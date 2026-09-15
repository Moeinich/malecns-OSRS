#!/usr/bin/env python3
"""Scored ablation harness: does the connectome's specific wiring do any work?

Runs N episodes per condition against the live stack, scores them, and reports
effect sizes with bootstrap confidence intervals rather than bare means. The
condition that matters is `shuffle` — a degree- and sign-preserving rewire. If a
shuffled connectome plays as well as the real one, the specific wiring
contributes nothing, and this harness says exactly that in those terms.

A negative result is a result. The report is written so it reads as one.

Usage:
    uv run python tools/ablation.py --episodes 10 --ticks 400 --out runs/ablation.json
    uv run python tools/ablation.py --conditions real shuffle --episodes 10
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from flybrain.connectome.loader import DEFAULT_PATH, load
from flybrain.engine.calibration import DEFAULT_CALIBRATION_PATH, UNCALIBRATED, Calibration
from flybrain.engine.calibration import load as load_calibration
from flybrain.engine.lif import LIFEngine
from flybrain.loop.agent import DEFAULT_TICK_MS, Ablation, Agent, AgentParams, default_encoder
from flybrain.loop.client import BridgeClient, default_socket_path
from flybrain.loop.types import StateUpdate
from flybrain.motor.decode import MotorIndex
from flybrain.reward import DopamineIndex, RewardRouter
from flybrain.sensory.collision import DEFAULT_COLLISION_PATH, CollisionGrid

DEFAULT_CONDITIONS = (
    "real",
    "lesion:DNp01",
    "lesion:DNa02",
    "lesion:optic",
    "ablate-network",
    "shuffle",
)

BASELINE = "real"
DEFAULT_PRIMARY = "xp_per_hr"

SCALAR_METRICS = (
    "kills_per_hr",
    "xp_per_hr",
    "deaths_per_hr",
    "mean_hp_fraction",
    "distance",
    "tortuosity",
    "time_to_first_attack_s",
    "mean_rate_hz",
)

#: Recorded in every report rather than papered over. Both are live as of writing.
CAVEATS = (
    (
        "The brain overruns its tick (~706 ms against a 360 ms deadline) and runs ~300 substeps "
        "instead of 600, so it simulates roughly half the biological time it reports. Scored "
        "numbers taken before that fix lands are provisional."
    ),
    (
        "The decoded steering differential is small (+0.21 / -0.32 Hz), so directed movement "
        "may be weak and the steering metrics (distance, tortuosity) correspondingly noisy."
    ),
)

SHUFFLE_DEGRADED = (
    "SHUFFLE DEGRADED vs real: the connectome's specific wiring does work beyond its degree "
    "sequence and transmitter signs."
)
SHUFFLE_MATCHED = (
    "SHUFFLE MATCHED real: the connectome's specific wiring contributes nothing beyond its "
    "degree sequence and transmitter signs."
)
SHUFFLE_BETTER = (
    "SHUFFLE OUTPERFORMED real: the connectome's specific wiring contributes nothing — a random "
    "rewire with the same degrees and signs played better."
)
SHUFFLE_ABSENT = "shuffle was not run: no verdict on the wiring."


class StackDown(RuntimeError):
    """The live stack is not answering, so no score was taken.

    Raised instead of returning zeros: a scored run against a dead game that
    quietly reports 0 kills for every condition looks exactly like a clean
    negative result and is worthless.
    """


# ------------------------------------------------------------------ conditions


def ablation_for(condition: str, seed: int) -> Ablation:
    """The `Ablation` one condition name means. The shuffle is `agent.py`'s."""
    if condition == BASELINE:
        return Ablation(seed=seed)
    if condition == "ablate-network":
        return Ablation(ablate_network=True, seed=seed)
    if condition == "shuffle":
        return Ablation(shuffle=True, seed=seed)
    if condition.startswith("lesion:"):
        return Ablation(lesions=tuple(condition.removeprefix("lesion:").split("+")), seed=seed)
    raise ValueError(f"unknown condition {condition!r}; known: {', '.join(DEFAULT_CONDITIONS)}")


# ------------------------------------------------------------------ recording


@dataclass(frozen=True, slots=True)
class TickRecord:
    tick: int
    action: str
    hp: int
    max_hp: int
    x: int
    z: int
    xp: int
    kills: int
    deaths: int
    mean_rate_hz: float
    overrun: bool


@dataclass(frozen=True)
class Episode:
    condition: str
    seed: int
    tick_ms: int
    wall_s: float
    records: tuple[TickRecord, ...]
    dropped_game_ticks: int = 0


class Recorder:
    """Turns (state, report) pairs into `TickRecord`s.

    Kills and deaths are edge-detected here because nothing upstream carries
    them: `TickReport` knows the action, the world state knows the consequence.
    """

    def __init__(self) -> None:
        self.records: list[TickRecord] = []
        self._engaged: set[int] = set()
        self._life_id: int | None = None

    def observe(self, update: StateUpdate, report: Any) -> TickRecord | None:
        """A tick with no player (logged out, loading) is driven but not scored."""
        player = update.state.player
        if player is None:
            return None
        record = TickRecord(
            tick=update.tick,
            action=report.action.kind,
            hp=player.hp,
            max_hp=player.max_hp,
            x=player.x,
            z=player.z,
            xp=sum(update.state.skills.values()),
            kills=self._kills(update),
            deaths=self._deaths(player),
            mean_rate_hz=float(report.mean_rate_hz),
            overrun=bool(report.overrun),
        )
        self.records.append(record)
        return record

    def _kills(self, update: StateUpdate) -> int:
        present = {npc.index: npc for npc in update.state.npcs}
        kills = 0
        for index in sorted(self._engaged):
            npc = present.get(index)
            if npc is None or (npc.hp is not None and npc.hp <= 0):
                kills += 1
                self._engaged.discard(index)
        player = update.state.player
        if player is None or player.target_type != "npc":
            return kills
        # Only a target still standing is engaged: the game leaves the dead
        # NPC's index on the player for a tick or two, and re-engaging it would
        # count the same kill again on the tick its corpse disappears.
        target = present.get(player.target_index)
        if target is not None and (target.hp is None or target.hp > 0):
            self._engaged.add(player.target_index)
        return kills

    def _deaths(self, player: Any) -> int:
        previous, self._life_id = self._life_id, player.life_id
        return 1 if previous is not None and player.life_id != previous else 0


def record_episode(
    states: Iterable[StateUpdate],
    tick: Callable[[StateUpdate], Any],
    *,
    condition: str,
    seed: int,
    ticks: int,
    tick_ms: int,
    dropped: Callable[[], int] = lambda: 0,
) -> Episode:
    """Drive at most `ticks` ticks and score them. No scored tick is a dead stack."""
    recorder = Recorder()
    started = time.monotonic()
    driven = 0
    for update in states:
        recorder.observe(update, tick(update))
        driven += 1
        if driven >= ticks:
            break
    if not recorder.records:
        raise StackDown(
            f"{condition}: {driven} ticks arrived and none carried a player — "
            f"the stack is down or the bot is not in game; refusing to score zeros"
        )
    return Episode(
        condition=condition,
        seed=seed,
        tick_ms=tick_ms,
        wall_s=time.monotonic() - started,
        records=tuple(recorder.records),
        dropped_game_ticks=dropped(),
    )


# ------------------------------------------------------------------- metrics


def metrics(episode: Episode) -> dict[str, float]:
    """Per-episode scores. Rates are per hour of *game* time, not wall time.

    Game time is `ticks * tick_ms`, so a run at a low `NODE_TICKRATE` is
    comparable with one at 600 ms — the tickrate is still recorded, because it
    changes how much biological time the brain gets per decision.
    """
    r = episode.records
    hours = len(r) * episode.tick_ms / 3_600_000
    out = {
        "ticks": float(len(r)),
        "kills_per_hr": sum(x.kills for x in r) / hours,
        "xp_per_hr": max(0, r[-1].xp - r[0].xp) / hours,
        "deaths_per_hr": sum(x.deaths for x in r) / hours,
        "mean_hp_fraction": _mean([x.hp / x.max_hp for x in r if x.max_hp > 0]),
        "mean_rate_hz": _mean([x.mean_rate_hz for x in r]),
        "overrun_fraction": _mean([float(x.overrun) for x in r]),
        "time_to_first_attack_s": _time_to_first_attack(episode),
    }
    out.update(_path(r))
    counts: dict[str, int] = {}
    for x in r:
        counts[x.action] = counts.get(x.action, 0) + 1
    out.update({f"action:{k}": v / len(r) for k, v in counts.items()})
    return out


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else math.nan


def _time_to_first_attack(episode: Episode) -> float:
    for i, record in enumerate(episode.records):
        if record.action == "attack_fovea":
            return i * episode.tick_ms / 1000
    return math.nan


def _path(records: Sequence[TickRecord]) -> dict[str, float]:
    """Travelled distance and tortuosity (path / net displacement).

    Tortuosity is undefined for a bot that never moved — `nan`, never 1.0,
    which would read as a straight line.
    """
    steps = [
        math.dist((a.x, a.z), (b.x, b.z)) for a, b in pairwise(records) if (a.x, a.z) != (b.x, b.z)
    ]
    path = float(sum(steps))
    net = math.dist((records[0].x, records[0].z), (records[-1].x, records[-1].z))
    return {
        "distance": path,
        "tortuosity": path / net if net > 0 else math.nan,
        "moving_fraction": len(steps) / max(1, len(records) - 1),
    }


def metric_names(per_condition: dict[str, list[dict[str, float]]]) -> list[str]:
    """Every metric any episode produced, scalars first then action fractions."""
    seen: set[str] = set()
    for runs in per_condition.values():
        for run in runs:
            seen.update(run)
    scalars = [m for m in SCALAR_METRICS if m in seen]
    rest = sorted(m for m in seen if m not in set(SCALAR_METRICS) | {"ticks"})
    return scalars + rest


# ---------------------------------------------------------------- statistics


@dataclass(frozen=True)
class Effect:
    """One metric, one condition, against the baseline. N=10 means CIs, not means."""

    metric: str
    condition: str
    baseline_mean: float
    mean: float
    delta: float
    lo: float
    hi: float
    hedges_g: float

    @property
    def significant(self) -> bool:
        """The CI of the difference excludes zero."""
        return math.isfinite(self.lo) and math.isfinite(self.hi) and (self.lo > 0 or self.hi < 0)

    @property
    def direction(self) -> str:
        if not self.significant:
            return "same"
        return "up" if self.delta > 0 else "down"


def bootstrap_ci(
    values: Sequence[float], rng: np.random.Generator, reps: int = 10_000, alpha: float = 0.05
) -> tuple[float, float]:
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < 2:
        return (math.nan, math.nan)
    draws = rng.choice(np.asarray(finite, dtype=float), size=(reps, len(finite)), replace=True)
    means = draws.mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def hedges_g(baseline: Sequence[float], treatment: Sequence[float]) -> float:
    """Standardised difference, small-sample corrected.

    With no within-condition spread and a real difference there is no
    standardised effect to quote: `nan`, never an infinity read as huge.
    """
    a = [v for v in baseline if math.isfinite(v)]
    b = [v for v in treatment if math.isfinite(v)]
    if len(a) < 2 or len(b) < 2:
        return math.nan
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    delta = statistics.fmean(b) - statistics.fmean(a)
    if pooled == 0:
        return 0.0 if delta == 0 else math.nan
    df = len(a) + len(b) - 2
    return (delta / pooled) * (1 - 3 / (4 * df - 1))


def compare(
    metric: str,
    condition: str,
    baseline: Sequence[float],
    treatment: Sequence[float],
    rng: np.random.Generator,
    reps: int = 10_000,
) -> Effect:
    """Bootstrap CI on the difference of means, plus Hedges' g."""
    a = [v for v in baseline if math.isfinite(v)]
    b = [v for v in treatment if math.isfinite(v)]
    lo = hi = math.nan
    if len(a) >= 2 and len(b) >= 2:
        arr_a = np.asarray(a, dtype=float)
        arr_b = np.asarray(b, dtype=float)
        diffs = rng.choice(arr_b, size=(reps, len(b))).mean(axis=1) - rng.choice(
            arr_a, size=(reps, len(a))
        ).mean(axis=1)
        lo, hi = (float(v) for v in np.quantile(diffs, [0.025, 0.975]))
    return Effect(
        metric=metric,
        condition=condition,
        baseline_mean=_mean(a),
        mean=_mean(b),
        delta=_mean(b) - _mean(a),
        lo=lo,
        hi=hi,
        hedges_g=hedges_g(a, b),
    )


def effects(
    per_condition: dict[str, list[dict[str, float]]], seed: int = 0, reps: int = 10_000
) -> dict[str, dict[str, Effect]]:
    """`{condition: {metric: Effect}}` against `real`. Seeded, so it reproduces."""
    if BASELINE not in per_condition:
        raise ValueError(f"no {BASELINE!r} condition to compare against")
    rng = np.random.default_rng(seed)
    names = metric_names(per_condition)
    out: dict[str, dict[str, Effect]] = {}
    for condition, runs in per_condition.items():
        if condition == BASELINE:
            continue
        out[condition] = {
            metric: compare(
                metric,
                condition,
                _column(per_condition[BASELINE], metric),
                _column(runs, metric),
                rng,
                reps,
            )
            for metric in names
        }
    return out


def _column(runs: Sequence[dict[str, float]], metric: str) -> list[float]:
    """A missing action key means the action never fired — a zero, not a gap."""
    default = 0.0 if metric.startswith("action:") else math.nan
    return [run.get(metric, default) for run in runs]


# ------------------------------------------------------------------ verdicts


def shuffle_verdict(effects_by_condition: dict[str, dict[str, Effect]], primary: str) -> str:
    """The sentence this whole harness exists to be able to print, either way."""
    effect = effects_by_condition.get("shuffle", {}).get(primary)
    if effect is None or not math.isfinite(effect.delta):
        return SHUFFLE_ABSENT
    if not effect.significant:
        return SHUFFLE_MATCHED
    return SHUFFLE_DEGRADED if effect.delta < 0 else SHUFFLE_BETTER


def expectations(
    effects_by_condition: dict[str, dict[str, Effect]], primary: str
) -> list[tuple[str, str, str]]:
    """Each stated expectation and whether this run met it.

    Stated up front, in the report, so a null result is legible rather than
    embarrassing: every line can read FAILED and the run is still a result.
    """
    out: list[tuple[str, str, str]] = []

    def get(condition: str, metric: str) -> Effect | None:
        return effects_by_condition.get(condition, {}).get(metric)

    ablate = get("ablate-network", primary)
    if ablate is not None:
        out.append(
            ("ablate-network collapses to chance", _held(_dropped(ablate)), _fmt_effect(ablate))
        )

    flee = get("lesion:DNp01", "action:flee")
    p_attack = get("lesion:DNp01", "action:attack_fovea")
    if flee is not None and p_attack is not None:
        out.append(
            (
                "lesion:DNp01 removes flee, foraging survives",
                _held(_dropped(flee) and not p_attack.significant),
                f"flee {_fmt_effect(flee)}; attack {_fmt_effect(p_attack)}",
            )
        )

    a_attack = get("lesion:DNa02", "action:attack_fovea")
    steering = get("lesion:DNa02", "tortuosity")
    if a_attack is not None and steering is not None:
        out.append(
            (
                "lesion:DNa02 undirects steering, attack survives",
                _held(steering.significant and not a_attack.significant),
                f"attack {_fmt_effect(a_attack)}; tortuosity {_fmt_effect(steering)}",
            )
        )

    if flee is not None and a_attack is not None and steering is not None:
        halves = _dropped(flee) and not a_attack.significant and steering.significant
        out.append(
            (
                "double dissociation DNp01 x DNa02",
                _held(halves),
                "flee lost with DNp01 while attack survived DNa02" if halves else "not both halves",
            )
        )

    shuffle = get("shuffle", primary)
    if shuffle is not None:
        out.append(("shuffle degrades vs real", _held(_dropped(shuffle)), _fmt_effect(shuffle)))
    return out


def _dropped(effect: Effect) -> bool:
    """Significantly below baseline — the only thing that counts as an ablation working."""
    return effect.significant and effect.delta < 0


def _held(met: bool) -> str:
    return "HELD" if met else "FAILED"


def _fmt_effect(e: Effect) -> str:
    return f"{e.delta:+.3g} [{e.lo:.3g}, {e.hi:.3g}] g={e.hedges_g:.2f}"


# ------------------------------------------------------------------- reports


@dataclass(frozen=True)
class RunMeta:
    tick_ms: int
    ticks_per_episode: int
    episodes: int
    seed: int
    primary: str
    dry_run: bool
    learn: bool
    calibration: str
    calibration_params: dict[str, Any]
    started: str


def report(
    per_condition: dict[str, list[dict[str, float]]],
    effects_by_condition: dict[str, dict[str, Effect]],
    meta: RunMeta,
) -> str:
    lines = [
        "=" * 78,
        "ABLATION — does the connectome's specific wiring do any work?",
        "=" * 78,
        (
            f"episodes/condition {meta.episodes}   ticks/episode "
            f"{meta.ticks_per_episode}   seed {meta.seed}"
        ),
        (
            f"NODE_TICKRATE      {meta.tick_ms} ms per game tick (a confound: it sets how "
            f"much biological time the brain gets per decision)"
        ),
        f"calibration        {meta.calibration}",
        (
            f"mode               {'dry-run (no commands sent)' if meta.dry_run else 'live'}, "
            f"learning {'ON' if meta.learn else 'OFF (weights frozen)'}"
        ),
        "",
        "-- means per condition " + "-" * 55,
    ]
    names = metric_names(per_condition)
    width = max((len(c) for c in per_condition), default=10)
    lines.append("metric".ljust(26) + "".join(c.rjust(width + 2) for c in per_condition))
    for metric in names:
        row = metric.ljust(26)
        for runs in per_condition.values():
            row += f"{_mean(_column(runs, metric)):>{width + 2}.3g}"
        lines.append(row)

    lines += ["", "-- effect vs real (delta, 95% bootstrap CI, Hedges' g) " + "-" * 23]
    for condition, by_metric in effects_by_condition.items():
        lines.append(f"[{condition}]")
        for metric in names:
            e = by_metric.get(metric)
            if e is None:
                continue
            flag = "*" if e.significant else " "
            lines.append(f"  {flag} {metric.ljust(24)} {_fmt_effect(e)}")

    lines += ["", "-- expectations " + "-" * 62]
    for name, held, detail in expectations(effects_by_condition, meta.primary):
        lines.append(f"  {held:<6} {name}  ({detail})")

    verdict = shuffle_verdict(effects_by_condition, meta.primary)
    lines += [
        "",
        "-- the verdict that matters " + "-" * 50,
        f"  primary metric: {meta.primary}",
        f"  {verdict}",
        "",
        "-- caveats " + "-" * 67,
    ]
    lines += [f"  - {c}" for c in CAVEATS]
    return "\n".join(lines) + "\n"


def to_json(
    per_condition: dict[str, list[dict[str, float]]],
    effects_by_condition: dict[str, dict[str, Effect]],
    meta: RunMeta,
) -> dict[str, Any]:
    return _jsonable(
        {
            "meta": vars(meta),
            "caveats": list(CAVEATS),
            "episodes": per_condition,
            "effects": {
                condition: {metric: vars(e) for metric, e in by_metric.items()}
                for condition, by_metric in effects_by_condition.items()
            },
            "expectations": [
                {"expectation": name, "verdict": held, "detail": detail}
                for name, held, detail in expectations(effects_by_condition, meta.primary)
            ],
            "shuffle_verdict": shuffle_verdict(effects_by_condition, meta.primary),
        }
    )


def _jsonable(value: Any) -> Any:
    """`nan` and `inf` are real answers here (undefined tortuosity, no attack);
    JSON has no word for them, so they travel as `null`."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# -------------------------------------------------------------------- runner


@dataclass(frozen=True)
class Stack:
    """Everything the harness needs from the live stack and the artifacts."""

    socket: str
    connectome_path: str
    collision_path: str
    calibration_path: str
    dry_run: bool
    learn: bool
    substeps: int | None


def connect(socket_path: str, timeout: float = 30.0) -> BridgeClient:
    """A connected client, or `StackDown`. Never a client that silently retries."""
    client = BridgeClient(socket_path, reconnect=False)
    try:
        client.connect()
    except OSError as exc:
        raise StackDown(
            f"no sidecar on {socket_path}: {exc}. Start the stack; a scored run against a "
            f"dead game is worthless."
        ) from exc
    # BridgeClient exposes no timeout: without one a dead sidecar blocks forever
    # in recv and the harness hangs instead of reporting the stack is down.
    client._sock.settimeout(timeout)
    return client


def alive_states(client: BridgeClient, timeout: float = 120.0) -> Iterator[StateUpdate]:
    """States from the first one where the bot is alive and in game: the respawn wait."""
    deadline = time.monotonic() + timeout
    waiting = True
    try:
        for update in client.states():
            player = update.state.player
            if waiting:
                if player is None or player.is_dead or not update.state.in_game:
                    if time.monotonic() > deadline:
                        raise StackDown(
                            f"bot never came back alive within {timeout:.0f} s — "
                            f"respawn it before scoring"
                        )
                    continue
                waiting = False
            yield update
    except (ConnectionError, OSError, TimeoutError) as exc:
        raise StackDown(f"the stack dropped mid-episode: {exc}") from exc


def build_agent(stack: Stack, condition: str, seed: int, client: BridgeClient) -> Agent:
    """The same wiring `flybrain.loop.run` builds, under one ablation."""
    connectome = load(stack.connectome_path)
    calibration = load_calibration(stack.calibration_path, connectome) or UNCALIBRATED
    W = calibration.apply(ablation_for(condition, seed).apply(connectome))
    engine = LIFEngine(W, seed=seed, **calibration.engine_kwargs())
    reward = None
    try:
        reward = RewardRouter(DopamineIndex.from_connectome(connectome), connectome.n)
    except KeyError as exc:
        print(f"no reward routed: {exc}", file=sys.stderr)
    return Agent(
        client=client,
        engine=engine,
        motor=MotorIndex.from_connectome(connectome),
        collision=CollisionGrid.load(stack.collision_path),
        encoder=default_encoder(connectome, calibration.encode_params()),
        params=AgentParams(substeps_per_subframe=stack.substeps, dry_run=stack.dry_run),
        tonic=calibration.tonic_drive(connectome.n),
        reward=reward,
    )


def run_condition(
    stack: Stack, condition: str, episodes: int, ticks: int, seed: int
) -> list[Episode]:
    out = []
    for i in range(episodes):
        episode_seed = seed + i
        client = connect(stack.socket)
        try:
            agent = build_agent(stack, condition, episode_seed, client)
            episode = record_episode(
                alive_states(client),
                agent.tick,
                condition=condition,
                seed=episode_seed,
                ticks=ticks,
                tick_ms=agent.tick_ms,
                dropped=lambda c=client: c.dropped_game_ticks,
            )
        finally:
            client.close()
        print(
            f"  {condition} episode {i + 1}/{episodes}: {len(episode.records)} ticks, "
            f"{episode.wall_s:.0f} s",
            file=sys.stderr,
        )
        out.append(episode)
    return out


# ----------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tools/ablation.py", description=__doc__)
    p.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    p.add_argument("--episodes", type=int, default=10, help="episodes per condition; N >= 10")
    p.add_argument("--ticks", type=int, default=400, help="scored ticks per episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--primary", default=DEFAULT_PRIMARY, help="the metric the verdict reads")
    p.add_argument("--socket", default=None)
    p.add_argument("--connectome", default=str(DEFAULT_PATH))
    p.add_argument("--collision", default=str(DEFAULT_COLLISION_PATH))
    p.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH))
    p.add_argument("--substeps", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="score the brain without moving the bot")
    p.add_argument("--out", default=None, help="write the JSON report here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 10:
        print(f"WARNING: {args.episodes} episodes; the CIs will not carry a claim", file=sys.stderr)
    if BASELINE not in args.conditions:
        raise SystemExit(
            f"--conditions must include {BASELINE!r}: everything is measured against it"
        )

    stack = Stack(
        socket=args.socket or default_socket_path(),
        connectome_path=args.connectome,
        collision_path=args.collision,
        calibration_path=args.calibration,
        dry_run=args.dry_run,
        learn=False,
        substeps=args.substeps,
    )
    per_condition: dict[str, list[dict[str, float]]] = {}
    tick_ms = DEFAULT_TICK_MS
    for condition in args.conditions:
        print(f"condition {condition}", file=sys.stderr)
        runs = run_condition(stack, condition, args.episodes, args.ticks, args.seed)
        tick_ms = runs[0].tick_ms
        per_condition[condition] = [metrics(e) for e in runs]

    calibration = load_calibration(args.calibration) or UNCALIBRATED
    meta = RunMeta(
        tick_ms=tick_ms,
        ticks_per_episode=args.ticks,
        episodes=args.episodes,
        seed=args.seed,
        primary=args.primary,
        dry_run=args.dry_run,
        learn=False,
        calibration=calibration.describe(),
        calibration_params=calibration_params(calibration),
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    computed = effects(per_condition, seed=args.seed)
    print(report(per_condition, computed, meta))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_json(per_condition, computed, meta), indent=2) + "\n")
        print(f"wrote {path}", file=sys.stderr)
    return 0


def calibration_params(calibration: Calibration) -> dict[str, Any]:
    """The tuning the scores were taken under, so a scored run is reproducible."""
    return {
        "gain": calibration.gain,
        "spontaneous_noise_std": calibration.spontaneous_noise_std,
        "i_max": calibration.i_max,
        "normalization": calibration.normalization,
        "incoming_cap": calibration.incoming_cap,
        "tonic_fraction": calibration.tonic_fraction,
        "b": calibration.b,
        "tau_w": calibration.tau_w,
        "git_rev": calibration.git_rev,
        "created": calibration.created,
    }


if __name__ == "__main__":
    raise SystemExit(main())
