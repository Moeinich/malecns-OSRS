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
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from itertools import count, pairwise
from pathlib import Path
from typing import Any

import numpy as np

from flybrain.app import (
    BROWSER,
    DEFAULT_BOT,
    DEFAULT_TICKRATE,
    LITE,
    ServiceFailed,
    Supervisor,
    bot_save_path,
    default_services,
)
from flybrain.connectome.loader import DEFAULT_PATH, Connectome, load
from flybrain.connectome.select import (
    MOTION_DETECTOR_TYPES,
    ON_RELAY_TYPES,
    VISUAL_INPUT_TYPES,
)
from flybrain.engine.calibration import DEFAULT_CALIBRATION_PATH, UNCALIBRATED, Calibration
from flybrain.engine.calibration import load as load_calibration
from flybrain.engine.lif import LIFEngine
from flybrain.loop.agent import (
    DEFAULT_TICK_MS,
    Ablation,
    Agent,
    AgentParams,
    _population,
    default_encoder,
)
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
DEFAULT_PRIMARY = "moving_fraction"

#: The Lumbridge spawn tile every scored episode starts from. Without it each
#: episode starts wherever the last one ended, and since conditions run in
#: sequence, position drift is a confound on every contrast.
DEFAULT_START = (3222, 3218)

#: How long a reset walk may take before the stack counts as down.
RESET_TIMEOUT_S = 90.0

#: How long to wait for the sidecar's socket to come back after `reset_bot`
#: restarts it, before the first episode of a condition tries to connect.
SIDECAR_RECONNECT_TIMEOUT_S = 30.0

SCALAR_METRICS = (
    "kills_per_hr",
    "xp_per_hr",
    "deaths_per_hr",
    "mean_hp_fraction",
    "distance",
    "tortuosity",
    "moving_fraction",
    "time_to_first_attack_s",
    "mean_rate_hz",
    "mean_ms_per_tick",
)

#: The sidecar's `deadlineFraction`. Source of truth is bridge/config.ts, which
#: reads the same variable; restated rather than imported so a drift is findable.
DEADLINE_FRACTION = float(os.environ.get("RS_DEADLINE_FRACTION") or 0.85)

#: An overrun fraction at or below this is not a material confound.
NEGLIGIBLE_OVERRUN = 0.01

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
SHUFFLE_DEGENERATE = (
    "NO VERDICT: the primary metric {metric} had no variance across the real and shuffle "
    "episodes — every episode scored the same value, so no effect could be significant and "
    "nothing can be concluded about the wiring."
)


class StackDown(RuntimeError):
    """The live stack is not answering, so no score was taken.

    Raised instead of returning zeros: a scored run against a dead game that
    quietly reports 0 kills for every condition looks exactly like a clean
    negative result and is worthless.
    """


# ------------------------------------------------------------------ conditions


#: Friendly lesion names that stand for a whole set of populations. A lesion
#: name is otherwise a cell type, and "optic" is not one. The optic lobe is the
#: 14 types `select.py` forces into the subgraph — the same grouping `hud.py`
#: colours as one legend entry — taken from there rather than restated here.
POPULATION_GROUPS: dict[str, tuple[str, ...]] = {
    "optic": MOTION_DETECTOR_TYPES + VISUAL_INPUT_TYPES + ON_RELAY_TYPES,
}


def ablation_for(condition: str, seed: int) -> Ablation:
    """The `Ablation` one condition name means. The shuffle is `agent.py`'s."""
    if condition == BASELINE:
        return Ablation(seed=seed)
    if condition == "ablate-network":
        return Ablation(ablate_network=True, seed=seed)
    if condition == "shuffle":
        return Ablation(shuffle=True, seed=seed)
    if condition.startswith("lesion:"):
        names = condition.removeprefix("lesion:").split("+")
        expanded = tuple(t for n in names for t in POPULATION_GROUPS.get(n, (n,)))
        return Ablation(lesions=expanded, seed=seed)
    raise ValueError(f"unknown condition {condition!r}; known: {', '.join(DEFAULT_CONDITIONS)}")


def validate_conditions(conditions: Sequence[str], connectome: Connectome, seed: int = 0) -> None:
    """Resolve every condition before any episode runs.

    A name this build has no population for costs a second here instead of
    twenty minutes of scored data that is then thrown away by the crash.
    """
    for condition in conditions:
        for name in ablation_for(condition, seed).lesions:
            _population(connectome, name)


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
    ms_total: float


@dataclass(frozen=True)
class Episode:
    condition: str
    seed: int
    tick_ms: int
    wall_s: float
    records: tuple[TickRecord, ...]
    dropped_game_ticks: int = 0
    #: Where the episode began: the acked reset tile, or the first scored tick
    #: when resets are off. Never blank, so a drifted run is visible afterwards.
    start: tuple[int, int] | None = None
    #: The tick length the sidecar measured, when it measured one. `None` is
    #: "never reported", not "matches the configured value".
    observed_tick_ms: float | None = None


class Recorder:
    """Turns (state, report) pairs into `TickRecord`s.

    Kills and deaths are edge-detected here because nothing upstream carries
    them: `TickReport` knows the action, the world state knows the consequence.
    """

    def __init__(self) -> None:
        self.records: list[TickRecord] = []
        self.observed_tick_ms: float | None = None
        self._engaged: set[int] = set()
        self._life_id: int | None = None

    def observe(self, update: StateUpdate, report: Any) -> TickRecord | None:
        """A tick with no player (logged out, loading) is driven but not scored."""
        if update.observed_tick_ms is not None:
            self.observed_tick_ms = float(update.observed_tick_ms)
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
            ms_total=float(report.ms_total),
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
        observed_tick_ms=recorder.observed_tick_ms,
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
        "mean_ms_per_tick": _mean([x.ms_total for x in r]),
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


def degenerate(per_condition: dict[str, list[dict[str, float]]], primary: str) -> bool:
    """True when the primary metric cannot carry a verdict at all.

    A metric that is the same number in every episode of `real` and `shuffle`
    makes every contrast non-significant by construction, which reads exactly
    like a clean "no difference" and is not one.
    """
    values = [
        v
        for condition in (BASELINE, "shuffle")
        for v in _column(per_condition.get(condition, []), primary)
        if math.isfinite(v)
    ]
    return len(values) < 2 or min(values) == max(values)


def _degenerate(primary: str) -> str:
    return SHUFFLE_DEGENERATE.format(metric=primary)


def shuffle_verdict(
    effects_by_condition: dict[str, dict[str, Effect]],
    primary: str,
    per_condition: dict[str, list[dict[str, float]]],
) -> str:
    """The sentence this whole harness exists to be able to print, either way."""
    if degenerate(per_condition, primary):
        return _degenerate(primary)
    effect = effects_by_condition.get("shuffle", {}).get(primary)
    if effect is None or not math.isfinite(effect.delta):
        return SHUFFLE_ABSENT
    if not effect.significant:
        return SHUFFLE_MATCHED
    return SHUFFLE_DEGRADED if effect.delta < 0 else SHUFFLE_BETTER


def expectations(
    effects_by_condition: dict[str, dict[str, Effect]],
    primary: str,
    per_condition: dict[str, list[dict[str, float]]],
) -> list[tuple[str, str, str]]:
    """Each stated expectation and whether this run met it.

    Stated up front, in the report, so a null result is legible rather than
    embarrassing: every line can read FAILED and the run is still a result.
    """
    out: list[tuple[str, str, str]] = []
    flat = degenerate(per_condition, primary)

    def on_primary(effect: Effect) -> tuple[str, str]:
        if flat:
            return ("NO VERDICT", f"{primary} had no variance across real and shuffle")
        return (_held(_dropped(effect)), _fmt_effect(effect))

    def get(condition: str, metric: str) -> Effect | None:
        return effects_by_condition.get(condition, {}).get(metric)

    ablate = get("ablate-network", primary)
    if ablate is not None:
        out.append(("ablate-network collapses to chance", *on_primary(ablate)))

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
        out.append(("shuffle degrades vs real", *on_primary(shuffle)))
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
    #: The tick the rates were scored against: the measured one where the stack
    #: measured it, else the one it was configured with.
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
    configured_tick_ms: int | None = None
    observed_tick_ms: float | None = None
    hud: bool = False
    start: tuple[int, int] | None = None
    fresh_per_condition: bool = False


def _tick_mismatch(meta: RunMeta) -> list[str]:
    """Said out loud when the measured tick and the configured one disagree."""
    observed = meta.observed_tick_ms
    if observed is None or meta.configured_tick_ms in (None, round(observed)):
        return []
    return [
        (
            f"TICK MISMATCH      configured {meta.configured_tick_ms} ms, observed "
            f"{observed:.1f} ms — the rates above are scored against the observed value"
        )
    ]


def caveats(per_condition: dict[str, list[dict[str, float]]], meta: RunMeta) -> tuple[str, ...]:
    """Recorded in every report, derived from the run they are attached to.

    A caveat quoting numbers from some earlier run reads authoritative and
    describes a run that is not this one, so every figure here is measured.
    """
    return (_overrun_caveat(per_condition, meta), _decoder_caveat(per_condition))


def _overrun_caveat(per_condition: dict[str, list[dict[str, float]]], meta: RunMeta) -> str:
    deadline = round(meta.tick_ms * DEADLINE_FRACTION)
    measured = [
        (condition, _mean(_column(runs, "mean_ms_per_tick")), fraction)
        for condition, runs in per_condition.items()
        if math.isfinite(fraction := _mean(_column(runs, "overrun_fraction")))
    ]
    head = (
        f"The brain's step runs against a {deadline} ms deadline ({DEADLINE_FRACTION:g} of the "
        f"{meta.tick_ms} ms tick) and the encoder drive goes stale on any tick it overruns. "
    )
    if not measured:
        return head + "No overrun was measured this run, so nothing can be said about staleness."
    detail = ", ".join(f"{c} {ms:.1f} ms / {f:.3g}" for c, ms, f in measured)
    mechanism = (
        "The cost of a step is edges touched = N x firing rate x degree, so a hotter condition "
        "costs more; overrun is then a STEP function of that cost against the deadline, so "
        "conditions clustered near it can show wildly different overrun fractions for a few ms "
        "of difference. Measured this run (mean_ms_per_tick / overrun_fraction): " + detail + ". "
    )
    overrunning = [(c, f) for c, _, f in measured if f > NEGLIGIBLE_OVERRUN]
    if not overrunning:
        worst = max(measured, key=lambda row: row[2])
        return (
            head
            + mechanism
            + (
                f"No condition overran materially this run (the worst, {worst[0]}, at "
                f"{worst[2]:.3g}), so staleness was not a live confound in THIS run."
            )
        )
    named = ", ".join(f"{c} ({f:.3g})" for c, f in overrunning)
    return (
        head
        + mechanism
        + (
            f"Staleness IS a live confound on any contrast involving {named}: those conditions ran "
            f"stale drive on that fraction of their ticks, and the conditions they are measured "
            f"against did not do so to the same degree."
        )
    )


def _decoder_caveat(per_condition: dict[str, list[dict[str, float]]]) -> str:
    head = (
        "The decoded drive sits near 0.10 with a steering differential of about +/-0.1 rad, so "
        "directed movement is slow and the steering metrics (distance, tortuosity) are "
        "correspondingly noisy. "
    )
    killing = [
        (condition, kills)
        for condition, runs in per_condition.items()
        if math.isfinite(kills := _mean(_column(runs, "kills_per_hr"))) and kills > 0
    ]
    if killing:
        return (
            head
            + "Kills were not zero in every condition this run: "
            + ", ".join(f"{c} {k:.3g}/hr" for c, k in killing)
        )
    return head + (
        "Zero kills in every condition is a result about the decoder's dynamic range, not a "
        "broken harness."
    )


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
        *_tick_mismatch(meta),
        f"calibration        {meta.calibration}",
        (
            f"mode               {'dry-run (no commands sent)' if meta.dry_run else 'live'}, "
            f"learning {'ON' if meta.learn else 'OFF (weights frozen)'}"
        ),
        f"hud                {'on' if meta.hud else 'off'}",
        (
            f"start              {meta.start[0]}, {meta.start[1]} (every episode reset here)"
            if meta.start
            else "start              not reset (every episode began where the last one ended)"
        ),
        (
            "character          fresh per condition"
            if meta.fresh_per_condition
            else "character          carried over between conditions"
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
    for name, held, detail in expectations(effects_by_condition, meta.primary, per_condition):
        lines.append(f"  {held:<6} {name}  ({detail})")

    verdict = shuffle_verdict(effects_by_condition, meta.primary, per_condition)
    lines += [
        "",
        "-- the verdict that matters " + "-" * 50,
        f"  primary metric: {meta.primary}",
        f"  {verdict}",
        "",
        "-- caveats " + "-" * 67,
    ]
    lines += [f"  - {c}" for c in caveats(per_condition, meta)]
    return "\n".join(lines) + "\n"


def to_json(
    per_condition: dict[str, list[dict[str, float]]],
    effects_by_condition: dict[str, dict[str, Effect]],
    meta: RunMeta,
) -> dict[str, Any]:
    return _jsonable(
        {
            "meta": vars(meta),
            "caveats": list(caveats(per_condition, meta)),
            "episodes": per_condition,
            "effects": {
                condition: {metric: vars(e) for metric, e in by_metric.items()}
                for condition, by_metric in effects_by_condition.items()
            },
            "expectations": [
                {"expectation": name, "verdict": held, "detail": detail}
                for name, held, detail in expectations(
                    effects_by_condition, meta.primary, per_condition
                )
            ],
            "shuffle_verdict": shuffle_verdict(effects_by_condition, meta.primary, per_condition),
        }
    )


def partial_json(per_condition: dict[str, list[dict[str, float]]], meta: RunMeta) -> dict[str, Any]:
    """The scores so far, without the cross-condition statistics.

    Written after every condition completes: a crash in condition five then
    costs one condition rather than every episode scored before it.
    """
    return _jsonable(
        {
            "meta": vars(meta),
            "caveats": list(caveats(per_condition, meta)),
            "episodes": per_condition,
            "partial": True,
        }
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


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
    hud: bool = False
    start: tuple[int, int] | None = DEFAULT_START
    fresh_per_condition: bool = False


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


def wait_for_sidecar(
    socket_path: str, timeout: float = SIDECAR_RECONNECT_TIMEOUT_S, interval: float = 0.5
) -> None:
    """Retry `connect` until the sidecar `reset_bot` just restarted answers.

    `connect` itself never retries — that contract is load-bearing everywhere
    else in this file — so the bounded retry lives here, only for the one
    connect that is expected to race a restart.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            connect(socket_path).close()
            return
        except StackDown:
            if time.monotonic() >= deadline:
                raise
            time.sleep(interval)


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


def build_agent(
    stack: Stack, condition: str, seed: int, client: BridgeClient, hud: Any = None
) -> Agent:
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
        calibration=calibration,
        spike_sink=hud.record_spikes if hud is not None and hud.enabled else None,
        reward=reward,
    )


def hud_tick(
    agent: Agent, hud: Any, feed: Any, condition: str, episode: int, episodes: int, ticks: int
) -> Callable[[StateUpdate], Any]:
    """`agent.tick` with the window drawn after it, scoring the same report."""
    counter = count(1)

    def tick(update: StateUpdate) -> Any:
        report = agent.tick(update)
        hud.label = (
            f"ablation - {condition} - episode {episode}/{episodes} - tick {next(counter)}/{ticks}"
        )
        if feed is not None and hud.should_draw(report.overrun):
            hud.game_frame = feed.read()
        hud.update(agent, report)
        return report

    return tick


def reset_to_start(client: BridgeClient, start: tuple[int, int]) -> tuple[int, int]:
    """Walk the bot back to `start`, or `StackDown`. Never scores a misplaced episode."""
    try:
        cmd_id = client.send_reset(*start)
        ok, x, z = client.wait_reset(cmd_id, RESET_TIMEOUT_S)
    except (ConnectionError, OSError, TimeoutError) as exc:
        raise StackDown(f"the stack dropped during the reset to {start}: {exc}") from exc
    if not ok:
        raise StackDown(f"the reset to {start} failed; the bot is at ({x}, {z})")
    if max(abs(x - start[0]), abs(z - start[1])) > 1:
        raise StackDown(f"the reset to {start} landed at ({x}, {z}), more than a tile away")
    return x, z


def run_condition(
    stack: Stack,
    condition: str,
    episodes: int,
    ticks: int,
    seed: int,
    hud: Any = None,
    feed: Any = None,
) -> list[Episode]:
    out = []
    for i in range(episodes):
        episode_seed = seed + i
        client = connect(stack.socket)
        try:
            agent = build_agent(stack, condition, episode_seed, client, hud)
            driver = (
                agent.tick
                if hud is None
                else hud_tick(agent, hud, feed, condition, i + 1, episodes, ticks)
            )
            start = reset_to_start(client, stack.start) if stack.start else None
            episode = record_episode(
                alive_states(client),
                driver,
                condition=condition,
                seed=episode_seed,
                ticks=ticks,
                tick_ms=agent.tick_ms,
                dropped=lambda c=client: c.dropped_game_ticks,
            )
            # Before the first state arrives `agent.tick_ms` is only the default,
            # so the rates would be scored against a tickrate the stack is not
            # running. Take it again once the episode has seen the handshake.
            first = episode.records[0]
            episode = replace(
                episode,
                tick_ms=agent.tick_ms,
                start=start if start is not None else (first.x, first.z),
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
    p.add_argument(
        "--start",
        type=int,
        nargs=2,
        metavar=("X", "Z"),
        default=list(DEFAULT_START),
        help="the tile every episode is reset to before it is scored",
    )
    p.add_argument(
        "--no-reset",
        action="store_true",
        help="score from wherever the last episode ended (position drift is then a confound)",
    )
    p.add_argument("--dry-run", action="store_true", help="score the brain without moving the bot")
    p.add_argument(
        "--fresh-per-condition",
        action="store_true",
        help="give the bot a brand-new character before each condition (HP, inventory, XP and "
        "loot otherwise carry over between conditions)",
    )
    p.add_argument("--tickrate", type=int, default=DEFAULT_TICKRATE)
    p.add_argument("--bot", default=DEFAULT_BOT)
    p.add_argument(
        "--client",
        choices=[LITE, BROWSER],
        default=LITE,
        help="lite: headless, no pixels. browser: Chromium on /bot, feeding the HUD. Only used "
        "with --fresh-per-condition",
    )
    p.add_argument("--out", default=None, help="write the JSON report here")
    p.add_argument(
        "--hud",
        action="store_true",
        help="live OpenCV telemetry window; q closes it, the run continues",
    )
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
        hud=args.hud,
        start=None if args.no_reset else (args.start[0], args.start[1]),
        fresh_per_condition=args.fresh_per_condition,
    )
    connectome = load(args.connectome)
    validate_conditions(args.conditions, connectome, args.seed)

    calibration = load_calibration(args.calibration) or UNCALIBRATED
    meta = RunMeta(
        tick_ms=DEFAULT_TICK_MS,
        ticks_per_episode=args.ticks,
        episodes=args.episodes,
        seed=args.seed,
        primary=args.primary,
        dry_run=args.dry_run,
        learn=False,
        calibration=calibration.describe(),
        calibration_params=calibration_params(calibration),
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
        hud=args.hud,
        start=stack.start,
        fresh_per_condition=args.fresh_per_condition,
    )

    sup = client_service = sidecar_service = None
    if args.fresh_per_condition:
        sup = Supervisor(default_services(args.tickrate, args.bot, args.client))
        client_service = next(s for s in sup.services if s.name in (LITE, BROWSER))
        sidecar_service = next(s for s in sup.services if s.name == "sidecar")

    hud = feed = None
    if args.hud:
        # Imported only here: the brain must not depend on OpenCV being installed.
        from flybrain.gamefeed import GameFeed
        from flybrain.hud import Hud

        feed = GameFeed()
        hud = Hud.create(
            populations=connectome.populations,
            soma_positions=connectome.soma_positions,
            n=connectome.n,
            dt_ms=calibration.engine_kwargs().get("dt_ms", 1.0),
            min_interval=1 / 6,
            log=lambda m: print(m, file=sys.stderr),
        )

    out_path = Path(args.out) if args.out else None
    per_condition: dict[str, list[dict[str, float]]] = {}
    try:
        for condition in args.conditions:
            print(f"condition {condition}", file=sys.stderr)
            if args.fresh_per_condition:
                try:
                    sup.reset_bot(
                        args.bot, client_service, sidecar_service, bot_save_path(args.bot)
                    )
                except (ServiceFailed, subprocess.CalledProcessError) as exc:
                    print(f"\n{exc}", file=sys.stderr)
                    return 1
                wait_for_sidecar(stack.socket)
            runs = run_condition(stack, condition, args.episodes, args.ticks, args.seed, hud, feed)
            meta = replace(meta, **tick_fields(runs))
            per_condition[condition] = [metrics(e) for e in runs]
            if out_path:
                write_json(out_path, partial_json(per_condition, meta))
                print(f"wrote {len(per_condition)} condition(s) to {out_path}", file=sys.stderr)
    finally:
        if hud is not None:
            hud.close()

    computed = effects(per_condition, seed=args.seed)
    print(report(per_condition, computed, meta))
    if out_path:
        write_json(out_path, to_json(per_condition, computed, meta))
        print(f"wrote {out_path}", file=sys.stderr)
    return 0


def tick_fields(episodes: Sequence[Episode]) -> dict[str, Any]:
    """The tick the scores are read against: measured where the stack measured it.

    Recorded alongside the configured value rather than in place of it, so a
    disagreement is reported instead of silently resolved.
    """
    measured = [e.observed_tick_ms for e in episodes if e.observed_tick_ms is not None]
    observed = _mean(measured) if measured else None
    configured = episodes[0].tick_ms
    return {
        "tick_ms": round(observed) if observed is not None else configured,
        "configured_tick_ms": configured,
        "observed_tick_ms": observed,
    }


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
