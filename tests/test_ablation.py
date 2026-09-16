"""Hermetic tests for the ablation harness: no game, no connectome.

Everything here is synthetic. What is under test is the scoring, the statistics
and the report — in particular that the report can state the shuffle verdict in
both directions, and that a dead stack raises instead of scoring zeros.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flybrain.app import ServiceFailed
from flybrain.loop.types import Npc, Player, StateUpdate, WorldState
from tools import ablation
from tools.ablation import (
    DEADLINE_FRACTION,
    DEFAULT_START,
    ENGAGEMENT_TICKS,
    SHUFFLE_ABSENT,
    SHUFFLE_BETTER,
    SHUFFLE_DEGENERATE,
    SHUFFLE_DEGRADED,
    SHUFFLE_MATCHED,
    UNCALIBRATED,
    Episode,
    RunMeta,
    Stack,
    StackDown,
    TickRecord,
    ablation_for,
    alive_states,
    build_agent,
    build_parser,
    compare,
    connect,
    effects,
    hedges_g,
    hud_tick,
    metric_names,
    metrics,
    partial_json,
    record_episode,
    report,
    reset_to_start,
    shuffle_verdict,
    tick_fields,
    to_json,
    validate_conditions,
    write_json,
)

# ------------------------------------------------------------------ fixtures


def player(*, hp=10, x=0, z=0, life_id=1, target=-1, dead=False) -> Player:
    return Player(
        name="flybot01",
        combat_level=3,
        hp=hp,
        max_hp=10,
        x=x,
        z=z,
        level=0,
        run_energy=100,
        anim_id=-1,
        in_combat=target >= 0,
        target_index=target,
        target_type="npc" if target >= 0 else "none",
        is_dead=dead,
        life_id=life_id,
    )


def npc(index: int, hp: int | None = 5) -> Npc:
    return Npc(
        id=1,
        index=index,
        name="Man",
        combat_level=2,
        x=1,
        z=1,
        size=1,
        distance=1,
        hp=hp,
        max_hp=5,
        in_combat=True,
        target_index=-1,
        reachable=True,
        options=("Attack",),
    )


def update(
    tick: int, p: Player | None, npcs=(), xp: int = 0, observed_tick_ms: float | None = None
) -> StateUpdate:
    return StateUpdate(
        revision=tick,
        tick=tick,
        dropped_since_last=0,
        deadline_ms=360,
        tick_ms=600,
        observed_tick_ms=observed_tick_ms,
        state=WorldState(
            tick=tick,
            in_game=p is not None,
            modal_open=False,
            player=p,
            npcs=tuple(npcs),
            ground_items=(),
            locs=(),
            inventory=(),
            skills={"attack": xp},
            op_rejected_count=0,
        ),
    )


class FakeAction:
    def __init__(self, kind: str, npc_index: int = -1) -> None:
        self.kind = kind
        self.npc_index = npc_index


class FakeReport:
    def __init__(
        self,
        kind: str,
        rate: float = 1.75,
        overrun: bool = False,
        ms_total: float = 42.0,
        npc_index: int = -1,
    ) -> None:
        self.action = FakeAction(kind, npc_index)
        self.mean_rate_hz = rate
        self.overrun = overrun
        self.ms_total = ms_total


def tick_fn(kinds):
    """A brain stub: hands back the next scripted action per tick.

    An entry is a kind, or `(kind, npc_index)` for a targeted `attack_fovea`.
    """
    it = iter(kinds)

    def tick(_update):
        entry = next(it)
        return (
            FakeReport(entry)
            if isinstance(entry, str)
            else FakeReport(entry[0], npc_index=entry[1])
        )

    return tick


def episode(records, condition="real", tick_ms=600) -> Episode:
    return Episode(condition=condition, seed=0, tick_ms=tick_ms, wall_s=1.0, records=tuple(records))


def record(**kw) -> TickRecord:
    base = {
        "tick": 0,
        "action": "walk",
        "hp": 10,
        "max_hp": 10,
        "x": 0,
        "z": 0,
        "xp": 0,
        "kills": 0,
        "retaliation_kills": 0,
        "deaths": 0,
        "mean_rate_hz": 1.75,
        "overrun": False,
        "ms_total": 42.0,
    }
    return TickRecord(**(base | kw))


def runs(values: list[float], metric: str, extra: dict | None = None) -> list[dict[str, float]]:
    return [{metric: v} | (extra or {}) for v in values]


# ---------------------------------------------------------------- conditions


def test_condition_names_map_to_the_ablations_agent_already_implements():
    assert ablation_for("real", 3) == ablation_for("real", 3)
    assert ablation_for("ablate-network", 1).ablate_network
    assert ablation_for("shuffle", 7).shuffle and ablation_for("shuffle", 7).seed == 7
    assert ablation_for("lesion:DNp01", 0).lesions == ("DNp01",)
    with pytest.raises(ValueError, match="unknown condition"):
        ablation_for("lesion-DNp01", 0)


def test_a_friendly_lesion_name_expands_to_every_population_it_stands_for():
    lesions = ablation_for("lesion:optic", 0).lesions
    assert set(lesions) == {
        *("T4", "T5", "L1", "L2", "L3", "L5", "Tm1", "Tm3"),
        *("Mi1", "Mi4", "Mi9", "C2", "C3", "CT1"),
    }
    assert ablation_for("lesion:optic+DNp01", 0).lesions[-1] == "DNp01"


class FakeConnectome:
    """A build that knows one population, so every optic type is missing."""

    def __init__(self):
        self.populations = {"DNp01": [0]}

    def population(self, name, side=None):
        return self.populations[name]


class FakeOpticConnectome:
    """Every type `lesion:optic` stands for, each its own block of cells."""

    def __init__(self):
        types = ablation_for("lesion:optic", 0).lesions
        self.populations = {
            t: np.arange(i * 10, i * 10 + 7, dtype=np.int64) for i, t in enumerate(types)
        }
        self.populations["DNp01"] = np.array([500, 501], dtype=np.int64)

    def population(self, name, side=None):
        return self.populations[name]


def test_a_lesion_hands_the_engine_every_cell_it_removes():
    c = FakeOpticConnectome()
    optic = ablation_for("lesion:optic", 0)
    expected = np.unique(np.concatenate([c.population(n) for n in optic.lesions]))
    assert optic.silenced(c).tolist() == expected.tolist()
    assert ablation_for("lesion:DNp01", 0).silenced(c).tolist() == [500, 501]
    assert ablation_for("real", 0).silenced(c).size == 0


def test_every_condition_is_resolved_before_any_episode_runs():
    validate_conditions(["real", "shuffle", "lesion:DNp01"], FakeConnectome())
    with pytest.raises(KeyError, match="no population 'T4' in this build; have: DNp01"):
        validate_conditions(["real", "lesion:optic"], FakeConnectome())
    with pytest.raises(ValueError, match="unknown condition"):
        validate_conditions(["nonsense"], FakeConnectome())


# ------------------------------------------------------------------- scoring


def test_records_kills_deaths_and_xp_from_the_world_state():
    states = [
        update(1, player(target=7), npcs=[npc(7)], xp=0),
        update(2, player(target=7), npcs=[npc(7, hp=0)], xp=50),
        update(3, player(hp=0, life_id=1), npcs=[], xp=50),
        update(4, player(life_id=2), npcs=[], xp=50),
    ]
    e = record_episode(
        states,
        tick_fn([("attack_fovea", 7), ("attack_fovea", 7), "flee", "walk"]),
        condition="real",
        seed=0,
        ticks=10,
        tick_ms=600,
    )
    assert [r.kills for r in e.records] == [0, 1, 0, 0]
    assert [r.deaths for r in e.records] == [0, 0, 0, 1]
    assert [r.xp for r in e.records] == [0, 50, 50, 50]


def test_an_npc_that_vanishes_while_engaged_counts_as_a_kill():
    states = [
        update(1, player(target=7), npcs=[npc(7)]),
        update(2, player(), npcs=[]),
    ]
    e = record_episode(
        states,
        tick_fn([("attack_fovea", 7), "walk"]),
        condition="real",
        seed=0,
        ticks=2,
        tick_ms=600,
    )
    assert sum(r.kills for r in e.records) == 1


def kills(kinds, states) -> tuple[int, int]:
    e = record_episode(
        states, tick_fn(kinds), condition="real", seed=0, ticks=len(states), tick_ms=600
    )
    return sum(r.kills for r in e.records), sum(r.retaliation_kills for r in e.records)


def test_a_kill_the_fly_never_attacked_is_the_engines_retaliation_not_a_kill():
    states = [
        update(1, player(target=7), npcs=[npc(7)]),
        update(2, player(target=7), npcs=[npc(7, hp=0)]),
    ]
    assert kills(["walk", "walk"], states) == (0, 1)


def test_a_kill_inside_the_engagement_window_after_an_attack_is_the_flys():
    states = [update(t, player(target=7), npcs=[npc(7)]) for t in range(1, ENGAGEMENT_TICKS)]
    states.append(update(ENGAGEMENT_TICKS, player(target=7), npcs=[npc(7, hp=0)]))
    kinds = [("attack_fovea", 7)] + ["walk"] * (len(states) - 1)
    assert kills(kinds, states) == (1, 0)


def test_fleeing_hands_the_kill_back_to_the_engine():
    states = [
        update(1, player(target=7), npcs=[npc(7)]),
        update(2, player(target=7), npcs=[npc(7)]),
        update(3, player(target=7), npcs=[npc(7, hp=0)]),
    ]
    assert kills([("attack_fovea", 7), "flee", "walk"], states) == (0, 1)


def test_an_attack_that_has_gone_stale_no_longer_claims_a_later_kill():
    late = ENGAGEMENT_TICKS + 2
    states = [update(1, player(), npcs=[npc(7)])]
    states.append(update(late, player(target=7), npcs=[npc(7)]))
    states.append(update(late + 1, player(target=7), npcs=[npc(7, hp=0)]))
    assert kills([("attack_fovea", 7), "walk", "walk"], states) == (0, 1)


def test_both_kill_rates_are_reported_side_by_side():
    records = [record(tick=0, kills=1), record(tick=1, retaliation_kills=2)]
    m = metrics(episode(records, tick_ms=600))
    assert m["retaliation_kills_per_hr"] == pytest.approx(2 * m["kills_per_hr"])
    per_condition = {"real": [m], "shuffle": [m]}
    text = report(per_condition, effects(per_condition, seed=0, reps=20), meta())
    assert "retaliation_kills_per_hr" in text
    names = metric_names(per_condition)
    assert names.index("retaliation_kills_per_hr") == names.index("kills_per_hr") + 1


def test_metrics_are_per_hour_of_game_time():
    records = [record(tick=i, xp=100 * i, kills=1 if i == 5 else 0) for i in range(10)]
    m = metrics(episode(records, tick_ms=600))
    hours = 10 * 600 / 3_600_000
    assert m["kills_per_hr"] == pytest.approx(1 / hours)
    assert m["xp_per_hr"] == pytest.approx(900 / hours)
    assert m["deaths_per_hr"] == 0.0
    assert m["mean_hp_fraction"] == pytest.approx(1.0)


def test_a_faster_tickrate_scales_the_rates_it_is_compared_against():
    records = [record(tick=i, kills=1) for i in range(10)]
    fast = metrics(episode(records, tick_ms=100))
    slow = metrics(episode(records, tick_ms=600))
    assert fast["kills_per_hr"] == pytest.approx(6 * slow["kills_per_hr"])


def test_path_length_tortuosity_and_time_to_first_attack():
    walk = [record(tick=i, x=i, z=0) for i in range(5)]
    straight = metrics(episode(walk))
    assert straight["distance"] == pytest.approx(4.0)
    assert straight["tortuosity"] == pytest.approx(1.0)

    out_and_back = [record(tick=0, x=0), record(tick=1, x=2), record(tick=2, x=0)]
    assert math.isnan(metrics(episode(out_and_back))["tortuosity"])

    still = metrics(episode([record(tick=i) for i in range(3)]))
    assert still["distance"] == 0.0
    assert math.isnan(still["tortuosity"])


def test_time_to_first_attack_is_nan_when_the_bot_never_attacks():
    fleeing = [record(tick=i, action="flee") for i in range(5)]
    assert math.isnan(metrics(episode(fleeing))["time_to_first_attack_s"])
    mixed = [record(tick=0, action="walk"), record(tick=1, action="attack_fovea")]
    assert metrics(episode(mixed))["time_to_first_attack_s"] == pytest.approx(0.6)


def test_the_action_histogram_is_a_first_class_metric():
    records = [record(action="flee") for _ in range(8)] + [record(action="walk") for _ in range(2)]
    m = metrics(episode(records))
    assert m["action:flee"] == pytest.approx(0.8)
    assert m["action:walk"] == pytest.approx(0.2)
    assert "action:attack_fovea" not in m


# ---------------------------------------------------------------- dead stack


def test_a_dead_stack_raises_rather_than_scoring_zeros():
    states = [update(i, None) for i in range(5)]
    with pytest.raises(StackDown, match="refusing to score zeros"):
        record_episode(
            states,
            tick_fn(["walk"] * 5),
            condition="real",
            seed=0,
            ticks=5,
            tick_ms=600,
        )


def test_no_states_at_all_is_a_dead_stack_too():
    with pytest.raises(StackDown):
        record_episode([], tick_fn([]), condition="shuffle", seed=0, ticks=5, tick_ms=600)


def test_connecting_to_a_missing_socket_raises_stack_down(tmp_path):
    with pytest.raises(StackDown, match="no sidecar"):
        connect(str(tmp_path / "nothing.sock"))


class FakeClient:
    def __init__(self, states, error: Exception | None = None) -> None:
        self._states = states
        self._error = error

    def states(self):
        yield from self._states
        if self._error is not None:
            raise self._error


def test_episodes_start_only_once_the_bot_is_alive_again():
    states = [
        update(1, player(dead=True)),
        update(2, None),
        update(3, player(x=4)),
        update(4, player(x=5)),
    ]
    got = list(alive_states(FakeClient(states)))
    assert [u.tick for u in got] == [3, 4]


def test_a_stack_that_drops_mid_episode_raises():
    states = [update(1, player())]
    with pytest.raises(StackDown, match="dropped mid-episode"):
        list(alive_states(FakeClient(states, error=ConnectionError("socket closed"))))


# --------------------------------------------------------------- statistics


def test_effect_size_and_ci_on_a_clear_difference():
    import numpy as np

    rng = np.random.default_rng(0)
    e = compare("xp_per_hr", "shuffle", [100.0] * 5 + [110.0] * 5, [50.0] * 5 + [60.0] * 5, rng)
    assert e.delta == pytest.approx(-50.0)
    assert e.hi < 0 and e.significant and e.direction == "down"
    assert e.hedges_g < -2


def test_identical_groups_are_not_a_significant_effect():
    import numpy as np

    values = [10.0, 12.0, 9.0, 11.0, 13.0, 8.0, 10.5, 11.5, 9.5, 12.5]
    e = compare("xp_per_hr", "shuffle", values, list(values), np.random.default_rng(0))
    assert not e.significant
    assert e.direction == "same"
    assert e.lo < 0 < e.hi


def test_hedges_g_is_corrected_and_sign_follows_the_difference():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [3.0, 4.0, 5.0, 6.0]
    assert hedges_g(a, b) > 0
    assert hedges_g(b, a) == pytest.approx(-hedges_g(a, b))
    assert hedges_g(a, b) == pytest.approx(1.3471, abs=1e-3)
    assert math.isnan(hedges_g([1.0], [2.0]))


def test_the_same_seed_reproduces_the_same_intervals():
    per_condition = {
        "real": runs([10, 11, 9, 12, 8, 10, 11, 9, 12, 10], "xp_per_hr"),
        "shuffle": runs([4, 5, 3, 6, 2, 4, 5, 3, 6, 4], "xp_per_hr"),
    }
    first = effects(per_condition, seed=7, reps=500)
    second = effects(per_condition, seed=7, reps=500)
    assert first == second
    assert effects(per_condition, seed=8, reps=500) != first


def test_effects_need_a_real_baseline():
    with pytest.raises(ValueError, match="real"):
        effects({"shuffle": runs([1.0, 2.0], "xp_per_hr")})


def test_a_missing_action_key_is_a_zero_not_a_gap():
    per_condition = {
        "real": [{"action:flee": 0.5} for _ in range(6)],
        "lesion:DNp01": [{} for _ in range(6)],
    }
    e = effects(per_condition, seed=0, reps=500)["lesion:DNp01"]["action:flee"]
    assert e.mean == 0.0
    assert e.delta == pytest.approx(-0.5)


# ------------------------------------------------------------------ verdicts


def meta(primary: str = "xp_per_hr", **kw) -> RunMeta:
    return RunMeta(
        **{"tick_ms": 100} | kw,
        ticks_per_episode=400,
        episodes=10,
        seed=0,
        primary=primary,
        dry_run=False,
        learn=False,
        calibration="gain 3.2  band 1-5 Hz",
        calibration_params={"gain": 3.2},
        started="2026-01-01T00:00:00",
    )


def scored(shuffle_values: list[float]) -> tuple[dict, dict]:
    real = [10.0, 11.0, 9.0, 12.0, 8.0, 10.0, 11.0, 9.0, 12.0, 10.0]
    per_condition = {
        "real": runs(real, "xp_per_hr"),
        "shuffle": runs(shuffle_values, "xp_per_hr"),
    }
    return per_condition, effects(per_condition, seed=0, reps=2000)


def test_a_shuffle_that_degrades_is_reported_as_the_wiring_doing_work():
    per_condition, computed = scored([3.0, 4.0, 2.0, 5.0, 1.0, 3.0, 4.0, 2.0, 5.0, 3.0])
    assert shuffle_verdict(computed, "xp_per_hr", per_condition) == SHUFFLE_DEGRADED
    assert SHUFFLE_DEGRADED in report(per_condition, computed, meta())


def test_a_shuffle_that_matches_is_reported_as_the_wiring_contributing_nothing():
    per_condition, computed = scored([10.0, 11.0, 9.0, 12.0, 8.0, 10.0, 11.0, 9.0, 12.0, 10.0])
    assert shuffle_verdict(computed, "xp_per_hr", per_condition) == SHUFFLE_MATCHED
    text = report(per_condition, computed, meta())
    assert SHUFFLE_MATCHED in text
    assert "contributes nothing" in text
    assert "FAILED shuffle degrades vs real" in text


def test_a_shuffle_that_wins_is_also_a_negative_result():
    per_condition, computed = scored([40.0, 41.0, 39.0, 42.0, 38.0, 40.0, 41.0, 39.0, 42.0, 40.0])
    assert shuffle_verdict(computed, "xp_per_hr", per_condition) == SHUFFLE_BETTER
    assert "contributes nothing" in report(per_condition, computed, meta())


def test_no_shuffle_condition_gives_no_verdict():
    real = runs([10.0, 11.0, 9.0, 12.0], "xp_per_hr")
    assert shuffle_verdict({}, "xp_per_hr", {"real": real}) == SHUFFLE_ABSENT


def test_a_primary_metric_with_no_variance_is_a_degeneracy_not_a_match():
    """A constant column makes every contrast non-significant by construction."""
    per_condition = {
        "real": runs([0.0] * 10, "xp_per_hr"),
        "shuffle": runs([0.0] * 10, "xp_per_hr"),
    }
    computed = effects(per_condition, seed=0, reps=500)
    verdict = shuffle_verdict(computed, "xp_per_hr", per_condition)
    assert verdict == SHUFFLE_DEGENERATE.format(metric="xp_per_hr")
    assert verdict != SHUFFLE_MATCHED

    text = report(per_condition, computed, meta())
    assert "NO VERDICT" in text
    assert SHUFFLE_MATCHED not in text
    assert "FAILED shuffle degrades vs real" not in text


def test_an_all_nan_primary_metric_is_degenerate_too():
    per_condition = {
        "real": runs([math.nan] * 6, "xp_per_hr"),
        "shuffle": runs([math.nan] * 6, "xp_per_hr"),
    }
    computed = effects(per_condition, seed=0, reps=200)
    assert shuffle_verdict(computed, "xp_per_hr", per_condition) == SHUFFLE_DEGENERATE.format(
        metric="xp_per_hr"
    )


# -------------------------------------------------------------------- report


def full_run() -> tuple[dict, dict]:
    def block(xp, flee, attack, tortuosity):
        return [
            {
                "xp_per_hr": xp + (i % 3),
                "action:flee": flee,
                "action:attack_fovea": attack,
                "tortuosity": tortuosity + (i % 3) * 0.01,
            }
            for i in range(10)
        ]

    per_condition = {
        "real": block(100, 0.3, 0.5, 1.2),
        "lesion:DNp01": block(90, 0.0, 0.5, 1.2),
        "lesion:DNa02": block(95, 0.3, 0.5, 3.0),
        "ablate-network": block(2, 0.3, 0.5, 1.2),
        "shuffle": block(20, 0.3, 0.5, 1.2),
    }
    return per_condition, effects(per_condition, seed=0, reps=2000)


def test_the_report_states_every_expectation_and_the_tickrate():
    per_condition, computed = full_run()
    text = report(per_condition, computed, meta())
    assert "NODE_TICKRATE      100 ms" in text
    assert "gain 3.2" in text
    for expectation in (
        "ablate-network collapses to chance",
        "lesion:DNp01 removes flee, foraging survives",
        "lesion:DNa02 undirects steering, attack survives",
        "double dissociation DNp01 x DNa02",
        "shuffle degrades vs real",
    ):
        assert expectation in text
    assert "HELD" in text
    assert "No overrun was measured this run" in text


def test_the_json_report_is_serialisable_and_carries_the_verdict():
    per_condition, computed = full_run()
    payload = to_json(per_condition, computed, meta())
    text = json.dumps(payload)
    assert json.loads(text)["shuffle_verdict"] == SHUFFLE_DEGRADED
    assert json.loads(text)["meta"]["calibration_params"]["gain"] == 3.2
    assert json.loads(text)["meta"]["tick_ms"] == 100


def test_undefined_metrics_travel_as_null_not_as_zero():
    per_condition = {
        "real": [{"tortuosity": math.nan} for _ in range(4)],
        "shuffle": [{"tortuosity": math.nan} for _ in range(4)],
    }
    payload = to_json(per_condition, effects(per_condition, seed=0, reps=200), meta())
    assert json.loads(json.dumps(payload))["episodes"]["real"][0]["tortuosity"] is None


def test_each_condition_is_written_as_it_completes(tmp_path):
    out = tmp_path / "nested" / "ablation.json"
    per_condition = {"real": runs([1.0, 2.0], "xp_per_hr")}
    write_json(out, partial_json(per_condition, meta()))
    first = json.loads(out.read_text())
    assert first["partial"] is True
    assert list(first["episodes"]) == ["real"]

    per_condition["shuffle"] = runs([3.0, 4.0], "xp_per_hr")
    write_json(out, partial_json(per_condition, meta()))
    assert list(json.loads(out.read_text())["episodes"]) == ["real", "shuffle"]


def test_the_tickrate_is_taken_after_the_episode_not_before_the_handshake():
    """`agent.tick_ms` is the 600 ms default until the sidecar says otherwise."""

    class LateAgent:
        tick_ms = 600

        def tick(self, update):
            self.tick_ms = 150
            return FakeReport("walk")

    agent = LateAgent()
    episode = record_episode(
        [update(i, player()) for i in range(3)],
        agent.tick,
        condition="real",
        seed=0,
        ticks=3,
        tick_ms=agent.tick_ms,
    )
    assert episode.tick_ms == 600, "the value read before the handshake is the stale one"
    assert dataclasses.replace(episode, tick_ms=agent.tick_ms).tick_ms == 150


def test_the_measured_tick_is_recorded_and_a_disagreement_is_reported():
    """The stack measured 150 ms while configured for 600; neither is dropped."""
    e = record_episode(
        [update(i, player(), observed_tick_ms=150.0) for i in range(3)],
        tick_fn(["walk"] * 3),
        condition="real",
        seed=0,
        ticks=3,
        tick_ms=600,
    )
    assert e.observed_tick_ms == pytest.approx(150.0)
    assert tick_fields([e]) == {
        "tick_ms": 150,
        "configured_tick_ms": 600,
        "observed_tick_ms": pytest.approx(150.0),
    }

    per_condition, computed = full_run()
    text = report(per_condition, computed, meta(**tick_fields([e])))
    assert "NODE_TICKRATE      150 ms" in text
    assert "configured 600 ms, observed 150.0 ms" in text


def test_an_unmeasured_tick_falls_back_to_the_configured_one_without_a_mismatch():
    e = record_episode(
        [update(i, player()) for i in range(2)],
        tick_fn(["walk"] * 2),
        condition="real",
        seed=0,
        ticks=2,
        tick_ms=600,
    )
    assert tick_fields([e]) == {
        "tick_ms": 600,
        "configured_tick_ms": 600,
        "observed_tick_ms": None,
    }
    per_condition, computed = full_run()
    assert "TICK MISMATCH" not in report(per_condition, computed, meta(**tick_fields([e])))


def test_compute_time_is_measured_per_episode_not_only_as_an_overrun_flag():
    m = metrics(episode([record(tick=i, ms_total=100.0 + i) for i in range(4)]))
    assert m["mean_ms_per_tick"] == pytest.approx(101.5)


# ------------------------------------------------------------------- caveats


def timed(
    ms: float, overrun: float, kills: float = 0.0, retaliation: float = 0.0
) -> list[dict[str, float]]:
    return [
        {
            "mean_ms_per_tick": ms,
            "overrun_fraction": overrun,
            "kills_per_hr": kills,
            "retaliation_kills_per_hr": retaliation,
        }
        for _ in range(4)
    ]


def test_the_overrun_caveat_is_measured_from_the_run_not_hardcoded():
    per_condition = {"real": timed(94.6, 0.733), "shuffle": timed(99.0, 1.0)}
    text = report(per_condition, effects(per_condition, seed=0, reps=200), meta(tick_ms=150))
    assert "94.6 ms / 0.733" in text
    assert "99.0 ms / 1" in text
    assert f"{round(150 * DEADLINE_FRACTION)} ms deadline" in text
    assert "LIVE CONFOUND" in text.upper()
    assert "shuffle (1)" in text and "real (0.733)" in text
    assert "not a live confound" not in text
    assert "applies equally" not in text


def test_a_run_that_meets_its_deadline_says_staleness_was_not_a_confound():
    per_condition = {"real": timed(40.0, 0.0), "shuffle": timed(41.0, 0.0)}
    text = report(per_condition, effects(per_condition, seed=0, reps=200), meta(tick_ms=200))
    assert "staleness was not a live confound in THIS run" in text
    assert f"{round(200 * DEADLINE_FRACTION)} ms deadline" in text
    assert "IS a live confound" not in text
    assert "applies equally" not in text


def test_the_derived_caveats_are_identical_in_the_text_and_json_reports():
    per_condition = {"real": timed(94.6, 0.733), "shuffle": timed(99.0, 1.0)}
    computed = effects(per_condition, seed=0, reps=200)
    m = meta(tick_ms=150)
    for caveat in to_json(per_condition, computed, m)["caveats"]:
        assert caveat in report(per_condition, computed, m)


def test_the_zero_kills_clause_is_conditional_on_the_measured_kills():
    m = meta(tick_ms=150)
    quiet = {"real": timed(40.0, 0.0), "shuffle": timed(40.0, 0.0)}
    assert "Zero kills in every condition" in report(quiet, effects(quiet, seed=0, reps=200), m)

    killing = {"real": timed(40.0, 0.0, kills=12.0), "shuffle": timed(40.0, 0.0)}
    text = report(killing, effects(killing, seed=0, reps=200), m)
    assert "Zero kills in every condition" not in text
    assert "real 12/hr" in text


def test_the_caveat_names_retaliation_kills_only_when_there_were_some():
    m = meta(tick_ms=150)
    quiet = {"real": timed(40.0, 0.0), "shuffle": timed(40.0, 0.0)}
    assert "Retaliation kills" not in report(quiet, effects(quiet, seed=0, reps=200), m)

    retaliating = {
        "real": timed(40.0, 0.0, retaliation=10.2),
        "shuffle": timed(40.0, 0.0),
    }
    text = report(retaliating, effects(retaliating, seed=0, reps=200), m)
    assert "Retaliation kills did happen (real 10.2/hr)" in text
    assert "excluded from kills_per_hr" in text


# ------------------------------------------------------------------------ hud


class FakeHud:
    """Everything `run_condition` asks of a hud, and nothing more."""

    enabled = True

    def __init__(self) -> None:
        self.label = None
        self.game_frame = None
        self.calls = []

    def record_spikes(self, fired) -> None:
        pass

    def update(self, agent, report) -> None:
        self.calls.append((agent, report, self.label))

    def close(self) -> None:
        pass


def test_the_hud_flag_reaches_the_stack_the_meta_and_the_report():
    args = build_parser().parse_args(["--hud"])
    assert args.hud
    assert Stack(
        socket="s",
        connectome_path="c",
        collision_path="x",
        calibration_path="k",
        dry_run=False,
        learn=False,
        substeps=None,
        hud=args.hud,
    ).hud

    per_condition, computed = full_run()
    assert "hud                on" in report(per_condition, computed, meta(hud=True))
    assert "hud                off" in report(per_condition, computed, meta())


def test_the_hud_driver_draws_every_tick_and_scores_the_report_unchanged():
    reports = [object(), object(), object()]
    agent = SimpleNamespace(tick=lambda update: reports.pop(0))
    fake = FakeHud()
    tick = hud_tick(agent, fake, None, "shuffle", 2, 5, 3)

    returned = [tick(None) for _ in range(3)]
    assert len(fake.calls) == 3
    assert [r for _a, r, _l in fake.calls] == returned
    assert all(a is agent for a, _r, _l in fake.calls)
    labels = [label for _a, _r, label in fake.calls]
    assert labels == [
        "ablation - shuffle - episode 2/5 - tick 1/3",
        "ablation - shuffle - episode 2/5 - tick 2/3",
        "ablation - shuffle - episode 2/5 - tick 3/3",
    ]


def _stub_build_agent(monkeypatch) -> list[dict]:
    """Strip `build_agent` down to the one wiring decision under test."""
    built: list[dict] = []
    sentinel = SimpleNamespace(
        populations={}, soma_positions=None, n=4, connectome_path=None, W=None
    )
    calibration = SimpleNamespace(
        apply=lambda w: w,
        engine_kwargs=dict,
        encode_params=lambda: None,
        tonic_drive=lambda n: None,
    )
    for name, value in (
        ("load", lambda path: sentinel),
        ("load_calibration", lambda path, c=None: calibration),
        (
            "ablation_for",
            lambda condition, seed: SimpleNamespace(apply=lambda c: None, silenced=lambda c: None),
        ),
        ("LIFEngine", lambda w, **kw: None),
        ("MotorIndex", SimpleNamespace(from_connectome=lambda c: None)),
        ("CollisionGrid", SimpleNamespace(load=lambda p: None)),
        ("default_encoder", lambda c, p: None),
        ("DopamineIndex", SimpleNamespace(from_connectome=lambda c: None)),
        ("RewardRouter", lambda idx, n: None),
        ("Agent", lambda **kw: built.append(kw)),
    ):
        monkeypatch.setattr(ablation, name, value)
    return built


def test_without_the_hud_the_agent_gets_no_spike_sink(monkeypatch):
    built = _stub_build_agent(monkeypatch)
    build_agent(
        Stack(
            socket="s",
            connectome_path="c",
            collision_path="x",
            calibration_path="k",
            dry_run=False,
            learn=False,
            substeps=None,
        ),
        "real",
        0,
        None,
    )
    assert built[0]["spike_sink"] is None


def test_with_the_hud_the_agent_records_spikes_into_it(monkeypatch):
    built = _stub_build_agent(monkeypatch)
    fake = FakeHud()
    build_agent(
        Stack(
            socket="s",
            connectome_path="c",
            collision_path="x",
            calibration_path="k",
            dry_run=False,
            learn=False,
            substeps=None,
            hud=True,
        ),
        "real",
        0,
        None,
        fake,
    )
    assert built[0]["spike_sink"] == fake.record_spikes


# -------------------------------------------------------------------- resets


class ResetClient:
    """Answers a reset with a fixed outcome, recording what it was asked for."""

    def __init__(self, ok: bool, landed: tuple[int, int]) -> None:
        self.ok = ok
        self.landed = landed
        self.asked: tuple[int, int] | None = None

    def send_reset(self, x: int, z: int) -> int:
        self.asked = (x, z)
        return 7

    def wait_reset(self, cmd_id: int, timeout_s: float) -> tuple[bool, int, int]:
        assert cmd_id == 7
        return self.ok, *self.landed


def test_every_episode_starts_from_the_same_tile():
    client = ResetClient(True, DEFAULT_START)
    assert reset_to_start(client, DEFAULT_START) == DEFAULT_START
    assert client.asked == DEFAULT_START


def test_a_failed_reset_is_a_dead_stack_not_a_scored_episode():
    with pytest.raises(StackDown, match="failed"):
        reset_to_start(ResetClient(False, (3300, 3190)), DEFAULT_START)


def test_a_reset_that_lands_two_tiles_away_is_refused():
    off = (DEFAULT_START[0] + 2, DEFAULT_START[1])
    with pytest.raises(StackDown, match="more than a tile"):
        reset_to_start(ResetClient(True, off), DEFAULT_START)


def test_one_tile_of_walk_tolerance_is_accepted():
    near = (DEFAULT_START[0] + 1, DEFAULT_START[1] - 1)
    assert reset_to_start(ResetClient(True, near), DEFAULT_START) == near


class SequencedResetClient(ResetClient):
    """Answers successive resets from a list of `(ok, (x, z))` outcomes."""

    def __init__(self, outcomes: list[tuple[bool, tuple[int, int]]]) -> None:
        super().__init__(*outcomes[0])
        self.outcomes = outcomes
        self.attempts = 0

    def wait_reset(self, cmd_id, timeout_s):
        ok, landed = self.outcomes[self.attempts]
        self.attempts += 1
        return ok, *landed


def test_a_reset_that_fails_once_is_retried_rather_than_killing_the_run():
    client = SequencedResetClient([(False, (3210, 3218)), (True, DEFAULT_START)])
    assert reset_to_start(client, DEFAULT_START) == DEFAULT_START
    assert client.attempts == 2


def test_two_failed_resets_raise_naming_both():
    client = SequencedResetClient([(False, (3210, 3218)), (False, (3300, 3190))])
    with pytest.raises(StackDown, match=r"\(3210, 3218\).*then.*\(3300, 3190\)"):
        reset_to_start(client, DEFAULT_START)


def test_a_diagonal_neighbour_is_accepted_without_a_retry():
    near = (DEFAULT_START[0] + 1, DEFAULT_START[1] + 1)
    client = SequencedResetClient([(True, near)])
    assert reset_to_start(client, DEFAULT_START) == near
    assert client.attempts == 1


def test_a_stack_that_drops_during_the_reset_raises():
    class Dropped(ResetClient):
        def wait_reset(self, cmd_id, timeout_s):
            raise ConnectionError("socket closed")

    with pytest.raises(StackDown, match="dropped during the reset"):
        reset_to_start(Dropped(True, DEFAULT_START), DEFAULT_START)


def test_the_start_tile_is_configurable_and_can_be_turned_off():
    args = build_parser().parse_args([])
    assert tuple(args.start) == DEFAULT_START
    assert not args.no_reset
    assert (
        Stack(
            socket="s",
            connectome_path="c",
            collision_path="x",
            calibration_path="k",
            dry_run=False,
            learn=False,
            substeps=None,
            start=(args.start[0], args.start[1]),
        ).start
        == DEFAULT_START
    )

    args = build_parser().parse_args(["--start", "3100", "3200", "--no-reset"])
    assert tuple(args.start) == (3100, 3200)
    assert args.no_reset
    assert (
        Stack(
            socket="s",
            connectome_path="c",
            collision_path="x",
            calibration_path="k",
            dry_run=False,
            learn=False,
            substeps=None,
            start=None if args.no_reset else (args.start[0], args.start[1]),
        ).start
        is None
    )


def test_the_report_says_where_every_episode_began():
    per_condition, computed = scored([10.0] * 10)
    assert "start              3222, 3218" in report(
        per_condition, computed, meta(start=(3222, 3218))
    )
    assert "not reset" in report(per_condition, computed, meta())


# ------------------------------------------------------- fresh-per-condition


def test_fresh_per_condition_flag_parses_into_stack_and_reads_both_ways_in_the_report():
    args = build_parser().parse_args(["--fresh-per-condition"])
    assert args.fresh_per_condition
    assert args.tickrate == ablation.DEFAULT_TICKRATE
    assert args.bot == ablation.DEFAULT_BOT
    assert args.client == ablation.LITE

    assert Stack(
        socket="s",
        connectome_path="c",
        collision_path="x",
        calibration_path="k",
        dry_run=False,
        learn=False,
        substeps=None,
        fresh_per_condition=True,
    ).fresh_per_condition

    per_condition, computed = full_run()
    assert "character          fresh per condition" in report(
        per_condition, computed, meta(fresh_per_condition=True)
    )
    assert "character          carried over between conditions" in report(
        per_condition, computed, meta()
    )


class StubSupervisor:
    """Records what `main()` asked of `reset_bot`, in call order."""

    def __init__(self, services, calls: list) -> None:
        self.services = services
        self.calls = calls

    def reset_bot(self, bot, client_service, sidecar_service, save_path):
        self.calls.append(("reset", bot))


def _fresh_per_condition_stubs(monkeypatch, calls: list, *, fail_on: str | None = None):
    services = [SimpleNamespace(name="lite"), SimpleNamespace(name="sidecar")]

    def make_supervisor(_services):
        sup = StubSupervisor(services, calls)
        if fail_on is not None:
            original = sup.reset_bot

            def reset_bot(bot, client_service, sidecar_service, save_path):
                original(bot, client_service, sidecar_service, save_path)
                if bot == fail_on:
                    raise ServiceFailed("sidecar: did not come back")

            sup.reset_bot = reset_bot
        return sup

    def stub_run_condition(stack, condition, episodes, ticks, seed, hud=None, feed=None):
        calls.append(("run", condition))
        return [episode([record()], condition=condition)]

    monkeypatch.setattr(ablation, "Supervisor", make_supervisor)
    monkeypatch.setattr(ablation, "default_services", lambda tickrate, bot, client: services)
    monkeypatch.setattr(ablation, "bot_save_path", lambda bot: Path("save"))
    monkeypatch.setattr(ablation, "wait_for_sidecar", lambda socket, **kw: None)
    monkeypatch.setattr(ablation, "load", lambda path: object())
    monkeypatch.setattr(ablation, "validate_conditions", lambda *a, **kw: None)
    monkeypatch.setattr(ablation, "load_calibration", lambda *a, **kw: UNCALIBRATED)
    monkeypatch.setattr(ablation, "run_condition", stub_run_condition)


def test_fresh_per_condition_resets_once_per_condition_before_its_first_episode(monkeypatch):
    calls: list = []
    _fresh_per_condition_stubs(monkeypatch, calls)

    rc = ablation.main(
        [
            "--conditions",
            "real",
            "shuffle",
            "--fresh-per-condition",
            "--episodes",
            "1",
            "--ticks",
            "1",
        ]
    )

    assert rc == 0
    assert calls == [
        ("reset", ablation.DEFAULT_BOT),
        ("run", "real"),
        ("reset", ablation.DEFAULT_BOT),
        ("run", "shuffle"),
    ]


def test_a_failed_reset_stops_the_run_before_scoring_the_condition(monkeypatch):
    calls: list = []
    _fresh_per_condition_stubs(monkeypatch, calls, fail_on=ablation.DEFAULT_BOT)

    rc = ablation.main(
        [
            "--conditions",
            "real",
            "shuffle",
            "--fresh-per-condition",
            "--episodes",
            "1",
            "--ticks",
            "1",
        ]
    )

    assert rc == 1
    assert calls == [("reset", ablation.DEFAULT_BOT)]
