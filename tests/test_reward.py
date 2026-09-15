"""Reward routing: game events -> current in the real DAN cells -> a dopamine rate."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from flybrain.connectome.loader import Connectome
from flybrain.engine.lif import LIFEngine
from flybrain.loop.types import parse_server_message
from flybrain.reward import (
    EVENT_KINDS,
    DopamineIndex,
    RewardParams,
    RewardRouter,
    UnknownEvent,
)

PAM = np.arange(0, 8)
PPL1 = np.arange(8, 12)
OTHER = np.arange(12, 20)
N = 20


def _connectome(populations: dict[str, np.ndarray] | None = None) -> Connectome:
    pops = (
        {"PAM": PAM.astype(np.int64), "PPL1": PPL1.astype(np.int64)}
        if populations is None
        else populations
    )
    return Connectome(
        W=sp.csc_matrix((N, N), dtype=np.float32),
        body_ids=np.arange(N, dtype=np.int64),
        populations=pops,
        provenance={"synthetic": True},
    )


def _router(params: RewardParams | None = None) -> RewardRouter:
    return RewardRouter(DopamineIndex.from_connectome(_connectome()), N, params)


def _combat(kind: str, damage: int = 0) -> dict:
    return {
        "tick": 1,
        "observationId": None,
        "type": kind,
        "damage": damage,
        "sourceType": "player",
        "sourceIndex": 0,
        "targetType": "npc",
        "targetIndex": 1,
    }


def _reward_msg(revision: int, events: list[dict], xp: dict[str, int] | None = None) -> dict:
    return {
        "t": "reward",
        "revision": revision,
        "combatEvents": events,
        "xpDelta": xp or {},
    }


def _player(hp: int = 10, is_dead: bool = False, life_id: int = 1):
    from flybrain.loop.types import Player

    return Player(
        name="flybot01",
        combat_level=3,
        hp=hp,
        max_hp=10,
        x=3200,
        z=3200,
        level=0,
        run_energy=100,
        anim_id=-1,
        in_combat=False,
        target_index=-1,
        target_type="none",
        is_dead=is_dead,
        life_id=life_id,
    )


# ------------------------------------------------------------------- routing


@pytest.mark.parametrize("kind", ["damage_dealt", "kill", "xp"])
def test_appetitive_events_drive_pam_and_only_pam(kind):
    r = _router()
    r.register_event(kind, 1.0)
    current = r.current()
    assert (current[PAM] > 0.0).all()
    assert (current[PPL1] == 0.0).all()
    assert (current[OTHER] == 0.0).all()


@pytest.mark.parametrize("kind", ["damage_taken", "death", "hp_drop"])
def test_aversive_events_drive_ppl1_and_only_ppl1(kind):
    r = _router()
    r.register_event(kind, 1.0)
    current = r.current()
    assert (current[PPL1] > 0.0).all()
    assert (current[PAM] == 0.0).all()


def test_the_populations_are_resolved_by_name_never_by_index():
    index = DopamineIndex.from_connectome(_connectome())
    assert np.array_equal(index.appetitive, PAM)
    assert np.array_equal(index.aversive, PPL1)


def test_a_build_without_the_dan_populations_raises_rather_than_injecting_nowhere():
    with pytest.raises(KeyError):
        DopamineIndex.from_connectome(_connectome({"MBON": np.arange(3)}))
    with pytest.raises(KeyError, match="no neurons"):
        DopamineIndex.from_connectome(
            _connectome({"PAM": np.array([], dtype=np.int64), "PPL1": PPL1})
        )


def test_an_unrouted_event_kind_is_refused():
    r = _router()
    with pytest.raises(UnknownEvent):
        r.register_event("loot_dropped", 1.0)
    assert set(EVENT_KINDS) == {
        "damage_dealt",
        "kill",
        "xp",
        "damage_taken",
        "death",
        "hp_drop",
    }


def test_register_event_is_the_seam_every_source_goes_through():
    r = _router()
    assert r.events == 0
    r.observe_reward(
        parse_server_message(
            _reward_msg(1, [_combat("damage_dealt", 4), _combat("kill")], {"Attack": 160})
        )
    )
    assert r.events == 3
    assert r.appetitive_drive > 0.0
    assert r.aversive_drive == 0.0


# ------------------------------------------------------- the message and the state


def test_a_bridge_reward_message_routes_both_ways():
    r = _router()
    msg = parse_server_message(
        _reward_msg(7, [_combat("damage_dealt", 6), _combat("damage_taken", 3)])
    )
    assert r.observe_reward(msg)
    assert r.appetitive_drive == pytest.approx(0.6)
    assert r.aversive_drive == pytest.approx(0.3)


def test_the_same_reward_message_is_not_injected_twice():
    r = _router()
    msg = parse_server_message(_reward_msg(7, [_combat("kill")]))
    assert r.observe_reward(msg)
    assert not r.observe_reward(msg)
    assert r.events == 1


def test_negative_xp_is_not_a_reward():
    r = _router()
    r.observe_reward(parse_server_message(_reward_msg(1, [], {"Hitpoints": -50})))
    assert r.appetitive_drive == 0.0
    assert r.aversive_drive == 0.0


def test_an_hp_drop_within_one_life_is_aversive():
    r = _router()
    r.observe_state(_player(hp=10), _player(hp=6))
    assert r.aversive_drive == pytest.approx(0.4)
    assert r.appetitive_drive == 0.0


def test_a_respawn_is_not_read_as_healing_or_as_damage():
    """`hp` jumps on a new life; the life id is what stops it becoming an event."""
    r = _router()
    r.observe_state(_player(hp=1, life_id=1), _player(hp=10, life_id=2))
    assert r.events == 0


def test_death_is_aversive_exactly_once():
    r = _router()
    r.observe_state(_player(), _player(hp=0, is_dead=True))
    r.observe_state(_player(hp=0, is_dead=True), _player(hp=0, is_dead=True))
    assert r.events == 1


# ------------------------------------------------------------------ the current


def test_the_drive_saturates_rather_than_growing_without_bound():
    p = RewardParams(i_dan=30.0)
    r = _router(p)
    r.register_event("kill", 1000.0)
    assert r.current()[PAM].max() == pytest.approx(30.0, rel=1e-3)


def test_dopamine_outlasts_the_event_but_decays():
    r = _router(RewardParams(retention=0.5))
    r.register_event("kill", 1.0)
    first = r.current()[PAM].max()
    r.decay()
    second = r.current()[PAM].max()
    assert 0.0 < second < first


# ----------------------------------------------------------------- the dopamine


def test_the_dopamine_term_is_the_dan_firing_rate_not_the_registered_magnitude():
    r = _router()
    rates = np.zeros(N, dtype=np.float32)
    r.dopamine(rates)  # baseline at the resting rate

    r.register_event("kill", 100.0)
    # The events are in; the cells have not fired, so there is no dopamine yet.
    assert r.dopamine(rates) == pytest.approx(0.0)

    rates[PAM] = 40.0
    assert r.dopamine(rates) > 0.0


def test_appetitive_and_aversive_cancel_in_the_cells():
    r = _router()
    rates = np.zeros(N, dtype=np.float32)
    r.dopamine(rates)
    rates[PAM] = 25.0
    rates[PPL1] = 25.0
    assert r.dopamine(rates) == pytest.approx(0.0)
    rates[PPL1] = 40.0
    assert r.dopamine(rates) < 0.0


def test_the_first_reading_seeds_the_baseline_instead_of_teaching():
    r = _router()
    rates = np.zeros(N, dtype=np.float32)
    rates[PAM] = 12.0
    assert r.baseline is None
    assert r.dopamine(rates) == 0.0
    assert r.baseline == pytest.approx(12.0)


def test_a_constant_dan_rate_stops_teaching():
    r = _router(RewardParams(baseline_ticks=4.0))
    rates = np.zeros(N, dtype=np.float32)
    r.dopamine(rates)
    rates[PAM] = 30.0
    terms = [r.dopamine(rates) for _ in range(40)]
    assert terms[0] > 0.0
    assert terms[-1] < terms[0] * 0.1


# ------------------------------------------------- reward reaching the real cells


def test_the_injected_current_actually_fires_the_dan_population():
    """The whole point: reward becomes current in named cells, then a rate."""
    W = sp.csc_matrix((N, N), dtype=np.float32)
    engine = LIFEngine(W, rate_window_ms=200.0, seed=0)
    r = _router()

    quiet = engine.get_firing_rates()
    for _ in range(200):
        engine.step(r.current())
    assert engine.get_firing_rates()[PAM].mean() == 0.0

    r.register_event("damage_dealt", 5.0)
    current = r.current()
    for _ in range(200):
        engine.step(current)
    rates = engine.get_firing_rates()
    assert rates[PAM].mean() > 0.0
    assert rates[PPL1].mean() == 0.0
    assert rates[OTHER].mean() == 0.0
    assert quiet.sum() == 0.0
