from __future__ import annotations

import math
import random

from flybrain.loop.types import Npc, Player, WorldState
from flybrain.sensory.heading import Heading, wrap_pi


def _player(x: int, z: int, **kw) -> Player:
    base = {
        "name": "flybot",
        "combat_level": 3,
        "hp": 10,
        "max_hp": 10,
        "x": x,
        "z": z,
        "level": 0,
        "run_energy": 100,
        "anim_id": -1,
        "in_combat": False,
        "target_index": -1,
        "target_type": "none",
        "is_dead": False,
        "life_id": 1,
    }
    base.update(kw)
    return Player(**base)


def _npc(index: int, x: int, z: int) -> Npc:
    return Npc(
        id=1,
        index=index,
        name="Chicken",
        combat_level=1,
        x=x,
        z=z,
        size=1,
        distance=1,
        hp=3,
        max_hp=3,
        in_combat=True,
        target_index=0,
        reachable=True,
        options=("Attack",),
    )


def _state(player: Player, npcs=()) -> WorldState:
    return WorldState(
        tick=1,
        in_game=True,
        modal_open=False,
        player=player,
        npcs=tuple(npcs),
        ground_items=(),
        locs=(),
        inventory=(),
        skills={},
        op_rejected_count=0,
    )


def _fixed() -> Heading:
    h = Heading(rng=random.Random(0))
    return h


def test_wrap_pi_is_a_half_open_interval():
    assert wrap_pi(math.pi) == math.pi or wrap_pi(math.pi) == -math.pi
    assert math.isclose(wrap_pi(3 * math.pi / 2), -math.pi / 2)
    assert math.isclose(wrap_pi(-3 * math.pi / 2), math.pi / 2)
    assert math.isclose(wrap_pi(0.3), 0.3)
    for a in (-10.0, -4.0, 0.0, 4.0, 10.0):
        assert -math.pi <= wrap_pi(a) <= math.pi


def test_turn_takes_the_short_way_across_the_pi_boundary():
    h = _fixed()
    h.update(_state(_player(0, 0)))
    h.theta = 3.0
    # Bearing ~-2.99 rad: the short way is forward across +pi, not back through 0.
    p = _player(0, 0, in_combat=True, target_index=7, target_type="npc")
    theta = h.update(_state(p, [_npc(7, -20, -3)]))
    assert abs(theta) > 3.0
    assert -math.pi <= theta <= math.pi


def test_combat_target_wins_over_displacement():
    h = _fixed()
    p0 = _player(0, 0)
    h.update(_state(p0))
    # Walk east for a few ticks so the EMA points east.
    for i in range(1, 5):
        h.update(_state(_player(i, 0)))
    assert abs(wrap_pi(h.theta - 0.0)) < abs(wrap_pi(h.theta - math.pi / 2))

    p = _player(4, 0, in_combat=True, target_index=7, target_type="npc")
    before = h.theta
    after = h.update(_state(p, [_npc(7, 4, 20)]))
    assert wrap_pi(after - before) > 0.0  # swung toward north, away from the EMA


def test_displacement_ema_smooths_the_eight_way_quantisation():
    h = _fixed()
    h.update(_state(_player(0, 0)))
    h.theta = 0.0
    # Alternating E / NE steps average to ~22.5 degrees; neither raw step is that.
    for k in range(1, 21):
        h.update(_state(_player(k, k // 2)))
    assert 0.15 < h.theta < 0.6


def test_stationary_holds_the_previous_heading():
    h = _fixed()
    h.update(_state(_player(5, 5)))
    for i in range(1, 4):
        h.update(_state(_player(5 + i, 5)))
    for _ in range(10):
        h.update(_state(_player(8, 5)))
    held = h.theta  # the displacement EMA has decayed below threshold by now
    for _ in range(10):
        h.update(_state(_player(8, 5)))
    assert math.isclose(h.theta, held, abs_tol=1e-12)


def test_respawn_reseeds():
    h = Heading(rng=random.Random(1))
    h.update(_state(_player(5, 5)))
    first = h.theta
    h.update(_state(_player(5, 5, life_id=2)))
    assert h.theta != first


def test_no_player_is_survivable():
    h = _fixed()
    theta = h.update(_state(None))
    assert -math.pi <= theta <= math.pi
