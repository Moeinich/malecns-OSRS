from __future__ import annotations

import math

from flybrain.loop.types import (
    AttackFovea,
    Eat,
    Flee,
    GroundItem,
    Idle,
    Item,
    Npc,
    PickupFovea,
    Player,
    Walk,
    WorldState,
)
from flybrain.motor.body import BodyParams, to_action
from flybrain.motor.decode import EgocentricCommand

NORTH = math.pi / 2
EAST = 0.0


def _cmd(**kw) -> EgocentricCommand:
    base = {
        "turn": 0.0,
        "drive": 0.0,
        "reverse": False,
        "escape": False,
        "attack": False,
        "eat": False,
        "pickup": False,
    }
    base.update(kw)
    return EgocentricCommand(**base)


def _player(x: int = 100, z: int = 100) -> Player:
    return Player(
        name="flybot",
        combat_level=3,
        hp=10,
        max_hp=10,
        x=x,
        z=z,
        level=0,
        run_energy=100,
        anim_id=-1,
        in_combat=False,
        target_index=-1,
        target_type="none",
        is_dead=False,
        life_id=1,
    )


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
        in_combat=False,
        target_index=-1,
        reachable=True,
        options=("Attack",),
    )


def _state(npcs=(), items=(), inventory=()) -> WorldState:
    return WorldState(
        tick=1,
        in_game=True,
        modal_open=False,
        player=_player(),
        npcs=tuple(npcs),
        ground_items=tuple(items),
        locs=(),
        inventory=tuple(inventory),
        skills={},
        op_rejected_count=0,
    )


# ---------------------------------------------------------------- the fovea


def test_an_npc_inside_the_fovea_wedge_is_attacked():
    state = _state([_npc(7, 100, 106)])  # 6 tiles north, dead ahead
    assert to_action(_cmd(attack=True), state, NORTH) == AttackFovea(npc_index=7)


def test_the_same_npc_outside_the_wedge_is_not_attacked_even_when_nearest():
    """The assertion the whole fovea rule exists for.

    Identical NPC, identical distance, only the gaze differs. Silently
    retargeting onto the nearest hostile would hand the network the answer.
    """
    state = _state([_npc(7, 100, 106)])
    assert to_action(_cmd(attack=True), state, NORTH) == AttackFovea(npc_index=7)
    assert to_action(_cmd(attack=True), state, EAST) == Idle()


def test_the_nearer_of_two_npcs_in_the_wedge_wins():
    state = _state([_npc(7, 100, 112), _npc(9, 100, 104)])
    assert to_action(_cmd(attack=True), state, NORTH) == AttackFovea(npc_index=9)


def test_pickup_obeys_the_same_wedge():
    item = GroundItem(id=526, name="Bones", count=1, x=100, z=105, distance=5, reachable=True)
    state = _state(items=[item])
    assert to_action(_cmd(pickup=True), state, NORTH) == PickupFovea(x=100, z=105, item_id=526)
    assert to_action(_cmd(pickup=True), state, EAST) == Idle()


def test_an_empty_fovea_refuses_rather_than_retargets():
    assert to_action(_cmd(attack=True), _state(), NORTH) == Idle()
    assert to_action(_cmd(pickup=True), _state(), NORTH) == Idle()


# ------------------------------------------------------------- locomotion


def test_heading_and_drive_produce_the_expected_target_tile():
    params = BodyParams()
    action = to_action(_cmd(drive=1.0), _state(), NORTH, params)
    assert action == Walk(x=100, z=100 + params.max_tiles, running=True)

    east = to_action(_cmd(drive=1.0), _state(), EAST, params)
    assert east == Walk(x=100 + params.max_tiles, z=100, running=True)


def test_turn_rotates_the_target_tile():
    action = to_action(_cmd(drive=1.0, turn=math.pi / 2), _state(), EAST)
    assert isinstance(action, Walk)
    assert (action.x, action.z) == (100, 107)


def test_zero_drive_is_idle_and_a_short_step_walks():
    assert to_action(_cmd(drive=0.0), _state(), NORTH) == Idle()
    action = to_action(_cmd(drive=0.3), _state(), NORTH)
    assert action == Walk(x=100, z=102, running=False)


def test_reverse_walks_behind():
    params = BodyParams()
    action = to_action(_cmd(reverse=True, drive=1.0), _state(), NORTH, params)
    assert action == Walk(x=100, z=100 - params.reverse_tiles, running=False)


# ------------------------------------------------------------------ reflexes


def test_escape_flees_directly_away_from_the_nearest_threat():
    params = BodyParams()
    action = to_action(_cmd(escape=True), _state([_npc(7, 100, 106)]), NORTH, params)
    assert action == Flee(x=100, z=100 - params.flee_tiles)


def test_escape_outranks_every_other_act():
    state = _state([_npc(7, 100, 106)])
    action = to_action(_cmd(escape=True, attack=True, drive=1.0), state, NORTH)
    assert isinstance(action, Flee)


def test_eat_picks_a_food_slot_and_refuses_when_there_is_none():
    food = Item(slot=3, id=2309, name="Bread", count=1)
    junk = Item(slot=0, id=526, name="Bones", count=1)
    assert to_action(_cmd(eat=True), _state(inventory=[junk, food]), NORTH) == Eat(slot=3)
    assert to_action(_cmd(eat=True), _state(inventory=[junk]), NORTH) == Idle()
