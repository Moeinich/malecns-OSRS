"""The fly's body: egocentric command plus position and heading -> a game action.

A body may know where it is, so unlike `decode.py` this module sees `WorldState`.
What it must never do is choose a *target*: the brain decides when to attack and
where to look, and the body only resolves the gaze. So an attack is aimed at
whatever happens to lie in the fovea, and if the fovea is empty the act is
refused. Silently retargeting onto "the nearest hostile" would hand the network
the answer, which is the exact cheat the decoder's signature exists to prevent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from flybrain.loop.types import (
    Action,
    AttackFovea,
    Eat,
    Flee,
    Idle,
    Npc,
    PickupFovea,
    Walk,
    WorldState,
)
from flybrain.motor.decode import EgocentricCommand

# The retina's central 7x7 px wedge, read as an angular window: half of 7 px
# subtended at the ~10 px radius where engagements happen. ~19 degrees.
FOVEA_HALF_ANGLE = math.atan2(3.5, 10.0)
FOVEA_RANGE_TILES = 20.0

ATTACK_OPTION = "attack"

FOOD_NAMES = frozenset(
    {
        "bread",
        "shrimps",
        "anchovies",
        "sardine",
        "herring",
        "trout",
        "salmon",
        "tuna",
        "cabbage",
        "potato",
        "onion",
        "banana",
        "cake",
        "meat pie",
        "redberry pie",
    }
)


@dataclass(frozen=True, slots=True)
class BodyParams:
    max_tiles: int = 7
    reverse_tiles: int = 3
    flee_tiles: int = 7
    fovea_half_angle: float = FOVEA_HALF_ANGLE
    fovea_range: float = FOVEA_RANGE_TILES


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _in_fovea(px: int, pz: int, tx: int, tz: int, heading: float, p: BodyParams) -> float | None:
    """Distance in tiles if the target sits inside the gaze wedge, else None."""
    dx, dz = tx - px, tz - pz
    distance = math.hypot(dx, dz)
    if distance == 0.0:
        return 0.0
    if distance > p.fovea_range:
        return None
    if abs(_wrap_pi(math.atan2(dz, dx) - heading)) > p.fovea_half_angle:
        return None
    return distance


def _step(px: int, pz: int, bearing: float, tiles: float) -> tuple[int, int]:
    return round(px + tiles * math.cos(bearing)), round(pz + tiles * math.sin(bearing))


def _is_attackable(npc: Npc) -> bool:
    return any(o.lower() == ATTACK_OPTION for o in npc.options)


def _is_food(name: str) -> bool:
    lowered = name.lower()
    return lowered in FOOD_NAMES or lowered.startswith("cooked ")


def to_action(
    cmd: EgocentricCommand,
    state: WorldState,
    heading: float,
    params: BodyParams | None = None,
) -> Action:
    if params is None:
        params = BodyParams()
    player = state.player
    if player is None:
        return Idle()
    px, pz = player.x, player.z

    if cmd.escape:
        threat = min(state.npcs, key=lambda n: math.hypot(n.x - px, n.z - pz), default=None)
        away = (
            math.atan2(pz - threat.z, px - threat.x)
            if threat is not None and (threat.x != px or threat.z != pz)
            else _wrap_pi(heading + math.pi)
        )
        x, z = _step(px, pz, away, params.flee_tiles)
        return Flee(x=x, z=z)

    if cmd.attack:
        attackable = [n for n in state.npcs if _is_attackable(n) and n.reachable]
        target = _nearest_in_fovea(attackable, px, pz, heading, params)
        return AttackFovea(npc_index=target.index) if target is not None else Idle()

    if cmd.pickup:
        item = _nearest_in_fovea(state.ground_items, px, pz, heading, params)
        return PickupFovea(x=item.x, z=item.z, item_id=item.id) if item is not None else Idle()

    if cmd.eat:
        food = next((i for i in state.inventory if _is_food(i.name)), None)
        return Eat(slot=food.slot) if food is not None else Idle()

    bearing = _wrap_pi(heading + cmd.turn)
    if cmd.reverse:
        x, z = _step(px, pz, _wrap_pi(bearing + math.pi), params.reverse_tiles)
        return Walk(x=x, z=z, running=False)

    tiles = cmd.drive * params.max_tiles
    if round(tiles) == 0:
        return Idle()
    x, z = _step(px, pz, bearing, tiles)
    if (x, z) == (px, pz):
        return Idle()
    return Walk(x=x, z=z, running=tiles >= params.max_tiles / 2)


def _nearest_in_fovea(candidates, px: int, pz: int, heading: float, params: BodyParams):
    best, best_distance = None, math.inf
    for c in candidates:
        distance = _in_fovea(px, pz, c.x, c.z, heading, params)
        if distance is not None and distance < best_distance:
            best, best_distance = c, distance
    return best
