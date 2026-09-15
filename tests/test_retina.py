from __future__ import annotations

import ast
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from flybrain.loop.types import GroundItem, Loc, Npc, Player, WorldState
from flybrain.sensory.collision import DEFAULT_COLLISION_PATH, CollisionGrid
from flybrain.sensory.retina import (
    CH_LOOT,
    CH_LUMINANCE,
    CH_RESOURCE,
    CH_THREAT,
    Retina,
)

NORTH = math.pi / 2
EAST = 0.0


# ------------------------------------------------------------------ fixtures


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


def _npc(index: int, x: int, z: int, combat_level: int = 1, in_combat: bool = False) -> Npc:
    return Npc(
        id=1,
        index=index,
        name="Chicken",
        combat_level=combat_level,
        x=x,
        z=z,
        size=1,
        distance=1,
        hp=3,
        max_hp=3,
        in_combat=in_combat,
        target_index=-1,
        reachable=True,
        options=("Attack",),
    )


def _state(player, npcs=(), items=(), locs=()) -> WorldState:
    return WorldState(
        tick=1,
        in_game=True,
        modal_open=False,
        player=player,
        npcs=tuple(npcs),
        ground_items=tuple(items),
        locs=tuple(locs),
        inventory=(),
        skills={},
        op_rejected_count=0,
    )


def _blank_grid(walkable: bool = False) -> CollisionGrid:
    fill = 1 if walkable else 0
    return CollisionGrid(np.full((200, 200), fill, dtype=np.uint8), x_min=0, z_min=0, level=0)


def _blob_grid(x: int, z: int, half: int = 2) -> CollisionGrid:
    grid = np.zeros((200, 200), dtype=np.uint8)
    grid[x - half : x + half + 1, z - half : z + half + 1] = 1
    return CollisionGrid(grid, x_min=0, z_min=0, level=0)


def _centroid(frame: np.ndarray, channel: int) -> tuple[float, float]:
    plane = frame[:, :, channel]
    total = plane.sum()
    assert total > 0, "channel is empty; the test fixture is wrong"
    rows, cols = np.indices(plane.shape)
    return float((rows * plane).sum() / total), float((cols * plane).sum() / total)


# ------------------------------------------------------------------ firewall


def test_sensory_modules_never_reach_the_motor_layer():
    allowed = {"flybrain.loop.types", "flybrain.sensory.collision", "flybrain.sensory.heading"}
    for name in ("heading.py", "retina.py"):
        path = Path(__file__).resolve().parent.parent / "flybrain" / "sensory" / name
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("flybrain"):
                assert node.module in allowed, (name, node.module)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("flybrain"), (name, alias.name)


# ------------------------------------------------------------------ geometry


def test_npc_ahead_lands_centre_top_and_moves_aside_on_a_quarter_turn():
    retina = Retina()
    centre = (retina.size - 1) / 2.0
    state = _state(_player(100, 100), npcs=[_npc(1, 100, 110)])
    grid = _blank_grid()

    ahead = retina.render(state, grid, NORTH)
    row, col = _centroid(ahead, CH_THREAT)
    assert abs(col - centre) < 1.0
    assert row < centre - 10.0

    aside = retina.render(state, grid, EAST)
    row_e, col_e = _centroid(aside, CH_THREAT)
    assert abs(row_e - centre) < 1.0
    assert col_e < centre - 10.0


def test_forward_translation_shifts_the_scene_backwards_uniformly():
    """Top-down orthographic egocentric projection: translation gives translational
    flow, not radial expansion. See the report note — expansion needs perspective."""
    retina = Retina()
    grid = _blob_grid(100, 114)
    prev = _state(_player(100, 100))
    cur = _state(_player(100, 104))

    frames = retina.render_subframes(prev, cur, grid, NORTH)
    rows = [_centroid(f, CH_LUMINANCE)[0] for f in frames]
    cols = [_centroid(f, CH_LUMINANCE)[1] for f in frames]

    assert all(b > a for a, b in pairwise(rows)), rows  # scene recedes, monotonically
    step = 1.0 * retina.px_per_tile  # 1 tile per sub-frame
    assert all(abs((b - a) - step) < 0.4 for a, b in pairwise(rows)), rows
    assert max(cols) - min(cols) < 0.25, cols  # no rotational component


def test_yaw_produces_rotational_shift_at_constant_radius():
    retina = Retina()
    centre = (retina.size - 1) / 2.0
    grid = _blob_grid(100, 114)
    state = _state(_player(100, 100))

    frames = retina.render_subframes(state, state, grid, EAST, prev_heading=NORTH)
    angles, radii = [], []
    for f in frames:
        row, col = _centroid(f, CH_LUMINANCE)
        dy, dx = centre - row, col - centre
        angles.append(math.atan2(dy, dx))
        radii.append(math.hypot(dy, dx))

    unwrapped = np.unwrap(angles)
    assert all(b > a for a, b in pairwise(unwrapped)), angles
    assert abs((unwrapped[-1] - unwrapped[0]) - (3 / 4) * (math.pi / 2)) < 0.15
    assert max(radii) - min(radii) < 0.1 * float(np.mean(radii))


def test_subframes_are_distinct_and_correctly_shaped():
    retina = Retina()
    grid = _blob_grid(100, 112)
    prev = _state(_player(100, 100), npcs=[_npc(1, 104, 100)])
    cur = _state(_player(100, 104), npcs=[_npc(1, 104, 104)])

    frames = retina.render_subframes(prev, cur, grid, NORTH, n=4)
    assert frames.shape == (4, retina.size, retina.size, 4)
    assert frames.dtype == np.float32
    for a, b in pairwise(frames):
        assert not np.array_equal(a, b)


# ------------------------------------------------------------------ channels


def test_channels_are_populated_and_bounded():
    retina = Retina()
    grid = _blank_grid(walkable=True)
    state = _state(
        _player(100, 100),
        npcs=[_npc(1, 100, 106, combat_level=9, in_combat=True)],
        items=[GroundItem(id=1, name="Bones", count=1, x=102, z=100, distance=2, reachable=True)],
        locs=[Loc(id=1, name="Tree", x=98, z=100, distance=2, options=("Chop down",))],
    )
    frame = retina.render(state, grid, NORTH)

    assert frame.shape == (retina.size, retina.size, 4)
    assert frame.min() >= 0.0 and frame.max() <= 1.0
    for ch in (CH_LUMINANCE, CH_THREAT, CH_LOOT, CH_RESOURCE):
        assert frame[:, :, ch].max() > 0.0, ch


def test_threat_scales_with_relative_combat_level():
    retina = Retina()
    grid = _blank_grid()
    weak = retina.render(_state(_player(100, 100), [_npc(1, 100, 106)]), grid, NORTH)
    strong = retina.render(
        _state(_player(100, 100), [_npc(1, 100, 106, combat_level=6)]), grid, NORTH
    )
    assert strong[:, :, CH_THREAT].sum() > weak[:, :, CH_THREAT].sum()


def test_locs_without_a_useful_option_are_not_resources():
    retina = Retina()
    grid = _blank_grid()
    state = _state(
        _player(100, 100),
        locs=[Loc(id=1, name="Wall", x=98, z=100, distance=2, options=("Examine",))],
    )
    assert retina.render(state, grid, NORTH)[:, :, CH_RESOURCE].max() == 0.0


def test_render_is_deterministic():
    retina = Retina()
    grid = _blob_grid(100, 110)
    state = _state(_player(100, 100), npcs=[_npc(1, 103, 107)])
    a = retina.render(state, grid, 0.7)
    b = retina.render(state, grid, 0.7)
    assert np.array_equal(a, b)


@pytest.mark.skipif(
    not DEFAULT_COLLISION_PATH.exists(),
    reason=f"{DEFAULT_COLLISION_PATH} missing; run tools/build_collision.py first",
)
def test_lumbridge_luminance_is_neither_empty_nor_solid():
    retina = Retina()
    grid = CollisionGrid.load(DEFAULT_COLLISION_PATH)
    frame = retina.render(_state(_player(3222, 3222)), grid, NORTH)
    luminance = frame[:, :, CH_LUMINANCE]
    assert 0.0 < luminance.mean() < 1.0
    assert luminance.max() > 0.9
    assert luminance.min() < 0.1
