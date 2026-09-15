from __future__ import annotations

import pytest

from flybrain.sensory.collision import DEFAULT_COLLISION_PATH, CollisionGrid

if not DEFAULT_COLLISION_PATH.exists():
    pytest.skip(
        f"{DEFAULT_COLLISION_PATH} missing; run tools/build_collision.py first",
        allow_module_level=True,
    )


def _grid() -> CollisionGrid:
    return CollisionGrid.load(DEFAULT_COLLISION_PATH)


def test_lumbridge_spawn_is_walkable():
    grid = _grid()
    assert grid.is_walkable(3222, 3222) is True


def test_patch_has_plausible_mix():
    grid = _grid()
    patch = grid.walkable_patch(3222, 3222, radius=20)
    assert patch.min() == 0.0
    assert patch.max() == 1.0
    assert 0 < patch.mean() < 1


def test_out_of_bounds_is_blocked():
    grid = _grid()
    assert grid.is_walkable(0, 0) is False

    far_x = grid.x_min - 10_000
    far_z = grid.z_min - 10_000
    patch = grid.walkable_patch(far_x, far_z, radius=5)
    assert patch.max() == 0.0


def test_patch_agrees_with_is_walkable():
    grid = _grid()
    cx, cz = 3222, 3222
    radius = 15
    patch = grid.walkable_patch(cx, cz, radius)
    for dx in (-15, -5, 0, 5, 15):
        for dz in (-15, -5, 0, 5, 15):
            i, j = dx + radius, dz + radius
            expected = grid.is_walkable(cx + dx, cz + dz)
            assert bool(patch[i, j]) == expected, (dx, dz)
