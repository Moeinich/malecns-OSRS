#!/usr/bin/env python3
"""Build a numpy occupancy grid from the rs-sdk collision data.

Usage:
    uv run python tools/build_collision.py [--region NAME] [--center-x X] [--center-z Z]
        [--radius R] [--level L] [--out-dir DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COLLISION_DATA_PATH = ROOT / "vendor" / "rs-sdk" / "sdk" / "collision-data.json"

WALK_BLOCKED = 2359552

# Mainland rectangle (level 0 only) that the SDK treats as allocated/open even
# when absent from data["zones"], so routing can cross real gaps in the source data.
MAINLAND_X_MIN, MAINLAND_X_MAX = 2304, 3392
MAINLAND_Z_MIN, MAINLAND_Z_MAX = 2944, 3584


def _zone_key(level: int, x: int, z: int) -> tuple[int, int, int]:
    return (level, x & ~7, z & ~7)


def _is_mainland(level: int, zone_x: int, zone_z: int) -> bool:
    return (
        level == 0
        and MAINLAND_X_MIN <= zone_x <= MAINLAND_X_MAX
        and MAINLAND_Z_MIN <= zone_z <= MAINLAND_Z_MAX
    )


def build_grid(data: dict, center_x: int, center_z: int, radius: int, level: int) -> np.ndarray:
    """Grid indexed as grid[i, j]: i indexes x from x_min..x_max, j indexes z from z_min..z_max."""
    zones = {tuple(z) for z in data["zones"]}
    tiles = {(t[0], t[1], t[2]): t[3] for t in data["tiles"]}

    x_min, x_max = center_x - radius, center_x + radius
    z_min, z_max = center_z - radius, center_z + radius
    size = 2 * radius + 1
    grid = np.zeros((size, size), dtype=np.uint8)

    for i, x in enumerate(range(x_min, x_max + 1)):
        for j, z in enumerate(range(z_min, z_max + 1)):
            zone = _zone_key(level, x, z)
            zone_allocated = zone in zones or _is_mainland(level, zone[1], zone[2])
            if not zone_allocated:
                continue
            flags = tiles.get((level, x, z), 0)
            if (flags & WALK_BLOCKED) == 0:
                grid[i, j] = 1

    return grid, x_min, z_min


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="lumbridge")
    parser.add_argument("--center-x", type=int, default=3222)
    parser.add_argument("--center-z", type=int, default=3222)
    parser.add_argument("--radius", type=int, default=128)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "cache"))
    args = parser.parse_args()

    with open(COLLISION_DATA_PATH) as f:
        data = json.load(f)

    grid, x_min, z_min = build_grid(data, args.center_x, args.center_z, args.radius, args.level)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"collision_{args.region}.npz"
    np.savez_compressed(out_path, grid=grid, x_min=x_min, z_min=z_min, level=args.level)

    print(f"grid shape: {grid.shape}, walkable fraction: {grid.mean():.4f}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
