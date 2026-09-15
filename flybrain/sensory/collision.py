from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_COLLISION_PATH = ROOT / "data" / "cache" / "collision_lumbridge.npz"


class CollisionGrid:
    """Occupancy grid loaded from a build_collision.py npz.

    grid[i, j]: i indexes x from x_min..x_max, j indexes z from z_min..z_max.
    """

    def __init__(self, grid: np.ndarray, x_min: int, z_min: int, level: int):
        self.grid = grid
        self.x_min = x_min
        self.z_min = z_min
        self.level = level

    @classmethod
    def load(cls, path: Path | str = DEFAULT_COLLISION_PATH) -> CollisionGrid:
        with np.load(path) as data:
            return cls(
                grid=data["grid"],
                x_min=int(data["x_min"]),
                z_min=int(data["z_min"]),
                level=int(data["level"]),
            )

    def is_walkable(self, x: int, z: int) -> bool:
        i, j = x - self.x_min, z - self.z_min
        if 0 <= i < self.grid.shape[0] and 0 <= j < self.grid.shape[1]:
            return bool(self.grid[i, j])
        return False

    def walkable_patch(self, cx: int, cz: int, radius: int) -> np.ndarray:
        size = 2 * radius + 1
        patch = np.zeros((size, size), dtype=np.float32)

        x_lo, x_hi = cx - radius, cx + radius
        z_lo, z_hi = cz - radius, cz + radius

        src_i0 = max(x_lo, self.x_min)
        src_i1 = min(x_hi, self.x_min + self.grid.shape[0] - 1)
        src_j0 = max(z_lo, self.z_min)
        src_j1 = min(z_hi, self.z_min + self.grid.shape[1] - 1)

        if src_i0 > src_i1 or src_j0 > src_j1:
            return patch

        dst_i0, dst_i1 = src_i0 - x_lo, src_i1 - x_lo
        dst_j0, dst_j1 = src_j0 - z_lo, src_j1 - z_lo

        grid_i0, grid_i1 = src_i0 - self.x_min, src_i1 - self.x_min
        grid_j0, grid_j1 = src_j0 - self.z_min, src_j1 - self.z_min

        patch[dst_i0 : dst_i1 + 1, dst_j0 : dst_j1 + 1] = self.grid[
            grid_i0 : grid_i1 + 1, grid_j0 : grid_j1 + 1
        ].astype(np.float32)

        return patch
