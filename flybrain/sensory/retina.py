"""Egocentric heading-aligned raster of the world state.

Four channels, float32 in [0, 1], player at the centre, forward (the heading) up.
This is a metaphor and is labelled one: a top-down tile raster is not what a fly
eye sees. The flow it produces is nonetheless real flow, computed nowhere — it
exists only as the difference between sub-frames and is left for T4/T5 to extract.

Nothing here may import from `flybrain.motor`, and nothing here knows that
actions exist — see AGENTS.md.
"""

from __future__ import annotations

import math

import numpy as np

from flybrain.loop.types import Loc, Npc, Player, WorldState
from flybrain.sensory.collision import CollisionGrid
from flybrain.sensory.heading import wrap_pi

CH_LUMINANCE = 0
CH_THREAT = 1
CH_LOOT = 2
CH_RESOURCE = 3
N_CHANNELS = 4

DEFAULT_SIZE = 60
# The plan's "60x60 at ~3 px/tile, ~20 tile radius" is arithmetically impossible
# (3 px/tile over 60 px is a 10-tile radius). The radius is the load-bearing half:
# the view has to reach past the ~15-tile state horizon, so resolution gives way.
DEFAULT_PX_PER_TILE = 1.5

SUBFRAMES = 4

RESOURCE_OPTIONS = frozenset(
    {
        "chop down",
        "mine",
        "fish",
        "net",
        "bait",
        "lure",
        "cage",
        "harpoon",
        "pick",
        "pick-fruit",
        "milk",
        "cook",
        "smelt",
        "smith",
        "spin",
        "fill",
        "drink",
        "take",
    }
)

ATTACK_OPTION = "attack"


def _is_resource(loc: Loc) -> bool:
    return any(o.lower() in RESOURCE_OPTIONS for o in loc.options)


def _is_attackable(npc: Npc) -> bool:
    return any(o.lower() == ATTACK_OPTION for o in npc.options)


def _threat_intensity(npc: Npc, player: Player | None) -> float:
    level = max(player.combat_level, 1) if player is not None else 1
    ratio = min(npc.combat_level / level, 2.0)
    value = 0.2 + 0.3 * ratio
    if npc.in_combat:
        value *= 1.5
    return min(value, 1.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class Retina:
    def __init__(self, size: int = DEFAULT_SIZE, px_per_tile: float = DEFAULT_PX_PER_TILE):
        self.size = size
        self.px_per_tile = px_per_tile
        self.view_radius = size / (2.0 * px_per_tile)
        # The sampling grid is rotated, so the source patch must cover the diagonal.
        self.patch_radius = math.ceil(self.view_radius * math.sqrt(2.0)) + 2
        self._patch_size = 2 * self.patch_radius + 1

        centre = (size - 1) / 2.0
        axis = (np.arange(size, dtype=np.float32) - centre) / px_per_tile
        self._right = axis[None, :]  # +x in image space, tiles
        self._forward = -axis[:, None]  # row 0 is straight ahead

    # -------------------------------------------------------------- public

    def render(
        self,
        state: WorldState,
        collision: CollisionGrid,
        heading: float,
        subframe_t: float = 1.0,
        prev_state: WorldState | None = None,
        prev_heading: float | None = None,
    ) -> np.ndarray:
        prev = prev_state if prev_state is not None else state
        th0 = heading if prev_heading is None else prev_heading
        return self._render_at(prev, state, collision, th0, heading, subframe_t)

    def render_subframes(
        self,
        prev_state: WorldState | None,
        state: WorldState,
        collision: CollisionGrid,
        heading: float,
        n: int = SUBFRAMES,
        *,
        prev_heading: float | None = None,
    ) -> np.ndarray:
        """`n` frames spanning one game tick.

        HR correlators have 20-50 ms delay lines; one frame per 400 ms tick carries
        no velocity information at all. Four sub-frames put the spacing at ~100 ms.
        """
        prev = prev_state if prev_state is not None else state
        th0 = heading if prev_heading is None else prev_heading
        out = np.empty((n, self.size, self.size, N_CHANNELS), dtype=np.float32)
        for k in range(n):
            out[k] = self._render_at(prev, state, collision, th0, heading, (k + 1) / n)
        return out

    # ------------------------------------------------------------- internal

    def _render_at(
        self,
        prev_state: WorldState,
        state: WorldState,
        collision: CollisionGrid,
        th0: float,
        th1: float,
        t: float,
    ) -> np.ndarray:
        player, prev_player = state.player, prev_state.player
        if player is None:
            return np.zeros((self.size, self.size, N_CHANNELS), dtype=np.float32)
        if prev_player is None:
            prev_player = player

        px = _lerp(prev_player.x, player.x, t)
        pz = _lerp(prev_player.z, player.z, t)
        theta = wrap_pi(th0 + t * wrap_pi(th1 - th0))

        cx, cz = round(px), round(pz)
        x0, z0 = cx - self.patch_radius, cz - self.patch_radius
        patch = self._build_patch(prev_state, state, collision, t, cx, cz, x0, z0)
        return self._sample(patch, theta, px - x0, pz - z0)

    def _build_patch(
        self,
        prev_state: WorldState,
        state: WorldState,
        collision: CollisionGrid,
        t: float,
        cx: int,
        cz: int,
        x0: int,
        z0: int,
    ) -> np.ndarray:
        """Channel-first `(4, S, S)` in the world frame; the rotation is a resampling of it."""
        patch = np.zeros((N_CHANNELS, self._patch_size, self._patch_size), dtype=np.float32)
        patch[CH_LUMINANCE] = collision.walkable_patch(cx, cz, self.patch_radius)

        prev_npcs = {n.index: n for n in prev_state.npcs}
        for npc in state.npcs:
            # An NPC with no Attack option is scenery: the engine refuses the act
            # silently, so painting it teaches the fly a move that never lands.
            if not _is_attackable(npc):
                continue
            was = prev_npcs.get(npc.index)
            nx = _lerp(was.x, npc.x, t) if was is not None else float(npc.x)
            nz = _lerp(was.z, npc.z, t) if was is not None else float(npc.z)
            self._splat(patch, CH_THREAT, nx - x0, nz - z0, _threat_intensity(npc, state.player))

        for item in state.ground_items:
            self._splat(patch, CH_LOOT, item.x - x0, item.z - z0, 1.0)

        for loc in state.locs:
            if _is_resource(loc):
                self._splat(patch, CH_RESOURCE, loc.x - x0, loc.z - z0, 1.0)

        np.clip(patch, 0.0, 1.0, out=patch)
        return patch

    def _splat(self, patch: np.ndarray, ch: int, fi: float, fj: float, value: float) -> None:
        """Bilinear splat in the world frame, so sub-tile motion is continuous."""
        i0, j0 = math.floor(fi), math.floor(fj)
        ti, tj = fi - i0, fj - j0
        n = self._patch_size
        for di, wi in ((0, 1.0 - ti), (1, ti)):
            for dj, wj in ((0, 1.0 - tj), (1, tj)):
                i, j = i0 + di, j0 + dj
                if 0 <= i < n and 0 <= j < n:
                    patch[ch, i, j] += value * wi * wj

    def _sample(self, patch: np.ndarray, theta: float, pi: float, pj: float) -> np.ndarray:
        cos, sin = math.cos(theta), math.sin(theta)
        fi = pi + self._forward * cos + self._right * sin
        fj = pj + self._forward * sin - self._right * cos

        n = self._patch_size
        i0 = np.floor(fi)
        j0 = np.floor(fj)
        ti = fi - i0
        tj = fj - j0
        i0 = i0.astype(np.int32)
        j0 = j0.astype(np.int32)

        inside = (i0 >= 0) & (i0 < n - 1) & (j0 >= 0) & (j0 < n - 1)
        # One flat index array over a 2-D base is markedly faster than a pair of
        # index arrays over a 3-D one; the corners are then constant offsets.
        base = np.clip(i0, 0, n - 2) * n + np.clip(j0, 0, n - 2)
        flat = patch.reshape(N_CHANNELS, -1)

        out = (
            ((1.0 - ti) * (1.0 - tj)) * flat[:, base]
            + (ti * (1.0 - tj)) * flat[:, base + n]
            + ((1.0 - ti) * tj) * flat[:, base + 1]
            + (ti * tj) * flat[:, base + n + 1]
        )
        out *= inside
        np.clip(out, 0.0, 1.0, out=out)
        return np.ascontiguousarray(out.transpose(1, 2, 0))


__all__ = [
    "CH_LOOT",
    "CH_LUMINANCE",
    "CH_RESOURCE",
    "CH_THREAT",
    "N_CHANNELS",
    "Retina",
]
