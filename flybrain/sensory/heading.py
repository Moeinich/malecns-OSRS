"""Synthesised body heading. OSRS exposes no facing angle, so derive one.

Angles are world-frame radians: `atan2(dz, dx)`, so 0 faces +x (east) and
+pi/2 faces +z (north).

Nothing here may import from `flybrain.motor` — see AGENTS.md.
"""

from __future__ import annotations

import math
import random

from flybrain.loop.types import Player, WorldState

K_TURN = 0.35
EMA_ALPHA = 0.4
MIN_DISPLACEMENT = 0.05


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Heading:
    """Priority cascade (combat target > EMA of displacement > hold) into a low-pass.

    The EMA smooths the 8-way tile quantisation *before* the atan2; smoothing the
    angle afterwards would leave 45-degree per-tick jumps in the rendered scene.
    """

    def __init__(
        self,
        k_turn: float = K_TURN,
        ema_alpha: float = EMA_ALPHA,
        rng: random.Random | None = None,
    ):
        self.k_turn = k_turn
        self.ema_alpha = ema_alpha
        self._rng = rng if rng is not None else random.Random()
        self.theta = 0.0
        self._seeded = False
        self._life_id: int | None = None
        self._prev_pos: tuple[int, int] | None = None
        self._ema = (0.0, 0.0)

    def _respawn(self) -> None:
        self.theta = self._rng.uniform(-math.pi, math.pi)
        self._seeded = True
        self._prev_pos = None
        self._ema = (0.0, 0.0)

    def update(self, state: WorldState) -> float:
        player = state.player
        if player is None:
            if not self._seeded:
                self._respawn()
            return self.theta

        if not self._seeded or player.life_id != self._life_id:
            self._respawn()
            self._life_id = player.life_id

        if self._prev_pos is not None:
            dx = player.x - self._prev_pos[0]
            dz = player.z - self._prev_pos[1]
            a = self.ema_alpha
            self._ema = ((1.0 - a) * self._ema[0] + a * dx, (1.0 - a) * self._ema[1] + a * dz)
        self._prev_pos = (player.x, player.z)

        target = self._combat_bearing(state, player)
        if target is None:
            ex, ez = self._ema
            if math.hypot(ex, ez) > MIN_DISPLACEMENT:
                target = math.atan2(ez, ex)

        if target is not None:
            self.theta = wrap_pi(self.theta + self.k_turn * wrap_pi(target - self.theta))
        return self.theta

    @staticmethod
    def _combat_bearing(state: WorldState, player: Player) -> float | None:
        """Free ground truth: the engine auto-faces your combat target."""
        if not player.in_combat or player.target_type != "npc" or player.target_index < 0:
            return None
        for npc in state.npcs:
            if npc.index == player.target_index:
                dx, dz = npc.x - player.x, npc.z - player.z
                return math.atan2(dz, dx) if (dx or dz) else None
        return None
