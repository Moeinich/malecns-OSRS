"""The fly's value system: game events in, dopaminergic current out.

Two things here are deliberate and load-bearing.

**The reward becomes current, not a number.** An event does not produce a
scalar that some update rule multiplies by. It produces an injection current
into the real PAM and PPL1 cells, resolved by name from the connectome, and the
learning signal downstream is those cells' *firing rate*. The dopamine term is
therefore a network variable — it has the network's dynamics, its latency and
its saturation — rather than a value computed in numpy and handed to the rule.
That is the same hole `motor/decode.py` closes on the output side.

**`register_event` is the only way in.** Combat messages, XP, HP drops and a
hand-written test all arrive through one seam, so the source of reward is
swappable without touching the routing or the injection.

Routing, from `docs`-level biology: appetitive (`damage_dealt`, `kill`,
positive XP) to PAM, aversive (`damage_taken`, `death`, HP loss) to PPL1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from flybrain.loop.types import Player, Reward

#: `kind` -> which dopaminergic population it drives.
APPETITIVE = ("damage_dealt", "kill", "xp")
AVERSIVE = ("damage_taken", "death", "hp_drop")
EVENT_KINDS = APPETITIVE + AVERSIVE


class UnknownEvent(ValueError):
    """A reward kind with no dopaminergic route. Never silently dropped."""


@dataclass(frozen=True, slots=True)
class RewardParams:
    #: Peak injected current at full saturation, in the same units as the
    #: encoder's `i_max`: a constant drive `I` pulls `v` to `v_rest + I`, and
    #: threshold is 15 mV above rest, so this is comfortably suprathreshold.
    i_dan: float = 30.0
    #: Damage is in hitpoints; a 10-damage hit saturates about as hard as a kill.
    damage_scale: float = 0.1
    kill: float = 1.0
    death: float = 1.0
    #: Raw XP, which comes in hundreds for a low-level kill.
    xp_scale: float = 0.002
    #: Fraction of the accumulated drive surviving one tick. Dopamine outlasts
    #: the event that caused it — that is what the eligibility trace bridges.
    retention: float = 0.5
    #: Time constant of the dopamine baseline, in ticks. The rule is driven by
    #: `rate - baseline`, so a constant DAN rate teaches nothing and only a
    #: departure from what the fly has recently been getting does.
    baseline_ticks: float = 50.0


class _Populations(Protocol):
    def population(self, name: str, side: str | None = ...) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class DopamineIndex:
    """Which neurons are the reward centres. Resolved by name, never by index."""

    appetitive: np.ndarray
    aversive: np.ndarray

    @classmethod
    def from_connectome(cls, c: _Populations) -> DopamineIndex:
        return cls(appetitive=_require(c, "PAM"), aversive=_require(c, "PPL1"))


def _require(c: _Populations, name: str) -> np.ndarray:
    idx = c.population(name)
    if len(idx) == 0:
        raise KeyError(
            f"population {name!r} resolves to no neurons in this build; "
            "reward would be injected nowhere and read back as a plausible zero"
        )
    return np.asarray(idx, dtype=np.int64)


class RewardRouter:
    """Accumulates events into DAN drive, and reads dopamine back off the rates."""

    def __init__(
        self,
        index: DopamineIndex,
        n: int,
        params: RewardParams | None = None,
    ) -> None:
        self.index = index
        self.n = int(n)
        self.params = params if params is not None else RewardParams()

        self.appetitive_drive = 0.0
        self.aversive_drive = 0.0
        #: `None` until the first tick: a baseline seeded at zero would make the
        #: very first rate look like a reward.
        self.baseline: float | None = None
        self.events = 0
        self._last_revision: int | None = None

    # ------------------------------------------------------------- the seam

    def register_event(self, kind: str, magnitude: float = 1.0) -> None:
        if kind not in EVENT_KINDS:
            raise UnknownEvent(f"reward kind {kind!r} is not one of {EVENT_KINDS}")
        if magnitude <= 0.0:
            return
        self.events += 1
        if kind in APPETITIVE:
            self.appetitive_drive += float(magnitude)
        else:
            self.aversive_drive += float(magnitude)

    # ------------------------------------------------------------- sources

    def observe_reward(self, reward: Reward | None) -> bool:
        """Route one `reward` message. Returns whether it was new.

        The client keeps only the latest reward, so the revision is what stops a
        tick with no new events from re-injecting the previous tick's.
        """
        if reward is None or reward.revision == self._last_revision:
            return False
        self._last_revision = reward.revision
        p = self.params
        for event in reward.combat_events:
            if event.type == "damage_dealt":
                self.register_event("damage_dealt", event.damage * p.damage_scale)
            elif event.type == "damage_taken":
                self.register_event("damage_taken", event.damage * p.damage_scale)
            elif event.type == "kill":
                self.register_event("kill", p.kill)
        gained = sum(v for v in reward.xp_delta.values() if v > 0)
        if gained:
            self.register_event("xp", gained * p.xp_scale)
        return True

    def observe_state(self, previous: Player | None, current: Player | None) -> None:
        """HP loss and death, which arrive as state rather than as events."""
        if current is None:
            return
        p = self.params
        if current.is_dead and (previous is None or not previous.is_dead):
            # The killing blow's HP drop is not a second event: dying once is one
            # thing that happened, and counting it twice would double the sting.
            self.register_event("death", p.death)
            return
        # A respawn restores HP, so a drop is only a drop within one life.
        if previous is not None and previous.life_id == current.life_id:
            lost = previous.hp - current.hp
            if lost > 0:
                self.register_event("hp_drop", lost * p.damage_scale)

    # ---------------------------------------------------------- the current

    def current(self) -> np.ndarray:
        """`float32[n]`, zero everywhere but the two DAN populations.

        `tanh` because a 40-damage hit is not four times the news that a
        10-damage one is, and an unsaturating drive would put PAM into the
        depolarisation block that makes a reward look like silence.
        """
        out = np.zeros(self.n, dtype=np.float32)
        i = np.float32(self.params.i_dan)
        if self.appetitive_drive > 0.0:
            out[self.index.appetitive] = i * np.float32(np.tanh(self.appetitive_drive))
        if self.aversive_drive > 0.0:
            out[self.index.aversive] = i * np.float32(np.tanh(self.aversive_drive))
        return out

    def decay(self) -> None:
        r = self.params.retention
        self.appetitive_drive *= r
        self.aversive_drive *= r

    # --------------------------------------------------------- the dopamine

    def dan_rates(self, rates: np.ndarray) -> tuple[float, float]:
        return (
            float(rates[self.index.appetitive].mean()),
            float(rates[self.index.aversive].mean()),
        )

    def dopamine(self, rates: np.ndarray) -> float:
        """`DAN_rate - DAN_baseline`, the third factor of the learning rule.

        `DAN_rate` is the signed valence rate — appetitive minus aversive — so a
        beating and a kill in the same tick cancel in the cells rather than in a
        Python branch. The baseline is an exponential average of it, updated
        after the term is taken: learning tracks the change, not the level.
        """
        appetitive, aversive = self.dan_rates(rates)
        rate = appetitive - aversive
        if self.baseline is None:
            self.baseline = rate
            return 0.0
        term = rate - self.baseline
        a = 1.0 / max(self.params.baseline_ticks, 1.0)
        self.baseline += a * term
        return term


__all__ = [
    "APPETITIVE",
    "AVERSIVE",
    "EVENT_KINDS",
    "DopamineIndex",
    "RewardParams",
    "RewardRouter",
    "UnknownEvent",
]
