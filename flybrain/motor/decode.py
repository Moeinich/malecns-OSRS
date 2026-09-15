"""Descending-neuron firing rates in, an egocentric command out.

`decode(rates, motor)` is the whole interface, and the fact that `WorldState`
is not in it is the point. FlyBrain computed `left_eye_drive - right_eye_drive`
in NumPy and injected the answer into cells it had labelled DNp20, so the
network could be deleted without changing behaviour. Here the only path from
the game to the decision runs through the simulated neurons: this module
imports nothing from `flybrain`, and `tests/test_decode.py` asserts both that
and the parameter list.

Turning egocentric into absolute needs to know where the body is, so it lives
in `motor/body.py` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class MotorParams:
    #: Full-scale turn for a fully lateralised DNa02 differential.
    turn_gain: float = math.pi / 2
    eps: float = 1e-6
    #: Pooled rate mapping to `drive == 1.0`.
    drive_max_hz: float = 40.0
    run_drive: float = 0.5
    reverse_hz: float = 20.0
    discrete_hz: float = 20.0
    refractory_ticks: int = 2


@dataclass(slots=True)
class _Debounce:
    tick: int = 0
    last: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class _Reflex:
    """Spikes counted inside the current tick, first-past-the-post.

    An all-or-nothing reflex cannot be read off an averaged rate: a Giant Fiber
    spike early in the tick is evicted from the rate window before the decoder
    looks. The loop writes the count here as the substeps run, because
    `decode(rates, motor)` may not grow a third parameter. These are spikes out
    of the LIF engine, never game state.
    """

    escape: int = 0
    #: Substep the first escape spike landed on, so latency can be reported.
    escape_substep: int | None = None


@dataclass(frozen=True, slots=True)
class EgocentricCommand:
    turn: float
    drive: float
    reverse: bool
    escape: bool
    attack: bool
    eat: bool
    pickup: bool


class _Populations(Protocol):
    """Structural view of `connectome.Connectome` — importing it would breach the firewall."""

    def population(self, name: str, side: str | None = ...) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class MotorIndex:
    """Which neurons are read out as what. Built once, never from game state."""

    steer_left: np.ndarray
    steer_right: np.ndarray
    drive: np.ndarray
    reverse: np.ndarray
    escape: np.ndarray
    attack: np.ndarray
    eat: np.ndarray
    pickup: np.ndarray
    params: MotorParams = MotorParams()
    _debounce: _Debounce = field(default_factory=_Debounce)
    _reflex: _Reflex = field(default_factory=_Reflex)

    def begin_tick(self) -> None:
        self._reflex.escape = 0
        self._reflex.escape_substep = None

    def observe_spikes(self, fired: np.ndarray, substep: int) -> None:
        """Accumulate one substep's reflex spikes. Called during the tick."""
        if not len(self.escape) or not len(fired):
            return
        n = int(np.isin(fired, self.escape).sum())
        if not n:
            return
        if self._reflex.escape == 0:
            self._reflex.escape_substep = substep
        self._reflex.escape += n

    @classmethod
    def from_connectome(cls, c: _Populations, params: MotorParams | None = None) -> MotorIndex:
        feeding = c.population("MBON")
        third = len(feeding) // 3
        return cls(
            steer_left=c.population("DNa02", side="left"),
            steer_right=c.population("DNa02", side="right"),
            drive=np.concatenate([c.population("DNp09"), c.population("DNpe017")]),
            reverse=c.population("MDN"),
            escape=c.population("DNp01"),
            # MaleCNS names no attack/feed descending pool, so the discrete acts
            # are read from disjoint MBON slices. Provisional, and the ablation
            # harness is what will say whether it carries anything.
            attack=feeding[:third],
            eat=feeding[third : 2 * third],
            pickup=feeding[2 * third :],
            params=params if params is not None else MotorParams(),
        )


def _pool(rates: np.ndarray, idx: np.ndarray) -> float:
    return float(rates[idx].mean()) if len(idx) else 0.0


def decode(rates: np.ndarray, motor: MotorIndex) -> EgocentricCommand:
    p = motor.params
    d = motor._debounce
    d.tick += 1

    left, right = _pool(rates, motor.steer_left), _pool(rates, motor.steer_right)
    turn = p.turn_gain * (left - right) / (left + right + p.eps)

    # The Giant Fiber is an all-or-nothing reflex: one spike anywhere in the
    # tick is the trigger, counted as it happened rather than averaged.
    escape = motor._reflex.escape > 0

    pooled = {
        "attack": _pool(rates, motor.attack),
        "eat": _pool(rates, motor.eat),
        "pickup": _pool(rates, motor.pickup),
    }
    above = {k: v for k, v in pooled.items() if v >= p.discrete_hz}
    winner = max(above, key=lambda k: above[k]) if above else None
    if winner is not None:
        if d.tick - d.last.get(winner, -(1 << 30)) < p.refractory_ticks:
            winner = None
        else:
            d.last[winner] = d.tick

    return EgocentricCommand(
        turn=turn,
        drive=min(_pool(rates, motor.drive) / p.drive_max_hz, 1.0),
        reverse=_pool(rates, motor.reverse) >= p.reverse_hz,
        escape=escape,
        attack=winner == "attack",
        eat=winner == "eat",
        pickup=winner == "pickup",
    )
