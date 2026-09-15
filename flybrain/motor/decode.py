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

import logging
import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

#: Descending families pooled for steering, and the reading behind the choice.
#: This is **our interpretation**, not a MaleCNS annotation: the dataset names
#: no "steering pool". `DNa*` (DNa01-DNa16 plus the DNae unnamed extension) is
#: the anterior-dorsal descending family that carries the canonical turning
#: cells DNa01/DNa02 into the leg neuropils, so it is the population whose
#: left/right imbalance is read as a turn. DNa02 stays in it as the canonical
#: member rather than being the whole readout: one cell emits ~2 spikes per
#: 600 ms tick, so a single-cell differential is Poisson noise.
STEER_FAMILIES = ("DNa", "DNa02")

#: Forward drive, same caveat. `DNb*` is the anterior-ventral descending family
#: containing DNb02, already paired with DNp09 in `cell_types.toml`; DNp09 and
#: DNpe017 stay named so the canonical drive cells are certainly included.
DRIVE_FAMILIES = ("DNb", "DNp09", "DNpe017")


@dataclass(frozen=True, slots=True)
class MotorParams:
    #: Full-scale turn for a fully lateralised steering differential.
    turn_gain: float = math.pi / 2
    eps: float = 1e-6
    #: Pooled rate mapping to `drive == 1.0`.
    drive_max_hz: float = 40.0
    run_drive: float = 0.5
    reverse_hz: float = 20.0
    discrete_hz: float = 20.0
    refractory_ticks: int = 2
    #: How far above its own running baseline a population must fire to count
    #: as a burst rather than the tonic floor every neuron now sits at.
    burst_ratio: float = 2.0
    baseline_alpha: float = 0.3


@dataclass(slots=True)
class _Debounce:
    tick: int = 0
    last: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class _Baseline:
    """Running activity floor per readout population, EMA over ticks.

    Absolute thresholds were written when the network was silent, so a single
    spike meant something. Every neuron now fires at a tonic-driven floor and
    the absolute thresholds trip together on every tick. A reflex is a
    deviation from what its own cells were already doing. These levels are
    derived from the network's own output, never from game state.
    """

    level: dict[str, float] = field(default_factory=dict)

    def burst(self, name: str, value: float, ratio: float, alpha: float) -> bool:
        base = self.level.get(name, 0.0)
        self.level[name] = base + alpha * (value - base)
        return value > base * ratio


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


def _pooled(c: _Populations, names: tuple[str, ...], side: str | None = None) -> np.ndarray:
    """Union of the named families, skipping any this build does not carry."""
    found = []
    for name in names:
        try:
            found.append(c.population(name, side=side))
        except KeyError:
            log.warning("population %r absent from this build; not pooled", name)
    if not found:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(found))


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
    _baseline: _Baseline = field(default_factory=_Baseline)

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
        """Resolve the readout populations by name.

        Steering and drive are pooled over whole descending families
        (`STEER_FAMILIES`, `DRIVE_FAMILIES`) with a left/right split, because
        the differential *is* the steering signal and a one-cell-per-side
        differential is noise. Escape is deliberately not pooled: `DNp01` is
        one Giant Fiber per side and that is the biology.
        """
        feeding = c.population("MBON")
        third = len(feeding) // 3
        return cls(
            steer_left=_pooled(c, STEER_FAMILIES, side="left"),
            steer_right=_pooled(c, STEER_FAMILIES, side="right"),
            drive=_pooled(c, DRIVE_FAMILIES),
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
    b = motor._baseline
    d.tick += 1

    left, right = _pool(rates, motor.steer_left), _pool(rates, motor.steer_right)
    turn = p.turn_gain * (left - right) / (left + right + p.eps)

    # The Giant Fiber is all-or-nothing within a tick, counted as it happened
    # rather than averaged — but against its own recent rate, since DNp01 fires
    # at the tonic floor and "any spike at all" is now every tick.
    escape = b.burst("escape", float(motor._reflex.escape), p.burst_ratio, p.baseline_alpha)

    pooled = {
        "attack": _pool(rates, motor.attack),
        "eat": _pool(rates, motor.eat),
        "pickup": _pool(rates, motor.pickup),
    }
    burst = {k: b.burst(k, v, p.burst_ratio, p.baseline_alpha) for k, v in pooled.items()}
    above = {k: v for k, v in pooled.items() if v >= p.discrete_hz and burst[k]}
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
