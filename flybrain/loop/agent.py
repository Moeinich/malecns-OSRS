"""The tick loop: state in, sub-frames, LIF substeps, decoded action out.

One game tick is rendered as four sub-frames; each is encoded to an injection
current that is then held constant for `substeps_per_subframe` LIF steps. At
the end of the tick the descending-neuron firing rates are decoded and the
body turns the egocentric command into an action.

Nothing here computes behaviour. Every value that reaches `to_action` came out
of `decode`, and everything `decode` saw came out of the simulated network — so
the ablations below are a real test of whether the wiring does any work.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp

from flybrain.connectome.loader import Connectome
from flybrain.engine.calibration import Calibration
from flybrain.engine.lif import LIFEngine
from flybrain.engine.plasticity import Plasticity
from flybrain.loop.client import BridgeClient
from flybrain.loop.types import Action, StateUpdate
from flybrain.motor.body import BodyParams, to_action
from flybrain.motor.decode import EgocentricCommand, MotorIndex, MotorParams, decode
from flybrain.reward import RewardRouter
from flybrain.sensory.collision import CollisionGrid
from flybrain.sensory.heading import Heading
from flybrain.sensory.retina import Retina

#: `float32[size, size, 4]` sub-frame -> `float32[N]` injection current.
Encoder = Callable[[np.ndarray], np.ndarray]

#: Only used when the sidecar has not said otherwise. Mirrors `DEFAULT_TICK_MS`
#: in `bridge/config.ts`; real Old School RuneScape runs at 600 ms.
DEFAULT_TICK_MS = 600


@dataclass(frozen=True)
class AgentParams:
    subframes: int = 4
    #: None derives it from the tick the sidecar reports, so the brain simulates
    #: exactly one tick of biological time whatever the server is running
    #: (600 ms / 4 sub-frames / dt 1 ms = 150). An explicit count overrides it.
    substeps_per_subframe: int | None = None
    rate_window_steps: int | None = None
    #: Fraction of the sidecar's stated deadline we are willing to spend.
    deadline_fraction: float = 1.0
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class TickReport:
    revision: int
    tick: int
    heading: float
    command: EgocentricCommand
    action: Action
    mean_rate_hz: float
    ms_total: float
    ms_retina: float
    ms_encode: float
    ms_lif: float
    ms_decode: float
    substeps: int
    overrun: bool
    dropped_game_ticks: int
    #: The sidecar's stated deadline for this tick, not the tick length.
    deadline_ms: float
    #: The rate vector `decode` was handed. Telemetry reads populations out of
    #: it; nothing in the loop may read it back into a decision.
    rates: np.ndarray
    #: Reflex spikes counted inside this tick, and the substep the first landed
    #: on. `None` substep means none fired — never drawn as a zero.
    escape_spikes: int
    escape_substep: int | None
    #: DAN population rates this tick and the dopamine term taken from them.
    #: `None` when no reward router is attached — never drawn as a zero, which
    #: is a real dopamine level.
    dan_appetitive_hz: float | None = None
    dan_aversive_hz: float | None = None
    dopamine: float | None = None
    #: Summed |dW| written into the simulated matrix. `None` with `--learn` off.
    weight_delta: float | None = None

    @property
    def kind(self) -> str:
        return self.action.kind


def scale_motor(motor: MotorIndex, calibration: Calibration | None) -> MotorIndex:
    """`motor` with its thresholds written in the calibrated network's own units.

    The decoder's gates are multiples of an operating rate, and the only honest
    source for that rate is the artifact the engine is running: an uncalibrated
    run is left alone, since its rates describe no network.
    """
    if calibration is None or not calibration.calibrated:
        return motor
    acceptance, rates = calibration.acceptance, calibration.rates
    if acceptance is None:
        return motor
    params = MotorParams.for_band(acceptance.target_hz, None if rates is None else rates.mean_hz)
    return replace(motor, params=params)


def default_encoder(connectome: Connectome, params: object | None = None) -> Encoder:
    """The real encoder, bound to one connectome. Needs the annotations feather."""
    from flybrain.sensory.encode import EncodeParams, encode

    bound = params if params is not None else EncodeParams()
    return lambda frame: encode(frame, connectome, bound)


class Agent:
    def __init__(
        self,
        client: BridgeClient,
        engine: LIFEngine,
        motor: MotorIndex,
        collision: CollisionGrid,
        encoder: Encoder,
        *,
        retina: Retina | None = None,
        heading: Heading | None = None,
        params: AgentParams | None = None,
        body_params: BodyParams | None = None,
        tonic: np.ndarray | None = None,
        calibration: Calibration | None = None,
        spike_sink: Callable[[np.ndarray], None] | None = None,
        reward: RewardRouter | None = None,
        plasticity: Plasticity | None = None,
    ) -> None:
        self.client = client
        self.engine = engine
        #: The decoder's thresholds are multiples of the rate the network is
        #: calibrated to, taken from the artifact the engine is running rather
        #: than written down twice.
        self.motor = scale_motor(motor, calibration)
        self.collision = collision
        self.encoder = encoder
        self.retina = retina if retina is not None else Retina()
        self.heading = heading if heading is not None else Heading()
        self.params = params if params is not None else AgentParams()
        self.body_params = body_params
        #: Constant current added to every encoded frame. `LIFEngine` has no tonic
        #: parameter, so the calibrated floor can only reach the live brain folded
        #: into the injected current — and if it does not reach it, the calibration
        #: describes a network this loop never runs.
        self.tonic = np.float32(0.0) if tonic is None else np.asarray(tonic, dtype=np.float32)
        #: Called with each substep's fired indices. Telemetry only; the loop
        #: never reads it back.
        self.spike_sink = spike_sink
        #: Routes game events into current injected at the real PAM/PPL1 cells.
        #: `None` leaves the brain rewardless, which is the pre-`--learn` loop.
        self.reward = reward
        #: Present only under `--learn`. Absent is the frozen-weights control.
        self.plasticity = plasticity

        self.ticks = 0
        self.overruns = 0
        self.last: TickReport | None = None
        #: The sub-frames this tick was built from, kept so telemetry can show
        #: the retina the brain was actually given rather than re-render one.
        self.last_frames: np.ndarray | None = None
        self.action_counts: dict[str, int] = {}
        self._ms_total = 0.0
        self._ms_lif = 0.0
        self._prev_state = None
        self._prev_heading: float | None = None
        self._state_tick_ms: int | None = None
        self._tick_mismatch_logged = False

    # ------------------------------------------------------------------ loop

    def run(self, max_ticks: int | None = None) -> Iterator[TickReport]:
        for update in self.client.states():
            yield self.tick(update)
            if max_ticks is not None and self.ticks >= max_ticks:
                return

    def tick(self, update: StateUpdate) -> TickReport:
        p = self.params
        self._observe_tick(update)
        t0 = time.perf_counter()
        budget = update.deadline_ms / 1000.0 * p.deadline_fraction

        state = update.state
        heading = self.heading.update(state)

        dan_current = None
        if self.reward is not None:
            self.reward.observe_reward(self.client.last_reward)
            self.reward.observe_state(
                self._prev_state.player if self._prev_state is not None else None,
                state.player,
            )
            dan_current = self.reward.current()

        frames = self.retina.render_subframes(
            self._prev_state,
            state,
            self.collision,
            heading,
            n=p.subframes,
            prev_heading=self._prev_heading,
        )
        self.last_frames = frames
        t_retina = time.perf_counter()

        per_subframe = self.substeps_per_subframe
        sink = self.spike_sink
        learn = self.plasticity
        self.motor.begin_tick()
        ms_encode = 0.0
        ms_lif = 0.0
        substeps = 0
        overrun = False
        for frame in frames:
            a = time.perf_counter()
            current = self.encoder(frame) + self.tonic
            if dan_current is not None:
                current = current + dan_current
            b = time.perf_counter()
            for k in range(per_subframe):
                fired = self.engine.step(current)
                self.motor.observe_spikes(fired, substeps + k)
                if learn is not None:
                    learn.observe_spikes(fired)
                if sink is not None:
                    sink(fired)
            c = time.perf_counter()
            ms_encode += (b - a) * 1e3
            ms_lif += (c - b) * 1e3
            substeps += per_subframe
            # Send what we have rather than stall the gateway: a late command is
            # worse than a coarse one, because the sidecar will dispatch the
            # continuation policy and our revision goes stale.
            if c - t0 > budget:
                overrun = True
                break

        t_decode = time.perf_counter()
        rates = self.engine.get_firing_rates(p.rate_window_steps)
        dan_hz = (None, None)
        dopamine = None
        weight_delta = None
        if self.reward is not None:
            dan_hz = self.reward.dan_rates(rates)
            # The dopamine term is the DAN cells' own rate, not the game reward
            # that drove them: the learning signal is a network variable.
            dopamine = self.reward.dopamine(rates)
            if learn is not None:
                weight_delta = learn.apply(dopamine)
            self.reward.decay()
        command = decode(rates, self.motor)
        action = to_action(command, state, heading, self.body_params)
        t_end = time.perf_counter()

        if p.dry_run:
            self.client.send_noop(update.revision)
        else:
            self.client.send_cmd(update.revision, action)

        self._prev_state = state
        self._prev_heading = heading
        self.ticks += 1
        if overrun:
            self.overruns += 1
        self.action_counts[action.kind] = self.action_counts.get(action.kind, 0) + 1

        report = TickReport(
            revision=update.revision,
            tick=update.tick,
            heading=heading,
            command=command,
            action=action,
            mean_rate_hz=float(rates.mean()),
            ms_total=(t_end - t0) * 1e3,
            ms_retina=(t_retina - t0) * 1e3,
            ms_encode=ms_encode,
            ms_lif=ms_lif,
            ms_decode=(t_end - t_decode) * 1e3,
            substeps=substeps,
            overrun=overrun,
            dropped_game_ticks=self.client.dropped_game_ticks,
            deadline_ms=float(update.deadline_ms),
            rates=rates,
            escape_spikes=self.motor._reflex.escape,
            escape_substep=self.motor._reflex.escape_substep,
            dan_appetitive_hz=dan_hz[0],
            dan_aversive_hz=dan_hz[1],
            dopamine=dopamine,
            weight_delta=weight_delta,
        )
        self._ms_total += report.ms_total
        self._ms_lif += report.ms_lif
        self.last = report
        return report

    def _observe_tick(self, update: StateUpdate) -> None:
        """Adopt the tick this state was actually built against.

        The sidecar measures the engine and corrects its own deadline mid-run,
        so the handshake value can be stale. Adopting it silently would change
        the substep count with nothing reporting it, which is the same class of
        bug as deriving it from a wrong tick — so say so, once.
        """
        before = self.substeps_per_subframe
        self._state_tick_ms = update.tick_ms
        self._ensure_rate_window()
        ready = self.client.ready
        if self._tick_mismatch_logged or ready is None or update.tick_ms == ready.tick_ms:
            return
        self._tick_mismatch_logged = True
        observed = (
            f"{update.observed_tick_ms:.1f} ms"
            if update.observed_tick_ms is not None
            else "not yet measured"
        )
        print(
            f"TICK MISMATCH: the sidecar announced {ready.tick_ms} ms at handshake, "
            f"but this state was built against {update.tick_ms} ms (observed {observed}). "
            f"Following it: substeps_per_subframe {before} -> {self.substeps_per_subframe}.",
            file=sys.stderr,
        )

    def _ensure_rate_window(self) -> None:
        """Keep the rate window at least one tick long.

        The decoder samples once per tick, so a window shorter than the tick
        evicts the tick's first substeps before they are ever read — silently,
        and worse the faster the server runs.
        """
        tick_ms = self.tick_ms
        pinned = self.params.rate_window_steps
        if pinned is not None and pinned * self.engine.dt_ms < tick_ms:
            raise ValueError(
                f"rate_window_steps={pinned} covers {pinned * self.engine.dt_ms:g} ms, "
                f"shorter than the {tick_ms} ms tick: every tick would lose its first "
                f"substeps unread."
            )
        if self.engine.rate_window_ms >= tick_ms:
            return
        print(
            f"RATE WINDOW: {self.engine.rate_window_ms:g} ms is shorter than the "
            f"{tick_ms} ms tick; widening it so no substep is evicted unread.",
            file=sys.stderr,
        )
        self.engine.set_rate_window_ms(float(tick_ms))

    @property
    def tick_ms(self) -> int:
        """The tick the sidecar last measured, else what it announced, else the default."""
        if self._state_tick_ms is not None:
            return self._state_tick_ms
        ready = self.client.ready
        return ready.tick_ms if ready is not None else DEFAULT_TICK_MS

    @property
    def substeps_per_subframe(self) -> int:
        """Derived from the reported tick unless `AgentParams` pins a count.

        Hardcoding the count is the real bug risk: the number would be right
        only for the tickrate it was written against, and a server started at a
        different one would leave the brain silently simulating the wrong amount
        of time per decision with nothing reporting it.
        """
        pinned = self.params.substeps_per_subframe
        if pinned is not None:
            return pinned
        return max(1, round(self.tick_ms / self.engine.dt_ms / self.params.subframes))

    # --------------------------------------------------------------- health

    @property
    def mean_ms_per_tick(self) -> float:
        return self._ms_total / self.ticks if self.ticks else 0.0

    @property
    def mean_ms_lif(self) -> float:
        return self._ms_lif / self.ticks if self.ticks else 0.0


# --------------------------------------------------------------- ablations
#
# Every one of these returns a new matrix. The cached .npz on disk and the
# connectome's own `W` are never touched: a lesion that outlived the process
# would quietly poison every later run.


@dataclass(frozen=True)
class Ablation:
    lesions: tuple[str, ...] = ()
    ablate_network: bool = False
    shuffle: bool = False
    seed: int = 0
    #: Filled in by `apply`, so the run summary can state what actually happened.
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        parts = []
        if self.ablate_network:
            parts.append("ablate-network")
        if self.shuffle:
            parts.append(f"shuffle(seed={self.seed})")
        parts += [f"lesion:{name}" for name in self.lesions]
        return "+".join(parts) if parts else "real"

    def silenced(self, connectome: Connectome) -> np.ndarray:
        """The neurons the engine must hold silent, alongside `apply`'s cut."""
        if not self.lesions:
            return np.empty(0, dtype=np.int64)
        return np.unique(
            np.concatenate([_population(connectome, n) for n in self.lesions]).astype(np.int64)
        )

    def apply(self, connectome: Connectome) -> sp.csc_matrix:
        W = connectome.W.copy()
        if self.ablate_network:
            W = ablate_network(W)
            self.notes.append(f"zeroed all {connectome.W.nnz} synapses")
        if self.shuffle:
            W, conflicts = shuffle_degree_preserving(W, self.seed)
            self.notes.append(f"degree-preserving rewire, {conflicts} unresolved conflicts")
        for name in self.lesions:
            idx = _population(connectome, name)
            W = lesion(W, idx)
            self.notes.append(f"lesion {name}: silenced {len(idx)} neurons")
        return W


def _population(connectome: Connectome, name: str) -> np.ndarray:
    try:
        return connectome.population(name)
    except KeyError:
        available = ", ".join(sorted(connectome.populations))
        raise KeyError(f"no population {name!r} in this build; have: {available}") from None


def lesion(W: sp.csc_matrix, idx: np.ndarray) -> sp.csc_matrix:
    """Cut a population's outgoing and incoming weights. Half of a lesion.

    The other half is `Ablation.silenced`, handed to the engine. Cutting the
    synapses alone leaves the cells firing: the calibrated tonic current reaches
    every neuron at 0.9 of threshold and noise carries them over it, so a
    population with its inhibition cut fires *more* — `lesion:DNp01` tripled
    escape in a scored run. Removing the cells has to mean no spikes at all.
    """
    W = W.tocsc(copy=True)
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return W
    column_of = np.repeat(np.arange(W.shape[1]), np.diff(W.indptr))
    mask = np.isin(column_of, idx) | np.isin(W.indices, idx)
    W.data[mask] = 0.0
    return W


def ablate_network(W: sp.csc_matrix) -> sp.csc_matrix:
    """Every synapse to zero, structure kept. Encoder and decoder stay intact."""
    W = W.tocsc(copy=True)
    W.data[:] = 0.0
    return W


def shuffle_degree_preserving(W: sp.csc_matrix, seed: int = 0) -> tuple[sp.csc_matrix, int]:
    """Rewire at random, preserving in-degree, out-degree and transmitter sign.

    The column of a CSC matrix is the presynaptic neuron, so permuting the
    postsynaptic index array globally leaves every column's length (out-degree)
    untouched and the multiset of targets (in-degree) unchanged, while the
    weights — and therefore Dale's-law sign, which is per presynaptic neuron —
    stay with the column they came from.

    This is the control that matters: if a shuffled connectome plays as well as
    the real one, the specific wiring contributes nothing.
    """
    W = W.tocsc(copy=True)
    rng = np.random.default_rng(seed)
    indptr = W.indptr
    indices = W.indices[rng.permutation(W.nnz)].astype(np.int64, copy=False)
    column_of = np.repeat(np.arange(W.shape[1], dtype=np.int64), np.diff(indptr))

    conflicts = _repair(indices, column_of, rng, W.shape[1])
    _sort_columns(indices, W.data, indptr)
    return sp.csc_matrix(
        (W.data, indices.astype(W.indices.dtype), indptr), shape=W.shape
    ), conflicts


def _conflicting(indices: np.ndarray, column_of: np.ndarray, n: int) -> np.ndarray:
    """Positions holding a self-loop or a second copy of an edge already present."""
    self_loops = indices == column_of
    key = column_of * n + indices
    order = np.argsort(key, kind="stable")
    duplicate = np.zeros(indices.size, dtype=bool)
    sorted_key = key[order]
    duplicate[order[1:]] = sorted_key[1:] == sorted_key[:-1]
    return np.flatnonzero(self_loops | duplicate)


def _repair(indices: np.ndarray, column_of: np.ndarray, rng, n: int, rounds: int = 64) -> int:
    """Swap the bad targets against random other positions until they are legal."""
    for _ in range(rounds):
        bad = _conflicting(indices, column_of, n)
        if bad.size == 0:
            return 0
        # Both sides of the swap must be unique and disjoint, or numpy's
        # last-write-wins would drop a target and silently change an in-degree.
        partner = np.unique(rng.integers(0, indices.size, size=bad.size))
        partner = partner[~np.isin(partner, bad)]
        k = min(partner.size, bad.size)
        left, right = bad[:k], partner[:k]
        held = indices[left].copy()
        indices[left] = indices[right]
        indices[right] = held
    return int(_conflicting(indices, column_of, n).size)


def _sort_columns(indices: np.ndarray, data: np.ndarray, indptr: np.ndarray) -> None:
    """Canonical order per column, weights carried along so no edge changes sign."""
    for j in range(len(indptr) - 1):
        lo, hi = indptr[j], indptr[j + 1]
        if hi - lo > 1:
            order = np.argsort(indices[lo:hi], kind="stable")
            indices[lo:hi] = indices[lo:hi][order]
            data[lo:hi] = data[lo:hi][order]


__all__ = [
    "DEFAULT_TICK_MS",
    "Ablation",
    "Agent",
    "AgentParams",
    "Encoder",
    "TickReport",
    "ablate_network",
    "default_encoder",
    "lesion",
    "shuffle_degree_preserving",
]
