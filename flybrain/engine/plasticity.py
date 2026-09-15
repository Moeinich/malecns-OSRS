"""Three-factor plasticity on the KC->MBON synapses, written into the simulated matrix.

**There is exactly one weight array in the process.** `LIFEngine` owns `self.W`
(CSC) and deposits from `self.W.data`; this module writes *in place* into that
same array at precomputed plastic positions. It never builds, copies or
reassigns a matrix. FlyBrain's learning was a no-op for precisely the opposite
reason: it updated a second copy that was never simulated, and every test it had
passed anyway because they all read back from the copy.

`tests/test_plasticity.py::test_stdp_mutates_the_simulated_weights` is the test
that catches that. Its second assertion — that the *response* moved, not only
the numbers — is the load-bearing one.

The rule is `dW = eta * eligibility * (DAN_rate - DAN_baseline)`, with the
dopamine term coming from `flybrain.reward` as a firing rate of real PAM/PPL1
cells rather than as a game reward. Eligibility is a per-synapse coincidence
trace decaying over ~1.5 s, which is what spans the gap between an action and
the reward it earns.

Plastic subset: KC->MBON only. That is where fly learning actually lives, it is
a tractable number of edges, and credit assignment onto it is interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.sparse as sp


class _Populations(Protocol):
    def population(self, name: str, side: str | None = ...) -> np.ndarray: ...


def plastic_indices(W: sp.csc_matrix, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Positions in `W.data` of the `pre -> post` edges, ascending.

    The stored matrix is `W[post, pre]`, so column `j` is presynaptic neuron `j`
    and `W.indices` holds postsynaptic rows. Resolve against the matrix the
    engine is actually handed — `Calibration.apply` and the ablations each
    return a *new* matrix, and indices computed against the raw build would
    address the wrong entries of it.
    """
    if not sp.isspmatrix_csc(W):
        raise TypeError(f"plastic indices need the engine's CSC matrix, got {type(W).__name__}")
    column_of = np.repeat(np.arange(W.shape[1], dtype=np.int64), np.diff(W.indptr))
    mask = np.isin(column_of, pre) & np.isin(W.indices, post)
    return np.flatnonzero(mask).astype(np.int64)


def kc_to_mbon(W: sp.csc_matrix, c: _Populations) -> np.ndarray:
    return plastic_indices(W, c.population("KC"), c.population("MBON"))


@dataclass(frozen=True, slots=True)
class PlasticityParams:
    #: Weight change per unit of eligibility per unit of dopamine rate (Hz).
    eta: float = 2e-4
    #: Eligibility decay, spanning the action -> reward gap.
    tau_eligibility_ms: float = 1500.0
    #: Pre- and postsynaptic spike traces, the coincidence window itself.
    tau_trace_ms: float = 20.0
    #: Ceiling as a multiple of each synapse's initial weight. Sign-preserving:
    #: a synapse may be driven to zero but never through it, because Dale's law
    #: is a property of the presynaptic cell and not of what it learned.
    w_max_scale: float = 3.0


class Plasticity:
    """Eligibility traces over one plastic edge set, and the update they gate."""

    def __init__(
        self,
        W: sp.csc_matrix,
        plastic_idx: np.ndarray,
        pre: np.ndarray,
        post: np.ndarray,
        *,
        dt_ms: float = 1.0,
        params: PlasticityParams | None = None,
    ) -> None:
        self.W = W
        self.params = params if params is not None else PlasticityParams()
        self.plastic_idx = np.asarray(plastic_idx, dtype=np.int64)
        self.pre = np.sort(np.asarray(pre, dtype=np.int64))
        self.post = np.sort(np.asarray(post, dtype=np.int64))

        n = W.shape[0]
        self._pre_lookup = np.full(n, -1, dtype=np.int64)
        self._pre_lookup[self.pre] = np.arange(self.pre.size)
        self._post_lookup = np.full(n, -1, dtype=np.int64)
        self._post_lookup[self.post] = np.arange(self.post.size)

        column_of = np.repeat(np.arange(W.shape[1], dtype=np.int64), np.diff(W.indptr))
        self._pre_local = self._pre_lookup[column_of[self.plastic_idx]]
        self._post_local = self._post_lookup[W.indices[self.plastic_idx]]
        if self.plastic_idx.size and (self._pre_local.min() < 0 or self._post_local.min() < 0):
            raise ValueError("plastic_idx contains an edge outside the pre/post populations")

        # CSC order is already grouped by column, so the plastic set is grouped
        # by presynaptic neuron with no sort; only the postsynaptic view needs one.
        self._pre_ptr = np.searchsorted(self._pre_local, np.arange(self.pre.size + 1))
        self._post_order = np.argsort(self._post_local, kind="stable")
        self._post_ptr = np.searchsorted(
            self._post_local[self._post_order], np.arange(self.post.size + 1)
        )

        self.w0 = W.data[self.plastic_idx].copy()
        hi = self.w0 * np.float32(self.params.w_max_scale)
        self._lo = np.minimum(hi, 0.0).astype(np.float32)
        self._hi = np.maximum(hi, 0.0).astype(np.float32)

        self._x = np.zeros(self.pre.size, dtype=np.float32)
        self._y = np.zeros(self.post.size, dtype=np.float32)
        self._e = np.zeros(self.plastic_idx.size, dtype=np.float64)
        # Eligibility decays lazily: `true = _e * _scale`. Decaying the array
        # itself every substep would cost one pass over every plastic synapse per
        # millisecond of simulated time, for an identical result.
        self._scale = 1.0

        self._ae = float(np.exp(-dt_ms / self.params.tau_eligibility_ms))
        self._at = float(np.exp(-dt_ms / self.params.tau_trace_ms))
        self.updates = 0
        self.total_abs_dw = 0.0

    @classmethod
    def attach(
        cls,
        engine,
        c: _Populations,
        *,
        params: PlasticityParams | None = None,
    ) -> Plasticity:
        """Bind to a built engine, using the `plastic_idx` it was constructed with.

        `LIFEngine.__init__` stores `plastic_idx` and reads it nowhere; this is
        the only thing that ever looks at it, which is what keeps the index set
        and the simulated matrix from drifting apart.
        """
        idx = engine.plastic_idx
        if idx is None:
            idx = kc_to_mbon(engine.W, c)
        return cls(
            engine.W,
            idx,
            c.population("KC"),
            c.population("MBON"),
            dt_ms=engine.dt_ms,
            params=params,
        )

    # ---------------------------------------------------------------- traces

    def observe_spikes(self, fired: np.ndarray) -> None:
        """One LIF substep's spikes. Cheap: work is proportional to spikes, not to N."""
        self._scale *= self._ae
        if self._scale < 1e-6:
            self._e *= self._scale
            self._scale = 1.0
        self._x *= self._at
        self._y *= self._at
        if not len(fired):
            return

        pre_fired = self._pre_lookup[fired]
        pre_fired = pre_fired[pre_fired >= 0]
        post_fired = self._post_lookup[fired]
        post_fired = post_fired[post_fired >= 0]
        if not pre_fired.size and not post_fired.size:
            return

        inv = 1.0 / self._scale
        # Coincidence, not order: the dopamine term supplies the sign, so a
        # symmetric trace is what the third factor is allowed to gate.
        if pre_fired.size:
            pos = _gather(self._pre_ptr, pre_fired)
            self._e[pos] += self._y[self._post_local[pos]] * inv
        if post_fired.size:
            pos = self._post_order[_gather(self._post_ptr, post_fired)]
            self._e[pos] += self._x[self._pre_local[pos]] * inv

        self._x[pre_fired] += 1.0
        self._y[post_fired] += 1.0

    @property
    def eligibility(self) -> np.ndarray:
        return self._e * self._scale

    # ---------------------------------------------------------------- update

    def apply(self, dopamine: float) -> float:
        """`dW = eta * eligibility * dopamine`, in place. Returns the summed |dW|.

        Writes through `W.data[plastic_idx]`, which *is* the array the engine
        deposits from — never a copy, and never a reassignment of `.data`.
        """
        if dopamine == 0.0 or not self.plastic_idx.size:
            return 0.0
        dw = (self.params.eta * dopamine * self.eligibility).astype(np.float32)
        data = self.W.data
        data[self.plastic_idx] = np.clip(data[self.plastic_idx] + dw, self._lo, self._hi)
        self.updates += 1
        moved = float(np.abs(dw).sum())
        self.total_abs_dw += moved
        return moved

    @property
    def weights(self) -> np.ndarray:
        return self.W.data[self.plastic_idx]

    @property
    def drift(self) -> float:
        """Mean |w - w0| over the plastic set — how far learning has moved."""
        if not self.plastic_idx.size:
            return 0.0
        return float(np.abs(self.weights - self.w0).mean())


def _gather(ptr: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Concatenated `ptr[r]:ptr[r+1]` slices, without a Python loop."""
    starts = ptr[rows]
    counts = ptr[rows + 1] - starts
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    offsets = starts - (counts.cumsum() - counts)
    return np.repeat(offsets, counts) + np.arange(total)


def probe_response(engine, current: np.ndarray, steps: int = 200) -> np.ndarray:
    """Firing rates under a fixed input, from a reset membrane state.

    The reset is the point: without it the probe measures whatever the network
    was left doing, so a weight change and a leftover transient are
    indistinguishable. `LIFEngine` has no such method and is not this lane's to
    change, so the reset is done here against its documented state.
    """
    engine.v[:] = engine.v_rest
    engine.g[:] = 0.0
    engine.w[:] = 0.0
    engine.refractory[:] = 0
    engine._ring[:] = 0.0
    engine._hist.clear()
    for _ in range(steps):
        engine.step(current)
    return engine.get_firing_rates(steps)


__all__ = [
    "Plasticity",
    "PlasticityParams",
    "kc_to_mbon",
    "plastic_indices",
    "probe_response",
]
