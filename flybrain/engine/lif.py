from __future__ import annotations

from collections import deque

import numpy as np
import scipy.sparse as sp

#: Offsets drawn from per step, so the pool costs `N + DEFAULT_NOISE_POOL`
#: float32 (4.7 MB at N=184,110) and repeats no sooner than this many steps.
#: 0 disables it and restores a fresh `standard_normal` every step.
#:
#: `standard_normal` over every neuron was 0.436 ms of a 1.16 ms step — 131 ms
#: of a 347 ms tick, the largest single dense cost left. Reading a rolling
#: window out of a pre-drawn pool is 0.016 ms. It is not an approximation of
#: the same process in the way holding a sample for k steps would be: within a
#: step the N values are still N distinct independent draws, and neuron `i` at
#: `t` and at `t+1` read pool entries `NOISE_STRIDE` apart, so they are
#: independent too. What it gives up is that the pool is finite — after
#: `DEFAULT_NOISE_POOL / gcd(stride, pool)` steps the sequence repeats, which
#: is 1,048,576 steps, 35 minutes of biological time at dt = 2.
DEFAULT_NOISE_POOL = 1 << 20

#: Odd, so it is coprime with the power-of-two pool and successive offsets walk
#: all of it rather than a short cycle. It is smaller than `N`, so consecutive
#: windows do overlap: neuron `i` at `t+1` reads the entry neuron `i + stride`
#: read at `t`. That is a cross-neuron, cross-time correlation, not one along
#: any single neuron's own noise sequence, and `tests/test_lif.py` measures what
#: it does to the rate distribution rather than assuming it does nothing.
NOISE_STRIDE = 16411


class LIFEngine:
    """Event-driven sparse leaky integrate-and-fire network.

    `W` is signed (transmitter sign already folded in) and CSC, so column `j`
    holds neuron `j`'s postsynaptic targets and a step gathers only the columns
    of the neurons that actually spiked.
    """

    def __init__(
        self,
        W: sp.csc_matrix,
        dt_ms: float = 1.0,
        v_rest: float = -65.0,
        v_reset: float = -70.0,
        v_thresh: float = -50.0,
        tau_m: float = 20.0,
        tau_s: float = 5.0,
        tau_w: float = 150.0,
        b: float = 2.0,
        refractory_ms: float = 2.0,
        transmission_delay_ms: float = 1.8,
        spontaneous_noise_std: float = 0.0,
        noise_pool: int = DEFAULT_NOISE_POOL,
        rate_window_ms: float = 500.0,
        plastic_idx: np.ndarray | None = None,
        silenced: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        if not sp.isspmatrix_csc(W):
            W = sp.csc_matrix(W)
        if W.shape[0] != W.shape[1]:
            raise ValueError(f"W must be square, got {W.shape}")

        self.W: sp.csc_matrix = W
        self.W.data = self.W.data.astype(np.float32, copy=False)
        self.plastic_idx = plastic_idx

        self.N: int = W.shape[0]
        self.dt_ms = float(dt_ms)
        self.v_rest = float(v_rest)
        self.v_reset = float(v_reset)
        self.v_thresh = float(v_thresh)
        self.tau_m = float(tau_m)
        self.tau_s = float(tau_s)
        self.tau_w = float(tau_w)
        self.b = float(b)
        self.spontaneous_noise_std = float(spontaneous_noise_std)

        self.av = float(np.exp(-self.dt_ms / self.tau_m))
        self.ag = float(np.exp(-self.dt_ms / self.tau_s))
        self.aw = float(np.exp(-self.dt_ms / self.tau_w))
        self._one_minus_av = 1.0 - self.av

        self.refractory_steps = round(refractory_ms / self.dt_ms)
        self.delay_slots = max(1, round(transmission_delay_ms / self.dt_ms))

        self.v = np.full(self.N, self.v_rest, dtype=np.float32)
        self.g = np.zeros(self.N, dtype=np.float32)
        # Spike-frequency adaptation current, kept separate from `g` so it is never
        # deposited through the delay ring. It only ever subtracts from `drive`, so a
        # silent neuron stays silent: adaptation cannot manufacture a spike.
        self.w = np.zeros(self.N, dtype=np.float32)
        self.refractory = np.zeros(self.N, dtype=np.int32)

        self._ring = np.zeros((self.delay_slots + 1, self.N), dtype=np.float32)
        self._ring_ptr = 0

        self._indptr = self.W.indptr
        self._indices = self.W.indices
        self._data = self.W.data

        self.rate_window_ms = float(rate_window_ms)
        self._hist_len = max(1, round(rate_window_ms / self.dt_ms))
        self._hist: deque[np.ndarray] = deque(maxlen=self._hist_len)
        self.steps = 0

        self._silenced = np.empty(0, dtype=np.int64)
        self.set_silenced(silenced)

        self._rng = np.random.default_rng(seed)
        self._noise_pool: np.ndarray | None = None
        self._noise_offset = 0
        if self.spontaneous_noise_std > 0.0 and noise_pool > 0:
            self._noise_period = int(noise_pool)
            self._noise_pool = self._rng.standard_normal(
                self.N + self._noise_period, dtype=np.float32
            )
            self._noise_pool *= np.float32(self.spontaneous_noise_std)

    def step(self, external_current: np.ndarray | None = None) -> np.ndarray:
        if external_current is None:
            drive = np.zeros(self.N, dtype=np.float32)
        else:
            if external_current.shape != (self.N,):
                raise ValueError(
                    f"external_current must have shape ({self.N},), got {external_current.shape}"
                )
            drive = external_current.astype(np.float32, copy=True)

        self.g *= self.ag
        self.w *= self.aw
        slot = self._ring[self._ring_ptr]
        self.g += slot
        slot[:] = 0.0
        drive += self.g
        drive -= self.w

        if self._noise_pool is not None:
            offset = self._noise_offset
            drive += self._noise_pool[offset : offset + self.N]
            self._noise_offset = (offset + NOISE_STRIDE) % self._noise_period
        elif self.spontaneous_noise_std > 0.0:
            # float32 directly, not `normal` then cast: the draw is the single
            # largest dense cost in a step (0.65 -> 0.41 ms at N=184,110, on a
            # 1.20 ms step), and a float64 buffer over every neuron is what it
            # was spending the difference on.
            noise = self._rng.standard_normal(self.N, dtype=np.float32)
            noise *= np.float32(self.spontaneous_noise_std)
            drive += noise

        active = self.refractory == 0
        v_next = self.v_rest + (self.v - self.v_rest) * self.av + drive * self._one_minus_av
        self.v = np.where(active, v_next, self.v_reset).astype(np.float32, copy=False)
        if self._silenced.size:
            self.v[self._silenced] = self.v_rest

        fired = np.flatnonzero(active & (self.v >= self.v_thresh)).astype(np.int32)

        np.subtract(self.refractory, 1, out=self.refractory, where=self.refractory > 0)
        if fired.size:
            self.v[fired] = self.v_reset
            self.refractory[fired] = self.refractory_steps
            self.w[fired] += self.b
            self._deposit(fired)

        self._ring_ptr = (self._ring_ptr + 1) % self._ring.shape[0]
        self._record(fired)
        self.steps += 1
        return fired

    @property
    def silenced(self) -> np.ndarray:
        """Indices removed from the simulation. Empty unless a lesion set them."""
        return self._silenced

    def set_silenced(self, idx: np.ndarray | None) -> None:
        """Remove these neurons: pinned at rest, so they never reach threshold.

        Nothing downstream of them is special-cased — a neuron that never fires
        never deposits, never adapts and reads 0 Hz. Spikes they emitted before
        this call still arrive: the delay ring holds summed current, not the
        contribution of any one source, so those cannot be withdrawn.
        """
        self._silenced = (
            np.empty(0, dtype=np.int64) if idx is None else np.asarray(idx, dtype=np.int64)
        )
        if self._silenced.size:
            self.v[self._silenced] = self.v_rest
            self.g[self._silenced] = 0.0
            self.w[self._silenced] = 0.0
            self.refractory[self._silenced] = 0

    def _deposit(self, fired: np.ndarray) -> None:
        starts = self._indptr[fired]
        counts = self._indptr[fired + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return
        offsets = starts - (counts.cumsum() - counts)
        pos = np.repeat(offsets, counts) + np.arange(total)
        target = self._ring[(self._ring_ptr + self.delay_slots) % self._ring.shape[0]]
        target += np.bincount(self._indices[pos], weights=self._data[pos], minlength=self.N).astype(
            np.float32
        )

    def _record(self, fired: np.ndarray) -> None:
        self._hist.append(fired)

    def set_rate_window_ms(self, rate_window_ms: float) -> None:
        """Resize the rate window, keeping whatever history still fits.

        A window shorter than the caller's sampling interval evicts the start of
        every interval before it is ever read, so the loop widens this rather
        than losing substeps.
        """
        n = max(1, round(rate_window_ms / self.dt_ms))
        self.rate_window_ms = float(rate_window_ms)
        if n == self._hist_len:
            return
        self._hist_len = n
        self._hist = deque(self._hist, maxlen=n)

    def get_firing_rates(self, window: int | None = None) -> np.ndarray:
        n = min(window if window is not None else self._hist_len, len(self._hist))
        if n <= 0:
            return np.zeros(self.N, dtype=np.float32)
        recent = list(self._hist)[-n:]
        counts = np.bincount(np.concatenate(recent), minlength=self.N).astype(np.float32)
        return counts / (n * self.dt_ms / 1000.0)
