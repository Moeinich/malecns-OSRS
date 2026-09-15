from __future__ import annotations

import numpy as np
import scipy.sparse as sp


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
        refractory_ms: float = 2.0,
        transmission_delay_ms: float = 1.8,
        spontaneous_noise_std: float = 0.0,
        rate_window_ms: float = 500.0,
        plastic_idx: np.ndarray | None = None,
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
        self.spontaneous_noise_std = float(spontaneous_noise_std)

        self.av = float(np.exp(-self.dt_ms / self.tau_m))
        self.ag = float(np.exp(-self.dt_ms / self.tau_s))
        self._one_minus_av = 1.0 - self.av

        self.refractory_steps = round(refractory_ms / self.dt_ms)
        self.delay_slots = max(1, round(transmission_delay_ms / self.dt_ms))

        self.v = np.full(self.N, self.v_rest, dtype=np.float32)
        self.g = np.zeros(self.N, dtype=np.float32)
        self.refractory = np.zeros(self.N, dtype=np.int32)

        self._ring = np.zeros((self.delay_slots + 1, self.N), dtype=np.float32)
        self._ring_ptr = 0

        self._indptr = self.W.indptr
        self._indices = self.W.indices
        self._data = self.W.data

        self._hist_len = max(1, round(rate_window_ms / self.dt_ms))
        self._hist: list[np.ndarray] = []
        self._hist_ptr = 0
        self.steps = 0

        self._rng = np.random.default_rng(seed)

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
        slot = self._ring[self._ring_ptr]
        self.g += slot
        slot[:] = 0.0
        drive += self.g

        if self.spontaneous_noise_std > 0.0:
            drive += self._rng.normal(0.0, self.spontaneous_noise_std, self.N).astype(np.float32)

        active = self.refractory == 0
        v_next = self.v_rest + (self.v - self.v_rest) * self.av + drive * self._one_minus_av
        self.v = np.where(active, v_next, self.v_reset).astype(np.float32, copy=False)

        fired = np.flatnonzero(active & (self.v >= self.v_thresh)).astype(np.int32)

        np.subtract(self.refractory, 1, out=self.refractory, where=self.refractory > 0)
        if fired.size:
            self.v[fired] = self.v_reset
            self.refractory[fired] = self.refractory_steps
            self._deposit(fired)

        self._ring_ptr = (self._ring_ptr + 1) % self._ring.shape[0]
        self._record(fired)
        self.steps += 1
        return fired

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
        if len(self._hist) < self._hist_len:
            self._hist.append(fired)
        else:
            self._hist[self._hist_ptr] = fired
        self._hist_ptr = (self._hist_ptr + 1) % self._hist_len

    def get_firing_rates(self, window: int | None = None) -> np.ndarray:
        n = min(window if window is not None else self._hist_len, len(self._hist))
        if n <= 0:
            return np.zeros(self.N, dtype=np.float32)
        order = [(self._hist_ptr - 1 - k) % len(self._hist) for k in range(n)]
        spikes = np.concatenate([self._hist[i] for i in order]) if n else np.empty(0, np.int32)
        counts = np.bincount(spikes, minlength=self.N).astype(np.float32)
        return counts / (n * self.dt_ms / 1000.0)
