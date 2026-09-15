"""Homeostatic gain search — find the synaptic scale at which the network computes.

A real connectome with an arbitrary synaptic gain either goes silent or saturates.
Measured on the 44,687-neuron build: 0.155 ms/step at 1% firing, 1.1 ms/step at 13%
under a uniform drive of mu=12 — 440 ms per 400 ms tick. Cost is edges touched per
step, so the firing rate *is* the compute budget, and the usable band has to be found
deliberately.

Two choices are load-bearing:

*The drive is sparse and restricted to the sensory populations*, never uniform. A
uniform current is exactly what produced the cliff, and it is a regime the real system
never occupies — every input arrives at the lamina/medulla columns. Calibrating against
a uniform drive tunes for a network that does not exist.

*Failure is reported, not rounded off.* If the network jumps from silent to saturated
with no gain in between it is bistable, not calibrated, and `CalibrationResult.success`
is False with both bracketing points attached. Returning the closest miss as a success
would hide the one outcome the search exists to detect.

Measured on the real build, that is exactly what happens: the mean rate goes from
0.10 Hz at gain 2.03285 to 5.23 Hz at 2.03399 — a 0.06% change in gain — and no scalar
multiplier holds the band, the upper bracket staying bimodal with 76.6% of cells silent.
Worse, the quiet side is *metastable*, not stable: the network reads near-zero over
2,000 steps and tens of Hz over 8,000, because the avalanche takes seconds of simulated
time to ignite. So `measure_steps` is part of the claim, not a tuning knob — a short
window will call a supercritical network silent.

These figures are from the corrected, correctly-oriented connectome. An earlier run on a
transposed matrix put the cliff at gain ~2.9131; fixing the orientation moved it to
~2.0334 but did not soften it, so the bistability is a property of the network and this
LIF model, not of that bug.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from flybrain.engine.lif import LIFEngine

#: Lamina/medulla columns — the only place sensory current enters the network.
SENSORY_POPULATIONS = ("L1", "L2", "L3", "Tm1")

#: Resting band for fly central neurons. At dt=1 ms, 5 Hz is 0.5% of neurons per
#: step — a fifth of the 1% benchmark, so in-band is comfortably inside budget.
DEFAULT_TARGET_HZ = (1.0, 5.0)

#: A neuron above this fraction of its refractory-limited maximum is saturated.
SATURATION_FRACTION = 0.5

#: Sarle's bimodality coefficient of a uniform distribution. Above it, the rate
#: distribution is not the lognormal a healthy connectome produces.
BIMODALITY_UNIFORM = 5.0 / 9.0

Drive = Callable[[int], np.ndarray]


@dataclass(frozen=True)
class RateSummary:
    """The shape of the firing-rate distribution, not just its mean."""

    mean_hz: float
    median_hz: float
    percentiles_hz: dict[int, float]
    silent_fraction: float
    saturated_fraction: float
    log_mean: float
    log_std: float
    bimodality: float
    histogram: np.ndarray
    bin_edges_hz: np.ndarray

    @property
    def looks_lognormal(self) -> bool:
        return self.bimodality <= BIMODALITY_UNIFORM


@dataclass(frozen=True)
class Measurement:
    gain: float
    ms_per_step: float
    rates: RateSummary


@dataclass(frozen=True)
class CalibrationResult:
    target_hz: tuple[float, float]
    success: bool
    measurement: Measurement | None = None
    lower_bracket: Measurement | None = None
    upper_bracket: Measurement | None = None
    failure: str | None = None
    iterations: int = 0
    threshold_offsets: np.ndarray | None = field(default=None, repr=False)

    @property
    def gain(self) -> float | None:
        return self.measurement.gain if self.measurement else None


def summarize_rates(rates: np.ndarray, max_hz: float) -> RateSummary:
    rates = np.asarray(rates, dtype=np.float64)
    active = rates[rates > 0.0]
    log_rates = np.log10(active) if active.size else np.empty(0)
    hist, edges = np.histogram(active, bins=20, range=(0.0, max(max_hz, 1e-9)))
    return RateSummary(
        mean_hz=float(rates.mean()),
        median_hz=float(np.median(rates)),
        percentiles_hz={p: float(np.percentile(rates, p)) for p in (10, 25, 50, 75, 90, 99)},
        silent_fraction=float((rates == 0.0).mean()),
        saturated_fraction=float((rates >= SATURATION_FRACTION * max_hz).mean()),
        log_mean=float(log_rates.mean()) if log_rates.size else float("nan"),
        log_std=float(log_rates.std()) if log_rates.size else float("nan"),
        bimodality=_bimodality(log_rates),
        histogram=hist,
        bin_edges_hz=edges,
    )


def _bimodality(x: np.ndarray) -> float:
    """Sarle's coefficient (skew^2 + 1) / kurtosis. Above 5/9 means two modes."""
    if x.size < 4:
        return float("nan")
    centred = x - x.mean()
    sd = centred.std()
    if sd == 0.0:
        return float("inf")
    m3 = float((centred**3).mean()) / sd**3
    m4 = float((centred**4).mean()) / sd**4
    return (m3**2 + 1.0) / m4


def sensory_drive(
    n: int,
    indices: np.ndarray,
    *,
    amplitude: float = 20.0,
    active_fraction: float = 0.01,
    frame_steps: int = 100,
    seed: int = 0,
) -> Drive:
    """A sparse current into `indices` — the stand-in for real encoder output.

    The pattern is held for `frame_steps` because the membrane charges over
    `tau_m`: a current resampled every step charges nothing and the network stays
    silent regardless of gain. 100 steps is one of the encoder's four sub-frames
    per 400 ms tick, so this is also what the real input looks like.

    Each frame is seeded from its own index rather than from a running generator,
    so the drive is a pure function of `step`. A stateful generator would hand a
    different input sequence to every iteration of the gain search, and the gains
    would not be comparable — which is how a metastable network gets mistaken for
    a calibrated one.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("sensory_drive needs at least one target index")
    k = max(1, round(indices.size * active_fraction))
    frames: dict[int, np.ndarray] = {}

    def drive(step: int) -> np.ndarray:
        frame = step // frame_steps
        if frame not in frames:
            frames.clear()
            rng = np.random.default_rng([seed, frame])
            current = np.zeros(n, dtype=np.float32)
            current[rng.choice(indices, k, replace=False)] = amplitude
            frames[frame] = current
        return frames[frame]

    return drive


def _as_drive(drive: Drive | np.ndarray | None, n: int, populations) -> Drive:
    if callable(drive):
        return drive
    if isinstance(drive, np.ndarray):
        constant = drive.astype(np.float32, copy=True)
        return lambda step: constant
    if populations:
        present = [populations[name] for name in SENSORY_POPULATIONS if name in populations]
        if not present:
            raise KeyError(f"none of {SENSORY_POPULATIONS} are in this build")
        return sensory_drive(n, np.concatenate(present))
    return sensory_drive(n, np.arange(n))


def measure_gain(
    W: sp.csc_matrix,
    gain: float,
    drive: Drive,
    *,
    dt_ms: float = 1.0,
    warmup_steps: int = 500,
    measure_steps: int = 2000,
    v_thresh: np.ndarray | None = None,
    engine_kwargs: dict | None = None,
) -> Measurement:
    """Run the engine at `gain` and report what the network actually does."""
    scaled = sp.csc_matrix(
        (W.data.astype(np.float32) * np.float32(gain), W.indices, W.indptr), shape=W.shape
    )
    engine = LIFEngine(
        scaled,
        dt_ms=dt_ms,
        rate_window_ms=measure_steps * dt_ms,
        **{"spontaneous_noise_std": 0.5, "seed": 0, **(engine_kwargs or {})},
    )
    if v_thresh is not None:
        engine.v_thresh = np.asarray(v_thresh, dtype=np.float32)
    max_hz = 1000.0 / (dt_ms * (engine.refractory_steps + 1))

    for step in range(warmup_steps):
        engine.step(drive(step))
    elapsed = 0.0
    for step in range(warmup_steps, warmup_steps + measure_steps):
        current = drive(step)
        t0 = time.perf_counter()
        engine.step(current)
        elapsed += time.perf_counter() - t0
    return Measurement(
        gain=float(gain),
        ms_per_step=elapsed * 1000.0 / measure_steps,
        rates=summarize_rates(engine.get_firing_rates(), max_hz),
    )


def calibrate_gain(
    W: sp.csc_matrix,
    *,
    target_hz: tuple[float, float] = DEFAULT_TARGET_HZ,
    drive: Drive | np.ndarray | None = None,
    populations: dict[str, np.ndarray] | None = None,
    gain_range: tuple[float, float] = (1e-3, 10.0),
    max_iter: int = 14,
    dt_ms: float = 1.0,
    warmup_steps: int = 500,
    measure_steps: int = 2000,
    trim_thresholds: bool = False,
    trim_rounds: int = 3,
    engine_kwargs: dict | None = None,
) -> CalibrationResult:
    """Binary-search a scalar synaptic multiplier landing the mean rate in band."""
    lo_band, hi_band = target_hz
    n = W.shape[0]
    drive_fn = _as_drive(drive, n, populations)
    kw = {
        "dt_ms": dt_ms,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "engine_kwargs": engine_kwargs,
    }

    def run(gain: float) -> Measurement:
        return measure_gain(W, gain, drive_fn, **kw)

    def done(m: Measurement, iterations: int) -> CalibrationResult:
        offsets = None
        if trim_thresholds:
            offsets = trim_population_thresholds(
                W,
                m.gain,
                drive_fn,
                populations or {},
                target_hz=target_hz,
                rounds=trim_rounds,
                **kw,
            )
        return CalibrationResult(
            target_hz=target_hz,
            success=True,
            measurement=m,
            iterations=iterations,
            threshold_offsets=offsets,
        )

    lo, hi = gain_range
    lo_m, hi_m = run(lo), run(hi)
    if lo_m.rates.mean_hz > hi_band:
        return CalibrationResult(
            target_hz=target_hz,
            success=False,
            lower_bracket=lo_m,
            upper_bracket=hi_m,
            failure=f"already above band at the minimum gain {lo:g} "
            f"({lo_m.rates.mean_hz:.2f} Hz) — widen gain_range downward",
            iterations=2,
        )
    if hi_m.rates.mean_hz < lo_band:
        return CalibrationResult(
            target_hz=target_hz,
            success=False,
            lower_bracket=lo_m,
            upper_bracket=hi_m,
            failure=f"still below band at the maximum gain {hi:g} "
            f"({hi_m.rates.mean_hz:.2f} Hz) — widen gain_range upward or raise the drive",
            iterations=2,
        )
    for m in (lo_m, hi_m):
        if lo_band <= m.rates.mean_hz <= hi_band:
            return done(m, 2)

    for i in range(max_iter):
        mid = float(np.sqrt(lo * hi))
        m = run(mid)
        if lo_band <= m.rates.mean_hz <= hi_band:
            return done(m, 3 + i)
        if m.rates.mean_hz < lo_band:
            lo, lo_m = mid, m
        else:
            hi, hi_m = mid, m

    return CalibrationResult(
        target_hz=target_hz,
        success=False,
        lower_bracket=lo_m,
        upper_bracket=hi_m,
        failure=(
            f"bistable: gain {lo_m.gain:.6g} gives {lo_m.rates.mean_hz:.2f} Hz and "
            f"{hi_m.gain:.6g} gives {hi_m.rates.mean_hz:.2f} Hz, with the band "
            f"{lo_band}-{hi_band} Hz never reached after {max_iter} bisections"
        ),
        iterations=2 + max_iter,
    )


def trim_population_thresholds(
    W: sp.csc_matrix,
    gain: float,
    drive: Drive,
    populations: dict[str, np.ndarray],
    *,
    target_hz: tuple[float, float] = DEFAULT_TARGET_HZ,
    rounds: int = 3,
    step_mv: float = 1.0,
    v_thresh: float = -50.0,
    **kw,
) -> np.ndarray:
    """Second stage: nudge each population's threshold toward the band. Off by default."""
    lo_band, hi_band = target_hz
    thresholds = np.full(W.shape[0], v_thresh, dtype=np.float32)
    named = {k: v for k, v in populations.items() if "|" not in k}
    for _ in range(rounds):
        for name, rate in _population_rates(W, gain, drive, thresholds, named, **kw).items():
            if rate > hi_band:
                thresholds[named[name]] += step_mv
            elif rate < lo_band:
                thresholds[named[name]] -= step_mv
    return thresholds


def _population_rates(
    W: sp.csc_matrix,
    gain: float,
    drive: Drive,
    thresholds: np.ndarray,
    named: dict[str, np.ndarray],
    *,
    dt_ms: float = 1.0,
    warmup_steps: int = 500,
    measure_steps: int = 2000,
    engine_kwargs: dict | None = None,
) -> dict[str, float]:
    scaled = sp.csc_matrix(
        (W.data.astype(np.float32) * np.float32(gain), W.indices, W.indptr), shape=W.shape
    )
    engine = LIFEngine(
        scaled,
        dt_ms=dt_ms,
        rate_window_ms=measure_steps * dt_ms,
        **{"spontaneous_noise_std": 0.5, "seed": 0, **(engine_kwargs or {})},
    )
    engine.v_thresh = thresholds
    for step in range(warmup_steps + measure_steps):
        engine.step(drive(step))
    rates = engine.get_firing_rates()
    return {name: float(rates[idx].mean()) for name, idx in named.items() if len(idx)}


def format_report(result: CalibrationResult) -> str:
    lo, hi = result.target_hz
    lines = [
        f"target band      {lo}-{hi} Hz",
        f"iterations       {result.iterations}",
        f"status           {'CALIBRATED' if result.success else 'FAILED'}",
    ]
    if result.failure:
        lines.append(f"reason           {result.failure}")
    for label, m in (
        ("", result.measurement),
        ("lower bracket", result.lower_bracket),
        ("upper bracket", result.upper_bracket),
    ):
        if m is None:
            continue
        r = m.rates
        lines += [
            "",
            f"--- {label or 'calibrated point'} ---",
            f"gain             {m.gain:.6g}",
            f"mean / median    {r.mean_hz:.3f} / {r.median_hz:.3f} Hz",
            "percentiles      " + "  ".join(f"p{p}={v:.2f}" for p, v in r.percentiles_hz.items()),
            f"silent           {r.silent_fraction:.1%}",
            f"saturated        {r.saturated_fraction:.1%}",
            (
                f"log10 rate       mean {r.log_mean:.3f}  std {r.log_std:.3f}  "
                f"bimodality {r.bimodality:.3f} "
                f"({'lognormal-ish' if r.looks_lognormal else 'BIMODAL'})"
            ),
            f"ms/step          {m.ms_per_step:.3f}",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from flybrain.connectome.loader import DEFAULT_PATH, load

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--path", type=Path, default=DEFAULT_PATH)
    p.add_argument("--target", type=float, nargs=2, default=list(DEFAULT_TARGET_HZ))
    p.add_argument("--gain-range", type=float, nargs=2, default=[1e-3, 10.0])
    p.add_argument("--max-iter", type=int, default=14)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--measure-steps", type=int, default=2000)
    p.add_argument("--amplitude", type=float, default=20.0)
    p.add_argument("--active-fraction", type=float, default=0.01)
    p.add_argument("--trim-thresholds", action="store_true")
    args = p.parse_args(argv)

    connectome = load(args.path)
    present = [
        connectome.populations[n] for n in SENSORY_POPULATIONS if n in connectome.populations
    ]
    drive = sensory_drive(
        connectome.n,
        np.concatenate(present),
        amplitude=args.amplitude,
        active_fraction=args.active_fraction,
    )
    result = calibrate_gain(
        connectome.W,
        target_hz=tuple(args.target),
        drive=drive,
        populations=connectome.populations,
        gain_range=tuple(args.gain_range),
        max_iter=args.max_iter,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        trim_thresholds=args.trim_thresholds,
    )
    print(f"connectome       {args.path}  N={connectome.n}  nnz={connectome.W.nnz}")
    print(format_report(result))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
