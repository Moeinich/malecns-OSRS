"""Homeostatic gain search — find the synaptic scale at which the network computes.

A real connectome with an arbitrary synaptic gain either goes silent or saturates.
Measured on the 44,687-neuron build: 0.155 ms/step at 1% firing, 1.1 ms/step at 13%
under a uniform drive of mu=12 — 1.1 s per 600 ms tick. Cost is edges touched per
step, so the firing rate *is* the compute budget, and the usable band has to be found
deliberately.

Three choices are load-bearing:

*The accepted point must beat the connectome-free null.* A rate in band is not
evidence the graph did anything: under the tonic that finally got the median
neuron firing, a zero weight matrix also scores 1.04 Hz, so the search would have
reported CALIBRATED at gain 0.001. Every accepted point is now measured against
the same drive with `W = 0` and has to exceed it by `Acceptance.null_margin`.
This is the shuffle control's logic applied to the search itself.


*The drive is the encoder's own output*, never uniform and no longer a stand-in for it.
A uniform current is exactly what produced the cliff, and it is a regime the real system
never occupies — every input arrives at the lamina/medulla columns. But a sparse pattern
over those columns is not what arrives either: `sensory_drive` lit 1% of the injection
layer where `encode()` lights 62-72% of it, a 55x difference in injected current that put
the bench at 2.20 Hz and the live loop at 4.5 Hz on the same gain. `encoder_drive` renders
real retina sub-frames and pushes them through `encode()`, so the figure the search
reports is the figure the loop runs at.

*Failure is reported, not rounded off.* If the network jumps from silent to saturated
with no gain in between it is bistable, not calibrated, and `CalibrationResult.success`
is False with both bracketing points attached. Returning the closest miss as a success
would hide the one outcome the search exists to detect.

Measured on the real build, that is exactly what happens: the mean rate goes from
0.10 Hz at gain 2.03285 to 5.23 Hz at 2.03399 — a 0.06% change in gain — and no scalar
multiplier holds the band, the upper bracket staying bimodal with 76.6% of cells silent.
Worse, the quiet side is *metastable*, not stable: the network reads near-zero over
2 s and tens of Hz over 8 s, because the avalanche takes seconds of simulated time to
ignite. So the measurement window is part of the claim, not a tuning knob — a short
window will call a supercritical network silent. The default is therefore tied to the
timescale on which ignition was actually observed (8 s of simulated time), not to a
round number, and it is held in *ms* so that `dt` cannot shorten it silently.

These figures are from the corrected, correctly-oriented connectome. An earlier run on a
transposed matrix put the cliff at gain ~2.9131; fixing the orientation moved it to
~2.0334 but did not soften it, so the bistability is a property of the network and this
LIF model, not of that bug.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from flybrain.connectome.loader import DEFAULT_INCOMING_CAP, normalize_incoming, superclasses
from flybrain.connectome.vocab import DESCENDING_SUPERCLASS
from flybrain.engine.lif import LIFEngine

#: Resting band for fly central neurons. At dt=1 ms, 5 Hz is 0.5% of neurons per
#: step — a fifth of the 1% benchmark, so in-band is comfortably inside budget.
#:
#: Deliberately *not* raised now that the tonic puts a ~1.04 Hz floor under the
#: band. Raising the floor to clear the null would be the wrong fix twice over:
#: the band is a claim about fly physiology, not about our drive, and a floor
#: tuned to sit above today's tonic silently stops being above tomorrow's. The
#: null is handled where it belongs, by `Acceptance.null_margin`.
DEFAULT_TARGET_HZ = (1.0, 5.0)

#: How far above the connectome-free null an accepted point has to sit.
#:
#: The measurement that forced this clause, on the full 184,110-neuron build
#: under the default normalisation and tonic:
#:
#:     tonic only, W = 0       mean 1.041 Hz, median 1.000
#:     real W, gain 0.001      mean 1.042 Hz, median 1.000
#:     real W, gain 0.4        mean 2.729 Hz, median 1.750
#:
#: A network with *no connectome whatsoever* scores inside the 1-5 Hz band, so
#: the search would have reported CALIBRATED at gain 0.001 with the graph
#: contributing nothing — FlyBrain's own failure mode, a noise generator with a
#: graph attached. 1.5x is the smallest margin that is unambiguously the network
#: rather than measurement scatter; the accepted point clears it at 2.6x.
DEFAULT_NULL_MARGIN = 1.5

#: The name of that clause, so a rejection can be recognised without parsing prose.
NULL_CLAUSE = "connectome_contribution"

#: How many standard deviations an attackable NPC in the fovea has to move the
#: target population's pooled rate.
#:
#: The clause exists because the calibration accepted a network that ticks in band
#: and does not compute. At gain 0.29907, tonic 0.9, capped/500, noise 2.83, dt 2
#: — a point that passes the band, the silent and saturated fractions, the
#: bimodality and the null by 2.2x — a prey stimulus dead ahead measures
#:
#:     injected cells      z = 38
#:     hop 1               z = 4.4   (0.02 on the shuffle, so it is the wiring)
#:     hop 2               z = -0.02, zero responding cells of 45,126
#:     descending_neuron   z = -0.04
#:
#: Every rate clause is a statement about the *distribution*; none of them can see
#: that the input never reaches the output. 2.0 is `tools/probe_prey.py`'s own bar.
DEFAULT_MIN_PROPAGATION_Z = 2.0

#: The name of that clause, so a rejection can be recognised without parsing prose.
PROPAGATION_CLAUSE = "propagation"

#: The propagation windows, in ticks, and the loop's own substeps per sub-frame.
#: 5 + 50 ticks of 4x19 steps is 4,180 steps per scene, 8,360 per evaluation —
#: seconds, not the 8 s the rate measurement needs, because a z between two scenes
#: is a difference and not a claim about metastability. `substeps` is the live
#: loop's floor rather than `FRAME_MS / dt`: the claim is about the regime the
#: brain actually runs in.
PROPAGATION_TICKS = 50
PROPAGATION_WARMUP_TICKS = 5
PROPAGATION_SUBSTEPS = 19

#: Where the probe puts its prey: 4 tiles dead ahead, inside the gaze wedge.
PROPAGATION_NPC_TILES = 4.0

#: Ticks of the walk the two propagation scenes run over. **1, i.e. standing
#: still**, unlike the gain measurement, and that is the difference between a
#: measurement and a number. `z` divides by the *unstimulated* scene's std, so a
#: walking baseline puts the walk's own frame-to-frame variance in the
#: denominator: measured at the calibrated point, the injected cells read z = 38
#: standing still and z = 0.14 over a 128-tick walk, for the same prey. The second
#: figure is a statement about how much the scene changes when you move, not about
#: whether the stimulus arrived, and under it no network could ever clear the
#: clause at hop 0, let alone downstream.
PROPAGATION_PATH_TICKS = 1

#: A neuron above this fraction of its refractory-limited maximum is saturated.
SATURATION_FRACTION = 0.5

#: Sarle's bimodality coefficient of a uniform distribution. Above it, the rate
#: distribution is not the lognormal a healthy connectome produces.
BIMODALITY_UNIFORM = 5.0 / 9.0

#: Membrane noise, in current units. One definition, read by calibration *and* by
#: the live brain — calibrating with noise and running without it makes the two
#: different networks, and with a subthreshold input the noise is the only thing
#: that ever initiates activity.
#:
#: It was 0.5, and at 0.5 it initiates nothing. The noise enters `v` through the
#: same `1 - exp(-dt/tau_m)` the drive does and is then low-passed by `tau_m`, so
#: its standing deviation on the membrane is `std * sqrt((1-av)/(1+av))` = 0.158
#: of the current figure: 0.079 mV against the 1.5 mV a 0.9 tonic leaves short of
#: threshold, which is 19 sigma and never happens. Measured floor rate on an
#: unconnected 2,000-cell net over 4 s, tonic 0.9:
#:
#:     std   0.5     1.0     2.0     4.0     8.0
#:     Hz    0.00    0.00    0.00    1.01    3.71
#:     silent 100%   100%    99.7%   0.1%    0.0%
#:
#: 4.0 is the first value that gives every neuron a floor, and it lands on
#: ornata's ~1.2 Hz. Against an encoder peak of 6.0 (a 6 mV displacement) the
#: noise contributes 0.63 mV, so the sensory signal is still 10x the jitter.
#:
#: **This is the value at dt = 1 ms**, and it does not transfer: see
#: `noise_std_for_dt`. The engine draws one sample per step, so a fixed std is a
#: different membrane process at every dt.
DEFAULT_SPONTANEOUS_NOISE_STD = 4.0

#: The `dt` the search and the live brain run at, in ms.
#:
#: It was 1.0, and at 1.0 the tick cannot be simulated. The per-step cost splits
#: into a dense O(N) part (voltage, conductance and adaptation decay, and the
#: noise draw over all 184,110 neurons) and an event-driven part (edges touched),
#: and only the dense part scales with the step count. Measured on the full build
#: at the calibrated operating point, per 600 ms tick:
#:
#:     dt  substeps  dense floor  total    mean Hz  x null
#:     1   600       433.5 ms     590.2 ms   2.15    2.06
#:     2   300       216.4 ms     352.8 ms   2.20    2.36
#:     3   200       144.5 ms     274.6 ms   2.30    2.69
#:
#: dt = 1 cannot fit the 360 ms deadline at any gain: its *dense floor alone*,
#: measured with a zero weight matrix and nothing firing, is 433 ms. Neither the
#: gain nor the band nor the sub-frame count is a lever on that — sub-frames in
#: particular are not a lever on anything here, since the substeps per tick are
#: `tick_ms / dt` however they are grouped.
#:
#: 2, not 3, because of what the derived constants do. `refractory_steps` is
#: `round(2.0 / dt)`: exact at dt = 2, and at dt = 3 it rounds to one step and
#: stretches the refractory period to 3 ms. `delay_slots` is `round(1.8 / dt)`:
#: 2 ms at dt = 2, 3 ms at dt = 3. dt = 3 buys 78 ms by distorting the two
#: constants that set the network's timescales; dt = 2 fits without that.
DEFAULT_DT_MS = 2.0


def noise_std_for_dt(
    dt_ms: float, tau_m: float = 20.0, std_at_1ms: float = DEFAULT_SPONTANEOUS_NOISE_STD
) -> float:
    """`spontaneous_noise_std` that holds the *membrane* noise fixed across `dt`.

    The engine draws one sample per step and passes it through the same
    `1 - exp(-dt/tau_m)` the drive gets, so a constant std is a coarser-and-louder
    process as `dt` grows: the standing deviation on `v` is
    `std * sqrt((1-av)/(1+av))`, which runs 0.158 -> 0.224 -> 0.274 of the current
    for dt = 1, 2, 3. Left alone that is not a smaller time step, it is a noisier
    neuron, and it shows up exactly where it does the most damage — the
    connectome-free null, which rose 1.041 -> 2.192 -> 2.848 Hz and ate the band
    the accepted point has to clear. Rescaled, the null holds at 1.04 -> 0.93 ->
    0.86 Hz and `dt` costs only time.
    """

    def sigma(dt: float) -> float:
        av = math.exp(-dt / tau_m)
        return math.sqrt((1.0 - av) / (1.0 + av))

    return std_at_1ms * sigma(1.0) / sigma(dt_ms)


#: `LIFEngine`'s own resting and threshold potentials. Repeated here so a tonic
#: can be expressed as a fraction of the distance between them rather than as a
#: bare current that silently changes meaning if either constant moves.
DEFAULT_V_REST = -65.0
DEFAULT_V_THRESH = -50.0

#: Tonic drive, as a fraction of `v_thresh - v_rest`.
#:
#: The engine's update is exact-exponential, so a constant drive `I` pulls `v` to
#: the fixed point `v_rest + I`: at 0.9 every neuron asymptotes 10% short of
#: threshold. Nothing fires from the tonic alone; what changes is that the
#: connectome and the noise then modulate *around* a floor instead of deciding
#: fire-versus-never, which is the failure mode adaptation could not touch —
#: adaptation only subtracts, so it is a no-op on a cell already below threshold.
DEFAULT_TONIC_FRACTION = 0.9

#: Which incoming-weight normalisation the search runs by default.
#:
#: `capped`, not `full`, and the sweep is why. All three modes were run at tonic
#: 0.9 over an 8,000-step window on the v1 build, gain 1.2 -> 3.5:
#:
#:     none    median 0.125 Hz, 48% silent, bimodal 0.51-0.56, at every gain
#:     full    median 1.000 Hz,  0% silent, but log_slope 0.00 — gain does nothing
#:     capped  median 1.875 -> 0.750 Hz, 12-40% silent, log_slope 0.60-1.12
#:
#: `full` fixes the silence and then throws the connectome away with it: bounding
#: every row at 1.0 leaves a recurrent contribution too small to compete with the
#: tonic, so the gain has no leverage and the brain is a noise generator with a
#: graph attached. `capped` keeps both — swept down, it holds the 1-5 Hz band
#: across gain 0.1 to 0.6, a 6x window where the old search had a cliff 0.06%
#: of a gain wide.
DEFAULT_NORMALIZATION = "capped"

Drive = Callable[[int], np.ndarray]

#: One game tick of simulated time, matching `bridge/config.ts`.
TICK_MS = 600
#: Sub-frames per tick, as `flybrain/loop/agent.py` renders them.
SUBFRAMES_PER_TICK = 4
#: One encoder sub-frame, in ms. The drive is held this long because the membrane
#: charges over tau_m; a current resampled every step charges nothing.
FRAME_MS = TICK_MS // SUBFRAMES_PER_TICK

#: The measurement window, in ms of simulated time.
#:
#: This is a claim, not a knob. The quiet branch of this network is metastable:
#: it reads 0.072 Hz over 2 s and 27.3 Hz over 8 s, because the avalanche needs
#: seconds of simulated time to ignite. A window shorter than the ignition
#: timescale calls a supercritical network calm, so the default is the window at
#: which ignition was actually observed — 8 s — and the cost of the longer run is
#: the price of the measurement being true.
#:
#: In *ms*, not steps, for the same reason `dt` is now calibrated: a window
#: pinned at 8,000 steps is 8 s at dt = 1 and 16 s at dt = 2, so the constant
#: would quietly stop meaning what its docstring says the moment dt moved.
DEFAULT_MEASURE_MS = 8000
DEFAULT_WARMUP_MS = 500


def steps_for(ms: float, dt_ms: float) -> int:
    """`ms` of simulated time as a step count at `dt_ms`. Never zero."""
    return max(1, round(ms / dt_ms))


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
    #: How far a prey stimulus moves the target population, and the injected
    #: cells, in standard deviations of the unstimulated scene. `None` means
    #: propagation was never measured — never that it passed.
    propagation_z: float | None = None
    input_z: float | None = None

    @property
    def looks_lognormal(self) -> bool:
        return self.bimodality <= BIMODALITY_UNIFORM


@dataclass(frozen=True)
class Acceptance:
    """What "calibrated" means, in one visible place.

    The mean rate is the only scalar monotone in gain, so it is what the search
    bisects on — but a mean is not a network. A measured point at 9.99 Hz mean was
    70.7% silent, bimodal, with a p99 of 172 Hz: two populations, one dead and one
    avalanching, whose average happens to look plausible. Every threshold here is a
    judgment call, which is why they are fields and not literals scattered through
    the search.

    `median_hz > 0` is the strongest clause, `null_margin` is the one that makes
    the band mean anything (`NULL_CLAUSE`), and `min_propagation_z` is the one
    that makes the whole network mean anything (`PROPAGATION_CLAUSE`): every other
    clause reads the rate distribution, and a distribution cannot show that the
    input never reaches the output.
    """

    target_hz: tuple[float, float] = DEFAULT_TARGET_HZ
    max_silent_fraction: float = 0.35
    max_saturated_fraction: float = 0.01
    max_p99_hz: float = 100.0
    require_median_above_zero: bool = True
    max_bimodality: float = BIMODALITY_UNIFORM
    null_margin: float = DEFAULT_NULL_MARGIN
    #: `None` skips the clause outright, for a drive that renders no scene. A
    #: number is a promise that propagation was measured: see `PROPAGATION_CLAUSE`.
    min_propagation_z: float | None = DEFAULT_MIN_PROPAGATION_Z
    propagation_superclass: str = DESCENDING_SUPERCLASS

    def reject(self, r: RateSummary, null_mean_hz: float | None = None) -> list[str]:
        """The clauses this rate distribution fails. Empty means accepted.

        `null_mean_hz` is the same network's rate with no connectome at all. It is
        optional only so a bare `RateSummary` can still be tested against the shape
        clauses; the search always supplies it.
        """
        lo, hi = self.target_hz
        p99 = r.percentiles_hz[99]
        contribution = None if not null_mean_hz else r.mean_hz / null_mean_hz
        z, z_in = r.propagation_z, r.input_z
        reached = z_in is not None and z_in >= (self.min_propagation_z or 0.0)
        no_propagation = (
            ""
            if z is None or self.min_propagation_z is None
            else (
                f"{PROPAGATION_CLAUSE}: prey in the fovea moves "
                f"{self.propagation_superclass} by z {z:.2f}, below the "
                f"{self.min_propagation_z:g} margin — "
                + (
                    f"input present (z {z_in:.2f} at the injected cells), propagation absent"
                    if reached
                    else "the stimulus never reached the injected cells either"
                )
            )
        )
        checks = (
            (not lo <= r.mean_hz <= hi, f"mean_hz {r.mean_hz:.3f} outside {lo}-{hi}"),
            (
                r.silent_fraction > self.max_silent_fraction,
                f"silent_fraction {r.silent_fraction:.1%} > {self.max_silent_fraction:.1%}",
            ),
            (
                r.saturated_fraction > self.max_saturated_fraction,
                (
                    f"saturated_fraction {r.saturated_fraction:.1%} > "
                    f"{self.max_saturated_fraction:.1%}"
                ),
            ),
            (p99 > self.max_p99_hz, f"p99_hz {p99:.1f} > {self.max_p99_hz}"),
            (
                self.require_median_above_zero and r.median_hz <= 0.0,
                "median_hz 0 — the median neuron never fires",
            ),
            (
                r.bimodality > self.max_bimodality,
                f"bimodality {r.bimodality:.3f} > {self.max_bimodality:.3f}",
            ),
            (
                contribution is not None and contribution < self.null_margin,
                (
                    f"{NULL_CLAUSE}: {r.mean_hz:.3f} Hz is {contribution:.2f}x the "
                    f"{null_mean_hz:.3f} Hz this drive produces with no connectome at "
                    f"all, below the {self.null_margin:g}x margin — the connectome is "
                    f"not contributing"
                    if contribution is not None
                    else ""
                ),
            ),
            (
                self.min_propagation_z is not None and z is not None and z < self.min_propagation_z,
                no_propagation,
            ),
        )
        return [why for failed, why in checks if failed]


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
    #: A point whose mean landed in band but whose distribution failed a clause.
    #: Never `measurement`: `measurement is not None` means calibrated, full stop.
    rejected: Measurement | None = None
    lower_bracket: Measurement | None = None
    upper_bracket: Measurement | None = None
    #: The same drive, tonic, noise and window with a zero weight matrix — what
    #: this run scores with no connectome at all. Measured only once a mean lands
    #: in band, because that is the only point whose acceptance it decides.
    null: Measurement | None = None
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
    amplitude: float,
    active_fraction: float = 0.01,
    dt_ms: float = DEFAULT_DT_MS,
    frame_steps: int | None = None,
    seed: int = 0,
) -> Drive:
    """A sparse current into `indices` — the stand-in for real encoder output.

    The pattern is held for `frame_steps` because the membrane charges over
    `tau_m`: a current resampled every step charges nothing and the network stays
    silent regardless of gain. `frame_steps` defaults to one of the encoder's four
    sub-frames per 600 ms tick *at `dt_ms`*, so this is also what the real input
    looks like — and it follows `dt` rather than pinning the frame to a step count
    that would mean a different length of held current at every `dt`.

    `amplitude` has no default on purpose. It used to be a bare 20.0 while the
    encoder capped at 6.0, so the search tuned a network the live brain never ran;
    callers now have to say which number they mean.

    Each frame is seeded from its own index rather than from a running generator,
    so the drive is a pure function of `step`. A stateful generator would hand a
    different input sequence to every iteration of the gain search, and the gains
    would not be comparable — which is how a metastable network gets mistaken for
    a calibrated one.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("sensory_drive needs at least one target index")
    if frame_steps is None:
        frame_steps = steps_for(FRAME_MS, dt_ms)
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


def encoder_drive(
    connectome,
    *,
    params=None,
    dt_ms: float = DEFAULT_DT_MS,
    frame_steps: int | None = None,
    subframes: int = SUBFRAMES_PER_TICK,
    collision=None,
    path_ticks: int = 128,
    npc_ahead: float | None = None,
    seed: int = 0,
) -> Drive:
    """The drive the live loop actually injects: real retina frames through `encode`.

    `sensory_drive` is a sparse stand-in and it is not what the encoder produces.
    Measured side by side on the v1 build under the calibrated `EncodeParams`
    (i_max 24), over the injection layer's 8,877 cells:

        sensory_drive     1.0% of the layer driven, flat 24.0, sum 2,136
        encode()         62-72% of the layer driven, mean 21.3 (p10 19.6,
                         median 22.6 = the Naka-Rushton ceiling), sum 118k-135k

    A 55x difference in injected current, so a gain calibrated against the sparse
    pattern lands the *bench* in band and the live loop at twice the rate and
    twice the event-driven cost — which is the whole of the missed deadline. Only
    the amplitude divergence was closed before; this closes the pattern.

    The scene is a deterministic walk over the real collision grid, so the drive
    stays a pure function of `step`: successive gains in the search must see an
    identical input sequence or their rates are not comparable. Frames are
    rendered a tick at a time and cached on the tick index, the same contract
    `sensory_drive` keeps on the frame index.
    """
    # Deferred: `flybrain.engine` must not depend on `flybrain.sensory` or the
    # loop. The injection layer and the raster have one definition each and this
    # is the point of reading them rather than approximating them.
    from flybrain.loop.types import Npc, Player, WorldState
    from flybrain.sensory.collision import CollisionGrid
    from flybrain.sensory.encode import EncodeParams, encode
    from flybrain.sensory.retina import Retina

    params = params if params is not None else EncodeParams()
    collision = collision if collision is not None else CollisionGrid.load()
    if frame_steps is None:
        frame_steps = steps_for(FRAME_MS, dt_ms)
    retina = Retina()
    path = _walk(collision, path_ticks, seed)

    def state(tick: int) -> WorldState:
        x, z = path[tick % len(path)]
        player = Player(
            name="calibrate",
            combat_level=3,
            hp=10,
            max_hp=10,
            x=int(x),
            z=int(z),
            level=collision.level,
            run_energy=100,
            anim_id=-1,
            in_combat=False,
            target_index=-1,
            target_type="none",
            is_dead=False,
            life_id=1,
        )
        return WorldState(
            tick=tick,
            in_game=True,
            modal_open=False,
            player=player,
            npcs=prey(tick, player),
            ground_items=(),
            locs=(),
            inventory=(),
            skills={},
            op_rejected_count=0,
        )

    def prey(tick: int, player: Player) -> tuple:
        """One attackable NPC `npc_ahead` tiles along the heading, or nothing.

        Copied from `tools/probe_prey.py` rather than imported: that module reads
        `flybrain.motor` at import, and `flybrain.engine` must not.
        """
        if npc_ahead is None:
            return ()
        bearing = heading(tick)
        return (
            Npc(
                id=1,
                index=1,
                name="prey",
                combat_level=1,
                x=round(player.x + npc_ahead * math.cos(bearing)),
                z=round(player.z + npc_ahead * math.sin(bearing)),
                size=1,
                distance=int(npc_ahead),
                hp=10,
                max_hp=10,
                in_combat=False,
                target_index=-1,
                reachable=True,
                options=("Attack",),
            ),
        )

    def heading(tick: int) -> float:
        x0, z0 = path[(tick - 1) % len(path)]
        x1, z1 = path[tick % len(path)]
        return math.atan2(z1 - z0, x1 - x0)

    cached: dict[int, np.ndarray] = {}

    def drive(step: int) -> np.ndarray:
        tick, sub = divmod(step // frame_steps, subframes)
        if tick not in cached:
            cached.clear()
            frames = retina.render_subframes(
                state(tick - 1),
                state(tick),
                collision,
                heading(tick),
                n=subframes,
                prev_heading=heading(tick - 1),
            )
            cached[tick] = np.stack([encode(f, connectome, params) for f in frames])
        return cached[tick][sub]

    return drive


def _walk(collision, ticks: int, seed: int) -> list[tuple[int, int]]:
    """A deterministic one-tile-per-tick walk over walkable ground.

    One tile per tick because that is what the body's step command produces, and
    the retina's flow — the only thing T4/T5 have to work with — is the
    difference between sub-frames of exactly that motion.
    """
    walkable = np.argwhere(collision.grid)
    if not walkable.size:
        raise ValueError("collision grid has no walkable tiles to calibrate over")
    rng = np.random.default_rng(seed)
    i, j = walkable[len(walkable) // 2]
    x, z = int(i) + collision.x_min, int(j) + collision.z_min
    steps = ((1, 0), (0, 1), (-1, 0), (0, -1))
    out = [(x, z)]
    for _ in range(ticks - 1):
        for k in rng.permutation(len(steps)):
            dx, dz = steps[k]
            if collision.is_walkable(x + dx, z + dz):
                x, z = x + dx, z + dz
                break
        out.append((x, z))
    return out


#: `(input_z, target_z)` at a candidate gain: the propagation measurement.
Propagation = Callable[[sp.csc_matrix, float], tuple[float, float]]


def propagation_probe(
    connectome,
    *,
    params=None,
    dt_ms: float = DEFAULT_DT_MS,
    tonic_fraction: float = 0.0,
    noise_std: float = DEFAULT_SPONTANEOUS_NOISE_STD,
    superclass: str = DESCENDING_SUPERCLASS,
    substeps: int = PROPAGATION_SUBSTEPS,
    ticks: int = PROPAGATION_TICKS,
    warmup_ticks: int = PROPAGATION_WARMUP_TICKS,
    npc_tiles: float = PROPAGATION_NPC_TILES,
    path_ticks: int = PROPAGATION_PATH_TICKS,
    collision=None,
    seed: int = 0,
    engine_kwargs: dict | None = None,
) -> Propagation:
    """Does a stimulus reach `superclass` at all? Returns a probe of `(W, gain)`.

    Two windows from the same seed over the calibration's own walk, differing only
    in whether an attackable NPC stands `npc_tiles` dead ahead. The pooled rate per
    tick of the target population gives `z = (meanB - meanA) / stdA`, and the same
    z at the injected cells says whether a z of zero downstream means "no
    propagation" or "no input".

    The hop-0 pool is every cell the encoder drives, not `probe_prey`'s chromatic
    row, so it is diluted by the cells the prey never touches: at the calibrated
    point it reads z 4.52 where the probe's narrower pool reads 38. Same statement
    — the stimulus arrives — an order of magnitude apart, so the two numbers are
    not interchangeable.

    The target population is resolved *now*, not at the first evaluation: a search
    that discovers after eight minutes that it cannot check the clause has already
    spent the eight minutes, and a clause that cannot be checked must not be
    claimed. Missing annotations are therefore an error here, and the caller's
    answer is `--no-propagation`, not silence.
    """
    codes = superclasses(connectome)
    if codes is None:
        raise RuntimeError(
            f"{PROPAGATION_CLAUSE}: no superclass annotations, so {superclass} cannot be "
            "located and propagation cannot be measured — build the annotations or pass "
            "--no-propagation to skip the clause explicitly"
        )
    code, labels = codes
    if superclass not in labels:
        raise RuntimeError(
            f"{PROPAGATION_CLAUSE}: superclass {superclass!r} is not in this build "
            f"({len(labels)} labels) — propagation cannot be measured"
        )
    targets = np.flatnonzero(code == labels.index(superclass))
    if not targets.size:
        raise RuntimeError(f"{PROPAGATION_CLAUSE}: no {superclass} cells in this build")

    if collision is None:
        from flybrain.sensory.collision import CollisionGrid

        collision = CollisionGrid.load()
    kw = {
        "params": params,
        "dt_ms": dt_ms,
        "frame_steps": substeps,
        "path_ticks": path_ticks,
        "collision": collision,
        "seed": seed,
    }
    empty = encoder_drive(connectome, **kw)
    prey = encoder_drive(connectome, npc_ahead=npc_tiles, **kw)
    injected = np.flatnonzero(prey(0) > 0.0)
    tonic = tonic_current(tonic_fraction) if tonic_fraction else 0.0
    window = substeps * SUBFRAMES_PER_TICK
    pools = {"input": injected, "target": targets}

    def run(W: sp.csc_matrix, gain: float, drive: Drive) -> dict[str, np.ndarray]:
        scaled = sp.csc_matrix(
            (W.data.astype(np.float32) * np.float32(gain), W.indices, W.indptr), shape=W.shape
        )
        engine = LIFEngine(
            scaled,
            dt_ms=dt_ms,
            rate_window_ms=window * dt_ms,
            **{"spontaneous_noise_std": noise_std, "seed": 0, **(engine_kwargs or {})},
        )
        driven = with_tonic(drive, W.shape[0], tonic)
        out: dict[str, list[float]] = {name: [] for name in pools}
        for tick in range(warmup_ticks + ticks):
            for step in range(tick * window, (tick + 1) * window):
                engine.step(driven(step))
            if tick < warmup_ticks:
                continue
            rates = engine.get_firing_rates(window)
            for name, idx in pools.items():
                out[name].append(float(rates[idx].mean()) if len(idx) else 0.0)
        return {k: np.asarray(v) for k, v in out.items()}

    def probe(W: sp.csc_matrix, gain: float) -> tuple[float, float]:
        a, b = run(W, gain, empty), run(W, gain, prey)

        def z(name: str) -> float:
            std = float(a[name].std())
            return 0.0 if std == 0.0 else float((b[name].mean() - a[name].mean()) / std)

        return z("input"), z("target")

    return probe


def tonic_current(
    fraction: float, v_rest: float = DEFAULT_V_REST, v_thresh: float = DEFAULT_V_THRESH
) -> float:
    """The constant current that parks `v` at `fraction` of the way to threshold."""
    return float(fraction) * (v_thresh - v_rest)


def with_tonic(drive: Drive, n: int, tonic: float) -> Drive:
    """`drive` plus a constant current into every neuron, including unwired ones."""
    if not tonic:
        return drive
    base = np.full(n, np.float32(tonic), dtype=np.float32)
    return lambda step: drive(step) + base


def _as_drive(
    drive: Drive | np.ndarray | None,
    n: int,
    populations,
    injection_types: Sequence[str],
    amplitude: float | None,
) -> Drive:
    if callable(drive):
        return drive
    if isinstance(drive, np.ndarray):
        constant = drive.astype(np.float32, copy=True)
        return lambda step: constant
    if amplitude is None:
        raise ValueError("deriving a drive needs drive_amplitude — say which current you mean")
    if populations:
        if not injection_types:
            raise ValueError(
                "deriving a drive from populations needs injection_types "
                "(flybrain.sensory.encode.LUMINANCE_TYPES is what the live path reads)"
            )
        present = [populations[name] for name in injection_types if name in populations]
        if not present:
            raise KeyError(f"none of {tuple(injection_types)} are in this build")
        return sensory_drive(n, np.concatenate(present), amplitude=amplitude)
    return sensory_drive(n, np.arange(n), amplitude=amplitude)


def measure_gain(
    W: sp.csc_matrix,
    gain: float,
    drive: Drive,
    *,
    dt_ms: float = DEFAULT_DT_MS,
    warmup_steps: int | None = None,
    measure_steps: int | None = None,
    v_thresh: np.ndarray | None = None,
    noise_std: float = DEFAULT_SPONTANEOUS_NOISE_STD,
    engine_kwargs: dict | None = None,
) -> Measurement:
    """Run the engine at `gain` and report what the network actually does."""
    warmup_steps = steps_for(DEFAULT_WARMUP_MS, dt_ms) if warmup_steps is None else warmup_steps
    measure_steps = steps_for(DEFAULT_MEASURE_MS, dt_ms) if measure_steps is None else measure_steps
    scaled = sp.csc_matrix(
        (W.data.astype(np.float32) * np.float32(gain), W.indices, W.indptr), shape=W.shape
    )
    engine = LIFEngine(
        scaled,
        dt_ms=dt_ms,
        rate_window_ms=measure_steps * dt_ms,
        **{"spontaneous_noise_std": noise_std, "seed": 0, **(engine_kwargs or {})},
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
    acceptance: Acceptance | None = None,
    drive: Drive | np.ndarray | None = None,
    populations: dict[str, np.ndarray] | None = None,
    injection_types: Sequence[str] = (),
    drive_amplitude: float | None = None,
    gain_range: tuple[float, float] = (1e-3, 10.0),
    max_iter: int = 14,
    dt_ms: float = DEFAULT_DT_MS,
    warmup_steps: int | None = None,
    measure_steps: int | None = None,
    trim_thresholds: bool = False,
    trim_rounds: int = 3,
    noise_std: float = DEFAULT_SPONTANEOUS_NOISE_STD,
    normalize: str = "none",
    incoming_cap: float = DEFAULT_INCOMING_CAP,
    tonic_fraction: float = 0.0,
    propagation: Propagation | None = None,
    engine_kwargs: dict | None = None,
) -> CalibrationResult:
    """Bisect a scalar synaptic multiplier on the mean rate; accept on `Acceptance`.

    The mean is the search variable because it is the only scalar monotone in gain.
    It is *not* the acceptance test — a mean in band over a dead median is a failure,
    and it is returned as one, on `rejected` rather than `measurement`.

    `normalize` and `tonic_fraction` are the two structural knobs. A scalar gain on
    the raw matrix cannot work: per-neuron net input has median 3.0 and mean 111.7,
    so one multiplier means a different thing to every cell. Normalising incoming
    weight gives it one meaning; the tonic gives every cell a floor to modulate
    around. Both are searched *with*, and both are carried on the artifact.
    """
    accept = acceptance or Acceptance(target_hz=target_hz)
    lo_band, hi_band = accept.target_hz
    n = W.shape[0]
    W = normalize_incoming(W, normalize, cap=incoming_cap)
    drive_fn = with_tonic(
        _as_drive(drive, n, populations, injection_types, drive_amplitude),
        n,
        tonic_current(tonic_fraction) if tonic_fraction else 0.0,
    )
    kw = {
        "dt_ms": dt_ms,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "noise_std": noise_std,
        "engine_kwargs": engine_kwargs,
    }

    def run(gain: float) -> Measurement:
        return measure_gain(W, gain, drive_fn, **kw)

    # `drive_fn`, the tonic, the noise and the window are all fixed for the lifetime
    # of this call, so the null is too: measured once here, never on the bisection
    # path. This is the cache per (drive, tonic, noise, steps).
    null_m = measure_gain(sp.csc_matrix((n, n), dtype=np.float32), 1.0, drive_fn, **kw)
    null_hz = null_m.rates.mean_hz

    # The null does not merely reject at the end — it moves the floor the search is
    # looking for. Under the tonic the null is 1.04 Hz, inside a 1-5 Hz band, so a
    # search that stopped at the first in-band mean would stop at gain 1e-3 and
    # report the connectome missing rather than go and find where it contributes.
    lo_search = max(lo_band, null_hz * accept.null_margin)

    def settle(m: Measurement, iterations: int) -> CalibrationResult:
        """The mean is in band. Whether that is a calibration is a separate question."""
        # Measured here and nowhere else, like the null: only the point that is
        # about to be accepted has to answer for propagation.
        if propagation is not None and accept.min_propagation_z is not None:
            input_z, target_z = propagation(W, m.gain)
            m = replace(m, rates=replace(m.rates, propagation_z=target_z, input_z=input_z))
        clauses = accept.reject(m.rates, null_hz)
        if clauses:
            return CalibrationResult(
                target_hz=accept.target_hz,
                success=False,
                rejected=m,
                null=null_m,
                failure=f"mean in band at gain {m.gain:.6g} but rejected: " + "; ".join(clauses),
                iterations=iterations,
            )
        offsets = None
        if trim_thresholds:
            offsets = trim_population_thresholds(
                W,
                m.gain,
                drive_fn,
                populations or {},
                target_hz=accept.target_hz,
                rounds=trim_rounds,
                **kw,
            )
        return CalibrationResult(
            target_hz=accept.target_hz,
            success=True,
            measurement=m,
            null=null_m,
            iterations=iterations,
            threshold_offsets=offsets,
        )

    def failed(lo_m, hi_m, why, iterations) -> CalibrationResult:
        return CalibrationResult(
            target_hz=accept.target_hz,
            success=False,
            lower_bracket=lo_m,
            upper_bracket=hi_m,
            null=null_m,
            failure=why,
            iterations=iterations,
        )

    if lo_search > hi_band:
        return failed(
            None,
            None,
            f"{NULL_CLAUSE}: this drive alone reaches {null_hz:.3f} Hz with no connectome, "
            f"so {accept.null_margin:g}x it ({lo_search:.3f} Hz) is already past the band's "
            f"{hi_band} Hz ceiling — no gain in it could be the network's doing",
            1,
        )

    lo, hi = gain_range
    lo_m, hi_m = run(lo), run(hi)
    if lo_m.rates.mean_hz > hi_band:
        return failed(
            lo_m,
            hi_m,
            f"already above band at the minimum gain {lo:g} "
            f"({lo_m.rates.mean_hz:.2f} Hz) — widen gain_range downward",
            3,
        )
    if hi_m.rates.mean_hz < lo_search:
        raised = (
            ""
            if lo_search <= lo_band
            else (
                f" — {NULL_CLAUSE}: the floor here is {lo_search:.3f} Hz, "
                f"{accept.null_margin:g}x the {null_hz:.3f} Hz this drive reaches with no "
                f"connectome at all, and the connectome never pulls away from it"
            )
        )
        return failed(
            lo_m,
            hi_m,
            f"still below band at the maximum gain {hi:g} "
            f"({hi_m.rates.mean_hz:.2f} Hz) — widen gain_range upward or raise the drive" + raised,
            3,
        )
    for m in (lo_m, hi_m):
        if lo_search <= m.rates.mean_hz <= hi_band:
            return settle(m, 3)

    for i in range(max_iter):
        mid = float(np.sqrt(lo * hi))
        m = run(mid)
        if lo_search <= m.rates.mean_hz <= hi_band:
            return settle(m, 4 + i)
        if m.rates.mean_hz < lo_search:
            lo, lo_m = mid, m
        else:
            hi, hi_m = mid, m

    return failed(
        lo_m,
        hi_m,
        f"bistable: gain {lo_m.gain:.6g} gives {lo_m.rates.mean_hz:.2f} Hz and "
        f"{hi_m.gain:.6g} gives {hi_m.rates.mean_hz:.2f} Hz, with the band "
        f"{lo_search:.3g}-{hi_band} Hz never reached after {max_iter} bisections",
        3 + max_iter,
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
    dt_ms: float = DEFAULT_DT_MS,
    warmup_steps: int | None = None,
    measure_steps: int | None = None,
    noise_std: float = DEFAULT_SPONTANEOUS_NOISE_STD,
    engine_kwargs: dict | None = None,
) -> dict[str, float]:
    warmup_steps = steps_for(DEFAULT_WARMUP_MS, dt_ms) if warmup_steps is None else warmup_steps
    measure_steps = steps_for(DEFAULT_MEASURE_MS, dt_ms) if measure_steps is None else measure_steps
    scaled = sp.csc_matrix(
        (W.data.astype(np.float32) * np.float32(gain), W.indices, W.indptr), shape=W.shape
    )
    engine = LIFEngine(
        scaled,
        dt_ms=dt_ms,
        rate_window_ms=measure_steps * dt_ms,
        **{"spontaneous_noise_std": noise_std, "seed": 0, **(engine_kwargs or {})},
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
    if result.null is not None:
        scored = result.measurement or result.rejected
        null_hz = result.null.rates.mean_hz
        ratio = f"{scored.rates.mean_hz / null_hz:.2f}x" if scored and null_hz else "n/a"
        lines.append(f"connectome-free  {null_hz:.3f} Hz null; the point scores {ratio} it")
    for label, m in (
        ("", result.measurement),
        ("rejected point", result.rejected),
        ("lower bracket", result.lower_bracket),
        ("upper bracket", result.upper_bracket),
        ("connectome-free null", result.null),
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
            "propagation      "
            + (
                "unchecked"
                if r.propagation_z is None
                else (
                    f"z {r.propagation_z:.2f} at the target population, "
                    f"{r.input_z:.2f} at the injected cells"
                )
            ),
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from flybrain.connectome.loader import DEFAULT_PATH, NORMALIZATION_MODES, load

    # Deferred: `calibration` imports this module, so the cycle only closes here.
    from flybrain.engine.calibration import DEFAULT_CALIBRATION_PATH, Calibration, Fingerprint

    # Imported here, not at module scope: `flybrain.engine` has no business
    # depending on `flybrain.sensory`, but the injection layer has exactly one
    # definition and it is the one the live encoder reads.
    from flybrain.sensory.encode import LUMINANCE_TYPES, EncodeParams

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--path", type=Path, default=DEFAULT_PATH)
    p.add_argument("--target", type=float, nargs=2, default=list(DEFAULT_TARGET_HZ))
    # The band under the default normalisation and tonic sits near 0.4; 1e-3 to
    # 10 brackets it with room on both sides.
    p.add_argument("--gain-range", type=float, nargs=2, default=[1e-3, 10.0])
    p.add_argument("--max-iter", type=int, default=14)
    p.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT_MS,
        help="LIF step in ms; the calibration is only valid at the dt it was measured at",
    )
    p.add_argument("--warmup-ms", type=float, default=DEFAULT_WARMUP_MS)
    p.add_argument("--measure-ms", type=float, default=DEFAULT_MEASURE_MS)
    p.add_argument("--amplitude", type=float, default=EncodeParams().i_max)
    p.add_argument("--active-fraction", type=float, default=0.01)
    p.add_argument(
        "--drive",
        choices=("encoder", "synthetic"),
        default="encoder",
        help="encoder: real retina frames through encode(). synthetic: sensory_drive's "
        "sparse stand-in, which drives 1%% of the injection layer where the encoder "
        "drives 60%% — see encoder_drive",
    )
    p.add_argument(
        "--noise-std",
        type=float,
        default=None,
        help="default holds the membrane noise fixed across dt; see noise_std_for_dt",
    )
    p.add_argument("--normalize", choices=NORMALIZATION_MODES, default=DEFAULT_NORMALIZATION)
    p.add_argument("--incoming-cap", type=float, default=DEFAULT_INCOMING_CAP)
    p.add_argument("--tonic-fraction", type=float, default=DEFAULT_TONIC_FRACTION)
    p.add_argument(
        "--null-margin",
        type=float,
        default=DEFAULT_NULL_MARGIN,
        help="how many times the connectome-free null the accepted rate must be",
    )
    p.add_argument(
        "--min-propagation-z",
        type=float,
        default=DEFAULT_MIN_PROPAGATION_Z,
        help="how many standard deviations prey in the fovea must move "
        f"{DESCENDING_SUPERCLASS}; see PROPAGATION_CLAUSE",
    )
    p.add_argument(
        "--no-propagation",
        action="store_true",
        help="skip the propagation clause and say so, rather than claim it was checked",
    )
    p.add_argument("--trim-thresholds", action="store_true")
    p.add_argument(
        "--save",
        nargs="?",
        const=str(DEFAULT_CALIBRATION_PATH),
        default=None,
        help="write the accepted calibration here so the live brain picks it up",
    )
    args = p.parse_args(argv)

    noise_std = noise_std_for_dt(args.dt) if args.noise_std is None else args.noise_std
    warmup_steps = steps_for(args.warmup_ms, args.dt)
    measure_steps = steps_for(args.measure_ms, args.dt)

    connectome = load(args.path)
    params = replace(EncodeParams(), i_max=args.amplitude)
    if args.drive == "encoder":
        drive = encoder_drive(connectome, params=params, dt_ms=args.dt)
        described = f"encoder_drive, encode() over a walk, i_max {args.amplitude:g}"
    else:
        present = [
            connectome.populations[n] for n in LUMINANCE_TYPES if n in connectome.populations
        ]
        drive = sensory_drive(
            connectome.n,
            np.concatenate(present),
            amplitude=args.amplitude,
            active_fraction=args.active_fraction,
            dt_ms=args.dt,
        )
        described = (
            f"sensory_drive over {'/'.join(LUMINANCE_TYPES)}, amplitude {args.amplitude:g}, "
            f"active_fraction {args.active_fraction:g}"
        )
    # The synthetic drive renders no scene, so there is no prey stimulus to
    # propagate — skipped out loud, never quietly passed.
    check_propagation = not args.no_propagation and args.drive == "encoder"
    acceptance = Acceptance(
        target_hz=tuple(args.target),
        null_margin=args.null_margin,
        min_propagation_z=args.min_propagation_z if check_propagation else None,
    )
    probe = (
        propagation_probe(
            connectome,
            params=params,
            dt_ms=args.dt,
            tonic_fraction=args.tonic_fraction,
            noise_std=noise_std,
        )
        if check_propagation
        else None
    )
    result = calibrate_gain(
        connectome.W,
        acceptance=acceptance,
        drive=drive,
        populations=connectome.populations,
        gain_range=tuple(args.gain_range),
        max_iter=args.max_iter,
        dt_ms=args.dt,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        noise_std=noise_std,
        normalize=args.normalize,
        incoming_cap=args.incoming_cap,
        tonic_fraction=args.tonic_fraction,
        trim_thresholds=args.trim_thresholds,
        propagation=probe,
    )
    print(f"connectome       {args.path}  N={connectome.n}  nnz={connectome.W.nnz}")
    print(f"normalize        {args.normalize}  cap {args.incoming_cap:g}")
    print(
        f"dt               {args.dt:g} ms  -> {steps_for(TICK_MS, args.dt)} substeps per "
        f"{TICK_MS} ms tick, {measure_steps} step window"
    )
    print(
        f"tonic            {args.tonic_fraction:g} of threshold distance "
        f"= {tonic_current(args.tonic_fraction):.3f}  noise_std {noise_std:g}"
    )
    print(
        "propagation      "
        + (
            f"{DESCENDING_SUPERCLASS} must move z >= {args.min_propagation_z:g}, "
            f"{PROPAGATION_WARMUP_TICKS}+{PROPAGATION_TICKS} ticks x "
            f"{SUBFRAMES_PER_TICK}x{PROPAGATION_SUBSTEPS} substeps per scene"
            if check_propagation
            else "NOT CHECKED — "
            + (
                "--no-propagation"
                if args.no_propagation
                else "the synthetic drive renders no scene to put prey in"
            )
        )
    )
    print(format_report(result))
    if args.save and result.success:
        path = Calibration(
            gain=result.gain,
            dt_ms=args.dt,
            spontaneous_noise_std=noise_std,
            i_max=args.amplitude,
            normalization=args.normalize,
            incoming_cap=args.incoming_cap,
            tonic_fraction=args.tonic_fraction or None,
            acceptance=acceptance,
            rates=result.measurement.rates,
            measure_steps=measure_steps,
            drive=f"{described}, tonic {args.tonic_fraction:g}",
            connectome=Fingerprint.of(connectome),
        ).save(args.save)
        print(f"saved            {path}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
