from __future__ import annotations

import math
import os
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp

from flybrain.connectome import loader, vocab
from flybrain.connectome.loader import normalize_incoming
from flybrain.engine import calibrate as calibrate_module
from flybrain.engine.calibrate import (
    BIMODALITY_UNIFORM,
    DEFAULT_DT_MS,
    DEFAULT_SPONTANEOUS_NOISE_STD,
    DEFAULT_TONIC_FRACTION,
    DEFAULT_V_REST,
    DEFAULT_V_THRESH,
    FRAME_MS,
    NULL_CLAUSE,
    SATURATION_FRACTION,
    SUBFRAMES_PER_TICK,
    TICK_MS,
    Acceptance,
    calibrate_gain,
    encoder_drive,
    format_report,
    measure_gain,
    noise_std_for_dt,
    sensory_drive,
    steps_for,
    summarize_rates,
    tonic_current,
    with_tonic,
)
from flybrain.engine.lif import LIFEngine

#: `dt_ms` is pinned, not defaulted. These fixtures are hand-tuned currents against
#: a fixed threshold, so they measure the network only at the `dt` they were tuned
#: at — and `DEFAULT_DT_MS` is a production choice that will move again.
FAST = {"dt_ms": 1.0, "warmup_steps": 200, "measure_steps": 400}
#: A 400-step window quantizes rates to 2.5 Hz, so at a 1-5 Hz mean almost every
#: neuron reads 0 or 1 spikes and the silent fraction measures the window, not the
#: network. Any test asserting the *acceptance* predicate needs a window that can
#: resolve the band it is accepting.
HONEST = {"dt_ms": 1.0, "warmup_steps": 500, "measure_steps": 4000}


def _graded_net(n: int = 500, p: float = 0.02, inhibitory: float = 0.2, seed: int = 1):
    """Sparse E/I random net whose mean rate is a graded function of gain."""
    rng = np.random.default_rng(seed)
    M = (rng.random((n, n)) < p).astype(np.float32)
    np.fill_diagonal(M, 0.0)
    sign = np.where(rng.random(n) < inhibitory, -2.0, 1.0).astype(np.float32)
    return sp.csc_matrix(M * sign[None, :])


def _bistable_net(n: int = 200):
    """All-to-all excitation, no inhibition: one spike recruits everything."""
    M = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(M, 0.0)
    return sp.csc_matrix(M)


def _drive(n: int, *, background: float, sensory: int, active_fraction: float, seed: int = 0):
    sparse = sensory_drive(
        n, np.arange(sensory), amplitude=25.0, active_fraction=active_fraction, seed=seed
    )
    return lambda step: sparse(step) + np.float32(background)


def test_summarize_rates_reports_the_distribution():
    rates = np.array([0.0, 0.0, 1.0, 2.0, 4.0, 100.0], dtype=np.float32)
    s = summarize_rates(rates, max_hz=200.0)
    assert s.mean_hz == pytest.approx(107.0 / 6)
    assert s.median_hz == pytest.approx(1.5)
    assert s.silent_fraction == pytest.approx(2 / 6)
    assert s.saturated_fraction == pytest.approx(1 / 6)  # only 100 >= 0.5 * 200
    assert s.percentiles_hz[50] == pytest.approx(1.5)
    assert s.log_mean == pytest.approx(np.log10([1.0, 2.0, 4.0, 100.0]).mean())
    assert s.histogram.sum() == 4  # silent neurons are excluded


def test_saturation_threshold_is_a_fraction_of_the_refractory_limit():
    max_hz = 200.0
    just_under = np.full(4, SATURATION_FRACTION * max_hz - 1e-3, dtype=np.float32)
    assert summarize_rates(just_under, max_hz).saturated_fraction == 0.0
    assert summarize_rates(just_under + 1.0, max_hz).saturated_fraction == 1.0


def test_bimodality_separates_one_mode_from_two():
    rng = np.random.default_rng(0)
    unimodal = summarize_rates(10 ** rng.normal(0.5, 0.3, 4000), max_hz=1e6)
    assert unimodal.bimodality < BIMODALITY_UNIFORM
    assert unimodal.looks_lognormal

    two_modes = np.concatenate([np.full(2000, 1.0), np.full(2000, 100.0)])
    split = summarize_rates(two_modes, max_hz=1e6)
    assert split.bimodality > BIMODALITY_UNIFORM
    assert not split.looks_lognormal


def test_too_low_gain_is_silent_and_too_high_gain_is_saturated():
    n = 500
    W = _graded_net(n)
    drive = _drive(n, background=12.0, sensory=50, active_fraction=0.1)

    cold = measure_gain(W, 1.0, drive, **FAST)
    assert cold.rates.mean_hz < 1.0
    assert cold.rates.silent_fraction > 0.9
    assert cold.rates.saturated_fraction == 0.0

    hot = measure_gain(W, 60.0, drive, **FAST)
    assert hot.rates.mean_hz > 5.0
    assert hot.rates.saturated_fraction > 0.2
    assert hot.ms_per_step > cold.ms_per_step  # firing rate is the compute budget


def test_search_finds_a_gain_in_band_when_one_exists():
    n = 500
    W = _graded_net(n)
    drive = _drive(n, background=12.0, sensory=50, active_fraction=0.1)

    result = calibrate_gain(W, drive=drive, gain_range=(1.0, 30.0), max_iter=12, **HONEST)

    assert result.success, result.failure
    assert result.failure is None
    assert 1.0 <= result.measurement.rates.mean_hz <= 5.0
    assert 1.0 < result.gain < 30.0
    assert "CALIBRATED" in format_report(result)


def test_bistable_network_is_reported_as_failed_not_as_a_near_miss():
    n = 200
    W = _bistable_net(n)
    drive = _drive(n, background=14.0, sensory=10, active_fraction=0.1)

    # Pinned: this fixture sits 1 mV below threshold, so the default membrane
    # noise now fires it on its own and there is no quiet bracket to find.
    result = calibrate_gain(
        W, drive=drive, gain_range=(1e-5, 10.0), max_iter=12, noise_std=0.5, **FAST
    )

    assert not result.success
    assert result.measurement is None
    assert "bistable" in result.failure
    assert result.lower_bracket.rates.mean_hz < 1.0
    assert result.upper_bracket.rates.mean_hz > 5.0
    assert result.upper_bracket.rates.saturated_fraction > 0.5
    # The brackets are adjacent: there is no gain in between, which is the finding.
    assert result.upper_bracket.gain / result.lower_bracket.gain < 1.05
    assert "FAILED" in format_report(result)


def test_gain_range_endpoints_that_miss_the_band_are_named():
    n = 500
    W = _graded_net(n)
    drive = _drive(n, background=12.0, sensory=50, active_fraction=0.1)

    too_cold = calibrate_gain(W, drive=drive, gain_range=(0.1, 1.0), max_iter=4, **FAST)
    assert not too_cold.success
    assert "below band at the maximum gain" in too_cold.failure

    too_hot = calibrate_gain(W, drive=drive, gain_range=(60.0, 200.0), max_iter=4, **FAST)
    assert not too_hot.success
    assert "above band at the minimum gain" in too_hot.failure


def test_sensory_drive_holds_a_pattern_for_a_whole_frame():
    drive = sensory_drive(50, np.arange(20), amplitude=20.0, active_fraction=0.25, frame_steps=10)
    assert np.array_equal(drive(0), drive(9))
    assert not np.array_equal(drive(0), drive(10))
    assert np.count_nonzero(drive(0)) == 5
    assert np.count_nonzero(drive(0)[20:]) == 0


@pytest.mark.parametrize("dt_ms", [1.0, 2.0, 3.0])
def test_the_drive_frame_is_one_sub_frame_of_a_real_tick(dt_ms: float):
    """Calibrating against an input shape the encoder never produces tunes for a
    network that does not exist, so the default frame is derived from the tick —
    in ms, so it stays one sub-frame of held current at every `dt`."""
    assert (TICK_MS, SUBFRAMES_PER_TICK, FRAME_MS) == (600, 4, 150)
    steps = steps_for(FRAME_MS, dt_ms)
    assert steps * dt_ms == pytest.approx(FRAME_MS, abs=dt_ms)
    drive = sensory_drive(50, np.arange(20), amplitude=20.0, active_fraction=0.25, dt_ms=dt_ms)
    assert np.array_equal(drive(0), drive(steps - 1))
    assert not np.array_equal(drive(0), drive(steps))


def test_the_default_dt_divides_the_tick_and_the_sub_frame():
    """`agent.py` derives substeps as `tick_ms / dt / subframes` and never rounds,
    so a `dt` that does not divide both silently drops biological time."""
    assert TICK_MS % DEFAULT_DT_MS == 0
    assert FRAME_MS % DEFAULT_DT_MS == 0


def test_the_noise_std_holds_the_membrane_noise_fixed_across_dt():
    """The bug that made a coarser step look expensive: the engine draws one sample
    per step, so a fixed std is a louder membrane at every larger `dt`, and the
    connectome-free null rose 1.041 -> 2.192 -> 2.848 Hz and ate the band."""

    def membrane_sigma(dt: float) -> float:
        av = math.exp(-dt / 20.0)
        return noise_std_for_dt(dt) * math.sqrt((1.0 - av) / (1.0 + av))

    assert noise_std_for_dt(1.0) == DEFAULT_SPONTANEOUS_NOISE_STD
    baseline = membrane_sigma(1.0)
    for dt in (0.5, 2.0, 3.0, 5.0):
        assert membrane_sigma(dt) == pytest.approx(baseline, rel=1e-9)
        # Coarser steps need *less* per-step current, not more.
        assert (noise_std_for_dt(dt) < DEFAULT_SPONTANEOUS_NOISE_STD) == (dt > 1.0)


def _feedforward_net(n: int):
    """No recurrence at all, so the gain is irrelevant and the drive sets the rate."""
    return sp.csc_matrix((n, n), dtype=np.float32)


def test_a_mean_in_band_over_a_silent_majority_is_rejected():
    """The predicate's whole point: 70% silent at a 4 Hz mean is not calibrated."""
    rates = np.concatenate([np.zeros(700), np.full(300, 13.0)])
    clauses = Acceptance().reject(summarize_rates(rates, max_hz=333.0))

    assert "mean_hz" not in " ".join(clauses)  # the mean alone would have passed
    assert any(c.startswith("silent_fraction") for c in clauses)
    assert any(c.startswith("median_hz") for c in clauses)


def test_a_network_whose_mean_is_in_band_but_is_mostly_silent_fails_the_search():
    n = 1000
    W = _feedforward_net(n)
    # 30% of cells held just above threshold; the rest see nothing at any gain.
    current = np.zeros(n, dtype=np.float32)
    current[: int(n * 0.3)] = 15.5

    # `null_margin=0` deliberately: this fixture has no connectome, so the null
    # clause would catch it first and the shape clauses — what it exists to test —
    # would never run.
    result = calibrate_gain(
        W,
        drive=current,
        acceptance=Acceptance(null_margin=0.0),
        gain_range=(0.5, 2.0),
        max_iter=4,
        **FAST,
    )

    assert not result.success
    assert result.measurement is None
    assert result.rejected is not None
    assert 1.0 <= result.rejected.rates.mean_hz <= 5.0
    assert result.rejected.rates.silent_fraction > 0.65
    assert "silent_fraction" in result.failure
    assert "median_hz" in result.failure
    assert "FAILED" in format_report(result)


@pytest.mark.skipif(
    not os.environ.get("FLYBRAIN_SLOW"),
    reason="calibrating the real 44k connectome is a manual path; set FLYBRAIN_SLOW=1",
)
def test_real_connectome_calibration_runs():
    from flybrain.connectome.loader import load
    from flybrain.sensory.encode import LUMINANCE_TYPES, EncodeParams

    c = load()
    result = calibrate_gain(
        c.W,
        populations=c.populations,
        injection_types=LUMINANCE_TYPES,
        drive_amplitude=EncodeParams().i_max,
        max_iter=10,
    )
    print(format_report(result))
    assert result.measurement or result.lower_bracket


def test_threshold_trim_moves_an_over_firing_population():
    n = 500
    W = _graded_net(n)
    drive = _drive(n, background=12.0, sensory=50, active_fraction=0.1)
    populations = {"sensory": np.arange(50), "rest": np.arange(50, n)}

    result = calibrate_gain(
        W,
        drive=drive,
        populations=populations,
        gain_range=(1.0, 30.0),
        max_iter=12,
        trim_thresholds=True,
        trim_rounds=2,
        **HONEST,
    )

    assert result.success, result.failure
    offsets = result.threshold_offsets
    assert offsets.shape == (n,)
    # The driven population is the one firing hardest, so its threshold rises.
    assert offsets[:50].mean() > offsets[50:].mean()


def _asymmetric_net(n: int = 200, seed: int = 3):
    """Wildly uneven in-degree: a few hubs, most cells with a handful of inputs."""
    rng = np.random.default_rng(seed)
    M = np.zeros((n, n), dtype=np.float32)
    for post in range(n):
        k = 3 if post % 4 else 120
        pre = rng.choice(n, k, replace=False)
        M[post, pre] = rng.integers(1, 40, k)
    np.fill_diagonal(M, 0.0)
    sign = np.where(rng.random(n) < 0.3, -1.0, 1.0).astype(np.float32)
    return sp.csc_matrix(M * sign[None, :])


def test_normalization_bounds_every_row_and_leaves_the_e_i_ratio_alone():
    """Row, not column: `W` is `W[post, pre]`, so incoming is `W.indices`."""
    W = _asymmetric_net()
    rows = np.abs(W).sum(axis=1).A1
    assert rows.max() / rows[rows > 0].min() > 50  # the imbalance being removed

    full = normalize_incoming(W, "full")
    np.testing.assert_allclose(np.abs(full).sum(axis=1).A1, 1.0, atol=1e-6)

    dense, scaled = W.toarray(), full.toarray()
    pos, neg = dense.clip(min=0).sum(axis=1), -dense.clip(max=0).sum(axis=1)
    spos, sneg = scaled.clip(min=0).sum(axis=1), -scaled.clip(max=0).sum(axis=1)
    np.testing.assert_allclose(spos, pos / rows, atol=1e-6)
    np.testing.assert_allclose(sneg, neg / rows, atol=1e-6)
    np.testing.assert_array_equal(np.sign(scaled), np.sign(dense))


def test_capped_normalization_only_touches_the_over_innervated():
    W = _asymmetric_net()
    rows = np.abs(W).sum(axis=1).A1
    cap = float(np.median(rows))
    capped = np.abs(normalize_incoming(W, "capped", cap=cap)).sum(axis=1).A1

    under = rows <= cap
    np.testing.assert_allclose(capped[under], rows[under], rtol=1e-5)
    np.testing.assert_allclose(capped[~under], cap, rtol=1e-5)
    assert normalize_incoming(W, "none") is W
    with pytest.raises(ValueError, match="not one of"):
        normalize_incoming(W, "by-column")


def test_normalizing_the_columns_instead_would_have_looked_fine():
    """The axis that fails silently: out-degree normalisation also 'sums to 1'."""
    W = _asymmetric_net()
    out_degree = np.abs(W).sum(axis=0).A1
    assert not np.allclose(out_degree, np.abs(W).sum(axis=1).A1)
    assert not np.allclose(np.abs(normalize_incoming(W, "full")).sum(axis=0).A1, 1.0)


def test_tonic_parks_the_membrane_just_below_threshold():
    """The fixed point of the exact-exponential update under a constant drive."""
    v_rest, v_thresh = DEFAULT_V_REST, DEFAULT_V_THRESH
    assert tonic_current(0.9) == pytest.approx(0.9 * (v_thresh - v_rest))

    engine = LIFEngine(sp.csc_matrix((4, 4), dtype=np.float32), spontaneous_noise_std=0.0)
    current = np.full(4, np.float32(tonic_current(DEFAULT_TONIC_FRACTION)), dtype=np.float32)
    for _ in range(400):
        assert engine.step(current).size == 0
    np.testing.assert_allclose(engine.v, v_rest + tonic_current(DEFAULT_TONIC_FRACTION), atol=1e-3)
    # A hair more and the same silent neuron fires: the floor is the whole point.
    hot = LIFEngine(sp.csc_matrix((4, 4), dtype=np.float32), spontaneous_noise_std=0.0)
    over = np.full(4, np.float32(tonic_current(1.01)), dtype=np.float32)
    assert any(hot.step(over).size for _ in range(400))


def test_with_tonic_lifts_every_neuron_including_the_undriven():
    base = sensory_drive(10, np.arange(3), amplitude=5.0, active_fraction=1.0)
    lifted = with_tonic(base, 10, 2.0)
    np.testing.assert_allclose(lifted(0), base(0) + 2.0)
    assert with_tonic(base, 10, 0.0) is base


def test_a_rate_in_band_that_the_null_also_reaches_is_rejected():
    """The clause the tonic made necessary: in-band is not evidence of a network."""
    rng = np.random.default_rng(0)
    r = summarize_rates(10 ** rng.normal(0.3, 0.2, 2000), max_hz=333.0)

    assert Acceptance().reject(r, None) == []  # every shape clause passes
    clauses = Acceptance().reject(r, r.mean_hz / 1.25)
    assert len(clauses) == 1
    assert clauses[0].startswith(NULL_CLAUSE)
    assert "1.25x" in clauses[0] and "not contributing" in clauses[0]
    # The same distribution over a null it clears by the margin is accepted.
    assert Acceptance().reject(r, r.mean_hz / 1.5) == []


def test_a_network_with_no_connectome_is_rejected_however_good_its_rate_looks():
    """FlyBrain's own failure mode: a noise generator with a graph attached.

    A zero weight matrix under a drive 1 mV short of threshold reads 2.2 Hz mean,
    2.25 Hz median, 0% silent, unimodal — it passes every other clause in
    `Acceptance`, at every gain, because the gain multiplies nothing.
    """
    n = 800
    W = _feedforward_net(n)
    current = np.full(n, np.float32(14.0), dtype=np.float32)

    result = calibrate_gain(W, drive=current, gain_range=(0.01, 10.0), max_iter=4, **HONEST)

    assert not result.success
    assert result.measurement is None
    # Every shape clause passes; it is only the null that rejects this.
    assert 1.0 <= result.upper_bracket.rates.mean_hz <= 5.0
    assert result.upper_bracket.rates.median_hz > 0.0
    assert result.upper_bracket.rates.silent_fraction == 0.0
    assert Acceptance().reject(result.upper_bracket.rates) == []
    assert NULL_CLAUSE in result.failure
    # The null is the same network, so no gain can pull away from it.
    assert result.null.rates.mean_hz == pytest.approx(result.upper_bracket.rates.mean_hz)
    assert "connectome-free" in format_report(result)


def test_the_null_is_measured_once_per_search_not_once_per_bisection():
    n = 500
    W = _graded_net(n)
    drive = _drive(n, background=12.0, sensory=50, active_fraction=0.1)
    runs: list[float] = []
    real = calibrate_module.measure_gain

    def counting(W_, gain, drive_, **kw):
        runs.append(float(W_.nnz))
        return real(W_, gain, drive_, **kw)

    with mock.patch.object(calibrate_module, "measure_gain", counting):
        result = calibrate_gain(W, drive=drive, gain_range=(1.0, 30.0), max_iter=12, **HONEST)

    assert result.success, result.failure
    assert runs.count(0.0) == 1  # exactly one zero-matrix run, whatever the iteration count
    assert result.null is not None
    assert result.measurement.rates.mean_hz >= result.null.rates.mean_hz * Acceptance().null_margin


@pytest.mark.skipif(
    not (loader.DEFAULT_PATH.exists() and vocab.DEFAULT_ANNOTATIONS_PATH.exists()),
    reason="connectome_v1.npz or the annotations feather is missing; run the build first",
)
def test_the_encoder_drives_sixty_times_what_the_synthetic_pattern_does():
    """The last live/calibration divergence, and why `encoder_drive` exists.

    Only the drive's *amplitude* was ever matched to the encoder; the *pattern*
    was not. `sensory_drive` lights 1% of the injection layer, `encode()` lights
    62-72% of it at close to the Naka-Rushton ceiling, so calibrating on the
    synthetic pattern put the bench at 2.20 Hz while the live loop ran 4.5 Hz —
    and since cost is edges touched, twice the rate is twice the tick, which is
    the whole of the missed 360 ms deadline.
    """
    from flybrain.connectome.loader import load
    from flybrain.sensory.encode import LUMINANCE_TYPES, EncodeParams

    c = load()
    params = EncodeParams()
    injection = np.unique(
        np.concatenate([c.populations[t] for t in LUMINANCE_TYPES if t in c.populations])
    )
    live = encoder_drive(c, params=params)(0)
    synthetic = sensory_drive(c.n, injection, amplitude=params.i_max)(0)

    def lit(current):
        return float((current[injection] > 0.0).mean())

    print(
        f"\ninjection layer {injection.size} cells | "
        f"synthetic {lit(synthetic):.1%} lit, sum {synthetic.sum():.0f} | "
        f"encoder {lit(live):.1%} lit, sum {live.sum():.0f}"
    )
    assert lit(synthetic) == pytest.approx(0.01, abs=0.002)
    assert lit(live) > 0.5
    assert live.sum() > 20.0 * synthetic.sum()
    # Sensory indices only, either way: the drive must not leak past the layer.
    assert np.count_nonzero(live) == np.count_nonzero(live[injection])


@pytest.mark.skipif(
    not (loader.DEFAULT_PATH.exists() and vocab.DEFAULT_ANNOTATIONS_PATH.exists()),
    reason="connectome_v1.npz or the annotations feather is missing; run the build first",
)
def test_the_encoder_drive_is_a_pure_function_of_the_step():
    """Successive gains in the search must see an identical input sequence.

    A drive that advanced a scene on each call would hand every bisection
    iteration a different input, and their rates would not be comparable — which
    is how a metastable network gets mistaken for a calibrated one.
    """
    from flybrain.connectome.loader import load

    c = load()
    drive = encoder_drive(c, dt_ms=2.0)
    steps = steps_for(FRAME_MS, 2.0)
    first = drive(0).copy()
    assert np.array_equal(drive(steps - 1), first)
    later = drive(steps * 9).copy()
    assert not np.array_equal(later, first)
    # Held for a whole sub-frame, and reproduced after the cache has rolled past.
    assert np.array_equal(drive(0), first)
    assert np.array_equal(drive(steps * 9), later)
