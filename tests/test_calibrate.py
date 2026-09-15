from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.sparse as sp

from flybrain.engine.calibrate import (
    BIMODALITY_UNIFORM,
    SATURATION_FRACTION,
    calibrate_gain,
    format_report,
    measure_gain,
    sensory_drive,
    summarize_rates,
)

FAST = {"warmup_steps": 200, "measure_steps": 400}


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

    result = calibrate_gain(W, drive=drive, gain_range=(1.0, 30.0), max_iter=12, **FAST)

    assert result.success, result.failure
    assert result.failure is None
    assert 1.0 <= result.measurement.rates.mean_hz <= 5.0
    assert 1.0 < result.gain < 30.0
    assert "CALIBRATED" in format_report(result)


def test_bistable_network_is_reported_as_failed_not_as_a_near_miss():
    n = 200
    W = _bistable_net(n)
    drive = _drive(n, background=14.0, sensory=10, active_fraction=0.1)

    result = calibrate_gain(W, drive=drive, gain_range=(1e-5, 10.0), max_iter=12, **FAST)

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
    drive = sensory_drive(50, np.arange(20), active_fraction=0.25, frame_steps=10)
    assert np.array_equal(drive(0), drive(9))
    assert not np.array_equal(drive(0), drive(10))
    assert np.count_nonzero(drive(0)) == 5
    assert np.count_nonzero(drive(0)[20:]) == 0


@pytest.mark.skipif(
    not os.environ.get("FLYBRAIN_SLOW"),
    reason="calibrating the real 44k connectome is a manual path; set FLYBRAIN_SLOW=1",
)
def test_real_connectome_calibration_runs():
    from flybrain.connectome.loader import load

    c = load()
    result = calibrate_gain(c.W, populations=c.populations, max_iter=10)
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
        **FAST,
    )

    assert result.success, result.failure
    offsets = result.threshold_offsets
    assert offsets.shape == (n,)
    # The driven population is the one firing hardest, so its threshold rises.
    assert offsets[:50].mean() > offsets[50:].mean()
