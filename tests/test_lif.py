from __future__ import annotations

import time
from itertools import pairwise

import numpy as np
import pytest
import scipy.sparse as sp

from flybrain.engine.lif import LIFEngine


def _empty(n: int) -> sp.csc_matrix:
    return sp.csc_matrix((n, n), dtype=np.float32)


def _engine(n: int = 2, W: sp.csc_matrix | None = None, **kw) -> LIFEngine:
    return LIFEngine(W if W is not None else _empty(n), **kw)


def _run(engine: LIFEngine, current: np.ndarray, steps: int) -> list[np.ndarray]:
    return [engine.step(current) for _ in range(steps)]


def _spike_steps(fired: list[np.ndarray], neuron: int) -> list[int]:
    return [k for k, f in enumerate(fired) if neuron in f]


@pytest.mark.parametrize("dt_ms", [1.0, 2.0, 3.0])
def test_suprathreshold_current_fires_and_rate_rises_with_current(dt_ms: float):
    """f-I is a claim about the neuron, so it has to survive `dt` moving.

    The window is 400 ms of simulated time, not 400 steps: the adapted asymptote
    is set by `b * tau_w`, which `dt` does not touch, but a window pinned in steps
    would measure a different stretch of the adaptation transient at every `dt`.
    """
    rates = []
    for amplitude in (16.0, 20.0, 40.0):
        eng = _engine(1, dt_ms=dt_ms)
        cur = np.full(1, amplitude, dtype=np.float32)
        _run(eng, cur, round(400.0 / dt_ms))
        rate = float(eng.get_firing_rates()[0])
        assert rate > 0.0, f"no spikes at I={amplitude}, dt={dt_ms}"
        rates.append(rate)
    assert rates[0] < rates[1] < rates[2], (dt_ms, rates)


def test_refractory_period_is_respected():
    eng = _engine(1)
    fired = _run(eng, np.full(1, 60.0, dtype=np.float32), 300)
    steps = _spike_steps(fired, 0)
    assert len(steps) > 5
    gaps = np.diff(steps)
    assert gaps.min() >= eng.refractory_steps, (gaps.min(), eng.refractory_steps)


def test_excitation_advances_and_inhibition_delays_postsynaptic_spike():
    def first_post_spike(weight: float) -> int | None:
        W = sp.csc_matrix(np.array([[0.0, 0.0], [weight, 0.0]], dtype=np.float32))
        eng = _engine(W=W)
        cur = np.array([40.0, 16.0], dtype=np.float32)
        steps = _spike_steps(_run(eng, cur, 400), 1)
        return steps[0] if steps else None

    baseline = first_post_spike(0.0)
    excited = first_post_spike(10.0)
    inhibited = first_post_spike(-10.0)

    assert baseline is not None and excited is not None
    assert excited < baseline
    assert inhibited is None or inhibited > baseline


def test_conduction_delay_is_exact():
    W = sp.csc_matrix(np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32))
    eng = _engine(W=W)
    cur = np.array([40.0, 0.0], dtype=np.float32)

    pre_spike = None
    post_g_change = None
    for k in range(200):
        fired = eng.step(cur)
        if pre_spike is None and 0 in fired:
            pre_spike = k
        if post_g_change is None and eng.g[1] != 0.0:
            post_g_change = k
            break

    assert pre_spike is not None and post_g_change is not None
    assert post_g_change - pre_spike == eng.delay_slots


def test_zero_input_is_silent_and_at_rest():
    eng = _engine(64, spontaneous_noise_std=0.0)
    cur = np.zeros(64, dtype=np.float32)
    for _ in range(200):
        assert eng.step(cur).size == 0
    assert np.allclose(eng.v, eng.v_rest)
    assert np.allclose(eng.get_firing_rates(), 0.0)


def test_step_time_at_scale():
    n = 20_000
    density = 2_000_000 / (n * n)
    rng = np.random.default_rng(0)
    W = sp.random(
        n,
        n,
        density=density,
        format="csc",
        dtype=np.float32,
        random_state=rng,
        data_rvs=lambda k: (rng.random(k, dtype=np.float32) - 0.5) * 0.02,
    )

    cur = np.zeros(n, dtype=np.float32)
    driven = rng.choice(n, size=int(0.03 * n), replace=False)
    cur[driven] = 450.0

    eng = LIFEngine(W)
    for _ in range(50):
        eng.step(cur)

    trials = 400
    counts = 0
    t0 = time.perf_counter()
    for _ in range(trials):
        counts += eng.step(cur).size
    mean_ms = (time.perf_counter() - t0) / trials * 1000.0
    frac = counts / trials / n

    print(f"\nN={n} nnz={W.nnz} firing={frac:.3%} mean step {mean_ms:.3f} ms")
    assert 0.005 < frac < 0.02, frac
    assert mean_ms < 0.5, mean_ms


def test_widening_the_rate_window_keeps_the_history_it_already_had():
    eng = _engine(1, rate_window_ms=50.0)
    _run(eng, np.full(1, 40.0, dtype=np.float32), 50)
    before = float(eng.get_firing_rates()[0])

    eng.set_rate_window_ms(600.0)
    assert eng.rate_window_ms == 600.0
    # Still 50 steps of history, so the rate is unchanged until more arrives.
    assert float(eng.get_firing_rates()[0]) == before

    _run(eng, np.zeros(1, dtype=np.float32), 550)
    assert len(eng._hist) == 600


def test_a_short_rate_window_evicts_an_early_spike():
    """Why `Agent` widens the window: the reason the escape reflex went deaf."""
    eng = _engine(1, rate_window_ms=500.0)
    pulse = np.full(1, 500.0, dtype=np.float32)
    zero = np.zeros(1, dtype=np.float32)
    for k in range(600):
        eng.step(pulse if k == 50 else zero)
    assert float(eng.get_firing_rates()[0]) == 0.0


def test_adaptation_lowers_the_rate_over_a_constant_current():
    eng = _engine(1)
    fired = _run(eng, np.full(1, 20.0, dtype=np.float32), 2000)
    steps = _spike_steps(fired, 0)
    q = len(fired) // 4
    first = sum(1 for k in steps if k < q)
    last = sum(1 for k in steps if k >= 3 * q)
    print(f"\nadaptation at I=20: first quarter {first} spikes, last quarter {last}")
    assert last < first, (first, last)


def test_zero_adaptation_leaves_the_unadapted_train_untouched():
    """`b=0.0` must be a true off switch, so the pre-adaptation numbers stay reachable."""
    cur = np.full(1, 20.0, dtype=np.float32)
    off = _engine(1, b=0.0)
    steps_off = _spike_steps(_run(off, cur, 2000), 0)

    assert np.all(off.w == 0.0)
    # Unadapted, a constant current gives a perfectly periodic train.
    assert len(set(np.diff(steps_off))) == 1, np.diff(steps_off)

    on = _engine(1)
    steps_on = _spike_steps(_run(on, cur, 2000), 0)
    assert len(steps_on) < len(steps_off), (len(steps_on), len(steps_off))


def _floor_rates(n: int, steps: int, **kw) -> np.ndarray:
    """Rates of an unconnected net held under a tonic — the membrane noise alone."""
    eng = _engine(n, rate_window_ms=steps, spontaneous_noise_std=4.0, seed=0, **kw)
    _run(eng, np.full(n, 13.5, dtype=np.float32), steps)
    return eng.get_firing_rates()


def test_the_noise_pool_leaves_the_floor_distribution_where_a_fresh_draw_puts_it():
    """The pool is a speed change, so it has to be a no-op on the statistics.

    `standard_normal` over every neuron was 0.436 ms of a 1.16 ms step at
    N=184,110 — 131 ms of a 347 ms tick — against 0.016 ms for a rolling window
    out of a pre-drawn pool. That trade is only free if the rate distribution
    does not move, and the rate distribution under a tonic with no connectome at
    all *is* the membrane noise: it is the connectome-free null every calibration
    is accepted against. Measured on the full build, 4,000 steps at dt 2:

        fresh draw   null 0.934 Hz, mean 2.197, median 1.625, silent 0.11%
        pool         null 0.942 Hz, mean 2.214, median 1.625, silent 0.11%

    so this pins the same equality at a size a test can afford.
    """
    fresh = _floor_rates(2_000, 4_000, noise_pool=0)
    pooled = _floor_rates(2_000, 4_000)
    print(
        f"\nfloor: fresh mean {fresh.mean():.3f} Hz median {np.median(fresh):.3f} "
        f"silent {(fresh == 0).mean():.2%} | "
        f"pool mean {pooled.mean():.3f} Hz median {np.median(pooled):.3f} "
        f"silent {(pooled == 0).mean():.2%}"
    )
    assert fresh.mean() > 0.0
    assert abs(pooled.mean() - fresh.mean()) < 0.05 * fresh.mean()
    assert np.median(pooled) == np.median(fresh)
    assert abs((pooled == 0).mean() - (fresh == 0).mean()) < 0.01


def test_the_noise_pool_hands_every_step_a_different_window():
    """Every step reads a window no earlier step read.

    The saving would be an illusion if the offset stalled or cycled short: the
    membrane would be integrating one held sample, which is the thing this
    deliberately is not. The windows do overlap once `N` exceeds the stride, but
    a *shifted* window still gives each neuron an entry it has not read, which is
    the property the noise process needs.
    """
    eng = _engine(64, spontaneous_noise_std=4.0, seed=0)
    assert eng._noise_pool is not None
    offsets = []
    for _ in range(500):
        offsets.append(eng._noise_offset)
        eng.step(np.zeros(64, dtype=np.float32))
    assert len(set(offsets)) == len(offsets)
    assert all(a != b for a, b in pairwise(offsets))

    off = _engine(64, spontaneous_noise_std=0.0)
    assert off._noise_pool is None


# --------------------------------------------------------------- silencing

#: The steady state of the membrane update is `v_rest + drive`, so this is 0.9
#: of the 15 mV to threshold — the floor the calibration puts under every
#: neuron, and the reason cutting a population's synapses does not silence it.
TONIC = 13.5


def test_a_silenced_neuron_never_fires_under_tonic_drive_and_noise():
    eng = _engine(3, silenced=np.array([1]), spontaneous_noise_std=60.0, seed=5)
    fired = _run(eng, np.full(3, TONIC, dtype=np.float32), 2000)
    counts = np.bincount(np.concatenate(fired), minlength=3)
    assert counts[1] == 0
    assert counts[0] > 0 and counts[2] > 0
    assert eng.get_firing_rates()[1] == 0.0
    assert eng.v[1] == pytest.approx(eng.v_rest)


def test_a_silenced_neuron_deposits_nothing_downstream():
    W = sp.csc_matrix(np.array([[0.0, 0.0], [40.0, 0.0]], dtype=np.float32))
    eng = _engine(W=W, silenced=np.array([0]))
    fired = _run(eng, np.array([60.0, 0.0], dtype=np.float32), 500)
    assert not np.concatenate(fired).size
    assert eng.v[1] == pytest.approx(eng.v_rest)


def test_silencing_nothing_leaves_the_spike_train_untouched():
    current = np.full(2, 20.0, dtype=np.float32)
    plain = _run(_engine(2, seed=3), current, 300)
    empty = _run(_engine(2, silenced=np.empty(0, dtype=np.int64), seed=3), current, 300)
    assert [f.tolist() for f in plain] == [f.tolist() for f in empty]


def test_silencing_after_the_fact_stops_a_firing_neuron():
    eng = _engine(2, spontaneous_noise_std=60.0, seed=5)
    current = np.full(2, 20.0, dtype=np.float32)
    assert np.concatenate(_run(eng, current, 200)).size
    eng.set_silenced(np.array([0]))
    after = np.concatenate(_run(eng, current, 500))
    assert 0 not in after
    assert 1 in after
