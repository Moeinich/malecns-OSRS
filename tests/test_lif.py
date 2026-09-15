from __future__ import annotations

import time

import numpy as np
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


def test_suprathreshold_current_fires_and_rate_rises_with_current():
    rates = []
    for amplitude in (16.0, 20.0, 40.0):
        eng = _engine(1)
        cur = np.full(1, amplitude, dtype=np.float32)
        _run(eng, cur, 400)
        rate = float(eng.get_firing_rates()[0])
        assert rate > 0.0, f"no spikes at I={amplitude}"
        rates.append(rate)
    assert rates[0] < rates[1] < rates[2], rates


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
