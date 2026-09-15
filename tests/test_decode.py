from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np

from flybrain.motor.decode import EgocentricCommand, MotorIndex, MotorParams, decode

N = 40

PARAMS = MotorParams()


SLOTS = {
    "steer_left": [0, 1],
    "steer_right": [2, 3],
    "drive": [4, 5],
    "reverse": [6, 7],
    "escape": [8, 9],
    "attack": [10, 11],
    "eat": [12, 13],
    "pickup": [14, 15],
}


def _index(**overrides) -> MotorIndex:
    kwargs = {k: np.array(v, dtype=np.int64) for k, v in SLOTS.items()}
    kwargs.update(overrides)
    return MotorIndex(params=PARAMS, **kwargs)


def _rates(**pools) -> np.ndarray:
    rates = np.zeros(N, dtype=np.float32)
    for name, value in pools.items():
        rates[SLOTS[name]] = value
    return rates


# ------------------------------------------------------------------ firewall


def test_decode_never_sees_game_state():
    """No import of any state or action type, and no path for one to arrive."""
    path = Path(__file__).resolve().parent.parent / "flybrain" / "motor" / "decode.py"
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("flybrain"), node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("flybrain"), alias.name


def test_decode_signature_is_exactly_rates_and_motor():
    assert list(inspect.signature(decode).parameters) == ["rates", "motor"]
    assert inspect.signature(decode).return_annotation == "EgocentricCommand"


# ------------------------------------------------------------------ steering


def test_dna02_imbalance_sets_the_turn_direction():
    left = decode(_rates(steer_left=30.0, steer_right=10.0), _index())
    right = decode(_rates(steer_left=10.0, steer_right=30.0), _index())
    assert left.turn > 0.0
    assert right.turn < 0.0
    assert left.turn == -right.turn


def test_zero_differential_gives_zero_turn():
    assert decode(_rates(steer_left=25.0, steer_right=25.0), _index()).turn == 0.0
    assert decode(_rates(), _index()).turn == 0.0


def test_drive_and_reverse_are_pooled_thresholds():
    cmd = decode(_rates(drive=PARAMS.drive_max_hz, reverse=PARAMS.reverse_hz), _index())
    assert cmd.drive == 1.0
    assert cmd.reverse
    assert decode(_rates(drive=1e9), _index()).drive == 1.0
    assert not decode(_rates(reverse=PARAMS.reverse_hz - 0.1), _index()).reverse


# -------------------------------------------------------------------- escape


def _fired(*neurons: int) -> np.ndarray:
    return np.array(neurons, dtype=np.int32)


def test_a_single_giant_fiber_spike_triggers_escape():
    motor = _index()
    motor.begin_tick()
    motor.observe_spikes(_fired(SLOTS["escape"][0]), 0)
    assert decode(_rates(), motor).escape
    assert motor._reflex.escape_substep == 0


def test_a_high_rate_without_a_spike_does_not_trigger_escape():
    motor = _index()
    motor.begin_tick()
    motor.observe_spikes(_fired(*SLOTS["steer_left"]), 0)
    assert not decode(_rates(escape=1e3), motor).escape


def test_begin_tick_clears_the_previous_ticks_spikes():
    motor = _index()
    motor.begin_tick()
    motor.observe_spikes(_fired(SLOTS["escape"][1]), 7)
    assert decode(_rates(), motor).escape
    motor.begin_tick()
    assert not decode(_rates(), motor).escape


# ------------------------------------------------------------------ discrete


def test_discrete_acts_are_winner_take_all():
    hot = PARAMS.discrete_hz
    cmd = decode(_rates(attack=hot + 5.0, eat=hot + 1.0, pickup=hot + 3.0), _index())
    assert (cmd.attack, cmd.eat, cmd.pickup) == (True, False, False)


def test_refractory_suppresses_a_repeat_inside_the_window():
    motor = _index()
    rates = _rates(attack=PARAMS.discrete_hz + 5.0)
    assert decode(rates, motor).attack
    assert not decode(rates, motor).attack
    # Past the window, but the same rate is now this pool's own baseline, so it
    # takes a fresh burst above it rather than merely holding the old rate.
    assert decode(_rates(attack=4 * PARAMS.discrete_hz), motor).attack


def test_below_threshold_is_no_act():
    cmd = decode(_rates(attack=PARAMS.discrete_hz - 0.1), _index())
    assert isinstance(cmd, EgocentricCommand)
    assert not (cmd.attack or cmd.eat or cmd.pickup)


# ------------------------------------------------- steering is a population


def _pool_rates(n_per_side: int, left_hz: float, right_hz: float) -> tuple[np.ndarray, MotorIndex]:
    """Two steering pools whose *first* cell votes against its own side.

    A one-cell-per-side readout would read the inverted pair and turn the wrong
    way; only pooling recovers the true imbalance. This is the regression lock
    on the finding that a 2-cell differential is Poisson noise.
    """
    left = np.arange(n_per_side, dtype=np.int64)
    right = left + n_per_side
    rates = np.zeros(2 * n_per_side, dtype=np.float32)
    rates[left], rates[right] = left_hz, right_hz
    rates[left[0]], rates[right[0]] = right_hz, left_hz
    empty = np.array([], dtype=np.int64)
    motor = MotorIndex(
        steer_left=left,
        steer_right=right,
        drive=empty,
        reverse=empty,
        escape=empty,
        attack=empty,
        eat=empty,
        pickup=empty,
        params=PARAMS,
    )
    return rates, motor


def test_a_pooled_differential_resolves_left_from_right_with_a_margin():
    margin = 0.2  # a 1-neuron readout of the inverted pair turns the wrong way entirely
    rates, motor = _pool_rates(26, 6.0, 4.0)
    assert decode(rates, motor).turn > margin
    rates, motor = _pool_rates(26, 4.0, 6.0)
    assert decode(rates, motor).turn < -margin


def test_one_cell_per_side_is_what_the_pool_has_to_beat():
    """The same rates read one cell per side: wrong sign, which is the bug."""
    rates, _ = _pool_rates(26, 6.0, 4.0)
    one = _pool_rates(26, 6.0, 4.0)[1]
    single = MotorIndex(
        **{
            **{k: getattr(one, k) for k in SLOTS},
            "steer_left": one.steer_left[:1],
            "steer_right": one.steer_right[:1],
        },
        params=PARAMS,
    )
    assert decode(rates, single).turn < 0.0


class _FakePopulations:
    def __init__(self, pops: dict[str, np.ndarray]):
        self._pops = pops

    def population(self, name: str, side: str | None = None) -> np.ndarray:
        key = f"{name}|{side[0].upper()}" if side else name
        if key not in self._pops:
            raise KeyError(key)
        return self._pops[key]


def test_from_connectome_pools_whole_descending_families():
    def block(start: int, n: int) -> np.ndarray:
        return np.arange(start, start + n, dtype=np.int64)

    pops = {
        "DNa|L": block(0, 26),
        "DNa|R": block(26, 26),
        "DNa02|L": block(0, 1),
        "DNa02|R": block(26, 1),
        "DNb": block(52, 40),
        "DNp09": block(92, 2),
        "DNpe017": block(94, 2),
        "MDN": block(96, 4),
        "DNp01": block(100, 2),
        "MBON": block(102, 97),
    }
    motor = MotorIndex.from_connectome(_FakePopulations(pops))
    assert len(motor.steer_left) == 26
    assert len(motor.steer_right) == 26
    assert set(pops["DNa02|L"]) <= set(motor.steer_left)
    assert set(pops["DNa02|R"]) <= set(motor.steer_right)
    assert len(motor.drive) == 44
    # The Giant Fiber is genuinely one cell per side and must not be pooled.
    assert len(motor.escape) == 2


def test_a_missing_family_falls_back_to_the_canonical_cells():
    pops = {
        "DNa02|L": np.array([0], dtype=np.int64),
        "DNa02|R": np.array([1], dtype=np.int64),
        "DNp09": np.array([2, 3], dtype=np.int64),
        "DNpe017": np.array([4, 5], dtype=np.int64),
        "MDN": np.array([6], dtype=np.int64),
        "DNp01": np.array([7, 8], dtype=np.int64),
        "MBON": np.arange(9, 18, dtype=np.int64),
    }
    motor = MotorIndex.from_connectome(_FakePopulations(pops))
    assert motor.steer_left.tolist() == [0]
    assert motor.drive.tolist() == [2, 3, 4, 5]


# ------------------------------------------------------- baseline-relative


def _burst_ticks(motor: MotorIndex, spikes: int, ticks: int) -> list[bool]:
    out = []
    for _ in range(ticks):
        motor.begin_tick()
        for k in range(spikes):
            motor.observe_spikes(_fired(SLOTS["escape"][k % 2]), k)
        out.append(decode(_rates(), motor).escape)
    return out


def test_escape_stops_firing_once_the_giant_fiber_baseline_settles():
    """DNp01 at the tonic floor spikes every tick; that is not a looming stimulus."""
    fired = _burst_ticks(_index(), spikes=4, ticks=12)
    assert fired[0]
    assert not any(fired[3:])


def test_a_burst_above_the_settled_baseline_still_triggers_escape():
    motor = _index()
    assert not any(_burst_ticks(motor, spikes=4, ticks=12)[3:])
    assert _burst_ticks(motor, spikes=40, ticks=1) == [True]


def test_a_discrete_act_at_its_own_baseline_does_not_trip():
    motor = _index()
    rates = _rates(attack=PARAMS.discrete_hz + 5.0)
    fired = [decode(rates, motor).attack for _ in range(30)]
    assert fired[0]
    assert not any(fired[10:])
