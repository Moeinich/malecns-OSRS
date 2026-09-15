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
    assert decode(rates, motor).attack


def test_below_threshold_is_no_act():
    cmd = decode(_rates(attack=PARAMS.discrete_hz - 0.1), _index())
    assert isinstance(cmd, EgocentricCommand)
    assert not (cmd.attack or cmd.eat or cmd.pickup)
