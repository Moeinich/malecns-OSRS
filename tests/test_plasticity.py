"""KC->MBON plasticity, and the one test FlyBrain lacked.

`test_stdp_mutates_the_simulated_weights` is why this file exists. Its first
assertion says the numbers moved; its second says the *response* moved, which is
the only thing that proves the mutated weights are on the simulated path rather
than in a second copy nobody runs.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from flybrain.connectome.loader import DEFAULT_PATH, Connectome, load
from flybrain.engine.calibration import Calibration
from flybrain.engine.lif import LIFEngine
from flybrain.engine.plasticity import (
    Plasticity,
    PlasticityParams,
    kc_to_mbon,
    plastic_indices,
    probe_response,
)
from flybrain.loop.agent import Agent, AgentParams
from flybrain.loop.types import StateUpdate, parse_server_message
from flybrain.motor.decode import MotorIndex
from flybrain.reward import DopamineIndex, RewardRouter
from flybrain.sensory.collision import CollisionGrid

# --------------------------------------------------- a mushroom body in miniature
#
# 20 Kenyon cells driven by a sensory slab, projecting onto 6 MBONs, which
# project onto a readout population. The readout is what makes the behavioural
# assertion possible: if KC->MBON weights change and nothing downstream of MBON
# changes, the weights were never simulated.

SENSORY = np.arange(0, 12)
KC = np.arange(12, 32)
MBON = np.arange(32, 38)
READOUT = np.arange(38, 44)
PAM = np.arange(44, 50)
PPL1 = np.arange(50, 54)
#: The descending pools `MotorIndex` insists on. Unwired here — this file tests
#: the plastic path, and `tests/test_agent.py` owns the motor readout.
MOTOR = {
    "DNa02|L": np.arange(54, 56),
    "DNa02|R": np.arange(56, 58),
    "DNp09": np.arange(58, 60),
    "DNpe017": np.arange(60, 62),
    "MDN": np.arange(62, 64),
    "DNp01": np.arange(64, 66),
}
N = 66

#: Long enough that one spike is not a tenth of the measured rate.
PROBE_STEPS = 400


def _populations() -> dict[str, np.ndarray]:
    return {
        "KC": KC.astype(np.int64),
        "MBON": MBON.astype(np.int64),
        "PAM": PAM.astype(np.int64),
        "PPL1": PPL1.astype(np.int64),
        "DNa02": np.concatenate([MOTOR["DNa02|L"], MOTOR["DNa02|R"]]).astype(np.int64),
        **{k: v.astype(np.int64) for k, v in MOTOR.items()},
    }


def _weights() -> sp.csc_matrix:
    """`W[post, pre]`, so `W[i, j]` is the synapse from `j` onto `i`."""
    W = np.zeros((N, N), dtype=np.float32)
    W[np.ix_(KC, SENSORY)] = 8.0
    # Tuned so MBON sits mid-range rather than saturated: a saturated readout
    # would be insensitive to the very weight change this file has to detect.
    W[np.ix_(MBON, KC)] = 4.0
    W[np.ix_(READOUT, MBON)] = 16.0
    return sp.csc_matrix(W)


def _connectome() -> Connectome:
    return Connectome(
        W=_weights(),
        body_ids=np.arange(N, dtype=np.int64),
        populations=_populations(),
        provenance={"synthetic": True},
    )


def _drive(amplitude: float = 26.0) -> np.ndarray:
    current = np.zeros(N, dtype=np.float32)
    current[SENSORY] = amplitude
    return current


def _engine(W: sp.csc_matrix, plastic_idx: np.ndarray | None = None) -> LIFEngine:
    return LIFEngine(W, plastic_idx=plastic_idx, rate_window_ms=400.0, b=0.5, seed=4)


def _learning_setup(eta: float = 5e-3):
    c = _connectome()
    W = c.W
    idx = kc_to_mbon(W, c)
    engine = _engine(W, idx)
    plasticity = Plasticity.attach(engine, c, params=PlasticityParams(eta=eta))
    reward = RewardRouter(DopamineIndex.from_connectome(c), N)
    return engine, plasticity, reward, idx


def _step_n(engine, plasticity, reward, n: int) -> None:
    current = _drive() + reward.current()
    for _ in range(n):
        plasticity.observe_spikes(engine.step(current))


# ------------------------------------------------------------ the load-bearing test


def test_stdp_mutates_the_simulated_weights():
    engine, plasticity, reward, idx = _learning_setup()
    fixed_input = _drive()

    r0 = probe_response(engine, fixed_input, PROBE_STEPS)
    w0 = engine.W.data[idx].copy()

    # A baseline first, from an unrewarded stretch: the rule is driven by
    # `DAN_rate - DAN_baseline`, so a constant dopamine level teaches nothing.
    _step_n(engine, plasticity, reward, 400)
    reward.dopamine(engine.get_firing_rates())

    reward.register_event("damage_dealt", 5.0)
    _step_n(engine, plasticity, reward, 400)
    moved = plasticity.apply(reward.dopamine(engine.get_firing_rates()))

    assert moved > 0.0
    assert not np.allclose(w0, engine.W.data[idx])
    assert not np.allclose(r0, probe_response(engine, fixed_input, PROBE_STEPS))


def test_the_response_that_moved_is_downstream_of_the_changed_synapses():
    """The second assertion above, localised: MBON and its readout are what moved."""
    engine, plasticity, reward, idx = _learning_setup()
    fixed_input = _drive()
    r0 = probe_response(engine, fixed_input, PROBE_STEPS)

    _step_n(engine, plasticity, reward, 400)
    reward.dopamine(engine.get_firing_rates())
    reward.register_event("kill", 6.0)
    _step_n(engine, plasticity, reward, 400)
    plasticity.apply(reward.dopamine(engine.get_firing_rates()))

    r1 = probe_response(engine, fixed_input, PROBE_STEPS)
    assert not np.allclose(r0[MBON], r1[MBON])
    assert not np.allclose(r0[READOUT], r1[READOUT])
    # Upstream of the plastic set nothing may move: the drive is identical and
    # no sensory->KC synapse is in `idx`.
    assert np.allclose(r0[KC], r1[KC])
    assert np.allclose(r0[SENSORY], r1[SENSORY])
    assert (engine.W.data[idx] > plasticity.w0).any()


def test_frozen_weights_leave_both_the_matrix_and_the_response_alone():
    """The `--learn`-off control: same dopamine, no write, no change."""
    engine, plasticity, reward, idx = _learning_setup()
    fixed_input = _drive()
    r0 = probe_response(engine, fixed_input, PROBE_STEPS)
    w0 = engine.W.data[idx].copy()

    _step_n(engine, plasticity, reward, 400)
    reward.dopamine(engine.get_firing_rates())
    reward.register_event("damage_dealt", 5.0)
    _step_n(engine, plasticity, reward, 400)
    # `plasticity.apply` is exactly what the agent skips without `--learn`.
    reward.dopamine(engine.get_firing_rates())

    assert np.array_equal(w0, engine.W.data[idx])
    assert np.allclose(r0, probe_response(engine, fixed_input, PROBE_STEPS))


# ----------------------------------------------------------------- the index set


def test_the_plastic_set_is_exactly_the_kc_to_mbon_edges():
    c = _connectome()
    idx = kc_to_mbon(c.W, c)
    assert idx.size == len(KC) * len(MBON)

    column_of = np.repeat(np.arange(N), np.diff(c.W.indptr))
    assert set(column_of[idx].tolist()) == set(KC.tolist())
    assert set(c.W.indices[idx].tolist()) == set(MBON.tolist())


def test_the_plastic_set_is_resolved_against_the_matrix_the_engine_gets():
    """Calibration hands the engine a new matrix, so the indices must address that one.

    Gain and normalisation leave the sparsity pattern alone, so the positions
    happen to coincide here — but the ablations do not, and resolving against
    `connectome.W` would then silently address the wrong entries.
    """
    c = _connectome()
    calibrated = Calibration(gain=0.5, normalization="capped").apply(c.W)
    assert calibrated.data is not c.W.data

    idx = kc_to_mbon(calibrated, c)
    engine = _engine(calibrated, idx)
    plasticity = Plasticity.attach(engine, c)
    assert np.array_equal(plasticity.weights, calibrated.data[idx])
    assert plasticity.W is engine.W


def test_plasticity_writes_into_the_engines_own_array_not_a_copy():
    engine, plasticity, _, idx = _learning_setup()
    assert plasticity.W.data is engine.W.data
    # `LIFEngine` deposits from `_data`, bound once in `__init__`.
    assert engine._data is engine.W.data

    plasticity._e[:] = 1.0
    plasticity._scale = 1.0
    before = engine._data[idx].copy()
    plasticity.apply(10.0)
    assert not np.array_equal(before, engine._data[idx])


def test_a_dense_matrix_is_refused_rather_than_silently_reoriented():
    with pytest.raises(TypeError):
        plastic_indices(np.zeros((4, 4)), np.array([0]), np.array([1]))


# ------------------------------------------------------------------ the traces


def test_eligibility_needs_coincidence_and_decays():
    engine, plasticity, reward, _ = _learning_setup()
    _step_n(engine, plasticity, reward, 120)
    peak = plasticity.eligibility.sum()
    assert peak > 0.0

    silent = np.zeros(N, dtype=np.float32)
    for _ in range(1500):
        plasticity.observe_spikes(engine.step(silent))
    assert plasticity.eligibility.sum() < peak * 0.5


def test_a_reward_with_no_prior_activity_moves_nothing():
    """No eligibility, no credit: dopamine alone is not a learning signal."""
    c = _connectome()
    idx = kc_to_mbon(c.W, c)
    engine = _engine(c.W, idx)
    plasticity = Plasticity.attach(engine, c)
    w0 = engine.W.data[idx].copy()
    assert plasticity.apply(25.0) == 0.0
    assert np.array_equal(w0, engine.W.data[idx])


def test_the_update_preserves_sign_and_respects_the_ceiling():
    c = _connectome()
    idx = kc_to_mbon(c.W, c)
    engine = _engine(c.W, idx)
    plasticity = Plasticity.attach(engine, c, params=PlasticityParams(eta=1.0, w_max_scale=2.0))
    plasticity._e[:] = 1.0
    plasticity._scale = 1.0

    plasticity.apply(1e4)
    assert np.allclose(plasticity.weights, plasticity.w0 * 2.0)
    plasticity.apply(-1e4)
    assert np.allclose(plasticity.weights, 0.0)


def test_opposite_dopamine_moves_the_weights_the_opposite_way():
    engine, plasticity, reward, idx = _learning_setup()
    _step_n(engine, plasticity, reward, 150)
    e = plasticity.eligibility.copy()
    w0 = engine.W.data[idx].copy()

    plasticity.apply(8.0)
    up = engine.W.data[idx] - w0
    engine.W.data[idx] = w0
    plasticity._e[:] = e
    plasticity._scale = 1.0
    plasticity.apply(-8.0)
    down = engine.W.data[idx] - w0

    assert up.max() > 0.0
    assert down.min() < 0.0
    assert np.allclose(up, -down)


def test_the_probe_is_reproducible_from_the_same_weights():
    engine, _, _, _ = _learning_setup()
    fixed_input = _drive()
    assert np.array_equal(
        probe_response(engine, fixed_input, PROBE_STEPS),
        probe_response(engine, fixed_input, PROBE_STEPS),
    )


# ------------------------------------------------------- the loop that uses it


class _StubClient:
    """Everything `Agent.tick` asks of a client, and nothing else."""

    def __init__(self) -> None:
        self.ready = None
        self.last_reward = None
        self.dropped_game_ticks = 0
        self.sent: list = []

    def send_cmd(self, revision: int, action) -> None:
        self.sent.append(action)

    def send_noop(self, revision: int) -> None:
        self.sent.append(None)


def _state(revision: int):
    return StateUpdate.from_dict(
        {
            "revision": revision,
            "tick": revision,
            "droppedSinceLast": 0,
            "deadlineMs": 100_000,
            "tickMs": 600,
            "observedTickMs": None,
            "state": {
                "tick": revision,
                "inGame": True,
                "modalOpen": False,
                "player": {
                    "name": "flybot01",
                    "combatLevel": 3,
                    "hp": 10,
                    "maxHp": 10,
                    "x": 3220,
                    "z": 3218,
                    "level": 0,
                    "runEnergy": 100,
                    "animId": -1,
                    "inCombat": False,
                    "targetIndex": -1,
                    "targetType": "none",
                    "isDead": False,
                    "lifeId": 1,
                },
                "npcs": [],
                "groundItems": [],
                "locs": [],
                "inventory": [],
                "skills": {},
                "opRejectedCount": 0,
            },
        }
    )


def _reward_message(revision: int):
    return parse_server_message(
        {
            "t": "reward",
            "revision": revision,
            "combatEvents": [
                {
                    "tick": revision,
                    "observationId": None,
                    "type": "damage_dealt",
                    "damage": 40,
                    "sourceType": "player",
                    "sourceIndex": 0,
                    "targetType": "npc",
                    "targetIndex": 1,
                }
            ],
            "xpDelta": {"Attack": 400},
        }
    )


def _loop_agent(learn: bool):
    c = _connectome()
    engine = _engine(c.W, kc_to_mbon(c.W, c))
    plasticity = Plasticity.attach(engine, c, params=PlasticityParams(eta=5e-3)) if learn else None
    client = _StubClient()
    agent = Agent(
        client=client,
        engine=engine,
        motor=MotorIndex.from_connectome(c),
        collision=CollisionGrid(
            grid=np.ones((40, 40), dtype=bool), x_min=3200, z_min=3200, level=0
        ),
        encoder=lambda frame: _drive(),
        params=AgentParams(substeps_per_subframe=100),
        reward=RewardRouter(DopamineIndex.from_connectome(c), N),
        plasticity=plasticity,
    )
    return agent, client, engine, plasticity


def test_the_tick_loop_injects_dopamine_and_writes_the_update():
    agent, client, engine, plasticity = _loop_agent(learn=True)
    idx = plasticity.plastic_idx
    w0 = engine.W.data[idx].copy()

    first = agent.tick(_state(1))
    assert first.dopamine == pytest.approx(0.0)  # the baseline tick
    assert first.weight_delta == 0.0

    client.last_reward = _reward_message(2)
    second = agent.tick(_state(2))

    assert agent.reward.events > 0
    assert second.dan_appetitive_hz > 0.0
    assert second.dan_aversive_hz == 0.0
    assert second.dopamine > 0.0
    assert second.weight_delta > 0.0
    assert not np.allclose(w0, engine.W.data[idx])


def test_the_tick_loop_without_learning_still_routes_dopamine_but_freezes_weights():
    """The `--learn`-off control: identical dynamics, no write."""
    agent, client, engine, _ = _loop_agent(learn=False)
    idx = kc_to_mbon(engine.W, _connectome())
    w0 = engine.W.data[idx].copy()

    agent.tick(_state(1))
    client.last_reward = _reward_message(2)
    report = agent.tick(_state(2))

    assert report.dopamine > 0.0
    assert report.weight_delta is None
    assert np.array_equal(w0, engine.W.data[idx])


# ----------------------------------------------------------- the real build


@pytest.mark.slow
def test_the_real_connectome_resolves_a_plastic_set_and_both_dan_populations():
    if not DEFAULT_PATH.exists():
        pytest.skip(f"no connectome at {DEFAULT_PATH}")

    c = load(DEFAULT_PATH)
    idx = kc_to_mbon(c.W, c)
    dan = DopamineIndex.from_connectome(c)

    assert len(c.population("KC")) == 4064
    assert len(c.population("MBON")) == 97
    assert len(dan.appetitive) == 316
    assert len(dan.aversive) == 16
    # Every KC->MBON edge and nothing else; a fraction of a percent of the graph.
    assert 0 < idx.size < c.W.nnz // 100
    print(f"\nplastic KC->MBON synapses: {idx.size} of {c.W.nnz}")
