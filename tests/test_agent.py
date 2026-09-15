"""The closed loop against a fake sidecar: a real unix socket, no game.

The connectome here is synthetic and tiny — the point of these tests is the
wiring of the loop and the ablation machinery, not the biology. The one test
that touches the real 44,687-neuron build is marked `slow` and skips itself
when the artifacts are absent.
"""

from __future__ import annotations

import json
import socket
import tempfile
import threading

import numpy as np
import pytest
import scipy.sparse as sp

from flybrain.connectome.loader import DEFAULT_PATH, Connectome
from flybrain.engine.lif import LIFEngine
from flybrain.loop.agent import (
    Ablation,
    Agent,
    AgentParams,
    ablate_network,
    lesion,
    shuffle_degree_preserving,
)
from flybrain.loop.client import BridgeClient
from flybrain.motor.decode import MotorIndex, MotorParams
from flybrain.sensory.collision import CollisionGrid
from flybrain.sensory.retina import CH_LOOT, CH_LUMINANCE, CH_THREAT, Retina

# ------------------------------------------------------------ fake sidecar


def ready_msg(tick_ms: int = 600) -> dict:
    return {
        "t": "ready",
        "protocol": 1,
        "username": "flybot01",
        "mode": "control",
        "role": "brain",
        "tickMs": tick_ms,
        "pathfindingWarm": True,
    }


def _npc(index: int, x: int, z: int, level: int = 20) -> dict:
    return {
        "id": 3266,
        "index": index,
        "name": f"Rat {index}",
        "combatLevel": level,
        "x": x,
        "z": z,
        "size": 1,
        "distance": 3,
        "hp": 5,
        "maxHp": 5,
        "inCombat": False,
        "targetIndex": -1,
        "reachable": True,
        "options": ["Attack"],
    }


def state_msg(revision: int, x: int, z: int, npcs=(), items=(), tick_ms: int = 600) -> dict:
    """Deadline and tick come from one value: the sidecar derives one from the other."""
    return {
        "t": "state",
        "revision": revision,
        "tick": 88000 + revision,
        "droppedSinceLast": 0,
        "deadlineMs": round(tick_ms * 0.6),
        "tickMs": tick_ms,
        "observedTickMs": None,
        "state": {
            "tick": 88000 + revision,
            "inGame": True,
            "modalOpen": False,
            "player": {
                "name": "flybot01",
                "combatLevel": 3,
                "hp": 10,
                "maxHp": 10,
                "x": x,
                "z": z,
                "level": 0,
                "runEnergy": 100,
                "animId": -1,
                "inCombat": False,
                "targetIndex": -1,
                "targetType": "none",
                "isDead": False,
                "lifeId": 1,
            },
            "npcs": list(npcs),
            "groundItems": list(items),
            "locs": [],
            "inventory": [{"slot": 0, "id": 2309, "name": "Bread", "count": 1}],
            "skills": {"Attack": 0, "Hitpoints": 1154},
            "opRejectedCount": 0,
        },
    }


class FakeSidecar:
    """Accepts one brain, replays a script of states, drains whatever comes back."""

    def __init__(self, path: str, script: list[dict], tick_ms: int = 600) -> None:
        self.path = path
        self.script = script
        self.ready = ready_msg(tick_ms)
        self.received: list[dict] = []
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(path)
        self.listener.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        conn, _ = self.listener.accept()
        with conn:
            conn.sendall((json.dumps(self.ready) + "\n").encode())
            for msg in self.script:
                conn.sendall((json.dumps(msg) + "\n").encode())
            buffer = b""
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    self.received.append(json.loads(line))

    def close(self) -> None:
        self.listener.close()


# ------------------------------------------------- the synthetic fly brain

N_SENSORY = 200
LEFT_LUM = slice(0, 80)
RIGHT_LUM = slice(80, 160)
THREAT = slice(160, 180)
LOOT = slice(180, 200)

SLOTS = {
    "DNa02|L": range(200, 202),
    "DNa02|R": range(202, 204),
    "DNp09": range(204, 206),
    "DNpe017": range(206, 208),
    "MDN": range(208, 212),
    "DNp01": range(212, 214),
    "MBON": range(214, 260),
}
N = 260


def _populations() -> dict[str, np.ndarray]:
    pops = {k: np.array(v, dtype=np.int64) for k, v in SLOTS.items()}
    pops["DNa02"] = np.concatenate([pops["DNa02|L"], pops["DNa02|R"]])
    return pops


def _weights() -> sp.csc_matrix:
    """Sensory slabs -> descending neurons. Column = presynaptic, row = post."""
    pops = _populations()
    W = np.zeros((N, N), dtype=np.float32)

    def project(pre: slice, post: str, w: float) -> None:
        W[np.ix_(pops[post], np.arange(N)[pre])] = w

    project(LEFT_LUM, "DNa02|L", 0.9)
    project(RIGHT_LUM, "DNa02|R", 0.9)
    for post in ("DNp09", "DNpe017"):
        project(LEFT_LUM, post, 0.45)
        project(RIGHT_LUM, post, 0.45)
    project(THREAT, "DNp01", 4.0)
    project(LOOT, "MBON", 1.2)
    # DNp01 is the only inhibitory presynaptic population here, so Dale's law
    # holds per column and the shuffle has a sign to preserve.
    W[np.ix_(pops["MDN"], pops["DNp01"])] = -1.2

    # Sparse background wiring: without it the graph is a few dense blocks over
    # 60 distinct target rows, where a degree-preserving rewire cannot avoid
    # duplicate edges. Real connectomes are not shaped like that.
    rng = np.random.default_rng(3)
    inhibitory = set(pops["DNp01"].tolist())
    for j in range(N):
        sign = -1.0 if j in inhibitory else 1.0
        for i in rng.choice(N, size=6, replace=False):
            if i != j and W[i, j] == 0.0:
                W[i, j] = sign * 0.06
    return sp.csc_matrix(W)


def _connectome() -> Connectome:
    return Connectome(
        W=_weights(),
        body_ids=np.arange(N, dtype=np.int64),
        populations=_populations(),
        provenance={"synthetic": True},
    )


def _encoder(frame: np.ndarray) -> np.ndarray:
    """Stand-in for `sensory.encode`, which needs the real annotations feather."""
    current = np.zeros(N, dtype=np.float32)
    lum = frame[:, :, CH_LUMINANCE]
    current[LEFT_LUM] = 40.0 * float(lum[:, :30].mean())
    current[RIGHT_LUM] = 40.0 * float(lum[:, 30:].mean())
    current[THREAT] = 60.0 * float(frame[:, :, CH_THREAT].max())
    current[LOOT] = 60.0 * float(frame[:, :, CH_LOOT].max())
    return current


def _collision() -> CollisionGrid:
    rng = np.random.default_rng(7)
    grid = rng.random((80, 80)) > 0.25
    return CollisionGrid(grid=grid, x_min=3200, z_min=3200, level=0)


def _script(n_ticks: int, tick_ms: int = 600) -> list[dict]:
    """The player drifts east; a rat closes in from tick 3, loot drops at tick 6."""
    out = []
    for k in range(n_ticks):
        x, z = 3222 + k // 2, 3218
        npcs = [_npc(1, x + max(6 - k, 2), z + 1)] if k >= 3 else []
        items = (
            [
                {
                    "id": 526,
                    "name": "Bones",
                    "count": 1,
                    "x": x + 1,
                    "z": z,
                    "distance": 1,
                    "reachable": True,
                }
            ]
            if k >= 6
            else []
        )
        out.append(state_msg(k + 1, x, z, npcs, items, tick_ms=tick_ms))
    return out


def _agent(sidecar: FakeSidecar, W: sp.csc_matrix, **params) -> Agent:
    connectome = _connectome()
    motor = MotorIndex.from_connectome(connectome, MotorParams(rate_window_s=0.5))
    return Agent(
        client=BridgeClient(sidecar.path, reconnect=False),
        engine=LIFEngine(W, seed=1),
        motor=motor,
        collision=_collision(),
        encoder=_encoder,
        retina=Retina(),
        params=AgentParams(substeps_per_subframe=params.pop("substeps", None), **params),
    )


def run_condition(
    ablation: Ablation | None,
    n_ticks: int = 12,
    tick_ms: int = 600,
    state_tick_ms: int | None = None,
    **params,
):
    connectome = _connectome()
    W = ablation.apply(connectome) if ablation is not None else connectome.W.copy()
    # Not pytest's tmp_path: those paths overflow the 104-byte AF_UNIX limit.
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        script = _script(n_ticks, tick_ms=tick_ms if state_tick_ms is None else state_tick_ms)
        sidecar = FakeSidecar(f"{tmp}/b.sock", script, tick_ms=tick_ms)
        agent = _agent(sidecar, W, **params)
        try:
            with agent.client:
                reports = list(agent.run(max_ticks=n_ticks))
        finally:
            sidecar.close()
    return agent, reports, sidecar.received


# ----------------------------------------------------------------- the loop


def test_a_synthetic_state_produces_a_well_formed_action():
    agent, reports, sent = run_condition(None)

    assert len(reports) == 12
    assert [m["t"] for m in sent] == ["hello"] + ["cmd"] * 12
    for message, report in zip(sent[1:], reports, strict=True):
        assert message["revision"] == report.revision
        assert message["action"] == report.action.to_dict()
        assert message["action"]["kind"] in {
            "walk",
            "flee",
            "attack_fovea",
            "pickup_fovea",
            "eat",
            "idle",
        }
    assert agent.last is not None
    assert agent.last.substeps == 600  # 4 sub-frames x 150, derived from the 600 ms tick
    assert all(r.mean_rate_hz >= 0.0 for r in reports)


def test_substeps_follow_the_tick_the_sidecar_reports():
    """The count is derived, not hardcoded: a server at another tickrate must not
    leave the brain simulating the wrong amount of biological time per decision."""
    fast, _, _ = run_condition(None, n_ticks=4, tick_ms=100)
    assert fast.tick_ms == 100
    assert fast.last.substeps == 100  # 4 x 25

    pinned, _, _ = run_condition(None, n_ticks=4, tick_ms=100, substeps=50)
    assert pinned.last.substeps == 200  # an explicit override still wins


def test_substeps_follow_the_measured_tick_over_the_handshake(capsys):
    """The sidecar measures the engine mid-run; `ready.tickMs` is only configured."""
    agent, _, _ = run_condition(None, n_ticks=4, tick_ms=600, state_tick_ms=400)
    assert agent.tick_ms == 400
    assert agent.last.substeps == 400  # 4 x 100, not the 600 the handshake claimed

    err = capsys.readouterr().err
    assert err.count("TICK MISMATCH") == 1  # once, not per tick
    assert "150 -> 100" in err  # the substep count, 4 sub-frames of each

    pinned, _, _ = run_condition(None, n_ticks=4, tick_ms=600, state_tick_ms=400, substeps=50)
    assert pinned.last.substeps == 200


def test_dry_run_sends_noop_but_still_computes():
    _, reports, sent = run_condition(None, dry_run=True)
    assert [m["t"] for m in sent] == ["hello"] + ["noop"] * 12
    assert any(r.action.kind != "idle" for r in reports)


def test_the_loop_keeps_up(capsys):
    agent, reports, _ = run_condition(None)
    with capsys.disabled():
        print(
            f"\n  synthetic N={N}: {agent.mean_ms_per_tick:.1f} ms/tick "
            f"(retina {np.mean([r.ms_retina for r in reports]):.2f} "
            f"encode {np.mean([r.ms_encode for r in reports]):.2f} "
            f"lif {agent.mean_ms_lif:.1f} "
            f"decode {np.mean([r.ms_decode for r in reports]):.2f})"
        )
    assert agent.overruns == 0
    assert agent.mean_ms_per_tick < 360.0  # 60% of a 600 ms tick


# ------------------------------------------------------------- ablations


def test_ablating_the_network_changes_and_flattens_behaviour():
    _, intact, _ = run_condition(None)
    _, ablated, _ = run_condition(Ablation(ablate_network=True))

    assert [r.action.to_dict() for r in ablated] != [r.action.to_dict() for r in intact]
    # Nothing reaches a descending neuron any more, so every tick decodes the same.
    assert {r.kind for r in ablated} == {"idle"}
    # The sensory slabs still fire — they get injected current — but nothing
    # downstream of a synapse does, so every tick decodes to the same command.
    assert {r.command for r in ablated} == {ablated[0].command}
    assert len({r.kind for r in intact}) > 1


def test_lesioning_dnp01_removes_escape_and_leaves_locomotion():
    _, intact, _ = run_condition(None)
    _, lesioned, _ = run_condition(Ablation(lesions=("DNp01",)))

    assert any(r.command.escape for r in intact), "the intact network never escaped"
    assert any(r.kind == "flee" for r in intact)

    assert not any(r.command.escape for r in lesioned)
    assert not any(r.kind == "flee" for r in lesioned)
    # The double dissociation: locomotion survives the lesion.
    assert any(r.kind == "walk" for r in lesioned)
    assert max(r.command.drive for r in lesioned) > 0.0


def test_lesion_silences_the_population_in_both_directions():
    c = _connectome()
    W = lesion(c.W, c.population("DNp01"))
    dense = W.toarray()
    idx = c.population("DNp01")
    assert not dense[idx, :].any()
    assert not dense[:, idx].any()
    assert dense.any(), "the lesion removed the rest of the network too"


def test_ablation_never_touches_the_connectomes_own_weights():
    c = _connectome()
    before = c.W.data.copy()
    for ablation in (
        Ablation(ablate_network=True),
        Ablation(shuffle=True, seed=3),
        Ablation(lesions=("DNa02",)),
    ):
        ablation.apply(c)
    assert np.array_equal(c.W.data, before)


def test_ablate_network_keeps_structure_but_zeroes_every_weight():
    c = _connectome()
    W = ablate_network(c.W)
    assert W.nnz == c.W.nnz
    assert not W.data.any()


# --------------------------------------------------------------- shuffle


def _in_degree(W: sp.csc_matrix) -> np.ndarray:
    return np.bincount(W.indices, minlength=W.shape[0])


def _out_degree(W: sp.csc_matrix) -> np.ndarray:
    return np.diff(W.indptr)


def _column_signs(W: sp.csc_matrix) -> list[set[float]]:
    return [
        {float(np.sign(v)) for v in W.data[W.indptr[j] : W.indptr[j + 1]]}
        for j in range(W.shape[1])
    ]


def test_shuffle_preserves_in_and_out_degree_and_sign_exactly():
    c = _connectome()
    W, conflicts = shuffle_degree_preserving(c.W, seed=11)

    assert np.array_equal(_out_degree(W), _out_degree(c.W))
    assert np.array_equal(_in_degree(W), _in_degree(c.W))
    assert _column_signs(W) == _column_signs(c.W)
    assert conflicts == 0
    assert not (W.indices == np.repeat(np.arange(N), _out_degree(W))).any(), "self-loop"
    # And it is actually a rewire, not the identity.
    assert not np.array_equal(W.indices, c.W.indices)


def test_shuffle_is_reproducible_under_a_seed():
    c = _connectome()
    a, _ = shuffle_degree_preserving(c.W, seed=5)
    b, _ = shuffle_degree_preserving(c.W, seed=5)
    other, _ = shuffle_degree_preserving(c.W, seed=6)
    assert np.array_equal(a.indices, b.indices)
    assert not np.array_equal(a.indices, other.indices)


def test_shuffle_changes_behaviour():
    _, intact, _ = run_condition(None)
    _, shuffled, _ = run_condition(Ablation(shuffle=True, seed=2))
    assert [r.action.to_dict() for r in shuffled] != [r.action.to_dict() for r in intact]


def test_ablation_label_and_notes_describe_the_condition():
    c = _connectome()
    ablation = Ablation(lesions=("DNp01",), shuffle=True, seed=4)
    ablation.apply(c)
    assert ablation.label == "shuffle(seed=4)+lesion:DNp01"
    assert any("DNp01" in note for note in ablation.notes)
    assert Ablation().label == "real"


def test_an_unknown_population_names_the_ones_that_exist():
    with pytest.raises(KeyError, match="DNa02"):
        Ablation(lesions=("DNz99",)).apply(_connectome())


# ------------------------------------------------------- the real artifact


@pytest.mark.slow
@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="connectome_v1.npz is missing")
def test_shuffle_preserves_degree_on_the_real_connectome(capsys):
    from flybrain.connectome.loader import load

    c = load()
    W, conflicts = shuffle_degree_preserving(c.W, seed=0)
    with capsys.disabled():
        print(f"\n  real connectome: n={c.n} nnz={c.W.nnz} unresolved conflicts={conflicts}")
    assert np.array_equal(_out_degree(W), _out_degree(c.W))
    assert np.array_equal(_in_degree(W), _in_degree(c.W))
    assert conflicts == 0
