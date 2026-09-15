from __future__ import annotations

import ast
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from flybrain.connectome import vocab
from flybrain.connectome.loader import DEFAULT_PATH, load
from flybrain.engine.lif import LIFEngine
from flybrain.sensory import encode as encode_mod
from flybrain.sensory.encode import EncodeParams, encode, map_builds
from flybrain.sensory.retina import CH_LUMINANCE, CH_THREAT, N_CHANNELS

SIZE = 60

needs_artifacts = pytest.mark.skipif(
    not (DEFAULT_PATH.exists() and vocab.DEFAULT_ANNOTATIONS_PATH.exists()),
    reason="connectome_v1.npz or the annotations feather is missing; run the build first",
)


@pytest.fixture(scope="module")
def connectome():
    return load()


def _frame(x0: int = 0, x1: int = SIZE, value: float = 1.0, channel: int = CH_LUMINANCE):
    frame = np.zeros((SIZE, SIZE, N_CHANNELS), dtype=np.float32)
    frame[:, x0:x1, channel] = value
    return frame


def _eyes(connectome, params: EncodeParams | None = None):
    inj = encode_mod._injection(SIZE, connectome, params or EncodeParams())
    return inj.eyes["L"], inj.eyes["R"]


# ------------------------------------------------------------------ firewall


def test_encoder_never_reaches_the_motor_layer():
    allowed_prefixes = ("flybrain.loop.types", "flybrain.sensory.", "flybrain.connectome.")
    path = Path(__file__).resolve().parent.parent / "flybrain" / "sensory" / "encode.py"
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("flybrain"):
            assert node.module.startswith(allowed_prefixes), node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("flybrain"), alias.name


def test_the_encoder_does_not_mention_actions():
    path = Path(__file__).resolve().parent.parent / "flybrain" / "sensory" / "encode.py"
    source = path.read_text().lower()
    for word in ("motorcommand", "sendwalk", "attack_fovea", "import flybrain.motor"):
        assert word not in source, word


# ------------------------------------------------------------ transduction


def test_naka_rushton_saturates_rather_than_growing_without_bound():
    params = EncodeParams()
    levels = np.array([0.0, 0.25, 1.0, 10.0, 1e6])
    out = encode_mod._naka_rushton(levels, params)
    assert out[0] == 0.0
    assert out[1] == pytest.approx(0.5)  # sigma is the half-saturation point
    assert np.all(np.diff(out) > 0)
    assert out[-1] < 1.0
    assert out[-1] == pytest.approx(1.0, abs=1e-6)


#: How far a saturated column must sit above threshold, in current units. The
#: value is arbitrary; the *relationship* is not, and this is what pins it.
MIN_HEADROOM = 5.0


@needs_artifacts
def test_a_saturated_column_can_actually_reach_threshold(connectome):
    """The encoder's ceiling is a sustained current and `v_ss = v_rest + I`, so it
    must clear `v_thresh - v_rest` on its own. At `i_max = 6.0` it sat 9 units short
    permanently and no synaptic gain could rescue it — the live brain read 0.00 Hz.
    A change to `i_max` or to the LIF's voltages now breaks a test instead."""
    engine = LIFEngine(sp.csc_matrix((1, 1), dtype=np.float32))
    threshold_distance = engine.v_thresh - engine.v_rest

    saturating = encode(_frame(value=1e3), connectome)

    assert saturating.max() > threshold_distance + MIN_HEADROOM


# ------------------------------------------------------------------ the map


@needs_artifacts
def test_vector_is_float32_over_the_whole_network(connectome):
    out = encode(_frame(), connectome)
    assert out.dtype == np.float32
    # Derived from the artifact, not pinned: the claim is that the encoder covers
    # exactly the network it was given, which stays true across rebuilds.
    assert out.shape == (connectome.n,)


@needs_artifacts
def test_nothing_outside_the_sensory_indices_is_touched(connectome):
    """The anti-cheat: a descending neuron must never see injected current."""
    frame = _frame()
    frame[:, :, :] = 1.0
    out = encode(frame, connectome)

    left, right = _eyes(connectome)
    sensory = np.concatenate([left.neurons, right.neurons, *left.chromatic, *right.chromatic])
    for name in ("DNa02", "DNp01", "MDN", "DNp09"):
        assert np.all(out[connectome.population(name)] == 0.0), name

    others = np.setdiff1d(np.arange(connectome.n), sensory)
    assert np.all(out[others] == 0.0)
    assert out[sensory].max() > 0.0


@needs_artifacts
def test_a_patch_on_the_left_drives_the_left_eye_harder(connectome):
    left, right = _eyes(connectome)

    on_the_left = encode(_frame(0, SIZE // 3), connectome)
    assert on_the_left[left.neurons].sum() > on_the_left[right.neurons].sum()

    on_the_right = encode(_frame(2 * SIZE // 3, SIZE), connectome)
    assert on_the_right[right.neurons].sum() > on_the_right[left.neurons].sum()


@needs_artifacts
def test_the_two_eyes_see_the_same_thing_when_the_scene_is_uniform(connectome):
    left, right = _eyes(connectome)
    out = encode(_frame(), connectome)
    per_cell_left = out[left.neurons].mean()
    per_cell_right = out[right.neurons].mean()
    assert per_cell_left == pytest.approx(per_cell_right, rel=0.01)


@needs_artifacts
def test_the_eyes_are_hemisphere_split_and_column_sized(connectome):
    left, right = _eyes(connectome)
    assert len(left.counts) <= 892 and len(right.counts) <= 892
    assert set(left.neurons.tolist()).isdisjoint(right.neurons.tolist())
    assert len(left.neurons) < SIZE * SIZE  # fewer columns than pixels: a downsample
    for name in ("L1", "L2", "Tm1", "Mi1"):
        assert len(np.intersect1d(connectome.population(name), left.neurons)) > 0, name
    # L3's hex assignment is right-eye only in MaleCNS v1.0 (the plan's "L3 50%"),
    # so its left-eye cells fall through to the non-retinotopic pool.
    assert len(np.intersect1d(connectome.population("L3"), right.neurons)) > 0
    assert len(np.intersect1d(connectome.population("L3"), left.neurons)) == 0


@needs_artifacts
def test_chromatic_channels_are_non_retinotopic_and_disjoint(connectome):
    left, _ = _eyes(connectome)
    slabs = [set(s.tolist()) for s in left.chromatic]
    for a, b in pairwise(slabs):
        assert a.isdisjoint(b)
    assert set(left.neurons.tolist()).isdisjoint(set().union(*slabs))

    out = encode(_frame(channel=CH_THREAT), connectome)
    threat = out[left.chromatic[0]]
    assert threat.max() > 0.0
    assert np.all(threat == threat[0])  # one scalar per eye, no spatial structure


# ------------------------------------------------------------ caching, timing


@needs_artifacts
def test_the_pixel_to_column_map_is_built_once(connectome):
    encode(_frame(), connectome)
    before = map_builds()
    for _ in range(5):
        encode(_frame(), connectome)
    assert map_builds() == before


@needs_artifacts
def test_encoding_is_deterministic(connectome):
    frame = _frame(10, 30)
    assert np.array_equal(encode(frame, connectome), encode(frame, connectome))


@needs_artifacts
def test_encode_fits_in_the_tick_budget(connectome):
    frame = _frame(10, 30)
    encode(frame, connectome)
    start = time.perf_counter()
    for _ in range(50):
        encode(frame, connectome)
    per_call_ms = (time.perf_counter() - start) / 50 * 1e3
    print(f"\nencode: {per_call_ms:.3f} ms/call ({4 * per_call_ms:.3f} ms per tick at 4 frames)")
    assert per_call_ms < 10.0  # it runs 4x per 400 ms tick


@needs_artifacts
def test_a_subframe_stack_is_rejected(connectome):
    with pytest.raises(ValueError):
        encode(np.zeros((4, SIZE, SIZE, N_CHANNELS), dtype=np.float32), connectome)
