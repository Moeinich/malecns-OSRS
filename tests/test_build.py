"""The real build, at a small K so it runs in the default suite.

These assertions are the ones FlyBrain could not have passed: the named cells
are really in the graph, the signs come from real transmitter calls, and the
bodyIds in the provenance are the ones in the release.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.sparse as sp
from pyarrow import feather

from flybrain.connectome import loader, vocab
from flybrain.connectome.build import DEFAULT_NT_PATH, BuildParams, build
from flybrain.connectome.registry import CellTypeRegistry
from flybrain.connectome.select import DEFAULT_WEIGHTS_PATH, SelectionParams
from flybrain.engine.lif import LIFEngine

pytestmark = pytest.mark.skipif(
    not all(
        p.exists() for p in (vocab.DEFAULT_ANNOTATIONS_PATH, DEFAULT_WEIGHTS_PATH, DEFAULT_NT_PATH)
    ),
    reason="MaleCNS release not fetched; run tools/fetch_connectome.py",
)

SMALL_K = 2000


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("connectome") / "connectome_test.npz"
    provenance = build(out, SelectionParams(k=SMALL_K), BuildParams())
    return out, provenance


@pytest.fixture(scope="module")
def connectome(built):
    return loader.load(built[0])


@pytest.fixture(scope="module")
def registry():
    return CellTypeRegistry(
        feather.read_table(vocab.DEFAULT_ANNOTATIONS_PATH, columns=list(vocab.REQUIRED_COLUMNS))
    )


@pytest.mark.parametrize("name", sorted(vocab.REQUIRED_TYPE_CARDINALITY))
def test_required_cell_type_survives_selection(connectome, name):
    """A required type pruned out of the subgraph is a network we cannot address."""
    members = connectome.population(name)
    assert members.size == vocab.REQUIRED_TYPE_CARDINALITY[name]


def test_steering_and_escape_are_laterally_resolved(connectome):
    for name in ("DNa02", "DNp01"):
        assert connectome.population(name, "L").size == 1
        assert connectome.population(name, "R").size == 1


def test_mushroom_body_and_central_complex_are_anchored(connectome):
    for name in ("KC", "MBON", "PPL1", "PAM"):
        assert connectome.population(name).size > 0
    assert connectome.population("KC").size == 4064


def test_weights_are_signed_and_inhibition_exists(connectome):
    data = connectome.W.data
    assert np.all(data != 0)
    assert (data < 0).any(), "a network with no inhibition is not this connectome"
    assert (data > 0).any()
    assert 0.1 < (data < 0).mean() < 0.9


def test_sign_is_a_row_scaling_not_a_per_edge_label(built, connectome):
    """Dale's law: every neuron's outgoing edges carry one sign."""
    csr = loader.load_csr(built[0])
    signs = np.sign(csr.data)
    for start, end in zip(csr.indptr[:-1], csr.indptr[1:]):
        if end > start:
            assert len(np.unique(signs[start:end])) == 1


def test_csc_and_csr_agree(built, connectome):
    csr = loader.load_csr(built[0])
    assert csr.shape == connectome.W.shape
    assert (csr.tocsc() - connectome.W).nnz == 0


def test_provenance_body_ids_match_the_annotations(built, registry):
    _, provenance = built
    recorded = provenance["population_body_ids"]
    assert recorded["DNa02"], "provenance must carry auditable bodyIds"
    for name in vocab.REQUIRED_TYPE_CARDINALITY:
        assert set(recorded[name]) <= set(registry.body_ids(name).tolist())
        assert set(recorded[f"{name}|L"]) <= set(registry.body_ids(name, "L").tolist())


def test_provenance_records_sources_and_parameters(built):
    _, provenance = built
    for name, source in provenance["sources"].items():
        assert source["sha256"] and name in source["url"]
    assert provenance["selection"]["k"] == SMALL_K
    assert provenance["selection"]["achieved_min_syn"] >= provenance["selection"]["min_syn"]
    assert provenance["achieved"]["n_edges"] <= SelectionParams().max_edges


def test_unknown_transmitter_fraction_is_reported_and_small(built):
    _, provenance = built
    assert provenance["achieved"]["unknown_nt_edge_fraction"] <= 0.05


def test_body_ids_of_a_population_are_auditable(connectome, registry):
    ids = connectome.body_ids_of("DNa02", "L")
    assert ids.size == 1
    assert ids[0] in registry.body_ids("DNa02", "L")


def test_engine_shares_the_one_weight_array(connectome):
    """FlyBrain's learning was a no-op because a second copy existed. None here."""
    engine = LIFEngine(connectome.W)
    assert engine.W is connectome.W
    assert engine.W.data is connectome.W.data
    before = float(connectome.W.data[0])
    engine.W.data[0] += 1.0
    assert connectome.W.data[0] == before + 1.0
    engine.W.data[0] = before


def test_loader_returns_a_csc_matrix_the_engine_accepts_unchanged(connectome):
    assert sp.isspmatrix_csc(connectome.W)
    assert connectome.W.data.dtype == np.float32


def test_population_lookup_refuses_an_unbuilt_name(connectome):
    with pytest.raises(KeyError):
        connectome.population("DNa02", "M")


@pytest.mark.skipif(
    not os.environ.get("FLYBRAIN_SLOW"),
    reason="full-K build is a manual path; set FLYBRAIN_SLOW=1",
)
def test_full_k_build(tmp_path):
    provenance = build(tmp_path / "full.npz", SelectionParams(), BuildParams())
    assert provenance["achieved"]["n_neurons"] > 20_000
    assert provenance["achieved"]["n_edges"] <= SelectionParams().max_edges
