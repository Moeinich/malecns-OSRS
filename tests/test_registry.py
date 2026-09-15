"""Registry resolves every required cell type by name from the real annotations."""

from __future__ import annotations

import numpy as np
import pytest
from pyarrow import feather

from flybrain.connectome import vocab
from flybrain.connectome.registry import (
    CELL_TYPES_PATH,
    CellTypeRegistry,
    MissingCellTypeError,
    _load_cell_types,
)

pytestmark = pytest.mark.skipif(
    not vocab.DEFAULT_ANNOTATIONS_PATH.exists(),
    reason="MaleCNS annotations not fetched; run tools/fetch_connectome.py",
)

#: Verified against the real file: (cardinality, left, right).
REQUIRED = {
    "DNp01": (2, 1, 1),
    "DNa02": (2, 1, 1),
    "DNpe017": (2, 1, 1),
    "MDN": (4, 2, 2),
    "DNp09": (2, 1, 1),
}


@pytest.fixture(scope="module")
def table():
    return feather.read_table(vocab.DEFAULT_ANNOTATIONS_PATH, columns=list(vocab.REQUIRED_COLUMNS))


@pytest.fixture(scope="module")
def registry(table):
    return CellTypeRegistry(table)


@pytest.mark.parametrize("name,expected", REQUIRED.items())
def test_required_type_resolves_with_verified_cardinality(registry, name, expected):
    total, left, right = expected
    assert len(registry.population(name)) == total
    assert len(registry.population(name, "L")) == left
    assert len(registry.population(name, "R")) == right


def test_body_ids_are_auditable(registry):
    body_ids = registry.body_ids("DNa02")
    assert body_ids.dtype == np.int64
    assert len(body_ids) == 2
    assert len(set(body_ids.tolist())) == 2


def test_require_passes_for_every_required_type(registry):
    registry.require(*REQUIRED)


def test_require_raises_on_a_bogus_name(registry):
    with pytest.raises(MissingCellTypeError):
        registry.require("DNp01", "DNzz999")


def test_prefix_resolution_does_not_swallow_longer_types(registry):
    """`Tm1` must not absorb `Tm12`/`Tm16` — that is how a population silently grows."""
    assert len(registry.population("Tm1")) == 1777


def test_prefix_resolution_widens_over_a_family(registry):
    assert len(registry.population("KC")) == 4064
    assert len(registry.population("PAM")) == 316


def test_hex_lattice_yields_the_real_column_count(registry):
    columns = registry.columns()
    assert len(columns) == vocab.N_OL_COLUMNS
    assert (min(h for h, _ in columns), max(h for h, _ in columns)) == vocab.HEX1_RANGE
    assert (min(h for _, h in columns), max(h for _, h in columns)) == vocab.HEX2_RANGE


def test_vocab_verify_passes_on_the_real_file(table):
    vocab.verify(table)


def test_cell_types_toml_round_trips_required_cardinalities():
    spec = _load_cell_types(CELL_TYPES_PATH)
    required = {name: s["expect"] for name, s in spec.items() if s.get("required")}
    assert required == {name: total for name, (total, _, _) in REQUIRED.items()}
