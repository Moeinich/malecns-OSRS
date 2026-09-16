"""The load-time superclass join — the artifact stores rows, not the column."""

from __future__ import annotations

import logging

import numpy as np
import pyarrow as pa
import pytest
from pyarrow import feather

from flybrain.connectome import loader, vocab


@pytest.fixture
def annotations(tmp_path):
    path = tmp_path / vocab.ANNOTATIONS_FILENAME
    feather.write_feather(
        pa.table(
            {
                "superclass": [
                    "descending_neuron",
                    None,
                    "ol_intrinsic",
                    "cb_intrinsic",
                    "ol_intrinsic",
                ]
            }
        ),
        path,
    )
    return path


@pytest.fixture
def artifact(tmp_path):
    path = tmp_path / "connectome.npz"
    np.savez(path, annotation_rows=np.array([4, 0, 1, 3], dtype=np.int64))
    return path


def test_the_join_follows_annotation_rows_rather_than_neuron_order(artifact, annotations):
    codes, labels = loader.superclasses(artifact, annotations)
    assert labels == ("cb_intrinsic", "descending_neuron", "ol_intrinsic")
    assert [labels[c] if c >= 0 else None for c in codes] == [
        "ol_intrinsic",
        "descending_neuron",
        None,
        "cb_intrinsic",
    ]


def test_an_unclassed_neuron_is_minus_one_not_a_label(artifact, annotations):
    codes, _labels = loader.superclasses(artifact, annotations)
    assert codes[2] == -1


def test_missing_annotations_are_a_log_line_and_None_not_a_failure(artifact, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=loader.__name__):
        assert loader.superclasses(artifact, tmp_path / "absent.feather") is None
    assert "absent.feather" in caplog.text


def test_a_superclass_outside_the_frozen_vocabulary_fails_loudly(artifact, tmp_path):
    path = tmp_path / "drifted.feather"
    feather.write_feather(pa.table({"superclass": ["brand_new"] * 5}), path)
    with pytest.raises(vocab.VocabDriftError):
        loader.superclasses(artifact, path)
