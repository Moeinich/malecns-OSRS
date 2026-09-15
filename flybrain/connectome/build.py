"""Compile the MaleCNS v1.0 release into a signed, runnable network.

    uv run python -m flybrain.connectome.build --out data/cache/connectome_v1.npz

Neurotransmitter identity is a property of the *presynaptic* neuron (Dale's
law), so signing is a scaling along the presynaptic axis of the unsigned
synapse counts, never a per-edge label. The emitted matrix is `W[post, pre]`:
the engine gathers column `j` when neuron `j` fires, so the presynaptic axis
must be the column axis and every column is sign-pure.

The artifact carries the source URLs, their SHA256s, the selection parameters
and the resolved bodyIds of every named population, so the claim "this is the
fly" is checkable against neuprint by anyone.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import scipy.sparse as sp
from pyarrow import feather

from flybrain.connectome import registry as registry_mod
from flybrain.connectome import select as select_mod
from flybrain.connectome import vocab
from flybrain.connectome.registry import CellTypeRegistry

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
NT_FILENAME = "body-neurotransmitters-male-cns-v1.0.feather"
DEFAULT_NT_PATH = vocab.DEFAULT_ANNOTATIONS_PATH.parent / NT_FILENAME
CHECKSUMS_PATH = ROOT / "tools" / "checksums.json"
BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"

#: The complete v1.0 neurotransmitter vocabulary. An unlisted value is drift,
#: not an unknown: it must fail the build rather than silently become +1.
NT_SIGN = {
    "acetylcholine": 1,
    "dopamine": 1,
    "octopamine": 1,
    "serotonin": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
}
NT_UNKNOWN = "unclear"

#: `consensus_nt` first; where it abstains, the per-celltype and per-body
#: predictions in turn. Each fallback is recorded in the provenance.
NT_COLUMNS = ("consensus_nt", "celltype_predicted_nt", "predicted_nt")

UNKNOWN_SIGNS = {"excitatory": 1, "inhibitory": -1, "zero": 0}

POPULATION_SEPARATOR = "|"

#: Real EM soma coordinates, `list<int64>` of `[x, y, z]`. Not in
#: `vocab.REQUIRED_COLUMNS`: nothing in the model reads it, only telemetry.
SOMA_COLUMN = "somaLocation"


class UnknownNeurotransmitterError(RuntimeError):
    """Too much of the network has no transmitter call to be worth simulating."""


@dataclass(frozen=True)
class BuildParams:
    unknown_nt: str = "excitatory"
    max_unknown_fraction: float = 0.05


def neurotransmitter_labels(
    body_ids: np.ndarray, nt_path: Path = DEFAULT_NT_PATH
) -> tuple[np.ndarray, dict[str, int]]:
    """Resolve one transmitter label per body, cascading over the prediction columns."""
    if not nt_path.exists():
        raise FileNotFoundError(f"neurotransmitters not found at {nt_path}; run fetch_connectome")

    table = feather.read_table(nt_path, columns=["body", *NT_COLUMNS])
    bodies = np.asarray(table.column("body").to_pylist(), dtype=np.int64)
    order = np.argsort(bodies, kind="stable")
    bodies = bodies[order]

    idx = select_mod.map_bodies(body_ids, bodies)
    present = idx >= 0
    labels = np.full(len(body_ids), NT_UNKNOWN, dtype=object)
    resolved_by = {"missing_from_table": int((~present).sum())}

    for column in NT_COLUMNS:
        pending = present & (labels == NT_UNKNOWN)
        if not pending.any():
            resolved_by[column] = 0
            continue
        values = np.array(table.column(column).to_pylist(), dtype=object)[order]
        candidate = values[idx[pending]]
        candidate = np.where(np.equal(candidate, None), NT_UNKNOWN, candidate)
        before = int((labels == NT_UNKNOWN).sum())
        labels[pending] = candidate
        resolved_by[column] = before - int((labels == NT_UNKNOWN).sum())

    unknown = set(np.unique(labels)) - set(NT_SIGN) - {NT_UNKNOWN}
    if unknown:
        raise UnknownNeurotransmitterError(f"unknown transmitter values in v1.0: {sorted(unknown)}")
    return labels, resolved_by


def sign_weights(
    weights: sp.csr_matrix, labels: np.ndarray, params: BuildParams
) -> tuple[sp.csr_matrix, dict[str, object]]:
    """Scale each presynaptic neuron's outgoing row by its transmitter sign."""
    unknown_sign = UNKNOWN_SIGNS[params.unknown_nt]
    signs = np.array(
        [NT_SIGN.get(label, unknown_sign) for label in labels],
        dtype=np.float32,
    )

    out_degree = np.diff(weights.indptr)
    is_unknown = np.array([label == NT_UNKNOWN for label in labels])
    unknown_edges = int(out_degree[is_unknown].sum())
    unknown_edge_fraction = unknown_edges / weights.nnz if weights.nnz else 0.0
    unknown_neuron_fraction = float(is_unknown.mean())

    log.info(
        "transmitter unknown for %d/%d neurons (%.2f%%) carrying %d/%d edges (%.2f%%); "
        "defaulting to %s",
        int(is_unknown.sum()),
        len(labels),
        100 * unknown_neuron_fraction,
        unknown_edges,
        weights.nnz,
        100 * unknown_edge_fraction,
        params.unknown_nt,
    )
    if unknown_edge_fraction > params.max_unknown_fraction:
        raise UnknownNeurotransmitterError(
            f"{unknown_edge_fraction:.2%} of edges have no transmitter call, over the "
            f"{params.max_unknown_fraction:.2%} limit; the sign of the network would be a guess"
        )

    signed = weights.copy()
    signed.data = signed.data * np.repeat(signs, out_degree)

    inhibitory_edges = int((signed.data < 0).sum())
    stats = {
        "unknown_nt_neuron_fraction": unknown_neuron_fraction,
        "unknown_nt_edge_fraction": unknown_edge_fraction,
        "unknown_nt_default": params.unknown_nt,
        "inhibitory_edge_fraction": inhibitory_edges / signed.nnz if signed.nnz else 0.0,
        "n_inhibitory_edges": inhibitory_edges,
        "transmitter_counts": {
            str(k): int(v) for k, v in zip(*np.unique(labels.astype(str), return_counts=True))
        },
    }
    return signed, stats


def soma_positions(
    rows: np.ndarray, annotations_path: Path = vocab.DEFAULT_ANNOTATIONS_PATH
) -> np.ndarray:
    """Soma coordinates for the selected annotation rows, `float32[N, 3]`.

    NaN where the release annotates no soma — about 11% of the selection, and
    almost all of the lamina. `(0, 0, 0)` would pile those at the origin and
    draw as a structure that is not in the fly.
    """
    column = feather.read_table(annotations_path, columns=[SOMA_COLUMN]).column(SOMA_COLUMN)
    array = column.combine_chunks()
    if isinstance(array, pa.ChunkedArray):
        array = array.chunk(0)

    offsets = np.asarray(array.offsets)
    start = offsets[rows]
    present = np.asarray(array.is_valid())[rows] & (offsets[rows + 1] - start == 3)

    out = np.full((len(rows), 3), np.nan, dtype=np.float32)
    out[present] = np.asarray(array.values)[start[present][:, None] + np.arange(3)]
    return out


def named_populations(registry: CellTypeRegistry, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Local indices for every declared cell type, plus its left/right halves."""
    with registry_mod.CELL_TYPES_PATH.open("rb") as f:
        names = list(tomllib.load(f))

    local = {row: i for i, row in enumerate(rows.tolist())}
    populations: dict[str, np.ndarray] = {}
    for name in names:
        for key, side in [
            (name, None),
            (f"{name}{POPULATION_SEPARATOR}L", "L"),
            (f"{name}{POPULATION_SEPARATOR}R", "R"),
        ]:
            members = [local[r] for r in registry.population(name, side).tolist() if r in local]
            if members or side is None:
                populations[key] = np.array(sorted(members), dtype=np.int64)
    return populations


def file_provenance() -> dict[str, dict[str, str]]:
    with CHECKSUMS_PATH.open() as f:
        checksums = json.load(f)
    return {name: {"url": BASE_URL + name, "sha256": sha} for name, sha in checksums.items()}


def build(
    out: Path,
    selection_params: select_mod.SelectionParams,
    build_params: BuildParams,
    annotations_path: Path = vocab.DEFAULT_ANNOTATIONS_PATH,
    weights_path: Path = select_mod.DEFAULT_WEIGHTS_PATH,
    nt_path: Path = DEFAULT_NT_PATH,
) -> dict[str, object]:
    table: pa.Table = feather.read_table(annotations_path, columns=list(vocab.REQUIRED_COLUMNS))
    vocab.verify(table)
    reg = CellTypeRegistry(table)
    reg.require(*reg.required_types())

    all_bodies = np.asarray(table.column("bodyId").to_pylist(), dtype=np.int64)
    graph = select_mod.load_annotated_graph(all_bodies, selection_params.min_syn, weights_path)
    selection = select_mod.select(table, reg, graph, selection_params)

    labels, resolved_by = neurotransmitter_labels(selection.body_ids, nt_path)
    signed, sign_stats = sign_weights(selection.weights, labels, build_params)

    populations = named_populations(reg, selection.rows)
    somas = soma_positions(selection.rows, annotations_path)
    soma_coverage = float(np.isfinite(somas[:, 0]).mean())
    log.info("soma coordinates for %.1f%% of the selection", 100 * soma_coverage)
    provenance = {
        "dataset": "MaleCNS v1.0 (CC-BY), minconf 0.5",
        "sources": file_provenance(),
        "selection": selection_params.as_dict() | {"achieved_min_syn": selection.achieved_min_syn},
        "achieved": selection.stats | sign_stats,
        "nt_resolution": resolved_by,
        "soma_position_coverage": soma_coverage,
        "population_body_ids": {
            name: selection.body_ids[idx].tolist() for name, idx in populations.items()
        },
        "population_counts": {name: len(idx) for name, idx in populations.items()},
    }

    # `signed` is pre-major (row = presynaptic), the orientation Dale's law is
    # applied in. The engine needs the presynaptic axis on the columns, so the
    # emitted matrix is its transpose, W[post, pre], in both storage orders.
    emitted = signed.T
    csc = emitted.tocsc()
    csc.sort_indices()
    csr = emitted.tocsr()
    csr.sort_indices()
    arrays = {
        "shape": np.array(csc.shape, dtype=np.int64),
        "body_ids": selection.body_ids,
        "annotation_rows": selection.rows,
        "soma_positions": somas,
        "csc_data": csc.data.astype(np.float32, copy=False),
        "csc_indices": csc.indices,
        "csc_indptr": csc.indptr,
        "csr_data": csr.data.astype(np.float32, copy=False),
        "csr_indices": csr.indices,
        "csr_indptr": csr.indptr,
        "provenance": np.array(json.dumps(provenance, indent=2)),
    }
    for name, idx in populations.items():
        arrays[f"pop{POPULATION_SEPARATOR}{name}"] = idx

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    log.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return provenance


def _print_table(provenance: dict[str, object]) -> None:
    achieved = provenance["achieved"]
    counts = provenance["population_counts"]
    print(f"\n{'population':<16}{'selected':>10}")
    print("-" * 26)
    for name, count in counts.items():
        print(f"{name:<16}{count:>10}")
    print("-" * 26)
    print(f"{'neurons':<16}{achieved['n_neurons']:>10}")
    print(f"{'edges':<16}{achieved['n_edges']:>10}")
    print(f"{'min_syn':<16}{provenance['selection']['achieved_min_syn']:>10}")
    print(f"{'inhibitory':<16}{achieved['inhibitory_edge_fraction']:>10.1%}")
    print(f"{'unknown NT':<16}{achieved['unknown_nt_edge_fraction']:>10.2%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "cache" / "connectome_v1.npz")
    parser.add_argument("--k", type=int, default=select_mod.SelectionParams.k)
    parser.add_argument("--min-syn", type=int, default=select_mod.SelectionParams.min_syn)
    parser.add_argument("--max-edges", type=int, default=select_mod.SelectionParams.max_edges)
    parser.add_argument("--unknown-nt", choices=sorted(UNKNOWN_SIGNS), default="excitatory")
    parser.add_argument("--max-unknown-fraction", type=float, default=0.05)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    provenance = build(
        args.out,
        select_mod.SelectionParams(k=args.k, min_syn=args.min_syn, max_edges=args.max_edges),
        BuildParams(args.unknown_nt, args.max_unknown_fraction),
    )
    _print_table(provenance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
