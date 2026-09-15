"""Subgraph selection over the full MaleCNS connectome.

The whole graph is 152 M edges over 1.8 M bodies. Over the *annotated* bodies
at `min_syn=3` it is 10.7 M edges over 184,526 neurons, which does fit the tick
budget, so the default is to keep all of it: `k=None` selects every connected
annotated neuron and no PPR runs. Selecting everything is not a heuristic, and
it is the only selection that cannot be wrong.

A smaller `k` still cuts by *connectivity*, not by hand: a neuron is kept when
it is both reachable from the visual input population and able to influence a
descending neuron. Anchors we depend on by name are forced in regardless of
score either way, because a network missing DNa02 is not a smaller model of the
fly, it is a different one.

Measured over the whole annotated graph, `min_syn` is not what starves the
median neuron -- the net-inhibitory fraction is 32.5% at `min_syn=1` and 33.5%
at `min_syn=5`, flat. The top-K cut was: it left the 55,599 kept neurons with
1.73 M of their 6.30 M edges, median net input 3.0 against 15.0 whole-brain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import scipy.sparse as sp
from pyarrow import ipc
from scipy.sparse.csgraph import connected_components

from flybrain.connectome import vocab
from flybrain.connectome.registry import CellTypeRegistry

log = logging.getLogger(__name__)

WEIGHTS_FILENAME = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
DEFAULT_WEIGHTS_PATH = vocab.DEFAULT_ANNOTATIONS_PATH.parent / WEIGHTS_FILENAME

#: Lamina/medulla injection sites — the retina drives these, so forward
#: reachability is measured from here and not from the photoreceptors (which
#: carry no hex assignment and are never injected into; see the plan).
VISUAL_INPUT_TYPES = ("L1", "L2", "L3", "Tm1")

#: The elementary motion detectors, all four preferred directions each —
#: direction selectivity is the entire point of rendering sub-frames per tick.
#: They lose the top-K cut among 89,403 ol_intrinsic neurons, and a network
#: without them computes no optic flow at all.
MOTION_DETECTOR_TYPES = ("T4", "T5")

#: The columnar medulla layer between the injection sites and the motion
#: detectors. Mi1 alone is not enough: it carries only 1,818 of L1's 16,023
#: out-edges, and T4 needs a fast input *and* a delayed one to be direction
#: selective. Mi1/Tm3 are the fast leg, Mi4/Mi9/C3 the delayed and inhibitory
#: one, CT1 the wide-field partner; L5 and C2 are here because they take 3,686
#: further L1 out-edges, which is what lifts L1 off the floor. None of them
#: survive the top-K PPR cut (Tm3 scores 2.57e-06 against a cut of 6.50e-06),
#: so each has to be forced in by name.
ON_RELAY_TYPES = ("Mi1", "Tm3", "Mi4", "Mi9", "C2", "C3", "L5", "CT1")

#: Named populations forced into the subgraph. The mushroom body is where
#: learning lives; the central complex is the heading compass.
ANCHOR_TYPES = ("KC", "MBON", "PPL1", "PAM", "EPG", "PEN", "PEG", "FB")

#: Everything the sensorimotor loop addresses by name must be present by
#: construction, not by score: the injection layer the retina writes into, the
#: motion detectors it drives, and the named circuits above.
FORCED_TYPES = VISUAL_INPUT_TYPES + ON_RELAY_TYPES + MOTION_DETECTOR_TYPES + ANCHOR_TYPES


@dataclass(frozen=True)
class SelectionParams:
    #: `None` keeps every connected annotated neuron and skips the PPR score.
    k: int | None = None
    min_syn: int = 3
    max_edges: int = 12_000_000
    max_min_syn: int = 64
    source_types: tuple[str, ...] = VISUAL_INPUT_TYPES
    anchor_types: tuple[str, ...] = FORCED_TYPES
    #: Resolved from the `superclass` column: the whole 1,314-cell motor bus.
    anchor_superclasses: tuple[str, ...] = (vocab.DESCENDING_SUPERCLASS,)
    #: Resolved from the `class` column. This is what carries the central
    #: complex: 2,950 cells, of which the `FB` prefix names only 575. The other
    #: 27 `FB*` cells carry no `class` at all, so both anchors are needed.
    anchor_classes: tuple[str, ...] = ("CX",)
    ppr_alpha: float = 0.85
    ppr_iters: int = 40

    def as_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "min_syn": self.min_syn,
            "max_edges": self.max_edges,
            "source_types": list(self.source_types),
            "anchor_types": list(self.anchor_types),
            "anchor_superclasses": list(self.anchor_superclasses),
            "anchor_classes": list(self.anchor_classes),
            "ppr_alpha": self.ppr_alpha,
            "ppr_iters": self.ppr_iters,
        }


@dataclass(frozen=True)
class Selection:
    """The chosen subgraph, in local index space."""

    #: Annotation row indices of the kept neurons, ascending. Local index `i`
    #: is `rows[i]`; nothing downstream may address a neuron any other way.
    rows: np.ndarray
    body_ids: np.ndarray
    #: Unsigned synapse counts, local x local, row = presynaptic.
    weights: sp.csr_matrix
    achieved_min_syn: int
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.rows)


def load_annotated_graph(
    body_ids: np.ndarray,
    min_syn: int,
    weights_path: Path | str = DEFAULT_WEIGHTS_PATH,
) -> sp.csr_matrix:
    """Stream the weights table into a CSR graph over the annotated bodies only.

    The file is 151.9 M rows and Feather v2 is LZ4-compressed, so memory-mapping
    saves nothing and a full materialization is ~3.6 GB. Each record batch is
    filtered down to annotated endpoints above `min_syn` before anything is kept.
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"weights not found at {path}; run tools/fetch_connectome.py")

    order = np.argsort(body_ids, kind="stable")
    sorted_bodies = body_ids[order]
    if len(np.unique(sorted_bodies)) != len(sorted_bodies):
        raise ValueError("duplicate bodyIds in the annotations; cannot build a body -> row map")

    pre_parts: list[np.ndarray] = []
    post_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    total_rows = 0

    with ipc.open_file(str(path)) as reader:
        _check_schema(reader.schema)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            total_rows += batch.num_rows
            weight = batch.column("weight").to_numpy(zero_copy_only=False)
            keep = weight >= min_syn
            if not keep.any():
                continue
            pre = map_bodies(
                batch.column("body_pre").to_numpy(zero_copy_only=False)[keep], sorted_bodies
            )
            post = map_bodies(
                batch.column("body_post").to_numpy(zero_copy_only=False)[keep], sorted_bodies
            )
            both = (pre >= 0) & (post >= 0)
            if not both.any():
                continue
            pre_parts.append(order[pre[both]])
            post_parts.append(order[post[both]])
            weight_parts.append(weight[keep][both].astype(np.float32))

    n = len(body_ids)
    if not pre_parts:
        return sp.csr_matrix((n, n), dtype=np.float32)

    graph = sp.coo_matrix(
        (
            np.concatenate(weight_parts),
            (np.concatenate(pre_parts), np.concatenate(post_parts)),
        ),
        shape=(n, n),
        dtype=np.float32,
    ).tocsr()
    log.info(
        "streamed %d rows -> %d edges over %d annotated bodies at min_syn=%d",
        total_rows,
        graph.nnz,
        n,
        min_syn,
    )
    return graph


def _check_schema(schema: pa.Schema) -> None:
    expected = {"body_pre": "int64", "body_post": "int64", "weight": "int64"}
    actual = {name: str(schema.field(name).type) for name in schema.names if name in expected}
    if actual != expected:
        raise ValueError(f"weights schema drifted: expected {expected}, found {dict(actual)}")


def map_bodies(values: np.ndarray, sorted_bodies: np.ndarray) -> np.ndarray:
    """Position of each value in `sorted_bodies`, or -1 when absent."""
    idx = np.searchsorted(sorted_bodies, values)
    np.clip(idx, 0, len(sorted_bodies) - 1, out=idx)
    return np.where(sorted_bodies[idx] == values, idx, -1)


def _ppr(graph: sp.csr_matrix, seed: np.ndarray, alpha: float, iters: int) -> np.ndarray:
    """Personalized PageRank restarting at `seed`, following edge direction."""
    if seed.sum() <= 0:
        raise ValueError("empty PPR seed")
    seed = seed / seed.sum()
    out_degree = np.asarray(graph.sum(axis=1)).ravel()
    nonzero = out_degree > 0
    scale = np.zeros_like(out_degree)
    scale[nonzero] = 1.0 / out_degree[nonzero]
    transposed = graph.T.tocsr()

    x = seed.copy()
    for _ in range(iters):
        dangling = float(x[~nonzero].sum())
        x = alpha * (transposed @ (x * scale)) + (alpha * dangling + 1.0 - alpha) * seed
    return x


def _indicator(n: int, rows: np.ndarray) -> np.ndarray:
    vector = np.zeros(n, dtype=np.float64)
    vector[rows] = 1.0
    return vector


def _rows_where(table: pa.Table, column: str, values: tuple[str, ...]) -> np.ndarray:
    if not values:
        return np.empty(0, dtype=np.int64)
    entries = table.column(column).to_pylist()
    wanted = set(values)
    return np.array([i for i, v in enumerate(entries) if v in wanted], dtype=np.int64)


def select(
    table: pa.Table,
    registry: CellTypeRegistry,
    graph: sp.csr_matrix,
    params: SelectionParams | None = None,
) -> Selection:
    """Score by bidirectional PPR, force in the anchors, prune, keep the DN component."""
    params = params or SelectionParams()
    n = graph.shape[0]
    connected = np.diff(graph.indptr) > 0
    connected[graph.indices] = True
    descending = _rows_where(table, "superclass", params.anchor_superclasses)
    if descending.size == 0:
        raise ValueError(f"no rows in superclass {params.anchor_superclasses}")

    all_bodies = np.asarray(table.column("bodyId").to_pylist(), dtype=np.int64)
    sources = np.unique(np.concatenate([registry.population(t) for t in params.source_types]))
    if sources.size == 0:
        raise ValueError(f"visual input types {params.source_types} resolved to nothing")

    if params.k is None:
        top = np.flatnonzero(connected)
    else:
        forward = _ppr(graph, _indicator(n, sources), params.ppr_alpha, params.ppr_iters)
        backward = _ppr(
            graph.T.tocsr(), _indicator(n, descending), params.ppr_alpha, params.ppr_iters
        )
        score = np.sqrt(forward * backward)
        top = np.argsort(score, kind="stable")[::-1][: params.k]
        top = top[score[top] > 0]

    anchors = [descending, _rows_where(table, "class", params.anchor_classes)]
    for name in params.anchor_types:
        rows = registry.population(name)
        if rows.size == 0:
            log.warning("anchor type %r does not resolve in the annotations; skipping", name)
            continue
        anchors.append(rows)
    forced = np.unique(np.concatenate(anchors))

    kept = np.unique(np.concatenate([top, forced]))
    log.info(
        "selected %d neurons: %d by %s, %d forced anchors",
        kept.size,
        top.size,
        "connectivity (no top-K cut)" if params.k is None else f"score (top-{params.k})",
        forced.size,
    )

    min_syn = params.min_syn
    sub = graph[kept][:, kept]
    while sub.nnz > params.max_edges and min_syn < params.max_min_syn:
        min_syn += 1
        sub.data[sub.data < min_syn] = 0.0
        sub.eliminate_zeros()
        log.info("pruned to min_syn=%d -> %d edges", min_syn, sub.nnz)
    if sub.nnz > params.max_edges:
        raise RuntimeError(
            f"{sub.nnz} edges still over budget {params.max_edges} at min_syn={min_syn}"
        )

    kept, sub = _largest_component_with(kept, sub, descending)

    rows = np.sort(kept)
    order = np.argsort(kept, kind="stable")
    sub = sub[order][:, order].tocsr()
    sub.sort_indices()

    stats = {
        "n_scored": int(top.size),
        "n_forced": int(forced.size),
        "n_neurons": int(rows.size),
        "n_edges": int(sub.nnz),
        "n_source_neurons": int(sources.size),
        "n_descending_selected": int(np.isin(rows, descending).sum()),
    }
    return Selection(
        rows=rows,
        body_ids=all_bodies[rows],
        weights=sub,
        achieved_min_syn=min_syn,
        stats=stats,
    )


def _largest_component_with(
    kept: np.ndarray, sub: sp.csr_matrix, descending: np.ndarray
) -> tuple[np.ndarray, sp.csr_matrix]:
    count, labels = connected_components(sub, directed=True, connection="weak")
    if count == 1:
        return kept, sub
    is_dn = np.isin(kept, descending)
    dn_per_label = np.bincount(labels[is_dn], minlength=count)
    winner = int(np.argmax(dn_per_label))
    mask = labels == winner
    log.info(
        "weak components: %d; keeping the one holding %d/%d descending neurons (%d neurons)",
        count,
        int(dn_per_label[winner]),
        int(is_dn.sum()),
        int(mask.sum()),
    )
    keep_idx = np.flatnonzero(mask)
    return kept[keep_idx], sub[keep_idx][:, keep_idx]
