"""Discovered MaleCNS v1.0 annotation vocabulary, asserted against the real file.

The `superclass`/`class` vocabularies are not publicly documented, so they are
discovered with `python -m flybrain.connectome.inspect` and frozen here.
`verify()` is the tripwire: upstream drift must fail loudly rather than quietly
resolve a named population to the wrong cells.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

ANNOTATIONS_FILENAME = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
DEFAULT_ANNOTATIONS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / ANNOTATIONS_FILENAME
)

N_ROWS = 211_577

#: Columns the registry and the builder actually read.
REQUIRED_COLUMNS = (
    "bodyId",
    "type",
    "instance",
    "class",
    "subclass",
    "superclass",
    "somaSide",
    "rootSide",
    "assignedOlHex1",
    "assignedOlHex2",
)

#: Every non-null `superclass` value, with its row count in v1.0.
SUPERCLASS_COUNTS = {
    "ol_intrinsic": 89403,
    "cb_intrinsic": 32164,
    "vnc_intrinsic": 13161,
    "visual_projection": 9201,
    "vnc_sensory": 6370,
    "ol_sensory": 6098,
    "cb_sensory": 4868,
    "ascending_neuron": 1846,
    "descending_neuron": 1314,
    "vnc_motor": 708,
    "visual_centrifugal": 563,
    "sensory_ascending": 537,
    "cb_motor": 107,
    "vnc_efferent": 94,
    "cb_endocrine": 72,
    "ENS": 50,
    "vnc_tbc": 38,
    "vnc_sensory_tbc": 36,
    "vnc_endocrine": 22,
    "cb_sensory_tbc": 14,
    "sensory_descending": 12,
    "efferent_ascending": 8,
    "efferent_descending": 4,
    "cb_efferent": 4,
    "visual_projection_tbc": 2,
    "sensory_ascending_tbc": 2,
    "descending_neuron_tbc": 2,
}
SUPERCLASSES = frozenset(SUPERCLASS_COUNTS)

#: Every non-null `class` value. Counts are checked only for the ones we rely on.
CLASSES = frozenset(
    {
        "ALIN",
        "ALLN",
        "ALON",
        "ALPN",
        "CX",
        "DAN",
        "Kenyon_Cell",
        "MBON",
        "SEZPN",
        "chemosensory",
        "gustatory",
        "hygrosensory",
        "mechanosensory",
        "mechanosensory_proprioceptive",
        "mechanosensory_tactile",
        "mechanosensory_tbc",
        "ol_bilateral",
        "olfactory",
        "thermosensory",
        "unknown_sensory",
        "visual",
    }
)

#: `somaSide` values. Laterality is central to the steering readout, so an
#: unknown side value must not be silently treated as "no side".
SOMA_SIDES = frozenset({"L", "R", "M"})

#: `assignedOlHex1/2` resolve to the fly's true ommatidial column count.
N_OL_COLUMNS = 892
HEX1_RANGE = (1, 36)
HEX2_RANGE = (1, 39)

#: The populations whose absence or resized cardinality invalidates the model.
#: Values are exact row counts under boundary-prefix resolution (see registry).
REQUIRED_TYPE_CARDINALITY = {
    "DNp01": 2,
    "DNa02": 2,
    "DNpe017": 2,
    "MDN": 4,
    "DNp09": 2,
}

#: Superclass of the whole motor bus; the decoder reads only from here.
DESCENDING_SUPERCLASS = "descending_neuron"


class VocabDriftError(RuntimeError):
    """The annotations no longer match the frozen vocabulary."""


def _distinct(table: pa.Table, column: str) -> set[str]:
    return {v for v in table.column(column).to_pylist() if v is not None}


def verify(table: pa.Table) -> None:
    """Raise `VocabDriftError` if the annotations drifted from this vocabulary.

    Checks the shape we depend on, not the whole file: required columns exist,
    `superclass`/`class`/`somaSide` introduce no unknown value, the hex lattice
    still has 892 columns in range, and every required type still has exactly
    the cardinality we resolved against.
    """
    from flybrain.connectome.registry import CellTypeRegistry

    missing = [c for c in REQUIRED_COLUMNS if c not in table.schema.names]
    if missing:
        raise VocabDriftError(f"annotations lost required columns: {missing}")

    if table.num_rows != N_ROWS:
        raise VocabDriftError(f"expected {N_ROWS} rows, got {table.num_rows}")

    for column, known in (
        ("superclass", SUPERCLASSES),
        ("class", CLASSES),
        ("somaSide", SOMA_SIDES),
    ):
        unknown = _distinct(table, column) - known
        if unknown:
            raise VocabDriftError(f"unknown {column} values: {sorted(unknown)}")

    registry = CellTypeRegistry(table)
    columns = registry.columns()
    if len(columns) != N_OL_COLUMNS:
        raise VocabDriftError(f"expected {N_OL_COLUMNS} ommatidial columns, got {len(columns)}")
    hex1 = [h1 for h1, _ in columns]
    hex2 = [h2 for _, h2 in columns]
    if (min(hex1), max(hex1)) != HEX1_RANGE or (min(hex2), max(hex2)) != HEX2_RANGE:
        raise VocabDriftError(
            f"hex lattice moved: hex1={min(hex1), max(hex1)} hex2={min(hex2), max(hex2)}"
        )

    for name, expected in REQUIRED_TYPE_CARDINALITY.items():
        found = len(registry.population(name))
        if found != expected:
            raise VocabDriftError(f"type {name!r}: expected {expected} cells, found {found}")
