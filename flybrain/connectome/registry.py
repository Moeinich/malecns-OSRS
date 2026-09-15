"""Name-addressed cell-type registry over the MaleCNS annotations.

Everything downstream addresses populations by name. There is no path by which
a caller obtains an index range for a named cell type, because that is exactly
how a connectome becomes fiction: an index assignment looks like a neuron until
someone checks the bodyId.
"""

from __future__ import annotations

import logging
import math
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
from pyarrow import feather

from flybrain.connectome import vocab

log = logging.getLogger(__name__)

CELL_TYPES_PATH = Path(__file__).resolve().parent / "cell_types.toml"

#: A bare prefix match would fold `Tm12` and `Tm16` into `Tm1`. A match is only
#: an ambiguity of the same type when the remainder starts with a separator.
_BOUNDARY_CHARS = "_-'(."


class MissingCellTypeError(LookupError):
    """A required cell type is absent. Never fall back to an index."""


def _load_cell_types(path: Path = CELL_TYPES_PATH) -> dict[str, dict[str, object]]:
    with path.open("rb") as f:
        return tomllib.load(f)


class CellTypeRegistry:
    """Resolves a cell-type name to row indices and bodyIds in the annotations."""

    def __init__(self, table: pa.Table, cell_types: dict[str, dict[str, object]] | None = None):
        self._table = table
        self._spec = _load_cell_types() if cell_types is None else cell_types

        self._body_ids = np.asarray(table.column("bodyId").to_pylist(), dtype=np.int64)
        self._types = table.column("type").to_pylist()
        self._instances = table.column("instance").to_pylist()
        self._soma_side = table.column("somaSide").to_pylist()
        self._root_side = table.column("rootSide").to_pylist()

        self._by_type: dict[str, list[int]] = defaultdict(list)
        for i, name in enumerate(self._types):
            if name is not None:
                self._by_type[name].append(i)

        self._sides = self._resolve_sides()
        self._cache: dict[str, np.ndarray] = {}

    @classmethod
    def from_feather(cls, path: Path | str | None = None) -> CellTypeRegistry:
        path = Path(path) if path is not None else vocab.DEFAULT_ANNOTATIONS_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"annotations not found at {path}; run tools/fetch_connectome.py first"
            )
        return cls(feather.read_table(path, columns=list(vocab.REQUIRED_COLUMNS)))

    def _resolve_sides(self) -> list[str | None]:
        """`somaSide` where present, else a logged heuristic on the instance suffix.

        Photoreceptors carry no `somaSide` at all, so a left/right readout built
        naively on that column would silently address half a population.
        """
        sides: list[str | None] = []
        heuristic = 0
        for i, side in enumerate(self._soma_side):
            if side is not None:
                sides.append(side)
                continue
            instance = self._instances[i] or ""
            guess = None
            if instance.endswith(("_L", "_R")):
                guess = instance[-1]
            elif self._root_side[i] in vocab.SOMA_SIDES:
                guess = self._root_side[i]
            if guess is not None:
                heuristic += 1
            sides.append(guess)
        if heuristic:
            log.info("somaSide missing for %d cells; inferred from instance/rootSide", heuristic)
        return sides

    def _match_rows(self, name: str) -> np.ndarray:
        """Row indices for `name`, widening over subtype ambiguity by prefix.

        `DNa02` and `DNa02_a` are the same population split by an annotator, so
        resolution widens into the larger population and logs every constituent.
        """
        spec = self._spec.get(name, {})
        mode = spec.get("match", "boundary")

        if mode == "exact":
            members = [name] if name in self._by_type else []
        elif mode == "prefix":
            members = sorted(k for k in self._by_type if k.startswith(name))
        else:
            members = sorted(
                k
                for k in self._by_type
                if k == name or (k.startswith(name) and k[len(name)] in _BOUNDARY_CHARS)
            )

        rows = np.array(
            sorted(i for k in members for i in self._by_type[k]),
            dtype=np.int64,
        )
        if len(members) > 1:
            if mode == "prefix":
                log.info("%s resolved over family %s -> %d cells", name, members, len(rows))
            else:
                log.info(
                    "%s widened by prefix over %s -> %d cells: bodyIds %s",
                    name,
                    members,
                    len(rows),
                    self._body_ids[rows].tolist(),
                )

        expected = spec.get("expect")
        if expected is not None and len(rows) != expected:
            log.warning(
                "%s: implausible cardinality, expected %s from cell_types.toml, found %d",
                name,
                expected,
                len(rows),
            )
        return rows

    def population(self, type_name: str, side: str | None = None) -> np.ndarray:
        """Row indices for `type_name`, optionally restricted to one soma side."""
        if type_name not in self._cache:
            self._cache[type_name] = self._match_rows(type_name)
        rows = self._cache[type_name]

        if len(rows) == 0 and self._spec.get(type_name, {}).get("required"):
            raise MissingCellTypeError(
                f"required cell type {type_name!r} is absent from the annotations"
            )
        if side is None:
            return rows

        side = side[0].upper()
        if side not in vocab.SOMA_SIDES:
            raise ValueError(f"unknown side {side!r}, expected one of {sorted(vocab.SOMA_SIDES)}")
        return rows[np.array([self._sides[i] == side for i in rows], dtype=bool)]

    def body_ids(self, type_name: str, side: str | None = None) -> np.ndarray:
        """MaleCNS bodyIds for `type_name` — paste these into neuprint to audit us."""
        return self._body_ids[self.population(type_name, side)]

    def require(self, *names: str) -> None:
        """Assert every named type resolves. A missing one is a hard error."""
        missing = {}
        for name in names:
            rows = self.population(name)
            if len(rows) == 0:
                missing[name] = "no cells"
            else:
                expected = self._spec.get(name, {}).get("expect")
                if expected is not None and len(rows) != expected:
                    missing[name] = f"{len(rows)} cells, expected {expected}"
        if missing:
            raise MissingCellTypeError(f"cannot resolve required cell types: {missing}")

    def required_types(self) -> list[str]:
        return [n for n, s in self._spec.items() if s.get("required")]

    def columns(self, side: str | None = None) -> dict[tuple[int, int], np.ndarray]:
        """Ommatidial lattice: `(assignedOlHex1, assignedOlHex2) -> row indices`.

        Hex coordinates are per-eye local, so pass `side` to disambiguate a
        column; without it the two eyes share a key.
        """
        hex1 = self._table.column("assignedOlHex1").to_pylist()
        hex2 = self._table.column("assignedOlHex2").to_pylist()
        lattice: dict[tuple[int, int], list[int]] = defaultdict(list)
        want = side[0].upper() if side else None
        for i, (a, b) in enumerate(zip(hex1, hex2)):
            if a is None or b is None or math.isnan(a) or math.isnan(b):
                continue
            if want is not None and self._sides[i] != want:
                continue
            lattice[(int(a), int(b))].append(i)
        return {k: np.array(v, dtype=np.int64) for k, v in sorted(lattice.items())}
