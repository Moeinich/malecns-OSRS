"""Print the real MaleCNS annotation schema, before anything hardcodes it.

    uv run python -m flybrain.connectome.inspect [--path FILE] [--top N]

The `superclass`/`class`/`subclass` vocabularies are not publicly documented.
This report is what `vocab.py` was frozen from; re-run it after any refetch.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import pyarrow as pa
from pyarrow import feather

from flybrain.connectome import vocab
from flybrain.connectome.registry import CellTypeRegistry

#: Types whose hex coverage and laterality decide where the retina injects.
PROBE_TYPES = ("L1", "L2", "L3", "Tm1", "T4a", "T5a", "R7p", "R8p", "R7y", "R8y")


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _counts(table: pa.Table, column: str, top: int) -> None:
    counts = Counter(table.column(column).to_pylist())
    nulls = counts.pop(None, 0)
    _rule(f"{column}: {len(counts)} distinct, {nulls} null")
    for value, n in counts.most_common(top):
        print(f"  {value:<32} {n:>7}")
    if len(counts) > top:
        print(f"  ... {len(counts) - top} more")


def report(table: pa.Table, top: int = 40) -> None:
    _rule(f"schema: {table.num_rows} rows, {len(table.schema.names)} columns")
    for name, dtype in zip(table.schema.names, table.schema.types):
        print(f"  {name:<24} {dtype}")

    for column in ("superclass", "class", "subclass", "somaSide"):
        _counts(table, column, top)

    types = [t for t in table.column("type").to_pylist() if t]
    prefixes = Counter()
    for name in types:
        head = name[:2] if len(name) > 1 else name
        prefixes[head] += 1
    _rule(f"type prefixes: {len(set(types))} distinct types, {len(types)} annotated cells")
    for prefix, n in prefixes.most_common(top):
        print(f"  {prefix:<8} {n:>7}")

    registry = CellTypeRegistry(table)
    lattice = registry.columns()
    hex1 = [h for h, _ in lattice]
    hex2 = [h for _, h in lattice]
    _rule(f"ommatidial lattice: {len(lattice)} columns")
    print(f"  hex1 {min(hex1)}..{max(hex1)}   hex2 {min(hex2)}..{max(hex2)}")
    print(f"  left eye {len(registry.columns('L'))}   right eye {len(registry.columns('R'))}")

    h1 = table.column("assignedOlHex1").to_pylist()
    soma = table.column("somaSide").to_pylist()
    _rule("hex coverage and somaSide availability")
    print(f"  {'type':<10} {'n':>6} {'hex%':>6}  somaSide")
    for name in PROBE_TYPES:
        rows = registry.population(name)
        if len(rows) == 0:
            print(f"  {name:<10} {'absent':>6}")
            continue
        mapped = sum(1 for i in rows if h1[i] is not None and not math.isnan(h1[i]))
        sides = Counter(soma[i] for i in rows)
        summary = ", ".join(f"{k or 'none'}={v}" for k, v in sorted(sides.items(), key=str))
        print(f"  {name:<10} {len(rows):>6} {100 * mapped // len(rows):>5}%  {summary}")

    _rule("required types")
    for name in registry.required_types():
        rows = registry.population(name)
        left = len(registry.population(name, "L"))
        right = len(registry.population(name, "R"))
        print(
            f"  {name:<10} n={len(rows):<4} L={left} R={right}  bodyIds={registry.body_ids(name)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=vocab.DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    table = feather.read_table(args.path)
    report(table, args.top)

    _rule("vocab.verify")
    try:
        vocab.verify(table)
    except vocab.VocabDriftError as exc:
        print(f"  DRIFT: {exc}")
        raise SystemExit(1) from exc
    print("  ok")


if __name__ == "__main__":
    main()
