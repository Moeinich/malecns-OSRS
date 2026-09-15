"""Load a compiled connectome — and hand out exactly one weight array.

STDP mutates `Connectome.W.data` in place. FlyBrain's learning was a no-op
because it wrote into a second copy of the graph while a different matrix was
simulated, so this module never materializes a second one: the CSR stored for
analysis is loaded only by an explicit call, and `engine.W is connectome.W`
must hold for the engine built here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from flybrain.connectome.build import POPULATION_SEPARATOR

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "cache" / "connectome_v1.npz"


@dataclass(frozen=True)
class Connectome:
    #: The one weight array in the process. Mutated in place by plasticity.
    W: sp.csc_matrix
    body_ids: np.ndarray
    populations: dict[str, np.ndarray]
    provenance: dict[str, object]

    @property
    def n(self) -> int:
        return self.W.shape[0]

    def population(self, name: str, side: str | None = None) -> np.ndarray:
        key = f"{name}{POPULATION_SEPARATOR}{side[0].upper()}" if side else name
        if key not in self.populations:
            raise KeyError(f"population {key!r} is not in this build")
        return self.populations[key]

    def body_ids_of(self, name: str, side: str | None = None) -> np.ndarray:
        """The bodyIds behind a named population — paste them into neuprint."""
        return self.body_ids[self.population(name, side)]


def load(path: Path | str = DEFAULT_PATH) -> Connectome:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no connectome at {path}; run python -m flybrain.connectome.build")
    with np.load(path, allow_pickle=False) as data:
        shape = tuple(int(v) for v in data["shape"])
        W = sp.csc_matrix(
            (
                data["csc_data"].astype(np.float32, copy=False),
                data["csc_indices"],
                data["csc_indptr"],
            ),
            shape=shape,
        )
        populations = {
            key.split(POPULATION_SEPARATOR, 1)[1]: data[key]
            for key in data.files
            if key.startswith(f"pop{POPULATION_SEPARATOR}")
        }
        return Connectome(
            W=W,
            body_ids=data["body_ids"],
            populations=populations,
            provenance=json.loads(str(data["provenance"])),
        )


def load_csr(path: Path | str = DEFAULT_PATH) -> sp.csr_matrix:
    """The analysis copy. Never hand this to the engine — it is a second array."""
    with np.load(Path(path), allow_pickle=False) as data:
        return sp.csr_matrix(
            (data["csr_data"], data["csr_indices"], data["csr_indptr"]),
            shape=tuple(int(v) for v in data["shape"]),
        )
