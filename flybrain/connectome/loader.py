"""Load a compiled connectome — and hand out exactly one weight array.

STDP mutates `Connectome.W.data` in place. FlyBrain's learning was a no-op
because it wrote into a second copy of the graph while a different matrix was
simulated, so this module never materializes a second one: the CSR stored for
analysis is loaded only by an explicit call, and `engine.W is connectome.W`
must hold for the engine built here.

The stored matrix is `W[post, pre]` and is handed to the engine in exactly that
orientation. Nothing here transposes: a flip applied on load is invisible to
every self-consistent test and is how an orientation bug survives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from flybrain.connectome.build import POPULATION_SEPARATOR

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "cache" / "connectome_v1.npz"

NORMALIZATION_MODES = ("none", "full", "capped")

#: Ceiling for `capped`, in summed |synapse count| of incoming weight. On the v1
#: build incoming |weight| has median 231 and mean 571, so 500 leaves roughly the
#: lower two thirds of the population at its own relative drive and pulls down
#: only the hub tail, which runs to 119,578.
DEFAULT_INCOMING_CAP = 500.0


def normalize_incoming(
    W: sp.csc_matrix, mode: str = "none", *, cap: float = DEFAULT_INCOMING_CAP
) -> sp.csc_matrix:
    """`W` with each neuron's total incoming |weight| bounded, as a new matrix.

    Not baked into the `.npz`: raw synapse counts stay on disk and this is a
    load-time choice, so the modes can be A/B'd against one build.

    The stored matrix is `W[post, pre]`, so "incoming to neuron `j`" is *row* `j`
    and the row index of every stored entry is `W.indices`. Bincounting the
    columns instead would normalise by out-degree, which produces a
    plausible-looking matrix that is wrong.

    Scaling a whole row by one positive number leaves every E/I ratio inside that
    row exactly where it was. What changes is that a single global gain then
    means the same thing to a neuron with three inputs and to one with three
    thousand, instead of 3.0 to one and 111.7 to the other.

    `full` sends every row to 1.0. `capped` rescales only rows above `cap`
    (`scale = cap / total`), leaving weakly-innervated cells their relative drive
    rather than amplifying them to parity with the hubs.
    """
    if mode == "none":
        return W
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"normalization mode {mode!r} is not one of {NORMALIZATION_MODES}")
    data = W.data.astype(np.float32, copy=True)
    incoming = np.bincount(W.indices, weights=np.abs(data), minlength=W.shape[0])
    if mode == "full":
        scale = 1.0 / np.maximum(incoming, 1.0)
    else:
        scale = np.minimum(1.0, cap / np.maximum(incoming, 1.0))
    data *= scale[W.indices].astype(np.float32)
    return sp.csc_matrix((data, W.indices.copy(), W.indptr.copy()), shape=W.shape)


@dataclass(frozen=True)
class Connectome:
    #: The one weight array in the process, `W[post, pre]` — column `j` is
    #: neuron `j`'s outgoing synapses. Mutated in place by plasticity.
    W: sp.csc_matrix
    body_ids: np.ndarray
    populations: dict[str, np.ndarray]
    provenance: dict[str, object]
    #: `float32[n, 3]` EM soma coordinates, NaN where the release annotates no
    #: soma. `None` only for a connectome that was not built from the release.
    soma_positions: np.ndarray | None = None

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
            soma_positions=data["soma_positions"],
        )


def load_csr(path: Path | str = DEFAULT_PATH) -> sp.csr_matrix:
    """The analysis copy. Never hand this to the engine — it is a second array."""
    with np.load(Path(path), allow_pickle=False) as data:
        return sp.csr_matrix(
            (data["csr_data"], data["csr_indices"], data["csr_indptr"]),
            shape=tuple(int(v) for v in data["shape"]),
        )
