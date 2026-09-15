"""The calibration artifact: what the search found, persisted so the brain can run it.

`flybrain.engine.calibrate` finds a gain; until this module existed nothing read
the answer back, so the live brain ran the raw connectome at an implicit gain of
1.0 and was silent. A calibration is therefore stored *next to* the connectome,
not inside it: a tuning result must not require rebuilding from a gigabyte of
feather files, and the same connectome can carry several.

The fingerprint is the point. A gain is only meaningful for the network it was
measured on, so `n`, `nnz` and the summed weight — enough to catch a rebuilt,
re-signed or ablated matrix — travel with it, and a mismatch is an error rather
than a quietly wrong simulation. A *missing* artifact is not an error: dev runs
and the ablation harness have to work without one. It is loud instead.

`apply` returns a new matrix. The loader's "one weight array in the process"
invariant means the cached `Connectome.W` is the array plasticity mutates; the
live path scales the copy `Ablation.apply` already made, never that one.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from flybrain.connectome.loader import DEFAULT_PATH, Connectome
from flybrain.engine.calibrate import Acceptance, RateSummary

DEFAULT_CALIBRATION_PATH = DEFAULT_PATH.with_name("calibration_v1.json")

FORMAT_VERSION = 1

UNCALIBRATED_BANNER = (
    "\n"
    "!!! UNCALIBRATED: running at gain 1.0, the brain will be silent !!!\n"
    f"    no calibration at {DEFAULT_CALIBRATION_PATH}\n"
    "    python -m flybrain.engine.calibrate finds the gain; --calibration points here\n"
)


class StaleCalibration(RuntimeError):
    """The artifact was measured on a different network than the one loaded."""


@dataclass(frozen=True)
class Fingerprint:
    """Cheap and sufficient: a rebuild, a re-sign or an ablation all move one of these."""

    n: int
    nnz: int
    weight_sum: float
    dataset: str = ""

    @classmethod
    def of(cls, connectome: Connectome) -> Fingerprint:
        W = connectome.W
        return cls(
            n=int(W.shape[0]),
            nnz=int(W.nnz),
            weight_sum=float(W.data.sum()),
            dataset=str(connectome.provenance.get("dataset", "")),
        )

    def describe(self) -> str:
        return f"n={self.n} nnz={self.nnz} sum={self.weight_sum:.6g} {self.dataset}"


@dataclass(frozen=True)
class Calibration:
    """One accepted tuning, and everything needed to reproduce or refute it."""

    gain: float
    spontaneous_noise_std: float | None = None
    i_max: float | None = None
    #: `LIFEngine`'s adaptation pair. `None` means "leave the engine's own default".
    b: float | None = None
    tau_w: float | None = None
    acceptance: Acceptance | None = None
    rates: RateSummary | None = None
    measure_steps: int | None = None
    #: What the network was driven with while this was measured.
    drive: str = ""
    connectome: Fingerprint | None = None
    git_rev: str = field(default_factory=lambda: git_rev())
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def calibrated(self) -> bool:
        return self.connectome is not None

    def apply(self, W: sp.csc_matrix) -> sp.csc_matrix:
        """`W` scaled by the gain, as a new matrix. The input is never touched."""
        return sp.csc_matrix(
            (W.data.astype(np.float32) * np.float32(self.gain), W.indices, W.indptr),
            shape=W.shape,
        )

    def engine_kwargs(self) -> dict[str, float]:
        """The `LIFEngine` arguments this calibration pins. Unset fields stay unset."""
        pairs = (
            ("spontaneous_noise_std", self.spontaneous_noise_std),
            ("b", self.b),
            ("tau_w", self.tau_w),
        )
        return {k: float(v) for k, v in pairs if v is not None}

    def encode_params(self) -> Any:
        """`EncodeParams` carrying the calibrated peak current."""
        # Imported here: `flybrain.engine` must not depend on `flybrain.sensory`.
        from flybrain.sensory.encode import EncodeParams

        params = EncodeParams()
        return params if self.i_max is None else replace(params, i_max=float(self.i_max))

    def describe(self) -> str:
        if not self.calibrated:
            return "UNCALIBRATED — gain 1.0, no band, no measured rate"
        lo, hi = (self.acceptance or Acceptance()).target_hz
        rate = f"{self.rates.mean_hz:.2f} Hz" if self.rates is not None else "n/a"
        return f"gain {self.gain:.6g}  band {lo}-{hi} Hz  measured {rate}"

    def save(self, path: Path | str = DEFAULT_CALIBRATION_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_encode(self), indent=2) + "\n")
        return path


#: What a run without an artifact is: the raw connectome, and the engine's own defaults.
UNCALIBRATED = Calibration(gain=1.0, git_rev="", created="")


def git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parent,
        )
    except OSError:
        return "unknown"
    return out.stdout.strip() or "unknown"


def load(
    path: Path | str = DEFAULT_CALIBRATION_PATH,
    connectome: Connectome | None = None,
) -> Calibration | None:
    """The stored calibration, or `None` if there is none. Stale raises."""
    path = Path(path)
    if not path.exists():
        return None
    c = _decode(json.loads(path.read_text()))
    if connectome is not None and c.connectome is not None:
        actual = Fingerprint.of(connectome)
        if actual != c.connectome:
            raise StaleCalibration(
                f"{path} was measured on a different network:\n"
                f"  calibrated  {c.connectome.describe()}\n"
                f"  loaded      {actual.describe()}\n"
                "re-run python -m flybrain.engine.calibrate against this build"
            )
    return c


def _encode(c: Calibration) -> dict[str, Any]:
    return {
        "format": FORMAT_VERSION,
        "gain": c.gain,
        "spontaneous_noise_std": c.spontaneous_noise_std,
        "i_max": c.i_max,
        "b": c.b,
        "tau_w": c.tau_w,
        "acceptance": None if c.acceptance is None else _acceptance_json(c.acceptance),
        "rates": None if c.rates is None else _rates_json(c.rates),
        "measure_steps": c.measure_steps,
        "drive": c.drive,
        "connectome": None if c.connectome is None else vars(c.connectome),
        "git_rev": c.git_rev,
        "created": c.created,
    }


def _decode(d: dict[str, Any]) -> Calibration:
    if int(d.get("format", 0)) != FORMAT_VERSION:
        raise ValueError(f"calibration format {d.get('format')!r}, expected {FORMAT_VERSION}")
    acceptance, rates, fp = d.get("acceptance"), d.get("rates"), d.get("connectome")
    return Calibration(
        gain=float(d["gain"]),
        spontaneous_noise_std=d.get("spontaneous_noise_std"),
        i_max=d.get("i_max"),
        b=d.get("b"),
        tau_w=d.get("tau_w"),
        acceptance=None if acceptance is None else _acceptance_from(acceptance),
        rates=None if rates is None else _rates_from(rates),
        measure_steps=d.get("measure_steps"),
        drive=d.get("drive", ""),
        connectome=None if fp is None else Fingerprint(**fp),
        git_rev=d.get("git_rev", ""),
        created=d.get("created", ""),
    )


def _acceptance_json(a: Acceptance) -> dict[str, Any]:
    return vars(a) | {"target_hz": list(a.target_hz)}


def _acceptance_from(d: dict[str, Any]) -> Acceptance:
    return Acceptance(**(d | {"target_hz": tuple(d["target_hz"])}))


def _rates_json(r: RateSummary) -> dict[str, Any]:
    return vars(r) | {
        "percentiles_hz": {str(p): v for p, v in r.percentiles_hz.items()},
        "histogram": r.histogram.tolist(),
        "bin_edges_hz": r.bin_edges_hz.tolist(),
    }


def _rates_from(d: dict[str, Any]) -> RateSummary:
    return RateSummary(
        **(
            d
            | {
                "percentiles_hz": {int(p): v for p, v in d["percentiles_hz"].items()},
                "histogram": np.asarray(d["histogram"]),
                "bin_edges_hz": np.asarray(d["bin_edges_hz"]),
            }
        )
    )


__all__ = [
    "DEFAULT_CALIBRATION_PATH",
    "UNCALIBRATED",
    "UNCALIBRATED_BANNER",
    "Calibration",
    "Fingerprint",
    "StaleCalibration",
    "git_rev",
    "load",
]
