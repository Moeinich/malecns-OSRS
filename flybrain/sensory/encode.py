"""Retina raster -> dense injection current over the whole network.

The raster is hex-binned down onto the real ommatidial lattice (892 columns per
eye, from `assignedOlHex1/2`) and injected at the lamina/medulla injection layer
(L1/L2/L3/Tm1/Mi1), never at photoreceptors: photoreceptors carry no hex
assignment in MaleCNS v1.0, and L1->Mi1->T4 / L2->Tm1->T5 means motion still has
to emerge through real circuitry. Nothing here computes flow.

**Chromatic injection is not retinotopic in v1, and this is a known limitation.**
The chromatic path proper (R7/R8, Dm8, Tm5) is either absent from this build or
carries no usable hex assignment (`R8p`/`R8y` are `somaSide == 'L'` only), so the
threat/loot/resource channels are injected as a per-eye scalar drive over the
injection-layer cells that have *no* hex assignment — the cells that cannot be
placed retinotopically anyway — partitioned into three disjoint slabs so the
channels stay distinguishable. The slab boundaries are arbitrary; only the
luminance pathway is anatomically placed.

Nothing here may import from `flybrain.motor`, and nothing here knows that
actions exist — see AGENTS.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flybrain.connectome.loader import Connectome
from flybrain.connectome.registry import CellTypeRegistry
from flybrain.sensory.retina import (
    CH_LOOT,
    CH_LUMINANCE,
    CH_RESOURCE,
    CH_THREAT,
    DEFAULT_PX_PER_TILE,
)

#: Achromatic injection layer. L1/L2/L3/Tm1 are the plan's targets; Mi1 is the
#: ON-pathway medulla cell sitting between L1 and T4 and is hex-mapped too.
LUMINANCE_TYPES = ("L1", "L2", "L3", "Tm1", "Mi1")

#: Non-retinotopic channels, in slab order. See the module docstring.
CHROMATIC_CHANNELS = (CH_THREAT, CH_LOOT, CH_RESOURCE)

SIDES = ("L", "R")

#: The two zones each chromatic channel is reduced over, in slab order.
ZONES = ("fovea", "periphery")

#: Half-angle of the gaze wedge the body aims discrete acts through. Restated
#: here rather than imported: `flybrain.sensory` may not see `flybrain.motor`,
#: and `motor/body.py` derives the same number from this retina's geometry.
FOVEA_HALF_ANGLE = math.atan2(3.5, 10.0)

#: Pixels one tile covers. A channel's level is its painted mass over this, so a
#: single tile-sized target reads as the intensity `retina._threat_intensity`
#: painted it with (0.2-1.0) — which straddles `sigma`, the responsive part of
#: the Naka-Rushton curve. Averaging over the whole ~1,800 px window instead put
#: one NPC at 1.5e-4, six orders under the luminance drive and under the noise
#: floor, so no spike time changed and the channel carried nothing.
TILE_AREA_PX = DEFAULT_PX_PER_TILE**2


@dataclass(frozen=True)
class EncodeParams:
    #: Peak injected current, in the LIF's units. This is a *sustained* current and
    #: `v_ss = v_rest + I`, so it has to clear `v_thresh - v_rest = 15` on its own:
    #: at the old 6.0 a maximally driven column cell sat 9 units below threshold
    #: permanently, at any synaptic gain, and the brain could not fire at all.
    #: 24.0 puts a saturated column ~9 units above it — the I=16 regime `tests/
    #: test_lif.py` already exercises. `test_encode.py` pins the relationship.
    i_max: float = 24.0
    #: Naka-Rushton exponent and half-saturation point.
    exponent: float = 2.0
    sigma: float = 0.25
    luminance_gain: float = 1.0
    threat_gain: float = 1.0
    loot_gain: float = 0.5
    resource_gain: float = 0.5
    #: Fraction of the raster width the two eyes share, binocularly.
    eye_overlap: float = 0.2
    annotations_path: Path | None = None

    def chromatic_gain(self, channel: int) -> float:
        return {
            CH_THREAT: self.threat_gain,
            CH_LOOT: self.loot_gain,
            CH_RESOURCE: self.resource_gain,
        }[channel]


@dataclass(frozen=True)
class _Eye:
    x0: int
    x1: int
    #: Flat raster indices of this eye's window, and the column each falls in.
    pixels: np.ndarray
    labels: np.ndarray
    counts: np.ndarray
    #: Columns no pixel was assigned to, and the nearest pixel to each.
    starved: np.ndarray
    starved_pixel: np.ndarray
    #: Network indices of the column cells, and the column each belongs to.
    neurons: np.ndarray
    column_of: np.ndarray
    #: Pixels of this eye's window inside the gaze wedge, and the rest.
    zones: tuple[np.ndarray, ...]
    #: One disjoint slab of non-retinotopic cells per chromatic channel per zone.
    chromatic: tuple[tuple[np.ndarray, ...], ...]


@dataclass(frozen=True)
class _Injection:
    n: int
    size: int
    eyes: dict[str, _Eye]


_CACHE: dict[tuple, _Injection] = {}
_BUILDS = 0


def map_builds() -> int:
    """How many times the pixel->column map has been built. Should stay at one."""
    return _BUILDS


def encode(
    raster: np.ndarray,
    connectome: Connectome,
    params: EncodeParams | None = None,
) -> np.ndarray:
    """One sub-frame -> `float32[N]` of injected current. Sensory indices only."""
    params = params or EncodeParams()
    if raster.ndim != 3:
        raise ValueError(f"expected one (size, size, channels) frame, got {raster.shape}")
    inj = _injection(raster.shape[0], connectome, params)

    out = np.zeros(inj.n, dtype=np.float32)
    luminance = np.ascontiguousarray(raster[:, :, CH_LUMINANCE]).ravel()
    for eye in inj.eyes.values():
        binned = np.bincount(
            eye.labels, weights=luminance[eye.pixels], minlength=len(eye.counts)
        ) / np.maximum(eye.counts, 1)
        # The lattice is finer than the raster in places, so some columns win no
        # pixel at all; they sample their nearest one instead of going blind.
        binned[eye.starved] = luminance[eye.starved_pixel]
        drive = params.i_max * params.luminance_gain * _naka_rushton(binned, params)
        out[eye.neurons] = drive[eye.column_of]

        window = raster[:, eye.x0 : eye.x1, :]
        for slabs, channel in zip(eye.chromatic, CHROMATIC_CHANNELS, strict=True):
            painted = window[:, :, channel]
            gain = params.i_max * params.chromatic_gain(channel)
            for slab, zone in zip(slabs, eye.zones, strict=True):
                level = float(painted[zone].sum()) / TILE_AREA_PX
                out[slab] = gain * _naka_rushton(level, params)
    return out


def _naka_rushton(level, params: EncodeParams):
    """`I = L^n / (L^n + sigma^n)` — saturates at 1, never grows without bound."""
    powered = np.power(np.clip(level, 0.0, None), params.exponent)
    return powered / (powered + params.sigma**params.exponent)


# ------------------------------------------------------- the pixel->column map


def _injection(size: int, connectome: Connectome, params: EncodeParams) -> _Injection:
    key = (size, connectome.n, params.eye_overlap, str(params.annotations_path))
    if key not in _CACHE:
        _CACHE[key] = _build(size, connectome, params)
    return _CACHE[key]


def _build(size: int, connectome: Connectome, params: EncodeParams) -> _Injection:
    global _BUILDS
    _BUILDS += 1

    reg = CellTypeRegistry.from_feather(params.annotations_path)
    to_network = {int(b): i for i, b in enumerate(connectome.body_ids)}

    body_of_row: dict[int, int] = {}
    layer: set[int] = set()
    for name in LUMINANCE_TYPES:
        for row, body in zip(reg.population(name), reg.body_ids(name), strict=True):
            index = to_network.get(int(body))
            if index is not None:
                body_of_row[int(row)] = index
                layer.add(index)

    half = round(size * (1.0 + params.eye_overlap) / 2.0)
    placed: set[int] = set()
    eyes: dict[str, _Eye] = {}
    for side in SIDES:
        x0, x1 = (0, half) if side == "L" else (size - half, size)
        eye = _build_eye(size, x0, x1, side == "R", side, reg, body_of_row)
        eyes[side] = eye
        placed.update(eye.neurons.tolist())

    unplaced = np.array(sorted(layer - placed), dtype=np.int64)
    for side, eye in eyes.items():
        eyes[side] = _with_chromatic(eye, unplaced, SIDES.index(side))
    return _Injection(n=connectome.n, size=size, eyes=eyes)


def _build_eye(
    size: int,
    x0: int,
    x1: int,
    mirror: bool,
    side: str,
    reg: CellTypeRegistry,
    body_of_row: dict[int, int],
) -> _Eye:
    keys, members = [], []
    for hexes, rows in reg.columns(side).items():
        cells = [body_of_row[int(r)] for r in rows if int(r) in body_of_row]
        if cells:
            keys.append(hexes)
            members.append(cells)
    if not keys:
        raise ValueError(f"no hex-mapped injection-layer cells on side {side!r}")

    axial = np.asarray(keys, dtype=np.float64)
    u = _unit(axial[:, 0] + 0.5 * axial[:, 1])
    v = _unit(axial[:, 1] * (np.sqrt(3.0) / 2.0))
    if mirror:
        u = 1.0 - u

    rows_i, cols_i = np.meshgrid(np.arange(size), np.arange(x0, x1), indexing="ij")
    pu = ((cols_i - x0) / max(x1 - x0 - 1, 1)).ravel()
    pv = (1.0 - rows_i / max(size - 1, 1)).ravel()  # raster row 0 is straight ahead
    distance = (pu[:, None] - u[None, :]) ** 2 + (pv[:, None] - v[None, :]) ** 2
    labels = distance.argmin(axis=1)
    pixels = (rows_i * size + cols_i).ravel()
    counts = np.bincount(labels, minlength=len(keys))
    starved = np.flatnonzero(counts == 0)

    centre = (size - 1) / 2.0
    ahead = centre - rows_i  # raster row 0 is straight ahead
    fovea = (ahead > 0) & (np.abs(cols_i - centre) <= ahead * np.tan(FOVEA_HALF_ANGLE))

    return _Eye(
        x0=x0,
        x1=x1,
        pixels=pixels,
        labels=labels.astype(np.int64),
        counts=counts.astype(np.float32),
        starved=starved,
        starved_pixel=pixels[distance[:, starved].argmin(axis=0)],
        neurons=np.array([i for cells in members for i in cells], dtype=np.int64),
        column_of=np.repeat(np.arange(len(keys)), [len(c) for c in members]),
        zones=(fovea, ~fovea),
        chromatic=(),
    )


def _unit(values: np.ndarray) -> np.ndarray:
    span = float(values.max() - values.min())
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


def _with_chromatic(eye: _Eye, unplaced: np.ndarray, offset: int) -> _Eye:
    """Split the unplaceable cells into per-side, per-channel, per-zone slabs."""
    per_side = len(CHROMATIC_CHANNELS) * len(ZONES)
    parts = np.array_split(unplaced, len(SIDES) * per_side)
    base = offset * per_side
    chromatic = tuple(
        tuple(parts[base + k * len(ZONES) + z] for z in range(len(ZONES)))
        for k in range(len(CHROMATIC_CHANNELS))
    )
    return _Eye(
        x0=eye.x0,
        x1=eye.x1,
        pixels=eye.pixels,
        labels=eye.labels,
        counts=eye.counts,
        starved=eye.starved,
        starved_pixel=eye.starved_pixel,
        neurons=eye.neurons,
        column_of=eye.column_of,
        zones=eye.zones,
        chromatic=chromatic,
    )


__all__ = ["EncodeParams", "encode", "map_builds"]
