"""One OpenCV window showing what the fly sees and what its brain is doing.

Everything is composited into a single numpy frame and shown once per draw.
That is not only simpler than a panel per window: on macOS Cocoa insists the
GUI runs on the main thread, and the tick loop *is* the main thread, so the
window is pumped from inside the loop rather than inverting the architecture.

The layout is the game on the left and the brain on the right, with the rest
of the telemetry under both. The game panel is a seam: `Hud.game_frame` takes
a BGR frame from a browser client and the panel draws it; with nothing attached
it says so rather than drawing black. The retina and world panels stay under it
either way — the synthesised retina the brain actually receives, beside the
world as the collision grid and `WorldState` say it really is. Disagreement
between those two is an encoder bug, visible at a glance.

The connectome panel plots the release's own `somaLocation` coordinates, not a
procedural stand-in, so its structure is the fly's and its brightness is the
network's real firing. A silent network must look dead there.

Absent data is drawn as "no data", never as a zero. Calibration has been wrong
four times here and every time it looked like a plausible number.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from flybrain.connectome.build import POPULATION_SEPARATOR
from flybrain.loop.types import Action, WorldState
from flybrain.motor.decode import EgocentricCommand
from flybrain.sensory.collision import CollisionGrid
from flybrain.sensory.retina import CH_LOOT, CH_LUMINANCE, CH_RESOURCE, CH_THREAT

if TYPE_CHECKING:  # pragma: no cover - import would drag cv2 into the brain
    from flybrain.loop.agent import Agent, TickReport

NO_DATA = "no data"

WIDTH = 1600
HEIGHT = 900

#: `name -> (x, y, w, h)`. Every panel draws inside its own rect and nowhere
#: else, which is what `tests/test_hud.py` pins. The game panel's inner area is
#: 887x499 - 16:9, so a real client frame fills it with no letterbox.
PANELS: dict[str, tuple[int, int, int, int]] = {
    "game": (8, 34, 903, 531),
    "connectome": (919, 34, 673, 300),
    "dn": (919, 342, 673, 223),
    "retina": (8, 573, 320, 260),
    "world": (336, 573, 244, 260),
    "rate": (588, 573, 376, 126),
    "raster": (588, 707, 376, 126),
    "health": (972, 573, 620, 260),
    "motor": (8, 841, 1584, 51),
}

TARGET_BAND_HZ = (1.0, 5.0)
WORLD_RADIUS_TILES = 24
RASTER_STEPS = 2000
QUIT_KEYS = frozenset({ord("q"), ord("Q"), 27})

_BG = (18, 18, 20)
_PANEL = (28, 28, 32)
_EDGE = (70, 70, 78)
_TEXT = (210, 210, 215)
_DIM = (130, 130, 138)
_WARN = (60, 170, 255)
_BAD = (60, 60, 240)
_GOOD = (120, 220, 130)
_ACCENT = (230, 190, 90)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Retina channel -> (label, BGR tint).
_CHANNELS = (
    (CH_LUMINANCE, "ch0 luminance / walls", (200, 200, 200)),
    (CH_THREAT, "ch1 threat / NPCs", (70, 70, 245)),
    (CH_LOOT, "ch2 loot", (80, 220, 245)),
    (CH_RESOURCE, "ch3 resources", (110, 220, 120)),
)

#: Firing rate that lights a soma fully. The target band's ceiling, so a
#: network inside its band glows and a silent one stays at the floor.
ACTIVITY_REF_HZ = 5.0
#: Brightness of a neuron firing at 0 Hz. Low enough that silence reads as dead.
_BRAIN_FLOOR = 0.13
_BRAIN_FOCAL = 6.0
_BRAIN_SPIN_PER_FRAME = 0.035

#: Functional groups for the connectome panel, resolved through the
#: connectome's own named populations. Never an index range: a neuron's row in
#: the matrix carries no anatomy.
_BRAIN_GROUPS: tuple[tuple[str, tuple[str, ...], tuple[int, int, int]], ...] = (
    (
        "optic lobe",
        ("T4", "T5", "L1", "L2", "L3", "L5", "Tm1", "Tm3", "Mi1", "Mi4", "Mi9", "C2", "C3", "CT1"),
        (70, 170, 245),
    ),
    ("mushroom body", ("KC", "MBON"), (120, 225, 140)),
    ("dopaminergic", ("PPL1", "PAM"), (110, 110, 250)),
    ("central complex", ("FB",), (230, 130, 225)),
    # Only the DN types we name and read. The network holds 1,314 descending
    # neurons; the rest are not separable from `populations` yet and fall into
    # "unnamed", so labelling this "descending" would read as a brain with 14.
    ("named DNs", ("DNp01", "DNa02", "DNp09", "DNpe017", "MDN", "DNp20"), (245, 245, 250)),
)
_BRAIN_OTHER = ("unnamed", (95, 95, 105))

_RASTER_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("T4/T5", ("T4", "T5")),
    ("L1-L3/Tm1/Mi1", ("L1", "L2", "L3", "Tm1", "Mi1")),
    ("KC", ("KC",)),
    ("named DNs", ("DNa02", "DNp09", "DNpe017", "MDN", "DNp01")),
)


# ------------------------------------------------------------------ snapshot


@dataclass(frozen=True, slots=True)
class WorldPatch:
    """The world as the state and the collision grid actually say it is."""

    walkable: np.ndarray
    x0: int
    z0: int
    player: tuple[int, int]
    heading: float
    npcs: tuple[tuple[int, int, bool], ...]
    items: tuple[tuple[int, int], ...]
    locs: tuple[tuple[int, int], ...]

    @classmethod
    def build(
        cls,
        state: WorldState,
        collision: CollisionGrid,
        heading: float,
        radius: int = WORLD_RADIUS_TILES,
    ) -> WorldPatch | None:
        if state.player is None:
            return None
        cx, cz = state.player.x, state.player.z
        return cls(
            walkable=collision.walkable_patch(cx, cz, radius),
            x0=cx - radius,
            z0=cz - radius,
            player=(cx, cz),
            heading=heading,
            npcs=tuple((n.x, n.z, n.in_combat) for n in state.npcs),
            items=tuple((g.x, g.z) for g in state.ground_items),
            locs=tuple((loc.x, loc.z) for loc in state.locs),
        )


@dataclass(frozen=True, slots=True)
class BrainCloud:
    """The release's soma coordinates, prepared once for the panel to project.

    Built from `Connectome.soma_positions`, which is NaN for every neuron the
    release annotates no soma for. Those are dropped from the *drawing* and
    counted in `positioned`; they stay in the network.
    """

    #: `float32[m, 3]`, centred and scaled by one scalar, so proportions stay
    #: the fly's. Permuted out of the EM axes into (horizontal, depth,
    #: vertical): the EM volume's `y` is dorsoventral, so it is the screen's
    #: vertical and the axis the cloud spins about. Plotting EM `z` there gives
    #: a top-down slice instead.
    xyz: np.ndarray
    #: Half-spans in those units — horizontal (the widest over a whole turn)
    #: and vertical. The panel is far wider than it is tall, so neither
    #: dimension alone may set the scale.
    half_width: float
    half_height: float
    #: `int64[m]` row in the rate vector, so activity can be gathered per point.
    index: np.ndarray
    colour: np.ndarray
    legend: tuple[tuple[str, tuple[int, int, int], int], ...]
    positioned: int
    total: int

    @classmethod
    def build(
        cls, positions: np.ndarray | None, populations: dict[str, np.ndarray]
    ) -> BrainCloud | None:
        if positions is None:
            return None
        positions = np.asarray(positions, dtype=np.float32)
        index = np.flatnonzero(np.isfinite(positions).all(axis=1))
        if not index.size:
            return None

        group = np.zeros(len(positions), dtype=np.int8)
        for g, (_label, members, _colour) in enumerate(_BRAIN_GROUPS, start=1):
            wanted = frozenset(members)
            for key, rows in populations.items():
                if key.split(POPULATION_SEPARATOR, 1)[0] in wanted:
                    group[rows] = g

        pts = positions[index][:, (0, 2, 1)]
        lo, hi = np.percentile(pts, (1.0, 99.0), axis=0)
        centre = np.median(pts, axis=0)
        pts -= centre
        half = np.maximum(hi - centre, centre - lo)
        span = max(float(half.max()), 1e-6)
        half /= span
        pts /= span

        group = group[index]
        palette = np.array([_BRAIN_OTHER[1], *(c for _l, _m, c in _BRAIN_GROUPS)], dtype=np.float32)
        labels = (_BRAIN_OTHER[0], *(label for label, _m, _c in _BRAIN_GROUPS))
        legend = tuple(
            (labels[g], tuple(int(v) for v in palette[g]), int((group == g).sum()))
            for g in range(len(labels))
            if (group == g).any()
        )
        return cls(
            xyz=pts.astype(np.float32),
            half_width=float(max(half[0], half[1])),
            half_height=float(half[2]),
            index=index,
            colour=palette[group],
            legend=legend,
            positioned=int(index.size),
            total=len(positions),
        )


@dataclass(frozen=True, slots=True)
class HudSnapshot:
    """Everything one frame draws. Every field may be None and must read as such."""

    #: The seam for the browser client: one BGR uint8 frame of the real game,
    #: any size. `None` draws the "not attached" placeholder, never black.
    game: np.ndarray | None = None
    brain: BrainCloud | None = None
    #: Per-point firing rate in Hz, aligned to `brain.index`.
    brain_activity: np.ndarray | None = None
    #: Rotation of the connectome about its vertical axis, radians.
    phase: float = 0.0
    n_neurons: int | None = None
    n_synapses: int | None = None
    fps: float | None = None
    retina: np.ndarray | None = None
    fovea_half_angle: float | None = None
    px_per_tile: float | None = None
    world: WorldPatch | None = None
    rate_hz: float | None = None
    rate_history: tuple[float, ...] = ()
    raster: np.ndarray | None = None
    raster_names: tuple[str, ...] = ()
    raster_ms: float | None = None
    dn: tuple[tuple[str, float | None], ...] = ()
    dn_differential: float | None = None
    escape_spikes: int | None = None
    escape_substep: int | None = None
    command: EgocentricCommand | None = None
    action: Action | None = None
    ack_phase: str | None = None
    ms: tuple[tuple[str, float | None], ...] = ()
    ms_total: float | None = None
    deadline_ms: float | None = None
    substeps: int | None = None
    dropped_game_ticks: int | None = None
    overruns: int | None = None
    render_ms: float | None = None
    tick: int | None = None
    revision: int | None = None


def snapshot_from_agent(
    agent: Agent,
    report: TickReport,
    *,
    raster: SpikeRaster | None = None,
    render_ms: float | None = None,
    brain: BrainCloud | None = None,
    game: np.ndarray | None = None,
    phase: float = 0.0,
    fps: float | None = None,
) -> HudSnapshot:
    """Read the agent's own state into a frame description. Draws nothing."""
    motor = agent.motor
    rates = report.rates
    state = agent._prev_state
    ack = agent.client.last_ack
    dn = (
        ("DNa02 L", _pool(rates, motor.steer_left)),
        ("DNa02 R", _pool(rates, motor.steer_right)),
        ("DNp09 drive", _pool(rates, motor.drive)),
        ("MDN reverse", _pool(rates, motor.reverse)),
        ("DNp01 escape", _pool(rates, motor.escape)),
    )
    left, right = dn[0][1], dn[1][1]
    differential = None
    if left is not None and right is not None:
        differential = (left - right) / (left + right + 1e-6)

    from flybrain.motor.body import FOVEA_HALF_ANGLE

    W = agent.engine.W
    return HudSnapshot(
        game=game,
        brain=brain,
        brain_activity=(rates[brain.index] if brain is not None and rates is not None else None),
        phase=phase,
        n_neurons=int(W.shape[0]),
        n_synapses=int(W.nnz),
        fps=fps,
        retina=agent.last_frames[-1] if agent.last_frames is not None else None,
        fovea_half_angle=FOVEA_HALF_ANGLE,
        px_per_tile=agent.retina.px_per_tile,
        world=(
            WorldPatch.build(state, agent.collision, report.heading) if state is not None else None
        ),
        rate_hz=report.mean_rate_hz,
        raster=raster.image_counts() if raster is not None else None,
        raster_names=raster.names if raster is not None else (),
        raster_ms=raster.span_ms if raster is not None else None,
        dn=dn,
        dn_differential=differential,
        escape_spikes=report.escape_spikes,
        escape_substep=report.escape_substep,
        command=report.command,
        action=report.action,
        ack_phase=ack.phase if ack is not None else None,
        ms=(
            ("retina", report.ms_retina),
            ("encode", report.ms_encode),
            ("LIF", report.ms_lif),
            ("decode", report.ms_decode),
        ),
        ms_total=report.ms_total,
        # The sidecar's own deadline, not the tick it was derived from: at a
        # 600 ms tick the deadline is 360, and the tick under-reports pressure.
        deadline_ms=_deadline_ms(report),
        substeps=report.substeps,
        dropped_game_ticks=report.dropped_game_ticks,
        overruns=agent.overruns,
        render_ms=render_ms,
        tick=report.tick,
        revision=report.revision,
    )


def _deadline_ms(report: TickReport) -> float | None:
    """`TickReport.deadline_ms` once the loop carries it; absent until then."""
    deadline = getattr(report, "deadline_ms", None)
    return None if deadline is None else float(deadline)


def _pool(rates: np.ndarray | None, idx: np.ndarray) -> float | None:
    if rates is None or not len(idx):
        return None
    return float(rates[idx].mean())


# -------------------------------------------------------------------- raster


class SpikeRaster:
    """Population activity over the last `steps` LIF substeps, as a ring buffer.

    Fed straight from the tick loop, so it must stay cheap: one boolean gather
    per substep that actually spiked, and nothing at all when none did.
    """

    def __init__(
        self,
        groups: dict[str, np.ndarray],
        n: int,
        steps: int = RASTER_STEPS,
        dt_ms: float = 1.0,
    ) -> None:
        self.names = tuple(groups)
        self.steps = steps
        self.dt_ms = dt_ms
        self._masks = np.zeros((len(groups), n), dtype=bool)
        for row, idx in enumerate(groups.values()):
            self._masks[row, idx] = True
        self.sizes = self._masks.sum(axis=1).astype(np.float32)
        self._counts = np.zeros((len(groups), steps), dtype=np.uint16)
        self._ptr = 0

    @property
    def span_ms(self) -> float:
        return self.steps * self.dt_ms

    def record(self, fired: np.ndarray) -> None:
        col = self._ptr
        if fired.size:
            self._counts[:, col] = self._masks[:, fired].sum(axis=1)
        else:
            self._counts[:, col] = 0
        self._ptr = (self._ptr + 1) % self.steps

    def image_counts(self) -> np.ndarray:
        """Chronological `(groups, steps)` counts, oldest column first."""
        return np.roll(self._counts, -self._ptr, axis=1)


def raster_groups(populations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Named raster rows from the connectome's populations, sides merged.

    Only groups with members are returned — an empty row would read as a silent
    population when it is really an absent one.
    """
    by_base: dict[str, list[np.ndarray]] = {}
    for key, idx in populations.items():
        by_base.setdefault(key.split("|", 1)[0], []).append(idx)

    out: dict[str, np.ndarray] = {}
    for label, members in _RASTER_GROUPS:
        parts = [a for name in members for a in by_base.get(name, ())]
        if parts:
            out[label] = np.unique(np.concatenate(parts))
    return out


# ------------------------------------------------------------------ drawing


def _fmt(value: float | None, spec: str = "{:.2f}") -> str:
    return NO_DATA if value is None else spec.format(value)


def _text(
    img: np.ndarray,
    s: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = _TEXT,
    scale: float = 0.4,
) -> None:
    cv2.putText(img, s, (x, y), _FONT, scale, color, 1, cv2.LINE_AA)


def _panel(img: np.ndarray, name: str, title: str) -> tuple[int, int, int, int]:
    x, y, w, h = PANELS[name]
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), _PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), _EDGE, 1)
    _text(img, title, x + 8, y + 15, _DIM, 0.38)
    return x + 8, y + 24, w - 16, h - 32


def _no_data(img: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    _text(img, NO_DATA, x + w // 2 - 24, y + h // 2, _DIM, 0.45)


def compose(snap: HudSnapshot) -> np.ndarray:
    img = np.full((HEIGHT, WIDTH, 3), _BG, dtype=np.uint8)
    _draw_status(img, snap)
    _draw_game(img, snap)
    _draw_connectome(img, snap)
    _draw_retina(img, snap)
    _draw_world(img, snap)
    _draw_rate(img, snap)
    _draw_raster(img, snap)
    _draw_dn(img, snap)
    _draw_health(img, snap)
    _draw_motor(img, snap)
    return img


def _draw_status(img: np.ndarray, snap: HudSnapshot) -> None:
    """The top strip. Outside every panel rect, so it belongs to no panel."""
    _text(img, "FLYBRAIN TELEMETRY", 10, 23, _ACCENT, 0.52)
    _text(
        img,
        f"{_fmt(snap.n_neurons, '{:,d}')} neurons   "
        f"{_fmt(snap.n_synapses, '{:,d}')} synapses   "
        f"tick {_fmt(snap.tick, '{:d}')}   rev {_fmt(snap.revision, '{:d}')}   "
        f"{_fmt(snap.rate_hz)} Hz   hud {_fmt(snap.fps, '{:.1f}')} fps",
        250,
        23,
        _TEXT,
        0.44,
    )


def _draw_game(img: np.ndarray, snap: HudSnapshot) -> None:
    rect = _panel(img, "game", "the game the fly is playing")
    x, y, w, h = rect
    frame = snap.game
    if frame is None:
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (24, 24, 28), -1)
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), _EDGE, 1)
        _text(img, "no game feed - browser client not attached", x + 20, y + h // 2, _DIM, 0.55)
        return

    frame = np.ascontiguousarray(frame)
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    tw, th = max(1, int(fw * scale)), max(1, int(fh * scale))
    ox, oy = x + (w - tw) // 2, y + (h - th) // 2
    img[oy : oy + th, ox : ox + tw] = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)


def _draw_connectome(img: np.ndarray, snap: HudSnapshot) -> None:
    cloud = snap.brain
    title = (
        f"3D connectome - MaleCNS v1.0 real somas, {cloud.positioned:,} positioned"
        if cloud is not None
        else "3D connectome - MaleCNS v1.0 real soma coordinates"
    )
    rect = _panel(img, "connectome", title)
    if cloud is None:
        _no_data(img, rect)
        return
    x, y, w, h = rect

    legend_h = 30
    ph = h - legend_h
    ca, sa = math.cos(snap.phase), math.sin(snap.phase)
    p = cloud.xyz
    # One matmul's worth of work, vectorised: spin about the vertical axis,
    # then a perspective divide. 49k points per frame leaves no room for a loop.
    across = p[:, 0] * ca - p[:, 1] * sa
    depth = p[:, 0] * sa + p[:, 1] * ca
    persp = _BRAIN_FOCAL / (_BRAIN_FOCAL + depth)
    # The nearest face is magnified most; divide it out or the cloud overruns.
    nearest = _BRAIN_FOCAL / max(_BRAIN_FOCAL - cloud.half_width, 1e-6)
    scale = 0.48 * min(ph / cloud.half_height, w / cloud.half_width) / nearest
    u = (x + w / 2 + across * persp * scale).astype(np.int32)
    v = (y + ph / 2 + p[:, 2] * persp * scale).astype(np.int32)

    activity = snap.brain_activity
    glow = (
        np.zeros(len(p), dtype=np.float32)
        if activity is None
        else np.clip(np.asarray(activity, dtype=np.float32) / ACTIVITY_REF_HZ, 0.0, 1.0)
    )
    shade = np.clip((persp - 0.7) / 0.6, 0.0, 1.0)
    weight = (_BRAIN_FLOOR + (1.0 - _BRAIN_FLOOR) * glow) * (0.35 + 0.65 * shade)

    inside = (u >= x) & (u < x + w) & (v >= y) & (v < y + ph)
    flat = (v[inside] - y) * w + (u[inside] - x)
    lit = cloud.colour[inside] * weight[inside, None]
    # Additive accumulation: order-independent, so no depth sort is needed, and
    # three bincounts cost a fraction of a per-point draw call.
    tile = np.empty((ph * w, 3), dtype=np.float32)
    for c in range(3):
        tile[:, c] = np.bincount(flat, weights=lit[:, c], minlength=ph * w)
    patch = img[y : y + ph, x : x + w]
    np.maximum(patch, np.clip(tile, 0, 255).reshape(ph, w, 3).astype(np.uint8), out=patch)

    peak = float(glow.max())
    if peak <= 0.0:
        _text(img, "SILENT - no neuron fired this tick", x + 6, y + ph - 6, _BAD, 0.4)

    column = w // 3
    for i, (label, colour, count) in enumerate(cloud.legend):
        cx = x + (i % 3) * column
        cy = y + ph + 10 + (i // 3) * 14
        cv2.rectangle(img, (cx, cy - 7), (cx + 8, cy + 1), colour, -1)
        _text(img, f"{label} {count:,}", cx + 13, cy, _DIM, 0.33)


def _draw_retina(img: np.ndarray, snap: HudSnapshot) -> None:
    rect = _panel(img, "retina", "what the fly perceives - retina 60x60x4, fovea outlined")
    if snap.retina is None:
        _no_data(img, rect)
        return
    x, y, w, h = rect
    cell = min(w // 2 - 8, (h - 44) // 2)
    frame = np.asarray(snap.retina, dtype=np.float32)
    for k, (ch, label, tint) in enumerate(_CHANNELS):
        cx = x + (k % 2) * (cell + 8)
        cy = y + (k // 2) * (cell + 22)
        plane = np.clip(frame[..., ch], 0.0, 1.0)
        up = cv2.resize(plane, (cell, cell), interpolation=cv2.INTER_NEAREST)
        tile = (up[..., None] * np.array(tint, dtype=np.float32)).astype(np.uint8)
        _draw_fovea(tile, snap.fovea_half_angle)
        img[cy : cy + cell, cx : cx + cell] = tile
        cv2.rectangle(img, (cx, cy), (cx + cell - 1, cy + cell - 1), _EDGE, 1)
        _text(img, label, cx, cy + cell + 12, _DIM, 0.33)


def _draw_fovea(tile: np.ndarray, half_angle: float | None) -> None:
    """The gaze wedge, forward being up. This is how the body picks a target."""
    if half_angle is None:
        return
    n = tile.shape[0]
    c = (n - 1) / 2.0
    reach = n
    for sign in (-1, 1):
        a = sign * half_angle
        end = (round(c + reach * math.sin(a)), round(c - reach * math.cos(a)))
        cv2.line(tile, (int(c), int(c)), end, _ACCENT, 1, cv2.LINE_AA)
    cv2.circle(tile, (int(c), int(c)), 2, _ACCENT, -1)


def _draw_world(img: np.ndarray, snap: HudSnapshot) -> None:
    rect = _panel(img, "world", "actually true - collision + WorldState")
    world = snap.world
    if world is None:
        _no_data(img, rect)
        return
    x, y, w, h = rect
    side = min(w, h - 18)
    # walkable_patch is [x, z]; transpose then flip so north is up.
    grid = np.flipud(np.asarray(world.walkable, dtype=np.float32).T)
    n = grid.shape[0]
    base = np.where(grid[..., None] > 0.5, np.float32(58.0), np.float32(22.0))
    tile = cv2.resize(
        np.repeat(base, 3, axis=2).astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST
    )
    scale = side / n

    def to_px(tx: int, tz: int) -> tuple[int, int]:
        return (
            int((tx - world.x0 + 0.5) * scale),
            int((n - 1 - (tz - world.z0) + 0.5) * scale),
        )

    for lx, lz in world.locs:
        cv2.circle(tile, to_px(lx, lz), 2, (110, 200, 120), -1)
    for ix, iz in world.items:
        cv2.circle(tile, to_px(ix, iz), 2, (80, 220, 245), -1)
    for nx, nz, in_combat in world.npcs:
        cv2.circle(tile, to_px(nx, nz), 3, (60, 60, 245) if in_combat else (90, 110, 235), -1)

    px, pz = to_px(*world.player)
    cv2.circle(tile, (px, pz), 4, (255, 255, 255), -1)
    reach = side * 0.18
    cv2.line(
        tile,
        (px, pz),
        (int(px + reach * math.cos(world.heading)), int(pz - reach * math.sin(world.heading))),
        _ACCENT,
        2,
        cv2.LINE_AA,
    )

    img[y : y + side, x : x + side] = tile
    cv2.rectangle(img, (x, y), (x + side - 1, y + side - 1), _EDGE, 1)
    _text(
        img,
        f"{world.player[0]},{world.player[1]}  {math.degrees(world.heading):+.0f} deg  "
        f"n{len(world.npcs)} i{len(world.items)} l{len(world.locs)}",
        x,
        y + side + 14,
        _DIM,
        0.33,
    )


def _draw_rate(img: np.ndarray, snap: HudSnapshot) -> None:
    rect = _panel(img, "rate", "mean firing rate vs 1-5 Hz target band")
    x, y, w, h = rect
    lo, hi = TARGET_BAND_HZ
    in_band = snap.rate_hz is not None and lo <= snap.rate_hz <= hi
    colour = _GOOD if in_band else _BAD
    _text(img, f"{_fmt(snap.rate_hz)} Hz", x, y + 22, colour, 0.75)
    _text(
        img,
        "IN BAND" if in_band else ("OUT OF BAND" if snap.rate_hz is not None else NO_DATA),
        x + 130,
        y + 22,
        colour,
        0.42,
    )

    plot_y = y + 32
    plot_h = h - 36
    if plot_h < 20:
        return
    top = max(hi * 1.4, max(snap.rate_history, default=0.0) * 1.15, 1e-3)

    def to_y(v: float) -> int:
        return int(plot_y + plot_h - min(v / top, 1.0) * plot_h)

    cv2.rectangle(img, (x, to_y(hi)), (x + w - 1, to_y(lo)), (40, 70, 45), -1)
    cv2.rectangle(img, (x, plot_y), (x + w - 1, plot_y + plot_h), _EDGE, 1)
    _text(img, f"{hi:g}", x + w - 22, to_y(hi) + 10, _DIM, 0.3)
    _text(img, f"{lo:g}", x + w - 22, to_y(lo) - 2, _DIM, 0.3)

    history = snap.rate_history
    if not history:
        _text(img, NO_DATA, x + w // 2 - 24, plot_y + plot_h // 2, _DIM, 0.4)
        return
    step = w / max(len(history), 2)
    points = np.array([[int(x + i * step), to_y(v)] for i, v in enumerate(history)], dtype=np.int32)
    cv2.polylines(img, [points], False, colour, 1, cv2.LINE_AA)


def _draw_raster(img: np.ndarray, snap: HudSnapshot) -> None:
    span = f"last {snap.raster_ms / 1000.0:.1f} s" if snap.raster_ms is not None else NO_DATA
    rect = _panel(img, "raster", f"spike raster by population, {span}")
    if snap.raster is None or not snap.raster_names:
        _no_data(img, rect)
        return
    x, y, w, h = rect
    counts = np.asarray(snap.raster, dtype=np.float32)
    rows = len(snap.raster_names)
    band = max(6, (h - 8) // rows)
    label_w = 96
    plot_w = w - label_w
    for r, name in enumerate(snap.raster_names):
        top = y + r * band
        _text(img, name, x, top + band - 4, _DIM, 0.33)
        row = counts[r]
        peak = float(row.max())
        norm = row / peak if peak > 0 else row
        strip = cv2.resize(norm[None, :], (plot_w, band - 2), interpolation=cv2.INTER_NEAREST)
        tint = np.array((120, 230, 255), dtype=np.float32)
        img[top : top + band - 2, x + label_w : x + label_w + plot_w] = (
            strip[..., None] * tint
        ).astype(np.uint8)
        if peak == 0:
            _text(img, "silent", x + label_w + 4, top + band - 5, _BAD, 0.3)


def _draw_dn(img: np.ndarray, snap: HudSnapshot) -> None:
    rect = _panel(img, "dn", "descending neurons - DNa02 differential is the steering signal")
    x, y, w, h = rect
    if not snap.dn:
        _no_data(img, rect)
        return

    _text(img, "DNa02 L-R", x, y + 16, _DIM, 0.36)
    d = snap.dn_differential
    if d is None:
        _text(img, NO_DATA, x + 90, y + 18, _DIM, 0.45)
    else:
        _text(img, f"{d:+.3f}", x + 90, y + 20, _ACCENT, 0.62)
        bar_x, bar_w = x + 180, w - 190
        mid = bar_x + bar_w // 2
        cv2.rectangle(img, (bar_x, y + 4), (bar_x + bar_w, y + 22), (40, 40, 46), -1)
        end = int(mid + max(-1.0, min(1.0, d)) * (bar_w // 2))
        cv2.rectangle(img, (min(mid, end), y + 4), (max(mid, end), y + 22), _ACCENT, -1)
        cv2.line(img, (mid, y + 2), (mid, y + 24), _TEXT, 1)

    values = [v for _, v in snap.dn if v is not None]
    top = max(max(values, default=0.0), 1e-3)
    row_y = y + 40
    for name, value in snap.dn:
        _text(img, name, x, row_y + 10, _DIM, 0.34)
        _text(img, f"{_fmt(value)} Hz", x + 96, row_y + 10, _TEXT, 0.34)
        bar_x, bar_w = x + 180, w - 190
        cv2.rectangle(img, (bar_x, row_y), (bar_x + bar_w, row_y + 12), (40, 40, 46), -1)
        if value is not None:
            cv2.rectangle(
                img,
                (bar_x, row_y),
                (bar_x + int(bar_w * value / top), row_y + 12),
                (150, 190, 110),
                -1,
            )
        row_y += 18

    spikes = snap.escape_spikes
    if spikes is None:
        _text(img, f"reflex counter: {NO_DATA}", x, y + h - 4, _DIM, 0.36)
    elif spikes > 0:
        cv2.rectangle(img, (x, y + h - 20), (x + 230, y + h - 2), _BAD, -1)
        at = _fmt(snap.escape_substep, "{:d}")
        _text(img, f"ESCAPE x{spikes} @ substep {at}", x + 6, y + h - 7, (255, 255, 255), 0.4)
    else:
        _text(img, "reflex counter: 0 escape spikes this tick", x, y + h - 4, _DIM, 0.36)


def _draw_health(img: np.ndarray, snap: HudSnapshot) -> None:
    rect = _panel(img, "health", "health - ms/tick vs deadline")
    x, y, w, _h = rect
    deadline = snap.deadline_ms
    _text(
        img,
        f"tick {_fmt(snap.tick, '{:d}')}  rev {_fmt(snap.revision, '{:d}')}  "
        f"total {_fmt(snap.ms_total, '{:.1f}')} ms / deadline {_fmt(deadline, '{:.0f}')} ms",
        x,
        y + 14,
        _TEXT,
        0.38,
    )

    bar_y = y + 24
    bar_h = 20
    cv2.rectangle(img, (x, bar_y), (x + w, bar_y + bar_h), (40, 40, 46), -1)
    tints = ((200, 160, 90), (120, 190, 220), (150, 120, 220), (120, 210, 150))
    if deadline and deadline > 0:
        cursor = float(x)
        for (name, value), tint in zip(snap.ms, tints, strict=False):
            if value is None:
                continue
            width = w * min(value / deadline, 1.0)
            cv2.rectangle(img, (int(cursor), bar_y), (int(cursor + width), bar_y + bar_h), tint, -1)
            cursor += width
        cv2.line(img, (x + w, bar_y - 2), (x + w, bar_y + bar_h + 2), _BAD, 2)
    else:
        _text(img, NO_DATA, x + w // 2 - 24, bar_y + 15, _DIM, 0.4)

    row_y = bar_y + bar_h + 18
    for (name, value), tint in zip(snap.ms, tints, strict=False):
        _text(img, f"{name} {_fmt(value, '{:.1f}')} ms", x, row_y, tint, 0.36)
        row_y += 16

    right = x + w // 2
    _text(
        img,
        f"substeps        {_fmt(snap.substeps, '{:d}')}",
        right,
        bar_y + bar_h + 18,
        _TEXT,
        0.36,
    )
    _text(
        img,
        f"dropped ticks   {_fmt(snap.dropped_game_ticks, '{:d}')}",
        right,
        bar_y + bar_h + 34,
        _TEXT,
        0.36,
    )
    _text(
        img,
        f"overruns        {_fmt(snap.overruns, '{:d}')}",
        right,
        bar_y + bar_h + 50,
        _TEXT,
        0.36,
    )
    _text(
        img,
        f"hud render      {_fmt(snap.render_ms, '{:.1f}')} ms",
        right,
        bar_y + bar_h + 66,
        _DIM,
        0.36,
    )


def _draw_motor(img: np.ndarray, snap: HudSnapshot) -> None:
    rect = _panel(img, "motor", "decoded command -> action sent -> ack phase")
    x, y, _w, h = rect
    c = snap.command
    if c is None:
        command = NO_DATA
    else:
        flags = " ".join(
            n
            for n, v in (
                ("reverse", c.reverse),
                ("escape", c.escape),
                ("attack", c.attack),
                ("eat", c.eat),
                ("pickup", c.pickup),
            )
            if v
        )
        command = f"turn {c.turn:+.3f} rad  drive {c.drive:.2f}  {flags or '-'}"

    action = NO_DATA if snap.action is None else _action_label(snap.action)
    phase = snap.ack_phase if snap.ack_phase is not None else NO_DATA

    base = y + h // 2 + 4
    _text(img, command, x, base, _TEXT, 0.42)
    _text(img, "->", x + 430, base, _DIM, 0.42)
    _text(img, action, x + 470, base, _ACCENT, 0.42)
    _text(img, "->", x + 840, base, _DIM, 0.42)
    _text(img, f"ack {phase}", x + 880, base, _GOOD if snap.ack_phase else _DIM, 0.42)


def _action_label(action: Action) -> str:
    fields = ", ".join(f"{k}={getattr(action, k)}" for k in getattr(action, "__slots__", ()))
    return f"{action.kind}({fields})" if fields else action.kind


# --------------------------------------------------------------------- window


def _probe_display(title: str) -> None:
    """Open and close a window once. Raises on a headless build or no display."""
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.destroyWindow(title)
    cv2.waitKey(1)


@dataclass
class Hud:
    """The window. Disabled by a headless probe, and never able to raise at a tick."""

    title: str = "flybrain telemetry"
    enabled: bool = True
    min_interval: float = 1.0 / 12.0
    history_len: int = 240
    raster: SpikeRaster | None = None
    brain: BrainCloud | None = None
    #: Set by whatever owns a browser client; drawn as-is next frame. Nothing
    #: here fetches it, so a missing client is a placeholder, not a stall.
    game_frame: np.ndarray | None = None
    render_ms: float | None = None
    _history: list[float] = field(default_factory=list)
    _last_draw: float = 0.0
    _phase: float = 0.0
    _fps: float | None = None
    log: Callable[[str], None] = print

    @classmethod
    def create(
        cls,
        *,
        populations: dict[str, np.ndarray] | None = None,
        n: int | None = None,
        soma_positions: np.ndarray | None = None,
        dt_ms: float = 1.0,
        probe: Callable[[str], None] = _probe_display,
        log: Callable[[str], None] = print,
        **kwargs: Any,
    ) -> Hud:
        hud = cls(log=log, **kwargs)
        try:
            probe(hud.title)
        except Exception as exc:  # noqa: BLE001 - any failure here means no display
            hud.enabled = False
            log(f"HUD disabled: no display ({type(exc).__name__}: {exc}). The brain runs on.")
            return hud
        if populations and n:
            groups = raster_groups(populations)
            if groups:
                hud.raster = SpikeRaster(groups, n, dt_ms=dt_ms)
            hud.brain = BrainCloud.build(soma_positions, populations)
            if hud.brain is None:
                log("HUD: no soma coordinates; the connectome panel reads 'no data'.")
        return hud

    # -------------------------------------------------------------- feeding

    def record_spikes(self, fired: np.ndarray) -> None:
        if self.raster is not None:
            self.raster.record(fired)

    def should_draw(self, over_budget: bool = False) -> bool:
        """Cheap enough to ask every tick; the render itself is the expensive part."""
        if not self.enabled or over_budget:
            return False
        return time.perf_counter() - self._last_draw >= self.min_interval

    def draw(self, snap: HudSnapshot) -> bool:
        """Composite and show one frame. Never raises; a failure disables the HUD."""
        if not self.enabled:
            return False
        t0 = time.perf_counter()
        try:
            cv2.imshow(self.title, compose(snap))
            key = cv2.waitKey(1) & 0xFF
        except Exception as exc:  # noqa: BLE001 - telemetry may never take the brain down
            self.enabled = False
            self.log(f"HUD disabled after a draw error ({type(exc).__name__}: {exc}).")
            return False
        now = time.perf_counter()
        if self._last_draw:
            self._fps = 1.0 / max(now - self._last_draw, 1e-6)
        self._last_draw = now
        self.render_ms = (now - t0) * 1e3
        self._phase += _BRAIN_SPIN_PER_FRAME
        if key in QUIT_KEYS:
            self.log("HUD closed (q). The brain keeps running.")
            self.close()
        return True

    def update(self, agent: Agent, report: TickReport) -> bool:
        """The one call the tick loop makes."""
        if report.mean_rate_hz is not None:
            self._history.append(report.mean_rate_hz)
            del self._history[: max(0, len(self._history) - self.history_len)]
        if not self.should_draw(over_budget=report.overrun):
            return False
        snap = snapshot_from_agent(
            agent,
            report,
            raster=self.raster,
            render_ms=self.render_ms,
            brain=self.brain,
            game=self.game_frame,
            phase=self._phase,
            fps=self._fps,
        )
        return self.draw(dataclasses.replace(snap, rate_history=tuple(self._history)))

    def close(self) -> None:
        self.enabled = False
        try:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)
        except Exception:  # noqa: BLE001, S110 - closing a window that is already gone
            pass


__all__ = [
    "ACTIVITY_REF_HZ",
    "HEIGHT",
    "NO_DATA",
    "PANELS",
    "WIDTH",
    "BrainCloud",
    "Hud",
    "HudSnapshot",
    "SpikeRaster",
    "WorldPatch",
    "compose",
    "raster_groups",
    "snapshot_from_agent",
]
