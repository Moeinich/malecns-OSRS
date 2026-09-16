"""Frame composition only. Nothing here opens a window.

`compose` is a pure function of a snapshot, so the interesting properties —
shape, panel isolation, and absent data reading as absent rather than as zero —
are all testable without a display. `Hud.create` is exercised with a probe that
fails, which is what a headless CI or an SSH session looks like.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from flybrain import hud
from flybrain.loop.types import GroundItem, Loc, Npc, Player, Walk, WorldState
from flybrain.motor.decode import EgocentricCommand
from flybrain.sensory.collision import CollisionGrid
from flybrain.sensory.retina import N_CHANNELS

# ------------------------------------------------------------------ fixtures


def _player(x: int = 3200, z: int = 3200) -> Player:
    return Player(
        name="flybot01",
        combat_level=3,
        hp=10,
        max_hp=10,
        x=x,
        z=z,
        level=0,
        run_energy=100,
        anim_id=-1,
        in_combat=False,
        target_index=-1,
        target_type="none",
        is_dead=False,
        life_id=1,
    )


def _state() -> WorldState:
    npc = Npc(
        id=1,
        index=7,
        name="Rat",
        combat_level=2,
        x=3203,
        z=3202,
        size=1,
        distance=3,
        hp=5,
        max_hp=5,
        in_combat=True,
        target_index=-1,
        reachable=True,
        options=("Attack",),
    )
    item = GroundItem(id=526, name="Bones", count=1, x=3198, z=3201, distance=2, reachable=True)
    loc = Loc(id=1276, name="Tree", x=3205, z=3205, distance=7, options=("Chop down",))
    return WorldState(
        tick=1,
        in_game=True,
        modal_open=False,
        player=_player(),
        npcs=(npc,),
        ground_items=(item,),
        locs=(loc,),
        inventory=(),
        skills={},
        op_rejected_count=0,
    )


def _collision() -> CollisionGrid:
    grid = np.ones((120, 120), dtype=np.uint8)
    grid[60:70, 60:64] = 0
    return CollisionGrid(grid=grid, x_min=3150, z_min=3150, level=0)


#: A stand-in connectome: real-shaped soma coordinates for 300 neurons, five
#: of them unannotated, plus the named populations the panel colours by.
N_CELLS = 4000
NO_SOMA = frozenset(range(3, N_CELLS, 97))
_RANGES = {
    "optic lobe": ((0, 800), (800, 1100)),
    "mushroom body": ((1100, 1900),),
    "dopaminergic": ((1900, 2100),),
    "central complex": ((2100, 2400),),
    "named DNs": ((2400, 2404),),
}


def _expected(label: str) -> int:
    return sum(len(set(range(lo, hi)) - NO_SOMA) for lo, hi in _RANGES[label])


def _soma_positions() -> np.ndarray:
    rng = np.random.default_rng(3)
    pos = rng.uniform([2468, 5820, 10483], [93600, 68904, 134218], size=(N_CELLS, 3))
    pos[sorted(NO_SOMA)] = np.nan
    return pos.astype(np.float32)


def _populations() -> dict[str, np.ndarray]:
    return {
        "T4": np.arange(0, 800),
        "T4|L": np.arange(0, 400),
        "L1": np.arange(800, 1100),
        "KC": np.arange(1100, 1900),
        "PAM": np.arange(1900, 2100),
        "FB": np.arange(2100, 2400),
        "DNa02": np.arange(2400, 2404),
    }


def _cloud() -> hud.BrainCloud:
    cloud = hud.BrainCloud.build(_soma_positions(), _populations())
    assert cloud is not None
    return cloud


def _game_frame() -> np.ndarray:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, :, 1] = 90
    return frame


def _full_snapshot() -> hud.HudSnapshot:
    rng = np.random.default_rng(0)
    retina = rng.random((60, 60, N_CHANNELS), dtype=np.float32)
    counts = rng.integers(0, 5, size=(4, 400)).astype(np.uint16)
    cloud = _cloud()
    return hud.HudSnapshot(
        game=_game_frame(),
        brain=cloud,
        brain_activity=rng.random(cloud.positioned, dtype=np.float32) * 6.0,
        phase=0.7,
        n_neurons=N_CELLS,
        n_synapses=123456,
        fps=11.8,
        retina=retina,
        fovea_half_angle=0.34,
        px_per_tile=1.5,
        world=hud.WorldPatch.build(_state(), _collision(), 0.7),
        rate_hz=2.4,
        rate_history=tuple(float(v) for v in rng.random(200) * 6),
        raster=counts,
        raster_names=("T4/T5", "L1-L3/Tm1/Mi1", "KC", "named DNs"),
        raster_ms=2000.0,
        dn=(
            ("DNa02 L", 3.0),
            ("DNa02 R", 1.0),
            ("DNp09 drive", 0.5),
            ("MDN reverse", 0.0),
            ("DNp01 escape", 0.0),
        ),
        dn_differential=0.5,
        escape_spikes=0,
        escape_substep=None,
        command=EgocentricCommand(
            turn=0.2, drive=0.4, reverse=False, escape=False, attack=False, eat=False, pickup=False
        ),
        action=Walk(x=3203, z=3204, running=True),
        ack_phase="dispatched",
        ms=(("retina", 4.0), ("encode", 6.0), ("LIF", 30.0), ("decode", 0.3)),
        ms_total=41.0,
        deadline_ms=360.0,
        substeps=600,
        dropped_game_ticks=0,
        overruns=0,
        render_ms=7.5,
        tick=88001,
        revision=12,
    )


# ------------------------------------------------------------- composition


def test_compose_returns_one_uint8_frame():
    frame = hud.compose(_full_snapshot())
    assert frame.shape == (hud.HEIGHT, hud.WIDTH, 3)
    assert frame.dtype == np.uint8


def test_empty_snapshot_still_composes():
    frame = hud.compose(hud.HudSnapshot())
    assert frame.shape == (hud.HEIGHT, hud.WIDTH, 3)


def test_panels_are_disjoint_and_inside_the_frame():
    mask = np.zeros((hud.HEIGHT, hud.WIDTH), dtype=np.uint8)
    for name, (x, y, w, h) in hud.PANELS.items():
        assert x >= 0 and y >= 0, name
        assert x + w <= hud.WIDTH and y + h <= hud.HEIGHT, name
        assert mask[y : y + h, x : x + w].max() == 0, f"{name} overlaps another panel"
        mask[y : y + h, x : x + w] = 1


@pytest.mark.parametrize(
    ("field", "value", "panel"),
    [
        ("game", None, "game"),
        ("brain", None, "connectome"),
        ("retina", None, "retina"),
        ("world", None, "world"),
        ("raster", None, "raster"),
        ("command", None, "motor"),
        ("substeps", None, "health"),
    ],
)
def test_a_panel_only_draws_inside_its_own_rect(field, value, panel):
    """Changing one panel's input must not move a pixel in any other panel."""
    base = _full_snapshot()
    other = hud.compose(dataclasses.replace(base, **{field: value}))
    reference = hud.compose(base)
    changed = np.any(reference != other, axis=2)
    assert changed.any(), f"{field} made no difference at all"

    x, y, w, h = hud.PANELS[panel]
    outside = changed.copy()
    outside[y : y + h, x : x + w] = False
    assert not outside.any(), f"{panel} drew outside its rect"


@pytest.mark.parametrize(
    "field", ["rate_hz", "ms_total", "substeps", "dropped_game_ticks", "escape_spikes"]
)
def test_absent_data_does_not_render_as_zero(field):
    """A missing number must read as missing. Plausible zeros have bitten this project."""
    base = _full_snapshot()
    absent = hud.compose(dataclasses.replace(base, **{field: None}))
    zero = hud.compose(dataclasses.replace(base, **{field: type(base.__getattribute__(field))(0)}))
    assert np.any(absent != zero), f"{field}=None looks identical to {field}=0"


def test_formatter_says_no_data_for_none():
    assert hud._fmt(None) == hud.NO_DATA
    assert hud._fmt(None, "{:d}") == hud.NO_DATA
    assert hud._fmt(0.0) == "0.00"


def test_missing_population_rates_are_none_not_zero():
    rates = np.zeros(10, dtype=np.float32)
    assert hud._pool(rates, np.array([], dtype=np.int64)) is None
    assert hud._pool(None, np.array([0, 1])) is None
    assert hud._pool(rates, np.array([0, 1])) == 0.0


# ------------------------------------------------------------------ raster


def test_raster_groups_skip_absent_populations():
    populations = {"T4|L": np.array([0, 1]), "T4|R": np.array([1, 2]), "Nope": np.array([9])}
    groups = hud.raster_groups(populations)
    assert set(groups) == {"T4/T5"}
    assert groups["T4/T5"].tolist() == [0, 1, 2]


def test_spike_raster_is_chronological_and_wraps():
    raster = hud.SpikeRaster({"a": np.array([0]), "b": np.array([1])}, n=2, steps=4)
    for fired in ([0], [], [1], [0, 1], [1]):
        raster.record(np.array(fired, dtype=np.int32))
    counts = raster.image_counts()
    assert counts.shape == (2, 4)
    # The first record was evicted; the last column is the most recent step.
    assert counts[:, -1].tolist() == [0, 1]
    assert counts[:, -2].tolist() == [1, 1]
    assert raster.span_ms == 4.0


# ---------------------------------------------------------------- headless


def test_headless_probe_disables_the_hud_without_raising():
    messages: list[str] = []

    def probe(_title: str) -> None:
        raise RuntimeError("cv2.imshow: no display")

    h = hud.Hud.create(probe=probe, log=messages.append)
    assert h.enabled is False
    assert h.raster is None
    assert len(messages) == 1 and "HUD disabled" in messages[0]


def test_a_disabled_hud_never_draws_or_records():
    h = hud.Hud.create(
        probe=lambda _t: (_ for _ in ()).throw(OSError("headless")), log=lambda _m: None
    )
    assert h.should_draw() is False
    assert h.draw(_full_snapshot()) is False
    h.record_spikes(np.array([0], dtype=np.int32))  # must not raise


def test_a_draw_failure_disables_the_hud_rather_than_killing_the_brain(monkeypatch):
    messages: list[str] = []
    h = hud.Hud(log=messages.append)
    monkeypatch.setattr(
        hud.cv2, "imshow", lambda *_a: (_ for _ in ()).throw(RuntimeError("window gone"))
    )
    assert h.draw(_full_snapshot()) is False
    assert h.enabled is False
    assert "HUD disabled" in messages[0]


def test_draw_is_throttled():
    h = hud.Hud(min_interval=10.0)
    h._last_draw = float("inf")
    assert h.should_draw() is False
    h._last_draw = 0.0
    assert h.should_draw() is True
    assert h.should_draw(over_budget=True) is False


# ---------------------------------------------------------------- connectome


def test_the_cloud_drops_unpositioned_neurons_without_inventing_a_position():
    cloud = _cloud()
    assert cloud.total == N_CELLS
    assert cloud.positioned == N_CELLS - len(NO_SOMA)
    assert not set(cloud.index.tolist()) & set(NO_SOMA)
    assert np.isfinite(cloud.xyz).all()


def test_the_cloud_colours_by_named_population_not_by_index_range():
    cloud = _cloud()
    by_label = {label: (colour, count) for label, colour, count in cloud.legend}
    for label in _RANGES:
        assert by_label[label][1] == _expected(label), label
    # Everything outside a named population must still be drawn, in its own hue.
    assert by_label["unnamed"][1] > 0
    assert len({colour for colour, _ in by_label.values()}) == len(by_label)


def test_a_missing_soma_array_is_no_data_rather_than_an_empty_brain():
    assert hud.BrainCloud.build(None, _populations()) is None
    assert hud.BrainCloud.build(np.full((4, 3), np.nan, np.float32), {}) is None


def test_a_silent_network_looks_dead_and_a_firing_one_does_not():
    """The panel is lit by real firing, so 0.00 Hz must not render as pretty."""
    base = _full_snapshot()
    cloud = base.brain
    x, y, w, h = hud.PANELS["connectome"]

    def lit(rate: float) -> float:
        """Mean brightness above the panel ground, inside the projection only.

        The margins are cut so the title, the legend chips and the SILENT
        marker cannot stand in for neurons that are not firing.
        """
        snap = dataclasses.replace(
            base, brain_activity=np.full(cloud.positioned, rate, dtype=np.float32)
        )
        plot = hud.compose(snap)[y + 40 : y + h - 60, x + 20 : x + w - 20].astype(np.int16)
        return float(np.clip(plot - 34, 0, None).mean())

    silent, firing = lit(0.0), lit(hud.ACTIVITY_REF_HZ)
    assert silent < 1.0, "a network at 0.00 Hz must look dead, not decorative"
    assert firing > 5.0 * silent


def test_rotation_moves_the_connectome_and_nothing_else():
    base = _full_snapshot()
    turned = hud.compose(dataclasses.replace(base, phase=base.phase + 1.0))
    changed = np.any(hud.compose(base) != turned, axis=2)
    assert changed.any()
    x, y, w, h = hud.PANELS["connectome"]
    changed[y : y + h, x : x + w] = False
    assert not changed.any()


# ---------------------------------------------------------------- game seam


def test_the_game_panel_says_it_has_no_feed_rather_than_drawing_black():
    frame = hud.compose(dataclasses.replace(_full_snapshot(), game=None))
    x, y, w, h = hud.PANELS["game"]
    assert frame[y : y + h, x : x + w].max() > 0, "the placeholder must be legible"


def test_the_game_panel_draws_the_frame_it_is_handed():
    """The seam the browser lane writes into: a BGR frame, any size."""
    snap = dataclasses.replace(_full_snapshot(), game=np.full((90, 160, 3), 200, np.uint8))
    x, y, w, h = hud.PANELS["game"]
    mask = (hud.compose(snap)[y : y + h, x : x + w] == 200).all(axis=2)
    rows, cols = np.nonzero(mask)
    box = (cols.max() - cols.min() + 1, rows.max() - rows.min() + 1)
    assert mask.sum() == box[0] * box[1], "the frame must land as one solid rect"
    assert abs(box[0] / box[1] - 16 / 9) < 0.01, "the inner rect must not letterbox 16:9"


# ------------------------------------------------------------------ deadline


class _FakeReport:
    """Only what `snapshot_from_agent` reads, plus the sidecar's own deadline."""

    def __init__(self, deadline_ms: float) -> None:
        self.deadline_ms = deadline_ms
        self.rates = np.zeros(4, dtype=np.float32)
        self.heading = 0.0
        self.mean_rate_hz = 0.0
        self.escape_spikes = 0
        self.escape_substep = None
        self.command = None
        self.action = None
        self.ms_retina = self.ms_encode = self.ms_lif = self.ms_decode = 1.0
        self.ms_total = 4.0
        self.substeps = 600
        self.dropped_game_ticks = 0
        self.tick = 1
        self.revision = 1


def _fake_agent():
    import scipy.sparse as sp

    empty = np.array([], dtype=np.int64)
    return SimpleNamespace(
        motor=SimpleNamespace(
            steer_left=empty, steer_right=empty, drive=empty, reverse=empty, escape=empty
        ),
        engine=SimpleNamespace(W=sp.csc_matrix((4, 4), dtype=np.float32)),
        client=SimpleNamespace(last_ack=None),
        retina=SimpleNamespace(px_per_tile=1.5),
        collision=_collision(),
        last_frames=None,
        overruns=0,
        tick_ms=600,
        _prev_state=None,
    )


def test_the_health_panel_shows_the_sidecar_deadline_not_the_tick():
    """360 ms of a 600 ms tick. Reading the tick hides how close we are."""
    snap = hud.snapshot_from_agent(_fake_agent(), _FakeReport(360.0))
    assert snap.deadline_ms == 360.0

    x, y, w, h = hud.PANELS["health"]
    panel = hud.compose(snap)[y : y + h, x : x + w]
    wrong = hud.compose(dataclasses.replace(snap, deadline_ms=600.0))[y : y + h, x : x + w]
    assert np.any(panel != wrong), "the deadline must be visible in the health panel"


def test_the_label_draws_in_the_strip_and_nowhere_else():
    base = _full_snapshot()
    labelled = hud.compose(dataclasses.replace(base, label="ablation - shuffle - tick 7/20"))
    plain = hud.compose(base)
    assert not np.array_equal(labelled[:30], plain[:30])
    assert np.array_equal(labelled[30:], plain[30:])


def test_no_label_renders_exactly_as_before():
    base = _full_snapshot()
    assert np.array_equal(hud.compose(dataclasses.replace(base, label=None)), hud.compose(base))


def test_the_label_is_right_aligned_and_clear_of_the_existing_strip_text():
    snap = dataclasses.replace(_full_snapshot(), label="x" * 40)
    strip = hud.compose(snap)[:30]
    lit = np.argwhere(strip.any(axis=2))[:, 1]
    assert lit.max() >= hud.WIDTH - 20
