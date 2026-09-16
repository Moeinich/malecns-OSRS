#!/usr/bin/env python3
"""Does an attackable NPC in the fovea change the network's output at all?

Offline: no game, no sidecar. Four scenes differing only in what the retina is
shown, the same calibrated engine and the same seed for each, and the decoder's
own pools read out of `engine.get_firing_rates`.

    A   empty scene, the calibration's stationary walk state, no NPCs
    B   + one attackable NPC, 4 tiles dead ahead (inside the gaze wedge)
    C   + the same NPC 4 tiles out but 60 degrees off-axis (outside it)
    B'  + a "Talk-to" NPC dead ahead; the retina paints nothing for it

Exits 0 only if the attack pool separates B from A, ignores B', and prefers the
fovea to the periphery. A z of 0.00 everywhere means the encoder handed the
network the same input twice.

Usage:
    uv run python -m tools.probe_prey
"""

from __future__ import annotations

import math
import sys

import numpy as np

from flybrain.connectome.loader import DEFAULT_PATH, load
from flybrain.engine.calibrate import _walk
from flybrain.engine.calibration import DEFAULT_CALIBRATION_PATH, UNCALIBRATED
from flybrain.engine.calibration import load as load_calibration
from flybrain.engine.lif import LIFEngine
from flybrain.loop.types import Npc, Player, WorldState
from flybrain.motor import MotorIndex
from flybrain.sensory.collision import CollisionGrid
from flybrain.sensory.encode import _injection, encode
from flybrain.sensory.retina import SUBFRAMES, Retina

WARMUP_TICKS = 30
TICKS = 150
SUBSTEPS = 19
SEED = 0

#: The bearing scene C places its NPC on, well outside the body's ~19 degree
#: gaze wedge. Hardcoded rather than imported: `tools/` may read `flybrain.motor`
#: but the wedge is the encoder's business here, not the body's.
OFF_AXIS = math.radians(60.0)
HEADING = 0.0
NPC_TILES = 4.0

DECODER_POOLS = (
    "steer_left",
    "steer_right",
    "drive",
    "reverse",
    "escape",
    "attack",
    "eat",
    "pickup",
)

#: Not a readout: the encoder's own chromatic cells, reported beside the pools so
#: a flat table says *where* the signal stopped rather than only that it did.
ENCODER_ROW = "chromatic-in"


def _player(x: int, z: int, level: int) -> Player:
    return Player(
        name="probe",
        combat_level=3,
        hp=10,
        max_hp=10,
        x=x,
        z=z,
        level=level,
        run_energy=100,
        anim_id=-1,
        in_combat=False,
        target_index=-1,
        target_type="none",
        is_dead=False,
        life_id=1,
    )


def _npc(x: int, z: int, options: tuple[str, ...]) -> Npc:
    return Npc(
        id=1,
        index=1,
        name="probe-npc",
        combat_level=1,
        x=x,
        z=z,
        size=1,
        distance=int(NPC_TILES),
        hp=10,
        max_hp=10,
        in_combat=False,
        target_index=-1,
        reachable=True,
        options=options,
    )


def _state(player: Player, npcs: tuple[Npc, ...]) -> WorldState:
    return WorldState(
        tick=0,
        in_game=True,
        modal_open=False,
        player=player,
        npcs=npcs,
        ground_items=(),
        locs=(),
        inventory=(),
        skills={},
        op_rejected_count=0,
    )


def _scenes(collision: CollisionGrid) -> dict[str, WorldState]:
    x, z = _walk(collision, 1, SEED)[0]
    player = _player(int(x), int(z), collision.level)

    def at(bearing: float, options: tuple[str, ...]) -> tuple[Npc, ...]:
        return (
            _npc(
                round(player.x + NPC_TILES * math.cos(HEADING + bearing)),
                round(player.z + NPC_TILES * math.sin(HEADING + bearing)),
                options,
            ),
        )

    return {
        "A": _state(player, ()),
        "B": _state(player, at(0.0, ("Attack",))),
        "C": _state(player, at(OFF_AXIS, ("Attack",))),
        "B'": _state(player, at(0.0, ("Talk-to",))),
    }


def _currents(state: WorldState, retina: Retina, collision, connectome, params, tonic):
    """The four sub-frame injection currents this scene holds constant forever."""
    frames = retina.render_subframes(
        state, state, collision, HEADING, n=SUBFRAMES, prev_heading=HEADING
    )
    return [encode(f, connectome, params) + tonic for f in frames]


def _run(currents, W, pools: dict[str, np.ndarray], kwargs) -> dict[str, np.ndarray]:
    engine = LIFEngine(W, seed=SEED, **kwargs)
    window = SUBFRAMES * SUBSTEPS
    out: dict[str, list[float]] = {name: [] for name in pools}
    for tick in range(WARMUP_TICKS + TICKS):
        for current in currents:
            for _ in range(SUBSTEPS):
                engine.step(current)
        if tick < WARMUP_TICKS:
            continue
        rates = engine.get_firing_rates(window)
        for name, idx in pools.items():
            out[name].append(float(rates[idx].mean()) if len(idx) else 0.0)
    return {k: np.asarray(v) for k, v in out.items()}


def _z(series: np.ndarray, base: np.ndarray) -> float:
    std = float(base.std())
    return 0.0 if std == 0.0 else float((series.mean() - base.mean()) / std)


def _fire_fractions(series: np.ndarray, base: np.ndarray, k: float) -> tuple[float, float]:
    cut = base.mean() + k * base.std()
    return float((series > cut).mean()), float((base > cut).mean())


def main() -> int:
    connectome = load(DEFAULT_PATH)
    calibration = load_calibration(DEFAULT_CALIBRATION_PATH, connectome) or UNCALIBRATED
    if not calibration.calibrated:
        print("no calibration artifact; the probe measures nothing", file=sys.stderr)
        return 1

    W = calibration.apply(connectome.W)
    motor = MotorIndex.from_connectome(connectome)
    tonic = calibration.tonic_drive(connectome.n)
    params = calibration.encode_params()
    collision = CollisionGrid.load()
    retina = Retina()

    eyes = _injection(retina.size, connectome, params).eyes.values()
    pools = {name: getattr(motor, name) for name in DECODER_POOLS}
    pools[ENCODER_ROW] = np.concatenate([s for e in eyes for ch in e.chromatic for s in ch])

    scenes = _scenes(collision)
    results = {
        name: _run(
            _currents(state, retina, collision, connectome, params, tonic),
            W,
            pools,
            calibration.engine_kwargs(),
        )
        for name, state in scenes.items()
    }

    print(calibration.describe())
    print(f"{WARMUP_TICKS} warm-up + {TICKS} ticks, {SUBFRAMES}x{SUBSTEPS} substeps\n")
    head = f"{'pool':<13}{'A mean':>9}{'A std':>8}{'B mean':>9}{'C mean':>9}{'B* mean':>9}"
    head += f"{'z(B-A)':>9}{'z(C-A)':>9}{'z(B*-A)':>9}   B>k*std (A false-fire)"
    print("B* is the Talk-to scene; chromatic-in is the encoder's own cells, not a readout")
    print(head)
    for name in (*DECODER_POOLS, ENCODER_ROW):
        a, b, c, bp = (results[s][name] for s in ("A", "B", "C", "B'"))
        gates = "  ".join(
            "k{}: {:.2f} ({:.2f})".format(k, *_fire_fractions(b, a, k)) for k in (1, 2, 3)
        )
        print(
            f"{name:<13}{a.mean():>9.3f}{a.std():>8.3f}{b.mean():>9.3f}{c.mean():>9.3f}"
            f"{bp.mean():>9.3f}{_z(b, a):>9.2f}{_z(c, a):>9.2f}{_z(bp, a):>9.2f}   {gates}"
        )

    attack = {s: results[s]["attack"] for s in ("A", "B", "C", "B'")}
    z_b, z_c, z_bp = (
        _z(attack["B"], attack["A"]),
        _z(attack["C"], attack["A"]),
        _z(attack["B'"], attack["A"]),
    )
    print()
    if z_b < 2.0:
        print(f"FAIL: attack z(B-A) = {z_b:.2f}, prey in the fovea does not reach the pool")
        return 1
    if z_bp > 0.5:
        print(f"FAIL: attack z(B'-A) = {z_bp:.2f}, a Talk-to NPC drives the pool too")
        return 1
    if z_b <= z_c:
        print(f"FAIL: attack z(B-A) = {z_b:.2f} <= z(C-A) = {z_c:.2f}, fovea is not preferred")
        return 1
    print(f"PASS: attack z(B-A) = {z_b:.2f}, z(C-A) = {z_c:.2f}, z(B'-A) = {z_bp:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
