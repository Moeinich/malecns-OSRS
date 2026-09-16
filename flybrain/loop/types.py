"""Python mirror of bridge/protocol.ts.

Every message is parsed by explicit field mapping, never `cls(**d)`: an upstream
rename must raise KeyError here rather than silently becoming a missing attribute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

PROTOCOL_VERSION = 1

Json = dict[str, Any]


# ---------------------------------------------------------------- state


@dataclass(frozen=True, slots=True)
class Player:
    name: str
    combat_level: int
    hp: int
    max_hp: int
    x: int
    z: int
    level: int
    run_energy: int
    anim_id: int
    in_combat: bool
    target_index: int
    target_type: str
    is_dead: bool
    life_id: int

    @classmethod
    def from_dict(cls, d: Json) -> Player:
        return cls(
            name=d["name"],
            combat_level=d["combatLevel"],
            hp=d["hp"],
            max_hp=d["maxHp"],
            x=d["x"],
            z=d["z"],
            level=d["level"],
            run_energy=d["runEnergy"],
            anim_id=d["animId"],
            in_combat=d["inCombat"],
            target_index=d["targetIndex"],
            target_type=d["targetType"],
            is_dead=d["isDead"],
            life_id=d["lifeId"],
        )


@dataclass(frozen=True, slots=True)
class Npc:
    """`x`/`z` are the route tile; the sidecar resolved `tileX ?? x` already."""

    id: int
    index: int
    name: str
    combat_level: int
    x: int
    z: int
    size: int
    distance: int
    hp: int | None
    max_hp: int | None
    in_combat: bool
    target_index: int
    reachable: bool
    options: tuple[str, ...]

    @property
    def attackable(self) -> bool:
        return any(o.lower() == "attack" for o in self.options)

    @classmethod
    def from_dict(cls, d: Json) -> Npc:
        return cls(
            id=d["id"],
            index=d["index"],
            name=d["name"],
            combat_level=d["combatLevel"],
            x=d["x"],
            z=d["z"],
            size=d["size"],
            distance=d["distance"],
            hp=d["hp"],
            max_hp=d["maxHp"],
            in_combat=d["inCombat"],
            target_index=d["targetIndex"],
            reachable=d["reachable"],
            options=tuple(d["options"]),
        )


@dataclass(frozen=True, slots=True)
class GroundItem:
    id: int
    name: str
    count: int
    x: int
    z: int
    distance: int
    reachable: bool

    @classmethod
    def from_dict(cls, d: Json) -> GroundItem:
        return cls(
            id=d["id"],
            name=d["name"],
            count=d["count"],
            x=d["x"],
            z=d["z"],
            distance=d["distance"],
            reachable=d["reachable"],
        )


@dataclass(frozen=True, slots=True)
class Loc:
    id: int
    name: str
    x: int
    z: int
    distance: int
    options: tuple[str, ...]

    @classmethod
    def from_dict(cls, d: Json) -> Loc:
        return cls(
            id=d["id"],
            name=d["name"],
            x=d["x"],
            z=d["z"],
            distance=d["distance"],
            options=tuple(d["options"]),
        )


@dataclass(frozen=True, slots=True)
class Item:
    slot: int
    id: int
    name: str
    count: int

    @classmethod
    def from_dict(cls, d: Json) -> Item:
        return cls(slot=d["slot"], id=d["id"], name=d["name"], count=d["count"])


@dataclass(frozen=True, slots=True)
class WorldState:
    tick: int
    in_game: bool
    modal_open: bool
    player: Player | None
    npcs: tuple[Npc, ...]
    ground_items: tuple[GroundItem, ...]
    locs: tuple[Loc, ...]
    inventory: tuple[Item, ...]
    skills: dict[str, int]
    op_rejected_count: int

    @classmethod
    def from_dict(cls, d: Json) -> WorldState:
        player = d["player"]
        return cls(
            tick=d["tick"],
            in_game=d["inGame"],
            modal_open=d["modalOpen"],
            player=Player.from_dict(player) if player is not None else None,
            npcs=tuple(Npc.from_dict(n) for n in d["npcs"]),
            ground_items=tuple(GroundItem.from_dict(g) for g in d["groundItems"]),
            locs=tuple(Loc.from_dict(loc) for loc in d["locs"]),
            inventory=tuple(Item.from_dict(i) for i in d["inventory"]),
            skills=dict(d["skills"]),
            op_rejected_count=d["opRejectedCount"],
        )


# ---------------------------------------------------------------- messages


@dataclass(frozen=True, slots=True)
class Ready:
    protocol: int
    username: str
    mode: str
    role: str
    tick_ms: int
    pathfinding_warm: bool

    @classmethod
    def from_dict(cls, d: Json) -> Ready:
        return cls(
            protocol=d["protocol"],
            username=d["username"],
            mode=d["mode"],
            role=d["role"],
            tick_ms=d["tickMs"],
            pathfinding_warm=d["pathfindingWarm"],
        )


@dataclass(frozen=True, slots=True)
class StateUpdate:
    revision: int
    tick: int
    dropped_since_last: int
    deadline_ms: int
    #: The tick the sidecar derived the deadline from, per state.
    tick_ms: int
    #: Raw rolling median; None until the sidecar's meter has samples.
    observed_tick_ms: float | None
    state: WorldState

    @classmethod
    def from_dict(cls, d: Json) -> StateUpdate:
        return cls(
            revision=d["revision"],
            tick=d["tick"],
            dropped_since_last=d["droppedSinceLast"],
            deadline_ms=d["deadlineMs"],
            tick_ms=d["tickMs"],
            observed_tick_ms=d.get("observedTickMs"),
            state=WorldState.from_dict(d["state"]),
        )


@dataclass(frozen=True, slots=True)
class CombatEvent:
    tick: int
    observation_id: int | None
    type: str
    damage: int
    source_type: str
    source_index: int
    target_type: str
    target_index: int

    @classmethod
    def from_dict(cls, d: Json) -> CombatEvent:
        return cls(
            tick=d["tick"],
            observation_id=d.get("observationId"),
            type=d["type"],
            damage=d["damage"],
            source_type=d["sourceType"],
            source_index=d["sourceIndex"],
            target_type=d["targetType"],
            target_index=d["targetIndex"],
        )


@dataclass(frozen=True, slots=True)
class Reward:
    revision: int
    combat_events: tuple[CombatEvent, ...]
    xp_delta: dict[str, int]

    @classmethod
    def from_dict(cls, d: Json) -> Reward:
        return cls(
            revision=d["revision"],
            combat_events=tuple(CombatEvent.from_dict(e) for e in d["combatEvents"]),
            xp_delta=dict(d["xpDelta"]),
        )


@dataclass(frozen=True, slots=True)
class Ack:
    cmd_id: int
    ok: bool
    phase: str
    op_rejected_delta: int
    message: str
    #: Where the player ended up; only a `reset` ack carries it.
    x: int | None = None
    z: int | None = None

    @classmethod
    def from_dict(cls, d: Json) -> Ack:
        return cls(
            cmd_id=d["cmdId"],
            ok=d["ok"],
            phase=d["phase"],
            op_rejected_delta=d["opRejectedDelta"],
            message=d["message"],
            x=d.get("x"),
            z=d.get("z"),
        )


@dataclass(frozen=True, slots=True)
class BridgeError:
    message: str

    @classmethod
    def from_dict(cls, d: Json) -> BridgeError:
        return cls(message=d["message"])


ServerMessage = Ready | StateUpdate | Reward | Ack | BridgeError

_PARSERS = {
    "ready": Ready.from_dict,
    "state": StateUpdate.from_dict,
    "reward": Reward.from_dict,
    "ack": Ack.from_dict,
    "error": BridgeError.from_dict,
}


class ProtocolError(RuntimeError):
    """The sidecar sent something we cannot parse. Never swallowed."""


def parse_server_message(raw: Json) -> ServerMessage:
    try:
        parser = _PARSERS[raw["t"]]
    except KeyError as exc:
        raise ProtocolError(f"unknown message type {raw.get('t')!r}") from exc
    try:
        return parser(raw)
    except (KeyError, TypeError) as exc:
        raise ProtocolError(f"malformed {raw['t']} message: {exc}") from exc


# ---------------------------------------------------------------- actions


@dataclass(frozen=True, slots=True)
class Walk:
    x: int
    z: int
    running: bool = True
    kind: ClassVar[str] = "walk"

    def to_dict(self) -> Json:
        return {"kind": "walk", "x": self.x, "z": self.z, "running": self.running}


@dataclass(frozen=True, slots=True)
class Flee:
    x: int
    z: int
    kind: ClassVar[str] = "flee"

    def to_dict(self) -> Json:
        return {"kind": "flee", "x": self.x, "z": self.z}


@dataclass(frozen=True, slots=True)
class AttackFovea:
    npc_index: int
    kind: ClassVar[str] = "attack_fovea"

    def to_dict(self) -> Json:
        return {"kind": "attack_fovea", "npcIndex": self.npc_index}


@dataclass(frozen=True, slots=True)
class PickupFovea:
    x: int
    z: int
    item_id: int
    kind: ClassVar[str] = "pickup_fovea"

    def to_dict(self) -> Json:
        return {"kind": "pickup_fovea", "x": self.x, "z": self.z, "itemId": self.item_id}


@dataclass(frozen=True, slots=True)
class Eat:
    slot: int
    kind: ClassVar[str] = "eat"

    def to_dict(self) -> Json:
        return {"kind": "eat", "slot": self.slot}


@dataclass(frozen=True, slots=True)
class Idle:
    kind: ClassVar[str] = "idle"

    def to_dict(self) -> Json:
        return {"kind": "idle"}


Action = Walk | Flee | AttackFovea | PickupFovea | Eat | Idle
