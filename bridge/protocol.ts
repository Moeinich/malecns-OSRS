/** NDJSON message schema spoken over the unix socket. One JSON object per line. */
import type { CombatEvent, SDKConnectionMode } from "../vendor/rs-sdk/sdk/types.ts";

export const PROTOCOL_VERSION = 1;

/** `brain` may send cmd; `observe` clients read only and can never pre-empt. */
export type Role = "brain" | "observe";

/**
 * Targets are resolved by the fly's body (motor/body.py), never by the sidecar:
 * the `_fovea` kinds carry what the brain was already looking at.
 */
export type MotorAction =
    | { kind: "walk"; x: number; z: number; running?: boolean }
    | { kind: "flee"; x: number; z: number }
    | { kind: "attack_fovea"; npcIndex: number }
    | { kind: "pickup_fovea"; x: number; z: number; itemId: number }
    | { kind: "eat"; slot: number }
    | { kind: "idle" };

export type MotorKind = MotorAction["kind"];

export interface NormalizedPlayer {
    name: string;
    combatLevel: number;
    hp: number;
    maxHp: number;
    x: number;
    z: number;
    level: number;
    runEnergy: number;
    animId: number;
    inCombat: boolean;
    targetIndex: number;
    targetType: "npc" | "player" | "none";
    isDead: boolean;
    lifeId: number;
}

export interface NormalizedNpc {
    id: number;
    index: number;
    name: string;
    combatLevel: number;
    /** Route tile (`tileX ?? x`), not the interpolated render position. */
    x: number;
    z: number;
    size: number;
    distance: number;
    hp: number | null;
    maxHp: number | null;
    inCombat: boolean;
    targetIndex: number;
    reachable: boolean;
    options: string[];
}

export interface NormalizedGroundItem {
    id: number;
    name: string;
    count: number;
    x: number;
    z: number;
    distance: number;
    reachable: boolean;
}

export interface NormalizedLoc {
    id: number;
    name: string;
    x: number;
    z: number;
    distance: number;
    options: string[];
}

export interface NormalizedItem {
    slot: number;
    id: number;
    name: string;
    count: number;
}

export interface NormalizedState {
    tick: number;
    inGame: boolean;
    modalOpen: boolean;
    player: NormalizedPlayer | null;
    npcs: NormalizedNpc[];
    groundItems: NormalizedGroundItem[];
    locs: NormalizedLoc[];
    inventory: NormalizedItem[];
    skills: Record<string, number>;
    opRejectedCount: number;
}

export interface ReadyMsg {
    t: "ready";
    protocol: number;
    username: string;
    mode: SDKConnectionMode;
    role: Role;
    tickMs: number;
    pathfindingWarm: boolean;
}

export interface StateMsg {
    t: "state";
    revision: number;
    tick: number;
    /** States conflated away since the last push. Depth-1: never a queue. */
    droppedSinceLast: number;
    deadlineMs: number;
    state: NormalizedState;
}

export interface RewardMsg {
    t: "reward";
    revision: number;
    combatEvents: CombatEvent[];
    xpDelta: Record<string, number>;
}

export type AckPhase =
    | "stale"
    | "duplicate"
    | "role"
    | "validation"
    | "routing"
    | "dispatch"
    | "observation"
    | "completion"
    | "continuation";

export interface AckMsg {
    t: "ack";
    cmdId: number;
    ok: boolean;
    phase: AckPhase;
    /** Rise in `opFeedback.opRejectedCount` across the dispatch: > 0 means refused. */
    opRejectedDelta: number;
    message: string;
}

export interface ErrorMsg {
    t: "error";
    message: string;
}

export type ServerMsg = ReadyMsg | StateMsg | RewardMsg | AckMsg | ErrorMsg;

export interface HelloMsg {
    t: "hello";
    protocol: number;
    role: Role;
}

export interface CmdMsg {
    t: "cmd";
    cmdId: number;
    revision: number;
    action: MotorAction;
}

export interface NoopMsg {
    t: "noop";
    revision: number;
}

export type ClientMsg = HelloMsg | CmdMsg | NoopMsg;

export function encode(msg: ServerMsg): string {
    return JSON.stringify(msg) + "\n";
}

export function parseClientMsg(line: string): ClientMsg | null {
    let raw: unknown;
    try {
        raw = JSON.parse(line);
    } catch {
        return null;
    }
    if (typeof raw !== "object" || raw === null) return null;
    const msg = raw as Record<string, unknown>;
    switch (msg.t) {
        case "hello":
            return msg.protocol === PROTOCOL_VERSION && (msg.role === "brain" || msg.role === "observe")
                ? { t: "hello", protocol: PROTOCOL_VERSION, role: msg.role }
                : null;
        case "cmd":
            return typeof msg.cmdId === "number" && typeof msg.revision === "number" && isMotorAction(msg.action)
                ? { t: "cmd", cmdId: msg.cmdId, revision: msg.revision, action: msg.action }
                : null;
        case "noop":
            return typeof msg.revision === "number" ? { t: "noop", revision: msg.revision } : null;
        default:
            return null;
    }
}

function isMotorAction(value: unknown): value is MotorAction {
    if (typeof value !== "object" || value === null) return false;
    const a = value as Record<string, unknown>;
    const num = (k: string) => typeof a[k] === "number";
    switch (a.kind) {
        case "walk":
            return num("x") && num("z");
        case "flee":
            return num("x") && num("z");
        case "attack_fovea":
            return num("npcIndex");
        case "pickup_fovea":
            return num("x") && num("z") && num("itemId");
        case "eat":
            return num("slot");
        case "idle":
            return true;
        default:
            return false;
    }
}
