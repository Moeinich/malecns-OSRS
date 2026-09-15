/**
 * The single place raw rs-sdk state becomes brain-facing state: tile positions
 * resolved once, XP and combat deltas cut once, everything else dropped.
 */
import type { BotWorldState, CombatEvent } from "../vendor/rs-sdk/sdk/types.ts";
import type {
    NormalizedGroundItem,
    NormalizedItem,
    NormalizedLoc,
    NormalizedNpc,
    NormalizedPlayer,
    NormalizedState,
} from "./protocol.ts";

export interface Observation {
    state: NormalizedState;
    combatEvents: CombatEvent[];
    xpDelta: Record<string, number>;
}

export class StateNormalizer {
    private lastXp = new Map<string, number>();
    private combatCursor = -1;

    normalize(raw: BotWorldState): Observation {
        const skills: Record<string, number> = {};
        const xpDelta: Record<string, number> = {};
        for (const skill of raw.skills) {
            skills[skill.name] = skill.experience;
            const previous = this.lastXp.get(skill.name);
            if (previous !== undefined && skill.experience > previous) {
                xpDelta[skill.name] = skill.experience - previous;
            }
            this.lastXp.set(skill.name, skill.experience);
        }

        const combatEvents = raw.combatEvents.filter((e) => cursorOf(e) > this.combatCursor);
        for (const event of combatEvents) {
            this.combatCursor = Math.max(this.combatCursor, cursorOf(event));
        }

        const state: NormalizedState = {
            tick: raw.tick,
            inGame: raw.inGame,
            modalOpen: raw.modalOpen,
            player: normalizePlayer(raw),
            npcs: raw.nearbyNpcs.map(normalizeNpc),
            groundItems: raw.groundItems.map(normalizeGroundItem),
            locs: raw.nearbyLocs.map(normalizeLoc),
            inventory: raw.inventory.map(normalizeItem),
            skills,
            opRejectedCount: raw.opFeedback?.opRejectedCount ?? 0,
        };
        return { state, combatEvents, xpDelta };
    }
}

/**
 * `npc.x/z` are interpolated render positions trailing a moving NPC by 1-3
 * tiles; `tileX/tileZ` is the authoritative route tile, and drops land on it.
 */
export function npcTile(npc: { x: number; z: number; tileX?: number; tileZ?: number }): { x: number; z: number } {
    return { x: npc.tileX ?? npc.x, z: npc.tileZ ?? npc.z };
}

function normalizePlayer(raw: BotWorldState): NormalizedPlayer | null {
    const p = raw.player;
    if (!p) return null;
    return {
        name: p.name,
        combatLevel: p.combatLevel,
        hp: p.hp,
        maxHp: p.maxHp,
        x: p.worldX,
        z: p.worldZ,
        level: p.level,
        runEnergy: p.runEnergy,
        animId: p.animId,
        inCombat: p.combat.inCombat,
        targetIndex: p.combat.targetIndex,
        targetType: p.combat.targetType,
        isDead: p.isDead,
        lifeId: p.lifeId,
    };
}

function normalizeNpc(npc: BotWorldState["nearbyNpcs"][number]): NormalizedNpc {
    const tile = npcTile(npc);
    return {
        id: npc.id,
        index: npc.index,
        name: npc.name,
        combatLevel: npc.combatLevel,
        x: tile.x,
        z: tile.z,
        size: npc.size ?? 1,
        distance: npc.distance,
        hp: npc.hp,
        maxHp: npc.maxHp,
        inCombat: npc.inCombat,
        targetIndex: npc.targetIndex,
        reachable: npc.reachable ?? true,
        options: npc.options,
    };
}

function normalizeGroundItem(item: BotWorldState["groundItems"][number]): NormalizedGroundItem {
    return {
        id: item.id,
        name: item.name,
        count: item.count,
        x: item.x,
        z: item.z,
        distance: item.distance,
        reachable: item.reachable ?? true,
    };
}

function normalizeLoc(loc: BotWorldState["nearbyLocs"][number]): NormalizedLoc {
    return { id: loc.id, name: loc.name, x: loc.x, z: loc.z, distance: loc.distance, options: loc.options };
}

function normalizeItem(item: BotWorldState["inventory"][number]): NormalizedItem {
    return { slot: item.slot, id: item.id, name: item.name, count: item.count };
}

/** `observationId` is monotonic across publications; tick is the older fallback. */
function cursorOf(event: CombatEvent): number {
    return event.observationId ?? event.tick;
}
