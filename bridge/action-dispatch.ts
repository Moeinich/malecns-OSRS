/**
 * `send*` success means bytes were written and nothing more: a refused op gets
 * an UNSET_MAP_FLAG and no message. Every dispatch therefore snapshots
 * `opFeedback.opRejectedCount`, waits a tick, and confirms an
 * action-appropriate state delta before it reports success.
 */
import type { Sdk } from "./config.ts";
import type { AckPhase, MotorAction } from "./protocol.ts";
import type { ActionResult, BotWorldState } from "../vendor/rs-sdk/sdk/types.ts";

export interface DispatchOutcome {
    ok: boolean;
    phase: AckPhase;
    opRejectedDelta: number;
    message: string;
}

export interface Dispatch extends DispatchOutcome {
    /**
     * Resolves a tick later with the observed truth. Kept separate from the
     * send so the cycle can close immediately: waiting for confirmation inside
     * the cycle costs the brain one conflated state per action.
     */
    verified: Promise<DispatchOutcome>;
}

export class ActionDispatcher {
    /** What the continuation policy repeats: a fly with a slow brain keeps walking. */
    private lastLocomotion: MotorAction | null = null;

    constructor(private readonly sdk: Sdk) {}

    continuation(): MotorAction {
        return this.lastLocomotion ?? { kind: "idle" };
    }

    async dispatch(action: MotorAction): Promise<Dispatch> {
        const before = this.sdk.getState();
        if (!before?.player) return rejected("validation", "no state");
        const rejectedBefore = before.opFeedback?.opRejectedCount ?? 0;

        let sent: ActionResult;
        switch (action.kind) {
            case "idle":
                return { ok: true, phase: "dispatch", opRejectedDelta: 0, message: "idle", verified: settled(true, "idle") };
            case "walk":
            case "flee": {
                this.lastLocomotion = action;
                sent = await this.sdk.sendWalk(action.x, action.z, action.kind === "flee" || action.running !== false);
                break;
            }
            case "attack_fovea": {
                const npc = before.nearbyNpcs.find((n) => n.index === action.npcIndex);
                if (!npc) return rejected("validation", `npc ${action.npcIndex} not in view`);
                const option = npc.optionsWithIndex.find((o) => /attack/i.test(o.text));
                if (!option) return rejected("validation", `${npc.name} has no Attack option`);
                sent = await this.sdk.sendInteractNpc(npc.index, option.opIndex);
                break;
            }
            case "pickup_fovea":
                sent = await this.sdk.sendPickup(action.x, action.z, action.itemId);
                break;
            case "eat": {
                const item = before.inventory.find((i) => i.slot === action.slot);
                if (!item) return rejected("validation", `slot ${action.slot} empty`);
                const option = item.optionsWithIndex.find((o) => /^(eat|drink)$/i.test(o.text));
                if (!option) return rejected("validation", `${item.name} is not edible`);
                sent = await this.sdk.sendUseItem(item.slot, option.opIndex);
                break;
            }
        }

        if (!sent.success) {
            const failure = fail(sent.phase ?? "dispatch", sent.message);
            return { ...failure, verified: Promise.resolve(failure) };
        }
        return {
            ok: true,
            phase: "dispatch",
            opRejectedDelta: 0,
            message: sent.message,
            verified: this.confirm(action, before, rejectedBefore),
        };
    }

    private async confirm(action: MotorAction, before: BotWorldState, rejectedBefore: number): Promise<DispatchOutcome> {
        // A walk requested during tick N is processed on N+1 and first visible
        // on N+2; everything else shows on the next tick.
        const settleTicks = action.kind === "walk" || action.kind === "flee" ? 2 : 1;
        const after = await this.settle(before.tick + settleTicks - 1);
        const opRejectedDelta = (after?.opFeedback?.opRejectedCount ?? rejectedBefore) - rejectedBefore;
        if (opRejectedDelta > 0) {
            return { ok: false, phase: "observation", opRejectedDelta, message: "op refused by the server" };
        }
        if (!after) return { ok: false, phase: "observation", opRejectedDelta, message: "no state after dispatch" };
        const observed = observe(action, before, after);
        return { ok: observed, phase: "observation", opRejectedDelta, message: observed ? "verified" : "no observable effect" };
    }

    /**
     * Wait for the game tick to advance, not merely for the next publication:
     * the SDK republishes state several times within one tick, and an op's
     * effect is not visible until the server has run the tick.
     */
    private async settle(from: number): Promise<BotWorldState | null> {
        try {
            return await this.sdk.waitForCondition((s) => s.tick > from, 5_000);
        } catch {
            return this.sdk.getState();
        }
    }
}

/** Action-appropriate state delta: the only evidence the server acted. */
function observe(action: MotorAction, before: BotWorldState, after: BotWorldState): boolean {
    const a = after.player;
    const b = before.player;
    if (!a || !b) return false;
    switch (action.kind) {
        case "idle":
            return true;
        case "walk":
        case "flee":
            return a.worldX !== b.worldX || a.worldZ !== b.worldZ || (a.worldX === action.x && a.worldZ === action.z);
        case "attack_fovea":
            return a.combat.inCombat || a.combat.targetIndex === action.npcIndex;
        case "pickup_fovea": {
            const count = (s: BotWorldState) =>
                s.inventory.filter((i) => i.id === action.itemId).reduce((n, i) => n + i.count, 0);
            const gone = !after.groundItems.some(
                (i) => i.id === action.itemId && i.x === action.x && i.z === action.z,
            );
            return count(after) > count(before) || gone;
        }
        case "eat": {
            const slotAfter = after.inventory.find((i) => i.slot === action.slot);
            const slotBefore = before.inventory.find((i) => i.slot === action.slot);
            return a.hp > b.hp || slotAfter?.id !== slotBefore?.id;
        }
    }
}

function fail(phase: AckPhase, message: string): DispatchOutcome {
    return { ok: false, phase, opRejectedDelta: 0, message };
}

function rejected(phase: AckPhase, message: string): Dispatch {
    const outcome = fail(phase, message);
    return { ...outcome, verified: Promise.resolve(outcome) };
}

function settled(ok: boolean, message: string): Promise<DispatchOutcome> {
    return Promise.resolve({ ok, phase: "observation" as const, opRejectedDelta: 0, message });
}
