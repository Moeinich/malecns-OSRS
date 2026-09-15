/**
 * Login, tutorial skip and respawn. Nothing behavioural belongs here: every
 * other decision is the network's.
 */
import type { Bot, Sdk } from "./config.ts";

/** The bot spawns behind a blocking modal where sendWalk returns phase "validation". */
export async function clearTutorial(sdk: Sdk, bot: Bot, log: (m: string) => void): Promise<void> {
    for (let attempt = 0; attempt < 3; attempt++) {
        const state = sdk.getState();
        if (!state || (!state.modalOpen && !state.dialog.isOpen)) return;
        const result = await bot.skipTutorial();
        log(`skipTutorial: ${result.success ? "ok" : "failed"} - ${result.message}`);
        if (!result.success) return;
    }
}

/** Death is not a decision the brain gets to make; wait the respawn out. */
export async function awaitRespawn(sdk: Sdk, log: (m: string) => void): Promise<void> {
    if (!sdk.getState()?.player?.isDead) return;
    log("dead - waiting for respawn");
    await sdk.waitForCondition((s) => s.player?.isDead === false, 60_000);
    log("respawned");
}

export async function login(sdk: Sdk, bot: Bot, log: (m: string) => void): Promise<void> {
    await sdk.connect();
    await sdk.waitForCondition((s) => s.inGame && s.player !== null, 60_000);
    await clearTutorial(sdk, bot, log);
    await awaitRespawn(sdk, log);
    const p = sdk.getState()?.player;
    log(`in game as ${p?.name} at (${p?.worldX}, ${p?.worldZ})`);
}
