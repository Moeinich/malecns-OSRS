/**
 * Connection config lives here, not in the submodule's `bots/<name>/bot.env`:
 * that file is inside vendor/ and invisible to our history, and a blank SERVER
 * there resolves to port 80 while the gateway URL is derived from it.
 */
import type { SDKConnectionMode } from "../vendor/rs-sdk/sdk/types.ts";

export interface BridgeConfig {
    username: string;
    password: string;
    server: string;
    gatewayUrl: string;
    sdkPath: string;
    socketPath: string;
    mode: SDKConnectionMode;
    tickMs: number;
    /** Fraction of a tick the brain gets before the continuation policy fires. */
    deadlineFraction: number;
}

const DEFAULT_SDK_PATH = new URL("../vendor/rs-sdk/sdk/", import.meta.url).href;

/**
 * Real Old School RuneScape runs at 600 ms. LostCity's engine defaults to 400
 * (`Environment.ts:54`), so every process in the stack has to be told 600 —
 * `scripts/dev.sh` passes `NODE_TICKRATE`, this is the sidecar's half. A low
 * tickrate for fast training runs stays available through `RS_TICK_MS`, which
 * must then match whatever the engine was started with.
 */
export const DEFAULT_TICK_MS = 600;

export function loadConfig(): BridgeConfig {
    const env = process.env;
    const username = env.RS_BOT_USERNAME || "flybot01";
    const server = env.RS_SERVER || "localhost:8888";
    const host = server.split(":")[0] || "localhost";
    return {
        username,
        password: env.RS_PASSWORD || "",
        server,
        gatewayUrl: env.RS_GATEWAY_URL || `ws://${host}:7780`,
        sdkPath: env.RS_SDK_PATH ? new URL("./", `file://${env.RS_SDK_PATH}/`).href : DEFAULT_SDK_PATH,
        socketPath: env.RS_BRIDGE_SOCKET || `/tmp/malecns-osrs/${username}.sock`,
        mode: env.RS_MODE === "observe" ? "observe" : "control",
        tickMs: Number(env.RS_TICK_MS) || DEFAULT_TICK_MS,
        deadlineFraction: 0.6,
    };
}

type SdkModule = typeof import("../vendor/rs-sdk/sdk/index.ts");
type ActionsModule = typeof import("../vendor/rs-sdk/sdk/actions.ts");
type PathfindingModule = typeof import("../vendor/rs-sdk/sdk/pathfinding.ts");

export interface LoadedSdk {
    BotSDK: SdkModule["BotSDK"];
    BotActions: ActionsModule["BotActions"];
    initPathfinding: PathfindingModule["initPathfinding"];
}

/** rs-sdk is not on npm; it is loaded from the submodule, or RS_SDK_PATH. */
export async function loadSdk(cfg: BridgeConfig): Promise<LoadedSdk> {
    const [index, actions, pathfinding] = await Promise.all([
        import(`${cfg.sdkPath}index.ts`) as Promise<SdkModule>,
        import(`${cfg.sdkPath}actions.ts`) as Promise<ActionsModule>,
        import(`${cfg.sdkPath}pathfinding.ts`) as Promise<PathfindingModule>,
    ]);
    return { BotSDK: index.BotSDK, BotActions: actions.BotActions, initPathfinding: pathfinding.initPathfinding };
}

export type Sdk = InstanceType<SdkModule["BotSDK"]>;
export type Bot = InstanceType<ActionsModule["BotActions"]>;
