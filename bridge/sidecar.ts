/**
 * The sidecar owns the rs-sdk connection and speaks NDJSON over a unix socket.
 *
 * Startup order is fixed: listen -> warm pathfinding -> connect the gateway ->
 * emit `ready`. Connecting before warming risks missed pings killing the socket.
 */
import type { Socket } from "bun";
import { mkdir, unlink } from "node:fs/promises";
import { dirname } from "node:path";
import { ActionDispatcher } from "./action-dispatch.ts";
import { loadConfig, loadSdk, type Bot, type BridgeConfig, type Sdk } from "./config.ts";
import {
    PROTOCOL_VERSION,
    encode,
    parseClientMsg,
    type AckMsg,
    type MotorAction,
    type ReadyMsg,
    type ResetMsg,
    type Role,
    type ServerMsg,
} from "./protocol.ts";
import { login, awaitRespawn } from "./scaffold.ts";
import { StateNormalizer, type Observation } from "./state-normalize.ts";

interface ClientCtx {
    id: number;
    role: Role | null;
    buffer: string;
    /** Unwritten tail: a state message exceeds one socket write, and the rest is lost otherwise. */
    outbox: Uint8Array;
}

const ENCODER = new TextEncoder();

type ClientSocket = Socket<ClientCtx>;

const log = (message: string) => console.error(`[bridge] ${message}`);

/** Relative disagreement between configured and observed tick we tolerate in silence. */
const TICK_TOLERANCE = 0.1;
/** Enough distinct ticks for the median to mean anything. */
const TICK_MIN_SAMPLES = 8;
const TICK_WINDOW = 21;
/** A larger gap is a respawn or a stall, not a tick rate. */
const TICK_MAX_GAP = 4;

/**
 * The tick, measured rather than trusted.
 *
 * `RS_TICK_MS` is an env var nothing validates against the engine it describes.
 * If the engine runs at 600 and the env says 400, the deadline is 60% of the
 * wrong number, the continuation policy fires early every tick, and every
 * `brainOverrun` we report is wrong — silently, because nothing compares them.
 */
class TickMeter {
    private readonly intervals: number[] = [];
    private lastTick = -1;
    private lastAt = 0;
    /** Rolling median of the per-tick interval, once there are enough samples. */
    observedMs: number | null = null;

    /**
     * Feed every state publication. The SDK republishes ~8x per game tick, so
     * only a change in `tick` is a sample; dividing by the tick delta keeps
     * conflated or skipped ticks from reading as a slower server.
     */
    sample(tick: number, now: number): void {
        const gap = tick - this.lastTick;
        if (this.lastTick >= 0 && gap > 0 && gap <= TICK_MAX_GAP) {
            this.intervals.push((now - this.lastAt) / gap);
            if (this.intervals.length > TICK_WINDOW) this.intervals.shift();
            if (this.intervals.length >= TICK_MIN_SAMPLES) this.observedMs = median(this.intervals);
        }
        if (gap > 0) {
            this.lastTick = tick;
            this.lastAt = now;
        }
    }
}

function median(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    return sorted.length % 2 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

class Sidecar {
    private readonly normalizer = new StateNormalizer();
    private readonly clients = new Set<ClientSocket>();
    private readonly meter = new TickMeter();
    private deadlineMs: number;
    private dispatcher!: ActionDispatcher;
    private sdk!: Sdk;
    private bot!: Bot;
    private ready: ReadyMsg | null = null;
    private nextClientId = 1;

    private revision = 0;
    private cycle: { revision: number; commanded: boolean; timer: ReturnType<typeof setTimeout> } | null = null;
    private pending: Observation | null = null;
    private droppedSinceLast = 0;
    /** The game tick the open (or last) cycle answered: at most one action per tick. */
    private lastCycleTick = -1;
    private awaitingRespawn = false;
    /** A reset owns the bot: no cycle, no deadline, no state goes out until it lands. */
    private resetting = false;

    /** Health counters, for the dashboard. */
    readonly health = { droppedStates: 0, brainOverrun: 0, opRejected: 0 };
    /** Configured vs observed tick, and the one the deadline is actually derived from. */
    readonly tick: { configuredMs: number; observedMs: number | null; effectiveMs: number };

    constructor(private readonly cfg: BridgeConfig) {
        this.deadlineMs = Math.round(cfg.tickMs * cfg.deadlineFraction);
        this.tick = { configuredMs: cfg.tickMs, observedMs: null, effectiveMs: cfg.tickMs };
    }

    async start(): Promise<{ sdk: Sdk; bot: Bot }> {
        await this.listen();

        const { BotSDK, BotActions, initPathfinding } = await loadSdk(this.cfg);
        const warmStart = Date.now();
        initPathfinding();
        log(`pathfinding warm in ${Date.now() - warmStart} ms`);

        const sdk = new BotSDK({
            botUsername: this.cfg.username,
            password: this.cfg.password,
            gatewayUrl: this.cfg.gatewayUrl,
            connectionMode: this.cfg.mode,
            autoLaunchBrowser: false,
        });
        const bot = new BotActions(sdk);
        this.sdk = sdk;
        this.bot = bot;
        this.dispatcher = new ActionDispatcher(sdk);
        await login(sdk, bot, log);

        sdk.onStateUpdate((raw) => this.onState(sdk, raw));
        this.ready = {
            t: "ready",
            protocol: PROTOCOL_VERSION,
            username: this.cfg.username,
            mode: this.cfg.mode,
            role: "brain",
            tickMs: this.cfg.tickMs,
            pathfindingWarm: true,
        };
        this.broadcast(this.ready);
        log(`ready on ${this.cfg.socketPath}`);
        return { sdk, bot };
    }

    private async listen(): Promise<void> {
        await mkdir(dirname(this.cfg.socketPath), { recursive: true });
        await unlink(this.cfg.socketPath).catch(() => {});
        Bun.listen<ClientCtx>({
            unix: this.cfg.socketPath,
            socket: {
                open: (socket) => {
                    socket.data = { id: this.nextClientId++, role: null, buffer: "", outbox: new Uint8Array(0) };
                    this.clients.add(socket);
                },
                close: (socket) => {
                    this.clients.delete(socket);
                },
                drain: (socket) => {
                    this.flush(socket);
                },
                error: (socket, error) => {
                    log(`client ${socket.data.id} error: ${error.message}`);
                    this.clients.delete(socket);
                },
                data: (socket, chunk) => {
                    socket.data.buffer += chunk.toString();
                    const lines = socket.data.buffer.split("\n");
                    socket.data.buffer = lines.pop() ?? "";
                    for (const line of lines) {
                        if (line.trim()) void this.onLine(socket, line);
                    }
                },
            },
        });
    }

    private async onLine(socket: ClientSocket, line: string): Promise<void> {
        const msg = parseClientMsg(line);
        if (!msg) return this.send(socket, { t: "error", message: "malformed message" });

        if (msg.t === "hello") {
            if (msg.role === "brain" && [...this.clients].some((c) => c !== socket && c.data.role === "brain")) {
                return this.send(socket, { t: "error", message: "a brain is already attached" });
            }
            socket.data.role = msg.role;
            if (this.ready) this.send(socket, { ...this.ready, role: msg.role });
            return;
        }

        const cmdId = msg.t === "noop" ? -1 : msg.cmdId;
        if (socket.data.role !== "brain") {
            return this.ack(socket, cmdId, false, "role", 0, "observe clients cannot act");
        }
        if (msg.t === "reset") return this.reset(socket, msg);
        // Stale-revision rejection: the brain answered a world that no longer exists.
        if (!this.cycle || this.cycle.revision !== msg.revision) {
            return this.ack(socket, cmdId, false, "stale", 0, `current revision ${this.cycle?.revision ?? "none"}`);
        }
        // At most one action per tick.
        if (this.cycle.commanded) {
            return this.ack(socket, cmdId, false, "duplicate", 0, "this revision was already answered");
        }

        this.cycle.commanded = true;
        clearTimeout(this.cycle.timer);

        if (msg.t === "noop") return this.closeCycle();

        const dispatch = await this.dispatcher.dispatch(msg.action);
        this.closeCycle();
        if (!dispatch.ok) {
            return this.ack(socket, msg.cmdId, false, dispatch.phase, 0, dispatch.message);
        }
        // The ack lands a tick later, carrying the observed truth rather than
        // "bytes were written" - a refused op is silent on the wire.
        const outcome = await dispatch.verified;
        this.health.opRejected += outcome.opRejectedDelta;
        this.ack(socket, msg.cmdId, outcome.ok, outcome.phase, outcome.opRejectedDelta, outcome.message);
    }

    /**
     * Walk to a fixed tile with the cycle suspended: the continuation policy
     * would otherwise issue a walk every tick and fight `walkTo` for the bot.
     */
    private async reset(socket: ClientSocket, msg: ResetMsg): Promise<void> {
        if (this.resetting) {
            return this.ack(socket, msg.cmdId, false, "reset", 0, "a reset is already running");
        }
        this.resetting = true;
        this.pending = null;
        this.droppedSinceLast = 0;
        this.closeCycle();
        const started = Date.now();
        let result: { success: boolean; message: string };
        try {
            result = await this.bot.walkTo(msg.x, msg.z, 1);
        } catch (error) {
            result = { success: false, message: error instanceof Error ? error.message : String(error) };
        } finally {
            this.resetting = false;
        }
        const player = this.sdk.getState()?.player;
        log(
            `reset -> (${msg.x}, ${msg.z}): ${result.success ? "ok" : "failed"} ${result.message} ` +
                `in ${Date.now() - started} ms`,
        );
        this.send(socket, {
            t: "ack",
            cmdId: msg.cmdId,
            ok: result.success,
            phase: "reset",
            opRejectedDelta: 0,
            message: result.message,
            x: player?.worldX ?? -1,
            z: player?.worldZ ?? -1,
        });
    }

    private onState(sdk: Sdk, raw: Parameters<Parameters<Sdk["onStateUpdate"]>[0]>[0]): void {
        if (this.resetting) return;
        if (raw.player?.isDead) {
            if (!this.awaitingRespawn) {
                this.awaitingRespawn = true;
                void awaitRespawn(sdk, log).finally(() => {
                    this.awaitingRespawn = false;
                });
            }
            return;
        }
        this.measureTick(raw.tick);
        // Conflation, depth 1: hold exactly one pending state, never a queue.
        // A queue would put the brain progressively behind reality while it
        // believes it is current.
        if (this.pending) {
            this.droppedSinceLast++;
            this.health.droppedStates++;
        }
        this.pending = this.normalizer.normalize(raw);
        this.beginIfDue();
    }

    /**
     * Prefer what the engine does over what the env var claims: a deadline built
     * on the wrong tick corrupts every overrun measurement it produces.
     */
    private measureTick(tick: number): void {
        this.meter.sample(tick, Date.now());
        const observed = this.meter.observedMs;
        if (observed === null) return;
        this.tick.observedMs = observed;
        // Hysteresis against the value in force, not against the configured one:
        // the median jitters a few ms either side and would otherwise re-decide,
        // and re-log, every tick.
        const inForce = this.tick.effectiveMs;
        if (Math.abs(observed - inForce) / inForce <= TICK_TOLERANCE) return;

        const agrees = Math.abs(observed - this.cfg.tickMs) / this.cfg.tickMs <= TICK_TOLERANCE;
        const effective = agrees ? this.cfg.tickMs : Math.round(observed / 10) * 10;
        this.tick.effectiveMs = effective;
        this.deadlineMs = Math.round(effective * this.cfg.deadlineFraction);
        log(
            agrees
                ? `tick back in agreement: configured ${this.cfg.tickMs} ms, observed ${observed.toFixed(1)} ms`
                : `TICK MISMATCH: configured ${this.cfg.tickMs} ms (RS_TICK_MS), observed ` +
                  `${observed.toFixed(1)} ms. Using the observed ${effective} ms for the deadline ` +
                  `(${this.deadlineMs} ms). Start the engine with NODE_TICKRATE=${this.cfg.tickMs}, ` +
                  `or set RS_TICK_MS=${effective} to match the engine that is running.`,
        );
    }

    /** The SDK republishes several times per game tick; the brain acts on the first of each. */
    private beginIfDue(): void {
        if (this.resetting || this.cycle || !this.pending || this.pending.state.tick === this.lastCycleTick) {
            return;
        }
        const observation = this.pending;
        this.pending = null;
        this.lastCycleTick = observation.state.tick;
        this.beginCycle(observation);
    }

    private beginCycle(observation: Observation): void {
        const revision = ++this.revision;
        const dropped = this.droppedSinceLast;
        this.droppedSinceLast = 0;
        this.cycle = {
            revision,
            commanded: false,
            timer: setTimeout(() => void this.onDeadline(revision), this.deadlineMs),
        };
        this.broadcast({
            t: "state",
            revision,
            tick: observation.state.tick,
            droppedSinceLast: dropped,
            deadlineMs: this.deadlineMs,
            tickMs: this.tick.effectiveMs,
            observedTickMs: this.tick.observedMs,
            state: observation.state,
        });
        if (observation.combatEvents.length > 0 || Object.keys(observation.xpDelta).length > 0) {
            this.broadcast({
                t: "reward",
                revision,
                combatEvents: observation.combatEvents,
                xpDelta: observation.xpDelta,
            });
        }
    }

    /** The brain missed its deadline: repeat the last locomotion rather than block the gateway. */
    private async onDeadline(revision: number): Promise<void> {
        if (!this.cycle || this.cycle.revision !== revision || this.cycle.commanded) return;
        this.cycle.commanded = true;
        this.health.brainOverrun++;
        const action: MotorAction = this.dispatcher.continuation();
        if (action.kind === "idle") return this.closeCycle();
        const dispatch = await this.dispatcher.dispatch(action);
        this.closeCycle();
        this.health.opRejected += (await dispatch.verified).opRejectedDelta;
    }

    private closeCycle(): void {
        if (this.cycle) clearTimeout(this.cycle.timer);
        this.cycle = null;
        this.beginIfDue();
    }

    private ack(
        socket: ClientSocket,
        cmdId: number,
        ok: boolean,
        phase: AckMsg["phase"],
        opRejectedDelta: number,
        message: string,
    ): void {
        this.send(socket, { t: "ack", cmdId, ok, phase, opRejectedDelta, message });
    }

    private send(socket: ClientSocket, msg: ServerMsg): void {
        this.enqueue(socket, ENCODER.encode(encode(msg)));
    }

    private broadcast(msg: ServerMsg): void {
        const line = ENCODER.encode(encode(msg));
        for (const client of this.clients) {
            if (client.data.role) this.enqueue(client, line);
        }
    }

    private enqueue(socket: ClientSocket, bytes: Uint8Array): void {
        const tail = socket.data.outbox;
        const outbox = new Uint8Array(tail.length + bytes.length);
        outbox.set(tail);
        outbox.set(bytes, tail.length);
        socket.data.outbox = outbox;
        this.flush(socket);
    }

    private flush(socket: ClientSocket): void {
        while (socket.data.outbox.length > 0) {
            const written = socket.write(socket.data.outbox);
            if (written <= 0) return;
            socket.data.outbox = socket.data.outbox.subarray(written);
        }
    }
}

/**
 * Walks the player +5 tiles through the real socket/cmd path, so the lane is
 * verifiable without a Python client.
 */
async function selftest(cfg: BridgeConfig): Promise<never> {
    /** +5 tiles, with fallbacks: the bot may already be standing against a wall. */
    const offsets = [
        [5, 0],
        [-5, 0],
        [0, 5],
        [0, -5],
    ] as const;
    let origin: { x: number; z: number } | null = null;
    let offsetIndex = 0;
    let cmdId = 0;
    let ticks = 0;
    let buffer = "";

    const finish = (code: number, message: string): never => {
        console.log(message);
        process.exit(code);
    };
    const timeout = setTimeout(() => finish(1, "TIMEOUT: never arrived"), 60_000);

    const socket = await Bun.connect<{}>({
        unix: cfg.socketPath,
        socket: {
            open: (s) => {
                s.write(JSON.stringify({ t: "hello", protocol: PROTOCOL_VERSION, role: "brain" }) + "\n");
            },
            data: (s, chunk) => {
                buffer += chunk.toString();
                const lines = buffer.split("\n");
                buffer = lines.pop() ?? "";
                for (const line of lines) {
                    if (!line.trim()) continue;
                    const msg = JSON.parse(line) as ServerMsg;
                    if (msg.t === "ready") {
                        console.log(`READY ${msg.username} mode=${msg.mode} tickMs=${msg.tickMs} warm=${msg.pathfindingWarm}`);
                        continue;
                    }
                    if (msg.t === "ack" && !msg.ok) {
                        console.log(`ack cmd=${msg.cmdId} REJECTED phase=${msg.phase} (${msg.message})`);
                        if (msg.phase === "validation" && ++offsetIndex >= offsets.length) {
                            clearTimeout(timeout);
                            finish(1, "FAILED: no walkable direction");
                        }
                        continue;
                    }
                    if (msg.t !== "state") continue;
                    const player = msg.state.player;
                    if (!player) continue;
                    origin ??= { x: player.x, z: player.z };
                    const offset = offsets[offsetIndex]!;
                    const target = { x: origin.x + offset[0], z: origin.z + offset[1] };
                    ticks++;
                    console.log(
                        `tick=${msg.state.tick} rev=${msg.revision} pos=(${player.x}, ${player.z}) ` +
                            `target=(${target.x}, ${target.z}) dropped=${msg.droppedSinceLast}`,
                    );
                    if (player.x === target.x && player.z === target.z) {
                        clearTimeout(timeout);
                        finish(0, `ARRIVED at (${player.x}, ${player.z}) after ${ticks} observed ticks`);
                    }
                    s.write(
                        JSON.stringify({
                            t: "cmd",
                            cmdId: ++cmdId,
                            revision: msg.revision,
                            action: { kind: "walk", x: target.x, z: target.z, running: false },
                        }) + "\n",
                    );
                }
            },
        },
    });
    socket.data = {};
    return await new Promise<never>(() => {});
}

const cfg = loadConfig();
const sidecar = new Sidecar(cfg);
await sidecar.start();
if (process.argv.includes("--selftest")) await selftest(cfg);
