/**
 * Publishes what the bot web client actually renders: headless Chrome drives
 * `/bot`, and the newest screencast frame is written out as a JPEG.
 *
 * A file, not the NDJSON socket. Base64 frames would bloat every state message,
 * the feed (~10 fps) and the brain (~1.7 fps at a 600 ms tick) want different
 * rates, and nothing on the shared control path has to change. Temp file plus
 * `rename`, so a reader never sees a torn frame.
 *
 * This REPLACES the lite client rather than joining it: the gateway pairs one
 * client per username, and a second would fight the first for the account.
 */
import { mkdir, rename, writeFile } from "node:fs/promises";

import puppeteer from "../vendor/rs-sdk/node_modules/puppeteer/lib/esm/puppeteer/puppeteer.js";
import { loadConfig } from "./config.ts";

/** ~10 fps. Chrome emits a frame per repaint; the rest are acked and dropped. */
const MIN_FRAME_INTERVAL_MS = 100;
const VIEWPORT = { width: 1024, height: 768 };
const REPORT_EVERY_MS = 10_000;
/** A cold client packs its cache before the login screen even appears. */
const LOGIN_TIMEOUT_MS = 120_000;

const cfg = loadConfig();
const runDir = process.env.FLYBRAIN_RUN_DIR || "/tmp/malecns-osrs";
const framePath = `${runDir}/game.jpg`;
const tmpPath = `${framePath}.tmp`;

if (!cfg.password) {
    console.error("game-feed: no RS_PASSWORD; the client will sit on the login screen");
}
await mkdir(runDir, { recursive: true });

const browser = await puppeteer.launch({
    headless: true,
    // The Chrome-for-Testing download is optional (and half-extracted on this
    // machine); the installed Chrome renders the client identically.
    channel: "chrome",
    defaultViewport: VIEWPORT,
});
const page = await browser.newPage();
page.on("console", (msg) => console.log(`client: ${msg.text()}`));

const url = new URL(`http://${cfg.server}/bot`);
url.searchParams.set("bot", cfg.username);
url.searchParams.set("password", cfg.password);
await page.goto(url.href, { waitUntil: "domcontentloaded" });

// Ready means in the game, not "a page loaded": an unbuilt client bundle
// renders a black loading screen forever, and a feed of that is worse than no
// feed at all.
try {
    await page.waitForFunction("window.gameClient?.ingame === true", { timeout: LOGIN_TIMEOUT_MS });
} catch {
    console.error(
        `game-feed: ${cfg.username} never reached the game within ${LOGIN_TIMEOUT_MS / 1000} s. ` +
            "If the page is blank, the bot client is not built: " +
            "(cd vendor/rs-sdk/server/webclient && BUILD_MODE=bot bun run bundle.ts)",
    );
    await browser.close();
    process.exit(1);
}

let lastWrite = 0;
let writing = false;
let rendered = 0;
let published = 0;

async function publish(data: Buffer): Promise<void> {
    try {
        await writeFile(tmpPath, data);
        await rename(tmpPath, framePath);
        published++;
    } catch (err) {
        console.error(`game-feed: write failed: ${err}`);
    }
}

const SCREENCAST = {
    format: "jpeg" as const,
    quality: 60,
    maxWidth: VIEWPORT.width,
    maxHeight: VIEWPORT.height,
};
/** The client re-logs-in by itself; the screencast does not come back with it. */
const STALL_MS = 15_000;

type Cdp = Awaited<ReturnType<typeof page.createCDPSession>>;

function attach(session: Cdp): void {
    session.on("Page.screencastFrame", (frame) => {
        // Ack first and unconditionally, or Chrome stops sending.
        void session.send("Page.screencastFrameAck", { sessionId: frame.sessionId }).catch(() => {});
        lastFrameAt = Date.now();
        rendered++;
        const now = Date.now();
        if (writing || now - lastWrite < MIN_FRAME_INTERVAL_MS) return;
        lastWrite = now;
        writing = true;
        void publish(Buffer.from(frame.data, "base64")).finally(() => {
            writing = false;
        });
    });
}

let cdp = await page.createCDPSession();
let lastFrameAt = Date.now();
let announcedLoggedOut = false;
attach(cdp);
await cdp.send("Page.startScreencast", SCREENCAST);

async function rearm(): Promise<void> {
    const stalled = Date.now() - lastFrameAt;
    if (stalled < STALL_MS) return;

    let inGame = false;
    try {
        inGame = (await page.evaluate("window.gameClient?.ingame === true")) === true;
    } catch {
        // page busy or navigating; try again next interval
    }
    if (!inGame) {
        if (!announcedLoggedOut) {
            console.log("game-feed: page left the game; waiting");
            announcedLoggedOut = true;
        }
        return;
    }
    announcedLoggedOut = false;

    try {
        await cdp.send("Page.stopScreencast").catch(() => {});
        await cdp.send("Page.startScreencast", SCREENCAST);
    } catch {
        cdp = await page.createCDPSession();
        attach(cdp);
        await cdp.send("Page.startScreencast", SCREENCAST);
        console.log("game-feed: CDP session was dead; reattached");
    }
    lastFrameAt = Date.now();
    console.log(
        `game-feed: screencast re-armed after ${(stalled / 1000).toFixed(0)} s without frames (page in game)`,
    );
}

console.log(`game-feed: ready on ${framePath} for ${cfg.username}@${cfg.server}`);

// Puppeteer spawns Chrome detached, so it is in its own process group and the
// supervisor's killpg never reaches it. Close it here, and make sure.
for (const sig of ["SIGTERM", "SIGINT", "SIGHUP"] as const) {
    process.on(sig, () => {
        const pid = browser.process()?.pid;
        void browser
            .close()
            .catch(() => {})
            .finally(() => {
                if (pid !== undefined) {
                    try {
                        process.kill(pid, "SIGKILL");
                    } catch {
                        // already gone
                    }
                }
                process.exit(0);
            });
    });
}

setInterval(() => {
    const fps = (published / REPORT_EVERY_MS) * 1000;
    console.log(`game-feed: ${fps.toFixed(1)} fps published, ${rendered} rendered`);
    rendered = 0;
    published = 0;
    void rearm().catch((err) => console.error(`game-feed: re-arm failed: ${err}`));
}, REPORT_EVERY_MS);
