"""Supervisor for the whole local stack: engine, gateway, lite client, sidecar.

Single source of truth for how the stack is started; `scripts/dev.sh` and
`scripts/stop.sh` are thin wrappers over `up --no-wait` and `down`. The one
exception is `vendor/rs-sdk/bots/<name>/bot.env` (SERVER, GATEWAY_URL,
TELEMETRY), which stays in `scripts/bot-env.sh` and is invoked from here rather
than reimplemented.

Two tiers. The long tier (engine, gateway, lite client) is slow — a cold engine
packs the cache for ~40 s — and reconnecting in control mode pre-empts the
bot's current controller, so it is restarted rarely. The short tier is cheap and
recycled constantly; `restart-short` never touches the long tier.

Children are spawned into their own process group and the *group* is signalled
on teardown, which is what keeps bun's own subprocesses from being orphaned.
Only groups this supervisor started (or was told to `--adopt`) are ever
signalled; a stack someone else brought up is attached to and left alone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RS_SDK = REPO_ROOT / "vendor" / "rs-sdk"
RUN_DIR = Path(os.environ.get("FLYBRAIN_RUN_DIR", "/tmp/malecns-osrs"))
STATE_PATH = RUN_DIR / "supervisor.json"

DEFAULT_TICKRATE = 600
DEFAULT_BOT = "flybot01"
GRACE_SECONDS = 5.0

LONG = "long"
SHORT = "short"

LITE = "lite"
BROWSER = "browser"


@dataclass(frozen=True)
class Service:
    name: str
    tier: str
    argv: list[str]
    cwd: Path
    ready: str | None = None
    ready_timeout: float = 30.0
    #: TCP port to probe for an already-running instance.
    port: int | None = None
    #: `pgrep -f` pattern, for services with no listening port.
    pattern: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    #: Command run once before the service starts (bot.env generation).
    prepare: list[str] | None = None

    def log_path(self, run_dir: Path = RUN_DIR) -> Path:
        return run_dir / "logs" / f"{self.name}.log"


def _client_service(client: str, bot: str) -> Service:
    """The one game client. The gateway pairs one per username, so `browser`
    replaces `lite` rather than joining it — connecting a second would fight the
    first for the account.

    Lite has no renderer by design, so the HUD's game panel is empty under it;
    `browser` is headless Chrome on `/bot`, publishing frames for
    `flybrain.gamefeed`. It costs a Chromium, which is why it is not the
    default, and it needs the client bundle built once
    (`BUILD_MODE=bot bun run bundle.ts` in `vendor/rs-sdk/server/webclient`);
    `scripts/webclient-bundle.sh` does that automatically, idempotently, then
    delegates to `bot-env.sh` for the password.
    """
    prepare = [str(REPO_ROOT / "scripts" / "bot-env.sh"), bot]
    if client == BROWSER:
        return Service(
            name="browser",
            tier=LONG,
            argv=["bun", "bridge/game-feed.ts"],
            cwd=REPO_ROOT,
            env={"RS_BOT_USERNAME": bot},
            pattern="bridge/game-feed.ts",
            ready="game-feed: ready on",
            ready_timeout=150.0,
            prepare=[str(REPO_ROOT / "scripts" / "webclient-bundle.sh"), bot],
        )
    return Service(
        name="lite",
        tier=LONG,
        argv=["bun", "src/lite/runner.ts", bot],
        cwd=RS_SDK / "server" / "webclient",
        pattern=f"src/lite/runner.ts {bot}",
        ready="Gateway connected, registering as",
        prepare=prepare,
    )


def default_services(
    tickrate: int = DEFAULT_TICKRATE, bot: str = DEFAULT_BOT, client: str = LITE
) -> list[Service]:
    """The real stack, in start order.

    `BUILD_VERIFY=false` or the engine aborts on a `.loc` checksum mismatch.
    `NODE_TICKRATE` because the engine defaults to 400 while real OSRS is 600
    (`::speed` cannot fix that from an ordinary account — it needs
    staffModLevel 4), and the sidecar's `RS_TICK_MS` has to agree with it.
    """
    return [
        Service(
            name="engine",
            tier=LONG,
            argv=["bun", "run", "src/app.ts"],
            cwd=RS_SDK / "server" / "engine",
            env={"BUILD_VERIFY": "false", "NODE_TICKRATE": str(tickrate)},
            port=8888,
            ready="World ready",
            ready_timeout=120.0,
        ),
        Service(
            name="gateway",
            tier=LONG,
            argv=["bun", "run", "gateway"],
            cwd=RS_SDK / "server" / "gateway",
            port=7780,
            ready="Gateway running",
        ),
        _client_service(client, bot),
        Service(
            name="sidecar",
            tier=SHORT,
            argv=["bun", "bridge/sidecar.ts"],
            cwd=REPO_ROOT,
            env={"RS_TICK_MS": str(tickrate), "RS_BOT_USERNAME": bot},
            pattern="bridge/sidecar.ts",
            ready="ready on",
        ),
        # SEAM: the brain (`python -m flybrain.loop.run`) and the dashboard join
        # the short tier here, as two more Service entries. Both are owned by
        # other lanes; wiring them in is an integration pass, not this file's
        # job. Nothing else has to change — tier membership is the whole
        # contract, and `restart-short` will pick them up for free.
    ]


def _prepare_env(service: Service) -> dict[str, str]:
    if not service.prepare:
        return {}
    out = subprocess.run(service.prepare, check=True, capture_output=True, text=True)
    return dict(
        line.split("=", 1)
        for line in out.stdout.splitlines()
        if re.match(r"^[A-Z][A-Z0-9_]*=", line)
    )


def port_listening(port: int) -> bool:
    return (
        subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def pgrep(pattern: str) -> list[int]:
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=False)
    return [int(p) for p in out.stdout.split()]


def probe(service: Service) -> int | None:
    """PID of an already-running instance, or None."""
    if service.port is not None and port_listening(service.port):
        pids = pgrep(" ".join(service.argv))
        return pids[0] if pids else -1
    if service.pattern:
        pids = [p for p in pgrep(service.pattern) if p != os.getpid()]
        if pids:
            return pids[0]
    return None


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


class ServiceFailed(RuntimeError):
    pass


class Supervisor:
    def __init__(
        self,
        services: list[Service],
        run_dir: Path = RUN_DIR,
        out=sys.stdout,
    ) -> None:
        self.services = services
        self.run_dir = run_dir
        self.state_path = run_dir / "supervisor.json"
        self.out = out
        self.procs: dict[str, subprocess.Popen] = {}
        self.reported_dead: set[str] = set()

    # --- state file -----------------------------------------------------

    def _load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except OSError, ValueError:
            return {}

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2))

    def _record(self, service: Service, pgid: int, adopted: bool) -> None:
        state = self._load_state()
        state[service.name] = {
            "pgid": pgid,
            "tier": service.tier,
            "log": str(service.log_path(self.run_dir)),
            "adopted": adopted,
        }
        self._save_state(state)

    def _forget(self, name: str) -> None:
        state = self._load_state()
        if state.pop(name, None) is not None:
            self._save_state(state)

    def _say(self, message: str) -> None:
        print(message, file=self.out, flush=True)

    # --- lifecycle ------------------------------------------------------

    def start(self, service: Service, adopt: bool = False) -> str:
        existing = probe(service)
        if existing is not None:
            if adopt and existing > 0:
                try:
                    self._record(service, os.getpgid(existing), adopted=True)
                    return "adopted"
                except ProcessLookupError:
                    pass
            else:
                return "attached"

        log = service.log_path(self.run_dir)
        log.parent.mkdir(parents=True, exist_ok=True)

        # `KEY=VALUE` lines from prepare become child env: that is how the bot
        # password reaches the client without anything here parsing bot.env.
        prepared = _prepare_env(service)

        with open(log, "wb") as handle:
            proc = subprocess.Popen(
                service.argv,
                cwd=service.cwd,
                env={**os.environ, **prepared, **service.env},
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        self.procs[service.name] = proc
        self._record(service, proc.pid, adopted=False)

        if service.ready:
            self._await_ready(service, proc)
        return "started"

    def _await_ready(self, service: Service, proc: subprocess.Popen) -> None:
        log = service.log_path(self.run_dir)
        deadline = time.monotonic() + service.ready_timeout
        assert service.ready
        while time.monotonic() < deadline:
            if service.ready in tail(log, 10_000):
                return
            if proc.poll() is not None:
                raise ServiceFailed(
                    f"{service.name}: exited {proc.returncode} before it was ready\n"
                    f"  log: {log}\n{tail(log)}"
                )
            time.sleep(0.25)
        raise ServiceFailed(
            f"{service.name}: no {service.ready!r} within {service.ready_timeout:.0f}s\n"
            f"  log: {log}\n{tail(log)}"
        )

    def up(self, tiers: tuple[str, ...] = (LONG, SHORT), adopt: bool = False) -> None:
        for service in self.services:
            if service.tier not in tiers:
                continue
            status = self.start(service, adopt=adopt)
            self._say(f"{service.name}: {status}")

    def stop(self, tiers: tuple[str, ...] = (LONG, SHORT), grace: float = GRACE_SECONDS) -> None:
        """Signal the process groups we own, in reverse start order. Idempotent.

        Reverse order because a dependent that outlives what it talks to just
        dies on its own and looks like a crash.
        """
        state = self._load_state()
        order = [s.name for s in reversed(self.services)]
        ordered = sorted(state.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else -1)
        for name, entry in ordered:
            if entry.get("tier") not in tiers:
                continue
            self._say(f"{name}: {self._kill_group(name, entry['pgid'], grace)}")
            self._forget(name)

    def _kill_group(self, name: str, pgid: int, grace: float) -> str:
        proc = self.procs.pop(name, None)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            if proc is not None:
                proc.wait()
            return "not running"
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if proc is not None:
                proc.poll()
            if not group_alive(pgid):
                if proc is not None:
                    proc.wait()
                return "stopped"
            time.sleep(0.1)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError, PermissionError:
            # Darwin answers EPERM for a group whose only member is a zombie:
            # already dead, just not yet reaped by whoever owns it.
            pass
        if proc is not None:
            proc.wait()
        return "killed"

    def restart_short(self, adopt: bool = False) -> None:
        self.stop(tiers=(SHORT,))
        self.up(tiers=(SHORT,), adopt=adopt)

    # --- observation ----------------------------------------------------

    def deaths(self) -> list[str]:
        """Children that exited on their own since the last call."""
        dead = []
        for name, proc in list(self.procs.items()):
            code = proc.poll()
            if code is None or name in self.reported_dead:
                continue
            self.reported_dead.add(name)
            service = next(s for s in self.services if s.name == name)
            log = service.log_path(self.run_dir)
            why = ""
            if name == "lite":
                # The lite client ends with a typed SessionEnd; anything other
                # than 'stopped' means the bot has left the game for good.
                for line in reversed(tail(log, 200).splitlines()):
                    if "Game session ended" in line:
                        why = f"\n  {line.strip()}"
                        break
            dead.append(f"{name}: DIED (exit {code})\n  log: {log}{why}")
        return dead

    def status(self) -> str:
        """Every `self.services` entry, plus any state-file row with no match
        among them — a service started under different flags (e.g. `browser`
        when invoked without `--client browser`), the way `stop` already
        tolerates via `order.index(...) if ... else -1`.
        """
        state = self._load_state()
        rows = [("SERVICE", "TIER", "STATE", "PGID", "PORT", "LOG")]
        known = {service.name for service in self.services}
        for service in self.services:
            entry = state.get(service.name)
            running = probe(service) is not None
            if entry and group_alive(entry["pgid"]):
                what = "adopted" if entry.get("adopted") else "ours"
            elif entry:
                what = "DEAD"
            else:
                what = "running (not ours)" if running else "stopped"
            rows.append(
                (
                    service.name,
                    service.tier,
                    what,
                    str(entry["pgid"]) if entry else "-",
                    str(service.port or "-"),
                    str(service.log_path(self.run_dir)),
                )
            )
        for name, entry in state.items():
            if name in known:
                continue
            alive = group_alive(entry["pgid"])
            what = ("adopted" if entry.get("adopted") else "ours") if alive else "DEAD"
            rows.append(
                (
                    name,
                    entry.get("tier", "-"),
                    f"{what} (not in current flags)",
                    str(entry["pgid"]),
                    "-",
                    entry.get("log", "-"),
                )
            )
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        return "\n".join(
            "  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in rows
        )

    def supervise(self) -> int:
        """Foreground: report deaths until a signal, then tear the stack down."""
        stopping = False

        def handler(signum, _frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

        while not stopping:
            for line in self.deaths():
                self._say(line)
            time.sleep(0.5)

        self._say("")
        self.stop()
        return 1 if self.reported_dead else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flybrain.app", description=__doc__)
    parser.add_argument("command", choices=["up", "down", "restart-short", "status"])
    parser.add_argument("--tickrate", type=int, default=DEFAULT_TICKRATE)
    parser.add_argument("--bot", default=DEFAULT_BOT)
    parser.add_argument(
        "--client",
        choices=[LITE, BROWSER],
        default=LITE,
        help="lite: headless, no pixels. browser: Chromium on /bot, feeding the HUD",
    )
    parser.add_argument(
        "--adopt",
        action="store_true",
        help="take ownership of services that are already running",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="start the stack and exit, leaving it running (dev.sh's behaviour)",
    )
    parser.add_argument(
        "--long-only", action="store_true", help="act on the long tier only (engine/gateway/lite)"
    )
    args = parser.parse_args(argv)

    sup = Supervisor(default_services(args.tickrate, args.bot, args.client))
    tiers = (LONG,) if args.long_only else (LONG, SHORT)

    if args.command == "status":
        print(sup.status())
        return 0
    if args.command == "down":
        sup.stop(tiers=tiers)
        return 0
    if args.command == "restart-short":
        sup.restart_short(adopt=args.adopt)
        print(sup.status())
        return 0

    try:
        sup.up(tiers=tiers, adopt=args.adopt)
    except (ServiceFailed, subprocess.CalledProcessError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        sup.stop(tiers=tiers)
        return 1
    print(sup.status())
    if args.no_wait:
        return 0
    print("\nsupervising; Ctrl-C tears the stack down", flush=True)
    return sup.supervise()


if __name__ == "__main__":
    sys.exit(main())
