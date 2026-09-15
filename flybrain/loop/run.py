"""CLI entry point: `python -m flybrain.loop.run`.

Builds the connectome, applies whatever ablation was asked for to a copy of the
weights, and drives the tick loop until the socket closes or `--ticks` is hit.
"""

from __future__ import annotations

import argparse
import sys
import time

from flybrain.connectome.loader import DEFAULT_PATH, load
from flybrain.engine.lif import LIFEngine
from flybrain.loop.agent import Ablation, Agent, AgentParams, default_encoder
from flybrain.loop.client import BridgeClient, default_socket_path
from flybrain.motor.decode import MotorIndex
from flybrain.sensory.collision import DEFAULT_COLLISION_PATH, CollisionGrid


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flybrain.loop.run", description=__doc__)
    p.add_argument("--socket", default=None, help="unix socket to the sidecar")
    p.add_argument("--connectome", default=str(DEFAULT_PATH))
    p.add_argument("--collision", default=str(DEFAULT_COLLISION_PATH))
    p.add_argument(
        "--lesion",
        action="append",
        default=[],
        metavar="POPULATION",
        help="silence a named population (repeatable), e.g. DNp01, DNa02, T4, L1",
    )
    p.add_argument("--ablate-network", action="store_true", help="zero every synapse")
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="degree- and sign-preserving random rewire; the control that matters",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ticks", type=int, default=None, help="stop after N ticks")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="compute everything, send noop — watch the brain without moving the bot",
    )
    p.add_argument("--status-every", type=int, default=25)
    p.add_argument("--substeps", type=int, default=100, help="LIF steps per sub-frame")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    connectome = load(args.connectome)
    ablation = Ablation(
        lesions=tuple(args.lesion),
        ablate_network=args.ablate_network,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    W = ablation.apply(connectome)
    print(f"condition: {ablation.label}", file=sys.stderr)
    for note in ablation.notes:
        print(f"  {note}", file=sys.stderr)

    agent = Agent(
        client=BridgeClient(args.socket or default_socket_path()),
        engine=LIFEngine(W, seed=args.seed),
        motor=MotorIndex.from_connectome(connectome),
        collision=CollisionGrid.load(args.collision),
        encoder=default_encoder(connectome),
        params=AgentParams(substeps_per_subframe=args.substeps, dry_run=args.dry_run),
    )

    started = time.monotonic()
    try:
        with agent.client:
            for report in agent.run(args.ticks):
                if args.status_every and agent.ticks % args.status_every == 0:
                    print(_status(agent, report), file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        print(_summary(agent, ablation, time.monotonic() - started), file=sys.stderr)
    return 0


def _status(agent: Agent, r) -> str:
    return (
        f"tick {r.tick} rev {r.revision} | {r.ms_total:6.1f} ms "
        f"(retina {r.ms_retina:.1f} encode {r.ms_encode:.1f} lif {r.ms_lif:.1f} "
        f"decode {r.ms_decode:.2f}) | {r.mean_rate_hz:5.2f} Hz | "
        f"{r.kind:<13} | dropped {r.dropped_game_ticks} overruns {agent.overruns}"
    )


def _summary(agent: Agent, ablation: Ablation, elapsed: float) -> str:
    counts = ", ".join(f"{k}={v}" for k, v in sorted(agent.action_counts.items())) or "none"
    rate = f"{agent.last.mean_rate_hz:.2f} Hz" if agent.last is not None else "n/a"
    return (
        f"\n--- {ablation.label} ---\n"
        f"ticks           {agent.ticks} in {elapsed:.1f} s\n"
        f"ms/tick         {agent.mean_ms_per_tick:.1f} (LIF {agent.mean_ms_lif:.1f})\n"
        f"overruns        {agent.overruns}\n"
        f"dropped ticks   {agent.client.dropped_game_ticks}\n"
        f"mean rate       {rate}\n"
        f"actions         {counts}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
