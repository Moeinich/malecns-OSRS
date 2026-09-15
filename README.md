# malecns-OSRS

A real MaleCNS v1.0 *Drosophila* connectome driving an OSRS-style bot: a spiking network built
from actual fly brain connectivity data reads game state through rs-sdk and issues actions back.
See `AGENTS.md` for the repo layout and where each piece lives.

**Status: in progress, and nothing here yet claims a fly is playing the game.** The spiking
engine, the cell-type registry and the collision grid are built and tested; the sensorimotor loop
is not closed. Whether the connectome's specific wiring contributes anything is an open empirical
question, and the ablation harness — in particular a degree-preserving shuffle control — exists to
answer it honestly, including negatively.

## Quick start

```bash
git submodule update --init --depth 1
uv sync
./scripts/dev.sh
```

`dev.sh` brings up the rs-sdk stack (engine, gateway, headless lite client) in order, skipping
anything already running. `./scripts/stop.sh` tears it down. See `--tickrate <ms>` and
`--bot <name>` flags in `scripts/dev.sh`.

## Verification gate

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Run from the repo root; see `AGENTS.md` for the full policy.

## Scope and ethics

This project targets **LostCity**, an open-source 2004scape emulator, via the bundled rs-sdk. It
must never be pointed at live Old School RuneScape — automating a bot against Jagex's live game
violates their terms of service and is bannable.
