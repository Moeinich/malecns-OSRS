#!/usr/bin/env bash
# Thin wrapper: flybrain/app.py is the single source of truth for how the stack
# is started (order, env, ready lines, tiers). Kept because the README and
# muscle memory point at it.
#
# `up --no-wait` starts and returns, as this script always has. The children are
# in their own process groups with their pgids recorded, so `stop.sh` can tear
# them down — that replaces the old `nohup ... & disown` (which also avoided a
# pipe hang; app.py gets the same effect from start_new_session + log files).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
exec uv run python -m flybrain.app up --no-wait "$@"
