#!/usr/bin/env bash
# Thin wrapper over the supervisor; see scripts/dev.sh. Only process groups the
# supervisor started (or adopted) are signalled — a stack someone else brought
# up is left alone.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
exec uv run python -m flybrain.app down "$@"
