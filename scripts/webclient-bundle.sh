#!/usr/bin/env bash
# Build the webclient's `/bot` bundle (browser client) if stale, then delegate
# to bot-env.sh so its `KEY=VALUE` stdout still reaches flybrain/app.py's
# `_prepare_env`. Idempotent: skips the ~40s build when `out/` is already
# newer than `src/`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEBCLIENT="$REPO_ROOT/vendor/rs-sdk/server/webclient"
OUT_DIR="$WEBCLIENT/out"
SRC_DIR="$WEBCLIENT/src"

# `_prepare_env` captures stdout and drops non-KEY=VALUE lines, so bun's build
# output must go to stderr or it silently vanishes and a slow first build
# looks like a hang.
if [[ -d "$OUT_DIR" ]] && [[ -z "$(find "$SRC_DIR" -newer "$OUT_DIR" -print -quit 2>/dev/null)" ]]; then
  echo "webclient-bundle: out/ is up to date, skipping build" >&2
else
  echo "webclient-bundle: building bot bundle (BUILD_MODE=bot bun run bundle.ts)..." >&2
  (cd "$WEBCLIENT" && BUILD_MODE=bot bun run bundle.ts) >&2
  echo "webclient-bundle: build complete" >&2
fi

exec "$REPO_ROOT/scripts/bot-env.sh" "$@"
