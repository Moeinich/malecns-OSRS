#!/usr/bin/env bash
# Bring up the local rs-sdk stack: engine, gateway, headless lite client.
# Idempotent: only starts what isn't already listening/running.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RS_SDK="$REPO_ROOT/vendor/rs-sdk"

TICKRATE=""
BOT_NAME="flybot01"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tickrate) TICKRATE="$2"; shift 2 ;;
    --bot) BOT_NAME="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

ENGINE_LOG=/tmp/dev-engine.log
GATEWAY_LOG=/tmp/dev-gateway.log
LITE_LOG=/tmp/dev-lite.log

port_listening() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

wait_for_log() {
  local log="$1" pattern="$2" timeout="$3" waited=0
  while (( waited < timeout )); do
    if grep -qF "$pattern" "$log" 2>/dev/null; then
      return 0
    fi
    sleep 1
    ((waited++))
  done
  return 1
}

ENGINE_STATUS="already running"
GATEWAY_STATUS="already running"
LITE_STATUS="already running"

# --- Engine (8888 HTTP, 43594 game, 8898) ---
if port_listening 8888; then
  echo "engine: already listening on 8888, skipping"
else
  echo "engine: starting (first run packs the cache, can take ~40s)"
  # env, not a bare assignment prefix: ${VAR:+K=V} expands after bash has parsed
  # the prefix, so it would be treated as a command name.
  ( cd "$RS_SDK/server/engine" && \
    nohup env BUILD_VERIFY=false ${TICKRATE:+NODE_TICKRATE="$TICKRATE"} \
    bun run src/app.ts > "$ENGINE_LOG" 2>&1 & disown ) >/dev/null 2>&1 </dev/null
  if wait_for_log "$ENGINE_LOG" "World ready" 90; then
    ENGINE_STATUS="started"
  else
    ENGINE_STATUS="FAILED (see $ENGINE_LOG)"
  fi
fi

# --- Gateway (7780) ---
if port_listening 7780; then
  echo "gateway: already listening on 7780, skipping"
else
  echo "gateway: starting"
  ( cd "$RS_SDK/server/gateway" && nohup bun run gateway > "$GATEWAY_LOG" 2>&1 & disown ) >/dev/null 2>&1 </dev/null
  if wait_for_log "$GATEWAY_LOG" "Gateway running" 30; then
    GATEWAY_STATUS="started"
  else
    GATEWAY_STATUS="FAILED (see $GATEWAY_LOG)"
  fi
fi

# --- Headless lite client (no listening port; check by process pattern) ---
LITE_PATTERN="src/lite/runner.ts $BOT_NAME"
if pgrep -f "$LITE_PATTERN" >/dev/null 2>&1; then
  echo "lite client ($BOT_NAME): already running, skipping"
else
  echo "lite client ($BOT_NAME): starting"
  "$REPO_ROOT/scripts/bot-env.sh" "$BOT_NAME"
  ( cd "$RS_SDK/server/webclient" && \
    nohup bun src/lite/runner.ts "$BOT_NAME" > "$LITE_LOG" 2>&1 & disown ) >/dev/null 2>&1 </dev/null
  if wait_for_log "$LITE_LOG" "Gateway connected, registering as" 30; then
    LITE_STATUS="started"
  else
    LITE_STATUS="FAILED (see $LITE_LOG)"
  fi
fi

echo
printf "%-14s %-18s %-9s %s\n" "SERVICE" "STATUS" "PORT" "LOG"
printf "%-14s %-18s %-9s %s\n" "engine" "$ENGINE_STATUS" "8888" "$ENGINE_LOG"
printf "%-14s %-18s %-9s %s\n" "gateway" "$GATEWAY_STATUS" "7780" "$GATEWAY_LOG"
printf "%-14s %-18s %-9s %s\n" "lite ($BOT_NAME)" "$LITE_STATUS" "-" "$LITE_LOG"

if [[ "$ENGINE_STATUS" == FAILED* || "$GATEWAY_STATUS" == FAILED* || "$LITE_STATUS" == FAILED* ]]; then
  exit 1
fi
