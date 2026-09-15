#!/usr/bin/env bash
# Stop the local rs-sdk stack started by dev.sh, by process pattern.
set -euo pipefail

PATTERNS=(
  "engine:bun run src/app.ts"
  "gateway:run gateway"
  "lite client:src/lite/runner.ts"
)

for entry in "${PATTERNS[@]}"; do
  name="${entry%%:*}"
  pattern="${entry#*:}"
  pids=$(pgrep -f "$pattern" || true)
  if [[ -z "$pids" ]]; then
    echo "$name: not running"
    continue
  fi
  echo "$name: stopping pid(s) $pids"
  # shellcheck disable=SC2086
  kill $pids
done
