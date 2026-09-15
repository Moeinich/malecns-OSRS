#!/usr/bin/env bash
# Create-or-fix vendor/rs-sdk/bots/<name>/bot.env for local dev.
#
# create-bot.ts's template defaults are wrong for local use (SERVER points at
# the public demo server, TELEMETRY is on, GATEWAY_URL is absent). This script
# regenerates the bot if missing and then forces the three local-dev values.
set -euo pipefail

BOT_NAME="${1:-flybot01}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RS_SDK="$REPO_ROOT/vendor/rs-sdk"
BOT_DIR="$RS_SDK/bots/$BOT_NAME"
ENV_FILE="$BOT_DIR/bot.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "bot-env: creating bot '$BOT_NAME'"
  (cd "$RS_SDK" && bun run bots/create-bot.ts "$BOT_NAME")
fi

# python3, not `sed -i ''`: sed is aliased to GNU gsed on this machine and the
# BSD-style `-i ''` form fails.
python3 - "$ENV_FILE" <<'PY'
import re
import sys

path = sys.argv[1]
with open(path) as f:
    text = f.read()

required = {
    # blank/missing SERVER resolves to port 80 -> ConnectionRefused on http://localhost/crc
    "SERVER": "localhost:8888",
    # otherwise the gateway URL is wrongly derived from SERVER and the client
    # connect/disconnect loops forever
    "GATEWAY_URL": "ws://localhost:7780",
    # default true sends bug reports to a shared third-party server
    "TELEMETRY": "false",
}

lines = text.splitlines()
seen = set()
for i, line in enumerate(lines):
    m = re.match(r"^([A-Z_]+)=", line)
    if m and m.group(1) in required:
        key = m.group(1)
        lines[i] = f"{key}={required[key]}"
        seen.add(key)

for key, value in required.items():
    if key not in seen:
        lines.append(f"{key}={value}")

with open(path, "w") as f:
    f.write("\n".join(lines) + "\n")
PY

echo "bot-env: $ENV_FILE ready"

# The password, for callers that need it (the browser client logs in over HTTP).
# `KEY=VALUE` on stdout is what flybrain/app.py folds into the child's env, so
# this file stays the only reader of bot.env. Not echoed into any log.
echo "RS_PASSWORD=$(grep -m1 '^PASSWORD=' "$ENV_FILE" | cut -d= -f2-)"
