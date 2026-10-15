#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# CAI Application proxy expects the UI on CDSW_APP_PORT (see Cloudera docs).
export CDSW_APP_PORT="${CDSW_APP_PORT:-5173}"
UI_PORT="${CDSW_APP_PORT}"

mkdir -p data backend/data

if ! tmux -f /exec-daemon/tmux.portal.conf has-session -t api 2>/dev/null; then
  tmux -f /exec-daemon/tmux.portal.conf new-session -d -s api -c "$ROOT/backend" \
    "source \"$ROOT/.venv/bin/activate\" && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 2>&1 | tee /tmp/api.log"
fi

if ! tmux -f /exec-daemon/tmux.portal.conf has-session -t web 2>/dev/null; then
  tmux -f /exec-daemon/tmux.portal.conf new-session -d -s web -c "$ROOT/frontend" \
    "export CDSW_APP_PORT=\"${CDSW_APP_PORT}\" && npm run dev 2>&1 | tee /tmp/web.log"
fi

for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/api/health >/dev/null \
    && curl -sf "http://127.0.0.1:${UI_PORT}/" >/dev/null; then
    echo "API and UI are up (UI port CDSW_APP_PORT=${UI_PORT})."
    exit 0
  fi
  sleep 1
done

echo "Timed out waiting for services; check /tmp/api.log and /tmp/web.log" >&2
exit 1
