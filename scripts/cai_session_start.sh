#!/usr/bin/env bash
# Bootstrap for a Cloudera AI (CML) workbench session.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export RAY_LOCAL_MODE=false

if [[ -n "${CDSW_IP_ADDRESS:-}${CML_IP_ADDRESS:-}" ]]; then
  if [[ -z "${CDSW_APP_PORT:-}" ]]; then
    echo "CDSW_APP_PORT must be set in a Cloudera AI Application / workbench session." >&2
    exit 1
  fi
  IP="${CDSW_IP_ADDRESS:-$CML_IP_ADDRESS}"
  export RAY_HEAD_ADDRESS="${IP}:6379"
  echo "CAI session detected at ${IP}; starting Ray head..."
  ray start --head --port=6379 --dashboard-port=8265 --include-dashboard=true || true
fi

bash .cursor/install.sh
bash .cursor/start.sh

echo "API: http://127.0.0.1:8000/docs"
echo "UI:  bound to CDSW_APP_PORT=${CDSW_APP_PORT} (use your CAI Application route URL)"
echo "Run: bash scripts/cai_smoke_test.sh"
