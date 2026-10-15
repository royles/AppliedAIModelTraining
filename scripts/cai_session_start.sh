#!/usr/bin/env bash
# Bootstrap for a Cloudera AI (CML) workbench session.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export RAY_LOCAL_MODE=false

if [[ -n "${CDSW_IP_ADDRESS:-}${CML_IP_ADDRESS:-}" ]]; then
  IP="${CDSW_IP_ADDRESS:-$CML_IP_ADDRESS}"
  export RAY_HEAD_ADDRESS="${IP}:6379"
  echo "CAI session detected at ${IP}; starting Ray head..."
  ray start --head --port=6379 --dashboard-port=8265 --include-dashboard=true || true
fi

bash .cursor/install.sh
bash .cursor/start.sh

echo "API: http://127.0.0.1:8000/docs"
echo "Run: bash scripts/cai_smoke_test.sh"
