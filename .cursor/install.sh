#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p data backend/data

if ! python3 -m venv /tmp/venv-probe 2>/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.12-venv
fi
rm -rf /tmp/venv-probe
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt

if ! command -v npm >/dev/null; then
  echo "npm is required for the React UI." >&2
  exit 1
fi

cd frontend
npm install
cd "$ROOT"

echo "Install complete."
