#!/usr/bin/env bash
set -euo pipefail

# Idempotent bootstrap for AppliedAIModelTraining.
# The repository currently has no package lockfiles or service dependencies.

if [[ ! -f README.md ]]; then
  echo "Expected README.md at repository root." >&2
  exit 1
fi

command -v python3 >/dev/null || { echo "python3 is required." >&2; exit 1; }
command -v node >/dev/null || { echo "node is required." >&2; exit 1; }

echo "Install complete (no project dependencies yet)."
