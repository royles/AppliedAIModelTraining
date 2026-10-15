#!/usr/bin/env bash
# End-to-end smoke test for CAI or local dev (API must be running on :8000).
set -euo pipefail

API="${API_BASE:-http://127.0.0.1:8000}"

echo "== Health =="
curl -sf "${API}/api/health" | python3 -m json.tool

echo "== Create experiment =="
EXP=$(curl -sf -X POST "${API}/api/experiments" \
  -H 'Content-Type: application/json' \
  -d '{"name":"CAI smoke","description":"automated test"}')
EXP_ID=$(echo "$EXP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "experiment_id=${EXP_ID}"

echo "== Submit training job (scratch_gpt2_tiny, 15 steps) =="
JOB=$(curl -sf -X POST "${API}/api/jobs" \
  -H 'Content-Type: application/json' \
  -d "{
    \"experiment_id\": ${EXP_ID},
    \"name\": \"cai-smoke-run\",
    \"preset\": \"scratch_gpt2_tiny\",
    \"openai_model_id\": \"cai-smoke-chat\",
    \"num_workers\": 1,
    \"gpus_per_worker\": ${GPUS_PER_WORKER:-0},
    \"max_steps\": 15,
    \"batch_size\": 2
  }")
JOB_ID=$(echo "$JOB" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "job_id=${JOB_ID}"

echo "== Poll job status =="
for i in $(seq 1 120); do
  STATUS=$(curl -sf "${API}/api/jobs/${JOB_ID}" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "  attempt ${i}: ${STATUS}"
  if [[ "$STATUS" == "succeeded" ]]; then
    break
  fi
  if [[ "$STATUS" == "failed" ]]; then
    curl -sf "${API}/api/jobs/${JOB_ID}" | python3 -m json.tool
    exit 1
  fi
  sleep 5
done

if [[ "${STATUS:-}" != "succeeded" ]]; then
  echo "Timed out waiting for job ${JOB_ID}" >&2
  exit 1
fi

echo "== OpenAI models =="
curl -sf "${API}/v1/models" | python3 -m json.tool

echo "== Chat completion =="
curl -sf -X POST "${API}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "cai-smoke-chat",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 32,
    "temperature": 0.7
  }' | python3 -m json.tool

echo "CAI smoke test passed."
