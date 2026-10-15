#!/usr/bin/env python3
"""End-to-end smoke test (API on :8000, optional CAI workers)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import time
import urllib.error
import urllib.request

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
GPUS_PER_WORKER = int(os.environ.get("GPUS_PER_WORKER", "0"))


def request_json(
    method: str,
    path: str,
    body: dict | None = None,
    timeout: float = 120.0,
) -> dict:
    url = f"{API_BASE}{path}"
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pretty(obj: dict) -> None:
    print(json.dumps(obj, indent=2))


def main() -> int:
    print("== Health ==")
    pretty(request_json("GET", "/api/health"))

    print("== Create experiment ==")
    exp = request_json(
        "POST",
        "/api/experiments",
        {"name": "CAI smoke", "description": "automated test"},
    )
    exp_id = exp["id"]
    print(f"experiment_id={exp_id}")

    print("== Submit training job (scratch_gpt2_tiny, 15 steps) ==")
    job = request_json(
        "POST",
        "/api/jobs",
        {
            "experiment_id": exp_id,
            "name": "cai-smoke-run",
            "preset": "scratch_gpt2_tiny",
            "openai_model_id": "cai-smoke-chat",
            "num_workers": 1,
            "gpus_per_worker": GPUS_PER_WORKER,
            "max_steps": 15,
            "batch_size": 2,
        },
        timeout=300.0,
    )
    job_id = job["id"]
    print(f"job_id={job_id}")

    print("== Poll job status ==")
    status = ""
    for attempt in range(1, 121):
        row = request_json("GET", f"/api/jobs/{job_id}")
        status = row["status"]
        print(f"  attempt {attempt}: {status}")
        if status == "succeeded":
            break
        if status == "failed":
            pretty(row)
            return 1
        time.sleep(5)

    if status != "succeeded":
        print(f"Timed out waiting for job {job_id}", file=sys.stderr)
        return 1

    print("== OpenAI models ==")
    pretty(request_json("GET", "/v1/models"))

    print("== Chat completion ==")
    pretty(
        request_json(
            "POST",
            "/v1/chat/completions",
            {
                "model": "cai-smoke-chat",
                "messages": [
                    {"role": "user", "content": "Say hello in one short sentence."}
                ],
                "max_tokens": 32,
                "temperature": 0.7,
            },
            timeout=300.0,
        )
    )

    print("CAI smoke test passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as exc:
        print(exc.read().decode(), file=sys.stderr)
        sys.exit(1)
