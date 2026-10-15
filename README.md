# Applied AI Model Training

A small model-training control plane with:

- **SQLite** state (`experiments`, `training_jobs`, `worker_groups`)
- **FastAPI** REST API (`backend/`)
- **React** dashboard (`frontend/`)
- **Ray Train** fine-tuning for transformer models
- **Cloudera CDSW / CML workers** integration to launch GPU Ray workers elastically

## Building from scratch

Training jobs default to **`init_mode: from_scratch`**. Configure core artifacts via `architecture`:

| Field | Meaning |
| --- | --- |
| `family` | `gpt2` (decoder / chat) or `bert` (encoder classifier) |
| `vocab_size` | Embedding table size |
| `hidden_size` | Model width (d_model) |
| `num_hidden_layers` | Depth |
| `num_attention_heads` | MHA head count (must divide `hidden_size`) |
| `intermediate_size` | FFN inner dim (default `4 × hidden_size`) |
| `max_position_embeddings` | Max sequence length |
| `tie_word_embeddings` | Shared input/output embeddings (causal LMs) |

Presets: `GET /api/transformer/presets`. Parameter estimate: `POST /api/transformer/architecture/estimate`.

Use `init_mode: pretrained` and `base_model` to fine-tune an existing Hugging Face checkpoint instead.

## Architecture

```
React UI ──► FastAPI ──► SQLite
                │
                ├── Ray Train (transformer fine-tune)
                └── cdsw.launch_workers / cml.workers_v1
                      └── ray start --block --address=<head>
```

On Cloudera Machine Learning, the session pod runs the API and Ray head; additional GPU workers are started with the [workers API](https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-workers-api.html). Set `CDSW_IP_ADDRESS` / `CML_*` resource IDs as documented by Cloudera.

Locally, workers are simulated and Ray runs in-process on CPU.

## Development

```bash
python3 scripts/install.py   # once (or after dependency changes)
python3 scripts/start.py       # starts API + UI; on CAI keeps UI in foreground
```

- API (direct, session only): http://127.0.0.1:8000/docs and http://127.0.0.1:8000/api/…
- **Application URL** (CAI): same host as the dashboard — **`<application-root>/api/…`** and **`<application-root>/v1/…`** (proxied by `ui_gateway` to port 8000)
- **UX** binds **`CDSW_APP_PORT`**; **API** binds **`PLATFORM_API_PORT`** (default `8000`) on `127.0.0.1` only

Create an experiment and start a training job from the UI, or via `POST /api/jobs`.

Scale GPU workers on a running job (CML/CDSW only):

```bash
curl -X POST http://127.0.0.1:8000/api/jobs/1/scale \
  -H 'Content-Type: application/json' \
  -d '{"target_workers": 3}'
```

## Cloudera AI (CAI) test run

In a **CML workbench session** (Python 3, project cloned from this repo):

1. Optional GPU worker settings (from your CAI admin / project settings):
   - `CML_ACCELERATOR_LABEL_ID`, `CML_RESOURCE_PROFILE_ID`, `CML_RESOURCE_GROUP_ID`
2. Install deps, start Ray head, API, and UI (UI binds to `CDSW_APP_PORT`):

```bash
python3 scripts/install.py
python3 scripts/start.py
```

On CAI, `start.py` runs the **API on :8000** (background) and the **UX gateway in the foreground on `CDSW_APP_PORT`**. The browser only needs the Application URL; the UI proxies API calls to port 8000 inside the session.

3. Run the smoke test (creates experiment → trains tiny scratch GPT-2 → calls `/v1/chat/completions`):

```bash
python3 scripts/cai_smoke_test.py
```

For GPU workers, set `GPUS_PER_WORKER=1` and pass `gpus_per_worker` on the job (UI or API). The platform uses [`cml.workers_v1`](https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-workers-api.html) / legacy `cdsw` to run `ray start --block --address=<head>` on worker engines.

`start.py` sets `RAY_LOCAL_MODE=false` on CAI so training uses the Ray cluster instead of an in-process local Ray.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `RAY_HEAD_ADDRESS` | Ray head `host:6379` when not auto-detected |
| `CML_ACCELERATOR_LABEL_ID` | GPU label for `launch_workers` |
| `CML_RESOURCE_PROFILE_ID` | Engine size profile |
| `CML_RESOURCE_GROUP_ID` | Target resource group |
| `RAY_LOCAL_MODE` | When `true` (default), use local Ray even if head is set |
| `CDSW_APP_PORT` | UX (React) port — required for CAI Application routes |
| `PLATFORM_API_PORT` | FastAPI backend port (default `8000`) |
| `PLATFORM_API_UPSTREAM` | Set on the UI gateway (default `http://127.0.0.1:8000`) |
| `RAY_SKIP_HEAD` | Set to `1` to skip `ray start --head` in `start.py` |
| `RAY_INCLUDE_DASHBOARD` | Set to `1` to enable Ray dashboard (needs `ray[default]` deps; off by default on CAI) |
