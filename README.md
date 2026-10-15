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
python3 scripts/install.py   # installs into **system Python** (no .venv)
python3 scripts/start.py     # auto-installs missing deps, then starts API + UX
```

All subprocesses (Ray, uvicorn, CDSW workers) use **`sys.executable`** / system Python (`PLATFORM_PYTHON` to override). `start.py` runs `pip install -r backend/requirements.txt` automatically when `uvicorn` / FastAPI are missing (set `PLATFORM_SKIP_DEP_CHECK=1` to skip).

- API (direct, session only): http://127.0.0.1:8000/docs and http://127.0.0.1:8000/api/…
- **Application URL** (CAI): same host as the dashboard — **`<application-root>/api/…`** and **`<application-root>/v1/…`** (served in-process by `ui_gateway`, same routes as port 8000)
- **UX** binds **`CDSW_APP_PORT`**; **API** binds **`PLATFORM_API_PORT`** (default `8000`) on `127.0.0.1` only

Create an experiment and start a training job from the UI, or via `POST /api/jobs`.

Ray and CDSW worker defaults are editable in the UI and persisted in SQLite (`GET`/`PUT` `/api/platform/settings`).

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

## Memory and resource profiles (CAI)

Ray OOM errors (`worker(s) were killed … low on memory`) mean the **container cgroup limit** is too small for Ray head + training workers. The platform passes `memory=` to `launch_workers`, but the **CML resource profile** (via `CML_RESOURCE_PROFILE_ID`) sets the real cap.

| Role | Minimum | Recommended |
| --- | --- | --- |
| **Session** (API, Ray head, `start.py`) | 8 GiB | **16 GiB** |
| **CDSW worker** (scratch tiny / CPU) | 8 GiB | **16 GiB** |
| **CDSW worker** (GPU / larger models) | 16 GiB | **32 GiB+** |

A **~4 GiB** profile is too small for Ray training on the session pod. Use separate **≥16 GiB** worker profiles for jobs, keep the session on **≥16 GiB** when running a local Ray head.

On small containers the platform automatically **caps Ray Train to 1 worker** and **batch size ≤ 2**. Override with care:

| Variable | Purpose |
| --- | --- |
| `DEFAULT_WORKER_MEMORY_GB` | GiB sent to `launch_workers` (default `16`; must match profile) |
| `DEFAULT_WORKER_CPU` | CPUs sent to `launch_workers` (default `4`) |
| `DEFAULT_GPUS_PER_WORKER` | Default GPUs per worker (default `1`) |
| `PLATFORM_CAI_LOW_MEMORY` | Set `1` to force low-memory caps; auto-detected when cgroup &lt; 6 GiB |
| `RAY_TRAIN_NUM_WORKERS` | Hard cap on Ray Train worker processes (e.g. `1`) |

Check `GET /api/health` → `details.memory` for detected limits and recommendations.

### Ray worker / Train log messages

- **`ModuleNotFoundError: No module named 'ray'`** on a worker engine — run `python3 scripts/install.py` on the **session** (system Python), then launch workers again. New worker bootstrap code installs `backend/requirements.txt` on first start if Ray is missing.
- **`PlacementGroupCleaner … Failed to query Ray Train Controller`** — usually transient while the head or workers are still starting, or when workers failed to join. Confirm workers show `ray start --block` running and the head is reachable on port `6379`.

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
| `DEFAULT_WORKER_MEMORY_GB` | CDSW worker RAM (GiB); default `16` |
| `DEFAULT_WORKER_CPU` | CDSW worker CPUs; default `4` |
| `DEFAULT_GPUS_PER_WORKER` | Default GPU count for workers; default `1` |
| `PLATFORM_CAI_LOW_MEMORY` | `1` = cap Ray Train workers and batch size on tiny sessions |
| `RAY_TRAIN_NUM_WORKERS` | Max Ray Train workers per job (overrides auto cap when &gt; 0) |
| `PLATFORM_PYTHON` | Override interpreter for start/install (default: current `python3`) |
| `PLATFORM_SKIP_DEP_CHECK` | Set `1` to skip automatic `pip install` in `start.py` |
