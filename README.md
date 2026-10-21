# Applied AI Model Training

A small model-training control plane with:

- **SQLite** state (`projects` / `experiments` table, `training_jobs`, `worker_groups`)
- **FastAPI** REST API (`backend/`)
- **React** dashboard (`frontend/`)
- **Ray Train** fine-tuning for transformer models
- **Cloudera CDSW / CML workers** integration to launch GPU Ray workers elastically
- **Model registry** with lifecycle states, auto metadata, and Hugging Face–style model cards ([docs/MODEL_REGISTRY.md](docs/MODEL_REGISTRY.md))

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

**Chat quality (from-scratch):** `starter_corpus.txt` uses verbose `User:` / `Assistant:` paragraphs without bracket suffixes. Re-train after `git pull` if an older corpus taught parenthetical filler. Use `temperature` 0.2–0.7 for smoke tests; tiny models still need more steps and data for fluent answers.

## Architecture

```
React UI ──► FastAPI ──► SQLite
                │
                ├── Ray Train (transformer fine-tune)
                └── cdsw.launch_workers / cml.workers_v1
                      └── ray start --block --address=<head>
```

On Cloudera Machine Learning, the session pod runs the API and Ray head; additional GPU workers are started with the [workers API](https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-workers-api.html). Workers reach the master via **`CDSW_MASTER_IP`** (Cloudera [worker network example](https://docs.cloudera.com/cdsw/1.10.5/parallel-computing/topics/cdsw-example--worker-network-communications.html)); the Ray head listens on the master pod IP (`CDSW_IP_ADDRESS:6379`). The in-session Train driver uses the same GCS address with **`_node_ip_address`** (see **[docs/CAI_RAY_NETWORKING.md](docs/CAI_RAY_NETWORKING.md)**). HTTP/API/UX stay on **`127.0.0.1`** only. Set `CML_*` resource IDs as documented by Cloudera.

Locally, workers are simulated and Ray runs in-process on CPU.

## Platform version

Release id **`x.y.z.b`** lives in [`VERSION`](VERSION) (start `1.1.0.0`). Bump rules: major = breaking, minor = new features, patch = minor changes, bugfix = fixes only. See [docs/VERSIONING.md](docs/VERSIONING.md). Shown in the UI sidebar as **Platform x.y.z.b**.

## Development

```bash
python3 scripts/install.py        # installs into **system Python** (no .venv)
python3 scripts/start.py          # auto-installs missing deps, then starts API + UX
python3 scripts/start.py --install  # full install + start (recommended on CAI)
```

### Production UI without Node (CAI / CDSW)

The React app is **precompiled into `frontend/dist/` and committed to git**. CAI sessions usually do **not** have `npm`; you do **not** need Node to run the platform.

1. `git pull origin main` — ensures `frontend/dist/` matches the latest UX.
2. `python3 scripts/start.py` — serves that bundle from `ui_gateway`.

Only use `python3 scripts/build_ui.py` on a machine with Node when changing `frontend/src`. On CAI, `build_ui.py` exits successfully if a current prebuilt `frontend/dist` is already present.

All subprocesses (Ray, uvicorn, CDSW workers) use **`sys.executable`** / system Python (`PLATFORM_PYTHON` to override). `start.py` runs `pip install -r backend/requirements.txt` automatically when `uvicorn` / FastAPI are missing (set `PLATFORM_SKIP_DEP_CHECK=1` to skip).

- API browser docs (Swagger): **http://127.0.0.1:8000/api** (same on the Application URL)
- API (direct, session only): http://127.0.0.1:8000/api/health, …
- **Application URL** (CAI): same host as the dashboard — **`<application-root>/api/…`** and **`<application-root>/v1/…`** (served in-process by `ui_gateway`, same routes as port 8000)
- **UX** binds **`CDSW_APP_PORT`**; **API** binds **`PLATFORM_API_PORT`** (default `8000`) on `127.0.0.1` only

Create a project and start a training job from the UI, or via `POST /api/jobs`.

Ray and CDSW worker defaults are editable in the UI and persisted in SQLite (`GET`/`PUT` `/api/platform/settings`). For **large multi-GPU worker nodes** (CPU, memory, GPUs per engine, CML profile IDs), see **[docs/CDSW_LARGE_GPU_WORKERS.md](docs/CDSW_LARGE_GPU_WORKERS.md)**.

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

3. Run the smoke test (creates project → trains tiny scratch GPT-2 → calls `/v1/chat/completions`):

```bash
python3 scripts/cai_smoke_test.py
```

For GPU workers, set `GPUS_PER_WORKER=1` and pass `gpus_per_worker` on the job (UI or API). The platform uses [`cml.workers_v1`](https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-workers-api.html) / legacy `cdsw` to run `ray start --block --address=<head>` on worker engines.

`start.py` sets `RAY_LOCAL_MODE=false` on CAI so training uses the Ray cluster instead of an in-process local Ray.

## Memory and resource profiles (CAI)

Ray OOM errors (`worker(s) were killed … low on memory`) mean the **container cgroup limit** is too small for Ray head + training workers. The platform passes `memory=` to `launch_workers`, but the **CML resource profile** (via `CML_RESOURCE_PROFILE_ID`) sets the real cap. A **~3.7 GiB session** cannot host Ray head, the API, and Ray Train control actors; repeated jobs also leave **detached `PlacementGroupCleaner` actors** on the head until you run **`ray stop`** and restart.

| Role | Minimum | Recommended |
| --- | --- | --- |
| **Session** (API, Ray head, `start.py`) | **8 GiB** (platform minimum) | **16 GiB** |
| **CDSW worker** (scratch tiny / CPU) | **8 GiB** (platform minimum) | **16 GiB** |
| **CDSW worker** (GPU / larger models) | 16 GiB | **32 GiB+** |

A **~4 GiB** profile is too small for Ray training on the session pod. Use separate **≥16 GiB** worker profiles for jobs, keep the session on **≥16 GiB** when running a local Ray head.

On **8 GiB** head/worker profiles the platform **caps Ray object store (~512 MiB)**, **advertised node memory**, **batch size ≤ 2**, and (session-only jobs) **1 Train rank**. Override with care:

| Variable | Purpose |
| --- | --- |
| `DEFAULT_WORKER_MEMORY_GB` | GiB sent to `launch_workers` (default **`8`** minimum; must match profile) |
| `PLATFORM_MIN_MEMORY_GB` | Treat instances ≤ this + 1 GiB as minimum tier for Ray memory caps (default `8`) |
| `DEFAULT_WORKER_CPU` | CPUs sent to `launch_workers` (default `4`) |
| `DEFAULT_GPUS_PER_WORKER` | Default GPUs per worker (default `1`) |
| `PLATFORM_CAI_LOW_MEMORY` | Set `1` to force 8 GiB-tier caps; auto-detected when session ≤ ~9 GiB |
| `RAY_TRAIN_NUM_WORKERS` | Cap Ray Train ranks for **session-only** jobs (ignored when CDSW worker engines are launched) |
| `PLATFORM_KEEP_WORKERS_ON_FAIL` | Set `1` to skip `stop_workers` after a failed job (debug worker / Ray logs) |
| `PLATFORM_NCCL_DEFAULTS` | Default **`1`**: apply NCCL env for GPU Ray Train (set **`0`** to disable) |
| `RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S` | Seconds to wait for Train workers (default **900** on CAI) |

Check `GET /api/health` → `details.memory` for detected limits and recommendations.

### Ray worker / Train log messages

- **`ModuleNotFoundError: No module named 'ray'`** on a worker engine — worker bootstrap runs `ray` in the project `.venv` when present; it pip-installs `backend/requirements.txt` into **that** interpreter (not the session-only check). First join can take several minutes. If install fails, run `python3 scripts/install.py` on the session, ensure workers can reach PyPI, or remove a broken `.venv` on the project so workers fall back to engine Python.
- **`PlacementGroupCleaner … State API`** — Benign while the Ray API server starts; enable dashboard on the head (`ray stop` + restart `start.py`). These lines are filtered from the task log UI when harmless.
- **`SynchronizationActor … ray.train.report`** — Usually meant rank 0 was reporting without other ranks (fixed in current `transformer_job.py`). If it persists past the first minute with stalled steps, check worker GPUs/logs.
- **`vocab_size` validation (must be >= 1024)`** — fixed: BPE may produce a smaller vocab than the target; architecture `vocab_size` is set to the trained tokenizer length (e.g. 884). Pull latest `main` if you still see the old limit.
- **`Failed to connect to the observability pubsub at address …:6379`** — the training driver could not reach Ray GCS quickly (often when using the pod IP from the same session). Restart the app after `git pull` and confirm `ray start --head` is running with GCS reachable on **`CDSW_IP_ADDRESS:6379`** (workers use **`CDSW_MASTER_IP`**).

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
| `RAY_INCLUDE_DASHBOARD` | Default **on** when `ray[default]` is installed; set `0` to disable dashboard/State API |
| `DEFAULT_WORKER_MEMORY_GB` | CDSW worker RAM (GiB); default **`8`** (minimum) |
| `DEFAULT_WORKER_CPU` | CDSW worker CPUs; default `4` |
| `DEFAULT_GPUS_PER_WORKER` | Default GPU count for workers; default `1` |
| `PLATFORM_CAI_LOW_MEMORY` | `1` = cap Ray Train workers and batch size on tiny sessions |
| `RAY_TRAIN_NUM_WORKERS` | Max Ray Train workers per job (overrides auto cap when &gt; 0) |
| `PLATFORM_NCCL_DEFAULTS` | Default **`1`**: NCCL / PyTorch distributed env for multi-GPU Train (`0` = off) |
| `PLATFORM_PYTHON` | Override interpreter for start/install (default: current `python3`) |
| `PLATFORM_SKIP_DEP_CHECK` | Set `1` to skip automatic `pip install` in `start.py` |
| `PLATFORM_UVICORN_LOG_LEVEL` | With `PLATFORM_ACCESS_LOG=1`, uvicorn level (default `info`) |
| `PLATFORM_ACCESS_LOG` | Set `1` for every HTTP line (default: **once per route**, then suppress 3s poll spam) |
