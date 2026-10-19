# Applied AI Model Training

A small model-training control plane with:

- **SQLite** state (`projects` / `experiments` table, `training_jobs`, `worker_groups`)
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

**Chat quality (from-scratch):** Models trained on `starter_corpus.txt` learn lines like `User: … (on a laptop)` / `Assistant: …` with parenthetical tags. `/v1/chat/completions` maps OpenAI messages to that `User:` / `Assistant:` format. Odd `((()))` or “demo project” phrases usually mean a **tiny model + high temperature** regurgitating training suffixes—use `temperature` 0.2–0.7, more steps, or a larger corpus without repetitive parentheticals.

## Architecture

```
React UI ──► FastAPI ──► SQLite
                │
                ├── Ray Train (transformer fine-tune)
                └── cdsw.launch_workers / cml.workers_v1
                      └── ray start --block --address=<head>
```

On Cloudera Machine Learning, the session pod runs the API and Ray head; additional GPU workers are started with the [workers API](https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-workers-api.html). Workers reach the master via **`CDSW_MASTER_IP`** / **`CDSW_IP_ADDRESS:6379`**; the in-session driver uses **`127.0.0.1:6379`**. See **[docs/CAI_RAY_NETWORKING.md](docs/CAI_RAY_NETWORKING.md)** and Cloudera’s [worker network communication](https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-worker-network-communication.html) guide. Set `CML_*` resource IDs as documented by Cloudera.

Locally, workers are simulated and Ray runs in-process on CPU.

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

3. Run the smoke test (creates project → trains tiny scratch GPT-2 → calls `/v1/chat/completions`):

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
- **`PlacementGroupCleaner … Failed to query Ray Train Controller`** — often a follow-on after a Ray Train worker crashed (check framework/training logs for the real error). Otherwise transient while the head or workers are still starting. Confirm workers show `ray start --block` running and the head is reachable on port `6379`.
- **`vocab_size` validation (must be >= 1024)`** — fixed: BPE may produce a smaller vocab than the target; architecture `vocab_size` is set to the trained tokenizer length (e.g. 884). Pull latest `main` if you still see the old limit.
- **`Failed to connect to the observability pubsub at address …:6379`** — the training driver could not reach Ray GCS quickly (often when using the pod IP from the same session). Restart the app after `git pull` (driver uses `127.0.0.1:6379` on CAI). If it repeats, confirm `ray start --head` is running and port `6379` is open on localhost.

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
