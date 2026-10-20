# Cloudera AI (CAI) Ray networking

This platform follows Cloudera’s distributed-computing and engine networking model. Official references:

- [CDSW 1.10.5 — Worker network communications (example)](https://docs.cloudera.com/cdsw/1.10.5/parallel-computing/topics/cdsw-example--worker-network-communications.html) — each worker gets **`CDSW_MASTER_IP`** (fully routable master engine IP); workers open TCP **to the master** (doc uses port 6000; Ray GCS uses **6379** the same way).
- [Worker network communication (CML cloud)](https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-worker-network-communication.html) — same idea on CAI: **`CML_MASTER_IP`** / **`CDSW_MASTER_IP`**.
- [Engine environment variables](https://docs.cloudera.com/machine-learning/cloud/engines/topics/ml-engine-environment-variables.html) — **`CDSW_IP_ADDRESS`** is how *other engines* reach *this* engine; HTTP on **`CDSW_APP_PORT`** / **`CDSW_READONLY_PORT`** should bind **`127.0.0.1`** inside the pod.
- [Workers API](https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-workers-api.html) — `launch_workers(..., code=...)` runs `ray start --block` on worker engines.

## Roles in this repo

| Role | Where it runs | Ray GCS / `ray.init` address | Why |
|------|----------------|------------------------------|-----|
| **Ray head** | Session (master) pod | `ray start --head` on port **6379** | Started by `scripts/start.py` (`start_ray_head`). |
| **Training driver** | Same session pod as head | **`CDSW_IP_ADDRESS:6379`** + **`_node_ip_address`** | Ray 2.x on Linux rewrites `127.0.0.1` to the pod IP in `ray.init` anyway (`resolve_ip_for_localhost`). The driver must match the head node IP in GCS (see `ray_driver_connect.py`). |
| **Ray workers** | Worker engines from `launch_workers` | **`CDSW_MASTER_IP:6379`** (CAI-set on workers; session also pins **`PLATFORM_RAY_HEAD_ADDRESS`**) | Same pattern as Cloudera’s socket example: connect to **master IP**, not the worker’s **`CDSW_IP_ADDRESS`**. |

### Cloudera worker → master pattern (Ray)

Cloudera’s [worker network example](https://docs.cloudera.com/cdsw/1.10.5/parallel-computing/topics/cdsw-example--worker-network-communications.html):

| Side | Doc pattern | This platform (Ray GCS) |
|------|-------------|-------------------------|
| **Master** | `socket.bind(("0.0.0.0", port))` — listen on all interfaces | `ray start --head --port=<GCS> --node-ip-address=<CDSW_IP_ADDRESS>` (GCS accepts worker TCP to pod IP) |
| **Worker** | `connect((os.environ["CDSW_MASTER_IP"], port))` | `ray start --block --address=<resolved head>` where `resolve_head()` prefers **`CDSW_MASTER_IP`** when it matches the session launch pin |
| **HTTP / UX** | N/A in worker doc | FastAPI + CDSW app port stay on **`127.0.0.1`** only (`scripts/common.BIND_HOST`) |

GCS port defaults to **6379**; override with **`PLATFORM_RAY_GCS_PORT`** (1024–65535) on session and workers (passed in `worker_launch_env`).

Canonical Python: `cai_networking.resolve_worker_ray_head_address()` and the same logic embedded in CDSW worker code via `cdsw_worker_codegen.embedded_worker_resolve_head_source()`.

After head start, `assert_master_gcs_worker_routable()` verifies TCP to **`CDSW_IP_ADDRESS:<GCS>`**, not only `127.0.0.1`.

If a connectivity probe shows **`tcp: true`** to `CDSW_MASTER_IP:<GCS>`, CAI **worker→master routing matches the doc**; remaining failures are usually Ray auth/version (not engine isolation).

## Environment variables

| Variable | Set on | Used for |
|----------|--------|----------|
| `CDSW_IP_ADDRESS` / `CML_IP_ADDRESS` | Every engine | Master’s IP as seen by other engines; worker Ray head address `host:6379`. |
| `CDSW_MASTER_IP` / `CML_MASTER_IP` | Worker engines (CAI-set) | Can be **stale** after session reschedule; Ray join uses **`PLATFORM_RAY_HEAD_ADDRESS`** / launch fallback instead. |
| `PLATFORM_RAY_HEAD_ADDRESS` | Worker engines at launch | Master `host:6379` pinned by the session when `launch_workers` runs. |
| `RAY_HEAD_ADDRESS` | Session after `start_ray_head` | Cached `CDSW_IP_ADDRESS:6379` for workers and UI. |
| `RAY_ADDRESS` | Session driver | Set to **`CDSW_IP_ADDRESS:6379`** after head is up (same GCS as workers). |
| `RAY_AUTH_TOKEN` | Session + workers | Ray 2.59+ **local** GCS token (no external IdP). Synced via `ray get-auth-token` after head start; copied to `data/.ray_gcs_auth_token` (project mount) and passed to CDSW workers. |
| `PLATFORM_RAY_GCS_AUTH_FILE` | Worker engines | Path to `data/.ray_gcs_auth_token` on the project. |
| `PLATFORM_NCCL_DEFAULTS` | Session + workers | Default **`1`**: apply NCCL / `TORCH_NCCL` env (`NCCL_ASYNC_ERROR_HANDLING`, socket interface `^docker0,lo`, etc.). Set **`0`** to skip. |

## NCCL (multi-GPU Ray Train)

PyTorch distributed on CUDA uses the **NCCL** backend. On CAI, worker engines often need explicit socket interface selection and async error handling so collectives fail fast instead of hanging.

When `PLATFORM_NCCL_DEFAULTS` is enabled (default), the platform merges defaults in:

- `apply_ray_train_runtime_env()` (session driver before Train)
- CDSW worker bootstrap env (`cdsw_workers.worker_launch_env`)
- Existing env vars are **never** overwritten

Set `NCCL_DEBUG=INFO` in the project/session if you need NCCL logs (not set by default).

## Code map

- `app/services/cai_networking.py` — resolve master IP, worker head.
- `app/services/ray_driver_connect.py` — `ray.init` with `_node_ip_address` + auth.
- `app/services/nccl_env.py` — default NCCL env for GPU Train (`PLATFORM_NCCL_DEFAULTS`).
- `app/services/cdsw_workers.py` — `build_worker_start_code`, `launch_workers`.
- `scripts/common.py` — `start_ray_head()`, `cai_session_ip()`.

## Why workers join but Ray Train fails

These are **different connection roles** on CAI:

| Step | Process | Mechanism | Address | Typical result |
|------|---------|-----------|---------|----------------|
| 1 | Session | `ray start --head` | GCS on `:6379` | Head registers with **pod IP** (`CDSW_IP_ADDRESS`) |
| 2 | CDSW worker engine | `ray start --block --address=…` | **`CDSW_MASTER_IP:6379`** (master pod IP) | **Succeeds** — remote engine → routable master IP |
| 3 | Training driver | `ray.init(address=…)` in **same pod as head** | **`CDSW_IP:6379`** and **`_node_ip_address=CDSW_IP`** | **Fails** if only logs say `127.0.0.1` but Ray canonicalizes to pod IP without matching the head node (60s timeout → `No node info found` / `Failed to connect … at <pod-ip>:6379`) |

The cluster map can look healthy while Train still fails:

- **TCP checks** to `100.100.x.x:6379` succeed (GCS listens on all interfaces).
- **Worker nodes** appear in `ray.nodes()` because they used the correct **worker** address.
- **Ray Train** runs a separate `ray.init` in the API process; it must register against the **same head node IP** Ray stored at `ray start --head`.

Framework log should show **`Ray driver GCS connect 100.100.x.x:6379; node_ip=100.100.x.x`**. If you only see `127.0.0.1` in UX logs but the error cites the pod IP, that is Ray rewriting localhost — upgrade to a build with `ray_driver_connect.py`.

## Troubleshooting

| Symptom | Check |
|---------|--------|
| `No node info found matching attributes: '<ip>, /tmp/ray'` | Do not pass `_temp_dir` to `ray.init` (fixed in `ray_driver_connect`). Session-only jobs restart the head via `ray stop` + `ray start --head`. |
| Workers OK, Train fails with `No node info found` / connect to `<pod-ip>:6379` after ~60s | Stale raylet vs GCS; `ray stop --force` and restart head; confirm `ray.init OK` in framework log. |
| `InvalidAuthToken` on `ray.init` | Backend missing `RAY_AUTH_TOKEN`; rerun `scripts/start.py` or copy `~/.ray/auth_token` into the API env. |
| `Ray GCS not reachable at 127.0.0.1:6379` | Head not running — `ss -ltn \| grep 6379`, rerun `python3 scripts/start.py`, inspect `start-user.log`. |
| Workers `scheduling` forever | Resource profile / GPU labels; worker code never reaches `ray start --block`. |
| Head in Framework UI ≠ session IP | Stale SQLite `ray_head_address`; live env wins after sync on API start. |
| `PlacementGroupCleaner … State API` | Head missing the Ray API server (dashboard). Run `ray stop` and restart the app; `start_ray_head` enables dashboard by default. Harmless if training still progresses. |
| Ray worker engines exit / `ray start --block exited` | Worker log: pip OOM during `requirements.txt`, wrong head IP, or GPU not advertised. Use **≥16 GiB** profile, project `.venv` (run `install.py` on session), and confirm `ray start` prints `gpus=1`. Set `PLATFORM_KEEP_WORKERS_ON_FAIL=1` to leave workers up after a failed job. Inspect `/tmp/ray/session_*/logs/ray_process_exit.log` on the worker. |
| `TrainController` requests `GPU: 1` but job fails | Session head is often CPU-only; GPU must be on a **joined worker node**. Framework log shows `Cluster resources before Train` — need `GPU ≥ 1`. |
