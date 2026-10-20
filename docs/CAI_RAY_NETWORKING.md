# Cloudera AI (CAI) Ray networking

This platform follows Cloudera’s distributed-computing and engine networking model. Official references:

- [Worker network communication](https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-worker-network-communication.html) — workers receive **`CML_MASTER_IP`** / **`CDSW_MASTER_IP`** (master engine IP).
- [Engine environment variables](https://docs.cloudera.com/machine-learning/cloud/engines/topics/ml-engine-environment-variables.html) — **`CDSW_IP_ADDRESS`** is how *other engines* reach *this* engine; HTTP on **`CDSW_APP_PORT`** / **`CDSW_READONLY_PORT`** should bind **`127.0.0.1`** inside the pod.
- [Workers API](https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-workers-api.html) — `launch_workers(..., code=...)` runs `ray start --block` on worker engines.

## Roles in this repo

| Role | Where it runs | Ray GCS / `ray.init` address | Why |
|------|----------------|------------------------------|-----|
| **Ray head** | Session (master) pod | `ray start --head` on port **6379** | Started by `scripts/start.py` (`start_ray_head`). |
| **Training driver** | Same session pod as head | **`127.0.0.1:6379`** | Same-pod traffic must use loopback (mirrors `CDSW_APP_PORT` guidance). Avoids GCS / observability pubsub timeouts to the pod IP. |
| **Ray workers** | Worker engines from `launch_workers` | **`CDSW_MASTER_IP:6379`** (or master’s `CDSW_IP_ADDRESS:6379` passed at launch) | Workers are separate engines; they must use the master’s cluster-routable IP. |

## Environment variables

| Variable | Set on | Used for |
|----------|--------|----------|
| `CDSW_IP_ADDRESS` / `CML_IP_ADDRESS` | Every engine | Master’s IP as seen by other engines; worker Ray head address `host:6379`. |
| `CDSW_MASTER_IP` / `CML_MASTER_IP` | Worker engines only | Master IP for outbound connections (preferred in worker bootstrap code). |
| `RAY_HEAD_ADDRESS` | Session after `start_ray_head` | Cached `CDSW_IP_ADDRESS:6379` for workers and UI. |
| `RAY_ADDRESS` | Session driver | Set to **`127.0.0.1:6379`** after head is up. |
| `PLATFORM_NCCL_DEFAULTS` | Session + workers | Default **`1`**: apply NCCL / `TORCH_NCCL` env (`NCCL_ASYNC_ERROR_HANDLING`, socket interface `^docker0,lo`, etc.). Set **`0`** to skip. |

## NCCL (multi-GPU Ray Train)

PyTorch distributed on CUDA uses the **NCCL** backend. On CAI, worker engines often need explicit socket interface selection and async error handling so collectives fail fast instead of hanging.

When `PLATFORM_NCCL_DEFAULTS` is enabled (default), the platform merges defaults in:

- `apply_ray_train_runtime_env()` (session driver before Train)
- CDSW worker bootstrap env (`cdsw_workers.worker_launch_env`)
- Existing env vars are **never** overwritten

Set `NCCL_DEBUG=INFO` in the project/session if you need NCCL logs (not set by default).

## Code map

- `app/services/cai_networking.py` — resolve master IP, worker head, driver loopback.
- `app/services/nccl_env.py` — default NCCL env for GPU Train (`PLATFORM_NCCL_DEFAULTS`).
- `app/services/cdsw_workers.py` — `build_worker_start_code`, `launch_workers`.
- `scripts/common.py` — `start_ray_head()`, `cai_session_ip()`.

## Why workers join but Ray Train fails

These are **different connection roles** on CAI:

| Step | Process | Mechanism | Address | Typical result |
|------|---------|-----------|---------|----------------|
| 1 | Session | `ray start --head` | GCS on `:6379` | Head registers with **pod IP** (`CDSW_IP_ADDRESS`) |
| 2 | CDSW worker engine | `ray start --block --address=…` | **`CDSW_MASTER_IP:6379`** (master pod IP) | **Succeeds** — remote engine → routable master IP |
| 3 | Training driver | `ray.init(address=…)` in **same pod as head** | Must be **`127.0.0.1:6379`** | **Fails** if set to pod IP (`No node info found` / `Failed to connect … at <pod-ip>:6379`) |

The cluster map can look healthy while Train still fails:

- **TCP checks** to `100.100.x.x:6379` succeed (GCS listens on all interfaces).
- **Worker nodes** appear in `ray.nodes()` because they used the correct **worker** address.
- **Ray Train** alone runs `ray.init` on the master pod; that code path must use **loopback**, not the address workers use.

Framework log should show **`Driver connecting via 127.0.0.1:6379`** and **`workers join 100.100.x.x:6379`**. If the traceback still cites the pod IP, the session is not running the loopback coercion (pull latest `main`, restart the API).

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Workers OK, Train fails with `No node info found` / connect to `<pod-ip>:6379` | Driver used worker head address; confirm framework log shows `127.0.0.1:6379` for the driver. |
| `Failed to connect to the observability pubsub at address <pod-ip>:6379` | Driver is using pod IP; upgrade and confirm logs show `driver=127.0.0.1:6379`. |
| `Ray GCS not reachable at 127.0.0.1:6379` | Head not running — `ss -ltn \| grep 6379`, rerun `python3 scripts/start.py`, inspect `start-user.log`. |
| Workers `scheduling` forever | Resource profile / GPU labels; worker code never reaches `ray start --block`. |
| Head in Framework UI ≠ session IP | Stale SQLite `ray_head_address`; live env wins after sync on API start. |
| `PlacementGroupCleaner … State API` | Head missing the Ray API server (dashboard). Run `ray stop` and restart the app; `start_ray_head` enables dashboard by default. Harmless if training still progresses. |
| Ray worker engines exit / `ray start --block exited` | Worker log: pip OOM during `requirements.txt`, wrong head IP, or GPU not advertised. Use **≥16 GiB** profile, project `.venv` (run `install.py` on session), and confirm `ray start` prints `gpus=1`. Set `PLATFORM_KEEP_WORKERS_ON_FAIL=1` to leave workers up after a failed job. Inspect `/tmp/ray/session_*/logs/ray_process_exit.log` on the worker. |
| `TrainController` requests `GPU: 1` but job fails | Session head is often CPU-only; GPU must be on a **joined worker node**. Framework log shows `Cluster resources before Train` — need `GPU ≥ 1`. |
