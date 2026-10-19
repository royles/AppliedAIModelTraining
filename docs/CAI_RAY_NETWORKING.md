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

## Code map

- `app/services/cai_networking.py` — resolve master IP, worker head, driver loopback.
- `app/services/cdsw_workers.py` — `build_worker_start_code`, `launch_workers`.
- `scripts/common.py` — `start_ray_head()`, `cai_session_ip()`.

## Troubleshooting

| Symptom | Check |
|---------|--------|
| `Failed to connect to the observability pubsub at address <pod-ip>:6379` | Driver is using pod IP; upgrade and confirm logs show `driver=127.0.0.1:6379`. |
| `Ray GCS not reachable at 127.0.0.1:6379` | Head not running — `ss -ltn \| grep 6379`, rerun `python3 scripts/start.py`, inspect `start-user.log`. |
| Workers `scheduling` forever | Resource profile / GPU labels; worker code never reaches `ray start --block`. |
| Head in Framework UI ≠ session IP | Stale SQLite `ray_head_address`; live env wins after sync on API start. |
