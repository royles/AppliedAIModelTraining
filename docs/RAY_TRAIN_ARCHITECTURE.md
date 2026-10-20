# Ray Train on Cloudera AI — what we are building

## Goal

Train a model with **Ray Train** (`TorchTrainer`) on a **Ray cluster**:

| Layer | Role | Where it runs |
|-------|------|----------------|
| **Ray head (GCS)** | Cluster control, scheduling | Master **session** (`ray start --head`) |
| **Ray nodes** | GPUs, object store, raylets | **CDSW worker engines** (`ray start --block --address=MASTER:6379`) |
| **Ray Train driver** | Calls `trainer.fit()`, owns the job | **Should not** be the fragile co-located path on the session when using distributed workers |
| **Ray Train workers** | Training ranks (PyTorch) | Scheduled onto Ray nodes with GPUs |

## What worked for you

**Distributed mode** (CDSW workers join the head, networking over `CDSW_MASTER_IP`) worked. Failures were **memory** on 8 GiB profiles, not worker join.

## What failed (session driver)

Running `ray.init()` + `TorchTrainer` **inside the FastAPI process on the same pod as the Ray head** is a different problem:

- Ray 2.x rewrites `127.0.0.1` to the pod IP.
- Driver must match head node metadata in GCS (`_node_ip_address`, no bogus `_temp_dir`).
- Head restarts and dashboard packages added more failure modes.

That path is only suitable for **session-only / dev** (no CDSW workers), and even then it is brittle.

## Recommended production path (implemented)

When **`num_workers > 0`** and **non-distributed low memory mode is off**:

1. Session starts / keeps **Ray head** only.
2. Platform launches **N GPU CDSW workers** (raylets) — unchanged.
3. Session waits until Ray reports **head + N worker nodes** (subprocess poll; API process never calls `ray.init`).
4. Platform launches **one extra CDSW engine** that runs **only the Ray Train driver** (no `ray start --block` on that engine).
5. That engine uses the shared **`init_ray_train_driver`** path with the GCS **auth token** from the session payload (not only `~/.ray` on the worker home).
6. Session API **polls** the driver result file and updates the job; it does **not** call `ray.init` for Train.

## Session-only / low-memory mode

- **Non-distributed low memory mode**: 1 Train rank on the session, no CDSW workers — uses session driver (tight memory; not the main target).
- For real training, use **distributed workers** + larger worker profiles (≥16 GiB GPU workers when possible).

## Operations checklist

1. Framework: **turn off** “non-distributed low memory mode” for distributed jobs.
2. `ray stop --force` only when debugging a stuck cluster; distributed jobs should not restart the head from the driver path.
3. Confirm `/api/health` shows current `ray_driver_connect_revision`.
4. Framework log should say **“Launching Ray Train driver on CDSW worker engine”** for distributed jobs.
