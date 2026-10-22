# Large GPU CDSW / CML workers

This platform launches **Cloudera AI worker engines** via `launch_workers` (see [ML workers API](https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-workers-api.html)). Each engine is a separate pod with the **CPU, memory, and `nvidia_gpu`** values you configure. Ray Train ranks run on those engines when **non-distributed low memory mode** is off.

## 1. Create a CML resource profile (admin)

In Cloudera AI, define a **resource profile** (and optional **accelerator label**) that matches the hardware you want, for example:

| Profile | Typical use |
| --- | --- |
| 16 CPU, 64 GiB, 1× GPU | Large single-GPU training |
| 32 CPU, 128 GiB, 1× GPU | From-scratch / large batch |
| 48 CPU, 256 GiB, 4× GPU | Multi-GPU on one engine |

Note the **resource profile ID**, **accelerator label ID**, and **resource group ID** from your environment.

## 2. Platform defaults (session / project)

**Framework Configuration → CDSW workers**:

1. Click a preset (**Large GPU**, **XL GPU**, etc.) or set **Worker CPU**, **Worker memory (GiB)**, and **GPUs per worker**.
2. Enter **CML resource profile ID** and **accelerator label ID** so engines schedule on GPU nodes.
3. Set **Default job worker count** to how many engines you want per new job (cluster size).
4. Ensure **Non-distributed low memory mode** is **unchecked** for distributed GPU jobs.

Optional environment overrides on the **session** (applied on first DB init):

```bash
export DEFAULT_WORKER_CPU=16
export DEFAULT_WORKER_MEMORY_GB=64
export DEFAULT_GPUS_PER_WORKER=1
export CML_RESOURCE_PROFILE_ID=<your-profile-id>
export CML_ACCELERATOR_LABEL_ID=<your-gpu-label-id>
export CML_RESOURCE_GROUP_ID=<optional-group-id>
```

Restart `python3 scripts/start.py` after changing env vars on a fresh database; existing SQLite settings are updated from the UI.

## 3. Per-job sizing (wizard)

**Model Tasks → Compute** step:

- **Worker engines (nodes)** — how many CDSW engines join the Ray cluster (e.g. `4` for four large GPU nodes).
- **CPUs / memory / GPUs per engine** — passed to `launch_workers` for **each** engine (must fit the CML profile).
- Presets mirror platform sizing; overrides are stored on the job as `worker_resources` in `config_json`.

Job logs show a line like: `4× CDSW engine(s): 32 CPU, 128 GiB RAM, 1 GPU/engine`.

## 4. Ray Train vs engine count

- **CDSW engines** = Ray worker **nodes** (processes running `ray start --block` with GPUs).
- **Ray Train workers** = training ranks (may be capped by `Ray Train worker cap` or memory hints).

For “many large GPU machines”, increase **worker engines** and keep **GPUs per engine ≥ 1**. The framework waits for cluster GPU capacity before starting Train.

## 5. Troubleshooting

| Symptom | Check |
| --- | --- |
| Workers launch but 0 GPU in Ray | Profile/accelerator IDs; worker logs for `nvidia_gpu`; `CML_ACCELERATOR_LABEL_ID` |
| `launch_workers` fails | CPU/memory/GPU request exceeds profile maximum |
| Training only on session | Low memory mode on; or `num_workers=0` / workers API unavailable |
| OOM on large model | Raise **memory per engine** and/or reduce batch size in the task |
| OOM with many workers | Corpus is loaded **once on the driver** and split across ranks; each worker tokenizes only its shard. Use **`dataset.max_samples`** and lower **max seq length** if token tensors still OOM |

See also [CAI_RAY_NETWORKING.md](CAI_RAY_NETWORKING.md) and the README architecture section.
