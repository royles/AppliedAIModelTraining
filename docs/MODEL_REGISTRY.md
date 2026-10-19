# Model registry

The platform maintains a **model registry** (SQLite `trained_models`) separate from raw training jobs. Each successful causal-LM job is registered automatically with metadata and a generated **model card**.

## Lifecycle states

| State | When to use |
| --- | --- |
| `created` | Reserved for future “register before train” flows |
| `training` | Job running (optional future hook) |
| `trained` | **Default** after a successful job—artifact on disk |
| `testing` | Smoke / qualitative eval via `/v1` |
| `benchmarked` | Quantitative benchmark JSON recorded |
| `production` | Approved for production serving |
| `retired` | Archived; hidden from `/v1/models` and blocked for chat |

Allowed transitions are enforced by `POST /api/registry/models/{id}/transition` (see `GET /api/registry/lifecycle`).

## API

- `GET /api/registry/models` — list entries
- `GET /api/registry/models/{id}` — detail + model card
- `PATCH /api/registry/models/{id}` — description, benchmark JSON, edit card
- `POST /api/registry/models/{id}/transition` — `{ "lifecycle_state": "production" }`
- `POST /api/registry/models/{id}/refresh-card` — regenerate card from job metadata

OpenAI-compatible `GET /v1/models` lists non-retired models in serving states.

## Best practices

1. **Train** → registry entry `trained` with auto metadata (job, dataset, hyperparameters, metrics).
2. **Test** → transition to `testing`; run `/v1/chat/completions` with low temperature.
3. **Benchmark** → `PATCH` with `benchmark` payload; state moves to `benchmarked`.
4. **Promote** → `production` only after review of the model card limitations section.
5. **Retire** → `retired` when superseded; delete the training job only if artifacts may be removed.

Model cards follow a Hugging Face–inspired layout (description, intended use, data, metrics, limitations, API example).
