import type { TrainingJob } from "./api";
import { resolveApiUrl } from "./api";

type Props = {
  job: TrainingJob;
  compact?: boolean;
};

export function ModelInferenceLinks({ job, compact = false }: Props) {
  if (job.status !== "succeeded" || !job.openai_model_id) {
    return null;
  }

  const modelId = job.openai_model_id;
  const modelMetaUrl = resolveApiUrl(`/v1/models/${encodeURIComponent(modelId)}`);
  const chatUrl = resolveApiUrl("/v1/chat/completions");
  const swaggerUrl = resolveApiUrl("/api");

  if (compact) {
    return (
      <p className="inference-links compact">
        <a href={modelMetaUrl} target="_blank" rel="noopener noreferrer">
          Model endpoint
        </a>
        <span aria-hidden> · </span>
        <a href={swaggerUrl} target="_blank" rel="noopener noreferrer">
          Try chat in Swagger
        </a>
      </p>
    );
  }

  return (
    <article className="card inference-links">
      <h3>Model ready for testing</h3>
      <p className="hint">
        OpenAI-compatible inference is live on this application host. Model id:{" "}
        <code>{modelId}</code>
      </p>
      <ul className="inference-link-list">
        <li>
          <a href={modelMetaUrl} target="_blank" rel="noopener noreferrer">
            GET /v1/models/{modelId}
          </a>
          <span className="hint"> — verify registration</span>
        </li>
        <li>
          <a href={chatUrl} target="_blank" rel="noopener noreferrer">
            POST /v1/chat/completions
          </a>
          <span className="hint"> — chat endpoint (use Swagger or curl)</span>
        </li>
        <li>
          <a href={swaggerUrl} target="_blank" rel="noopener noreferrer">
            Open Swagger (/api)
          </a>
          <span className="hint"> — try POST /v1/chat/completions interactively</span>
        </li>
      </ul>
      <details className="inference-curl">
        <summary>Example curl</summary>
        <pre>
{`curl -sS "${chatUrl}" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${modelId}",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'`}
        </pre>
      </details>
    </article>
  );
}
