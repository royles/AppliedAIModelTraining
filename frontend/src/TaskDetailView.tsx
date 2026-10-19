import { useCallback, useEffect, useRef, useState } from "react";
import { api, type JobLogs, type TrainingJob } from "./api";
import { jobArchitectureSummary, jobInitMode } from "./jobUtils";
import { ModelInferenceLinks } from "./ModelInferenceLinks";
import { confirmDeleteTask } from "./taskDelete";
import { formatLogLines } from "./logFormat";

type Props = {
  job: TrainingJob;
  onBack: () => void;
  onDelete: (jobId: number) => void | Promise<void>;
  onRestart: (jobId: number, resume: boolean) => void | Promise<void>;
  onJobUpdated?: (job: TrainingJob) => void;
  onOpenRegistry?: () => void;
};

const SCROLL_STICK_THRESHOLD_PX = 48;

function isNearBottom(el: HTMLDivElement): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= SCROLL_STICK_THRESHOLD_PX;
}

function scrollLogToBottom(el: HTMLDivElement | null) {
  if (el) {
    el.scrollTop = el.scrollHeight;
  }
}

export function TaskDetailView({ job, onBack, onDelete, onRestart, onJobUpdated, onOpenRegistry }: Props) {
  const [logs, setLogs] = useState<JobLogs | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [restarting, setRestarting] = useState(false);
  const rayStatusLogRef = useRef<HTMLDivElement>(null);
  const frameworkLogRef = useRef<HTMLDivElement>(null);
  const trainingLogRef = useRef<HTMLDivElement>(null);
  const lastRayStatusTextRef = useRef("");
  const lastFrameworkTextRef = useRef("");
  const lastTrainingTextRef = useRef("");
  const scrollStickRef = useRef({ ray: true, framework: true, training: true });

  const loadLogs = useCallback(async () => {
    try {
      scrollStickRef.current = {
        ray: !rayStatusLogRef.current || isNearBottom(rayStatusLogRef.current),
        framework: !frameworkLogRef.current || isNearBottom(frameworkLogRef.current),
        training: !trainingLogRef.current || isNearBottom(trainingLogRef.current),
      };
      const [data, freshJob] = await Promise.all([
        api.getJobLogs(job.id),
        api.getJob(job.id),
      ]);
      setLogs(data);
      onJobUpdated?.(freshJob);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [job.id, onJobUpdated]);

  useEffect(() => {
    loadLogs();
    const id = window.setInterval(loadLogs, 5000);
    return () => window.clearInterval(id);
  }, [loadLogs]);

  const rayStatusText = logs
    ? JSON.stringify(logs.ray_status, null, 2)
    : "Loading Ray status…";

  useEffect(() => {
    lastRayStatusTextRef.current = "";
    lastFrameworkTextRef.current = "";
    lastTrainingTextRef.current = "";
    scrollStickRef.current = { ray: true, framework: true, training: true };
  }, [job.id]);

  useEffect(() => {
    if (!logs) return;

    const frameworkText = formatLogLines(logs.framework);
    const trainingText = formatLogLines(logs.training);

    if (rayStatusText !== lastRayStatusTextRef.current) {
      if (scrollStickRef.current.ray) {
        scrollLogToBottom(rayStatusLogRef.current);
      }
      lastRayStatusTextRef.current = rayStatusText;
    }

    if (frameworkText !== lastFrameworkTextRef.current) {
      if (scrollStickRef.current.framework) {
        scrollLogToBottom(frameworkLogRef.current);
      }
      lastFrameworkTextRef.current = frameworkText;
    }

    if (trainingText !== lastTrainingTextRef.current) {
      if (scrollStickRef.current.training) {
        scrollLogToBottom(trainingLogRef.current);
      }
      lastTrainingTextRef.current = trainingText;
    }
  }, [logs, rayStatusText]);

  return (
    <div className="task-detail">
      <div className="task-detail-toolbar">
        <button type="button" className="btn-secondary" onClick={onBack}>
          ← All tasks
        </button>
        <div className="task-detail-title-wrap">
          <h2>{job.name}</h2>
          <span className={`pill ${job.status}`}>{job.status}</span>
        </div>
        <div className="task-detail-actions">
          {job.status === "failed" && (
            <button
              type="button"
              className="btn-secondary"
              disabled={restarting}
              onClick={() => {
                const resume = window.confirm(
                  "Restart this task with the same parameters?\n\n" +
                    "OK = resume from the last saved epoch/step if available.\n" +
                    "Cancel = start training over (keeps prior artifacts on disk).",
                );
                setRestarting(true);
                void onRestart(job.id, resume).finally(() => setRestarting(false));
              }}
            >
              {restarting ? "Restarting…" : "Restart task"}
            </button>
          )}
          <button
            type="button"
            className="btn-danger task-detail-delete"
            onClick={() => {
              if (confirmDeleteTask(job)) {
                void onDelete(job.id);
              }
            }}
          >
            Delete task
          </button>
        </div>
      </div>

      <article className="card task-detail-summary">
        <ul className="kv">
          <li><span>Task id</span><span>#{job.id}</span></li>
          <li><span>Init</span><span>{jobInitMode(job)}</span></li>
          <li><span>Architecture</span><span>{jobArchitectureSummary(job)}</span></li>
          <li><span>Workers / GPU</span><span>{job.num_workers} / {job.gpus_per_worker}</span></li>
          <li><span>Max steps</span><span>{job.max_steps}</span></li>
          <li><span>Ray</span><span><code>{job.ray_address ?? "local"}</code></span></li>
          {job.error_message && (
            <li className="task-error">
              <span>Error</span>
              <span>{job.error_message}</span>
            </li>
          )}
        </ul>
      </article>

      <ModelInferenceLinks job={job} onOpenRegistry={onOpenRegistry} />

      {error && <div className="banner error">{error}</div>}

      <div className="log-split">
        <section className="card log-panel log-panel-framework">
          <h3>Framework &amp; Ray status</h3>
          <p className="hint">Provisioning, workers, Ray cluster snapshot (refreshes every 5s)</p>
          <div className="log-ray-status" ref={rayStatusLogRef}>
            <strong>Ray status</strong>
            <pre>{rayStatusText}</pre>
          </div>
          <div className="log-stream log-stream-framework" ref={frameworkLogRef}>
            <pre>{formatLogLines(logs?.framework ?? [])}</pre>
          </div>
        </section>
        <section className="card log-panel log-panel-training">
          <h3>Training steps</h3>
          <p className="hint">Loss, learning rate, and checkpoint steps from the train loop</p>
          <div className="log-stream log-stream-training" ref={trainingLogRef}>
            <pre>{formatLogLines(logs?.training ?? [])}</pre>
          </div>
        </section>
      </div>
    </div>
  );
}
