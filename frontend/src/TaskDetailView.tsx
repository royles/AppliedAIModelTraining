import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type JobLogs, type TrainingJob } from "./api";
import { trainingJobListFieldsEqual } from "./jobUtils";
import { TaskCard } from "./TaskCard";
import { canRunJob, canStopJob } from "./jobActions";
import { formatLogLines, mergeSerialJobLogs } from "./logFormat";
import { RayClusterMap } from "./RayClusterMap";
import { rayStatusFingerprint } from "./rayStatusFields";

type Props = {
  job: TrainingJob;
  onBack: () => void;
  onRequestDelete: () => void;
  onRun: () => void;
  onStop: () => void | Promise<void>;
  actionBusy?: boolean;
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

export function TaskDetailView({
  job,
  onBack,
  onRequestDelete,
  onRun,
  onStop,
  actionBusy,
  onJobUpdated,
  onOpenRegistry,
}: Props) {
  const [logs, setLogs] = useState<JobLogs | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);
  const serialLogRef = useRef<HTMLDivElement>(null);
  const lastSerialTextRef = useRef("");
  const scrollStickRef = useRef(true);
  const onJobUpdatedRef = useRef(onJobUpdated);
  onJobUpdatedRef.current = onJobUpdated;
  const jobRef = useRef(job);
  jobRef.current = job;

  const serialLogText = useMemo(() => {
    if (!logs) return "";
    return formatLogLines(mergeSerialJobLogs(logs.framework, logs.training));
  }, [logs]);

  const logsInFlightRef = useRef(false);

  const loadLogs = useCallback(async () => {
    if (logsInFlightRef.current) {
      return;
    }
    logsInFlightRef.current = true;
    try {
      scrollStickRef.current = !serialLogRef.current || isNearBottom(serialLogRef.current);
      const [data, freshJob] = await Promise.all([
        api.getJobLogs(job.id),
        api.getJob(job.id),
      ]);
      setLogs(data);
      if (!trainingJobListFieldsEqual(jobRef.current, freshJob)) {
        onJobUpdatedRef.current?.(freshJob);
      }
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      logsInFlightRef.current = false;
    }
  }, [job.id]);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const schedule = () => {
      if (cancelled) return;
      const active =
        job.status === "running" ||
        job.status === "provisioning" ||
        job.status === "queued";
      const pollMs = active ? 3000 : 15000;
      timer = window.setTimeout(async () => {
        await loadLogs();
        schedule();
      }, pollMs);
    };

    void loadLogs();
    schedule();

    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [loadLogs, job.status]);

  const rayStatusKey = logs ? rayStatusFingerprint(logs.ray_status) : "";

  useEffect(() => {
    lastSerialTextRef.current = "";
    scrollStickRef.current = true;
  }, [job.id]);

  useEffect(() => {
    if (!logs || !serialLogText) return;
    if (serialLogText !== lastSerialTextRef.current) {
      if (scrollStickRef.current) {
        requestAnimationFrame(() => {
          scrollLogToBottom(serialLogRef.current);
        });
      }
      lastSerialTextRef.current = serialLogText;
    }
  }, [logs, rayStatusKey, serialLogText]);

  return (
    <div className="task-detail">
      <header className="task-detail-toolbar">
        <button type="button" className="btn-secondary task-detail-back" onClick={onBack}>
          ← Tasks
        </button>
        <div className="task-detail-actions">
          <button
            type="button"
            disabled={actionBusy || !canRunJob(job.status)}
            onClick={onRun}
          >
            Run
          </button>
          <button
            type="button"
            className="btn-secondary"
            disabled={actionBusy || stopping || !canStopJob(job.status)}
            onClick={() => {
              setStopping(true);
              void Promise.resolve(onStop()).finally(() => setStopping(false));
            }}
          >
            {stopping ? "Stopping…" : "Stop"}
          </button>
          <button
            type="button"
            className="btn-danger task-detail-delete"
            disabled={actionBusy}
            onClick={onRequestDelete}
          >
            Delete
          </button>
        </div>
      </header>

      {error && <div className="banner error">{error}</div>}

      <div className="task-detail-summary-row">
        <TaskCard job={job} variant="detail" onOpenRegistry={onOpenRegistry} />

        <section className="card task-detail-log-panel" aria-labelledby="task-run-log-title">
          <h3 id="task-run-log-title" className="task-detail-log-title">
            Provisioning &amp; training log
          </h3>
          <p className="hint">Framework and training output in time order</p>
          <div className="log-stream log-stream-serial" ref={serialLogRef}>
            <pre>{serialLogText || (logs ? "No log entries yet." : "Loading…")}</pre>
          </div>
        </section>
      </div>

      <section className="card log-panel log-panel-run log-panel-cluster">
        <h3>Framework &amp; Ray cluster</h3>
        <p className="hint">
          Control and Train on the top row; Ray workers on the bottom. Links use the band above and the corridor
          between rows. Refreshes every 5s.
        </p>
        <div className="log-ray-status ray-cluster-panel ray-cluster-panel-full">
          <RayClusterMap
            status={logs?.ray_status}
            jobStatus={job.status}
            loading={!logs}
            cdswEngines={job.num_workers}
          />
        </div>
      </section>
    </div>
  );
}
