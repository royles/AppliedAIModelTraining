import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type JobLogs, type TrainingJob } from "./api";
import { trainingJobListFieldsEqual } from "./jobUtils";
import { TaskCard } from "./TaskCard";
import { confirmDeleteTask } from "./taskDelete";
import { formatLogLines, mergeSerialJobLogs } from "./logFormat";
import { RayClusterMap } from "./RayClusterMap";
import { rayStatusFingerprint } from "./rayStatusFields";

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

  const loadLogs = useCallback(async () => {
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
    }
  }, [job.id]);

  useEffect(() => {
    loadLogs();
    const id = window.setInterval(loadLogs, 5000);
    return () => window.clearInterval(id);
  }, [loadLogs]);

  const rayStatusKey = logs ? rayStatusFingerprint(logs.ray_status) : "";

  useEffect(() => {
    lastSerialTextRef.current = "";
    scrollStickRef.current = true;
  }, [job.id]);

  useEffect(() => {
    if (!logs || !serialLogText) return;
    if (serialLogText !== lastSerialTextRef.current) {
      if (scrollStickRef.current) {
        scrollLogToBottom(serialLogRef.current);
      }
      lastSerialTextRef.current = serialLogText;
    }
  }, [logs, rayStatusKey, serialLogText]);

  return (
    <div className="task-detail">
      <header className="task-detail-top">
        <button type="button" className="btn-secondary task-detail-back" onClick={onBack}>
          ← Tasks
        </button>

        <TaskCard job={job} variant="detail" onOpenRegistry={onOpenRegistry} />

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
              {restarting ? "Restarting…" : "Restart"}
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
            Delete
          </button>
        </div>
      </header>

      {error && <div className="banner error">{error}</div>}

      <section className="card log-panel log-panel-run">
        <h3>Framework &amp; Ray cluster</h3>
        <p className="hint">Left-to-right: driver → head → Train → workers → train ranks (updates every 5s)</p>
        <div className="log-ray-status ray-cluster-panel ray-cluster-panel-full">
          <RayClusterMap
            status={logs?.ray_status}
            jobStatus={job.status}
            loading={!logs}
            cdswEngines={job.num_workers}
          />
        </div>
        <h4 className="task-run-log-title">Provisioning &amp; training log</h4>
        <p className="hint">Framework and training output in time order</p>
        <div className="log-stream log-stream-serial" ref={serialLogRef}>
          <pre>{serialLogText || (logs ? "No log entries yet." : "Loading…")}</pre>
        </div>
      </section>
    </div>
  );
}
