import { useEffect, useState } from "react";
import { api, type Health, type TrainingJob } from "./api";
import {
  parseActiveJobsSummary,
  rayIndicatorFromHealth,
  tasksIndicator,
  type IndicatorTone,
} from "./platformStatus";
import { UI_POLL_INTERVAL_MS } from "./pollIntervals";

type Props = {
  jobs: TrainingJob[];
  onOpenTask?: (jobId: number) => void;
};

function toneClass(tone: IndicatorTone): string {
  return `pill platform-status-pill tone-${tone}`;
}

function StatusPill({
  label,
  title,
  tone,
  onClick,
}: {
  label: string;
  title: string;
  tone: IndicatorTone;
  onClick?: () => void;
}) {
  const Tag = onClick ? "button" : "span";
  return (
    <Tag
      type={onClick ? "button" : undefined}
      className={toneClass(tone)}
      title={title}
      onClick={onClick}
    >
      <span className="platform-status-dot" aria-hidden />
      {label}
    </Tag>
  );
}

export function PlatformStatusBar({ jobs, onOpenTask }: Props) {
  const [uxReady, setUxReady] = useState(true);
  const [apiTone, setApiTone] = useState<IndicatorTone>("loading");
  const [apiTitle, setApiTitle] = useState("Checking API…");
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        await api.uiBuildInfo();
        if (!cancelled) {
          setUxReady(true);
        }
      } catch {
        if (!cancelled) {
          setUxReady(true);
        }
      }

      try {
        const h = await api.health();
        if (!cancelled) {
          setHealth(h);
          setApiTone("ok");
          setApiTitle("API connected · /api/health OK");
        }
      } catch (e) {
        if (!cancelled) {
          setHealth(null);
          setApiTone("error");
          setApiTitle(e instanceof Error ? e.message : "API unreachable");
        }
      }
    }

    poll();
    const id = window.setInterval(poll, UI_POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const apiOk = apiTone === "ok";
  const ray = rayIndicatorFromHealth(health, apiOk);
  const jobSummary = parseActiveJobsSummary(health?.details);
  const tasks = tasksIndicator(jobs, jobSummary);
  const firstActiveId = jobSummary?.jobs[0]?.id ?? jobs.find((j) =>
    ["queued", "provisioning", "running"].includes(j.status),
  )?.id;

  const liveSummary = `${apiOk ? "API connected" : "API offline"} · ${ray.label} · ${tasks.label}`;

  return (
    <header className="platform-status-bar" aria-label="Platform status">
      <div className="visually-hidden" aria-live="polite" aria-atomic="true">
        {liveSummary}
      </div>
      <div className="platform-status-inner">
        <StatusPill
          label="UX ready"
          title={
            uxReady
              ? "Application shell loaded; static assets served"
              : "Loading interface…"
          }
          tone={uxReady ? "ok" : "loading"}
        />
        <StatusPill
          label={apiOk ? "API connected" : apiTone === "error" ? "API offline" : "API…"}
          title={apiTitle}
          tone={apiTone}
        />
        <StatusPill label={ray.label} title={ray.title} tone={ray.tone} />
        <StatusPill
          label={tasks.label}
          title={tasks.title}
          tone={tasks.tone}
          onClick={
            firstActiveId != null && onOpenTask
              ? () => onOpenTask(firstActiveId)
              : undefined
          }
        />
      </div>
    </header>
  );
}
