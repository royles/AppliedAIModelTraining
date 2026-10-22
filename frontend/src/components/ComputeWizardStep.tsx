import { PlatformSettingsPanel } from "../PlatformSettingsPanel";
import { JobWorkerFields } from "./JobWorkerFields";
import type { WorkerSizePreset } from "../workerSizePresets";

type Props = {
  selectedProjectId: number | null;
  onProjectRaySettingsSaved?: () => void;
  workerPresetId: string | null;
  startBusy: boolean;
  jobWorkers: number;
  jobGpus: number;
  jobWorkerCpu: number;
  jobWorkerMemoryGb: number;
  onApplyWorkerPreset: (p: WorkerSizePreset) => void;
  onClearWorkerPreset: () => void;
  onJobWorkersChange: (n: number) => void;
  onJobGpusChange: (n: number) => void;
  onJobWorkerCpuChange: (n: number) => void;
  onJobWorkerMemoryGbChange: (n: number) => void;
};

export function ComputeWizardStep({
  selectedProjectId,
  onProjectRaySettingsSaved,
  ...jobProps
}: Props) {
  return (
    <div className="compute-wizard-step">
      <section className="compute-wizard-section" aria-labelledby="compute-job-heading">
        <h4 id="compute-job-heading" className="architecture-section-title">
          This job run
        </h4>
        <JobWorkerFields {...jobProps} />
      </section>

      <section className="compute-wizard-section" aria-labelledby="compute-cluster-heading">
        <h4 id="compute-cluster-heading" className="architecture-section-title">
          Project Ray &amp; CML placement
        </h4>
        <p className="hint architecture-section-lead">
          Cluster runtime and CML IDs are saved on the project. Engine count and sizing above apply only to this
          job&apos;s blueprint.
        </p>
        {selectedProjectId != null ? (
          <PlatformSettingsPanel
            embedded
            projectId={selectedProjectId}
            title="Project cluster settings"
            saveLabel="Save project settings"
            showCdswWorkerSizing={false}
            onSaved={onProjectRaySettingsSaved}
          />
        ) : (
          <p className="hint wizard-step-footnote">Select a project on step 1 to edit cluster settings.</p>
        )}
      </section>
    </div>
  );
}
