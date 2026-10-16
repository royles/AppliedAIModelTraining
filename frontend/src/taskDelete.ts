export function confirmDeleteTask(job: { id: number; name: string }): boolean {
  return window.confirm(
    `Delete task #${job.id} "${job.name}"?\n\nThis removes the job record, logs, and registered /v1 model entry. Running workers may take a moment to stop.`,
  );
}
