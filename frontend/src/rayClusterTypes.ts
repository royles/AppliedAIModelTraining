export type RayClusterMetric = {
  key: string;
  label: string;
  value: number;
  display: string;
  unit: string;
};

export type RayClusterComponent = {
  id: string;
  role: string;
  title: string;
  subtitle: string;
  healthy: boolean;
  host: string;
  node_id?: string;
  cdsw_worker_id?: string | null;
  metrics: RayClusterMetric[];
};

export type RayClusterLink = {
  from: string;
  to: string;
  healthy: boolean;
  kind: string;
  detail?: string;
};

export type RayClusterTopology = {
  version?: number;
  gcs_address?: string | null;
  gcs_reachable?: boolean;
  components?: RayClusterComponent[];
  links?: RayClusterLink[];
  summary?: Record<string, number>;
  distributed?: DistributedTrainingStatus;
  note?: string;
};

export type DistributedTrainingStatus = {
  backend: string;
  collective_backend?: string | null;
  nccl_defaults_enabled?: boolean;
  mode_label?: string;
  multi_train_ranks?: boolean;
  multi_engine_cluster?: boolean;
  status: string;
  active: boolean;
  multi_worker: boolean;
  multi_node: boolean;
  summary: string;
  configured: {
    cdsw_worker_engines: number;
    ray_train_workers: number;
    gpus_per_worker: number;
  };
  observed: {
    world_size: number;
    ranks_live: number;
    all_ranks_reported: boolean;
  };
  ranks: Array<{
    rank: number;
    display_rank: number;
    world_size: number;
    device: string;
    host?: string;
    state?: string;
  }>;
};

export function parseTopology(status: Record<string, unknown> | null | undefined): RayClusterTopology | null {
  const topo = status?.topology;
  if (!topo || typeof topo !== "object") {
    return null;
  }
  return topo as RayClusterTopology;
}

export function parseDistributed(status: Record<string, unknown> | null | undefined): DistributedTrainingStatus | null {
  const d = status?.distributed ?? (status?.topology as RayClusterTopology | undefined)?.distributed;
  if (!d || typeof d !== "object") {
    return null;
  }
  return d as DistributedTrainingStatus;
}
