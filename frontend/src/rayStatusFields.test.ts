import assert from "node:assert/strict";
import { rayStatusFingerprint, rayStatusToFields } from "./rayStatusFields";

const sample = {
  initialized: true,
  runtime: "initialized",
  nodes: 2,
  job_ray_address: "127.0.0.1:6379",
  cluster_resources: { GPU: 1, CPU: 8.0, node: 1.0 },
};

const fields = rayStatusToFields(sample);
const labels = fields.map((f) => f.label);
assert.deepEqual(labels.slice(0, 4), [
  "Job Ray address",
  "Ray initialized (API process)",
  "Runtime",
  "Nodes",
]);

const resourceLabels = fields.filter((f) => f.id.startsWith("cluster_resources.")).map((f) => f.label);
assert.deepEqual(resourceLabels, ["CPU", "GPU", "node"]);

const shuffled = {
  ...sample,
  cluster_resources: { node: 1.0, GPU: 1, CPU: 8.0 },
};
assert.equal(rayStatusFingerprint(sample), rayStatusFingerprint(shuffled));

console.log("rayStatusFields.test.ts: ok");
