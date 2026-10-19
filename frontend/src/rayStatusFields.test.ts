import assert from "node:assert/strict";
import {
  rayResourceBarMax,
  rayStatusFingerprint,
  rayStatusToDashboard,
  rayStatusToFields,
} from "./rayStatusFields";

const sample = {
  initialized: true,
  runtime: "initialized",
  nodes: 2,
  job_ray_address: "127.0.0.1:6379",
  cluster_resources: { GPU: 1, CPU: 8.0, node: 1.0 },
};

const dash = rayStatusToDashboard(sample);
assert.equal(dash.stats.find((s) => s.id === "initialized")?.format, "pill");
assert.deepEqual(dash.resources.map((r) => r.label), ["CPU", "GPU", "node"]);
assert.equal(dash.resources[0].accent, "cpu");
assert.equal(rayResourceBarMax(dash), 8);

const fields = rayStatusToFields(sample);
assert.ok(fields.length >= 4);

const shuffled = {
  ...sample,
  cluster_resources: { node: 1.0, GPU: 1, CPU: 8.0 },
};
assert.equal(rayStatusFingerprint(sample), rayStatusFingerprint(shuffled));

console.log("rayStatusFields.test.ts: ok");
