#!/usr/bin/env node
import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const indexHtml = path.join(root, "dist", "index.html");

if (!existsSync(indexHtml)) {
  console.error("Run npm run build first (dist/index.html missing).");
  process.exit(1);
}

const fileUrl = `file://${indexHtml}`;
const result = spawnSync(
  "npx",
  ["--yes", "@axe-core/cli", fileUrl, "--tags", "wcag2a,wcag2aa", "--exit"],
  { cwd: root, stdio: "inherit", shell: false },
);

process.exit(result.status ?? 1);
