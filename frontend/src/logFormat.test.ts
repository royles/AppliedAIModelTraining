import assert from "node:assert/strict";
import { boxColumnAligned, formatLogEntry, formatLogLines } from "./logFormat";

const TS = "2026-10-19T14:34:11.185693+00:00";
const BANNER = `╔═══════════════════════════╗
║ Model task provisioning ║
╚═══════════════════════════╝`;

const prefixLen = `[${TS}] `.length;

const formatted = formatLogEntry(TS, BANNER);
assert.equal(
  formatted,
  `[${TS}] ╔═══════════════════════════╗\n` +
    `${" ".repeat(prefixLen)}║ Model task provisioning ║\n` +
    `${" ".repeat(prefixLen)}╚═══════════════════════════╝`,
);

assert.ok(boxColumnAligned(formatted, prefixLen));

const multi = formatLogLines([
  { ts: TS, message: BANNER },
  { ts: TS, message: "[>>] single line" },
]);
assert.ok(multi.includes("single line"));
assert.ok(boxColumnAligned(formatLogEntry(TS, BANNER), prefixLen));

console.log("logFormat.test.ts: ok");
