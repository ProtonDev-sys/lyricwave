import { spawn } from "node:child_process";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { requireVenvPython, terminateProcessTree } from "./local-runtime.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let python;
try {
  python = requireVenvPython(root);
} catch (error) {
  console.error(`\n${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
}

const child = spawn(python, process.argv.slice(2), {
  cwd: root,
  stdio: "inherit",
  detached: process.platform !== "win32",
  env: { ...process.env, PYTHONUNBUFFERED: "1" },
});

let stopping = false;
async function stop(exitCode) {
  if (stopping) return;
  stopping = true;
  await terminateProcessTree(child);
  process.exit(exitCode);
}

child.on("error", (error) => {
  console.error(error);
  void stop(1);
});
child.on("exit", (code, signal) => {
  if (stopping) return;
  process.exit(code ?? (signal ? 1 : 0));
});
process.on("SIGINT", () => void stop(130));
process.on("SIGTERM", () => void stop(143));
