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

const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const spawnOptions = {
  cwd: root,
  stdio: "inherit",
  detached: process.platform !== "win32",
};
const children = [
  spawn(
    python,
    ["-m", "uvicorn", "backend.server:app", "--host", "127.0.0.1", "--port", "8008"],
    {
      ...spawnOptions,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    },
  ),
  spawn(npm, ["run", "dev:site"], spawnOptions),
];

let stopping = false;
function stop(exitCode = 0) {
  if (stopping) return;
  stopping = true;
  children.forEach(terminateProcessTree);
  const fallback = setTimeout(() => process.exit(exitCode), 1_500);
  fallback.unref();
  Promise.all(
    children.map(
      (child) =>
        new Promise((resolve) => {
          if (child.exitCode !== null || child.signalCode !== null) resolve();
          else child.once("exit", resolve);
        }),
    ),
  ).finally(() => process.exit(exitCode));
}

for (const child of children) {
  child.on("error", (error) => {
    console.error(error);
    stop(1);
  });
  child.on("exit", (code, signal) => {
    if (!stopping && code !== null) stop(code);
    if (!stopping && signal) stop(1);
  });
}

process.on("SIGINT", () => stop(0));
process.on("SIGTERM", () => stop(0));
