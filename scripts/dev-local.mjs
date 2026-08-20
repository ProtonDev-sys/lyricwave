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
async function stop(exitCode = 0) {
  if (stopping) return;
  stopping = true;
  await Promise.allSettled(children.map((child) => terminateProcessTree(child)));
  process.exit(exitCode);
}

for (const child of children) {
  child.on("error", (error) => {
    console.error(error);
    void stop(1);
  });
  child.on("exit", (code, signal) => {
    if (stopping) return;
    void stop(code ?? (signal ? 1 : 0));
  });
}

process.on("SIGINT", () => void stop(0));
process.on("SIGTERM", () => void stop(0));
