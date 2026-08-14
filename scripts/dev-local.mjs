import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const python =
  process.platform === "win32"
    ? path.join(root, ".venv", "Scripts", "python.exe")
    : path.join(root, ".venv", "bin", "python");

if (!existsSync(python)) {
  console.error("\nLocal engine is not installed. Run `npm run setup:engine` once, then retry.\n");
  process.exit(1);
}

const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const children = [
  spawn(
    python,
    ["-m", "uvicorn", "backend.server:app", "--host", "127.0.0.1", "--port", "8008"],
    {
      cwd: root,
      stdio: "inherit",
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    },
  ),
  spawn(npm, ["run", "dev:site"], { cwd: root, stdio: "inherit" }),
];

let stopping = false;
function stop(exitCode = 0) {
  if (stopping) return;
  stopping = true;
  for (const child of children) {
    if (!child.killed) child.kill();
  }
  setTimeout(() => process.exit(exitCode), 250);
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
