import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";

export function resolveVenvPython(root) {
  return process.platform === "win32"
    ? path.join(root, ".venv", "Scripts", "python.exe")
    : path.join(root, ".venv", "bin", "python");
}

export function requireVenvPython(root) {
  const python = resolveVenvPython(root);
  if (!existsSync(python)) {
    throw new Error(
      "Local engine is not installed. Run `npm run setup:engine` once, then retry.",
    );
  }
  return python;
}

export function terminateProcessTree(child) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;

  if (process.platform === "win32" && child.pid) {
    spawnSync("taskkill", ["/pid", String(child.pid), "/t", "/f"], {
      stdio: "ignore",
      windowsHide: true,
    });
    return;
  }

  try {
    if (child.pid) process.kill(-child.pid, "SIGTERM");
    else child.kill("SIGTERM");
  } catch {
    try {
      child.kill("SIGTERM");
    } catch {
      // The process already exited.
    }
  }
}
