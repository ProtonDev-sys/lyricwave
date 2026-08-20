import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";

const DEFAULT_GRACE_MS = 1_500;
const DEFAULT_KILL_WAIT_MS = 500;

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

function hasExited(child) {
  return !child || child.exitCode !== null || child.signalCode !== null;
}

function waitForExit(child, timeoutMs) {
  if (hasExited(child)) return Promise.resolve(true);
  return new Promise((resolve) => {
    let settled = false;
    let timer;
    const finish = (exited) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      child.off?.("exit", onExit);
      resolve(exited);
    };
    const onExit = () => finish(true);
    child.once("exit", onExit);
    timer = setTimeout(() => finish(false), Math.max(0, timeoutMs));
    timer.unref?.();
  });
}

export function signalProcessTree(
  child,
  signal,
  {
    platform = process.platform,
    killProcess = process.kill.bind(process),
    spawnSyncProcess = spawnSync,
  } = {},
) {
  if (hasExited(child)) return false;

  if (platform === "win32" && child.pid) {
    const completed = spawnSyncProcess(
      "taskkill",
      ["/pid", String(child.pid), "/t", "/f"],
      { stdio: "ignore", windowsHide: true },
    );
    if (!completed?.error && completed?.status === 0) return true;
  } else if (child.pid) {
    try {
      killProcess(-child.pid, signal);
      return true;
    } catch {
      // Fall back to the direct child handle when no process group exists.
    }
  }

  try {
    return child.kill(signal);
  } catch {
    return false;
  }
}

export async function terminateProcessTree(
  child,
  {
    platform = process.platform,
    graceMs = DEFAULT_GRACE_MS,
    killWaitMs = DEFAULT_KILL_WAIT_MS,
    killProcess = process.kill.bind(process),
    spawnSyncProcess = spawnSync,
  } = {},
) {
  if (hasExited(child)) return;

  const signalOptions = { platform, killProcess, spawnSyncProcess };
  if (platform === "win32") {
    const exit = waitForExit(child, killWaitMs);
    signalProcessTree(child, "SIGKILL", signalOptions);
    await exit;
    return;
  }

  const gracefulExit = waitForExit(child, graceMs);
  signalProcessTree(child, "SIGTERM", signalOptions);
  if (await gracefulExit) return;

  const forcedExit = waitForExit(child, killWaitMs);
  signalProcessTree(child, "SIGKILL", signalOptions);
  await forcedExit;
}
