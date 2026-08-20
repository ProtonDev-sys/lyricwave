import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";

import {
  signalProcessTree,
  terminateProcessTree,
} from "../scripts/local-runtime.mjs";

class FakeChild extends EventEmitter {
  constructor({ pid = 4321, exited = false } = {}) {
    super();
    this.pid = pid;
    this.exitCode = exited ? 0 : null;
    this.signalCode = null;
    this.directSignals = [];
  }

  kill(signal) {
    this.directSignals.push(signal);
    return true;
  }

  exit(code = 0) {
    this.exitCode = code;
    this.emit("exit", code, null);
  }
}

test("signals a POSIX process group rather than only the direct child", () => {
  const child = new FakeChild();
  const signals = [];
  const result = signalProcessTree(child, "SIGTERM", {
    platform: "linux",
    killProcess: (pid, signal) => signals.push([pid, signal]),
  });
  assert.equal(result, true);
  assert.deepEqual(signals, [[-child.pid, "SIGTERM"]]);
  assert.deepEqual(child.directSignals, []);
});

test("falls back to the direct child when the POSIX group is unavailable", () => {
  const child = new FakeChild();
  const result = signalProcessTree(child, "SIGTERM", {
    platform: "linux",
    killProcess: () => {
      throw new Error("missing group");
    },
  });
  assert.equal(result, true);
  assert.deepEqual(child.directSignals, ["SIGTERM"]);
});

test("uses taskkill for a Windows process tree", () => {
  const child = new FakeChild();
  const calls = [];
  const result = signalProcessTree(child, "SIGKILL", {
    platform: "win32",
    spawnSyncProcess: (...arguments_) => {
      calls.push(arguments_);
      return { status: 0 };
    },
  });
  assert.equal(result, true);
  assert.deepEqual(calls, [
    [
      "taskkill",
      ["/pid", String(child.pid), "/t", "/f"],
      { stdio: "ignore", windowsHide: true },
    ],
  ]);
  assert.deepEqual(child.directSignals, []);
});

test("graceful termination stops after the child exits", async () => {
  const child = new FakeChild();
  const signals = [];
  await terminateProcessTree(child, {
    platform: "linux",
    graceMs: 20,
    killWaitMs: 20,
    killProcess: (pid, signal) => {
      signals.push([pid, signal]);
      queueMicrotask(() => child.exit());
    },
  });
  assert.deepEqual(signals, [[-child.pid, "SIGTERM"]]);
});

test("termination escalates to SIGKILL after the grace period", async () => {
  const child = new FakeChild();
  const signals = [];
  await terminateProcessTree(child, {
    platform: "linux",
    graceMs: 1,
    killWaitMs: 20,
    killProcess: (pid, signal) => {
      signals.push([pid, signal]);
      if (signal === "SIGKILL") queueMicrotask(() => child.exit());
    },
  });
  assert.deepEqual(signals, [
    [-child.pid, "SIGTERM"],
    [-child.pid, "SIGKILL"],
  ]);
});

test("an already exited child is not signalled", async () => {
  const child = new FakeChild({ exited: true });
  let calls = 0;
  await terminateProcessTree(child, {
    platform: "linux",
    killProcess: () => {
      calls += 1;
    },
  });
  assert.equal(calls, 0);
  assert.deepEqual(child.directSignals, []);
});
