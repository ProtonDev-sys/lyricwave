import { spawn } from "node:child_process";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const windows = process.platform === "win32";
const command = windows ? "powershell.exe" : "bash";
const args = windows
  ? [
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      path.join(root, "scripts", "setup-local.ps1"),
    ]
  : [path.join(root, "scripts", "setup-local.sh")];

const child = spawn(command, args, { cwd: root, stdio: "inherit" });
child.on("error", (error) => {
  console.error(error);
  process.exitCode = 1;
});
child.on("exit", (code, signal) => {
  if (code !== null) process.exit(code);
  process.exit(signal ? 1 : 0);
});
