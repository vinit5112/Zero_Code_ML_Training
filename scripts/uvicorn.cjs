/**
 * Run uvicorn from the repo root, preferring .venv's Python when present.
 * Used by `npm run dev` so one command works on Windows, macOS, and Linux.
 */
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const win = process.platform === "win32";
const venvPython = path.join(
  root,
  win ? ".venv\\Scripts\\python.exe" : ".venv/bin/python",
);

const executable = fs.existsSync(venvPython) ? venvPython : win ? "python" : "python3";

const port = process.env.API_PORT || process.env.PORT || "8000";

const child = spawn(
  executable,
  [
    "-m",
    "uvicorn",
    "app.main:app",
    "--reload",
    "--host",
    "127.0.0.1",
    "--port",
    port,
  ],
  {
    cwd: root,
    stdio: "inherit",
    env: process.env,
  },
);

child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  process.exit(code ?? 1);
});
