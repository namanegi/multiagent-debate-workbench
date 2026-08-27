"""Run the API and web development servers together."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def npm_executable() -> str:
    return "npm.cmd" if os.name == "nt" else shutil.which("npm") or "npm"


def uv_executable() -> str:
    return "uv.exe" if os.name == "nt" else shutil.which("uv") or "uv"


def main() -> int:
    commands = [
        [
            uv_executable(),
            "run",
            "--directory",
            "apps/api",
            "uvicorn",
            "debate_api.main:app",
            "--app-dir",
            "src",
            "--reload",
            "--port",
            "8000",
        ],
        [npm_executable(), "--prefix", "apps/web", "run", "dev"],
    ]
    processes: list[subprocess.Popen[bytes]] = []

    try:
        for command in commands:
            processes.append(subprocess.Popen(command, cwd=ROOT))

        while True:
            completed = [process for process in processes if process.poll() is not None]
            if completed:
                return max(process.returncode or 0 for process in completed)
            time.sleep(0.25)
    except KeyboardInterrupt:
        return 130
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in processes:
            if process.poll() is None:
                process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
