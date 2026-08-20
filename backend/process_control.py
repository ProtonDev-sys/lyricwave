from __future__ import annotations

import os
import signal
import subprocess
from typing import Any


_CREATE_NO_WINDOW = int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
_BELOW_NORMAL_PRIORITY_CLASS = int(
    getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
)


def worker_process_options(platform_name: str | None = None) -> dict[str, Any]:
    """Return Popen options that isolate every model worker and its descendants."""

    platform = os.name if platform_name is None else platform_name
    if platform == "nt":
        return {
            "creationflags": _CREATE_NO_WINDOW | _BELOW_NORMAL_PRIORITY_CLASS,
        }
    return {"start_new_session": True}


def _wait(process: subprocess.Popen[str], timeout: float) -> bool:
    try:
        process.wait(timeout=timeout)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


def _terminate_directly(
    process: subprocess.Popen[str],
    terminate_timeout: float,
    kill_timeout: float,
) -> None:
    try:
        process.terminate()
    except OSError:
        pass
    else:
        if _wait(process, terminate_timeout):
            return

    try:
        process.kill()
    except OSError:
        return
    _wait(process, kill_timeout)


def _terminate_windows_tree(
    process: subprocess.Popen[str],
    terminate_timeout: float,
) -> bool:
    try:
        completed = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
            timeout=max(1.0, terminate_timeout),
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and _wait(
        process,
        min(3.0, terminate_timeout),
    )


def _terminate_posix_group(
    process: subprocess.Popen[str],
    terminate_timeout: float,
    kill_timeout: float,
) -> bool:
    try:
        process_group = os.getpgid(process.pid)
    except ProcessLookupError:
        return True
    except OSError:
        return False

    # Only signal a group that this process leads. This prevents a
    # compatibility fallback or externally supplied process from
    # terminating lyricwave itself.
    if process_group != process.pid:
        return False

    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    if _wait(process, terminate_timeout):
        return True

    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    _wait(process, kill_timeout)
    return True


def terminate_process_tree(
    process: subprocess.Popen[str],
    *,
    platform_name: str | None = None,
    terminate_timeout: float = 10.0,
    kill_timeout: float = 5.0,
) -> None:
    """Terminate a model worker and every child process it started."""

    if process.poll() is not None:
        return

    platform = os.name if platform_name is None else platform_name
    terminated = (
        _terminate_windows_tree(process, terminate_timeout)
        if platform == "nt"
        else _terminate_posix_group(
            process,
            terminate_timeout,
            kill_timeout,
        )
    )
    if not terminated and process.poll() is None:
        _terminate_directly(process, terminate_timeout, kill_timeout)
