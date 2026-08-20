from __future__ import annotations

import subprocess
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from backend.config import TERMINAL_STAGES
from backend.job_state import JobState


ProcessTerminator = Callable[[subprocess.Popen[str]], None]


def attach_job_future(job: JobState, future: Future[Any]) -> None:
    """Attach a submitted future without losing a concurrent cancellation request."""

    with job.lock:
        if not job.cancelled.is_set():
            if not future.done():
                job.future = future
            return

    # Cancellation can win the race between registry insertion and executor
    # submission. Remove pending work immediately instead of leaving a no-op
    # future ahead of later tracks in the single-worker queue.
    if future.cancel():
        return
    with job.lock:
        if not future.done():
            job.future = future


def clear_job_execution(job: JobState) -> None:
    """Drop executor and subprocess handles when a worker reaches its hard boundary."""

    with job.lock:
        job.future = None
        job.process = None


def cancel_job_execution(
    job: JobState,
    *,
    status: str,
    terminate_process: ProcessTerminator,
) -> bool:
    """Cancel pending work or signal and terminate a running job.

    Returns ``True`` when the executor removed the future before it started.
    """

    with job.lock:
        future = job.future
        process = job.process
        terminal = job.stage in TERMINAL_STAGES
        future_pending = future is not None and not future.done()
        process_running = process is not None and process.poll() is None
        if terminal and not future_pending and not process_running:
            return False
        job.cancelled.set()

    if not terminal:
        job.update(stage="cancelled", status=status)

    cancelled_before_start = bool(future is not None and future.cancel())
    if cancelled_before_start:
        with job.lock:
            if job.future is future:
                job.future = None

    if process_running and process is not None:
        terminate_process(process)
    return cancelled_before_start
