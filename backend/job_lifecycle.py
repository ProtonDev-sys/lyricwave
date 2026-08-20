from __future__ import annotations

import os
import shutil
import threading
import time
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path

from backend.config import TERMINAL_STAGES
from backend.job_state import JobState


_QUEUE_RETRY_AFTER_SECONDS = 5


def _bounded_integer_env(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Read an integer setting while keeping unsafe values inside fixed bounds."""

    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def max_pending_jobs() -> int:
    """Return the maximum number of uploads, queued jobs, and active jobs."""

    return _bounded_integer_env("LYRICWAVE_MAX_PENDING_JOBS", 2, 1, 16)


def job_retention_seconds() -> int:
    """Return how long terminal job directories remain available for restoration."""

    hours = _bounded_integer_env("LYRICWAVE_JOB_RETENTION_HOURS", 24, 1, 24 * 30)
    return hours * 60 * 60


def cleanup_interval_seconds() -> int:
    """Return the minimum interval between opportunistic cleanup passes."""

    return _bounded_integer_env("LYRICWAVE_CLEANUP_INTERVAL_SECONDS", 15 * 60, 30, 6 * 60 * 60)


class JobQueueFull(RuntimeError):
    """Raised before upload storage when the bounded local queue has no capacity."""

    retry_after_seconds = _QUEUE_RETRY_AFTER_SECONDS

    def __init__(self, capacity: int) -> None:
        super().__init__(
            "The local GPU queue is full. Wait for a current track to finish or cancel it, "
            "then try again."
        )
        self.capacity = capacity


class JobLifecycle:
    """Coordinate admission and retention for the in-memory and on-disk job registry."""

    def __init__(
        self,
        jobs: MutableMapping[str, JobState],
        jobs_lock: threading.RLock,
    ) -> None:
        self._jobs = jobs
        self._jobs_lock = jobs_lock
        self._lock = threading.RLock()
        self._reserved_uploads = 0
        self._last_cleanup = 0.0

    @staticmethod
    def _job_is_busy(job: JobState) -> bool:
        with job.lock:
            future = job.future
            process = job.process
            future_pending = future is not None and not future.done()
            process_running = process is not None and process.poll() is None
            return job.stage not in TERMINAL_STAGES or future_pending or process_running

    def busy_count(self) -> int:
        """Count active jobs plus upload slots reserved before registration."""

        with self._lock:
            with self._jobs_lock:
                jobs = list(self._jobs.values())
            return self._reserved_uploads + sum(self._job_is_busy(job) for job in jobs)

    @contextmanager
    def reserve(self, job_root: Path) -> Iterator[None]:
        """Reserve queue capacity before accepting bytes from an upload."""

        self.cleanup(job_root)
        with self._lock:
            with self._jobs_lock:
                jobs = list(self._jobs.values())
            busy_jobs = sum(self._job_is_busy(job) for job in jobs)
            capacity = max_pending_jobs()
            if busy_jobs + self._reserved_uploads >= capacity:
                raise JobQueueFull(capacity)
            self._reserved_uploads += 1
        try:
            yield
        finally:
            with self._lock:
                self._reserved_uploads = max(0, self._reserved_uploads - 1)

    def cleanup(self, job_root: Path, *, force: bool = False) -> int:
        """Remove expired terminal jobs from memory and disk without touching active work."""

        now_monotonic = time.monotonic()
        with self._lock:
            if (
                not force
                and now_monotonic - self._last_cleanup < cleanup_interval_seconds()
            ):
                return 0
            self._last_cleanup = now_monotonic

        cutoff = time.time() - job_retention_seconds()
        removed_ids: set[str] = set()
        active_ids: set[str] = set()

        with self._jobs_lock:
            for job_id, job in list(self._jobs.items()):
                if self._job_is_busy(job):
                    active_ids.add(job_id)
                    continue
                try:
                    modified = job.work_dir.stat().st_mtime
                except OSError:
                    modified = 0.0
                if modified < cutoff:
                    self._jobs.pop(job_id, None)
                    removed_ids.add(job_id)

        if not job_root.exists():
            return len(removed_ids)

        for directory in job_root.iterdir():
            try:
                if (
                    directory.is_dir()
                    and directory.name not in active_ids
                    and directory.stat().st_mtime < cutoff
                ):
                    shutil.rmtree(directory, ignore_errors=False)
                    removed_ids.add(directory.name)
            except OSError:
                continue
        return len(removed_ids)
