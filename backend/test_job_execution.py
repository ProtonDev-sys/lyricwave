from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from backend import server
from backend.job_execution import (
    attach_job_future,
    cancel_job_execution,
    clear_job_execution,
)
from backend.job_state import JobState


class JobExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.job = JobState(
            id="a" * 32,
            filename="track.mp3",
            source_path=root / "source.mp3",
            work_dir=root,
            language="english",
            quality="fast",
            duration=10,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_attach_tracks_a_pending_future(self) -> None:
        future: Future[Any] = Future()
        attach_job_future(self.job, future)
        self.assertIs(self.job.future, future)

    def test_attach_cancels_work_when_cancellation_won_the_race(self) -> None:
        self.job.cancelled.set()
        future: Future[Any] = Future()
        attach_job_future(self.job, future)
        self.assertTrue(future.cancelled())
        self.assertIsNone(self.job.future)

    def test_cancel_removes_a_pending_future_before_start(self) -> None:
        future: Future[Any] = Future()
        attach_job_future(self.job, future)

        cancelled = cancel_job_execution(
            self.job,
            status="Stopped",
            terminate_process=lambda _: self.fail("No process should be terminated."),
        )

        self.assertTrue(cancelled)
        self.assertTrue(future.cancelled())
        self.assertIsNone(self.job.future)
        self.assertTrue(self.job.cancelled.is_set())
        self.assertEqual(self.job.stage, "cancelled")

    def test_running_future_stays_visible_until_worker_cleanup(self) -> None:
        future: Future[Any] = Future()
        self.assertTrue(future.set_running_or_notify_cancel())
        attach_job_future(self.job, future)

        cancelled = cancel_job_execution(
            self.job,
            status="Stopped",
            terminate_process=lambda _: self.fail("No process should be terminated."),
        )

        self.assertFalse(cancelled)
        self.assertIs(self.job.future, future)
        clear_job_execution(self.job)
        self.assertIsNone(self.job.future)
        self.assertIsNone(self.job.process)

    def test_completed_job_without_live_handles_is_not_changed(self) -> None:
        self.job.stage = "complete"
        cancelled = cancel_job_execution(
            self.job,
            status="Stopped",
            terminate_process=lambda _: self.fail("No process should be terminated."),
        )
        self.assertFalse(cancelled)
        self.assertFalse(self.job.cancelled.is_set())
        self.assertEqual(self.job.stage, "complete")


class ServerCancellationTest(unittest.TestCase):
    def tearDown(self) -> None:
        with server.JOBS_LOCK:
            server.JOBS.clear()

    def test_cancel_endpoint_removes_queued_executor_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = JobState(
                id="b" * 32,
                filename="queued.mp3",
                source_path=root / "source.mp3",
                work_dir=root,
                language="english",
                quality="fast",
                duration=10,
            )
            future: Future[Any] = Future()
            attach_job_future(job, future)
            with server.JOBS_LOCK:
                server.JOBS[job.id] = job

            response = server.cancel_job(job.id)

            self.assertEqual(response.status_code, 200)
            self.assertTrue(future.cancelled())
            self.assertIsNone(job.future)
            self.assertEqual(job.stage, "cancelled")


if __name__ == "__main__":
    unittest.main()
