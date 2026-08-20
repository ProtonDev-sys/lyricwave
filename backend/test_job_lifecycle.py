import asyncio
import io
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

from backend import server
from backend.job_lifecycle import (
    JobLifecycle,
    JobQueueFull,
    cleanup_interval_seconds,
    job_retention_seconds,
    max_pending_jobs,
)
from backend.job_state import JobState


class LifecycleSettingsTest(unittest.TestCase):
    def test_queue_and_retention_settings_are_bounded(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LYRICWAVE_MAX_PENDING_JOBS": "1000",
                "LYRICWAVE_JOB_RETENTION_HOURS": "0",
                "LYRICWAVE_CLEANUP_INTERVAL_SECONDS": "invalid",
            },
            clear=True,
        ):
            self.assertEqual(max_pending_jobs(), 16)
            self.assertEqual(job_retention_seconds(), 60 * 60)
            self.assertEqual(cleanup_interval_seconds(), 15 * 60)


class JobLifecycleTest(unittest.TestCase):
    @staticmethod
    def _job(root: Path, job_id: str, stage: str) -> JobState:
        work_dir = root / job_id
        work_dir.mkdir(parents=True)
        return JobState(
            id=job_id,
            filename="track.mp3",
            source_path=work_dir / "source.mp3",
            work_dir=work_dir,
            language="english",
            quality="fast",
            duration=10,
            stage=stage,
        )

    def test_reservations_prevent_unbounded_uploads(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"LYRICWAVE_MAX_PENDING_JOBS": "1"},
            clear=True,
        ):
            root = Path(directory)
            jobs: dict[str, JobState] = {}
            lifecycle = JobLifecycle(jobs, threading.RLock())
            with lifecycle.reserve(root):
                self.assertEqual(lifecycle.busy_count(), 1)
                with self.assertRaises(JobQueueFull):
                    with lifecycle.reserve(root):
                        self.fail("A second upload should not receive a queue slot.")
            self.assertEqual(lifecycle.busy_count(), 0)

    def test_cleanup_removes_only_expired_terminal_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"LYRICWAVE_JOB_RETENTION_HOURS": "1"},
            clear=True,
        ):
            root = Path(directory)
            terminal = self._job(root, "a" * 32, "complete")
            active = self._job(root, "b" * 32, "queued")
            old_time = time.time() - 2 * 60 * 60
            os.utime(terminal.work_dir, (old_time, old_time))
            os.utime(active.work_dir, (old_time, old_time))
            jobs = {terminal.id: terminal, active.id: active}
            lifecycle = JobLifecycle(jobs, threading.RLock())

            removed = lifecycle.cleanup(root, force=True)

            self.assertEqual(removed, 1)
            self.assertNotIn(terminal.id, jobs)
            self.assertFalse(terminal.work_dir.exists())
            self.assertIn(active.id, jobs)
            self.assertTrue(active.work_dir.exists())


class ApiAdmissionTest(unittest.TestCase):
    @staticmethod
    def _runtime() -> dict[str, object]:
        return {
            "ready": True,
            "cuda": True,
            "device": "Test GPU",
            "ffmpeg": True,
            "ffprobe": True,
            "demucs": True,
            "transformers": True,
        }

    def tearDown(self) -> None:
        with server.JOBS_LOCK:
            server.JOBS.clear()

    def test_full_queue_rejects_before_audio_is_stored_or_probed(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"LYRICWAVE_MAX_PENDING_JOBS": "1"},
            clear=True,
        ), patch.object(server, "JOB_ROOT", Path(directory)), patch.object(
            server, "_runtime_info", return_value=self._runtime()
        ), patch.object(
            server,
            "_probe_duration",
            side_effect=AssertionError("A rejected upload must not be probed."),
        ):
            active_dir = Path(directory) / ("c" * 32)
            active_dir.mkdir()
            active = JobState(
                id="c" * 32,
                filename="active.mp3",
                source_path=active_dir / "source.mp3",
                work_dir=active_dir,
                language="english",
                quality="fast",
                duration=10,
                stage="queued",
            )
            with server.JOBS_LOCK:
                server.JOBS[active.id] = active

            upload = UploadFile(filename="next.mp3", file=io.BytesIO(b"audio"))
            with self.assertRaises(HTTPException) as context:
                asyncio.run(
                    server.create_job(upload, language="english", quality="fast")
                )

            self.assertEqual(context.exception.status_code, 429)
            self.assertEqual(context.exception.headers, {"Retry-After": "5"})
            self.assertTrue(upload.file.closed)
            self.assertEqual([path.name for path in Path(directory).iterdir()], [active.id])


if __name__ == "__main__":
    unittest.main()
