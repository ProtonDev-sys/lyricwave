import asyncio
import io
import json
import os
import tempfile
import threading
import types
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

from backend import server
from backend.config import (
    demucs_model_name,
    normalise_language,
    normalise_quality,
    whisper_model_id,
)
from backend.inference_worker import _vram_fraction
from backend.job_state import JobState, write_json_atomic


class ConfigAndStateTest(unittest.TestCase):
    def test_language_aliases_are_normalised_and_unknown_values_rejected(self) -> None:
        self.assertEqual(normalise_language("EN"), "english")
        self.assertEqual(normalise_language("Auto-detect"), "auto")
        self.assertEqual(normalise_language("pt"), "portuguese")
        with self.assertRaises(ValueError):
            normalise_language("klingon")
        with self.assertRaises(ValueError):
            normalise_quality("ultra")

    def test_model_overrides_are_mode_specific(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LYRICWAVE_WHISPER_MODEL": "example/shared-whisper",
                "LYRICWAVE_ACCURATE_WHISPER_MODEL": "example/accurate-whisper",
                "LYRICWAVE_FAST_DEMUCS_MODEL": "example/fast-demucs",
            },
            clear=True,
        ):
            self.assertEqual(whisper_model_id("fast"), "example/shared-whisper")
            self.assertEqual(whisper_model_id("accurate"), "example/accurate-whisper")
            self.assertEqual(demucs_model_name("fast"), "example/fast-demucs")

    def test_atomic_json_writes_remain_valid_under_contention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "status.json"

            def write(index: int) -> None:
                write_json_atomic(target, {"index": index, "text": "lyric"})

            threads = [threading.Thread(target=write, args=(index,)) for index in range(20)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertIn(payload["index"], range(20))
            self.assertEqual(payload["text"], "lyric")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_vram_fraction_is_bounded(self) -> None:
        with patch.dict(os.environ, {"LYRICWAVE_VRAM_FRACTION": "0.83"}, clear=True):
            self.assertEqual(_vram_fraction(), 0.83)
        with patch.dict(os.environ, {"LYRICWAVE_VRAM_FRACTION": "2"}, clear=True):
            self.assertEqual(_vram_fraction(), 0.95)
        with patch.dict(os.environ, {"LYRICWAVE_VRAM_FRACTION": "invalid"}, clear=True):
            self.assertEqual(_vram_fraction(), 0.70)


class ApiBoundaryTest(unittest.TestCase):
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

    def test_invalid_audio_is_removed_after_probe_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            server, "JOB_ROOT", Path(directory)
        ), patch.object(server, "_runtime_info", return_value=self._runtime()), patch.object(
            server, "_probe_duration", side_effect=ValueError("bad audio")
        ):
            upload = UploadFile(filename="broken.mp3", file=io.BytesIO(b"not audio"))
            with self.assertRaises(HTTPException) as context:
                asyncio.run(server.create_job(upload, language="english", quality="fast"))
            self.assertEqual(context.exception.status_code, 400)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_valid_job_normalises_language_and_persists_metadata(self) -> None:
        submissions: list[tuple[object, ...]] = []
        submitted_futures: list[Future[None]] = []

        class FakeExecutor:
            def submit(self, *args: object) -> Future[None]:
                submissions.append(args)
                future: Future[None] = Future()
                submitted_futures.append(future)
                return future

        with tempfile.TemporaryDirectory() as directory, patch.object(
            server, "JOB_ROOT", Path(directory)
        ), patch.object(server, "_runtime_info", return_value=self._runtime()), patch.object(
            server, "_probe_duration", return_value=123.5
        ), patch.object(server, "EXECUTOR", FakeExecutor()):
            upload = UploadFile(filename="track.mp3", file=io.BytesIO(b"audio bytes"))
            response = asyncio.run(
                server.create_job(upload, language="EN", quality="FAST")
            )
            self.assertEqual(response["quality"], "fast")
            self.assertEqual(len(submissions), 1)
            self.assertEqual(len(submitted_futures), 1)
            job_id = str(response["id"])
            with server.JOBS_LOCK:
                self.assertIs(server.JOBS[job_id].future, submitted_futures[0])
            metadata = json.loads(
                (Path(directory) / job_id / "job.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["language"], "english")
            self.assertEqual(metadata["schema"], 2)
            submitted_futures[0].cancel()

    def test_blocking_admission_checks_run_off_the_event_loop(self) -> None:
        caller_thread = threading.get_ident()
        runtime_threads: list[int] = []
        probe_threads: list[int] = []
        submitted_futures: list[Future[None]] = []

        class FakeExecutor:
            def submit(self, *args: object) -> Future[None]:
                future: Future[None] = Future()
                submitted_futures.append(future)
                return future

        def runtime_probe() -> dict[str, object]:
            runtime_threads.append(threading.get_ident())
            return self._runtime()

        def duration_probe(_: Path) -> float:
            probe_threads.append(threading.get_ident())
            return 4.25

        with tempfile.TemporaryDirectory() as directory, patch.object(
            server, "JOB_ROOT", Path(directory)
        ), patch.object(
            server, "_runtime_info", side_effect=runtime_probe
        ), patch.object(
            server, "_probe_duration", side_effect=duration_probe
        ), patch.object(server, "EXECUTOR", FakeExecutor()):
            upload = UploadFile(
                filename="track.mp3",
                file=io.BytesIO(b"audio bytes"),
            )
            response = asyncio.run(
                server.create_job(upload, language="english", quality="fast")
            )

        self.assertEqual(response["duration"], 4.25)
        self.assertEqual(len(runtime_threads), 1)
        self.assertEqual(len(probe_threads), 1)
        self.assertNotEqual(runtime_threads[0], caller_thread)
        self.assertNotEqual(probe_threads[0], caller_thread)
        self.assertEqual(len(submitted_futures), 1)
        submitted_futures[0].cancel()

    def test_job_public_payload_uses_overridden_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "LYRICWAVE_FAST_WHISPER_MODEL": "example/whisper",
                "LYRICWAVE_FAST_DEMUCS_MODEL": "example/demucs",
            },
            clear=True,
        ):
            root = Path(directory)
            job = JobState(
                id="a" * 32,
                filename="track.mp3",
                source_path=root / "track.mp3",
                work_dir=root,
                language="english",
                quality="fast",
                duration=10,
            )
            payload = job.public(include_result=False)
            self.assertEqual(payload["transcription_model"], "whisper")
            self.assertEqual(payload["separation_model"], "example/demucs")


if __name__ == "__main__":
    unittest.main()
