import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import server
from backend.job_state import JobState


class WorkerModelMetadataTest(unittest.TestCase):
    def tearDown(self) -> None:
        with server.JOBS_LOCK:
            server.JOBS.clear()

    @staticmethod
    def _job(root: Path, *, progress_file: Path | None = None) -> JobState:
        return JobState(
            id="a" * 32,
            filename="track.mp3",
            source_path=root / "source.mp3",
            work_dir=root,
            language="auto",
            quality="balanced",
            duration=10,
            progress_file=progress_file,
        )

    def test_progress_file_contains_the_models_actually_loaded_by_the_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress_path = root / "progress.json"
            job = self._job(root, progress_file=progress_path)
            job.update(
                transcription_backend="whisper",
                transcription_model="whisper-large-v3-turbo",
                alignment_model="wav2vec2-base-960h",
            )

            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["transcription_backend"], "whisper")
            self.assertEqual(payload["transcription_model"], "whisper-large-v3-turbo")
            self.assertEqual(payload["alignment_model"], "wav2vec2-base-960h")

    def test_worker_metadata_parser_ignores_empty_or_non_string_values(self) -> None:
        updates = server._worker_model_updates(
            {
                "transcription_backend": " whisper ",
                "transcription_model": "whisper-large-v3",
                "alignment_model": "",
                "unrelated": 123,
            }
        )
        self.assertEqual(
            updates,
            {
                "transcription_backend": "whisper",
                "transcription_model": "whisper-large-v3",
            },
        )
        self.assertEqual(server._worker_model_updates(None), {})

    def test_restored_job_preserves_fallback_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            server, "JOB_ROOT", Path(directory)
        ):
            job_id = "b" * 32
            work_dir = Path(directory) / job_id
            work_dir.mkdir()
            (work_dir / "source.mp3").write_bytes(b"legacy source")
            (work_dir / "job.json").write_text(
                json.dumps(
                    {
                        "schema": 2,
                        "filename": "track.mp3",
                        "language": "auto",
                        "quality": "balanced",
                        "duration": 10,
                        "device": "Test GPU",
                    }
                ),
                encoding="utf-8",
            )
            (work_dir / "inference-result.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "words": [{"text": "hello", "start": 0.1, "end": 0.5}],
                        "transcription_backend": "whisper",
                        "transcription_model": "whisper-large-v3-turbo",
                        "alignment_model": "wav2vec2-base-960h",
                    }
                ),
                encoding="utf-8",
            )

            restored = server._restore_completed_job(job_id)

            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.transcription_backend, "whisper")
            self.assertEqual(restored.transcription_model, "whisper-large-v3-turbo")
            self.assertEqual(restored.alignment_model, "wav2vec2-base-960h")
            public = restored.public()
            self.assertEqual(public["transcription_model"], "whisper-large-v3-turbo")
            self.assertEqual(public["alignment_model"], "wav2vec2-base-960h")


if __name__ == "__main__":
    unittest.main()
