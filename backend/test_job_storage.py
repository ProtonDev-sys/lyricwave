from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import server
from backend.job_state import JobState
from backend.job_storage import (
    discard_uploaded_source,
    restored_duration,
    restored_filename,
    restored_source_path,
)


class JobStorageTest(unittest.TestCase):
    @staticmethod
    def _job(root: Path, source_path: Path) -> JobState:
        return JobState(
            id="a" * 32,
            filename="Artist - Track.mp3",
            source_path=source_path,
            work_dir=root,
            language="english",
            quality="fast",
            duration=8.5,
        )

    def test_discard_removes_only_the_uploaded_mix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp3"
            source.write_bytes(b"original mix")
            vocal = root / "separated" / "htdemucs" / "track" / "vocals.wav"
            vocal.parent.mkdir(parents=True)
            vocal.write_bytes(b"vocal stem")
            job = self._job(root, source)

            removed = discard_uploaded_source(job, vocal)

            self.assertTrue(removed)
            self.assertFalse(source.exists())
            self.assertEqual(vocal.read_bytes(), b"vocal stem")

    def test_discard_refuses_a_source_outside_the_job_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            source = Path(outside) / "source.mp3"
            source.write_bytes(b"outside")
            vocal = root / "vocals.wav"
            vocal.write_bytes(b"vocal")
            job = self._job(root, source)

            self.assertFalse(discard_uploaded_source(job, vocal))
            self.assertTrue(source.exists())

    def test_metadata_duration_does_not_probe_a_removed_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.discarded"

            duration = restored_duration(
                {"duration": 12.25},
                source,
                lambda _: self.fail("Metadata restoration must not probe removed audio."),
            )

            self.assertEqual(duration, 12.25)

    def test_legacy_job_can_probe_an_existing_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.flac"
            source.write_bytes(b"legacy")
            seen: list[Path] = []

            duration = restored_duration(
                {},
                source,
                lambda path: seen.append(path) or 7.75,
            )

            self.assertEqual(duration, 7.75)
            self.assertEqual(seen, [source])
            self.assertEqual(restored_source_path(Path(directory)), source)

    def test_filename_is_restored_from_sanitized_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            placeholder = Path(directory) / "source.discarded"
            self.assertEqual(
                restored_filename({"filename": "../Artist - Track.mp3"}, placeholder),
                "Artist - Track.mp3",
            )


class CompletedJobRestoreTest(unittest.TestCase):
    def tearDown(self) -> None:
        with server.JOBS_LOCK:
            server.JOBS.clear()

    def test_completed_job_restores_without_the_uploaded_mix(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            server, "JOB_ROOT", Path(directory)
        ), patch.object(
            server,
            "_probe_duration",
            side_effect=AssertionError("A source-free completed job must use metadata."),
        ):
            job_id = "b" * 32
            work_dir = Path(directory) / job_id
            vocal = work_dir / "separated" / "htdemucs" / "track" / "vocals.wav"
            vocal.parent.mkdir(parents=True)
            vocal.write_bytes(b"vocal")
            (work_dir / "job.json").write_text(
                json.dumps(
                    {
                        "schema": 2,
                        "filename": "Artist - Restored.mp3",
                        "language": "english",
                        "quality": "fast",
                        "duration": 8.5,
                        "device": "Test GPU",
                        "created_at": "2026-08-20T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            (work_dir / "inference-result.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "words": [
                            {
                                "text": "hello",
                                "start": 0.1,
                                "end": 0.5,
                                "kind": "lead",
                                "phrase": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            restored = server._restore_completed_job(job_id)

            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.filename, "Artist - Restored.mp3")
            self.assertEqual(restored.duration, 8.5)
            self.assertEqual(restored.vocal_path, vocal)
            self.assertEqual(restored.source_path.name, "source.discarded")
            self.assertFalse(restored.source_path.exists())


if __name__ == "__main__":
    unittest.main()
