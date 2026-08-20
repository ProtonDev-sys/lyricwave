from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.job_state import JobState


class ExportProcessingMetadataTest(unittest.TestCase):
    def test_public_job_exposes_full_reproducible_model_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "LYRICWAVE_FAST_WHISPER_MODEL": "owner/custom-whisper",
                "LYRICWAVE_FAST_DEMUCS_MODEL": "custom-demucs",
                "LYRICWAVE_FAST_ALIGNER_MODEL": "owner/custom-aligner",
            },
            clear=True,
        ):
            root = Path(directory)
            job = JobState(
                id="a" * 32,
                filename="track.ogg",
                source_path=root / "track.ogg",
                work_dir=root,
                language="english",
                quality="fast",
                duration=12.5,
                device="Test GPU",
            )
            payload = job.public(include_result=False)

        self.assertEqual(payload["language"], "english")
        self.assertEqual(payload["quality"], "fast")
        self.assertEqual(payload["device"], "Test GPU")
        self.assertEqual(payload["separation_model"], "custom-demucs")
        self.assertEqual(payload["transcription_model"], "custom-whisper")
        self.assertEqual(
            payload["transcription_model_id"],
            "owner/custom-whisper",
        )
        self.assertEqual(
            payload["alignment_model_requested"],
            "owner/custom-aligner",
        )


if __name__ == "__main__":
    unittest.main()
