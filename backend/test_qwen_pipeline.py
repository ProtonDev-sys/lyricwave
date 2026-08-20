import os
import types
import unittest
from unittest.mock import patch

import numpy as np

from backend.inference_pipeline import _align_segment_words


class QwenPipelineRoutingTest(unittest.TestCase):
    def test_recommended_profile_routes_supported_languages_to_qwen(self) -> None:
        raw_words = [{"text": "hola", "start": 0.0, "end": 0.5}]
        segment = {
            "audio": np.full(8_000, 0.1, dtype=np.float32),
            "source_layer": "center",
            "language": "Spanish",
        }
        updates: list[dict[str, object]] = []
        job = types.SimpleNamespace(
            id="test-job",
            quality="balanced",
            language="auto",
            update=lambda **values: updates.append(values),
        )
        with patch(
            "backend.inference_pipeline.align_words_qwen",
            return_value=[{"text": "hola", "start": 0.1, "end": 0.4}],
        ), patch(
            "backend.inference_pipeline.qwen_alignment_model_name",
            return_value="Qwen3-ForcedAligner-0.6B-hf",
        ):
            words, confidence = _align_segment_words(job, segment, raw_words)
        self.assertEqual(words[0]["start"], 0.1)
        self.assertEqual(confidence, 1.0)
        self.assertIn(
            {"alignment_model": "Qwen3-ForcedAligner-0.6B-hf"},
            updates,
        )

    def test_none_backend_preserves_phrase_timing_without_loading_an_aligner(self) -> None:
        raw_words = [{"text": "hello", "start": 0.0, "end": 0.5}]
        segment = {
            "audio": np.full(8_000, 0.1, dtype=np.float32),
            "source_layer": "center",
            "language": "English",
        }
        job = types.SimpleNamespace(
            id="test-job",
            quality="balanced",
            language="auto",
            update=lambda **_: None,
        )
        with patch.dict(
            os.environ,
            {"LYRICWAVE_BALANCED_ALIGNER_BACKEND": "none"},
            clear=True,
        ), patch(
            "backend.inference_pipeline.align_words_qwen"
        ) as qwen_aligner, patch(
            "backend.inference_pipeline._align_words_ctc"
        ) as ctc_aligner:
            words, confidence = _align_segment_words(job, segment, raw_words)
        self.assertEqual(words, raw_words)
        self.assertEqual(confidence, 0.0)
        qwen_aligner.assert_not_called()
        ctc_aligner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
