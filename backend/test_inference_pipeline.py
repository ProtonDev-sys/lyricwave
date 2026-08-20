import os
import sys
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np

from backend import ctc_alignment
from backend.ctc_alignment import (
    _alignment_model_candidates,
    _alignment_model_id,
)
from backend.inference_pipeline import _align_segment_words, _transcribe_view
from backend.inference_regions import (
    _adaptive_vocal_regions,
    _proportional_word_timings,
    _segment_uses_english_alignment,
)


class InferencePipelineHelpersTest(unittest.TestCase):
    def test_auto_detected_english_uses_ctc_alignment(self) -> None:
        self.assertTrue(_segment_uses_english_alignment("auto", "English"))
        self.assertTrue(_segment_uses_english_alignment("english", None))
        self.assertFalse(_segment_uses_english_alignment("auto", "Spanish"))
        self.assertFalse(_segment_uses_english_alignment("spanish", "English"))

    def test_quality_selects_a_larger_accurate_aligner(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_alignment_model_id("fast"), "facebook/wav2vec2-base-960h")
            self.assertEqual(
                _alignment_model_id("accurate"),
                "facebook/wav2vec2-large-960h-lv60-self",
            )

        with patch.dict(
            os.environ,
            {"LYRICWAVE_ACCURATE_ALIGNER_MODEL": "example/custom-aligner"},
            clear=True,
        ):
            self.assertEqual(_alignment_model_id("accurate"), "example/custom-aligner")
            self.assertEqual(
                _alignment_model_candidates("accurate"),
                [
                    "example/custom-aligner",
                    "facebook/wav2vec2-large-960h-lv60-self",
                    "facebook/wav2vec2-base-960h",
                ],
            )

    def test_successful_model_fallback_is_cached_for_the_job(self) -> None:
        model_calls: list[str] = []

        class FakeProcessorFactory:
            @staticmethod
            def from_pretrained(model_id: str) -> object:
                return {"model_id": model_id}

        class FakeModel:
            def eval(self) -> None:
                return None

            def to(self, device: str) -> None:
                self.device = device

        class FakeModelFactory:
            @staticmethod
            def from_pretrained(model_id: str, **_: object) -> FakeModel:
                model_calls.append(model_id)
                if model_id.endswith("lv60-self"):
                    raise RuntimeError("simulated CUDA OOM")
                return FakeModel()

        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = object()
        fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModelForCTC = FakeModelFactory
        fake_transformers.AutoProcessor = FakeProcessorFactory
        job = types.SimpleNamespace(
            quality="accurate",
            id="test-job",
            update=lambda **_: None,
        )

        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "transformers": fake_transformers},
        ), patch.dict(os.environ, {}, clear=True), patch("builtins.print"):
            ctc_alignment.release_alignment_model()
            try:
                first_model, _ = ctc_alignment._load_ctc_aligner(job)
                second_model, _ = ctc_alignment._load_ctc_aligner(job)
                self.assertIs(first_model, second_model)
                self.assertEqual(
                    model_calls,
                    [
                        "facebook/wav2vec2-large-960h-lv60-self",
                        "facebook/wav2vec2-base-960h",
                    ],
                )
            finally:
                ctc_alignment.release_alignment_model()

    def test_missing_ctc_path_falls_back_to_whisper_phrase_timing(self) -> None:
        raw_words = [{"text": "still", "start": 0.0, "end": 0.5}]
        segment = {
            "audio": np.full(8_000, 0.1, dtype=np.float32),
            "source_layer": "center",
            "language": "English",
        }
        job = types.SimpleNamespace(id="test-job", quality="fast")

        with patch(
            "backend.inference_pipeline._align_words_ctc",
            return_value=[],
        ), patch("builtins.print"):
            words, confidence = _align_segment_words(job, segment, raw_words)
        self.assertIs(words, raw_words)
        self.assertEqual(confidence, 0.0)

    def test_missing_side_ctc_path_is_rejected(self) -> None:
        raw_words = [{"text": "maybe", "start": 0.0, "end": 0.5}]
        segment = {
            "audio": np.full(8_000, 0.1, dtype=np.float32),
            "source_layer": "side",
            "language": "English",
        }
        job = types.SimpleNamespace(id="test-job", quality="accurate")
        with patch(
            "backend.inference_pipeline._align_words_ctc",
            return_value=[],
        ), patch("builtins.print"):
            words, confidence = _align_segment_words(job, segment, raw_words)
        self.assertEqual(words, [])
        self.assertEqual(confidence, 0.0)

    def test_low_confidence_lead_alignment_preserves_whisper_words(self) -> None:
        raw_words = [{"text": "sung", "start": 0.0, "end": 0.5}]
        aligned = [
            {
                "text": "sung",
                "start": 0.1,
                "end": 0.4,
                "_confidence": 0.01,
            }
        ]
        segment = {"audio": np.ones(1000), "source_layer": "center", "language": "English"}
        job = types.SimpleNamespace(id="test-job", quality="fast")
        with patch("backend.inference_pipeline._align_words_ctc", return_value=aligned), patch(
            "builtins.print"
        ):
            words, _ = _align_segment_words(job, segment, raw_words)
        self.assertIs(words, raw_words)

    def test_low_confidence_side_alignment_is_rejected(self) -> None:
        raw_words = [{"text": "maybe", "start": 0.0, "end": 0.5}]
        aligned = [
            {
                "text": "maybe",
                "start": 0.1,
                "end": 0.4,
                "_confidence": 0.01,
            }
        ]
        segment = {"audio": np.ones(1000), "source_layer": "side", "language": "English"}
        job = types.SimpleNamespace(id="test-job", quality="accurate")
        with patch("backend.inference_pipeline._align_words_ctc", return_value=aligned), patch(
            "builtins.print"
        ):
            words, _ = _align_segment_words(job, segment, raw_words)
        self.assertEqual(words, [])

    def test_transcriber_language_and_generation_settings_are_preserved(self) -> None:
        calls: list[dict[str, object]] = []
        job = types.SimpleNamespace(
            language="auto",
            quality="accurate",
            cancelled=threading.Event(),
            update=lambda **_: None,
        )

        def transcriber(_: np.ndarray, **kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return {
                "text": "a extraordinary",
                "chunks": [
                    {
                        "text": "a extraordinary",
                        "timestamp": (0.1, 1.1),
                        "language": "English",
                    }
                ],
            }

        waveform = np.full(2 * 16_000, 0.1, dtype=np.float32)
        segments = _transcribe_view(
            job,
            transcriber,
            waveform,
            [(0, waveform.size)],
            "center",
            62.0,
            16.0,
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["language"], "English")
        self.assertGreater(
            segments[0]["words"][1]["end"] - segments[0]["words"][1]["start"],
            segments[0]["words"][0]["end"] - segments[0]["words"][0]["start"],
        )
        self.assertTrue(calls[0]["return_language"])
        self.assertEqual(calls[0]["generate_kwargs"]["max_new_tokens"], 256)

    def test_fallback_word_timing_is_proportional_to_token_length(self) -> None:
        timings = _proportional_word_timings(
            ["a", "much", "longer"],
            start=1.0,
            duration=3.0,
        )
        durations = [end - start for start, end in timings]
        self.assertGreater(durations[2], durations[0])
        self.assertEqual(timings[0][0], 1.0)
        self.assertEqual(timings[-1][1], 4.0)

    def test_adaptive_regions_skip_a_silent_intro(self) -> None:
        sample_rate = 1_000
        waveform = np.concatenate(
            (
                np.zeros(5 * sample_rate, dtype=np.float32),
                np.full(10 * sample_rate, 0.20, dtype=np.float32),
            )
        )
        regions = _adaptive_vocal_regions(waveform, sample_rate=sample_rate)
        self.assertEqual(len(regions), 1)
        self.assertGreater(regions[0][0], 4 * sample_rate)
        self.assertEqual(regions[0][1], waveform.size)

    def test_quiet_verse_survives_a_much_louder_chorus(self) -> None:
        sample_rate = 1_000
        waveform = np.concatenate(
            (
                np.zeros(2 * sample_rate, dtype=np.float32),
                np.full(4 * sample_rate, 0.015, dtype=np.float32),
                np.zeros(8 * sample_rate, dtype=np.float32),
                np.full(4 * sample_rate, 0.25, dtype=np.float32),
            )
        )
        regions = _adaptive_vocal_regions(waveform, sample_rate=sample_rate)
        self.assertEqual(len(regions), 2)
        self.assertLess(regions[0][0], 2 * sample_rate)
        self.assertGreater(regions[0][1], 5 * sample_rate)
        self.assertLess(regions[1][0], 14 * sample_rate)
        self.assertEqual(regions[1][1], waveform.size)

    def test_extreme_dynamic_range_does_not_hide_soft_vocals(self) -> None:
        sample_rate = 1_000
        waveform = np.concatenate(
            (
                np.full(3 * sample_rate, 1.0, dtype=np.float32),
                np.zeros(2 * sample_rate, dtype=np.float32),
                np.full(2 * sample_rate, 0.002, dtype=np.float32),
            )
        )
        regions = _adaptive_vocal_regions(waveform, sample_rate=sample_rate)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0][0], 0)
        self.assertEqual(regions[0][1], waveform.size)

    def test_short_quiet_bridge_survives_after_a_loud_section(self) -> None:
        sample_rate = 1_000
        waveform = np.concatenate(
            (
                np.full(4 * sample_rate, 0.25, dtype=np.float32),
                np.zeros(int(1.5 * sample_rate), dtype=np.float32),
                np.full(int(1.2 * sample_rate), 0.015, dtype=np.float32),
            )
        )
        regions = _adaptive_vocal_regions(waveform, sample_rate=sample_rate)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0][0], 0)
        self.assertEqual(regions[0][1], waveform.size)

    def test_long_vocal_regions_are_split_without_gaps(self) -> None:
        sample_rate = 1_000
        waveform = np.full(70 * sample_rate, 0.12, dtype=np.float32)
        regions = _adaptive_vocal_regions(waveform, sample_rate=sample_rate)
        self.assertGreater(len(regions), 1)
        self.assertEqual(regions[0][0], 0)
        self.assertEqual(regions[-1][1], waveform.size)
        self.assertTrue(
            all(end - start <= int(28.5 * sample_rate) for start, end in regions)
        )
        self.assertTrue(
            all(
                regions[index][1] == regions[index + 1][0]
                for index in range(len(regions) - 1)
            )
        )

    def test_near_silent_side_noise_does_not_create_a_region(self) -> None:
        sample_rate = 1_000
        side = np.full(12 * sample_rate, 0.0002, dtype=np.float32)
        self.assertEqual(_adaptive_vocal_regions(side, sample_rate=sample_rate), [])

    def test_an_isolated_side_response_creates_its_own_region(self) -> None:
        sample_rate = 1_000
        side = np.zeros(12 * sample_rate, dtype=np.float32)
        side[7 * sample_rate : 8 * sample_rate] = 0.08
        regions = _adaptive_vocal_regions(side, sample_rate=sample_rate)
        self.assertEqual(len(regions), 1)
        self.assertLess(regions[0][0], 7 * sample_rate)
        self.assertGreater(regions[0][1], 8 * sample_rate)


if __name__ == "__main__":
    unittest.main()
