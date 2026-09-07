"""GPU-free structural regressions; these do not claim a measured ASR WER gain."""
from __future__ import annotations

import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.inference_pipeline import _align_segment_words, _transcribe_view, transcribe_vocals
from backend.lyric_processing import deduplicate_words, polish_word_timings
from backend.qwen_alignment import timestamps_to_words


def item(text: str, start: object = 0.0, end: object = 0.5) -> dict:
    return {"text": text, "start_time": start, "end_time": end}


def job() -> SimpleNamespace:
    return SimpleNamespace(id="test", quality="fast", language="english", cancelled=threading.Event(), update=lambda **_: None)


class AlignmentIntegrityTest(unittest.TestCase):
    def test_same_count_wrong_text_is_not_assigned_to_display_words(self):
        self.assertEqual(timestamps_to_words([{"text": "one"}, {"text": "two"}], [item("two"), item("one", .5, 1)]), [])

    def test_missing_or_extra_words_do_not_count_as_full_coverage(self):
        for target in [[item("one")], [item("one"), item("two", .5, 1), item("three", 1, 1.5)]]:
            with self.subTest(target=target):
                self.assertEqual(timestamps_to_words([{"text": "one"}, {"text": "two"}], target), [])

    def test_malformed_intervals_reject_whole_phrase(self):
        for start, end in [(None, 1), ("bad", 1), (float("nan"), 1), (0, float("inf")), (-.1, 1), (1, 1), (2, 1), (0, None)]:
            with self.subTest(start=start, end=end):
                self.assertEqual(timestamps_to_words([{"text": "one"}], [item("one", start, end)]), [])
        self.assertEqual(timestamps_to_words([{"text": "one"}], [{"text": "one"}]), [])

    def test_out_of_order_and_out_of_audio_bounds_are_rejected(self):
        raw = [{"text": "one"}, {"text": "two"}]
        self.assertEqual(timestamps_to_words(raw, [item("one", .5, 1), item("two", .1, .4)]), [])
        self.assertEqual(timestamps_to_words(raw, [item("one"), item("two", .5, 2)], 1), [])
        self.assertEqual(timestamps_to_words(raw, [item("one"), item("two", .5, 1)], float("nan")), [])

    def test_small_duration_rounding_is_clamped_not_extended(self):
        result = timestamps_to_words([{"text": "one"}], [item("one", .1, 1.01)], 1)
        self.assertEqual(result[0]["end"], 1)

    def test_unicode_punctuation_and_display_form_survive(self):
        raw = [{"text": "Ｆｉｒｓｔ,"}, {"text": "café!"}]
        result = timestamps_to_words(raw, [item("first"), item("café", .5, 1)])
        self.assertEqual([word["text"] for word in result], [word["text"] for word in raw])
        self.assertEqual(result[0]["_timing_source"], "qwen")

    def test_cjk_segmentation_carries_correct_source_metadata(self):
        raw = [{"text": "你好", "_kind": "lead", "_explicit_adlib": False}, {"text": "世界", "_kind": "adlib", "_explicit_adlib": True}]
        result = timestamps_to_words(raw, [item("你", 0, .2), item("好", .2, .4), item("世界", .4, .8)])
        self.assertEqual([word["_kind"] for word in result], ["lead", "lead", "adlib"])
        self.assertTrue(result[-1]["_explicit_adlib"])

    def test_resegmentation_never_merges_different_vocal_layers(self):
        self.assertEqual(timestamps_to_words([{"text": "one", "_kind": "lead"}, {"text": "two", "_kind": "adlib"}], [item("onetwo")]), [])

    def test_equal_counts_can_still_require_resegmentation(self):
        result = timestamps_to_words([{"text": "ab"}, {"text": "c"}], [item("a"), item("bc", .5, 1)])
        self.assertEqual([word["text"] for word in result], ["a", "bc"])

    def test_new_alignment_removes_stale_character_timing(self):
        raw = [{"text": "one", "_timing": [{"start": 8, "end": 9, "fill": 1}]}]
        self.assertNotIn("_timing", timestamps_to_words(raw, [item("one")])[0])
        self.assertIn("_timing", raw[0])


class PipelineAccuracyTest(unittest.TestCase):
    def test_malformed_asr_timestamps_do_not_crash_or_escape_crop(self):
        chunks = [{"text": "bad", "timestamp": values} for values in [(float("nan"), 1), (0, float("inf")), ("bad", 1), ([], 1), (2, 1), (5, 6)]]
        chunks.append({"text": "valid", "timestamp": (-1, 5)})
        segments = _transcribe_view(job(), lambda *_args, **_kwargs: {"chunks": chunks}, np.ones(16000, dtype=np.float32), [(0, 16000)], "center", 0, 1)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["words"][0]["text"], "valid")
        self.assertLessEqual(segments[0]["words"][0]["end"], 1)
        self.assertEqual(segments[0]["words"][0]["_timing_source"], "estimated")

    def test_ctc_exception_keeps_lead_but_not_unverified_side_text(self):
        raw = [{"text": "one", "start": 0, "end": 1}]
        for layer in ["center", "side"]:
            with self.subTest(layer=layer), patch("backend.inference_pipeline.alignment_backend", return_value="ctc"), patch("backend.inference_pipeline._align_words_ctc", side_effect=RuntimeError("unavailable")), patch("builtins.print"):
                result, _ = _align_segment_words(job(), {"source_layer": layer, "audio": np.ones(16000), "language": "English"}, raw)
                self.assertEqual(result, raw if layer == "center" else [])

    def test_held_note_longer_than_four_seconds_is_not_deleted(self):
        segment = {"start_time": 0, "audio": np.ones(96000), "words": [{"text": "home", "start": 0, "end": 6, "_timing_source": "estimated"}], "source_layer": "center", "language": "English", "kind": "lead", "explicit_adlib": False}
        with patch("backend.inference_pipeline.decode_mono", return_value=np.ones(96000)), patch("backend.inference_pipeline.load_whisper_pipeline"), patch("backend.inference_pipeline._adaptive_vocal_regions", return_value=[(0, 96000)]), patch("backend.inference_pipeline._transcribe_view", return_value=[segment]), patch("backend.inference_pipeline.alignment_backend", return_value="none"), patch("backend.inference_pipeline.release_whisper_model"), patch("backend.inference_pipeline.release_alignment_model"), patch("backend.inference_pipeline.release_qwen_alignment_model"):
            result = transcribe_vocals(job(), Path("unused.wav"))
        self.assertEqual(result[0]["text"], "home")
        self.assertEqual(result[0]["end"], 6)
        self.assertEqual(result[0]["timing_source"], "estimated")

    def test_polishing_adds_no_unrequested_onset_bias(self):
        raw = [{"text": "one", "start": 1.125, "end": 2.5, "_segment": 0}]
        result = polish_word_timings(raw)
        self.assertEqual((result[0]["start"], result[0]["end"]), (1.125, 2.5))
        self.assertEqual(raw[0]["start"], 1.125)

    def test_public_timing_provenance_is_whitelisted(self):
        for source in ["qwen", "ctc", "estimated", "unexpected"]:
            with self.subTest(source=source):
                result = deduplicate_words([{"text": "one", "start": 0, "end": 1, "_timing_source": source}])
                self.assertEqual(result[0].get("timing_source"), None if source == "unexpected" else source)
