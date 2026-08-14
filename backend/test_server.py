import unittest

import numpy as np

from backend.server import (
    _alignment_spoken_form,
    _ctc_gap_is_silent,
    _deduplicate_words,
    _parenthetical_adlib_flags,
    _polish_word_timings,
    _rms_vocal_regions,
    _resolve_overlapping_words,
    group_words_into_lines,
)


class TimingHelpersTest(unittest.TestCase):
    def test_ctc_alignment_expands_numeric_lyrics(self) -> None:
        self.assertEqual(_alignment_spoken_form("21"), "TWENTYONE")
        self.assertEqual(_alignment_spoken_form("24/7"), "TWENTYFOURSEVEN")
        self.assertEqual(_alignment_spoken_form("$100"), "ONEHUNDREDDOLLARS")
        self.assertEqual(_alignment_spoken_form("3.14"), "THREEPOINTONEFOUR")
        self.assertEqual(_alignment_spoken_form("21st"), "TWENTYFIRST")

    def test_overlap_duplicates_are_removed(self) -> None:
        words = [
            {"text": "hello", "start": 1.0, "end": 1.3},
            {"text": "hello", "start": 1.2, "end": 1.5},
            {"text": "world", "start": 1.6, "end": 2.0},
        ]
        result = _deduplicate_words(words)
        self.assertEqual([word["text"] for word in result], ["hello", "world"])
        self.assertEqual(result[0]["end"], 1.5)

    def test_lines_break_on_long_gaps(self) -> None:
        words = [
            {"text": "first", "start": 1.0, "end": 1.3},
            {"text": "line", "start": 1.4, "end": 1.8},
            {"text": "second", "start": 3.0, "end": 3.4},
        ]
        lines = group_words_into_lines(words)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[1]["start"], 3.0)

    def test_higher_confidence_window_wins_overlap(self) -> None:
        words = [
            {
                "text": "correct",
                "start": 1.0,
                "end": 1.4,
                "_segment": 1,
                "_segment_confidence": 0.8,
                "_word_confidence": 0.7,
            },
            {
                "text": "guess",
                "start": 1.1,
                "end": 1.5,
                "_segment": 2,
                "_segment_confidence": 0.2,
                "_word_confidence": 0.3,
            },
            {
                "text": "continues",
                "start": 1.6,
                "end": 2.0,
                "_segment": 2,
                "_segment_confidence": 0.2,
                "_word_confidence": 0.4,
            },
        ]
        result = _deduplicate_words(_resolve_overlapping_words(words))
        self.assertEqual([word["text"] for word in result], ["correct", "continues"])

    def test_repeated_lyric_in_one_phrase_is_preserved(self) -> None:
        words = [
            {"text": "yeah", "start": 1.0, "end": 1.2, "_segment": 3},
            {"text": "yeah", "start": 1.3, "end": 1.5, "_segment": 3},
        ]
        result = _deduplicate_words(words)
        self.assertEqual([word["text"] for word in result], ["yeah", "yeah"])

    def test_character_timing_survives_public_word_cleanup(self) -> None:
        words = [
            {
                "text": "conflicted",
                "start": 1.0,
                "end": 2.2,
                "_timing": [
                    {"start": 1.05, "end": 1.18, "fill": 0.1},
                    {"start": 1.18, "end": 1.31, "fill": 0.2},
                    {"start": 1.31, "end": 1.44, "fill": 0.3},
                    {
                        "start": 1.82,
                        "end": 1.95,
                        "fill": 0.4,
                        "pause_before": True,
                    },
                    {"start": 1.95, "end": 2.14, "fill": 1.0},
                ],
            }
        ]
        result = _deduplicate_words(words)
        self.assertEqual(len(result[0]["timing"]), 5)
        self.assertEqual(result[0]["timing"][2]["fill"], 0.3)
        self.assertEqual(result[0]["timing"][3]["start"], 1.82)
        self.assertTrue(result[0]["timing"][3]["pause_before"])
        self.assertEqual(result[0]["timing"][-1]["fill"], 1.0)

    def test_ctc_gap_distinguishes_silence_from_a_held_note(self) -> None:
        sample_rate = 16_000
        held = np.full(sample_rate, 0.08, dtype=np.float32)
        self.assertFalse(
            _ctc_gap_is_silent(held, 0.2, 0.8, 0.1, 0.9, sample_rate)
        )

        split = held.copy()
        split[int(0.3 * sample_rate) : int(0.7 * sample_rate)] = 0
        self.assertTrue(
            _ctc_gap_is_silent(split, 0.2, 0.8, 0.1, 0.9, sample_rate)
        )

    def test_phrase_boundaries_and_adlibs_create_distinct_lines(self) -> None:
        words = [
            {"text": "lead", "start": 1.0, "end": 1.3, "kind": "lead", "phrase": 1},
            {"text": "vocal", "start": 1.4, "end": 1.8, "kind": "lead", "phrase": 1},
            {"text": "new", "start": 2.0, "end": 2.2, "kind": "lead", "phrase": 2},
            {"text": "phrase", "start": 2.3, "end": 2.7, "kind": "lead", "phrase": 2},
            {"text": "(yeah)", "start": 2.1, "end": 2.5, "kind": "adlib", "phrase": 7},
        ]
        lines = group_words_into_lines(words)
        self.assertEqual([line["kind"] for line in lines], ["lead", "lead", "adlib"])
        self.assertEqual([len(line["words"]) for line in lines], [2, 2, 1])

    def test_timing_polish_removes_lead_overlap(self) -> None:
        words = [
            {"text": "one", "start": 1.0, "end": 1.5, "_segment": 1, "_kind": "lead"},
            {"text": "two", "start": 1.4, "end": 1.8, "_segment": 2, "_kind": "lead"},
        ]
        result = _polish_word_timings(words)
        self.assertLessEqual(result[0]["end"], result[1]["start"])

    def test_rms_regions_skip_a_silent_intro(self) -> None:
        sample_rate = 100
        waveform = np.concatenate(
            (
                np.zeros(500, dtype=np.float32),
                np.full(1000, 0.25, dtype=np.float32),
            )
        )
        regions = _rms_vocal_regions(waveform, sample_rate=sample_rate)
        self.assertEqual(len(regions), 1)
        self.assertGreater(regions[0][0], 400)
        self.assertEqual(regions[0][1], waveform.size)

    def test_parenthetical_words_are_marked_as_one_adlib(self) -> None:
        flags = _parenthetical_adlib_flags(["main", "(backing", "vocal)", "main"])
        self.assertEqual(flags, [False, True, True, False])


if __name__ == "__main__":
    unittest.main()
