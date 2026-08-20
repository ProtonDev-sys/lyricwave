from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from backend.lyric_evaluation import WordToken, evaluate, load_tokens, main, normalize_text


def token(text: str, start: float | None = None, end: float | None = None) -> WordToken:
    return WordToken(text, normalize_text(text), start, end, "lead")


class LyricEvaluationTest(unittest.TestCase):
    def test_normalization_is_case_and_punctuation_insensitive(self) -> None:
        self.assertEqual(normalize_text("  HéLLo—WORLD!  "), "hélloworld")
        self.assertEqual(normalize_text("Don’t"), "don't")

    def test_export_loader_flattens_lines_and_excludes_adlibs_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "track.json"
            path.write_text(
                json.dumps(
                    {
                        "lines": [
                            {
                                "kind": "lead",
                                "words": [
                                    {"text": "Main", "start": 1, "end": 1.4}
                                ],
                            },
                            {
                                "kind": "adlib",
                                "words": [
                                    {"text": "yeah", "start": 1.2, "end": 1.5}
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                [item.normalized for item in load_tokens(path)],
                ["main"],
            )
            self.assertEqual(
                [
                    item.normalized
                    for item in load_tokens(path, include_adlibs=True)
                ],
                ["main", "yeah"],
            )

    def test_exact_text_reports_timing_error_and_bias(self) -> None:
        reference = [token("one", 1.0, 1.4), token("two", 2.0, 2.3)]
        candidate = [token("one", 1.1, 1.5), token("two", 2.2, 2.4)]
        metrics = evaluate(reference, candidate)
        self.assertEqual(metrics["wer"], 0)
        self.assertEqual(metrics["matches"], 2)
        self.assertEqual(metrics["timing"]["onset_mae_ms"], 150.0)
        self.assertEqual(metrics["timing"]["onset_bias_ms"], 150.0)
        self.assertEqual(metrics["timing"]["onset_p90_ms"], 190.0)
        self.assertEqual(metrics["timing"]["reference_coverage"], 1.0)

    def test_word_errors_are_split_into_standard_operations(self) -> None:
        reference = [token("one"), token("two"), token("three")]
        candidate = [token("one"), token("too"), token("four"), token("five")]
        metrics = evaluate(reference, candidate)
        self.assertEqual(metrics["matches"], 1)
        self.assertEqual(metrics["substitutions"], 2)
        self.assertEqual(metrics["deletions"], 0)
        self.assertEqual(metrics["insertions"], 1)
        self.assertEqual(metrics["wer"], 1.0)

    def test_missing_timestamps_keep_text_metrics_and_report_no_timing(self) -> None:
        metrics = evaluate([token("one")], [token("one")])
        self.assertEqual(metrics["wer"], 0)
        self.assertEqual(metrics["timing"]["matched_words"], 0)
        self.assertIsNone(metrics["timing"]["onset_mae_ms"])

    def test_cli_compares_multiple_labelled_candidates_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.json"
            fast = root / "fast.json"
            accurate = root / "accurate.json"
            payload = {
                "words": [{"text": "hello", "start": 1, "end": 1.3}]
            }
            reference.write_text(json.dumps(payload), encoding="utf-8")
            fast.write_text(json.dumps(payload), encoding="utf-8")
            accurate.write_text(
                json.dumps({"words": [{"text": "wrong"}]}),
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main(
                    [
                        str(reference),
                        f"fast={fast}",
                        f"accurate={accurate}",
                        "--json",
                    ]
                )
            self.assertEqual(result, 0)
            parsed = json.loads(output.getvalue())
            self.assertEqual(
                [item["label"] for item in parsed["results"]],
                ["fast", "accurate"],
            )
            self.assertEqual(parsed["results"][0]["metrics"]["wer"], 0)
            self.assertEqual(parsed["results"][1]["metrics"]["wer"], 1.0)

    def test_invalid_partial_timestamps_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "track.json"
            path.write_text(
                json.dumps({"words": [{"text": "word", "start": 1}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "both start and end"):
                load_tokens(path)

    def test_empty_reference_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference contains no"):
            evaluate([], [token("word")])


if __name__ == "__main__":
    unittest.main()
