import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from backend.qwen_alignment import (
    align_words_qwen,
    qwen_alignment_language,
    timestamps_to_words,
)


class QwenAlignmentTest(unittest.TestCase):
    def test_language_selection_supports_interface_languages(self) -> None:
        self.assertEqual(qwen_alignment_language("auto", "English"), "English")
        self.assertEqual(qwen_alignment_language("spanish", None), "Spanish")
        self.assertEqual(qwen_alignment_language("auto", "Arabic"), None)

    def test_timestamp_mapping_preserves_display_tokens(self) -> None:
        raw = [
            {"text": "Hello,", "_kind": "lead"},
            {"text": "world", "_kind": "lead"},
        ]
        aligned = timestamps_to_words(
            raw,
            [
                {"text": "Hello", "start_time": 0.1, "end_time": 0.4},
                {"text": "world", "start_time": 0.4, "end_time": 0.8},
            ],
        )
        self.assertEqual([word["text"] for word in aligned], ["Hello,", "world"])
        self.assertEqual(aligned[1]["end"], 0.8)

    def test_aligner_uses_transformers_native_processor_contract(self) -> None:
        class Batch(dict):
            def to(self, *_: object) -> "Batch":
                return self

        class Processor:
            def prepare_forced_aligner_inputs(self, **kwargs: object):
                self.request = kwargs
                return Batch(input_ids=np.zeros((1, 2), dtype=np.int64)), [["one", "two"]]

            def decode_forced_alignment(self, **_: object):
                return [[
                    {"text": "one", "start_time": 0.0, "end_time": 0.3},
                    {"text": "two", "start_time": 0.3, "end_time": 0.7},
                ]]

        class Model:
            device = "cuda:0"
            dtype = object()
            config = types.SimpleNamespace(timestamp_token_id=1)

            def __call__(self, **_: object):
                return types.SimpleNamespace(logits=np.zeros((1, 2, 2)))

        job = types.SimpleNamespace(language="english", quality="balanced")
        fake_torch = types.SimpleNamespace(inference_mode=lambda: _NullContext())
        processor = Processor()
        with patch(
            "backend.qwen_alignment._load_qwen_aligner",
            return_value=(Model(), processor),
        ), patch.dict(sys.modules, {"torch": fake_torch}):
            words = align_words_qwen(
                job,
                np.ones(16_000, dtype=np.float32),
                [{"text": "one"}, {"text": "two"}],
                "English",
            )
        self.assertEqual([word["text"] for word in words], ["one", "two"])
        self.assertEqual(processor.request["language"], "English")


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
