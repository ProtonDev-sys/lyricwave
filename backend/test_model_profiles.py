import os
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from backend.config import (
    DEFAULT_QUALITY,
    alignment_backend,
    alignment_model_id,
    asr_backend,
    asr_model_id,
    fallback_asr_model_id,
    public_model_profiles,
)
from backend.model_runtime import QwenASRAdapter


class ModelProfileTest(unittest.TestCase):
    def test_recommended_profile_is_the_default(self) -> None:
        self.assertEqual(DEFAULT_QUALITY, "balanced")
        self.assertEqual(asr_backend("fast"), "whisper")
        self.assertEqual(asr_backend("balanced"), "qwen3")
        self.assertEqual(asr_model_id("balanced"), "Qwen/Qwen3-ASR-0.6B-hf")
        self.assertEqual(asr_model_id("accurate"), "Qwen/Qwen3-ASR-1.7B-hf")
        self.assertEqual(
            fallback_asr_model_id("accurate"),
            "openai/whisper-large-v3",
        )
        self.assertEqual(alignment_backend("balanced"), "qwen3")
        self.assertEqual(
            alignment_model_id("balanced"),
            "Qwen/Qwen3-ForcedAligner-0.6B-hf",
        )
        self.assertEqual(
            [item["id"] for item in public_model_profiles()],
            ["fast", "balanced", "accurate"],
        )

    def test_generic_model_overrides_are_mode_specific(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LYRICWAVE_BALANCED_ASR_MODEL": "example/qwen",
                "LYRICWAVE_ACCURATE_FALLBACK_ASR_MODEL": "example/whisper",
                "LYRICWAVE_BALANCED_ALIGNER_MODEL": "example/aligner",
            },
            clear=True,
        ):
            self.assertEqual(asr_model_id("balanced"), "example/qwen")
            self.assertEqual(fallback_asr_model_id("accurate"), "example/whisper")
            self.assertEqual(alignment_model_id("balanced"), "example/aligner")

    def test_backend_only_overrides_select_compatible_default_models(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LYRICWAVE_BALANCED_ASR_BACKEND": "whisper",
                "LYRICWAVE_BALANCED_ALIGNER_BACKEND": "ctc",
            },
            clear=True,
        ):
            self.assertEqual(
                asr_model_id("balanced"),
                "openai/whisper-large-v3-turbo",
            )
            self.assertEqual(
                alignment_model_id("balanced"),
                "facebook/wav2vec2-base-960h",
            )

        with patch.dict(
            os.environ,
            {
                "LYRICWAVE_FAST_ASR_BACKEND": "qwen3",
                "LYRICWAVE_FAST_ALIGNER_BACKEND": "qwen3",
            },
            clear=True,
        ):
            self.assertEqual(asr_model_id("fast"), "Qwen/Qwen3-ASR-0.6B-hf")
            self.assertEqual(
                alignment_model_id("fast"),
                "Qwen/Qwen3-ForcedAligner-0.6B-hf",
            )

    def test_alignment_can_be_explicitly_disabled(self) -> None:
        with patch.dict(
            os.environ,
            {"LYRICWAVE_BALANCED_ALIGNER_BACKEND": "none"},
            clear=True,
        ):
            self.assertEqual(alignment_backend("balanced"), "none")
            self.assertEqual(alignment_model_id("balanced"), "")


class QwenAdapterTest(unittest.TestCase):
    def test_adapter_returns_the_existing_pipeline_shape(self) -> None:
        class Batch(dict):
            def to(self, *_: object) -> "Batch":
                return self

        class Processor:
            def apply_transcription_request(self, **kwargs: object) -> Batch:
                self.request = kwargs
                return Batch(input_ids=np.zeros((1, 3), dtype=np.int64))

            def decode(self, *_: object, **__: object) -> list[dict[str, str]]:
                return [{"language": "English", "transcription": "test lyric"}]

        class Output:
            def __getitem__(self, _: object) -> "Output":
                return self

        class Model:
            device = "cuda:0"
            dtype = object()

            def generate(self, **_: object) -> Output:
                return Output()

        fake_torch = types.SimpleNamespace(inference_mode=lambda: _NullContext())
        processor = Processor()
        with patch.dict(sys.modules, {"torch": fake_torch}):
            result = QwenASRAdapter(Model(), processor)(
                np.ones(16_000, dtype=np.float32),
                generate_kwargs={"language": "english", "max_new_tokens": 64},
            )
        self.assertEqual(result["text"], "test lyric")
        self.assertEqual(result["language"], "English")
        self.assertEqual(result["chunks"][0]["timestamp"], (0.0, 1.0))
        self.assertEqual(processor.request["language"], "English")


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
