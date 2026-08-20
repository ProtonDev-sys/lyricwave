from __future__ import annotations

import gc
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import asr_backend, asr_model_id, fallback_asr_model_id
from backend.inference_regions import SAMPLE_RATE


_ASR_PIPELINE: Any | None = None
_ASR_CACHE_KEY = ""
_ASR_LOCK = threading.RLock()

_LANGUAGE_NAMES = {
    "english": "English",
    "spanish": "Spanish",
    "french": "French",
    "german": "German",
    "italian": "Italian",
    "portuguese": "Portuguese",
    "japanese": "Japanese",
    "korean": "Korean",
}


class QwenASRAdapter:
    """Expose Qwen3-ASR through the result shape used by the lyric pipeline."""

    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor

    def __call__(self, audio: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        import torch

        generation = dict(kwargs.get("generate_kwargs") or {})
        language = generation.get("language")
        if language:
            language = _LANGUAGE_NAMES.get(str(language).lower(), str(language))
        inputs = self.processor.apply_transcription_request(
            audio=np.asarray(audio, dtype=np.float32),
            language=language,
        ).to(self.model.device, self.model.dtype)
        max_new_tokens = int(generation.get("max_new_tokens", 256))
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        parsed = self.processor.decode(generated_ids, return_format="parsed")[0]
        text = str(parsed.get("transcription") or "").strip()
        detected_language = parsed.get("language") or language
        duration = len(audio) / SAMPLE_RATE
        return {
            "text": text,
            "language": detected_language,
            "chunks": [
                {
                    "text": text,
                    "timestamp": (0.0, duration),
                    "language": detected_language,
                }
            ]
            if text
            else [],
        }


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def release_asr_model() -> None:
    global _ASR_CACHE_KEY, _ASR_PIPELINE

    with _ASR_LOCK:
        _ASR_PIPELINE = None
        _ASR_CACHE_KEY = ""
    _release_cuda_cache()


def release_whisper_model() -> None:
    """Compatibility alias retained for existing worker and server imports."""

    release_asr_model()


def _load_whisper(model_id: str) -> Any:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation=os.environ.get("LYRICWAVE_WHISPER_ATTENTION", "sdpa"),
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.to("cuda:0")
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch.float16,
        device=0,
    )


def _load_qwen(model_id: str) -> QwenASRAdapter:
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation=os.environ.get("LYRICWAVE_QWEN_ATTENTION", "sdpa"),
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    model.to("cuda:0")
    return QwenASRAdapter(model, processor)


def load_whisper_pipeline(job: Any) -> Any:
    """Load the selected ASR backend, with a configured Whisper fallback."""

    global _ASR_CACHE_KEY, _ASR_PIPELINE

    requested_backend = asr_backend(job.quality)
    requested_model_id = asr_model_id(job.quality)
    requested_key = f"{requested_backend}:{requested_model_id}"
    with _ASR_LOCK:
        if _ASR_PIPELINE is not None and _ASR_CACHE_KEY == requested_key:
            job.update(
                progress=60.0,
                status=f"Using {requested_model_id.split('/')[-1]} on the GPU",
                transcription_backend=requested_backend,
                transcription_model=requested_model_id.split("/")[-1],
                transcription_model_id=requested_model_id,
            )
            return _ASR_PIPELINE

        if _ASR_PIPELINE is not None:
            release_asr_model()

        job.update(
            progress=60.0,
            status=f"Loading {requested_model_id.split('/')[-1]}",
        )
        try:
            pipeline_instance = (
                _load_qwen(requested_model_id)
                if requested_backend == "qwen3"
                else _load_whisper(requested_model_id)
            )
            actual_backend = requested_backend
            actual_model_id = requested_model_id
        except Exception as primary_error:
            fallback_model_id = fallback_asr_model_id(job.quality)
            if requested_backend == "whisper" or fallback_model_id == requested_model_id:
                raise
            print(
                f"[asr:{job.id[:8]}] {requested_model_id} unavailable; "
                f"falling back to {fallback_model_id}: {primary_error}",
                flush=True,
            )
            release_asr_model()
            job.update(status=f"Loading fallback {fallback_model_id.split('/')[-1]}")
            pipeline_instance = _load_whisper(fallback_model_id)
            actual_backend = "whisper"
            actual_model_id = fallback_model_id

        _ASR_PIPELINE = pipeline_instance
        _ASR_CACHE_KEY = f"{actual_backend}:{actual_model_id}"
        job.update(
            transcription_backend=actual_backend,
            transcription_model=actual_model_id.split("/")[-1],
            transcription_model_id=actual_model_id,
        )
        return _ASR_PIPELINE


def decode_mono(path: Path, view: str = "center") -> np.ndarray:
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
    ]
    if view == "side":
        command.extend(["-af", "pan=mono|c0=0.5*c0-0.5*c1,volume=1.75"])
    elif view != "center":
        raise ValueError(f"Unknown vocal view: {view}")
    command.extend(
        [
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-",
        ]
    )
    completed = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(completed.stdout, dtype=np.float32).copy()
