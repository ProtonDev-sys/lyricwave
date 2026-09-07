from __future__ import annotations

import gc
import math
import re
import threading
import unicodedata
from typing import Any

import numpy as np

from backend.config import alignment_model_id


_SUPPORTED_LANGUAGES = {
    "chinese": "Chinese",
    "zh": "Chinese",
    "english": "English",
    "en": "English",
    "cantonese": "Cantonese",
    "yue": "Cantonese",
    "french": "French",
    "fr": "French",
    "german": "German",
    "de": "German",
    "italian": "Italian",
    "it": "Italian",
    "japanese": "Japanese",
    "ja": "Japanese",
    "korean": "Korean",
    "ko": "Korean",
    "portuguese": "Portuguese",
    "pt": "Portuguese",
    "russian": "Russian",
    "ru": "Russian",
    "spanish": "Spanish",
    "es": "Spanish",
}

_ALIGN_MODEL: Any | None = None
_ALIGN_PROCESSOR: Any | None = None
_ALIGN_MODEL_ID = ""
_ALIGN_LOCK = threading.RLock()


def qwen_alignment_language(job_language: str, detected_language: object) -> str | None:
    requested = str(job_language or "").strip().lower().replace("-", "")
    detected = str(detected_language or "").strip().lower().replace("-", "")
    if requested not in {"", "auto", "autodetect"}:
        return _SUPPORTED_LANGUAGES.get(requested)
    return _SUPPORTED_LANGUAGES.get(detected)


def qwen_alignment_model_name() -> str:
    return _ALIGN_MODEL_ID.split("/")[-1]


def release_qwen_alignment_model() -> None:
    global _ALIGN_MODEL, _ALIGN_MODEL_ID, _ALIGN_PROCESSOR

    with _ALIGN_LOCK:
        _ALIGN_MODEL = None
        _ALIGN_PROCESSOR = None
        _ALIGN_MODEL_ID = ""
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _load_qwen_aligner(job: Any) -> tuple[Any, Any]:
    global _ALIGN_MODEL, _ALIGN_MODEL_ID, _ALIGN_PROCESSOR

    import torch
    from transformers import AutoModelForTokenClassification, AutoProcessor

    model_id = alignment_model_id(job.quality)
    with _ALIGN_LOCK:
        if (
            _ALIGN_MODEL is not None
            and _ALIGN_PROCESSOR is not None
            and _ALIGN_MODEL_ID == model_id
        ):
            return _ALIGN_MODEL, _ALIGN_PROCESSOR
        if _ALIGN_MODEL is not None or _ALIGN_PROCESSOR is not None:
            release_qwen_alignment_model()

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        job.update(status=f"Loading {model_id.split('/')[-1]} word aligner")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForTokenClassification.from_pretrained(
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.eval()
        model.to("cuda:0")
        _ALIGN_MODEL = model
        _ALIGN_PROCESSOR = processor
        _ALIGN_MODEL_ID = model_id
        job.update(
            alignment_model=model_id.split("/")[-1],
            alignment_model_id=model_id,
        )
        return model, processor


def _normalise_word(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE)


def timestamps_to_words(
    raw_words: list[dict[str, Any]],
    timestamp_items: list[dict[str, Any]],
    audio_duration: float | None = None,
) -> list[dict[str, Any]]:
    """Validate the entire alignment before associating text with timestamps.

    Token counts alone do not establish correspondence. Require lossless text
    coverage, finite ordered intervals, and crop-local bounds. Re-tokenization
    (notably CJK) is allowed, but metadata follows the actual source characters.
    Returning [] deliberately invokes the pipeline's phrase/CTC fallback.
    """
    if not raw_words or not timestamp_items:
        return []
    source_tokens = [_normalise_word(word.get("text", "")) for word in raw_words]
    target_tokens = [_normalise_word(item.get("text", "")) for item in timestamp_items]
    if (
        not all(source_tokens)
        or not all(target_tokens)
        or "".join(source_tokens) != "".join(target_tokens)
    ):
        return []
    if audio_duration is not None and (
        not math.isfinite(audio_duration) or audio_duration <= 0
    ):
        return []

    intervals: list[tuple[float, float]] = []
    for item in timestamp_items:
        try:
            start = float(item["start_time"])
            end = float(item["end_time"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return []
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            return []
        if intervals and (start < intervals[-1][0] or end < intervals[-1][1]):
            return []
        if audio_duration is not None and end > audio_duration + 0.02:
            return []
        if audio_duration is not None:
            end = min(end, audio_duration)
            if end <= start:
                return []
        intervals.append((start, end))

    exact_tokens = source_tokens == target_tokens
    source_ends: list[int] = []
    position = 0
    for token in source_tokens:
        position += len(token)
        source_ends.append(position)
    aligned: list[dict[str, Any]] = []
    source_index = 0
    position = 0
    for index, (item, token, interval) in enumerate(zip(timestamp_items, target_tokens, intervals)):
        while source_index + 1 < len(source_ends) and position >= source_ends[source_index]:
            source_index += 1
        template = raw_words[source_index]
        # A merged token must not inherit a lead/ad-lib label from only one half.
        last_source = source_index
        while last_source + 1 < len(source_ends) and position + len(token) > source_ends[last_source]:
            last_source += 1
        if any(
            word.get("_kind", "lead") != template.get("_kind", "lead")
            for word in raw_words[source_index:last_source + 1]
        ):
            return []
        word = dict(raw_words[index] if exact_tokens else template)
        word.pop("_timing", None)
        word.update(
            text=str(raw_words[index]["text"] if exact_tokens else item["text"]).strip(),
            start=interval[0],
            end=interval[1],
            # Structural acceptance score, not a calibrated ASR probability.
            _confidence=1.0,
            _timing_source="qwen",
        )
        aligned.append(word)
        position += len(token)
    return aligned


def align_words_qwen(
    job: Any,
    audio: np.ndarray,
    raw_words: list[dict[str, Any]],
    detected_language: object,
) -> list[dict[str, Any]]:
    import torch

    language = qwen_alignment_language(job.language, detected_language)
    if not language or not raw_words or audio.size < 320:
        return []
    transcript = " ".join(
        str(word.get("text", "")) for word in raw_words
    ).strip()
    if not transcript:
        return []

    model, processor = _load_qwen_aligner(job)
    inputs, word_lists = processor.prepare_forced_aligner_inputs(
        audio=np.asarray(audio, dtype=np.float32),
        transcript=transcript,
        language=language,
    )
    inputs = inputs.to(model.device, model.dtype)
    with torch.inference_mode():
        outputs = model(**inputs)
    timestamp_batches = processor.decode_forced_alignment(
        logits=outputs.logits,
        input_ids=inputs["input_ids"],
        word_lists=word_lists,
        timestamp_token_id=model.config.timestamp_token_id,
    )
    items = timestamp_batches[0] if timestamp_batches else []
    return timestamps_to_words(raw_words, list(items), audio.size / 16_000)
