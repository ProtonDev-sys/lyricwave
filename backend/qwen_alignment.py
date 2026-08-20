from __future__ import annotations

import gc
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
) -> list[dict[str, Any]]:
    """Preserve display tokens when possible and otherwise use aligner segmentation."""

    if not raw_words or not timestamp_items:
        return []
    exact_count = len(raw_words) == len(timestamp_items)
    fallback_template = raw_words[0]
    aligned: list[dict[str, Any]] = []
    for index, item in enumerate(timestamp_items):
        start = float(item.get("start_time", 0.0))
        end = float(item.get("end_time", start))
        if end <= start:
            continue
        template = raw_words[index] if exact_count else fallback_template
        item_text = str(item.get("text", "")).strip()
        display_text = str(template.get("text", "")).strip() if exact_count else item_text
        if not display_text:
            continue
        word = dict(template)
        word.update(
            {
                "text": display_text,
                "start": start,
                "end": end,
                "_confidence": 1.0,
            }
        )
        aligned.append(word)

    if exact_count:
        return aligned

    source_text = _normalise_word(
        "".join(str(word.get("text", "")) for word in raw_words)
    )
    aligned_text = _normalise_word(
        "".join(str(word.get("text", "")) for word in aligned)
    )
    if (
        source_text
        and aligned_text
        and source_text not in aligned_text
        and aligned_text not in source_text
    ):
        return []
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
    return timestamps_to_words(raw_words, list(items))
