from __future__ import annotations

import gc
import os
import threading
from typing import Any

import numpy as np

from backend.inference_regions import SAMPLE_RATE
from backend.alignment_text import (
    alignment_spoken_form,
    ctc_gap_is_silent,
    ctc_token_spans,
)


_DEFAULT_ALIGNMENT_MODELS = {
    "fast": "facebook/wav2vec2-base-960h",
    "accurate": "facebook/wav2vec2-large-960h-lv60-self",
}

_ALIGN_MODEL: Any | None = None
_ALIGN_PROCESSOR: Any | None = None
_ALIGN_MODEL_ID = ""
_ALIGN_REQUEST_ID = ""
_ALIGN_LOCK = threading.RLock()


def alignment_model_name() -> str:
    return _ALIGN_MODEL_ID.split("/")[-1]


def _alignment_model_id(quality: str) -> str:
    mode = "accurate" if quality == "accurate" else "fast"
    mode_override = os.environ.get(f"LYRICWAVE_{mode.upper()}_ALIGNER_MODEL", "").strip()
    shared_override = os.environ.get("LYRICWAVE_ALIGNER_MODEL", "").strip()
    return mode_override or shared_override or _DEFAULT_ALIGNMENT_MODELS[mode]


def _alignment_model_candidates(quality: str) -> list[str]:
    mode = "accurate" if quality == "accurate" else "fast"
    requested = _alignment_model_id(mode)
    return list(
        dict.fromkeys(
            (
                requested,
                _DEFAULT_ALIGNMENT_MODELS[mode],
                _DEFAULT_ALIGNMENT_MODELS["fast"],
            )
        )
    )


def release_alignment_model() -> None:
    global _ALIGN_MODEL, _ALIGN_MODEL_ID, _ALIGN_PROCESSOR, _ALIGN_REQUEST_ID

    with _ALIGN_LOCK:
        _ALIGN_MODEL = None
        _ALIGN_PROCESSOR = None
        _ALIGN_MODEL_ID = ""
        _ALIGN_REQUEST_ID = ""
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _load_ctc_aligner(job: Any) -> tuple[Any, Any]:
    global _ALIGN_MODEL, _ALIGN_MODEL_ID, _ALIGN_PROCESSOR, _ALIGN_REQUEST_ID

    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    requested_model_id = _alignment_model_id(job.quality)
    candidates = _alignment_model_candidates(job.quality)

    def load(model_id: str) -> tuple[Any, Any]:
        global _ALIGN_MODEL, _ALIGN_MODEL_ID, _ALIGN_PROCESSOR, _ALIGN_REQUEST_ID

        job.update(status=f"Loading {model_id.split('/')[-1]} word aligner")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCTC.from_pretrained(
            model_id,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.to("cuda:0")
        _ALIGN_PROCESSOR = processor
        _ALIGN_MODEL = model
        _ALIGN_MODEL_ID = model_id
        _ALIGN_REQUEST_ID = requested_model_id
        job.update(alignment_model=model_id.split("/")[-1])
        return model, processor

    with _ALIGN_LOCK:
        if (
            _ALIGN_MODEL is not None
            and _ALIGN_PROCESSOR is not None
            and _ALIGN_REQUEST_ID == requested_model_id
        ):
            return _ALIGN_MODEL, _ALIGN_PROCESSOR

        if _ALIGN_MODEL is not None or _ALIGN_PROCESSOR is not None:
            release_alignment_model()

        failures: list[str] = []
        for candidate_index, model_id in enumerate(candidates):
            try:
                return load(model_id)
            except Exception as error:
                failures.append(f"{model_id}: {error}")
            release_alignment_model()
            if candidate_index + 1 < len(candidates):
                next_model_id = candidates[candidate_index + 1]
                print(
                    f"[align:{job.id[:8]}] {model_id} unavailable; "
                    f"falling back to {next_model_id}: {failures[-1]}",
                    flush=True,
                )

        raise RuntimeError(
            "Could not load a CTC alignment model: " + " | ".join(failures)
        )


def _align_words_ctc(
    job: Any,
    audio: np.ndarray,
    words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import torch

    model, processor = _load_ctc_aligner(job)
    tokenizer = processor.tokenizer
    blank_id = int(tokenizer.pad_token_id)
    delimiter = tokenizer.word_delimiter_token or "|"
    delimiter_id = int(tokenizer.convert_tokens_to_ids(delimiter))

    target_ids: list[int] = []
    token_owners: list[int | None] = []
    owner_token_indexes: dict[int, list[int]] = {}
    valid_words: list[tuple[int, dict[str, Any]]] = []
    for word_index, word in enumerate(words):
        normalized = alignment_spoken_form(str(word["text"]))
        character_ids: list[int] = []
        for character in normalized:
            token_id = int(tokenizer.convert_tokens_to_ids(character))
            if token_id == tokenizer.unk_token_id:
                continue
            character_ids.append(token_id)
        if not character_ids:
            continue
        if target_ids:
            target_ids.append(delimiter_id)
            token_owners.append(None)
        first_token_index = len(target_ids)
        target_ids.extend(character_ids)
        token_owners.extend([word_index] * len(character_ids))
        owner_token_indexes[word_index] = list(
            range(first_token_index, first_token_index + len(character_ids))
        )
        valid_words.append((word_index, word))

    if not target_ids or audio.size < 320:
        return []

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).input_values
    inputs = inputs.to(device="cuda:0", dtype=torch.float16)
    with torch.inference_mode():
        emissions = model(inputs).logits[0].float().log_softmax(dim=-1).cpu()
    del inputs

    token_spans = ctc_token_spans(emissions, target_ids, blank_id)
    if not token_spans:
        return []
    seconds_per_frame = (audio.size / SAMPLE_RATE) / emissions.shape[0]

    refined_spans: list[tuple[int, int]] = []
    for token_index, (start, end) in enumerate(token_spans):
        token_scores = emissions[start:end, target_ids[token_index]]
        blank_scores = emissions[start:end, blank_id]
        active = torch.nonzero(
            token_scores >= blank_scores,
            as_tuple=False,
        ).flatten()
        if active.numel():
            refined_spans.append(
                (start + int(active[0]), start + int(active[-1]) + 1)
            )
        else:
            peak = start + int(token_scores.argmax())
            refined_spans.append((peak, peak + 1))

    by_owner: dict[int, list[tuple[int, int]]] = {}
    for token_index, span in enumerate(refined_spans):
        owner = token_owners[token_index]
        if owner is not None:
            by_owner.setdefault(owner, []).append(span)

    aligned: list[dict[str, Any]] = []
    for word_index, word in valid_words:
        spans = by_owner.get(word_index)
        if not spans:
            continue
        token_indexes = owner_token_indexes[word_index]
        best_token_log_probs = [
            float(
                emissions[
                    token_spans[token_index][0] : token_spans[token_index][1],
                    target_ids[token_index],
                ].max()
            )
            for token_index in token_indexes
        ]
        confidence = (
            float(np.exp(np.mean(best_token_log_probs)))
            if best_token_log_probs
            else 0.0
        )
        word_start = spans[0][0] * seconds_per_frame
        word_end = spans[-1][1] * seconds_per_frame
        timing: list[dict[str, Any]] = []
        for index, (start, end) in enumerate(spans):
            unit: dict[str, Any] = {
                "start": round(start * seconds_per_frame, 3),
                "end": round(max(start + 1, end) * seconds_per_frame, 3),
                "fill": round((index + 1) / len(spans), 4),
            }
            if index > 0:
                previous_end = spans[index - 1][1] * seconds_per_frame
                current_start = start * seconds_per_frame
                if ctc_gap_is_silent(
                    audio,
                    previous_end,
                    current_start,
                    word_start,
                    word_end,
                ):
                    unit["pause_before"] = True
            timing.append(unit)
        aligned.append(
            {
                "text": word["text"],
                "start": round(word_start, 3),
                "end": round(word_end, 3),
                "_timing": timing,
                "_confidence": round(confidence, 4),
                "_kind": str(word.get("_kind", "lead")),
                "_explicit_adlib": bool(word.get("_explicit_adlib", False)),
                "_source_index": word_index,
            }
        )
    return aligned


def _mean_alignment_confidence(aligned: list[dict[str, Any]]) -> float:
    weights = [
        max(1, len(alignment_spoken_form(str(word["text"]))))
        for word in aligned
    ]
    if not weights:
        return 0.0
    return float(
        np.average(
            [float(word.get("_confidence", 0.0)) for word in aligned],
            weights=weights,
        )
    )


def _alignment_coverage(
    aligned: list[dict[str, Any]],
    raw_words: list[dict[str, Any]],
) -> float:
    expected = sum(
        max(1, len(alignment_spoken_form(str(word.get("text", "")))))
        for word in raw_words
        if alignment_spoken_form(str(word.get("text", "")))
    )
    covered = sum(
        max(1, len(alignment_spoken_form(str(word.get("text", "")))))
        for word in aligned
    )
    return min(1.0, covered / expected) if expected else 0.0
