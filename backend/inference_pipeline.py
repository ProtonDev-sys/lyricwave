from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import alignment_backend, side_pass_enabled
from backend.ctc_alignment import (
    _align_words_ctc,
    _alignment_coverage,
    _mean_alignment_confidence,
    alignment_model_name,
    release_alignment_model,
)
from backend.inference_regions import (
    SAMPLE_RATE,
    _adaptive_vocal_regions,
    _normalise_language,
    _proportional_word_timings,
    _segment_uses_english_alignment,
)
from backend.job_state import JobCancelled, JobState
from backend.lyric_processing import (
    COMMON_HALLUCINATION_PATTERNS,
    MIN_ADLIB_ALIGNMENT_CONFIDENCE,
    MIN_ALIGNMENT_CONFIDENCE,
    deduplicate_words,
    filter_secondary_adlibs,
    looks_like_adlib_phrase,
    parenthetical_adlib_flags,
    polish_word_timings,
)
from backend.model_runtime import (
    decode_mono,
    load_whisper_pipeline,
    release_whisper_model,
)
from backend.qwen_alignment import (
    align_words_qwen,
    qwen_alignment_language,
    qwen_alignment_model_name,
    release_qwen_alignment_model,
)


def _check_cancelled(job: JobState) -> None:
    if job.cancelled.is_set():
        raise JobCancelled()


def _transcribe_view(
    job: JobState,
    transcriber: Any,
    waveform: np.ndarray,
    regions: list[tuple[int, int]],
    layer: str,
    progress_start: float,
    progress_span: float,
) -> list[dict[str, Any]]:
    language = (
        None
        if _normalise_language(job.language) in {"auto", "autodetect"}
        else job.language
    )
    generation: dict[str, Any] = {
        "task": "transcribe",
        "num_beams": 2 if job.quality == "accurate" else 1,
        "do_sample": False,
        "max_new_tokens": 256,
        "condition_on_prev_tokens": False,
    }
    if language:
        generation["language"] = language

    segments: list[dict[str, Any]] = []
    for index, (start_sample, end_sample) in enumerate(regions):
        _check_cancelled(job)
        audio_chunk = np.ascontiguousarray(waveform[start_sample:end_sample])
        start_time = start_sample / SAMPLE_RATE
        if (
            audio_chunk.size < 320
            or float(np.sqrt(np.mean(np.square(audio_chunk)))) < 0.0001
        ):
            continue

        job.update(
            stage="transcribing",
            progress=progress_start + (index / max(1, len(regions))) * progress_span,
            status=(
                f"Listening for ad-libs {index + 1}/{len(regions)}"
                if layer == "side"
                else f"Transcribing lyric region {index + 1}/{len(regions)}"
            ),
        )
        result = transcriber(
            audio_chunk,
            return_timestamps=True,
            return_language=True,
            generate_kwargs=generation,
            decoder_kwargs={"clean_up_tokenization_spaces": False},
        )
        chunk_duration = audio_chunk.size / SAMPLE_RATE
        timestamped_chunks = result.get("chunks")
        if not isinstance(timestamped_chunks, list) or not timestamped_chunks:
            timestamped_chunks = [
                {
                    "text": result.get("text", ""),
                    "timestamp": (0.0, chunk_duration),
                    "language": result.get("language"),
                }
            ]

        for chunk in timestamped_chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_text = str(chunk.get("text", ""))
            pieces = [
                piece
                for piece in chunk_text.split()
                if any(character.isalnum() for character in piece)
            ]
            if not pieces:
                continue

            timestamp = chunk.get("timestamp")
            if isinstance(timestamp, (tuple, list)) and len(timestamp) == 2:
                segment_start = float(timestamp[0] or 0.0)
                segment_end = float(
                    timestamp[1] if timestamp[1] is not None else chunk_duration
                )
            else:
                segment_start, segment_end = 0.0, chunk_duration
            segment_start = max(0.0, min(chunk_duration, segment_start))
            segment_end = max(segment_start + 0.04, min(chunk_duration, segment_end))

            crop_start = max(0.0, segment_start - 0.3)
            crop_end = min(chunk_duration, segment_end + 0.3)
            first_sample = int(crop_start * SAMPLE_RATE)
            last_sample = max(first_sample + 1, int(crop_end * SAMPLE_RATE))
            cropped_audio = np.ascontiguousarray(
                audio_chunk[first_sample:last_sample]
            ).copy()
            local_start = segment_start - crop_start
            spoken_duration = max(0.04, segment_end - segment_start)
            phrase_is_adlib = layer == "side" or looks_like_adlib_phrase(
                chunk_text,
                pieces,
            )
            parenthetical_flags = parenthetical_adlib_flags(pieces)
            fallback_timings = _proportional_word_timings(
                pieces,
                local_start,
                spoken_duration,
            )
            raw_words = [
                {
                    "text": piece,
                    "start": fallback_timings[word_index][0],
                    "end": fallback_timings[word_index][1],
                    "_kind": (
                        "adlib"
                        if phrase_is_adlib or parenthetical_flags[word_index]
                        else "lead"
                    ),
                    "_explicit_adlib": bool(
                        phrase_is_adlib or parenthetical_flags[word_index]
                    ),
                }
                for word_index, piece in enumerate(pieces)
            ]
            segments.append(
                {
                    "start_time": start_time + crop_start,
                    "audio": cropped_audio,
                    "words": raw_words,
                    "kind": "adlib" if phrase_is_adlib else "lead",
                    "source_layer": layer,
                    "explicit_adlib": phrase_is_adlib,
                    "language": chunk.get("language") or result.get("language"),
                }
            )
    return segments


def _align_segment_words(
    job: JobState,
    segment: dict[str, Any],
    raw_words: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    if alignment_backend(job.quality) == "qwen3":
        try:
            aligned = align_words_qwen(
                job,
                segment["audio"],
                raw_words,
                segment.get("language"),
            )
            if aligned:
                job.update(alignment_model=qwen_alignment_model_name())
                return aligned, 1.0
        except Exception as qwen_error:
            print(
                f"[align:{job.id[:8]}] Qwen alignment unavailable: {qwen_error}",
                flush=True,
            )
        if segment["source_layer"] == "side":
            return [], 0.0
        if not _segment_uses_english_alignment(
            job.language, segment.get("language")
        ):
            return raw_words, 0.0

    try:
        aligned = _align_words_ctc(job, segment["audio"], raw_words)
        if not aligned:
            fallback = [] if segment["source_layer"] == "side" else raw_words
            print(
                f"[align:{job.id[:8]}] no CTC path; "
                f"{'dropping side phrase' if not fallback else 'using phrase timing'}",
                flush=True,
            )
            return fallback, 0.0

        confidence = _mean_alignment_confidence(aligned)
        coverage = _alignment_coverage(aligned, raw_words)
        aligned_text = " ".join(str(word["text"]) for word in aligned)
        looks_hallucinated = any(
            re.search(pattern, aligned_text, flags=re.IGNORECASE)
            for pattern in COMMON_HALLUCINATION_PATTERNS
        )
        minimum_confidence = (
            MIN_ADLIB_ALIGNMENT_CONFIDENCE
            if segment["source_layer"] == "side"
            else MIN_ALIGNMENT_CONFIDENCE
        )
        accepted = (
            confidence >= minimum_confidence
            and coverage >= 0.80
            and not looks_hallucinated
        )
        print(
            f"[align:{job.id[:8]}] "
            f"model={alignment_model_name()} "
            f"language={segment.get('language') or 'unknown'} "
            f"confidence={confidence:.3f} coverage={coverage:.2%} "
            f"accepted={accepted} text={aligned_text}",
            flush=True,
        )
        if accepted:
            return aligned, confidence
        if looks_hallucinated or segment["source_layer"] == "side":
            return [], confidence
        # Low CTC confidence is common for stylised singing. Preserve Whisper's
        # phrase and approximate timing rather than silently dropping lead lyrics.
        return raw_words, confidence
    except Exception as alignment_error:
        print(
            f"[align:{job.id[:8]}] alignment unavailable; "
            f"using phrase timing: {alignment_error}",
            flush=True,
        )
        return raw_words, 0.0


def transcribe_vocals(job: JobState, vocal_path: Path) -> list[dict[str, Any]]:
    job.update(
        stage="transcribing",
        progress=59.0,
        status="Preparing the isolated vocal",
    )
    waveform = decode_mono(vocal_path)
    if waveform.size == 0:
        raise RuntimeError("The isolated vocal stem was empty.")

    transcriber = load_whisper_pipeline(job)
    _check_cancelled(job)

    use_side_pass = side_pass_enabled(job.quality)
    regions = _adaptive_vocal_regions(waveform)
    if not regions:
        release_whisper_model()
        return []

    try:
        segments = _transcribe_view(
            job,
            transcriber,
            waveform,
            regions,
            "center",
            62.0,
            16.0 if use_side_pass else 25.0,
        )

        if use_side_pass:
            _check_cancelled(job)
            try:
                side_waveform = decode_mono(vocal_path, "side")
                if side_waveform.size:
                    side_regions = _adaptive_vocal_regions(side_waveform)
                    if side_regions:
                        segments.extend(
                            _transcribe_view(
                                job,
                                transcriber,
                                side_waveform,
                                side_regions,
                                "side",
                                78.0,
                                9.0,
                            )
                        )
            except subprocess.CalledProcessError as side_error:
                print(
                    f"[adlibs:{job.id[:8]}] stereo side pass skipped: {side_error}",
                    flush=True,
                )
    finally:
        del waveform
        del transcriber
        release_whisper_model()

    words: list[dict[str, Any]] = []
    try:
        for index, segment in enumerate(segments):
            _check_cancelled(job)
            raw_words = list(segment["words"])
            use_english_alignment = _segment_uses_english_alignment(
                job.language,
                segment.get("language"),
            )
            use_qwen_alignment = (
                alignment_backend(job.quality) == "qwen3"
                and qwen_alignment_language(
                    job.language, segment.get("language")
                )
                is not None
            )
            needs_alignment = use_qwen_alignment or use_english_alignment
            local_words = raw_words if not needs_alignment else []
            segment_confidence = 1.0 if not needs_alignment else 0.0

            if needs_alignment and raw_words:
                job.update(
                    progress=88.0 + (index / max(1, len(segments))) * 9.0,
                    status=f"Aligning phrase {index + 1}/{len(segments)}",
                )
                local_words, segment_confidence = _align_segment_words(
                    job,
                    segment,
                    raw_words,
                )

            start_time = float(segment["start_time"])
            for word in local_words:
                word_start = start_time + float(word["start"])
                word_end = start_time + float(word["end"])
                if word_end - word_start > 4.0:
                    print(
                        f"[align:{job.id[:8]}] dropped overlong word "
                        f"{word['text']} ({word_end - word_start:.2f}s)",
                        flush=True,
                    )
                    continue
                words.append(
                    {
                        "text": str(word["text"]),
                        "start": round(max(0.0, word_start), 3),
                        "end": round(max(word_start + 0.04, word_end), 3),
                        "_timing": [
                            {
                                "start": round(
                                    max(0.0, start_time + float(unit["start"])),
                                    3,
                                ),
                                "end": round(
                                    max(
                                        start_time + float(unit["start"]) + 0.008,
                                        start_time + float(unit["end"]),
                                    ),
                                    3,
                                ),
                                "fill": round(float(unit["fill"]), 4),
                                "pause_before": bool(
                                    unit.get("pause_before", False)
                                ),
                            }
                            for unit in word.get("_timing", [])
                        ],
                        "_segment": index,
                        "_segment_confidence": segment_confidence,
                        "_word_confidence": float(
                            word.get("_confidence", segment_confidence)
                        ),
                        "_kind": str(word.get("_kind", segment["kind"])),
                        "_source_layer": str(segment["source_layer"]),
                        "_explicit_adlib": bool(
                            word.get(
                                "_explicit_adlib",
                                segment["explicit_adlib"],
                            )
                        ),
                    }
                )
    finally:
        release_alignment_model()
        release_qwen_alignment_model()

    job.update(progress=98.0, status="Polishing word timings")
    lead_words = [word for word in words if str(word.get("_kind")) == "lead"]
    adlib_words = [word for word in words if str(word.get("_kind")) == "adlib"]
    adlib_words = filter_secondary_adlibs(adlib_words, lead_words)
    polished = polish_word_timings(lead_words) + polish_word_timings(adlib_words)
    return deduplicate_words(polished)
