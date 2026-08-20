from __future__ import annotations

import re
import subprocess
from typing import Any

import numpy as np

from backend.ctc_alignment import (
    _align_words_ctc,
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


def _transcribe_view(
    core: Any,
    job: Any,
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
        # The previous 160-token cap could truncate dense rap or fast sung
        # passages inside an otherwise valid 29-second region.
        "max_new_tokens": 256,
        "condition_on_prev_tokens": False,
    }
    if language:
        generation["language"] = language

    segments: list[dict[str, Any]] = []
    for index, (start_sample, end_sample) in enumerate(regions):
        core._check_cancelled(job)
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
            phrase_is_adlib = layer == "side" or core._looks_like_adlib_phrase(
                chunk_text,
                pieces,
            )
            parenthetical_flags = core._parenthetical_adlib_flags(pieces)
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
    core: Any,
    job: Any,
    segment: dict[str, Any],
    raw_words: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    try:
        aligned = _align_words_ctc(
            core,
            job,
            segment["audio"],
            raw_words,
        )
        if not aligned:
            print(
                f"[align:{job.id[:8]}] no CTC path; using phrase timing",
                flush=True,
            )
            return raw_words, 0.0

        confidence = _mean_alignment_confidence(core, aligned)
        aligned_text = " ".join(str(word["text"]) for word in aligned)
        looks_hallucinated = any(
            re.search(pattern, aligned_text, flags=re.IGNORECASE)
            for pattern in core.COMMON_HALLUCINATION_PATTERNS
        )
        minimum_confidence = (
            core.MIN_ADLIB_ALIGNMENT_CONFIDENCE
            if segment["source_layer"] == "side"
            else core.MIN_ALIGNMENT_CONFIDENCE
        )
        accepted = confidence >= minimum_confidence and not looks_hallucinated
        print(
            f"[align:{job.id[:8]}] "
            f"model={alignment_model_name()} "
            f"language={segment.get('language') or 'unknown'} "
            f"confidence={confidence:.3f} "
            f"accepted={accepted} text={aligned_text}",
            flush=True,
        )
        return (aligned if accepted else []), confidence
    except Exception as alignment_error:
        # A model/cache problem should not erase a transcription that Whisper
        # already produced. Low-confidence acoustic alignment, by contrast, is
        # rejected above as a likely hallucination.
        print(
            f"[align:{job.id[:8]}] alignment unavailable; "
            f"using phrase timing: {alignment_error}",
            flush=True,
        )
        return raw_words, 0.0


def transcribe_vocals(job: Any, vocal_path: Any) -> list[dict[str, Any]]:
    # Import lazily so the pure segmentation/timing helpers remain lightweight
    # and can be tested without importing FastAPI or Transformers.
    from backend import server as core

    job.update(
        stage="transcribing",
        progress=59.0,
        status="Preparing the isolated vocal",
    )
    waveform = core._decode_mono(vocal_path)
    if waveform.size == 0:
        raise RuntimeError("The isolated vocal stem was empty.")

    transcriber = core._load_whisper_pipeline(job)
    core._check_cancelled(job)

    use_side_pass = job.quality == "accurate"
    regions = _adaptive_vocal_regions(waveform)
    if not regions:
        regions = core._rms_vocal_regions(waveform)
    segments = _transcribe_view(
        core,
        job,
        transcriber,
        waveform,
        regions,
        "center",
        62.0,
        16.0 if use_side_pass else 25.0,
    )

    if use_side_pass:
        core._check_cancelled(job)
        try:
            side_waveform = core._decode_mono(vocal_path, "side")
            if side_waveform.size:
                # The side channel has its own activity map. Reusing lead-vocal
                # regions misses responses that happen during a lead pause.
                side_regions = _adaptive_vocal_regions(side_waveform)
                if side_regions:
                    segments.extend(
                        _transcribe_view(
                            core,
                            job,
                            transcriber,
                            side_waveform,
                            side_regions,
                            "side",
                            78.0,
                            9.0,
                        )
                    )
            del side_waveform
        except subprocess.CalledProcessError as side_error:
            print(
                f"[adlibs:{job.id[:8]}] stereo side pass skipped: {side_error}",
                flush=True,
            )

    del waveform
    del transcriber
    core._release_whisper_model()

    words: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        core._check_cancelled(job)
        raw_words = list(segment["words"])
        use_english_alignment = _segment_uses_english_alignment(
            job.language,
            segment.get("language"),
        )
        local_words = raw_words if not use_english_alignment else []
        segment_confidence = 1.0 if not use_english_alignment else 0.0

        if use_english_alignment and raw_words:
            job.update(
                progress=88.0 + (index / max(1, len(segments))) * 9.0,
                status=f"Tightening phrase {index + 1}/{len(segments)}",
            )
            local_words, segment_confidence = _align_segment_words(
                core,
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

    release_alignment_model()
    job.update(progress=98.0, status="Polishing word timings")
    lead_words = [
        word for word in words if str(word.get("_kind")) == "lead"
    ]
    adlib_words = [
        word for word in words if str(word.get("_kind")) == "adlib"
    ]
    adlib_words = core._filter_secondary_adlibs(adlib_words, lead_words)
    polished = core._polish_word_timings(
        lead_words
    ) + core._polish_word_timings(adlib_words)
    return core._deduplicate_words(polished)
