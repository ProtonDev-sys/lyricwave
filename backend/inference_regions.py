from __future__ import annotations

import re

import numpy as np


SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.05
_ENGLISH_LANGUAGE_NAMES = {"en", "eng", "english"}


def _normalise_language(value: object) -> str:
    return re.sub(r"[^a-z]+", "", str(value or "").strip().lower())


def _segment_uses_english_alignment(
    requested_language: str,
    detected_language: object,
) -> bool:
    requested = _normalise_language(requested_language)
    if requested in _ENGLISH_LANGUAGE_NAMES:
        return True
    if requested not in {"", "auto", "autodetect"}:
        return False
    return _normalise_language(detected_language) in _ENGLISH_LANGUAGE_NAMES


def _word_weight(text: str) -> float:
    letters = re.sub(r"[^\w]+", "", text, flags=re.UNICODE).replace("_", "")
    return max(1.0, float(len(letters)) ** 0.72)


def _proportional_word_timings(
    pieces: list[str],
    start: float,
    duration: float,
) -> list[tuple[float, float]]:
    if not pieces:
        return []
    safe_duration = max(0.04, float(duration))
    weights = np.asarray([_word_weight(piece) for piece in pieces], dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(weights)))
    cumulative /= cumulative[-1]
    return [
        (
            round(start + safe_duration * float(cumulative[index]), 3),
            round(start + safe_duration * float(cumulative[index + 1]), 3),
        )
        for index in range(len(pieces))
    ]


def _fill_short_gaps(active: np.ndarray, maximum_gap_frames: int) -> None:
    index = 0
    while index < active.size:
        if active[index]:
            index += 1
            continue
        start = index
        while index < active.size and not active[index]:
            index += 1
        if start > 0 and index < active.size and index - start <= maximum_gap_frames:
            active[start:index] = True


def _remove_short_runs(active: np.ndarray, minimum_run_frames: int) -> None:
    index = 0
    while index < active.size:
        if not active[index]:
            index += 1
            continue
        start = index
        while index < active.size and active[index]:
            index += 1
        if index - start < minimum_run_frames:
            active[start:index] = False


def _adaptive_vocal_regions(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> list[tuple[int, int]]:
    """Create Whisper-sized regions using local rather than global vocal energy."""

    if waveform.size < max(3, int(0.15 * sample_rate)):
        return []

    hop_samples = max(1, int(FRAME_SECONDS * sample_rate))
    padded_size = int(np.ceil(waveform.size / hop_samples)) * hop_samples
    padded = np.pad(waveform, (0, padded_size - waveform.size))
    frames = padded.reshape(-1, hop_samples)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    peak = float(rms.max(initial=0.0))
    if peak < 0.0005:
        return []

    noise_floor = float(np.percentile(rms, 18))
    context_frames = max(1, int(3.0 / FRAME_SECONDS))
    trailing_windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(rms, (context_frames, 0), mode="edge"),
        context_frames + 1,
    )
    leading_windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(rms, (0, context_frames), mode="edge"),
        context_frames + 1,
    )
    local_reference = np.minimum(
        np.percentile(trailing_windows, 85, axis=1),
        np.percentile(leading_windows, 85, axis=1),
    )

    on_threshold = np.maximum(
        0.00025,
        noise_floor + np.maximum(0.0, local_reference - noise_floor) * 0.20,
    )
    off_threshold = np.maximum(
        0.00018,
        noise_floor + np.maximum(0.0, local_reference - noise_floor) * 0.075,
    )

    active = np.zeros(rms.size, dtype=bool)
    is_active = False
    for index, level in enumerate(rms):
        threshold = off_threshold[index] if is_active else on_threshold[index]
        is_active = bool(level >= threshold)
        active[index] = is_active

    _fill_short_gaps(active, max(1, int(0.95 / FRAME_SECONDS)))
    _remove_short_runs(active, max(1, int(0.12 / FRAME_SECONDS)))

    padding_samples = int(0.28 * sample_rate)
    runs: list[tuple[int, int]] = []
    index = 0
    while index < active.size:
        if not active[index]:
            index += 1
            continue
        run_start = index
        while index < active.size and active[index]:
            index += 1
        runs.append(
            (
                max(0, run_start * hop_samples - padding_samples),
                min(waveform.size, index * hop_samples + padding_samples),
            )
        )

    if not runs:
        return []

    consolidated: list[tuple[int, int]] = []
    for start, end in runs:
        if consolidated and start <= consolidated[-1][1]:
            consolidated[-1] = (consolidated[-1][0], max(consolidated[-1][1], end))
        else:
            consolidated.append((start, end))

    maximum_samples = int(28.5 * sample_rate)
    preferred_minimum = int(16.0 * sample_rate)
    maximum_merge_gap = int(6.0 * sample_rate)
    merged: list[tuple[int, int]] = []
    current_start, current_end = consolidated[0]
    for run_start, run_end in consolidated[1:]:
        if (
            run_end - current_start <= maximum_samples
            and run_start - current_end <= maximum_merge_gap
        ):
            current_end = run_end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = run_start, run_end
    merged.append((current_start, current_end))

    regions: list[tuple[int, int]] = []
    for region_start, region_end in merged:
        while region_end - region_start > maximum_samples:
            search_start = region_start + preferred_minimum
            search_end = region_start + maximum_samples
            first_frame = max(0, search_start // hop_samples)
            last_frame = min(rms.size, max(first_frame + 1, search_end // hop_samples))
            quiet_frame = first_frame + int(np.argmin(rms[first_frame:last_frame]))
            cut = max(region_start + preferred_minimum, quiet_frame * hop_samples)
            cut = min(region_end - int(0.2 * sample_rate), cut)
            regions.append((region_start, cut))
            region_start = cut
        if region_end - region_start >= int(0.2 * sample_rate):
            regions.append((region_start, region_end))
    return regions
