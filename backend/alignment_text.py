from __future__ import annotations

import re
from typing import Any

import numpy as np


_SMALL_CARDINALS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS_CARDINALS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_SMALL_ORDINALS = {
    0: "zeroth",
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
}
_TENS_ORDINALS = {
    20: "twentieth",
    30: "thirtieth",
    40: "fortieth",
    50: "fiftieth",
    60: "sixtieth",
    70: "seventieth",
    80: "eightieth",
    90: "ninetieth",
}


def integer_to_english(value: int) -> str:
    if value < 0:
        return f"minus {integer_to_english(-value)}"
    if value < 20:
        return _SMALL_CARDINALS[value]
    if value < 100:
        tens, remainder = divmod(value, 10)
        return _TENS_CARDINALS[tens] + (
            f" {_SMALL_CARDINALS[remainder]}" if remainder else ""
        )
    if value < 1_000:
        hundreds, remainder = divmod(value, 100)
        return f"{_SMALL_CARDINALS[hundreds]} hundred" + (
            f" {integer_to_english(remainder)}" if remainder else ""
        )
    for scale, label in (
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
    ):
        if value >= scale:
            leading, remainder = divmod(value, scale)
            return f"{integer_to_english(leading)} {label}" + (
                f" {integer_to_english(remainder)}" if remainder else ""
            )
    return " ".join(_SMALL_CARDINALS[int(digit)] for digit in str(value))


def ordinal_to_english(value: int) -> str:
    if value in _SMALL_ORDINALS:
        return _SMALL_ORDINALS[value]
    if value in _TENS_ORDINALS:
        return _TENS_ORDINALS[value]
    if value < 100:
        tens, remainder = divmod(value, 10)
        return f"{_TENS_CARDINALS[tens]} {_SMALL_ORDINALS[remainder]}"
    for scale, label in (
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
        (100, "hundred"),
    ):
        if value >= scale:
            leading, remainder = divmod(value, scale)
            if remainder:
                return f"{integer_to_english(leading)} {label} {ordinal_to_english(remainder)}"
            return f"{integer_to_english(leading)} {label}th"
    return integer_to_english(value)


def alignment_spoken_form(text: str) -> str:
    """Expand written numbers for CTC without changing displayed lyric text."""

    expanded = str(text)

    def decimal_words(raw: str) -> str:
        clean = raw.replace(",", "")
        if "." in clean:
            whole, fraction = clean.split(".", 1)
            leading = integer_to_english(int(whole or "0"))
            trailing = " ".join(_SMALL_CARDINALS[int(digit)] for digit in fraction)
            return f"{leading} point {trailing}"
        return integer_to_english(int(clean))

    currency_names = {"$": "dollars", "£": "pounds", "€": "euros"}
    expanded = re.sub(
        r"([$£€])\s*(\d[\d,]*(?:\.\d+)?)",
        lambda match: f"{decimal_words(match.group(2))} {currency_names[match.group(1)]}",
        expanded,
    )
    expanded = re.sub(
        r"(\d[\d,]*(?:\.\d+)?)\s*%",
        lambda match: f"{decimal_words(match.group(1))} percent",
        expanded,
    )
    expanded = re.sub(
        r"\b(\d[\d,]*)(?:st|nd|rd|th)\b",
        lambda match: ordinal_to_english(int(match.group(1).replace(",", ""))),
        expanded,
        flags=re.IGNORECASE,
    )
    expanded = re.sub(
        r"\d[\d,]*(?:\.\d+)?",
        lambda match: decimal_words(match.group(0)),
        expanded,
    )
    return re.sub(r"[^A-Z]+", "", expanded.upper())


def ctc_gap_is_silent(
    audio: np.ndarray,
    gap_start: float,
    gap_end: float,
    word_start: float,
    word_end: float,
    sample_rate: int = 16_000,
) -> bool:
    gap_duration = gap_end - gap_start
    if gap_duration < 0.18 or audio.size < 320:
        return False

    edge_trim = min(0.045, gap_duration * 0.18)
    inner_start = max(0, int((gap_start + edge_trim) * sample_rate))
    inner_end = min(audio.size, int((gap_end - edge_trim) * sample_rate))
    reference_start = max(0, int(word_start * sample_rate))
    reference_end = min(audio.size, int(word_end * sample_rate))
    if inner_end - inner_start < int(0.09 * sample_rate):
        return False
    if reference_end - reference_start < int(0.05 * sample_rate):
        return False

    frame_samples = max(1, int(0.025 * sample_rate))

    def frame_rms(samples: np.ndarray) -> np.ndarray:
        padded_size = int(np.ceil(samples.size / frame_samples)) * frame_samples
        padded = np.pad(samples, (0, padded_size - samples.size))
        frames = padded.reshape(-1, frame_samples)
        return np.sqrt(np.mean(np.square(frames), axis=1))

    reference_rms = frame_rms(audio[reference_start:reference_end])
    voiced_level = float(np.percentile(reference_rms, 82))
    if voiced_level < 0.001:
        return False

    gap_rms = frame_rms(audio[inner_start:inner_end])
    quiet = gap_rms <= max(0.0008, voiced_level * 0.22)
    longest_quiet_run = 0
    current_quiet_run = 0
    for is_quiet in quiet:
        current_quiet_run = current_quiet_run + 1 if bool(is_quiet) else 0
        longest_quiet_run = max(longest_quiet_run, current_quiet_run)
    quiet_seconds = longest_quiet_run * frame_samples / sample_rate
    return quiet_seconds >= 0.10 and float(np.mean(quiet)) >= 0.45


def ctc_token_spans(
    emissions: Any,
    target_ids: list[int],
    blank_id: int,
) -> list[tuple[int, int]]:
    import torch

    target = torch.tensor(target_ids, dtype=torch.long)
    extended = torch.full((target.numel() * 2 + 1,), blank_id, dtype=torch.long)
    extended[1::2] = target
    frame_count = int(emissions.shape[0])
    state_count = int(extended.numel())
    if frame_count < target.numel():
        return []

    negative = -1.0e9
    scores = torch.full((state_count,), negative)
    scores[0] = 0.0
    backpointers = torch.zeros((frame_count, state_count), dtype=torch.int8)
    state_indexes = torch.arange(state_count)
    can_skip = (state_indexes >= 2) & (extended != blank_id)
    can_skip &= extended != torch.roll(extended, 2)

    for frame in range(frame_count):
        stay = scores
        advance = torch.roll(scores, 1)
        advance[0] = negative
        skip = torch.roll(scores, 2)
        skip[:2] = negative
        skip = torch.where(can_skip, skip, negative)
        candidates = torch.stack((stay, advance, skip), dim=0)
        best_scores, moves = candidates.max(dim=0)
        scores = best_scores + emissions[frame, extended]
        backpointers[frame] = moves.to(torch.int8)

    state = state_count - 1
    if state_count > 1 and scores[state - 1] > scores[state]:
        state -= 1
    path = torch.empty((frame_count,), dtype=torch.long)
    for frame in range(frame_count - 1, -1, -1):
        path[frame] = state
        state -= int(backpointers[frame, state])
        if state < 0:
            state = 0

    spans: list[tuple[int, int]] = []
    for token_index in range(len(target_ids)):
        token_state = token_index * 2 + 1
        frames = torch.nonzero(path == token_state, as_tuple=False).flatten()
        if frames.numel() == 0:
            return []
        spans.append((int(frames[0]), int(frames[-1]) + 1))
    return spans

