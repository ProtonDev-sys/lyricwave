from __future__ import annotations

import re
from typing import Any

import numpy as np

from backend.alignment_text import alignment_spoken_form, ctc_gap_is_silent, ctc_token_spans


MIN_ALIGNMENT_CONFIDENCE = 0.10
MIN_ADLIB_ALIGNMENT_CONFIDENCE = 0.17
COMMON_HALLUCINATION_PATTERNS = (
    r"\bthank(?:s| you) for watching\b",
    r"\bplease subscribe\b",
    r"\bsubtitles? by\b",
    r"\bamara\.?org\b",
)
ADLIB_WORDS = {
    "ah",
    "aha",
    "ayy",
    "ay",
    "eh",
    "ha",
    "hey",
    "hmm",
    "huh",
    "mm",
    "mmm",
    "nah",
    "oh",
    "ooh",
    "okay",
    "uh",
    "uhh",
    "whoa",
    "woah",
    "woo",
    "yeah",
    "yep",
    "yo",
}


def looks_like_adlib_phrase(text: str, pieces: list[str]) -> bool:
    stripped = text.strip()
    explicitly_parenthetical = (
        (stripped.startswith("(") and stripped.endswith(")"))
        or (stripped.startswith("[") and stripped.endswith("]"))
    )
    normalized = [re.sub(r"[^a-z]+", "", piece.lower()) for piece in pieces]
    normalized = [piece for piece in normalized if piece]
    interjections = sum(piece in ADLIB_WORDS for piece in normalized)
    return explicitly_parenthetical or bool(
        normalized
        and len(normalized) <= 8
        and interjections / len(normalized) >= 0.70
    )


def parenthetical_adlib_flags(pieces: list[str]) -> list[bool]:
    flags: list[bool] = []
    inside = False
    for piece in pieces:
        if re.match(r"^[\"']?[\(\[]", piece):
            inside = True
        flags.append(inside)
        if re.search(r"[\)\]][^\w]*$", piece):
            inside = False
    return flags


def filter_secondary_adlibs(
    adlibs: list[dict[str, Any]],
    lead_words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep short, distinct side-channel phrases and reject leaked lead vocals."""

    grouped: dict[int, list[dict[str, Any]]] = {}
    for word in adlibs:
        grouped.setdefault(int(word.get("_segment", -1)), []).append(word)

    retained: list[dict[str, Any]] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda word: float(word["start"]))
        if not ordered:
            continue
        if str(ordered[0].get("_source_layer")) != "side":
            retained.extend(ordered)
            continue
        if len(ordered) > 6:
            continue
        start = float(ordered[0]["start"]) - 0.45
        end = max(float(word["end"]) for word in ordered) + 0.45
        nearby_lead = [
            word
            for word in lead_words
            if float(word["end"]) >= start and float(word["start"]) <= end
        ]
        side_tokens = [
            re.sub(r"[^\w']+", "", str(word["text"]).lower()) for word in ordered
        ]
        lead_tokens = {
            re.sub(r"[^\w']+", "", str(word["text"]).lower()) for word in nearby_lead
        }
        side_tokens = [token for token in side_tokens if token]
        duplicate_ratio = (
            sum(token in lead_tokens for token in side_tokens) / len(side_tokens)
            if side_tokens
            else 1.0
        )
        if duplicate_ratio < 0.65:
            retained.extend(ordered)
    return retained


def polish_word_timings(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose audible onsets and remove impossible same-layer overlaps."""

    grouped: dict[int, list[dict[str, Any]]] = {}
    for original in words:
        grouped.setdefault(int(original.get("_segment", -1)), []).append(dict(original))
    phrases = [
        sorted(group, key=lambda word: (float(word["start"]), float(word["end"])))
        for group in grouped.values()
    ]
    phrases.sort(
        key=lambda group: (
            float(group[0]["start"]),
            int(group[0].get("_segment", -1)),
        )
    )

    for phrase in phrases:
        for word in phrase:
            original_start = float(word["start"])
            word["start"] = round(max(0.0, original_start - 0.055), 3)
            word["end"] = round(
                max(float(word["start"]) + 0.045, float(word["end"]) + 0.018),
                3,
            )
        for index in range(1, len(phrase)):
            previous = phrase[index - 1]
            current = phrase[index]
            if float(current["start"]) >= float(previous["end"]):
                continue
            boundary = (float(previous["end"]) + float(current["start"])) / 2
            boundary = max(float(previous["start"]) + 0.04, boundary)
            boundary = min(float(current["end"]) - 0.04, boundary)
            previous["end"] = round(boundary, 3)
            current["start"] = round(boundary + 0.006, 3)

    if phrases and str(phrases[0][0].get("_kind")) != "adlib":
        for index in range(1, len(phrases)):
            previous = phrases[index - 1][-1]
            current = phrases[index][0]
            if float(current["start"]) >= float(previous["end"]):
                continue
            boundary = (float(previous["end"]) + float(current["start"])) / 2
            boundary = max(float(previous["start"]) + 0.04, boundary)
            boundary = min(float(current["end"]) - 0.04, boundary)
            previous["end"] = round(boundary, 3)
            current["start"] = round(boundary + 0.006, 3)
    return [word for phrase in phrases for word in phrase]


def deduplicate_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        words,
        key=lambda word: (
            float(word["start"]),
            0 if str(word.get("_kind", "lead")) == "lead" else 1,
            float(word["end"]),
        ),
    )
    result: list[dict[str, Any]] = []
    for word in ordered:
        normalized = re.sub(r"[^\w']+", "", str(word["text"]).lower())
        if not normalized:
            continue
        duplicate: dict[str, Any] | None = None
        for previous in reversed(result[-12:]):
            previous_normalized = re.sub(
                r"[^\w']+", "", str(previous["text"]).lower()
            )
            same_kind = str(word.get("_kind", "lead")) == str(
                previous.get("_kind", "lead")
            )
            current_phrase = int(word.get("_segment", -1))
            previous_phrase = int(previous.get("_segment", -1))
            different_phrase = (
                current_phrase < 0
                or previous_phrase < 0
                or current_phrase != previous_phrase
            )
            start_distance = abs(float(word["start"]) - float(previous["start"]))
            overlaps = min(float(word["end"]), float(previous["end"])) > max(
                float(word["start"]), float(previous["start"])
            )
            if (
                normalized == previous_normalized
                and same_kind
                and different_phrase
                and start_distance < 0.7
                and (overlaps or start_distance < 0.16)
            ):
                duplicate = previous
                break
        if duplicate is not None:
            if float(word["end"]) > float(duplicate["end"]):
                duplicate["end"] = word["end"]
            continue
        result.append(word)

    public_words: list[dict[str, Any]] = []
    for word in result:
        public_word = {
            "text": word["text"],
            "start": word["start"],
            "end": word["end"],
            "kind": str(word.get("_kind", "lead")),
            "phrase": int(word.get("_segment", -1)),
        }
        word_start = float(word["start"])
        word_end = float(word["end"])
        previous_fill = 0.0
        timing: list[dict[str, Any]] = []
        for unit in word.get("_timing", []):
            unit_start = max(word_start, float(unit["start"]))
            unit_end = min(word_end, float(unit["end"]))
            unit_fill = max(previous_fill, min(1.0, float(unit["fill"])))
            if unit_end <= unit_start or unit_fill <= previous_fill:
                continue
            public_unit: dict[str, Any] = {
                "start": round(unit_start, 3),
                "end": round(unit_end, 3),
                "fill": round(unit_fill, 4),
            }
            if bool(unit.get("pause_before", False)):
                public_unit["pause_before"] = True
            timing.append(public_unit)
            previous_fill = unit_fill
        if timing:
            timing[-1]["fill"] = 1.0
            public_word["timing"] = timing
        public_words.append(public_word)
    return public_words


def resolve_overlapping_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for word in words:
        grouped.setdefault(int(word.get("_segment", -1)), []).append(word)
    ranked = sorted(
        grouped.values(),
        key=lambda group: (
            -float(group[0].get("_segment_confidence", 0.0)),
            float(min(word["start"] for word in group)),
        ),
    )
    accepted: list[dict[str, Any]] = []
    claimed_intervals: list[tuple[float, float]] = []
    for group in ranked:
        ordered = sorted(group, key=lambda word: (float(word["start"]), float(word["end"])))
        retained: list[dict[str, Any]] = []
        for original in ordered:
            word = dict(original)
            rejected = False
            for start, end in claimed_intervals:
                word_start = float(word["start"])
                word_end = float(word["end"])
                midpoint = (word_start + word_end) / 2
                if start - 0.025 <= midpoint <= end + 0.025:
                    rejected = True
                    break
                if word_start < end < word_end:
                    word["start"] = round(end + 0.008, 3)
                if word_start < start < word_end:
                    word["end"] = round(start - 0.008, 3)
                if float(word["end"]) - float(word["start"]) < 0.04:
                    rejected = True
                    break
            if not rejected:
                retained.append(word)
        if not retained:
            continue
        accepted.extend(retained)
        claimed_intervals.append(
            (
                float(retained[0]["start"]),
                max(float(word["end"]) for word in retained),
            )
        )
    return sorted(accepted, key=lambda word: (float(word["start"]), float(word["end"])))


def _group_line_stream(
    words: list[dict[str, Any]],
    kind: str,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    max_words = 7 if kind == "adlib" else 12
    max_duration = 5.2 if kind == "adlib" else 7.2
    maximum_gap = 0.90 if kind == "adlib" else 0.52
    for word in sorted(words, key=lambda item: (float(item["start"]), float(item["end"]))):
        previous = current[-1] if current else None
        gap = float(word["start"]) - float(previous["end"]) if previous else 0.0
        line_duration = float(word["end"]) - float(current[0]["start"]) if current else 0.0
        phrase_changed = bool(
            previous and int(word.get("phrase", -1)) != int(previous.get("phrase", -1))
        )
        punctuation_break = bool(
            previous and re.search(r"[.!?…]$", str(previous["text"])) and len(current) >= 3
        )
        phrase_break = bool(
            phrase_changed and (gap > 0.06 or len(current) >= 4 or line_duration >= 2.5)
        )
        if current and (
            gap > maximum_gap
            or len(current) >= max_words
            or line_duration > max_duration
            or punctuation_break
            or phrase_break
        ):
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups


def group_words_into_lines(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not words:
        return []

    groups: list[tuple[str, list[dict[str, Any]]]] = []
    for kind in ("lead", "adlib"):
        stream_words = [word for word in words if str(word.get("kind", "lead")) == kind]
        groups.extend((kind, group) for group in _group_line_stream(stream_words, kind))
    groups.sort(
        key=lambda item: (
            float(item[1][0]["start"]),
            0 if item[0] == "lead" else 1,
        )
    )

    return [
        {
            "id": f"line-{index}-{kind}-{line[0]['start']}",
            "start": line[0]["start"],
            "end": line[-1]["end"],
            "words": line,
            "kind": kind,
        }
        for index, (kind, line) in enumerate(groups)
    ]
