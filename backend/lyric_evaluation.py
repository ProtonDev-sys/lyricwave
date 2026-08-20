from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


_WORD_CHARACTERS = re.compile(r"[^\w']+", flags=re.UNICODE)


@dataclass(frozen=True)
class WordToken:
    text: str
    normalized: str
    start: float | None
    end: float | None
    kind: str


@dataclass(frozen=True)
class AlignmentStep:
    operation: str
    reference: WordToken | None
    candidate: WordToken | None


@dataclass(frozen=True)
class ErrorCounts:
    matches: int
    substitutions: int
    deletions: int
    insertions: int


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.casefold().replace("’", "'").replace("_", "")
    return _WORD_CHARACTERS.sub("", text)


def _optional_timestamp(value: object) -> float | None:
    if value is None or value == "":
        return None
    timestamp = float(value)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError(f"Invalid word timestamp: {value!r}")
    return timestamp


def _word_from_mapping(
    value: dict[str, Any],
    *,
    inherited_kind: str = "lead",
) -> WordToken | None:
    text = str(value.get("text", "")).strip()
    normalized = normalize_text(text)
    if not normalized:
        return None
    start = _optional_timestamp(value.get("start"))
    end = _optional_timestamp(value.get("end"))
    if (start is None) != (end is None):
        raise ValueError(f"Word {text!r} must provide both start and end timestamps.")
    if start is not None and end is not None and end < start:
        raise ValueError(f"Word {text!r} ends before it starts.")
    kind = str(value.get("kind") or inherited_kind or "lead").strip().lower()
    return WordToken(text=text, normalized=normalized, start=start, end=end, kind=kind)


def _raw_words(payload: object) -> Iterable[tuple[dict[str, Any], str]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                yield {"text": item}, "lead"
            elif isinstance(item, dict):
                yield item, str(item.get("kind") or "lead")
            else:
                raise ValueError("Word arrays may contain only strings or objects.")
        return

    if not isinstance(payload, dict):
        raise ValueError("Timing JSON must be an object or a word array.")
    words = payload.get("words")
    if isinstance(words, list):
        yield from _raw_words(words)
        return
    lines = payload.get("lines")
    if not isinstance(lines, list):
        raise ValueError("Timing JSON must contain a 'words' or 'lines' array.")
    for line in lines:
        if not isinstance(line, dict):
            raise ValueError("Every lyric line must be an object.")
        line_kind = str(line.get("kind") or "lead")
        line_words = line.get("words")
        if not isinstance(line_words, list):
            raise ValueError("Every lyric line must contain a word array.")
        for item in line_words:
            if isinstance(item, str):
                yield {"text": item}, line_kind
            elif isinstance(item, dict):
                yield item, line_kind
            else:
                raise ValueError("Line word arrays may contain only strings or objects.")


def load_tokens(path: Path, *, include_adlibs: bool = False) -> list[WordToken]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read timing JSON from {path}: {error}") from error

    tokens: list[WordToken] = []
    for raw_word, inherited_kind in _raw_words(payload):
        token = _word_from_mapping(raw_word, inherited_kind=inherited_kind)
        if token is None:
            continue
        if not include_adlibs and token.kind == "adlib":
            continue
        tokens.append(token)
    return tokens


def align_words(
    reference: Sequence[WordToken],
    candidate: Sequence[WordToken],
) -> list[AlignmentStep]:
    rows = len(reference) + 1
    columns = len(candidate) + 1
    costs = [[0] * columns for _ in range(rows)]
    moves = [[""] * columns for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row
        moves[row][0] = "deletion"
    for column in range(1, columns):
        costs[0][column] = column
        moves[0][column] = "insertion"

    priority = {"match": 0, "substitution": 1, "deletion": 2, "insertion": 3}
    for row in range(1, rows):
        for column in range(1, columns):
            same = reference[row - 1].normalized == candidate[column - 1].normalized
            diagonal_operation = "match" if same else "substitution"
            options = [
                (costs[row - 1][column - 1] + (0 if same else 1), diagonal_operation),
                (costs[row - 1][column] + 1, "deletion"),
                (costs[row][column - 1] + 1, "insertion"),
            ]
            cost, operation = min(options, key=lambda item: (item[0], priority[item[1]]))
            costs[row][column] = cost
            moves[row][column] = operation

    alignment: list[AlignmentStep] = []
    row = len(reference)
    column = len(candidate)
    while row or column:
        operation = moves[row][column]
        if operation in {"match", "substitution"}:
            alignment.append(
                AlignmentStep(operation, reference[row - 1], candidate[column - 1])
            )
            row -= 1
            column -= 1
        elif operation == "deletion":
            alignment.append(AlignmentStep(operation, reference[row - 1], None))
            row -= 1
        elif operation == "insertion":
            alignment.append(AlignmentStep(operation, None, candidate[column - 1]))
            column -= 1
        else:
            raise RuntimeError("Word alignment backtrace is incomplete.")
    alignment.reverse()
    return alignment


def _edit_distance(reference: Sequence[str], candidate: Sequence[str]) -> int:
    previous = list(range(len(candidate) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, candidate_item in enumerate(candidate, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (reference_item != candidate_item),
                )
            )
        previous = current
    return previous[-1]


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _milliseconds(value: float | None) -> float | None:
    return None if value is None else round(value * 1000, 3)


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _timing_metrics(alignment: Sequence[AlignmentStep], reference_count: int) -> dict[str, Any]:
    onset_deltas: list[float] = []
    offset_deltas: list[float] = []
    duration_errors: list[float] = []
    for step in alignment:
        if step.operation != "match" or step.reference is None or step.candidate is None:
            continue
        reference = step.reference
        candidate = step.candidate
        if None in {reference.start, reference.end, candidate.start, candidate.end}:
            continue
        assert reference.start is not None and reference.end is not None
        assert candidate.start is not None and candidate.end is not None
        onset_deltas.append(candidate.start - reference.start)
        offset_deltas.append(candidate.end - reference.end)
        reference_duration = reference.end - reference.start
        candidate_duration = candidate.end - candidate.start
        duration_errors.append(abs(candidate_duration - reference_duration))

    onset_errors = [abs(value) for value in onset_deltas]
    offset_errors = [abs(value) for value in offset_deltas]
    matched = len(onset_errors)
    return {
        "matched_words": matched,
        "reference_coverage": round(matched / reference_count, 6) if reference_count else 0.0,
        "onset_bias_ms": _milliseconds(_mean(onset_deltas)),
        "onset_mae_ms": _milliseconds(_mean(onset_errors)),
        "onset_median_ms": _milliseconds(_percentile(onset_errors, 0.5)),
        "onset_p90_ms": _milliseconds(_percentile(onset_errors, 0.9)),
        "offset_bias_ms": _milliseconds(_mean(offset_deltas)),
        "offset_mae_ms": _milliseconds(_mean(offset_errors)),
        "offset_median_ms": _milliseconds(_percentile(offset_errors, 0.5)),
        "offset_p90_ms": _milliseconds(_percentile(offset_errors, 0.9)),
        "duration_mae_ms": _milliseconds(_mean(duration_errors)),
        "onset_within_50ms": round(sum(value <= 0.05 for value in onset_errors) / matched, 6)
        if matched
        else None,
        "onset_within_100ms": round(sum(value <= 0.10 for value in onset_errors) / matched, 6)
        if matched
        else None,
        "onset_within_250ms": round(sum(value <= 0.25 for value in onset_errors) / matched, 6)
        if matched
        else None,
    }


def evaluate(reference: Sequence[WordToken], candidate: Sequence[WordToken]) -> dict[str, Any]:
    if not reference:
        raise ValueError("The reference contains no evaluable lyric words.")
    alignment = align_words(reference, candidate)
    counts = ErrorCounts(
        matches=sum(step.operation == "match" for step in alignment),
        substitutions=sum(step.operation == "substitution" for step in alignment),
        deletions=sum(step.operation == "deletion" for step in alignment),
        insertions=sum(step.operation == "insertion" for step in alignment),
    )
    reference_characters = list(" ".join(token.normalized for token in reference))
    candidate_characters = list(" ".join(token.normalized for token in candidate))
    character_errors = _edit_distance(reference_characters, candidate_characters)
    word_errors = counts.substitutions + counts.deletions + counts.insertions
    return {
        "reference_words": len(reference),
        "candidate_words": len(candidate),
        **asdict(counts),
        "word_errors": word_errors,
        "wer": round(word_errors / len(reference), 6),
        "character_errors": character_errors,
        "cer": round(character_errors / max(1, len(reference_characters)), 6),
        "timing": _timing_metrics(alignment, len(reference)),
    }


def _candidate_spec(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError("Candidate labels cannot be empty.")
        return label, Path(raw_path)
    path = Path(value)
    return path.stem, path


def _format_metric(value: object, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    return f"{numeric * 100:.2f}%" if percent else f"{numeric:.1f}"


def _render_text(reference_path: Path, results: Sequence[dict[str, Any]]) -> str:
    lines = [f"Reference: {reference_path}"]
    for result in results:
        metrics = result["metrics"]
        timing = metrics["timing"]
        lines.extend(
            [
                "",
                str(result["label"]),
                f"  WER: {_format_metric(metrics['wer'], percent=True)}",
                f"  CER: {_format_metric(metrics['cer'], percent=True)}",
                (
                    "  S/D/I: "
                    f"{metrics['substitutions']}/"
                    f"{metrics['deletions']}/"
                    f"{metrics['insertions']}"
                ),
                f"  Timing coverage: {_format_metric(timing['reference_coverage'], percent=True)}",
                f"  Onset MAE / P90: {_format_metric(timing['onset_mae_ms'])} / "
                f"{_format_metric(timing['onset_p90_ms'])} ms",
                f"  Onset bias: {_format_metric(timing['onset_bias_ms'])} ms",
                f"  Offset MAE / P90: {_format_metric(timing['offset_mae_ms'])} / "
                f"{_format_metric(timing['offset_p90_ms'])} ms",
            ]
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare lyricwave word-timing exports against a reference.",
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument(
        "candidates",
        nargs="+",
        help="Candidate JSON paths, optionally labelled as name=path.json.",
    )
    parser.add_argument(
        "--include-adlibs",
        action="store_true",
        help="Include ad-lib lines. Lead lyrics are evaluated by default.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        reference = load_tokens(
            arguments.reference,
            include_adlibs=arguments.include_adlibs,
        )
        results: list[dict[str, Any]] = []
        for specification in arguments.candidates:
            label, path = _candidate_spec(specification)
            candidate = load_tokens(path, include_adlibs=arguments.include_adlibs)
            results.append(
                {
                    "label": label,
                    "path": str(path),
                    "metrics": evaluate(reference, candidate),
                }
            )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    payload = {
        "reference": str(arguments.reference),
        "include_adlibs": bool(arguments.include_adlibs),
        "results": results,
    }
    if arguments.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render_text(arguments.reference, results))
    return 0
