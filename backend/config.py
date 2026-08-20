from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
JOB_ROOT: Final = PROJECT_ROOT / ".local-data" / "jobs"

MAX_FILE_SIZE: Final = 500 * 1024 * 1024
MAX_DURATION_SECONDS: Final = 3 * 60 * 60
ALLOWED_EXTENSIONS: Final = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".webm",
}
TERMINAL_STAGES: Final = {"complete", "error", "cancelled"}
DEFAULT_QUALITY: Final = "balanced"

_DEFAULT_QWEN_ASR_MODELS: Final = {
    "fast": "Qwen/Qwen3-ASR-0.6B-hf",
    "balanced": "Qwen/Qwen3-ASR-0.6B-hf",
    "accurate": "Qwen/Qwen3-ASR-1.7B-hf",
}
_DEFAULT_CTC_ALIGNER_MODELS: Final = {
    "fast": "facebook/wav2vec2-base-960h",
    "balanced": "facebook/wav2vec2-base-960h",
    "accurate": "facebook/wav2vec2-large-960h-lv60-self",
}
_DEFAULT_QWEN_ALIGNER_MODEL: Final = "Qwen/Qwen3-ForcedAligner-0.6B-hf"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    label: str
    description: str
    demucs: str
    demucs_passes: int
    asr_backend: str
    asr_model: str
    fallback_asr_model: str
    aligner_backend: str
    aligner_model: str
    side_pass: bool


PRESETS: Final[dict[str, ModelProfile]] = {
    "fast": ModelProfile(
        label="Fast",
        description="Lowest VRAM · Whisper large-v3-turbo · English word alignment",
        demucs="htdemucs",
        demucs_passes=1,
        asr_backend="whisper",
        asr_model="openai/whisper-large-v3-turbo",
        fallback_asr_model="openai/whisper-large-v3-turbo",
        aligner_backend="ctc",
        aligner_model="facebook/wav2vec2-base-960h",
        side_pass=False,
    ),
    "balanced": ModelProfile(
        label="Recommended",
        description="Qwen3-ASR 0.6B · multilingual word alignment · standard separation",
        demucs="htdemucs",
        demucs_passes=1,
        asr_backend="qwen3",
        asr_model="Qwen/Qwen3-ASR-0.6B-hf",
        fallback_asr_model="openai/whisper-large-v3-turbo",
        aligner_backend="qwen3",
        aligner_model="Qwen/Qwen3-ForcedAligner-0.6B-hf",
        side_pass=False,
    ),
    "accurate": ModelProfile(
        label="Best quality",
        description="Qwen3-ASR 1.7B · fine-tuned separation · background-vocal pass",
        demucs="htdemucs_ft",
        demucs_passes=4,
        asr_backend="qwen3",
        asr_model="Qwen/Qwen3-ASR-1.7B-hf",
        fallback_asr_model="openai/whisper-large-v3",
        aligner_backend="qwen3",
        aligner_model="Qwen/Qwen3-ForcedAligner-0.6B-hf",
        side_pass=True,
    ),
}

_LANGUAGE_ALIASES: Final = {
    "auto": "auto",
    "autodetect": "auto",
    "en": "english",
    "eng": "english",
    "english": "english",
    "es": "spanish",
    "spa": "spanish",
    "spanish": "spanish",
    "fr": "french",
    "fra": "french",
    "fre": "french",
    "french": "french",
    "de": "german",
    "deu": "german",
    "ger": "german",
    "german": "german",
    "it": "italian",
    "ita": "italian",
    "italian": "italian",
    "pt": "portuguese",
    "por": "portuguese",
    "portuguese": "portuguese",
    "ja": "japanese",
    "jpn": "japanese",
    "japanese": "japanese",
    "ko": "korean",
    "kor": "korean",
    "korean": "korean",
}


def normalise_quality(value: object) -> str:
    quality = str(value or "").strip().lower()
    if quality not in PRESETS:
        raise ValueError("Unknown processing profile.")
    return quality


def normalise_language(value: object) -> str:
    compact = "".join(
        character
        for character in str(value or "").strip().lower()
        if character.isalpha()
    )
    language = _LANGUAGE_ALIASES.get(compact)
    if not language:
        raise ValueError(
            "Unsupported lyrics language. Choose Auto-detect, English, Spanish, French, "
            "German, Italian, Portuguese, Japanese, or Korean."
        )
    return language


def cpu_thread_count() -> int:
    """Return one bounded thread budget shared by native and PyTorch workers."""

    available = max(1, int(os.cpu_count() or 1))
    maximum = min(32, available)
    default = min(8, maximum)
    raw = os.environ.get("LYRICWAVE_CPU_THREADS", "").strip()
    if not raw:
        return default
    try:
        requested = int(raw)
    except ValueError:
        return default
    return max(1, min(maximum, requested))


def model_profile(quality: str) -> ModelProfile:
    return PRESETS[normalise_quality(quality)]


def _model_override(kind: str, quality: str) -> str:
    mode = normalise_quality(quality).upper()
    mode_value = os.environ.get(f"LYRICWAVE_{mode}_{kind}_MODEL", "").strip()
    shared_value = os.environ.get(f"LYRICWAVE_{kind}_MODEL", "").strip()
    return mode_value or shared_value


def _backend_override(kind: str, quality: str) -> str:
    mode = normalise_quality(quality).upper()
    return (
        os.environ.get(f"LYRICWAVE_{mode}_{kind}_BACKEND", "").strip().lower()
        or os.environ.get(f"LYRICWAVE_{kind}_BACKEND", "").strip().lower()
    )


def _legacy_whisper_override(quality: str) -> str:
    mode = normalise_quality(quality).upper()
    return (
        os.environ.get(f"LYRICWAVE_{mode}_WHISPER_MODEL", "").strip()
        or os.environ.get("LYRICWAVE_WHISPER_MODEL", "").strip()
    )


def asr_backend(quality: str) -> str:
    mode = normalise_quality(quality)
    backend = _backend_override("ASR", mode) or model_profile(mode).asr_backend
    if backend not in {"qwen3", "whisper"}:
        raise ValueError("LYRICWAVE_ASR_BACKEND must be qwen3 or whisper.")
    return backend


def asr_model_id(quality: str) -> str:
    mode = normalise_quality(quality)
    override = _model_override("ASR", mode)
    if override:
        return override

    profile = model_profile(mode)
    backend = asr_backend(mode)
    if backend == profile.asr_backend:
        if backend == "whisper":
            return _legacy_whisper_override(mode) or profile.asr_model
        return profile.asr_model
    if backend == "whisper":
        return _legacy_whisper_override(mode) or profile.fallback_asr_model
    return _DEFAULT_QWEN_ASR_MODELS[mode]


def fallback_asr_model_id(quality: str) -> str:
    mode = normalise_quality(quality)
    return (
        _model_override("FALLBACK_ASR", mode)
        or _legacy_whisper_override(mode)
        or model_profile(mode).fallback_asr_model
    )


def whisper_model_id(quality: str) -> str:
    """Return the Whisper model used directly or as the automatic fallback."""

    mode = normalise_quality(quality)
    return asr_model_id(mode) if asr_backend(mode) == "whisper" else fallback_asr_model_id(mode)


def demucs_model_name(quality: str) -> str:
    mode = normalise_quality(quality)
    return _model_override("DEMUCS", mode) or model_profile(mode).demucs


def demucs_pass_count(quality: str) -> int:
    return model_profile(quality).demucs_passes


def alignment_backend(quality: str) -> str:
    mode = normalise_quality(quality)
    backend = _backend_override("ALIGNER", mode) or model_profile(mode).aligner_backend
    if backend not in {"ctc", "qwen3", "none"}:
        raise ValueError("LYRICWAVE_ALIGNER_BACKEND must be ctc, qwen3, or none.")
    return backend


def alignment_model_id(quality: str) -> str:
    mode = normalise_quality(quality)
    override = _model_override("ALIGNER", mode)
    if override:
        return override

    profile = model_profile(mode)
    backend = alignment_backend(mode)
    if backend == "none":
        return ""
    if backend == profile.aligner_backend:
        return profile.aligner_model
    if backend == "qwen3":
        return _DEFAULT_QWEN_ALIGNER_MODEL
    return _DEFAULT_CTC_ALIGNER_MODELS[mode]


def side_pass_enabled(quality: str) -> bool:
    return model_profile(quality).side_pass


def public_model_profiles() -> list[dict[str, object]]:
    profiles: list[dict[str, object]] = []
    for profile_id, profile in PRESETS.items():
        profiles.append(
            {
                "id": profile_id,
                "label": profile.label,
                "description": profile.description,
                "demucs_model": demucs_model_name(profile_id),
                "transcription_model": asr_model_id(profile_id),
                "alignment_model": alignment_model_id(profile_id),
                "side_pass": side_pass_enabled(profile_id),
            }
        )
    return profiles
