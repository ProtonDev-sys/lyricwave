from __future__ import annotations

import os
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

PRESETS: Final = {
    "accurate": {
        "label": "Accurate",
        "demucs": "htdemucs_ft",
        "whisper": "openai/whisper-large-v3",
        "demucs_passes": 4,
    },
    "fast": {
        "label": "Fast",
        "demucs": "htdemucs",
        "whisper": "openai/whisper-large-v3-turbo",
        "demucs_passes": 1,
    },
}

# These are the choices exposed by the interface. Common ISO aliases are
# accepted at the API boundary and converted to the model-facing names.
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
        raise ValueError("Unknown processing quality.")
    return quality


def normalise_language(value: object) -> str:
    compact = "".join(character for character in str(value or "").strip().lower() if character.isalpha())
    language = _LANGUAGE_ALIASES.get(compact)
    if not language:
        raise ValueError(
            "Unsupported lyrics language. Choose Auto-detect, English, Spanish, French, "
            "German, Italian, Portuguese, Japanese, or Korean."
        )
    return language


def _model_override(kind: str, quality: str) -> str:
    mode = normalise_quality(quality).upper()
    mode_value = os.environ.get(f"LYRICWAVE_{mode}_{kind}_MODEL", "").strip()
    shared_value = os.environ.get(f"LYRICWAVE_{kind}_MODEL", "").strip()
    return mode_value or shared_value


def whisper_model_id(quality: str) -> str:
    mode = normalise_quality(quality)
    return _model_override("WHISPER", mode) or str(PRESETS[mode]["whisper"])


def demucs_model_name(quality: str) -> str:
    mode = normalise_quality(quality)
    return _model_override("DEMUCS", mode) or str(PRESETS[mode]["demucs"])


def demucs_pass_count(quality: str) -> int:
    return int(PRESETS[normalise_quality(quality)]["demucs_passes"])
