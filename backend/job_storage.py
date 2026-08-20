from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from backend.config import ALLOWED_EXTENSIONS
from backend.job_state import JobState


_REMOVED_SOURCE_NAME = "source.discarded"
DurationProbe = Callable[[Path], float]


def discard_uploaded_source(job: JobState, vocal_path: Path) -> bool:
    """Delete the uploaded mix after separation without touching generated output."""

    with job.lock:
        source_path = job.source_path
        work_dir = job.work_dir

    try:
        source_resolved = source_path.resolve(strict=False)
        work_dir_resolved = work_dir.resolve(strict=False)
        vocal_resolved = vocal_path.resolve(strict=False)
        if not source_resolved.is_relative_to(work_dir_resolved):
            return False
        if source_resolved == vocal_resolved or not source_path.is_file():
            return False
        source_path.unlink()
        return True
    except OSError as error:
        # Retention cleanup remains the fallback. Failing to reclaim disk must
        # not discard an otherwise valid multi-minute inference result.
        print(f"[storage:{job.id[:8]}] could not discard source: {error}", flush=True)
        return False


def restored_source_path(work_dir: Path) -> Path:
    """Return a legacy source file or a stable non-existent placeholder path."""

    candidates = sorted(
        path
        for path in work_dir.glob("source.*")
        if path.suffix.lower() in ALLOWED_EXTENSIONS and path.is_file()
    )
    return candidates[0] if candidates else work_dir / _REMOVED_SOURCE_NAME


def restored_duration(
    metadata: dict[str, Any],
    source_path: Path,
    probe_duration: DurationProbe,
) -> float:
    """Restore duration from metadata, probing only legacy jobs that still have audio."""

    raw_duration = metadata.get("duration")
    if raw_duration not in (None, ""):
        duration = float(raw_duration)
    elif source_path.is_file():
        duration = float(probe_duration(source_path))
    else:
        raise ValueError("Completed job metadata does not contain a valid duration.")

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Completed job duration is not valid.")
    return duration


def restored_filename(metadata: dict[str, Any], source_path: Path) -> str:
    """Return the original display name without requiring the uploaded mix."""

    stored = str(metadata.get("filename", "")).strip()
    if stored:
        return Path(stored).name
    return source_path.name if source_path.is_file() else "Restored track"
