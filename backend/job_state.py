from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.config import demucs_model_name, normalise_quality, whisper_model_id
from backend.ctc_alignment import alignment_model_id


class JobCancelled(RuntimeError):
    pass


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Replace a JSON file without exposing partially written content.

    Progress files are read several times a second and Windows antivirus/indexing
    can briefly hold the destination. A unique same-directory temporary file and
    bounded retries keep concurrent jobs from colliding and make replacement
    resilient without hiding persistent filesystem failures.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    try:
        for attempt in range(12):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(0.025 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class JobState:
    id: str
    filename: str
    source_path: Path
    work_dir: Path
    language: str
    quality: str
    duration: float
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    stage: str = "queued"
    progress: float = 1.0
    status: str = "Waiting for the local GPU"
    error: str = ""
    words: list[dict[str, Any]] = field(default_factory=list)
    lines: list[dict[str, Any]] = field(default_factory=list)
    vocal_path: Path | None = None
    device: str = ""
    future: Future[Any] | None = field(default=None, repr=False)
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    log_tail: deque[str] = field(default_factory=lambda: deque(maxlen=24), repr=False)
    progress_file: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.quality = normalise_quality(self.quality)
        self.duration = max(0.0, float(self.duration))
        self.progress = max(0.0, min(100.0, float(self.progress)))

    def update(self, **values: Any) -> None:
        with self.lock:
            for key, value in values.items():
                if key == "progress":
                    value = max(0.0, min(100.0, float(value)))
                setattr(self, key, value)
            if self.progress_file:
                payload = {
                    "stage": self.stage,
                    "progress": self.progress,
                    "status": self.status,
                    "error": self.error,
                }
                try:
                    write_json_atomic(self.progress_file, payload)
                except OSError as error:
                    # Missing one status tick must not abort a long inference job.
                    print(f"[progress:{self.id[:8]}] {error}", flush=True)

    def public(self, include_result: bool = True) -> dict[str, Any]:
        with self.lock:
            payload: dict[str, Any] = {
                "id": self.id,
                "filename": self.filename,
                "stage": self.stage,
                "progress": round(self.progress, 1),
                "status": self.status,
                "error": self.error,
                "duration": self.duration,
                "quality": self.quality,
                "language": self.language,
                "device": self.device,
                "separation_model": demucs_model_name(self.quality),
                "transcription_model": whisper_model_id(self.quality).split("/")[-1],
                "transcription_model_id": whisper_model_id(self.quality),
                "alignment_model_requested": alignment_model_id(self.quality),
                "created_at": self.created_at,
                "vocal_url": f"/api/jobs/{self.id}/vocals" if self.vocal_path else None,
            }
            if include_result and self.stage == "complete":
                payload["words"] = list(self.words)
                payload["lines"] = list(self.lines)
            return payload
