from __future__ import annotations

import gc
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOB_ROOT = PROJECT_ROOT / ".local-data" / "jobs"
JOB_ROOT.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 500 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".webm"}
TERMINAL_STAGES = {"complete", "error", "cancelled"}

PRESETS = {
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
MIN_ALIGNMENT_CONFIDENCE = 0.10
MIN_ADLIB_ALIGNMENT_CONFIDENCE = 0.17
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
COMMON_HALLUCINATION_PATTERNS = (
    r"\bthank(?:s| you) for watching\b",
    r"\bplease subscribe\b",
    r"\bsubtitles? by\b",
    r"\bamara\.?org\b",
)


class JobCancelled(RuntimeError):
    pass


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    for attempt in range(12):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(0.025 * (attempt + 1))


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
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    log_tail: deque[str] = field(default_factory=lambda: deque(maxlen=24), repr=False)
    progress_file: Path | None = field(default=None, repr=False)

    def update(self, **values: Any) -> None:
        with self.lock:
            for key, value in values.items():
                setattr(self, key, value)
            if self.progress_file:
                payload = {
                    "stage": self.stage,
                    "progress": self.progress,
                    "status": self.status,
                    "error": self.error,
                }
                try:
                    _write_json_atomic(self.progress_file, payload)
                except OSError as error:
                    # Status is polled several times a second; losing one update
                    # must never abort a multi-minute inference job on Windows.
                    print(f"[progress:{self.id[:8]}] {error}", flush=True)

    def public(self, include_result: bool = True) -> dict[str, Any]:
        with self.lock:
            preset = PRESETS[self.quality]
            payload: dict[str, Any] = {
                "id": self.id,
                "filename": self.filename,
                "stage": self.stage,
                "progress": round(self.progress, 1),
                "status": self.status,
                "error": self.error,
                "duration": self.duration,
                "quality": self.quality,
                "device": self.device,
                "separation_model": preset["demucs"],
                "transcription_model": preset["whisper"].split("/")[-1],
                "created_at": self.created_at,
                "vocal_url": f"/api/jobs/{self.id}/vocals" if self.vocal_path else None,
            }
            if include_result and self.stage == "complete":
                payload["words"] = list(self.words)
                payload["lines"] = list(self.lines)
            return payload


JOBS: dict[str, JobState] = {}
JOBS_LOCK = threading.RLock()
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lyricwave-gpu")

_ASR_PIPELINE: Any | None = None
_ASR_MODEL_ID = ""
_ASR_LOCK = threading.RLock()
_ALIGN_MODEL: Any | None = None
_ALIGN_PROCESSOR: Any | None = None
_ALIGN_LOCK = threading.RLock()
_RUNTIME_CACHE: tuple[float, dict[str, Any]] | None = None
_RUNTIME_LOCK = threading.RLock()


app = FastAPI(title="lyricwave local engine", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_store(request: Any, call_next: Any) -> Any:
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


def _package_ready(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _runtime_info() -> dict[str, Any]:
    global _RUNTIME_CACHE

    with _RUNTIME_LOCK:
        if _RUNTIME_CACHE and time.monotonic() - _RUNTIME_CACHE[0] < 60:
            return dict(_RUNTIME_CACHE[1])

    info: dict[str, Any] = {
        "ready": False,
        "cuda": False,
        "device": "CPU",
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "demucs": _package_ready("demucs"),
        "transformers": _package_ready("transformers"),
    }
    probe = (
        "import json,torch; available=bool(torch.cuda.is_available()); "
        "print(json.dumps({'torch':torch.__version__,'cuda':available,"
        "'device':torch.cuda.get_device_name(0) if available else 'CPU'}))"
    )
    try:
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
            creationflags=creation_flags,
        )
        info.update(json.loads(completed.stdout.strip().splitlines()[-1]))
    except Exception as error:  # pragma: no cover - diagnostic path
        info["torch_error"] = str(error)
    info["ready"] = bool(
        info["cuda"] and info["ffmpeg"] and info["demucs"] and info["transformers"]
    )
    with _RUNTIME_LOCK:
        _RUNTIME_CACHE = (time.monotonic(), dict(info))
    return info


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            # GPU inference does not benefit from monopolising every CPU thread,
            # and leaving headroom keeps Windows and the browser responsive.
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
        }
    )
    return environment


def _worker_creation_flags() -> int:
    if os.name != "nt":
        return 0
    return subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "lyricwave-local", **_runtime_info()}


@app.post("/api/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    quality: str = Form("fast"),
) -> dict[str, Any]:
    runtime = _runtime_info()
    if not runtime["ready"]:
        raise HTTPException(
            status_code=503,
            detail="The local GPU engine is not ready. Run npm run setup:engine, then restart npm run dev.",
        )
    if quality not in PRESETS:
        raise HTTPException(status_code=400, detail="Unknown processing quality.")

    original_name = Path(file.filename or "track").name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Choose a supported audio file.")

    job_id = uuid.uuid4().hex
    work_dir = JOB_ROOT / job_id
    work_dir.mkdir(parents=True, exist_ok=False)
    source_path = work_dir / f"source{extension}"
    size = 0
    try:
        with source_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="Audio files are limited to 500 MB.")
                destination.write(chunk)
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    finally:
        await file.close()

    duration = _probe_duration(source_path)
    job = JobState(
        id=job_id,
        filename=original_name,
        source_path=source_path,
        work_dir=work_dir,
        language=language,
        quality=quality,
        duration=duration,
        device=str(runtime["device"]),
    )
    _write_json_atomic(
        work_dir / "job.json",
        {
            "filename": original_name,
            "language": language,
            "quality": quality,
            "duration": duration,
            "device": str(runtime["device"]),
            "created_at": job.created_at,
        },
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
    EXECUTOR.submit(_run_job, job)
    return job.public(include_result=False)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return _require_job(job_id).public()


@app.get("/api/jobs/{job_id}/vocals")
def get_vocals(job_id: str) -> FileResponse:
    job = _require_job(job_id)
    with job.lock:
        vocal_path = job.vocal_path
    if not vocal_path or not vocal_path.exists():
        raise HTTPException(status_code=404, detail="The vocal stem is not ready yet.")
    return FileResponse(vocal_path, media_type="audio/wav", filename=f"{vocal_path.stem}.wav")


@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str) -> JSONResponse:
    job = _require_job(job_id)
    job.cancelled.set()
    with job.lock:
        process = job.process
        if job.stage not in TERMINAL_STAGES:
            job.stage = "cancelled"
            job.status = "Stopped"
    if process and process.poll() is None:
        process.terminate()
    return JSONResponse({"ok": True})


def _require_job(job_id: str) -> JobState:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        job = _restore_completed_job(job_id)
        if job:
            with JOBS_LOCK:
                job = JOBS.setdefault(job_id, job)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown processing job.")
    return job


def _restore_completed_job(job_id: str) -> JobState | None:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        return None
    work_dir = JOB_ROOT / job_id
    result_path = work_dir / "inference-result.json"
    if not work_dir.is_dir() or not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        words = result.get("words")
        if not result.get("ok") or not isinstance(words, list) or not words:
            return None
        metadata_path = work_dir / "job.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        source_path = next(
            path
            for path in work_dir.glob("source.*")
            if path.suffix.lower() in ALLOWED_EXTENSIONS
        )
        quality = str(metadata.get("quality", "accurate"))
        if quality not in PRESETS:
            quality = "accurate"
        restored = JobState(
            id=job_id,
            filename=str(metadata.get("filename", source_path.name)),
            source_path=source_path,
            work_dir=work_dir,
            language=str(metadata.get("language", "english")),
            quality=quality,
            duration=float(metadata.get("duration") or _probe_duration(source_path)),
            created_at=str(
                metadata.get("created_at")
                or datetime.fromtimestamp(work_dir.stat().st_mtime, UTC).isoformat()
            ),
            stage="complete",
            progress=100.0,
            status=f"{len(words)} words synced locally",
            words=words,
            lines=group_words_into_lines(words),
            device=str(metadata.get("device", "Local GPU")),
        )
        vocal_candidates = list((work_dir / "separated").rglob("vocals.wav"))
        if vocal_candidates:
            restored.vocal_path = max(vocal_candidates, key=lambda path: path.stat().st_mtime)
        return restored
    except (FileNotFoundError, StopIteration, ValueError, TypeError, json.JSONDecodeError, OSError):
        return None


def _probe_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return max(0.0, float(completed.stdout.strip()))


def _run_job(job: JobState) -> None:
    try:
        _check_cancelled(job)
        vocal_path = _separate_vocals(job)
        job.update(vocal_path=vocal_path, progress=58.0)
        _check_cancelled(job)
        words = _transcribe_vocals_isolated(job, vocal_path)
        if not words:
            raise RuntimeError(
                "Whisper could not find clear sung words. Try Accurate mode and set the lyrics language explicitly."
            )
        lines = group_words_into_lines(words)
        job.update(
            words=words,
            lines=lines,
            stage="complete",
            progress=100.0,
            status=f"{len(words)} words synced locally",
        )
    except JobCancelled:
        job.update(stage="cancelled", status="Stopped")
    except Exception as error:  # pragma: no cover - exercised by real model failures
        traceback.print_exc()
        message = _friendly_error(error, job)
        job.update(stage="error", status="Processing stopped", error=message)
    finally:
        with job.lock:
            job.process = None


def _check_cancelled(job: JobState) -> None:
    if job.cancelled.is_set():
        raise JobCancelled()


def _friendly_error(error: Exception, job: JobState) -> str:
    raw = str(error).strip()
    lowered = raw.lower()
    if "out of memory" in lowered:
        return "The GPU ran out of memory. Close other GPU-heavy apps or retry in Fast mode."
    if "cuda" in lowered and ("not available" in lowered or "no kernel image" in lowered):
        return "CUDA could not use this GPU. Re-run npm run setup:engine to repair the local model runtime."
    if "connection" in lowered or "download" in lowered:
        return "A model download was interrupted. Check the connection and try again; completed files stay cached."
    if raw:
        return raw.splitlines()[-1][:360]
    with job.lock:
        return job.log_tail[-1][:360] if job.log_tail else "The local model stopped unexpectedly."


def _separate_vocals(job: JobState) -> Path:
    preset = PRESETS[job.quality]
    model_name = str(preset["demucs"])
    pass_count = int(preset["demucs_passes"])
    output_root = job.work_dir / "separated"
    job.update(
        stage="separating",
        progress=6.0,
        status=f"Loading {model_name} on the GPU",
    )

    command = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        model_name,
        "--two-stems",
        "vocals",
        "--shifts",
        "1",
        "--overlap",
        "0.25",
        "-d",
        "cuda",
        "-o",
        str(output_root),
        str(job.source_path),
    ]
    environment = _worker_environment()
    environment["NO_COLOR"] = "1"
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=0,
        env=environment,
        creationflags=_worker_creation_flags(),
    )
    job.update(process=process)

    progress_pattern = re.compile(r"(?<!\d)(\d{1,3})%")
    last_percent = -1
    current_pass = 0
    buffer = ""
    assert process.stdout is not None
    while True:
        character = process.stdout.read(1)
        if character == "" and process.poll() is not None:
            if buffer:
                _record_demucs_line(job, buffer)
            break
        if character not in {"\r", "\n"}:
            buffer += character
            continue
        line = buffer.strip()
        buffer = ""
        if not line:
            continue
        _record_demucs_line(job, line)
        match = progress_pattern.search(line)
        if match:
            percent = min(100, int(match.group(1)))
            if last_percent >= 75 and percent <= 25 and current_pass < pass_count - 1:
                current_pass += 1
            last_percent = percent
            combined = ((current_pass * 100) + percent) / (pass_count * 100)
            job.update(
                progress=min(55.0, 8.0 + combined * 47.0),
                status=(
                    f"Separating vocals · pass {current_pass + 1}/{pass_count} · {percent}%"
                    if pass_count > 1
                    else f"Separating vocals · {percent}%"
                ),
            )
        _check_cancelled(job)

    return_code = process.wait()
    if job.cancelled.is_set():
        raise JobCancelled()
    if return_code != 0:
        with job.lock:
            detail = job.log_tail[-1] if job.log_tail else f"Demucs exited with code {return_code}."
        raise RuntimeError(detail)

    candidates = sorted(output_root.rglob("vocals.wav"))
    if not candidates:
        raise RuntimeError("Demucs finished but did not create a vocal stem.")
    job.update(status="Vocal stem isolated", progress=58.0)
    return candidates[0]


def _record_demucs_line(job: JobState, line: str) -> None:
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line).strip()
    if not clean:
        return
    with job.lock:
        job.log_tail.append(clean)
    print(f"[demucs:{job.id[:8]}] {clean}", flush=True)


def _transcribe_vocals_isolated(job: JobState, vocal_path: Path) -> list[dict[str, Any]]:
    """Run the memory-heavy ML stages in a process that exits after one job.

    PyTorch's Windows allocator can retain several gigabytes even after models are
    deleted. Keeping inference outside the API process makes process exit the hard
    cleanup boundary for both system RAM and CUDA memory.
    """

    progress_path = job.work_dir / "inference-progress.json"
    result_path = job.work_dir / "inference-result.json"
    log_path = job.work_dir / "inference.log"
    for path in (progress_path, result_path, log_path):
        path.unlink(missing_ok=True)

    command = [
        sys.executable,
        "-m",
        "backend.inference_worker",
        "--vocal",
        str(vocal_path),
        "--work-dir",
        str(job.work_dir),
        "--filename",
        job.filename,
        "--language",
        job.language,
        "--quality",
        job.quality,
        "--duration",
        str(job.duration),
        "--progress",
        str(progress_path),
        "--result",
        str(result_path),
    ]
    environment = _worker_environment()
    job.update(
        stage="transcribing",
        progress=59.0,
        status="Starting a clean GPU transcription worker",
    )

    log_stream = log_path.open("w", encoding="utf-8", errors="replace", buffering=1)
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            creationflags=_worker_creation_flags(),
        )
        job.update(process=process)
        last_progress_mtime = -1
        while process.poll() is None:
            if job.cancelled.is_set():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise JobCancelled()

            try:
                progress_mtime = progress_path.stat().st_mtime_ns
                if progress_mtime != last_progress_mtime:
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    updates = {
                        key: progress[key]
                        for key in ("stage", "progress", "status", "error")
                        if key in progress
                    }
                    if updates:
                        job.update(**updates)
                    last_progress_mtime = progress_mtime
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                # Atomic replacement makes partial reads unlikely, but a missed
                # status tick is harmless and the next poll will pick it up.
                pass
            time.sleep(0.25)

        return_code = process.wait()
    finally:
        log_stream.close()

    if job.cancelled.is_set():
        raise JobCancelled()

    log_tail = ""
    try:
        with log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            log_file.seek(max(0, log_file.tell() - 32_768))
            log_tail = log_file.read().decode("utf-8", errors="replace")
        with job.lock:
            job.log_tail.extend(line for line in log_tail.splitlines()[-24:] if line.strip())
    except OSError:
        pass

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        detail = next((line.strip() for line in reversed(log_tail.splitlines()) if line.strip()), "")
        raise RuntimeError(
            detail or f"The GPU worker exited with code {return_code} without returning a result."
        ) from error

    if return_code != 0 or not result.get("ok"):
        raise RuntimeError(str(result.get("error") or f"The GPU worker exited with code {return_code}."))
    words = result.get("words")
    if not isinstance(words, list):
        raise RuntimeError("The GPU worker returned an invalid transcription result.")
    return words


def _release_whisper_model() -> None:
    global _ASR_MODEL_ID, _ASR_PIPELINE
    with _ASR_LOCK:
        _ASR_PIPELINE = None
        _ASR_MODEL_ID = ""
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _release_alignment_model() -> None:
    global _ALIGN_MODEL, _ALIGN_PROCESSOR
    with _ALIGN_LOCK:
        _ALIGN_MODEL = None
        _ALIGN_PROCESSOR = None
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _release_models() -> None:
    _release_whisper_model()
    _release_alignment_model()


def _load_whisper_pipeline(job: JobState) -> Any:
    global _ASR_MODEL_ID, _ASR_PIPELINE

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    model_id = str(PRESETS[job.quality]["whisper"])
    with _ASR_LOCK:
        if _ASR_PIPELINE is not None and _ASR_MODEL_ID != model_id:
            del _ASR_PIPELINE
            _ASR_PIPELINE = None
            _ASR_MODEL_ID = ""
            gc.collect()
            torch.cuda.empty_cache()

        if _ASR_PIPELINE is None:
            job.update(progress=60.0, status=f"Loading {model_id.split('/')[-1]} (first run downloads it)")
            torch.backends.cuda.matmul.allow_tf32 = True
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="sdpa",
            )
            processor = AutoProcessor.from_pretrained(model_id)
            model.to("cuda:0")
            _ASR_PIPELINE = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                dtype=torch.float16,
                device=0,
            )
            _ASR_MODEL_ID = model_id
        else:
            job.update(progress=60.0, status=f"Using {model_id.split('/')[-1]} on the GPU")
        return _ASR_PIPELINE


def _decode_mono(path: Path, view: str = "center") -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
    ]
    if view == "side":
        # Panned doubles and responses are often quiet in a mono fold-down.
        # The stereo difference suppresses the centered lead vocal and exposes
        # those layers without loading a second separation model.
        command.extend(["-af", "pan=mono|c0=0.5*c0-0.5*c1,volume=1.75"])
    command.extend(
        [
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ]
    )
    completed = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(completed.stdout, dtype=np.float32).copy()


def _looks_like_adlib_phrase(text: str, pieces: list[str]) -> bool:
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


def _parenthetical_adlib_flags(pieces: list[str]) -> list[bool]:
    flags: list[bool] = []
    inside = False
    for piece in pieces:
        if re.match(r"^[\"']?[\(\[]", piece):
            inside = True
        flags.append(inside)
        if re.search(r"[\)\]][^\w]*$", piece):
            inside = False
    return flags


def _rms_vocal_regions(
    waveform: np.ndarray,
    sample_rate: int = 16_000,
) -> list[tuple[int, int]]:
    """Create Whisper-sized regions from singing activity in the vocal stem.

    Speech-oriented timestamp segmentation tends to cut sustained or stylised
    singing. RMS activity from the separated stem provides music-aware cuts and
    avoids feeding long silent intros to Whisper, where hallucinations begin.
    """

    if waveform.size < 320:
        return []
    hop_samples = int(0.05 * sample_rate)
    padded_size = int(np.ceil(waveform.size / hop_samples)) * hop_samples
    padded = np.pad(waveform, (0, padded_size - waveform.size))
    frames = padded.reshape(-1, hop_samples)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    peak = float(rms.max(initial=0.0))
    if peak < 0.0005:
        return [(0, waveform.size)]
    active = rms >= peak * 0.10

    # A pause must last a full second before it is safe to cut a lyrical phrase.
    minimum_silence_frames = int(1.0 / 0.05)
    index = 0
    while index < active.size:
        if active[index]:
            index += 1
            continue
        gap_start = index
        while index < active.size and not active[index]:
            index += 1
        if (
            gap_start > 0
            and index < active.size
            and index - gap_start < minimum_silence_frames
        ):
            active[gap_start:index] = True

    runs: list[tuple[int, int]] = []
    index = 0
    while index < active.size:
        if not active[index]:
            index += 1
            continue
        run_start = index
        while index < active.size and active[index]:
            index += 1
        if index - run_start >= 3:
            runs.append(
                (
                    max(0, run_start * hop_samples - int(0.25 * sample_rate)),
                    min(waveform.size, index * hop_samples + int(0.25 * sample_rate)),
                )
            )
    if not runs:
        return [(0, waveform.size)]

    maximum_samples = int(29.0 * sample_rate)
    preferred_minimum = int(18.0 * sample_rate)
    maximum_merge_gap = int(7.0 * sample_rate)
    merged: list[tuple[int, int]] = []
    current_start, current_end = runs[0]
    for run_start, run_end in runs[1:]:
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
            regions.append((region_start, cut))
            region_start = cut
        if region_end - region_start >= int(0.2 * sample_rate):
            regions.append((region_start, region_end))
    return regions


def _transcribe_view(
    job: JobState,
    transcriber: Any,
    waveform: np.ndarray,
    regions: list[tuple[int, int]],
    layer: str,
    progress_start: float,
    progress_span: float,
) -> list[dict[str, Any]]:
    sample_rate = 16_000
    language = None if job.language == "auto" else job.language
    generation: dict[str, Any] = {
        "task": "transcribe",
        "num_beams": 2 if job.quality == "accurate" else 1,
        "do_sample": False,
        "max_new_tokens": 160,
        "condition_on_prev_tokens": False,
    }
    if language:
        generation["language"] = language

    segments: list[dict[str, Any]] = []
    for index, (start_sample, end_sample) in enumerate(regions):
        _check_cancelled(job)
        audio_chunk = waveform[start_sample:end_sample]
        start_time = start_sample / sample_rate
        if audio_chunk.size < 320 or float(np.sqrt(np.mean(np.square(audio_chunk)))) < 0.0001:
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
            # Segment timestamps come from Whisper's timestamp tokens. Unlike
            # `"word"` timestamps, this does not retain decoder attention maps.
            return_timestamps=True,
            generate_kwargs=generation,
            decoder_kwargs={"clean_up_tokenization_spaces": False},
        )
        chunk_duration = audio_chunk.size / sample_rate
        timestamped_chunks = result.get("chunks")
        if not isinstance(timestamped_chunks, list) or not timestamped_chunks:
            timestamped_chunks = [
                {"text": result.get("text", ""), "timestamp": (0.0, chunk_duration)}
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
                segment_end = float(timestamp[1] if timestamp[1] is not None else chunk_duration)
            else:
                segment_start, segment_end = 0.0, chunk_duration
            segment_start = max(0.0, min(chunk_duration, segment_start))
            segment_end = max(segment_start + 0.04, min(chunk_duration, segment_end))

            # A small amount of acoustic context helps the CTC aligner without
            # letting it attach a first word to an ad-lib several seconds away.
            crop_start = max(0.0, segment_start - 0.3)
            crop_end = min(chunk_duration, segment_end + 0.3)
            cropped_audio = np.ascontiguousarray(audio_chunk[
                int(crop_start * sample_rate) : max(
                    int(crop_start * sample_rate) + 1,
                    int(crop_end * sample_rate),
                )
            ]).copy()
            local_start = segment_start - crop_start
            spoken_duration = max(0.04, segment_end - segment_start)
            phrase_is_adlib = layer == "side" or _looks_like_adlib_phrase(
                chunk_text, pieces
            )
            parenthetical_flags = _parenthetical_adlib_flags(pieces)
            raw_words = [
                {
                    "text": piece,
                    "start": round(local_start + (word_index / len(pieces)) * spoken_duration, 3),
                    "end": round(
                        local_start + ((word_index + 1) / len(pieces)) * spoken_duration,
                        3,
                    ),
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
                }
            )
    return segments


def _filter_secondary_adlibs(
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
        # A long side-channel sentence is almost always the centered lead leaking
        # through stereo effects. Short responses are the useful secondary layer.
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
        if duplicate_ratio >= 0.65:
            continue
        retained.extend(ordered)
    return retained


def _polish_word_timings(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose the audible onset and remove impossible same-layer overlaps."""

    grouped: dict[int, list[dict[str, Any]]] = {}
    for original in words:
        grouped.setdefault(int(original.get("_segment", -1)), []).append(dict(original))
    phrases = [
        sorted(group, key=lambda word: (float(word["start"]), float(word["end"])))
        for group in grouped.values()
    ]
    phrases.sort(key=lambda group: (float(group[0]["start"]), int(group[0].get("_segment", -1))))

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

    # Whisper timestamp sections can touch or slightly overlap. Their text order
    # is still authoritative, so place the last word of one lead phrase before
    # the first word of the next instead of interleaving their text by timestamp.
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


def _transcribe_vocals(job: JobState, vocal_path: Path) -> list[dict[str, Any]]:
    job.update(stage="transcribing", progress=59.0, status="Preparing the isolated vocal")
    waveform = _decode_mono(vocal_path)
    if waveform.size == 0:
        raise RuntimeError("The isolated vocal stem was empty.")
    transcriber = _load_whisper_pipeline(job)
    _check_cancelled(job)

    use_side_pass = job.quality == "accurate"
    regions = _rms_vocal_regions(waveform)
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
            side_waveform = _decode_mono(vocal_path, "side")
            if side_waveform.size:
                segments.extend(
                    _transcribe_view(
                        job,
                        transcriber,
                        side_waveform,
                        regions,
                        "side",
                        78.0,
                        9.0,
                    )
                )
            del side_waveform
        except subprocess.CalledProcessError as side_error:
            print(f"[adlibs:{job.id[:8]}] stereo side pass skipped: {side_error}", flush=True)

    use_english_alignment = job.language.lower() in {"en", "english"}
    del waveform
    del transcriber
    _release_whisper_model()

    words: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        _check_cancelled(job)
        raw_words = list(segment["words"])
        local_words = raw_words if not use_english_alignment else []
        segment_confidence = 1.0 if not use_english_alignment else 0.0
        if use_english_alignment and raw_words:
            job.update(
                progress=88.0 + (index / max(1, len(segments))) * 9.0,
                status=f"Tightening phrase {index + 1}/{len(segments)}",
            )
            try:
                aligned = _align_words_ctc(segment["audio"], raw_words)
                if aligned:
                    confidence = sum(
                        float(word.get("_confidence", 0.0)) for word in aligned
                    ) / len(aligned)
                    segment_confidence = confidence
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
                    accepted = confidence >= minimum_confidence and not looks_hallucinated
                    if accepted:
                        local_words = aligned
                    print(
                        f"[align:{job.id[:8]}] confidence={confidence:.3f} "
                        f"accepted={accepted} text={aligned_text}",
                        flush=True,
                    )
            except Exception as alignment_error:
                print(f"[align:{job.id[:8]}] {alignment_error}", flush=True)

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
                                max(0.0, start_time + float(unit["start"])), 3
                            ),
                            "end": round(
                                max(
                                    start_time + float(unit["start"]) + 0.008,
                                    start_time + float(unit["end"]),
                                ),
                                3,
                            ),
                            "fill": round(float(unit["fill"]), 4),
                            "pause_before": bool(unit.get("pause_before", False)),
                        }
                        for unit in word.get("_timing", [])
                    ],
                    "_segment": index,
                    "_segment_confidence": segment_confidence,
                    "_word_confidence": float(word.get("_confidence", segment_confidence)),
                    "_kind": str(word.get("_kind", segment["kind"])),
                    "_source_layer": str(segment["source_layer"]),
                    "_explicit_adlib": bool(
                        word.get("_explicit_adlib", segment["explicit_adlib"])
                    ),
                }
            )

    _release_alignment_model()
    job.update(progress=98.0, status="Polishing word timings")
    # RMS-VAD regions do not overlap, so every acoustically accepted phrase is
    # meaningful. The old fixed-window resolver could mistakenly erase a short
    # first/last word where adjacent Whisper timestamp sections touched.
    lead_words = [word for word in words if str(word.get("_kind")) == "lead"]
    adlib_words = [word for word in words if str(word.get("_kind")) == "adlib"]
    adlib_words = _filter_secondary_adlibs(adlib_words, lead_words)
    polished = _polish_word_timings(lead_words) + _polish_word_timings(adlib_words)
    return _deduplicate_words(polished)


def _load_ctc_aligner() -> tuple[Any, Any]:
    global _ALIGN_MODEL, _ALIGN_PROCESSOR

    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    with _ALIGN_LOCK:
        if _ALIGN_MODEL is None or _ALIGN_PROCESSOR is None:
            model_id = "facebook/wav2vec2-base-960h"
            _ALIGN_PROCESSOR = AutoProcessor.from_pretrained(model_id)
            _ALIGN_MODEL = AutoModelForCTC.from_pretrained(
                model_id,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            _ALIGN_MODEL.eval()
        _ALIGN_MODEL.to("cuda:0")
        return _ALIGN_MODEL, _ALIGN_PROCESSOR


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


def _integer_to_english(value: int) -> str:
    """Return a compact English cardinal suitable for acoustic alignment."""

    if value < 0:
        return f"minus {_integer_to_english(-value)}"
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
            f" {_integer_to_english(remainder)}" if remainder else ""
        )
    for scale, label in (
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
    ):
        if value >= scale:
            leading, remainder = divmod(value, scale)
            return f"{_integer_to_english(leading)} {label}" + (
                f" {_integer_to_english(remainder)}" if remainder else ""
            )
    return " ".join(_SMALL_CARDINALS[int(digit)] for digit in str(value))


def _ordinal_to_english(value: int) -> str:
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
                return f"{_integer_to_english(leading)} {label} {_ordinal_to_english(remainder)}"
            return f"{_integer_to_english(leading)} {label}th"
    return _integer_to_english(value)


def _alignment_spoken_form(text: str) -> str:
    """Expand written numbers without changing the lyric text shown to the user.

    Wav2Vec2's English CTC alphabet has no digit tokens. Previously, a Whisper
    token such as ``21`` normalized to an empty string and silently disappeared
    from the timed lyric result. Expanding only for alignment preserves the
    original display/export form while giving the acoustic model letters to
    align against.
    """

    expanded = str(text)

    def decimal_words(raw: str) -> str:
        clean = raw.replace(",", "")
        if "." in clean:
            whole, fraction = clean.split(".", 1)
            leading = _integer_to_english(int(whole or "0"))
            trailing = " ".join(_SMALL_CARDINALS[int(digit)] for digit in fraction)
            return f"{leading} point {trailing}"
        return _integer_to_english(int(clean))

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
        lambda match: _ordinal_to_english(int(match.group(1).replace(",", ""))),
        expanded,
        flags=re.IGNORECASE,
    )
    expanded = re.sub(
        r"\d[\d,]*(?:\.\d+)?",
        lambda match: decimal_words(match.group(0)),
        expanded,
    )
    return re.sub(r"[^A-Z]+", "", expanded.upper())


def _ctc_gap_is_silent(
    audio: np.ndarray,
    gap_start: float,
    gap_end: float,
    word_start: float,
    word_end: float,
    sample_rate: int = 16_000,
) -> bool:
    """Return whether a CTC blank is a real pause rather than a held sound.

    CTC emits narrow character peaks with blank frames between them. Those
    blanks occur both during sustained vowels and during genuinely split words,
    so duration alone makes long notes look frozen. We mark a pause only when a
    run of low vocal energy is present relative to the surrounding word.
    """

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


def _align_words_ctc(
    audio: np.ndarray,
    words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import torch

    model, processor = _load_ctc_aligner()
    tokenizer = processor.tokenizer
    blank_id = int(tokenizer.pad_token_id)
    delimiter = tokenizer.word_delimiter_token or "|"
    delimiter_id = int(tokenizer.convert_tokens_to_ids(delimiter))

    target_ids: list[int] = []
    token_owners: list[int | None] = []
    valid_words: list[tuple[int, dict[str, Any]]] = []
    for word_index, word in enumerate(words):
        normalized = _alignment_spoken_form(str(word["text"]))
        character_ids: list[int] = []
        for character in normalized:
            token_id = int(tokenizer.convert_tokens_to_ids(character))
            if token_id == tokenizer.unk_token_id:
                continue
            character_ids.append(token_id)
        if not character_ids:
            continue
        if target_ids:
            target_ids.append(delimiter_id)
            token_owners.append(None)
        target_ids.extend(character_ids)
        token_owners.extend([word_index] * len(character_ids))
        valid_words.append((word_index, word))

    if not target_ids or audio.size < 320:
        return []

    inputs = processor(audio, sampling_rate=16_000, return_tensors="pt").input_values
    inputs = inputs.to(device="cuda:0", dtype=torch.float16)
    with torch.inference_mode():
        emissions = model(inputs).logits[0].float().log_softmax(dim=-1).cpu()
    del inputs

    token_spans = _ctc_token_spans(emissions, target_ids, blank_id)
    if not token_spans:
        return []
    seconds_per_frame = (audio.size / 16_000) / emissions.shape[0]

    refined_spans: list[tuple[int, int]] = []
    for token_index, (start, end) in enumerate(token_spans):
        token_scores = emissions[start:end, target_ids[token_index]]
        blank_scores = emissions[start:end, blank_id]
        active = torch.nonzero(token_scores >= blank_scores, as_tuple=False).flatten()
        if active.numel():
            refined_spans.append((start + int(active[0]), start + int(active[-1]) + 1))
        else:
            peak = start + int(token_scores.argmax())
            refined_spans.append((peak, peak + 1))

    by_owner: dict[int, list[tuple[int, int]]] = {}
    for token_index, span in enumerate(refined_spans):
        owner = token_owners[token_index]
        if owner is not None:
            by_owner.setdefault(owner, []).append(span)

    aligned: list[dict[str, Any]] = []
    for word_index, word in valid_words:
        spans = by_owner.get(word_index)
        if not spans:
            continue
        owned_tokens = {
            token_index for token_index, owner in enumerate(token_owners) if owner == word_index
        }
        best_token_log_probs = [
            float(emissions[start:end, target_ids[token_index]].max())
            for token_index, (start, end) in enumerate(token_spans)
            if token_index in owned_tokens
        ]
        confidence = (
            float(np.exp(np.mean(best_token_log_probs))) if best_token_log_probs else 0.0
        )
        word_start = spans[0][0] * seconds_per_frame
        word_end = spans[-1][1] * seconds_per_frame
        timing: list[dict[str, Any]] = []
        for index, (start, end) in enumerate(spans):
            unit: dict[str, Any] = {
                "start": round(start * seconds_per_frame, 3),
                "end": round(max(start + 1, end) * seconds_per_frame, 3),
                "fill": round((index + 1) / len(spans), 4),
            }
            if index > 0:
                previous_end = spans[index - 1][1] * seconds_per_frame
                current_start = start * seconds_per_frame
                if _ctc_gap_is_silent(
                    audio,
                    previous_end,
                    current_start,
                    word_start,
                    word_end,
                ):
                    unit["pause_before"] = True
            timing.append(unit)
        aligned.append(
            {
                "text": word["text"],
                "start": round(spans[0][0] * seconds_per_frame, 3),
                "end": round(spans[-1][1] * seconds_per_frame, 3),
                "_timing": timing,
                "_confidence": round(confidence, 4),
                "_kind": str(word.get("_kind", "lead")),
                "_explicit_adlib": bool(word.get("_explicit_adlib", False)),
            }
        )
    return aligned


def _ctc_token_spans(
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


def _deduplicate_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            previous_normalized = re.sub(r"[^\w']+", "", str(previous["text"]).lower())
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


def _resolve_overlapping_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer the more acoustically credible window when overlap disagrees.

    Whisper sees overlapping 20-second windows. A phrase crossing the seam can
    consequently be emitted twice with slightly different timing or wording.
    CTC confidence lets us keep the stronger word at each point in time while
    retaining the non-overlapping continuation from either phrase.
    """

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
            phrase_changed
            and (gap > 0.06 or len(current) >= 4 or line_duration >= 2.5)
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
    groups.sort(key=lambda item: (float(item[1][0]["start"]), 0 if item[0] == "lead" else 1))

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


def _cleanup_old_jobs() -> None:
    cutoff = time.time() - 24 * 60 * 60
    for directory in JOB_ROOT.iterdir():
        try:
            if directory.is_dir() and directory.stat().st_mtime < cutoff:
                shutil.rmtree(directory, ignore_errors=True)
        except OSError:
            continue


_cleanup_old_jobs()
