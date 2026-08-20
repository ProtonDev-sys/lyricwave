from __future__ import annotations

import hmac
import importlib.util
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.config import (
    ALLOWED_EXTENSIONS,
    cpu_thread_count,
    JOB_ROOT,
    MAX_DURATION_SECONDS,
    MAX_FILE_SIZE,
    PRESETS,
    PROJECT_ROOT,
    demucs_model_name,
    demucs_pass_count,
    normalise_language,
    normalise_quality,
)
from backend.ctc_alignment import release_alignment_model
from backend.inference_regions import _adaptive_vocal_regions
from backend.job_execution import (
    attach_job_future,
    cancel_job_execution,
    clear_job_execution,
)
from backend.job_lifecycle import (
    JobLifecycle,
    JobQueueFull,
    max_pending_jobs,
)
from backend.job_state import JobCancelled, JobState, write_json_atomic
from backend.lyric_processing import (
    COMMON_HALLUCINATION_PATTERNS,
    MIN_ADLIB_ALIGNMENT_CONFIDENCE,
    MIN_ALIGNMENT_CONFIDENCE,
    alignment_spoken_form as _alignment_spoken_form,
    ctc_gap_is_silent as _ctc_gap_is_silent,
    ctc_token_spans as _ctc_token_spans,
    deduplicate_words as _deduplicate_words,
    filter_secondary_adlibs as _filter_secondary_adlibs,
    group_words_into_lines,
    looks_like_adlib_phrase as _looks_like_adlib_phrase,
    parenthetical_adlib_flags as _parenthetical_adlib_flags,
    polish_word_timings as _polish_word_timings,
    resolve_overlapping_words as _resolve_overlapping_words,
)
from backend.model_runtime import (
    decode_mono as _decode_mono,
    load_whisper_pipeline as _load_whisper_pipeline,
    release_whisper_model as _release_whisper_model,
)
from backend.process_control import (
    terminate_process_tree as _terminate_process,
    worker_process_options,
)
from backend.upload_limits import JobUploadSizeLimitMiddleware


# Compatibility aliases for the existing test suite and local scripts. The
# implementations now live in focused modules rather than in this API server.
_write_json_atomic = write_json_atomic
_rms_vocal_regions = _adaptive_vocal_regions

JOB_ROOT.mkdir(parents=True, exist_ok=True)
JOBS: dict[str, JobState] = {}
JOBS_LOCK = threading.RLock()
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lyricwave-gpu")
JOB_LIFECYCLE = JobLifecycle(JOBS, JOBS_LOCK)
_RUNTIME_CACHE: tuple[float, dict[str, Any]] | None = None
_RUNTIME_LOCK = threading.RLock()
REQUEST_TOKEN_HEADER = "X-Lyricwave-Token"
REQUEST_TOKEN = secrets.token_urlsafe(32)
_LOCAL_ORIGIN_PATTERN = re.compile(
    r"https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?"
)
_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _release_models() -> None:
    _release_whisper_model()
    release_alignment_model()


def _cleanup_old_jobs() -> int:
    return JOB_LIFECYCLE.cleanup(JOB_ROOT, force=True)


def _shutdown_jobs() -> None:
    with JOBS_LOCK:
        jobs = list(JOBS.values())
    for job in jobs:
        cancel_job_execution(
            job,
            status="Engine stopped",
            terminate_process=_terminate_process,
        )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    _cleanup_old_jobs()
    yield
    _shutdown_jobs()
    EXECUTOR.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="lyricwave local engine",
    version="0.3.0",
    lifespan=lifespan,
)
app.add_middleware(JobUploadSizeLimitMiddleware)


@app.middleware("http")
async def local_security_headers(request: Request, call_next: Any) -> Any:
    origin = request.headers.get("origin")
    if origin and not _LOCAL_ORIGIN_PATTERN.fullmatch(origin):
        response = JSONResponse(
            status_code=403,
            content={
                "detail": "Use the local lyricwave interface to access this engine."
            },
        )
    elif (
        request.method in _MUTATING_METHODS
        and request.url.path.startswith("/api/")
        and not hmac.compare_digest(
            request.headers.get(REQUEST_TOKEN_HEADER, ""),
            REQUEST_TOKEN,
        )
    ):
        response = JSONResponse(
            status_code=403,
            content={
                "detail": "The local engine request token is missing or invalid."
            },
        )
    else:
        try:
            if (
                request.method == "POST"
                and request.url.path.rstrip("/") == "/api/jobs"
            ):
                # Middleware runs before FastAPI parses multipart form data, so a
                # full queue rejects the request before UploadFile can spool bytes.
                with JOB_LIFECYCLE.reserve(JOB_ROOT):
                    response = await call_next(request)
            else:
                response = await call_next(request)
        except JobQueueFull as error:
            response = JSONResponse(
                status_code=429,
                content={"detail": str(error)},
                headers={"Retry-After": str(error.retry_after_seconds)},
            )
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=(
        r"https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?"
    ),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", REQUEST_TOKEN_HEADER],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "[::1]"],
)


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
        "ffprobe": shutil.which("ffprobe") is not None,
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
        info["cuda"]
        and info["ffmpeg"]
        and info["ffprobe"]
        and info["demucs"]
        and info["transformers"]
    )
    with _RUNTIME_LOCK:
        _RUNTIME_CACHE = (time.monotonic(), dict(info))
    return info


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    thread_budget = str(cpu_thread_count())
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "OMP_NUM_THREADS": thread_budget,
            "MKL_NUM_THREADS": thread_budget,
            "NUMEXPR_NUM_THREADS": thread_budget,
        }
    )
    return environment


@app.get("/api/health")
def health() -> dict[str, Any]:
    JOB_LIFECYCLE.cleanup(JOB_ROOT)
    return {
        "ok": True,
        "service": "lyricwave-local",
        "pending_jobs": JOB_LIFECYCLE.busy_count(),
        "queue_capacity": max_pending_jobs(),
        "request_token": REQUEST_TOKEN,
        **_runtime_info(),
    }


@app.post("/api/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    quality: str = Form("fast"),
) -> dict[str, Any]:
    try:
        runtime = await run_in_threadpool(_runtime_info)
        if not runtime["ready"]:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The local GPU engine is not ready. Run npm run setup:engine, "
                    "then restart npm run dev."
                ),
            )
        try:
            quality = normalise_quality(quality)
            language = normalise_language(language)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

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
                        raise HTTPException(
                            status_code=413,
                            detail="Audio files are limited to 500 MB.",
                        )
                    destination.write(chunk)
            if size == 0:
                raise HTTPException(
                    status_code=400,
                    detail="The selected audio file is empty.",
                )
            duration = await run_in_threadpool(_probe_duration, source_path)
            if duration > MAX_DURATION_SECONDS:
                raise HTTPException(
                    status_code=413,
                    detail="Audio duration is limited to three hours.",
                )
        except HTTPException:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail="The selected file could not be decoded as supported audio.",
            ) from error

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
        try:
            write_json_atomic(
                work_dir / "job.json",
                {
                    "schema": 2,
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
            future = EXECUTOR.submit(_run_job, job)
            attach_job_future(job, future)
        except Exception:
            with JOBS_LOCK:
                JOBS.pop(job_id, None)
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
        return job.public(include_result=False)
    finally:
        await file.close()


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
    return FileResponse(
        vocal_path,
        media_type="audio/wav",
        filename=f"{vocal_path.stem}.wav",
    )


@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str) -> JSONResponse:
    job = _require_job(job_id)
    cancel_job_execution(
        job,
        status="Stopped",
        terminate_process=_terminate_process,
    )
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
        try:
            quality = normalise_quality(metadata.get("quality", "accurate"))
        except ValueError:
            quality = "accurate"
        try:
            language = normalise_language(metadata.get("language", "english"))
        except ValueError:
            language = "english"
        duration = float(metadata.get("duration") or _probe_duration(source_path))
        restored = JobState(
            id=job_id,
            filename=str(metadata.get("filename", source_path.name)),
            source_path=source_path,
            work_dir=work_dir,
            language=language,
            quality=quality,
            duration=duration,
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
            restored.vocal_path = max(
                vocal_candidates,
                key=lambda path: path.stat().st_mtime,
            )
        return restored
    except (
        FileNotFoundError,
        StopIteration,
        ValueError,
        TypeError,
        json.JSONDecodeError,
        OSError,
        subprocess.SubprocessError,
    ):
        return None


def _probe_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-nostdin",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    duration = float(completed.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Audio duration is not valid.")
    return duration


def _run_job(job: JobState) -> None:
    try:
        _check_cancelled(job)
        vocal_path = _separate_vocals(job)
        job.update(vocal_path=vocal_path, progress=58.0)
        _check_cancelled(job)
        words = _transcribe_vocals_isolated(job, vocal_path)
        if not words:
            raise RuntimeError(
                "Whisper could not find clear sung words. Try Accurate mode and "
                "set the lyrics language explicitly."
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
        clear_job_execution(job)


def _check_cancelled(job: JobState) -> None:
    if job.cancelled.is_set():
        raise JobCancelled()


def _friendly_error(error: Exception, job: JobState) -> str:
    raw = str(error).strip()
    lowered = raw.lower()
    if "out of memory" in lowered:
        return "The GPU ran out of memory. Close other GPU-heavy apps or retry in Fast mode."
    if "cuda" in lowered and ("not available" in lowered or "no kernel image" in lowered):
        return (
            "CUDA could not use this GPU. Re-run npm run setup:engine to repair "
            "the local model runtime."
        )
    if "connection" in lowered or "download" in lowered:
        return (
            "A model download was interrupted. Check the connection and try again; "
            "completed files stay cached."
        )
    if raw:
        return raw.splitlines()[-1][:360]
    with job.lock:
        return (
            job.log_tail[-1][:360]
            if job.log_tail
            else "The local model stopped unexpectedly."
        )


def _separate_vocals(job: JobState) -> Path:
    model_name = demucs_model_name(job.quality)
    pass_count = demucs_pass_count(job.quality)
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
        **worker_process_options(),
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
            detail = (
                job.log_tail[-1]
                if job.log_tail
                else f"Demucs exited with code {return_code}."
            )
        raise RuntimeError(detail)

    candidates = list(output_root.rglob("vocals.wav"))
    if not candidates:
        raise RuntimeError("Demucs finished but did not create a vocal stem.")
    vocal_path = max(candidates, key=lambda path: path.stat().st_mtime)
    job.update(status="Vocal stem isolated", progress=58.0)
    return vocal_path


def _record_demucs_line(job: JobState, line: str) -> None:
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line).strip()
    if not clean:
        return
    with job.lock:
        job.log_tail.append(clean)
    print(f"[demucs:{job.id[:8]}] {clean}", flush=True)


def _transcribe_vocals_isolated(
    job: JobState,
    vocal_path: Path,
) -> list[dict[str, Any]]:
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
    job.update(
        stage="transcribing",
        progress=59.0,
        status="Starting a clean GPU transcription worker",
    )

    with log_path.open(
        "w",
        encoding="utf-8",
        errors="replace",
        buffering=1,
    ) as log_stream:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            env=_worker_environment(),
            **worker_process_options(),
        )
        job.update(process=process)
        last_progress_mtime = -1
        while process.poll() is None:
            if job.cancelled.is_set():
                _terminate_process(process)
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
                pass
            time.sleep(0.25)

        return_code = process.wait()

    if job.cancelled.is_set():
        raise JobCancelled()

    log_tail = ""
    try:
        with log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            log_file.seek(max(0, log_file.tell() - 32_768))
            log_tail = log_file.read().decode("utf-8", errors="replace")
        with job.lock:
            job.log_tail.extend(
                line for line in log_tail.splitlines()[-24:] if line.strip()
            )
    except OSError:
        pass

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        detail = next(
            (line.strip() for line in reversed(log_tail.splitlines()) if line.strip()),
            "",
        )
        raise RuntimeError(
            detail
            or f"The GPU worker exited with code {return_code} without returning a result."
        ) from error

    if return_code != 0 or not result.get("ok"):
        raise RuntimeError(
            str(result.get("error") or f"The GPU worker exited with code {return_code}.")
        )
    words = result.get("words")
    if not isinstance(words, list) or any(not isinstance(word, dict) for word in words):
        raise RuntimeError("The GPU worker returned an invalid transcription result.")
    return words


_cleanup_old_jobs()
