from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} block, found {count}.")
    return text.replace(old, new, 1)


def patch_server() -> None:
    path = ROOT / "backend" / "server.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from backend.job_state import JobCancelled, JobState, write_json_atomic\n",
        "from backend.job_lifecycle import (\n"
        "    JobLifecycle,\n"
        "    JobQueueFull,\n"
        "    max_pending_jobs,\n"
        ")\n"
        "from backend.job_state import JobCancelled, JobState, write_json_atomic\n",
        "job lifecycle import",
    )
    text = replace_once(
        text,
        "EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix=\"lyricwave-gpu\")\n"
        "_RUNTIME_CACHE: tuple[float, dict[str, Any]] | None = None\n",
        "EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix=\"lyricwave-gpu\")\n"
        "JOB_LIFECYCLE = JobLifecycle(JOBS, JOBS_LOCK)\n"
        "_RUNTIME_CACHE: tuple[float, dict[str, Any]] | None = None\n",
        "job lifecycle registry",
    )

    cleanup_pattern = re.compile(
        r"def _cleanup_old_jobs\(\) -> None:\n.*?\n\n\ndef _shutdown_jobs",
        flags=re.DOTALL,
    )
    text, count = cleanup_pattern.subn(
        "def _cleanup_old_jobs() -> int:\n"
        "    return JOB_LIFECYCLE.cleanup(JOB_ROOT, force=True)\n\n\n"
        "def _shutdown_jobs",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Expected one cleanup function, found {count}.")

    text = replace_once(
        text,
        "@app.get(\"/api/health\")\n"
        "def health() -> dict[str, Any]:\n"
        "    return {\"ok\": True, \"service\": \"lyricwave-local\", **_runtime_info()}\n",
        "@app.get(\"/api/health\")\n"
        "def health() -> dict[str, Any]:\n"
        "    JOB_LIFECYCLE.cleanup(JOB_ROOT)\n"
        "    return {\n"
        "        \"ok\": True,\n"
        "        \"service\": \"lyricwave-local\",\n"
        "        \"pending_jobs\": JOB_LIFECYCLE.busy_count(),\n"
        "        \"queue_capacity\": max_pending_jobs(),\n"
        "        **_runtime_info(),\n"
        "    }\n",
        "health endpoint",
    )

    create_job_pattern = re.compile(
        r"@app\.post\(\"/api/jobs\", status_code=202\)\n"
        r"async def create_job\(.*?\n\n\n"
        r"(?=@app\.get\(\"/api/jobs/\{job_id\}\"\))",
        flags=re.DOTALL,
    )
    create_job = '''@app.post("/api/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    quality: str = Form("fast"),
) -> dict[str, Any]:
    try:
        runtime = _runtime_info()
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

        try:
            with JOB_LIFECYCLE.reserve(JOB_ROOT):
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
                    duration = _probe_duration(source_path)
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
                    EXECUTOR.submit(_run_job, job)
                except Exception:
                    with JOBS_LOCK:
                        JOBS.pop(job_id, None)
                    shutil.rmtree(work_dir, ignore_errors=True)
                    raise
                return job.public(include_result=False)
        except JobQueueFull as error:
            raise HTTPException(
                status_code=429,
                detail=str(error),
                headers={"Retry-After": str(error.retry_after_seconds)},
            ) from error
    finally:
        await file.close()


'''
    text, count = create_job_pattern.subn(create_job, text, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one create_job endpoint, found {count}.")

    path.write_text(text, encoding="utf-8")


def patch_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '$env:LYRICWAVE_VRAM_FRACTION = "0.75"\n'
        "npm run dev\n",
        '$env:LYRICWAVE_VRAM_FRACTION = "0.75"\n'
        '$env:LYRICWAVE_MAX_PENDING_JOBS = "2"\n'
        '$env:LYRICWAVE_JOB_RETENTION_HOURS = "24"\n'
        '$env:LYRICWAVE_CLEANUP_INTERVAL_SECONDS = "900"\n'
        "npm run dev\n",
        "runtime override example",
    )
    text = replace_once(
        text,
        "`LYRICWAVE_VRAM_FRACTION` is clamped to the range `0.20`–`0.95`.\n",
        "`LYRICWAVE_VRAM_FRACTION` is clamped to the range `0.20`–`0.95`. Queue\n"
        "capacity is clamped to 1–16 jobs, retention to 1–720 hours, and cleanup\n"
        "frequency to 30–21,600 seconds.\n",
        "override bounds",
    )
    text = replace_once(
        text,
        "caches. Local jobs stay under the ignored `.local-data` directory and expire after 24\n"
        "hours.\n\n"
        "The engine validates processing mode, supported language, file extension, decoded\n",
        "caches. Local jobs stay under the ignored `.local-data` directory and expire after 24\n"
        "hours by default. Expired terminal jobs are pruned from memory and disk during engine\n"
        "startup and later API activity, so a long-running engine does not accumulate stale data.\n\n"
        "The single-GPU queue is bounded before upload bytes are stored. By default it accepts\n"
        "two pending items in total—uploads, queued jobs, and active jobs—and returns HTTP 429\n"
        "with `Retry-After` when full. This prevents multiple browser tabs or clients from building\n"
        "an unbounded backlog of large local files.\n\n"
        "The engine validates processing mode, supported language, file extension, decoded\n",
        "local retention description",
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    patch_server()
    patch_readme()
