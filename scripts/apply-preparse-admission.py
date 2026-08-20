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
    middleware_registration = '''app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=(
        r"https?://(?:localhost|127\\.0\\.0\\.1|\\[::1\\])(?::\\d+)?"
    ),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", REQUEST_TOKEN_HEADER],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "[::1]"],
)


'''
    text = replace_once(
        text,
        middleware_registration,
        "",
        "early middleware registration",
    )
    old_security = '''@app.middleware("http")
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
        response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response
'''
    new_security = '''@app.middleware("http")
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
        r"https?://(?:localhost|127\\.0\\.0\\.1|\\[::1\\])(?::\\d+)?"
    ),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", REQUEST_TOKEN_HEADER],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "[::1]"],
)
'''
    text = replace_once(text, old_security, new_security, "request guard middleware")

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
    finally:
        await file.close()


'''
    text, count = create_job_pattern.subn(create_job, text, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one create_job endpoint, found {count}.")
    path.write_text(text, encoding="utf-8")


def patch_lifecycle_test() -> None:
    path = ROOT / "backend" / "test_job_lifecycle.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace("import io\n", "")
    text = text.replace("from fastapi import HTTPException, UploadFile\n", "from fastapi import Request\n")
    api_pattern = re.compile(
        r"class ApiAdmissionTest\(unittest\.TestCase\):.*?\n\n\nif __name__ == \"__main__\":",
        flags=re.DOTALL,
    )
    replacement = '''class ApiAdmissionTest(unittest.TestCase):
    @staticmethod
    def _request() -> Request:
        return Request(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/jobs",
                "raw_path": b"/api/jobs",
                "query_string": b"",
                "headers": [
                    (b"host", b"127.0.0.1:8008"),
                    (b"origin", b"http://localhost:3000"),
                    (
                        server.REQUEST_TOKEN_HEADER.lower().encode("ascii"),
                        server.REQUEST_TOKEN.encode("ascii"),
                    ),
                ],
                "client": ("127.0.0.1", 50000),
                "server": ("127.0.0.1", 8008),
            }
        )

    def tearDown(self) -> None:
        with server.JOBS_LOCK:
            server.JOBS.clear()

    def test_full_queue_rejects_before_multipart_parsing(self) -> None:
        async def must_not_parse_body(_: Request):
            raise AssertionError("A full queue must reject before endpoint body parsing.")

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"LYRICWAVE_MAX_PENDING_JOBS": "1"},
            clear=True,
        ), patch.object(server, "JOB_ROOT", Path(directory)):
            active_dir = Path(directory) / ("c" * 32)
            active_dir.mkdir()
            active = JobState(
                id="c" * 32,
                filename="active.mp3",
                source_path=active_dir / "source.mp3",
                work_dir=active_dir,
                language="english",
                quality="fast",
                duration=10,
                stage="queued",
            )
            with server.JOBS_LOCK:
                server.JOBS[active.id] = active

            response = asyncio.run(
                server.local_security_headers(self._request(), must_not_parse_body)
            )

            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.headers["Retry-After"], "5")
            self.assertEqual([item.name for item in Path(directory).iterdir()], [active.id])


if __name__ == "__main__":'''
    text, count = api_pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one API admission test class, found {count}.")
    path.write_text(text, encoding="utf-8")


def patch_security_test() -> None:
    path = ROOT / "backend" / "test_local_security.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "    def test_health_exposes_the_token_to_authorized_local_clients(self) -> None:\n",
        "    def test_middleware_order_keeps_cors_outside_direct_guard_responses(self) -> None:\n"
        "        middleware_names = [item.cls.__name__ for item in server.app.user_middleware]\n"
        "        self.assertEqual(\n"
        "            middleware_names[:3],\n"
        "            [\"TrustedHostMiddleware\", \"CORSMiddleware\", \"BaseHTTPMiddleware\"],\n"
        "        )\n\n"
        "    def test_health_exposes_the_token_to_authorized_local_clients(self) -> None:\n",
        "middleware ordering test",
    )
    path.write_text(text, encoding="utf-8")


def patch_frontend_test() -> None:
    path = ROOT / "tests" / "rendered-html.test.mjs"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "  assert.match(backend, /hmac\\.compare_digest/);\n",
        "  assert.match(backend, /hmac\\.compare_digest/);\n"
        "  assert.match(backend, /with JOB_LIFECYCLE\\.reserve\\(JOB_ROOT\\)/);\n"
        "  assert.match(backend, /before FastAPI parses multipart form data/);\n",
        "pre-parse admission assertions",
    )
    path.write_text(text, encoding="utf-8")


def patch_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "The single-GPU queue is bounded before upload bytes are stored. By default it accepts\n",
        "The single-GPU queue is reserved in middleware before FastAPI parses or spools multipart\n"
        "upload data. By default it accepts\n",
        "queue admission description",
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    patch_server()
    patch_lifecycle_test()
    patch_security_test()
    patch_frontend_test()
    patch_readme()
