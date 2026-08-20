from __future__ import annotations

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
        "import importlib.util\nimport json\nimport math\n",
        "import hmac\nimport importlib.util\nimport json\nimport math\n",
        "hmac import",
    )
    text = replace_once(
        text,
        "import re\nimport shutil\n",
        "import re\nimport secrets\nimport shutil\n",
        "secrets import",
    )
    text = replace_once(
        text,
        "from fastapi import FastAPI, File, Form, HTTPException, UploadFile\n",
        "from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile\n",
        "request import",
    )
    text = replace_once(
        text,
        "from fastapi.responses import FileResponse, JSONResponse\n",
        "from fastapi.responses import FileResponse, JSONResponse\n"
        "from starlette.middleware.trustedhost import TrustedHostMiddleware\n",
        "trusted host import",
    )
    text = replace_once(
        text,
        "_RUNTIME_CACHE: tuple[float, dict[str, Any]] | None = None\n"
        "_RUNTIME_LOCK = threading.RLock()\n",
        "_RUNTIME_CACHE: tuple[float, dict[str, Any]] | None = None\n"
        "_RUNTIME_LOCK = threading.RLock()\n"
        "REQUEST_TOKEN_HEADER = \"X-Lyricwave-Token\"\n"
        "REQUEST_TOKEN = secrets.token_urlsafe(32)\n"
        "_LOCAL_ORIGIN_PATTERN = re.compile(\n"
        "    r\"https?://(?:localhost|127\\.0\\.0\\.1|\\[::1\\])(?::\\d+)?\"\n"
        ")\n"
        "_MUTATING_METHODS = frozenset({\"POST\", \"PUT\", \"PATCH\", \"DELETE\"})\n",
        "security constants",
    )
    text = replace_once(
        text,
        "app.add_middleware(\n"
        "    CORSMiddleware,\n"
        "    allow_origin_regex=r\"https?://(localhost|127\\.0\\.0\\.1)(:\\d+)?\",\n"
        "    allow_credentials=False,\n"
        "    allow_methods=[\"GET\", \"POST\", \"DELETE\", \"OPTIONS\"],\n"
        "    allow_headers=[\"*\"],\n"
        ")\n",
        "app.add_middleware(\n"
        "    CORSMiddleware,\n"
        "    allow_origin_regex=(\n"
        "        r\"https?://(?:localhost|127\\.0\\.0\\.1|\\[::1\\])(?::\\d+)?\"\n"
        "    ),\n"
        "    allow_credentials=False,\n"
        "    allow_methods=[\"GET\", \"POST\", \"DELETE\", \"OPTIONS\"],\n"
        "    allow_headers=[\"Content-Type\", REQUEST_TOKEN_HEADER],\n"
        ")\n"
        "app.add_middleware(\n"
        "    TrustedHostMiddleware,\n"
        "    allowed_hosts=[\"localhost\", \"127.0.0.1\", \"[::1]\"],\n"
        ")\n",
        "API middleware registration",
    )
    text = replace_once(
        text,
        "@app.middleware(\"http\")\n"
        "async def local_security_headers(request: Any, call_next: Any) -> Any:\n"
        "    response = await call_next(request)\n"
        "    response.headers[\"Cache-Control\"] = \"no-store\"\n"
        "    response.headers[\"X-Content-Type-Options\"] = \"nosniff\"\n"
        "    response.headers[\"Referrer-Policy\"] = \"no-referrer\"\n"
        "    return response\n",
        "@app.middleware(\"http\")\n"
        "async def local_security_headers(request: Request, call_next: Any) -> Any:\n"
        "    origin = request.headers.get(\"origin\")\n"
        "    if origin and not _LOCAL_ORIGIN_PATTERN.fullmatch(origin):\n"
        "        response = JSONResponse(\n"
        "            status_code=403,\n"
        "            content={\n"
        "                \"detail\": \"Use the local lyricwave interface to access this engine.\"\n"
        "            },\n"
        "        )\n"
        "    elif (\n"
        "        request.method in _MUTATING_METHODS\n"
        "        and request.url.path.startswith(\"/api/\")\n"
        "        and not hmac.compare_digest(\n"
        "            request.headers.get(REQUEST_TOKEN_HEADER, \"\"),\n"
        "            REQUEST_TOKEN,\n"
        "        )\n"
        "    ):\n"
        "        response = JSONResponse(\n"
        "            status_code=403,\n"
        "            content={\n"
        "                \"detail\": \"The local engine request token is missing or invalid.\"\n"
        "            },\n"
        "        )\n"
        "    else:\n"
        "        response = await call_next(request)\n"
        "    response.headers[\"Cache-Control\"] = \"no-store\"\n"
        "    response.headers[\"X-Content-Type-Options\"] = \"nosniff\"\n"
        "    response.headers[\"Referrer-Policy\"] = \"no-referrer\"\n"
        "    return response\n",
        "local security middleware",
    )
    text = replace_once(
        text,
        "        \"queue_capacity\": max_pending_jobs(),\n"
        "        **_runtime_info(),\n",
        "        \"queue_capacity\": max_pending_jobs(),\n"
        "        \"request_token\": REQUEST_TOKEN,\n"
        "        **_runtime_info(),\n",
        "health request token",
    )
    path.write_text(text, encoding="utf-8")


def patch_page() -> None:
    path = ROOT / "app" / "page.tsx"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "  transformers: boolean;\n};\n",
        "  transformers: boolean;\n"
        "  request_token: string;\n"
        "  pending_jobs?: number;\n"
        "  queue_capacity?: number;\n"
        "};\n",
        "backend health token type",
    )
    text = replace_once(
        text,
        "const LOCAL_API_URL =\n"
        "  process.env.NEXT_PUBLIC_LYRICWAVE_API_URL?.replace(/\\/$/, \"\") ?? \"http://127.0.0.1:8008\";\n",
        "const LOCAL_API_URL =\n"
        "  process.env.NEXT_PUBLIC_LYRICWAVE_API_URL?.replace(/\\/$/, \"\") ?? \"http://127.0.0.1:8008\";\n"
        "const LOCAL_REQUEST_TOKEN_HEADER = \"X-Lyricwave-Token\";\n",
        "frontend token header constant",
    )
    text = replace_once(
        text,
        "  quality: \"accurate\" | \"fast\",\n"
        "  onProgress: (percent: number) => void,\n",
        "  quality: \"accurate\" | \"fast\",\n"
        "  requestToken: string,\n"
        "  onProgress: (percent: number) => void,\n",
        "upload token argument",
    )
    text = replace_once(
        text,
        "    request.open(\"POST\", `${LOCAL_API_URL}/api/jobs`);\n"
        "    request.timeout = 60_000;\n",
        "    request.open(\"POST\", `${LOCAL_API_URL}/api/jobs`);\n"
        "    request.setRequestHeader(LOCAL_REQUEST_TOKEN_HEADER, requestToken);\n"
        "    request.timeout = 60_000;\n",
        "upload token header",
    )
    text = replace_once(
        text,
        "  const uploadRequestRef = useRef<XMLHttpRequest | null>(null);\n"
        "  const activeWordIndexesRef = useRef<number[]>([]);\n",
        "  const uploadRequestRef = useRef<XMLHttpRequest | null>(null);\n"
        "  const requestTokenRef = useRef(\"\");\n"
        "  const activeWordIndexesRef = useRef<number[]>([]);\n",
        "request token ref",
    )
    text = replace_once(
        text,
        "      const health = (await response.json()) as BackendHealth;\n"
        "      setEngineHealth(health);\n"
        "      setEngineState(health.ready ? \"online\" : \"offline\");\n",
        "      const health = (await response.json()) as BackendHealth;\n"
        "      if (!health.request_token) {\n"
        "        throw new Error(\"The local engine did not return a request token.\");\n"
        "      }\n"
        "      requestTokenRef.current = health.request_token;\n"
        "      setEngineHealth(health);\n"
        "      setEngineState(health.ready ? \"online\" : \"offline\");\n",
        "health token capture",
    )
    text = replace_once(
        text,
        "    if (jobId) {\n"
        "      void fetch(`${LOCAL_API_URL}/api/jobs/${jobId}`, {\n"
        "        method: \"DELETE\",\n"
        "        keepalive: true,\n"
        "      }).catch(() => {});\n"
        "    }\n",
        "    if (jobId) {\n"
        "      const requestToken = requestTokenRef.current;\n"
        "      void fetch(`${LOCAL_API_URL}/api/jobs/${jobId}`, {\n"
        "        method: \"DELETE\",\n"
        "        keepalive: true,\n"
        "        headers: requestToken\n"
        "          ? { [LOCAL_REQUEST_TOKEN_HEADER]: requestToken }\n"
        "          : undefined,\n"
        "      }).catch(() => {});\n"
        "    }\n",
        "cancel token header",
    )
    text = replace_once(
        text,
        "        const created = await uploadToLocalEngine(\n"
        "          nextFile,\n"
        "          language,\n"
        "          quality,\n",
        "        const requestToken = requestTokenRef.current;\n"
        "        if (!requestToken) {\n"
        "          throw new Error(\"The local engine request token is unavailable.\");\n"
        "        }\n\n"
        "        const created = await uploadToLocalEngine(\n"
        "          nextFile,\n"
        "          language,\n"
        "          quality,\n"
        "          requestToken,\n",
        "upload token use",
    )
    path.write_text(text, encoding="utf-8")


def patch_frontend_test() -> None:
    path = ROOT / "tests" / "rendered-html.test.mjs"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "  assert.match(page, /FormData\\(\\)/);\n",
        "  assert.match(page, /FormData\\(\\)/);\n"
        "  assert.match(page, /X-Lyricwave-Token/);\n"
        "  assert.match(page, /request_token/);\n",
        "frontend token assertions",
    )
    text = replace_once(
        text,
        "  assert.match(backend, /backend\\.inference_worker/);\n",
        "  assert.match(backend, /backend\\.inference_worker/);\n"
        "  assert.match(backend, /TrustedHostMiddleware/);\n"
        "  assert.match(backend, /hmac\\.compare_digest/);\n",
        "backend security assertions",
    )
    path.write_text(text, encoding="utf-8")


def patch_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "uploads are removed immediately. The API is bound to loopback, accepts browser origins\n"
        "from localhost only, and marks responses as non-cacheable.\n",
        "uploads are removed immediately. The API is bound to loopback, validates the Host header,\n"
        "rejects browser origins outside localhost, and marks responses as non-cacheable. Each\n"
        "engine process also creates a random request token. The local interface reads it from the\n"
        "health endpoint and supplies it through `X-Lyricwave-Token` for POST and DELETE requests;\n"
        "cross-origin pages cannot read the token or submit mutation requests without it.\n",
        "local API security description",
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    patch_server()
    patch_page()
    patch_frontend_test()
    patch_readme()
