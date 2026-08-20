from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.config import MAX_FILE_SIZE


MULTIPART_OVERHEAD_ALLOWANCE = 2 * 1024 * 1024
MAX_JOB_REQUEST_BODY_SIZE = MAX_FILE_SIZE + MULTIPART_OVERHEAD_ALLOWANCE
UPLOAD_TOO_LARGE_DETAIL = "Audio files are limited to 500 MB."


class RequestBodyTooLarge(RuntimeError):
    pass


def _content_length(scope: Scope) -> int | None:
    values = [
        value
        for name, value in scope.get("headers", [])
        if name.lower() == b"content-length"
    ]
    if len(values) != 1:
        return None
    try:
        value = int(values[0].decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        return None
    return value if value >= 0 else None


def _is_job_upload(scope: Scope) -> bool:
    return bool(
        scope.get("type") == "http"
        and str(scope.get("method", "")).upper() == "POST"
        and str(scope.get("path", "")).rstrip("/") == "/api/jobs"
    )


class JobUploadSizeLimitMiddleware:
    """Reject oversized upload bodies before multipart parsing can spool them."""

    def __init__(
        self,
        app: ASGIApp,
        max_body_size: int = MAX_JOB_REQUEST_BODY_SIZE,
    ) -> None:
        if max_body_size < 1:
            raise ValueError("max_body_size must be positive")
        self.app = app
        self.max_body_size = int(max_body_size)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            status_code=413,
            content={"detail": UPLOAD_TOO_LARGE_DETAIL},
        )
        await response(scope, receive, send)

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if not _is_job_upload(scope):
            await self.app(scope, receive, send)
            return

        declared_length = _content_length(scope)
        if declared_length is not None and declared_length > self.max_body_size:
            await self._reject(scope, receive, send)
            return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_size:
                    raise RequestBodyTooLarge()
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if response_started:
                raise
            await self._reject(scope, receive, send)
