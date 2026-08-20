from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import Iterable
from typing import Any

from backend.upload_limits import JobUploadSizeLimitMiddleware


class JobUploadSizeLimitMiddlewareTest(unittest.TestCase):
    @staticmethod
    def _run(
        *,
        max_body_size: int = 8,
        path: str = "/api/jobs",
        method: str = "POST",
        headers: Iterable[tuple[bytes, bytes]] = (),
        chunks: Iterable[bytes] = (),
    ) -> tuple[list[dict[str, Any]], bool, int]:
        sent: list[dict[str, Any]] = []
        entered_app = False
        receive_calls = 0
        bodies = list(chunks)

        async def receive() -> dict[str, Any]:
            nonlocal receive_calls
            receive_calls += 1
            if not bodies:
                return {"type": "http.request", "body": b"", "more_body": False}
            body = bodies.pop(0)
            return {
                "type": "http.request",
                "body": body,
                "more_body": bool(bodies),
            }

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        async def downstream(
            scope: dict[str, Any],
            inner_receive: Any,
            inner_send: Any,
        ) -> None:
            nonlocal entered_app
            entered_app = True
            while True:
                message = await inner_receive()
                if not message.get("more_body", False):
                    break
            await inner_send(
                {
                    "type": "http.response.start",
                    "status": 204,
                    "headers": [],
                }
            )
            await inner_send({"type": "http.response.body", "body": b""})

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": list(headers),
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8008),
        }
        middleware = JobUploadSizeLimitMiddleware(
            downstream,
            max_body_size=max_body_size,
        )
        asyncio.run(middleware(scope, receive, send))
        return sent, entered_app, receive_calls

    @staticmethod
    def _status(messages: list[dict[str, Any]]) -> int:
        return int(
            next(
                message["status"]
                for message in messages
                if message["type"] == "http.response.start"
            )
        )

    @staticmethod
    def _json_body(messages: list[dict[str, Any]]) -> dict[str, Any]:
        body = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        )
        return json.loads(body)

    def test_declared_oversize_is_rejected_without_reading_the_body(self) -> None:
        messages, entered_app, receive_calls = self._run(
            headers=[(b"content-length", b"9")],
            chunks=[b"ignored"],
        )
        self.assertEqual(self._status(messages), 413)
        self.assertFalse(entered_app)
        self.assertEqual(receive_calls, 0)
        self.assertIn("500 MB", self._json_body(messages)["detail"])

    def test_streaming_body_is_stopped_when_the_actual_bytes_exceed_the_limit(self) -> None:
        messages, entered_app, receive_calls = self._run(
            chunks=[b"1234", b"56789"],
        )
        self.assertEqual(self._status(messages), 413)
        self.assertTrue(entered_app)
        self.assertEqual(receive_calls, 2)

    def test_body_at_the_limit_reaches_the_application(self) -> None:
        messages, entered_app, _ = self._run(
            headers=[(b"content-length", b"8")],
            chunks=[b"1234", b"5678"],
        )
        self.assertEqual(self._status(messages), 204)
        self.assertTrue(entered_app)

    def test_unrelated_routes_are_not_limited(self) -> None:
        messages, entered_app, _ = self._run(
            path="/api/health",
            method="GET",
            headers=[(b"content-length", b"100")],
            chunks=[b"x" * 100],
        )
        self.assertEqual(self._status(messages), 204)
        self.assertTrue(entered_app)

    def test_invalid_content_length_falls_back_to_stream_counting(self) -> None:
        messages, _, _ = self._run(
            headers=[(b"content-length", b"invalid")],
            chunks=[b"123456789"],
        )
        self.assertEqual(self._status(messages), 413)


if __name__ == "__main__":
    unittest.main()
