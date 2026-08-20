import asyncio
import json
import unittest

from fastapi import Request
from fastapi.responses import JSONResponse

from backend import server


class LocalRequestSecurityTest(unittest.TestCase):
    @staticmethod
    def _request(
        method: str,
        *,
        origin: str | None = None,
        token: str | None = None,
        path: str = "/api/jobs",
    ) -> Request:
        headers = [(b"host", b"127.0.0.1:8008")]
        if origin is not None:
            headers.append((b"origin", origin.encode("ascii")))
        if token is not None:
            headers.append(
                (server.REQUEST_TOKEN_HEADER.lower().encode("ascii"), token.encode("ascii"))
            )
        return Request(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": method,
                "scheme": "http",
                "path": path,
                "raw_path": path.encode("ascii"),
                "query_string": b"",
                "headers": headers,
                "client": ("127.0.0.1", 50000),
                "server": ("127.0.0.1", 8008),
            }
        )

    @staticmethod
    async def _next(_: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    def _run(self, request: Request) -> JSONResponse:
        return asyncio.run(server.local_security_headers(request, self._next))

    def test_foreign_browser_origin_is_rejected_even_with_the_token(self) -> None:
        response = self._run(
            self._request(
                "POST",
                origin="https://example.test",
                token=server.REQUEST_TOKEN,
            )
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("local lyricwave interface", json.loads(response.body)["detail"])

    def test_local_mutation_requires_the_health_token(self) -> None:
        response = self._run(
            self._request("POST", origin="http://localhost:3000")
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("request token", json.loads(response.body)["detail"])

    def test_local_mutation_with_the_token_is_allowed(self) -> None:
        response = self._run(
            self._request(
                "DELETE",
                origin="http://127.0.0.1:3000",
                token=server.REQUEST_TOKEN,
                path="/api/jobs/" + "a" * 32,
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

    def test_originless_local_client_can_use_the_token(self) -> None:
        response = self._run(
            self._request("POST", token=server.REQUEST_TOKEN)
        )
        self.assertEqual(response.status_code, 200)

    def test_foreign_origin_cannot_read_the_health_token(self) -> None:
        response = self._run(
            self._request("GET", origin="https://example.test", path="/api/health")
        )
        self.assertEqual(response.status_code, 403)

    def test_middleware_order_keeps_cors_outside_direct_guard_responses(self) -> None:
        middleware_names = [item.cls.__name__ for item in server.app.user_middleware]
        self.assertEqual(
            middleware_names[:3],
            ["TrustedHostMiddleware", "CORSMiddleware", "BaseHTTPMiddleware"],
        )

    def test_health_exposes_the_token_to_authorized_local_clients(self) -> None:
        health = server.health()
        self.assertEqual(health["request_token"], server.REQUEST_TOKEN)
        self.assertGreaterEqual(len(server.REQUEST_TOKEN), 32)


if __name__ == "__main__":
    unittest.main()
