from __future__ import annotations

import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend import server


class RuntimeInfoConcurrencyTest(unittest.TestCase):
    def setUp(self) -> None:
        with server._RUNTIME_LOCK:
            self.original_cache = server._RUNTIME_CACHE
            server._RUNTIME_CACHE = None

    def tearDown(self) -> None:
        with server._RUNTIME_LOCK:
            server._RUNTIME_CACHE = self.original_cache

    def test_simultaneous_cache_misses_share_one_torch_probe(self) -> None:
        worker_count = 8
        start = threading.Barrier(worker_count)
        probe_lock = threading.Lock()
        probe_calls = 0
        results: list[dict[str, object]] = []
        errors: list[BaseException] = []

        def completed_probe(*_: object, **__: object) -> SimpleNamespace:
            nonlocal probe_calls
            with probe_lock:
                probe_calls += 1
            time.sleep(0.06)
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "torch": "test",
                        "cuda": True,
                        "device": "Test GPU",
                    }
                )
            )

        def worker() -> None:
            try:
                start.wait(timeout=2)
                results.append(server._runtime_info())
            except BaseException as error:  # pragma: no cover - assertion aid
                errors.append(error)

        with patch.object(server.shutil, "which", return_value="/usr/bin/tool"), patch.object(
            server, "_package_ready", return_value=True
        ), patch.object(server.subprocess, "run", side_effect=completed_probe):
            threads = [threading.Thread(target=worker) for _ in range(worker_count)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)

        self.assertEqual(errors, [])
        self.assertEqual(probe_calls, 1)
        self.assertEqual(len(results), worker_count)
        self.assertTrue(all(result["ready"] is True for result in results))
        self.assertTrue(all(result["device"] == "Test GPU" for result in results))
        self.assertIsNot(results[0], results[1])

    def test_cached_results_are_returned_as_independent_copies(self) -> None:
        with patch.object(server.shutil, "which", return_value="/usr/bin/tool"), patch.object(
            server, "_package_ready", return_value=True
        ), patch.object(
            server.subprocess,
            "run",
            return_value=SimpleNamespace(
                stdout='{"torch":"test","cuda":true,"device":"Test GPU"}'
            ),
        ):
            first = server._runtime_info()
            first["device"] = "mutated"
            second = server._runtime_info()

        self.assertEqual(second["device"], "Test GPU")


if __name__ == "__main__":
    unittest.main()
