from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from backend import server
from backend.config import cpu_thread_count


class CpuThreadBudgetTest(unittest.TestCase):
    def test_default_uses_at_most_eight_available_threads(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "backend.config.os.cpu_count", return_value=16
        ):
            self.assertEqual(cpu_thread_count(), 8)

    def test_default_respects_small_machines(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "backend.config.os.cpu_count", return_value=2
        ):
            self.assertEqual(cpu_thread_count(), 2)

    def test_invalid_override_falls_back_to_the_bounded_default(self) -> None:
        with patch.dict(
            os.environ, {"LYRICWAVE_CPU_THREADS": "invalid"}, clear=True
        ), patch("backend.config.os.cpu_count", return_value=12):
            self.assertEqual(cpu_thread_count(), 8)

    def test_override_is_clamped_to_one_and_available_capacity(self) -> None:
        with patch("backend.config.os.cpu_count", return_value=12):
            with patch.dict(
                os.environ, {"LYRICWAVE_CPU_THREADS": "0"}, clear=True
            ):
                self.assertEqual(cpu_thread_count(), 1)
            with patch.dict(
                os.environ, {"LYRICWAVE_CPU_THREADS": "128"}, clear=True
            ):
                self.assertEqual(cpu_thread_count(), 12)

    def test_global_cap_prevents_extreme_native_thread_pools(self) -> None:
        with patch.dict(
            os.environ, {"LYRICWAVE_CPU_THREADS": "64"}, clear=True
        ), patch("backend.config.os.cpu_count", return_value=96):
            self.assertEqual(cpu_thread_count(), 32)

    def test_worker_environment_uses_one_normalized_budget(self) -> None:
        with patch.object(server, "cpu_thread_count", return_value=5):
            environment = server._worker_environment()
        self.assertEqual(environment["OMP_NUM_THREADS"], "5")
        self.assertEqual(environment["MKL_NUM_THREADS"], "5")
        self.assertEqual(environment["NUMEXPR_NUM_THREADS"], "5")


if __name__ == "__main__":
    unittest.main()
