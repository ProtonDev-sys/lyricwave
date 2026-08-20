from __future__ import annotations

import signal
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from backend import process_control


class FakeProcess:
    def __init__(
        self,
        *,
        pid: int = 1234,
        poll_result: int | None = None,
    ) -> None:
        self.pid = pid
        self.poll_result = poll_result
        self.terminate = Mock()
        self.kill = Mock()
        self.wait = Mock(return_value=0)

    def poll(self) -> int | None:
        return self.poll_result


class ProcessControlTest(unittest.TestCase):
    def test_worker_options_create_a_windows_process_tree(self) -> None:
        options = process_control.worker_process_options("nt")
        self.assertEqual(
            options["creationflags"],
            process_control._CREATE_NO_WINDOW
            | process_control._BELOW_NORMAL_PRIORITY_CLASS,
        )
        self.assertNotIn("start_new_session", options)

    def test_worker_options_create_a_posix_session(self) -> None:
        self.assertEqual(
            process_control.worker_process_options("posix"),
            {"start_new_session": True},
        )

    def test_completed_process_is_untouched(self) -> None:
        process = FakeProcess(poll_result=0)
        with patch.object(process_control.os, "killpg") as killpg:
            process_control.terminate_process_tree(
                process,
                platform_name="posix",
            )
        killpg.assert_not_called()
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_posix_process_group_is_terminated(self) -> None:
        process = FakeProcess()
        with (
            patch.object(
                process_control.os,
                "getpgid",
                return_value=process.pid,
            ),
            patch.object(process_control.os, "killpg") as killpg,
        ):
            process_control.terminate_process_tree(
                process,
                platform_name="posix",
            )
        killpg.assert_called_once_with(process.pid, signal.SIGTERM)
        process.wait.assert_called_once_with(timeout=10.0)
        process.terminate.assert_not_called()

    def test_posix_timeout_escalates_to_the_process_group(self) -> None:
        process = FakeProcess()
        process.wait.side_effect = [
            subprocess.TimeoutExpired("worker", 0.01),
            0,
        ]
        with (
            patch.object(
                process_control.os,
                "getpgid",
                return_value=process.pid,
            ),
            patch.object(process_control.os, "killpg") as killpg,
        ):
            process_control.terminate_process_tree(
                process,
                platform_name="posix",
                terminate_timeout=0.01,
                kill_timeout=0.02,
            )
        self.assertEqual(
            killpg.call_args_list,
            [
                call(process.pid, signal.SIGTERM),
                call(process.pid, signal.SIGKILL),
            ],
        )
        self.assertEqual(
            process.wait.call_args_list,
            [call(timeout=0.01), call(timeout=0.02)],
        )

    def test_non_leader_process_uses_safe_direct_fallback(self) -> None:
        process = FakeProcess()
        with (
            patch.object(
                process_control.os,
                "getpgid",
                return_value=999,
            ),
            patch.object(process_control.os, "killpg") as killpg,
        ):
            process_control.terminate_process_tree(
                process,
                platform_name="posix",
            )
        killpg.assert_not_called()
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=10.0)

    def test_windows_uses_taskkill_for_the_process_tree(self) -> None:
        process = FakeProcess()
        with patch.object(
            process_control.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0),
        ) as run:
            process_control.terminate_process_tree(
                process,
                platform_name="nt",
            )
        run.assert_called_once_with(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
            timeout=10.0,
            creationflags=process_control._CREATE_NO_WINDOW,
        )
        process.wait.assert_called_once_with(timeout=3.0)
        process.terminate.assert_not_called()

    def test_failed_windows_tree_kill_falls_back_and_escalates(self) -> None:
        process = FakeProcess()
        process.wait.side_effect = [
            subprocess.TimeoutExpired("worker", 0.01),
            0,
        ]
        with patch.object(
            process_control.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=1),
        ):
            process_control.terminate_process_tree(
                process,
                platform_name="nt",
                terminate_timeout=0.01,
                kill_timeout=0.02,
            )
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(
            process.wait.call_args_list,
            [call(timeout=0.01), call(timeout=0.02)],
        )

    def test_server_wires_both_model_workers_to_isolated_trees(self) -> None:
        server = Path(__file__).with_name("server.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(server.count("**worker_process_options(),"), 2)
        self.assertNotIn("def _worker_creation_flags", server)
        self.assertNotIn("def _terminate_process", server)


if __name__ == "__main__":
    unittest.main()
