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
        "    PROJECT_ROOT,\n    TERMINAL_STAGES,\n    demucs_model_name,\n",
        "    PROJECT_ROOT,\n    demucs_model_name,\n",
        "terminal stage import",
    )
    text = replace_once(
        text,
        "from backend.job_lifecycle import (\n"
        "    JobLifecycle,\n"
        "    JobQueueFull,\n"
        "    max_pending_jobs,\n"
        ")\n"
        "from backend.job_state import JobCancelled, JobState, write_json_atomic\n",
        "from backend.job_execution import (\n"
        "    attach_job_future,\n"
        "    cancel_job_execution,\n"
        "    clear_job_execution,\n"
        ")\n"
        "from backend.job_lifecycle import (\n"
        "    JobLifecycle,\n"
        "    JobQueueFull,\n"
        "    max_pending_jobs,\n"
        ")\n"
        "from backend.job_state import JobCancelled, JobState, write_json_atomic\n",
        "job execution import",
    )
    text = replace_once(
        text,
        "def _shutdown_jobs() -> None:\n"
        "    with JOBS_LOCK:\n"
        "        jobs = list(JOBS.values())\n"
        "    for job in jobs:\n"
        "        with job.lock:\n"
        "            process = job.process\n"
        "            terminal = job.stage in TERMINAL_STAGES\n"
        "        if terminal:\n"
        "            continue\n"
        "        job.cancelled.set()\n"
        "        job.update(stage=\"cancelled\", status=\"Engine stopped\")\n"
        "        if process:\n"
        "            _terminate_process(process)\n",
        "def _shutdown_jobs() -> None:\n"
        "    with JOBS_LOCK:\n"
        "        jobs = list(JOBS.values())\n"
        "    for job in jobs:\n"
        "        cancel_job_execution(\n"
        "            job,\n"
        "            status=\"Engine stopped\",\n"
        "            terminate_process=_terminate_process,\n"
        "        )\n",
        "shutdown cancellation",
    )
    text = replace_once(
        text,
        "            with JOBS_LOCK:\n"
        "                JOBS[job_id] = job\n"
        "            EXECUTOR.submit(_run_job, job)\n",
        "            with JOBS_LOCK:\n"
        "                JOBS[job_id] = job\n"
        "            future = EXECUTOR.submit(_run_job, job)\n"
        "            attach_job_future(job, future)\n",
        "executor submission",
    )
    text = replace_once(
        text,
        "@app.delete(\"/api/jobs/{job_id}\")\n"
        "def cancel_job(job_id: str) -> JSONResponse:\n"
        "    job = _require_job(job_id)\n"
        "    job.cancelled.set()\n"
        "    with job.lock:\n"
        "        process = job.process\n"
        "        terminal = job.stage in TERMINAL_STAGES\n"
        "    if not terminal:\n"
        "        job.update(stage=\"cancelled\", status=\"Stopped\")\n"
        "    if process:\n"
        "        _terminate_process(process)\n"
        "    return JSONResponse({\"ok\": True})\n",
        "@app.delete(\"/api/jobs/{job_id}\")\n"
        "def cancel_job(job_id: str) -> JSONResponse:\n"
        "    job = _require_job(job_id)\n"
        "    cancel_job_execution(\n"
        "        job,\n"
        "        status=\"Stopped\",\n"
        "        terminate_process=_terminate_process,\n"
        "    )\n"
        "    return JSONResponse({\"ok\": True})\n",
        "cancel endpoint",
    )
    text = replace_once(
        text,
        "    finally:\n"
        "        with job.lock:\n"
        "            job.process = None\n\n\n"
        "def _check_cancelled(job: JobState) -> None:\n",
        "    finally:\n"
        "        clear_job_execution(job)\n\n\n"
        "def _check_cancelled(job: JobState) -> None:\n",
        "worker cleanup",
    )
    path.write_text(text, encoding="utf-8")


def patch_lifecycle_test() -> None:
    path = ROOT / "backend" / "test_job_lifecycle.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "import unittest\nfrom pathlib import Path\n",
        "import unittest\nfrom concurrent.futures import Future\nfrom pathlib import Path\n"
        "from typing import Any\n",
        "future test imports",
    )
    text = replace_once(
        text,
        "    def test_cleanup_removes_only_expired_terminal_jobs(self) -> None:\n",
        "    def test_terminal_job_with_a_live_future_remains_busy(self) -> None:\n"
        "        with tempfile.TemporaryDirectory() as directory:\n"
        "            root = Path(directory)\n"
        "            job = self._job(root, \"d\" * 32, \"cancelled\")\n"
        "            future: Future[Any] = Future()\n"
        "            job.future = future\n"
        "            lifecycle = JobLifecycle({job.id: job}, threading.RLock())\n\n"
        "            self.assertEqual(lifecycle.busy_count(), 1)\n"
        "            future.cancel()\n"
        "            self.assertEqual(lifecycle.busy_count(), 0)\n\n"
        "    def test_cleanup_removes_only_expired_terminal_jobs(self) -> None:\n",
        "future lifecycle test",
    )
    path.write_text(text, encoding="utf-8")


def patch_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "with `Retry-After` when full. This prevents multiple browser tabs or clients from building\n"
        "an unbounded backlog of large local files.\n",
        "with `Retry-After` when full. Cancelling a job that has not reached the GPU removes its\n"
        "executor future immediately, so stopped tracks cannot remain ahead of later work. This\n"
        "prevents multiple browser tabs or clients from building an unbounded backlog of large\n"
        "local files.\n",
        "queued cancellation documentation",
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    patch_server()
    patch_lifecycle_test()
    patch_readme()
