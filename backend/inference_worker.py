from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from backend.config import cpu_thread_count
from backend.ctc_alignment import release_alignment_model
from backend.inference_pipeline import transcribe_vocals
from backend.job_state import JobState, write_json_atomic
from backend.model_runtime import release_whisper_model


def _vram_fraction() -> float:
    raw = os.environ.get("LYRICWAVE_VRAM_FRACTION", "0.70").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.70
    return max(0.20, min(0.95, value))


def main() -> int:
    parser = argparse.ArgumentParser(description="Disposable lyricwave GPU inference worker")
    parser.add_argument("--vocal", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--quality", choices=("accurate", "fast"), required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--result", required=True)
    arguments = parser.parse_args()

    work_dir = Path(arguments.work_dir).resolve()
    vocal_path = Path(arguments.vocal).resolve()
    progress_path = Path(arguments.progress).resolve()
    result_path = Path(arguments.result).resolve()
    job = JobState(
        id=work_dir.name or "worker",
        filename=arguments.filename,
        source_path=vocal_path,
        work_dir=work_dir,
        language=arguments.language,
        quality=arguments.quality,
        duration=arguments.duration,
        device="cuda",
        stage="transcribing",
        progress=59.0,
        status="Starting the disposable GPU worker",
        progress_file=progress_path,
    )
    job.update()

    try:
        import torch

        cpu_threads = cpu_thread_count()
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(min(2, cpu_threads))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to the inference worker.")
        torch.cuda.set_per_process_memory_fraction(_vram_fraction(), device=0)

        words = transcribe_vocals(job, vocal_path)
        write_json_atomic(result_path, {"ok": True, "words": words})
        return 0
    except Exception as error:
        traceback.print_exc()
        message = str(error).strip() or error.__class__.__name__
        job.update(stage="error", status="Processing stopped", error=message)
        write_json_atomic(result_path, {"ok": False, "error": message})
        return 1
    finally:
        release_alignment_model()
        release_whisper_model()


if __name__ == "__main__":
    sys.exit(main())
