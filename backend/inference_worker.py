from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from backend.ctc_alignment import release_alignment_model
from backend.inference_pipeline import transcribe_vocals
from backend.server import JobState, _release_models, _write_json_atomic


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _write_json_atomic(path, payload)


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

    # Cap the caching allocator well below total VRAM. A bad input should fail
    # with a handled CUDA OOM instead of pressuring the display driver.
    import torch

    torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
    torch.set_num_interop_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the inference worker.")
    torch.cuda.set_per_process_memory_fraction(0.70, device=0)

    work_dir = Path(arguments.work_dir).resolve()
    vocal_path = Path(arguments.vocal).resolve()
    progress_path = Path(arguments.progress).resolve()
    result_path = Path(arguments.result).resolve()
    job = JobState(
        id="worker",
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
        words = transcribe_vocals(job, vocal_path)
        _write_json(result_path, {"ok": True, "words": words})
        return 0
    except Exception as error:
        traceback.print_exc()
        message = str(error).strip() or error.__class__.__name__
        job.update(stage="error", status="Processing stopped", error=message)
        _write_json(result_path, {"ok": False, "error": message})
        return 1
    finally:
        release_alignment_model()
        _release_models()


if __name__ == "__main__":
    sys.exit(main())
