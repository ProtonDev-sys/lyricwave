from __future__ import annotations

import gc
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import whisper_model_id
from backend.inference_regions import SAMPLE_RATE


_ASR_PIPELINE: Any | None = None
_ASR_MODEL_ID = ""
_ASR_LOCK = threading.RLock()


def release_whisper_model() -> None:
    global _ASR_MODEL_ID, _ASR_PIPELINE

    with _ASR_LOCK:
        _ASR_PIPELINE = None
        _ASR_MODEL_ID = ""
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def load_whisper_pipeline(job: Any) -> Any:
    global _ASR_MODEL_ID, _ASR_PIPELINE

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    model_id = whisper_model_id(job.quality)
    with _ASR_LOCK:
        if _ASR_PIPELINE is not None and _ASR_MODEL_ID != model_id:
            _ASR_PIPELINE = None
            _ASR_MODEL_ID = ""
            gc.collect()
            torch.cuda.empty_cache()

        if _ASR_PIPELINE is None:
            job.update(
                progress=60.0,
                status=f"Loading {model_id.split('/')[-1]} (first run downloads it)",
            )
            torch.backends.cuda.matmul.allow_tf32 = True
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation=os.environ.get(
                    "LYRICWAVE_WHISPER_ATTENTION",
                    "sdpa",
                ),
            )
            processor = AutoProcessor.from_pretrained(model_id)
            model.to("cuda:0")
            _ASR_PIPELINE = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                dtype=torch.float16,
                device=0,
            )
            _ASR_MODEL_ID = model_id
        else:
            job.update(
                progress=60.0,
                status=f"Using {model_id.split('/')[-1]} on the GPU",
            )
        return _ASR_PIPELINE


def decode_mono(path: Path, view: str = "center") -> np.ndarray:
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
    ]
    if view == "side":
        command.extend(["-af", "pan=mono|c0=0.5*c0-0.5*c1,volume=1.75"])
    elif view != "center":
        raise ValueError(f"Unknown vocal view: {view}")
    command.extend(
        [
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-",
        ]
    )
    completed = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(completed.stdout, dtype=np.float32).copy()
