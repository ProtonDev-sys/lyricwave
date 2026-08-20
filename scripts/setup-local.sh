#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
PYTHON_BIN="${LYRICWAVE_PYTHON:-python3.12}"
TORCH_VERSION="${LYRICWAVE_TORCH_VERSION:-2.12.1}"
TORCH_INDEX_URL="${LYRICWAVE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.12 was not found. Set LYRICWAVE_PYTHON to a compatible Python executable." >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "FFmpeg and ffprobe must be installed and available on PATH." >&2
  exit 1
fi

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  echo "Creating the lyricwave Python environment..."
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

PYTHON_PATH="$VENV_PATH/bin/python"
echo "Installing the CUDA build of PyTorch..."
"$PYTHON_PATH" -m pip install --upgrade pip
"$PYTHON_PATH" -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"

echo "Installing Demucs, Whisper, and the localhost API..."
"$PYTHON_PATH" -m pip install -r "$PROJECT_ROOT/backend/requirements.txt"

echo "Checking the GPU runtime..."
"$PYTHON_PATH" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available"; import demucs, transformers; print("Ready:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| transformers", transformers.__version__)'

echo "Local engine setup complete. Run: npm run dev"
