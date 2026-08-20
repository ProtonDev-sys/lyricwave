$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
$TorchVersion = if ($env:LYRICWAVE_TORCH_VERSION) { $env:LYRICWAVE_TORCH_VERSION } else { "2.12.1" }
$TorchIndex = if ($env:LYRICWAVE_TORCH_INDEX_URL) { $env:LYRICWAVE_TORCH_INDEX_URL } else { "https://download.pytorch.org/whl/cu130" }

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    throw "FFmpeg and ffprobe must be installed and available on PATH."
}

if (-not (Test-Path -LiteralPath $PythonPath)) {
    Write-Host "Creating the lyricwave Python environment..."
    if ($env:LYRICWAVE_PYTHON) {
        & $env:LYRICWAVE_PYTHON -m venv $VenvPath
    } else {
        py -3.12 -m venv $VenvPath
    }
}

Write-Host "Installing the CUDA build of PyTorch..."
& $PythonPath -m pip install --upgrade pip
& $PythonPath -m pip install "torch==$TorchVersion" --index-url $TorchIndex

Write-Host "Installing Demucs, Whisper, and the localhost API..."
& $PythonPath -m pip install -r (Join-Path $ProjectRoot "backend\requirements.txt")

Write-Host "Checking the GPU runtime..."
& $PythonPath -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available"; import demucs, transformers; print("Ready:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| transformers", transformers.__version__)'

Write-Host "Local engine setup complete. Run: npm run dev"
