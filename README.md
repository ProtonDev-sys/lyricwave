# lyricwave

Turn a local song into word-synced live lyrics without sending the audio off the PC.

## What it does

- plays MP3, WAV, FLAC, M4A, AAC, OGG, and WebM immediately in the browser
- sends the selected file only to a FastAPI service bound to `127.0.0.1`
- isolates a vocal stem on the local NVIDIA GPU with HTDemucs or fine-tuned HTDemucs
- uses vocal-stem RMS activity to give Whisper musical regions instead of speech-style long-form cuts
- aligns English words and characters with Wav2Vec2 CTC, including pauses inside a sung word
- expands written numbers acoustically without changing the displayed lyrics
- keeps Whisper's phrase boundaries while regrouping them into readable lyric lines
- scans the stereo side vocal in Accurate mode for short background responses and ad-libs
- renders lead and secondary lyric lines with a direct audio-clock karaoke animation
- lets every lyric word seek to its own timestamp while line tracking stays inside the lyric panel
- plays either the original mix or the isolated vocal with seek and volume controls
- exports line-timed LRC or exact word timing JSON

The audio file stays on this computer. The first Accurate run downloads several GB of
model files; Hugging Face and PyTorch cache them for later runs.

Each transcription runs in a disposable, below-normal-priority GPU worker with a
VRAM cap. When a job finishes or is cancelled, that process exits so Windows reclaims
its model RAM and CUDA allocations instead of retaining them in the API server.
Completed jobs can be restored from `.local-data` after an engine restart, including
their isolated-vocal playback URL.

## Run locally

Requires Node.js 22.13+, Python 3.12, FFmpeg, and an NVIDIA GPU. On Windows:

```powershell
npm install
npm run setup:engine
npm run dev
```

Open `http://localhost:3000`. `npm run dev` starts both the interface and the private
inference API on `http://127.0.0.1:8008`.

The setup script creates `.venv`, installs the CUDA 13 PyTorch wheel, Demucs, and
Transformers. Fast is the recommended default and uses `htdemucs` with
`whisper-large-v3-turbo`. Accurate mode uses `htdemucs_ft` with
`whisper-large-v3` when a slower second pass is worth trying.

## Model downloads and local data

This repository intentionally contains no model weights, songs, separated stems, or
completed transcription jobs. The first transcription downloads the required model
files into the normal Hugging Face and PyTorch caches on that computer. Local jobs
stay under the ignored `.local-data` directory, and common audio and model-weight
extensions are ignored as an additional guard against accidental commits.

## Verify

```bash
npm run build
npm test
```

Model inference begins only after a user selects an audio file. The original mix is
playable during processing, and the isolated vocal becomes selectable when ready.
