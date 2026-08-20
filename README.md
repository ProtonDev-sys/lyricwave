# lyricwave

Turn a local song into word-synced live lyrics without sending the audio off the PC.

## What it does

- plays MP3, WAV, FLAC, M4A, AAC, OGG, and WebM immediately in the browser
- sends the selected file only to a FastAPI service bound to `127.0.0.1`
- isolates a vocal stem on the local NVIDIA GPU with HTDemucs or fine-tuned HTDemucs
- finds vocal activity with a local, adaptive energy threshold so a loud chorus does not hide a quiet verse
- gives the stereo side channel its own activity scan in Accurate mode, allowing isolated responses and ad-libs to be found during lead-vocal pauses
- transcribes musical regions with multilingual Whisper and preserves its phrase boundaries
- automatically applies English CTC alignment when Whisper detects English, even when the language control is set to Auto-detect
- aligns English words and characters with Wav2Vec2 CTC, including pauses inside a sustained or split sung word
- uses token-length-aware fallback timing instead of giving every word an equal share of a phrase
- expands written numbers acoustically without changing the displayed lyrics
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

## Processing modes

Fast is the recommended default. It uses:

- `htdemucs`
- `openai/whisper-large-v3-turbo`
- `facebook/wav2vec2-base-960h` for English word and character alignment

Accurate uses:

- `htdemucs_ft`
- `openai/whisper-large-v3`
- `facebook/wav2vec2-large-960h-lv60-self` for higher-capacity English alignment
- an independent stereo-side pass for short background responses and ad-libs

The Whisper checkpoints remain separate from alignment: Whisper determines the lyric
text and phrase boundaries, then the CTC model tightens English word and character
timing after Whisper has been released from GPU memory.

The alignment checkpoint can be overridden without changing source code:

```powershell
$env:LYRICWAVE_ALIGNER_MODEL = "owner/model"
$env:LYRICWAVE_FAST_ALIGNER_MODEL = "owner/fast-model"
$env:LYRICWAVE_ACCURATE_ALIGNER_MODEL = "owner/accurate-model"
npm run dev
```

A mode-specific variable takes precedence over the shared variable. Custom models
must expose a Wav2Vec2-compatible CTC alphabet containing English letters and a word
delimiter.

## Model downloads and local data

This repository intentionally contains no model weights, songs, separated stems, or
completed transcription jobs. The first transcription downloads the required model
files into the normal Hugging Face and PyTorch caches on that computer. Local jobs
stay under the ignored `.local-data` directory, and common audio and model-weight
extensions are ignored as an additional guard against accidental commits.

## Verify

```powershell
npm run build
npm run test:frontend
npm run test:python
npm test
```

The Python test command discovers every `backend/test_*.py` module, including
segmentation and model-selection regression tests. Model inference begins only after a
user selects an audio file. The original mix is playable during processing, and the
isolated vocal becomes selectable when ready.
