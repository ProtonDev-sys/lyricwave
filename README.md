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
- falls back to Whisper phrase timing instead of deleting lead lyrics when sung pronunciation produces weak CTC confidence
- uses token-length-aware fallback timing instead of giving every word an equal share of a phrase
- expands written numbers acoustically without changing the displayed lyrics
- renders lead and secondary lyric lines with a direct audio-clock karaoke animation
- lets every lyric word seek to its own timestamp while line tracking stays inside the lyric panel
- plays either the original mix or the isolated vocal with seek and volume controls
- exports line-timed LRC or exact word timing JSON

The audio file stays on this computer. The first Accurate run downloads several GB of
model files; Hugging Face and PyTorch cache them for later runs.

Each transcription runs in a disposable, below-normal-priority GPU worker with a
configurable VRAM cap. Model workers start in an isolated Windows process tree or POSIX
session, so cancellation terminates their Demucs, FFmpeg, and inference descendants as
well as the parent. When a job finishes or is cancelled, the operating system reclaims
its model RAM and CUDA allocations instead of retaining them in the API server.
Completed jobs can be restored from `.local-data` after an engine restart, including
their isolated-vocal playback URL.

## Run locally

Requires Node.js 22.13+, Python 3.12, FFmpeg/ffprobe, and an NVIDIA GPU with a
compatible CUDA runtime.

```bash
npm install
npm run setup:engine
npm run dev
```

`npm run setup:engine` selects the PowerShell installer on Windows and the Bash
installer on Linux. Set `LYRICWAVE_PYTHON` when Python 3.12 is not exposed under the
usual launcher name.

Open `http://localhost:3000`. `npm run dev` starts both the interface and the private
inference API on `http://127.0.0.1:8008`. The launcher and engine terminate their
complete process trees on Ctrl+C or job cancellation on both Windows and POSIX systems.

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

## Frontend runtime

Completed lyric results are indexed once with cumulative word offsets and prefix
maximum interval ends. During playback, active words and lines are found with binary
search plus a bounded overlap walk instead of scanning the entire song on every
animation frame. The lyric-line tree is memoized, and visual active/past state is
updated directly on the relevant elements so the player clock does not rebuild every
word ten times per second.

Transient localhost polling failures, timeouts, rate limits, and server-side 5xx
responses use capped exponential backoff. An expensive GPU job therefore continues
through a brief browser-to-engine interruption instead of being cancelled immediately.
Playback-source changes attach their metadata and error handlers before loading the
new source, and blocked automatic resume is surfaced to the user.

## Runtime and model overrides

Model checkpoints can be changed without editing source code. A mode-specific setting
takes precedence over the shared setting.

```powershell
$env:LYRICWAVE_ALIGNER_MODEL = "owner/model"
$env:LYRICWAVE_ACCURATE_ALIGNER_MODEL = "owner/accurate-aligner"
$env:LYRICWAVE_WHISPER_MODEL = "owner/whisper-compatible-model"
$env:LYRICWAVE_FAST_WHISPER_MODEL = "owner/fast-whisper-model"
$env:LYRICWAVE_DEMUCS_MODEL = "custom-demucs-model"
$env:LYRICWAVE_VRAM_FRACTION = "0.75"
$env:LYRICWAVE_MAX_PENDING_JOBS = "2"
$env:LYRICWAVE_JOB_RETENTION_HOURS = "24"
$env:LYRICWAVE_CLEANUP_INTERVAL_SECONDS = "900"
npm run dev
```

Custom transcription checkpoints must be compatible with Transformers'
`AutoModelForSpeechSeq2Seq` Whisper timestamp path. Custom aligners must expose a
Wav2Vec2-compatible CTC alphabet containing English letters and a word delimiter.
`LYRICWAVE_VRAM_FRACTION` is clamped to the range `0.20`–`0.95`. Queue capacity is
clamped to 1–16 jobs, retention to 1–720 hours, and cleanup frequency to
30–21,600 seconds.

The setup scripts also accept `LYRICWAVE_TORCH_VERSION` and
`LYRICWAVE_TORCH_INDEX_URL`, allowing the CUDA wheel to be updated independently of
application code.

## Local data and API safeguards

This repository intentionally contains no model weights, songs, separated stems, or
completed transcription jobs. Model files use the normal Hugging Face and PyTorch
caches. Local jobs stay under the ignored `.local-data` directory and expire after 24
hours by default. Expired terminal jobs are pruned from memory and disk during engine
startup and later API activity, so a long-running engine does not accumulate stale data.

The raw upload body is bounded before multipart parsing. Declared oversized requests
are rejected without reading them, and streamed or misleading requests are stopped once
their actual bytes exceed the multipart allowance. The single-GPU queue is also
reserved before FastAPI parses or spools multipart data. By default it accepts two
pending items in total—uploads, queued jobs, and active jobs—and returns HTTP 429 with
`Retry-After` when full.

Cancelling a job that has not reached the GPU removes its executor future immediately,
so stopped tracks cannot remain ahead of later work. If an upload response arrives
after the interface has already switched tracks, that exact superseded job is cancelled
instead of becoming hidden queue work. This prevents multiple browser tabs or clients
from building an unbounded backlog of large local files.

The engine validates processing mode, supported language, file extension, decoded
audio duration, maximum size, and maximum duration before queuing GPU work. Invalid
uploads are removed immediately. The API is bound to loopback, validates the Host
header, rejects browser origins outside localhost, and marks responses as non-cacheable.
Each engine process also creates a random request token. The local interface reads it
from the health endpoint and supplies it through `X-Lyricwave-Token` for POST and DELETE
requests; cross-origin pages cannot read the token or submit mutation requests without
it. If the engine restarts and rotates this token, the interface refreshes health and
retries one rejected mutation with the new token rather than entering a retry loop.

## Model benchmarking

Use exported word-timing JSON to compare Fast, Accurate, or custom model runs against a
manually corrected reference from the same track:

```bash
npm run benchmark:lyrics -- reference.json fast=fast.json accurate=accurate.json
```

The evaluator reports word error rate, character error rate, substitutions/deletions/
insertions, exact-word timing coverage, onset and offset error, timing percentiles, and
signed onset bias. Lead lyrics are measured by default because overlapping ad-libs can
make sequence metrics ambiguous. Include them explicitly when the reference annotates
the secondary layer:

```bash
npm run benchmark:lyrics -- reference.json candidate.json --include-adlibs
npm run benchmark:lyrics -- reference.json candidate.json --json
```

Reference and candidate files may use lyricwave's exported `lines` structure or a
top-level `words` array. Text-only references still produce WER/CER; timing metrics are
calculated only for exact aligned words where both files contain start and end times.
This makes model or threshold changes measurable on a representative private corpus
without committing any songs, lyrics, or generated outputs.

## Verify

```bash
npm run audit:production
npm run lint
npm run typecheck
npm run build
npm run test:frontend
npm run test:python
npm run check
```

`npm run test:python` invokes `python -m unittest backend -v`, so the backend package
itself is a stable discovery target for every `backend/test_*.py` module. The GitHub
Actions workflow runs the production dependency audit, ESLint, explicit TypeScript
checking, frontend build/tests, Python bytecode compilation, and the GPU-free backend
regression suite on each pull request and push to `main`. Model inference begins only
after a user selects an audio file; CUDA model downloads and real-track inference
remain local runtime checks.
