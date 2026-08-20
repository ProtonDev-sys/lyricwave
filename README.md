# lyricwave

Turn a local song into word-synced live lyrics without sending the audio off the PC.

## What it does

- plays MP3, WAV, FLAC, M4A, AAC, OGG, and WebM immediately in the browser
- sends the selected file only to a FastAPI service bound to `127.0.0.1`
- isolates a vocal stem on the local NVIDIA GPU with HTDemucs or fine-tuned HTDemucs
- finds vocal activity with a local, adaptive energy threshold so a loud chorus does not hide a quiet verse
- gives the stereo side channel its own activity scan in Best quality mode, allowing isolated responses and ad-libs to be found during lead-vocal pauses
- transcribes musical regions with a selectable local ASR profile and preserves phrase boundaries
- uses multilingual forced alignment where supported, with English CTC timing as a fallback
- falls back to Whisper instead of failing the job when a selected Qwen checkpoint cannot load
- falls back to phrase timing instead of deleting lead lyrics when sung pronunciation produces weak alignment confidence
- uses token-length-aware fallback timing instead of giving every word an equal share of a phrase
- expands written numbers acoustically without changing the displayed lyrics
- renders lead and secondary lyric lines with a direct audio-clock karaoke animation
- lets every lyric word seek to its own timestamp while line tracking stays inside the lyric panel
- plays either the original mix or the isolated vocal with seek and volume controls
- exports line-timed LRC or exact word timing JSON with processing provenance

The audio file stays on this computer. Model files are downloaded into the normal
Hugging Face and PyTorch caches and reused on later runs.

Each transcription runs in a disposable, below-normal-priority GPU worker with a
configurable VRAM cap. When a job finishes or is cancelled, that process exits so the
operating system reclaims its model RAM and CUDA allocations instead of retaining
them in the API server. Completed jobs can be restored from `.local-data` after an
engine restart, including their isolated-vocal playback URL.

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

## Model profiles

The interface exposes three profiles and defaults to Recommended:

| Profile | Vocal separation | Transcription | Word timing |
| --- | --- | --- | --- |
| Fast | `htdemucs` | `openai/whisper-large-v3-turbo` | `facebook/wav2vec2-base-960h` for English |
| Recommended | `htdemucs` | `Qwen/Qwen3-ASR-0.6B-hf` | `Qwen/Qwen3-ForcedAligner-0.6B-hf` |
| Best quality | `htdemucs_ft` with four passes | `Qwen/Qwen3-ASR-1.7B-hf` | Qwen forced alignment plus a background-vocal pass |

The browser reads this profile list from the local engine health response, so labels,
defaults, checkpoints, and backend execution use one configuration source.

Recommended and Best quality fall back to Whisper large-v3-turbo and large-v3
respectively if a Qwen checkpoint cannot load. English timing then falls back to the
existing Wav2Vec2 CTC path when needed. Models are downloaded once into the normal
local Hugging Face cache.

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
$env:LYRICWAVE_ASR_MODEL = "owner/asr-model"
$env:LYRICWAVE_BALANCED_ASR_MODEL = "owner/recommended-asr"
$env:LYRICWAVE_FALLBACK_ASR_MODEL = "owner/whisper-fallback"
$env:LYRICWAVE_DEMUCS_MODEL = "custom-demucs-model"
$env:LYRICWAVE_VRAM_FRACTION = "0.75"
$env:LYRICWAVE_CPU_THREADS = "8"
$env:LYRICWAVE_MAX_PENDING_JOBS = "2"
$env:LYRICWAVE_JOB_RETENTION_HOURS = "24"
$env:LYRICWAVE_CLEANUP_INTERVAL_SECONDS = "900"
npm run dev
```

Custom ASR checkpoints must match the selected `qwen3` or `whisper` backend.
Custom aligners must match the selected `qwen3` or `ctc` backend. Backend and model
overrides can be shared or profile-specific.
`LYRICWAVE_VRAM_FRACTION` is clamped to the range `0.20`–`0.95`.
`LYRICWAVE_CPU_THREADS` is parsed once for native math libraries and PyTorch, defaults
to at most eight available logical CPUs, and is clamped to 1–32 without exceeding the
reported machine capacity. Queue capacity is clamped to 1–16 jobs, retention to 1–720
hours, and cleanup frequency to 30–21,600 seconds.

The setup scripts also accept `LYRICWAVE_TORCH_VERSION` and
`LYRICWAVE_TORCH_INDEX_URL`, allowing the CUDA wheel to be updated independently of
application code.

## Local data and API safeguards

This repository intentionally contains no model weights, songs, separated stems, or
completed transcription jobs. Model files use the normal Hugging Face and PyTorch
caches. Local jobs stay under the ignored `.local-data` directory and expire after 24
hours by default. Expired terminal jobs are pruned from memory and disk during engine
startup and later API activity, so a long-running engine does not accumulate stale data.

The single-GPU queue is reserved in middleware before FastAPI parses or spools multipart
upload data. By default it accepts two pending items in total—uploads, queued jobs, and
active jobs—and returns HTTP 429 with `Retry-After` when full. Cancelling a job that has
not reached the GPU removes its executor future immediately, so stopped tracks cannot
remain ahead of later work. This prevents multiple browser tabs or clients from building
an unbounded backlog of large local files.

The engine validates processing profile, supported language, file extension, decoded
audio duration, maximum size, and maximum duration before queuing GPU work. Invalid
uploads are removed immediately. The API is bound to loopback, validates the Host header,
rejects browser origins outside localhost, and marks responses as non-cacheable. Each
engine process also creates a random request token. The local interface reads it from the
health endpoint and supplies it through `X-Lyricwave-Token` for POST and DELETE requests;
cross-origin pages cannot read the token or submit mutation requests without it.

## Evaluate model changes

Use exported word-timing JSON to compare Fast, Recommended, Best quality, or custom model runs
against a private reference without committing songs or lyrics:

```bash
npm run benchmark:lyrics -- reference.json fast.json recommended.json best.json
npm run benchmark:lyrics -- reference.json recommended=recommended.json best=best.json --json
```

Reference and candidate files may use lyricwave's exported `lines` structure or a
top-level `words` array. Timing exports include the selected profile, language, device,
Demucs model, actual transcription backend/model, actual alignment model, and requested
alignment checkpoint. Text-only references still produce WER/CER; timing metrics are
calculated only for exact aligned words where both files contain start and end times.
This makes model or threshold changes measurable on a representative private corpus
without committing any songs, lyrics, or generated outputs.

## Verify

```bash
npm run audit:production
npm run audit:dev:check
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
