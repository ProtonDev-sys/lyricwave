# lyricwave

Turn a local song into word-synced live lyrics without uploading the audio.

## What it does

- decodes MP3, WAV, FLAC, M4A, AAC, OGG, and WebM audio in the browser
- isolates a vocal stem with HTDemucs through WebGPU, with a lighter local fallback
- transcribes the vocal with multilingual Whisper and requests word timestamps
- renders karaoke-style live lyrics with click-to-seek lines
- plays either the original mix or the isolated vocal with seek and volume controls
- exports line-timed LRC or exact word timing JSON

The audio file stays on the device. The first run downloads the AI model files from
Hugging Face; browsers can reuse cached model data on later runs.

## Run locally

Requires Node.js 22.13 or newer.

```bash
npm install
npm run dev
```

Open `http://localhost:3000`. For best performance, use a current Chromium-based
browser with WebGPU enabled.

## Verify

```bash
npm run build
npm test
```

Model inference begins only after a user selects an audio file. Long tracks and
devices without WebGPU will take considerably longer to process.
