import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    {
      ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the lyricwave workspace", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>lyricwave — Your song, word for word<\/title>/i);
  assert.match(html, /Turn any song into live lyrics/);
  assert.match(html, /Checking local engine/);
  assert.match(html, /Drop your song here/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("uses the modular localhost inference engine and word-level results", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const timeline = await readFile(new URL("../app/lyric-timeline.js", import.meta.url), "utf8");
  const backend = await readFile(new URL("../backend/server.py", import.meta.url), "utf8");
  const config = await readFile(new URL("../backend/config.py", import.meta.url), "utf8");
  const pipeline = await readFile(
    new URL("../backend/inference_pipeline.py", import.meta.url),
    "utf8",
  );
  const alignment = await readFile(
    new URL("../backend/ctc_alignment.py", import.meta.url),
    "utf8",
  );
  const inferenceWorker = await readFile(
    new URL("../backend/inference_worker.py", import.meta.url),
    "utf8",
  );

  assert.match(page, /http:\/\/127\.0\.0\.1:8008/);
  assert.match(page, /FormData\(\)/);
  assert.match(page, /URL\.createObjectURL/);
  assert.match(page, /memo\(function LyricsLines/);
  assert.match(page, /findActiveIntervalIndexes/);
  assert.match(page, /RetryablePollingError/);
  assert.doesNotMatch(page, /wordTimeline\s*\.map\(/);
  assert.match(timeline, /buildPrefixMaxEnds/);
  assert.match(timeline, /pollingRetryDelay/);
  assert.match(config, /htdemucs_ft/);
  assert.match(config, /whisper-large-v3/);
  assert.match(pipeline, /return_timestamps=True/);
  assert.doesNotMatch(pipeline, /return_timestamps="word"/);
  assert.match(alignment, /AutoModelForCTC/);
  assert.match(backend, /backend\.inference_worker/);
  assert.match(inferenceWorker, /set_per_process_memory_fraction\(_vram_fraction\(\)/);
  assert.doesNotMatch(backend, /AutoModelForSpeechSeq2Seq|AutoModelForCTC/);
  assert.doesNotMatch(page, /@browserai|onnxruntime-web|@huggingface\/transformers/);
});
