import assert from "node:assert/strict";
import test from "node:test";

import {
  LYRIC_TIMING_SCHEMA,
  buildTimingExport,
} from "../app/lyric-export.js";


test("builds a versioned export with reproducible processing metadata", () => {
  const lines = [
    {
      id: "line-0",
      start: 1,
      end: 1.4,
      words: [{ text: "hello", start: 1, end: 1.4 }],
    },
  ];
  const processing = {
    jobId: "abc123",
    createdAt: "2026-08-20T18:00:00Z",
    quality: "accurate",
    language: "english",
    device: "Test GPU",
    separationModel: "htdemucs_ft",
    transcriptionModel: "openai/whisper-large-v3",
    alignmentModelRequested: "facebook/wav2vec2-large-960h-lv60-self",
  };

  const payload = buildTimingExport({
    title: "Track",
    artist: "Artist",
    duration: 12.5,
    lines,
    processing,
    exportedAt: "2026-08-20T19:00:00Z",
  });

  assert.equal(payload.schema, LYRIC_TIMING_SCHEMA);
  assert.equal(payload.generatedOnDevice, true);
  assert.equal(payload.exportedAt, "2026-08-20T19:00:00Z");
  assert.deepEqual(payload.processing, processing);
  assert.strictEqual(payload.lines, lines);
});

test("keeps the processing field explicit when metadata is unavailable", () => {
  const payload = buildTimingExport({
    title: "Track",
    artist: "Local audio",
    duration: 0,
    lines: [],
    exportedAt: "2026-08-20T19:00:00Z",
  });

  assert.equal(payload.processing, null);
  assert.deepEqual(payload.lines, []);
});
