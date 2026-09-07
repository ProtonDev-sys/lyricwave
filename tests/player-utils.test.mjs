import assert from "node:assert/strict";
import test from "node:test";
import { buildLrc, formatLrcTimestamp, lyricClockTime, parseTrackName } from "../app/player-utils.js";
import { buildTimingExport } from "../app/lyric-export.js";

test("filename metadata follows Artist - Title without inventing an artist", () => {
  assert.deepEqual(parseTrackName("Our Band - Night_song.flac"), { artist: "Our Band", title: "Night song" });
  assert.deepEqual(parseTrackName("歌手 — 曲名.wav"), { artist: "歌手", title: "曲名" });
  assert.deepEqual(parseTrackName("Night_song.mp3"), { artist: "Local audio", title: "Night song" });
  assert.deepEqual(parseTrackName(".wav"), { artist: "Local audio", title: "Untitled track" });
});

test("positive calibration delays lyrics and zero introduces no hidden bias", () => {
  assert.equal(lyricClockTime(12.5), 12.5);
  assert.equal(lyricClockTime(12.5, 200), 12.3);
  assert.equal(lyricClockTime(12.5, -200), 12.7);
  assert.equal(lyricClockTime(0.1, 200), 0);
  assert.equal(lyricClockTime(12.5, 9000), 10.5);
  assert.equal(lyricClockTime(NaN, Infinity), 0);
});

test("LRC carries rounding into minutes and hours without invalid 60-second fields", () => {
  for (const [time, expected] of [[0, "[00:00.00]"], [59.999, "[01:00.00]"], [3599.999, "[60:00.00]"], [61.234, "[01:01.23]"], [-1, "[00:00.00]"], [NaN, "[00:00.00]"]]) {
    assert.equal(formatLrcTimestamp(time), expected);
  }
});

test("LRC calibration shifts timestamps, not the original measurements", () => {
  const lines = [{ start: 1, words: [{ text: "One", start: 1, end: 2, timing_source: "estimated" }] }];
  const before = structuredClone(lines);
  assert.equal(buildLrc(lines, 250), "[00:01.25]One");
  assert.equal(buildLrc(lines, -2000), "[00:00.00]One");
  const exported = buildTimingExport({ title: "Test", artist: "Local", duration: 10, lines, processing: null, playbackOffsetMs: 250 });
  assert.deepEqual(lines, before);
  assert.equal(exported.playbackOffsetMs, 250);
  assert.equal(exported.lines[0].words[0].start, 1);
  assert.equal(exported.lines[0].words[0].timing_source, "estimated");
});
