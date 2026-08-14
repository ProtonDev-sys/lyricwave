import assert from "node:assert/strict";
import test from "node:test";

import { wordFillAt } from "../app/lyric-timing.js";

test("moves continuously between acoustic character landmarks", () => {
  const word = {
    start: 1,
    end: 1.5,
    timing: [
      { start: 1.04, end: 1.08, fill: 0.25 },
      { start: 1.16, end: 1.2, fill: 0.5 },
      { start: 1.28, end: 1.32, fill: 0.75 },
      { start: 1.4, end: 1.44, fill: 1 },
    ],
  };

  assert.ok(Math.abs(wordFillAt(word, 1.14) - 0.375) < 1e-9);
  const frameSamples = Array.from({ length: 12 }, (_, index) =>
    wordFillAt(word, 1.08 + index / 60),
  );
  assert.ok(new Set(frameSamples).size > 9);
});

test("holds the lyric fill only across an acoustically marked pause", () => {
  const word = {
    start: 1,
    end: 2.2,
    timing: [
      { start: 1.05, end: 1.2, fill: 0.3 },
      { start: 1.8, end: 2.1, fill: 1, pause_before: true },
    ],
  };

  assert.equal(wordFillAt(word, 1.5), 0.3);
  assert.ok(wordFillAt(word, 1.9) > 0.3);
  assert.ok(wordFillAt(word, 1.9) < 1);
});

test("falls back to a linear fill when acoustic character spans are absent", () => {
  assert.equal(wordFillAt({ start: 2, end: 4 }, 3), 0.5);
});
