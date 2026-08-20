import assert from "node:assert/strict";
import test from "node:test";

import {
  buildLineWordOffsets,
  buildPrefixMaxEnds,
  findActiveIntervalIndexes,
  findLastStartedIndex,
  pollingRetryDelay,
} from "../app/lyric-timeline.js";

test("computes flat word offsets in one cumulative pass", () => {
  assert.deepEqual(
    buildLineWordOffsets([
      { words: [{}, {}] },
      { words: [{}] },
      { words: [{}, {}, {}] },
    ]),
    [0, 2, 3],
  );
});

test("finds the latest started interval with binary search", () => {
  const entries = [
    { start: 1, end: 2 },
    { start: 3, end: 4 },
    { start: 3, end: 5 },
    { start: 8, end: 9 },
  ];
  assert.equal(findLastStartedIndex(entries, 0.9), -1);
  assert.equal(findLastStartedIndex(entries, 1), 0);
  assert.equal(findLastStartedIndex(entries, 3), 2);
  assert.equal(findLastStartedIndex(entries, 100), 3);
});

test("returns overlapping lead and ad-lib intervals without scanning the full history", () => {
  const entries = [
    { start: 0, end: 0.5 },
    { start: 1, end: 4 },
    { start: 1.5, end: 2.2 },
    { start: 1.8, end: 2.6 },
    { start: 9, end: 10 },
  ];
  const prefix = buildPrefixMaxEnds(entries);
  assert.deepEqual(findActiveIntervalIndexes(entries, prefix, 2), [1, 2, 3]);
  assert.deepEqual(findActiveIntervalIndexes(entries, prefix, 4), []);
  assert.deepEqual(findActiveIntervalIndexes(entries, prefix, 9.5), [4]);
});

test("treats interval endings as exclusive", () => {
  const entries = [
    { start: 1, end: 2 },
    { start: 2, end: 3 },
  ];
  const prefix = buildPrefixMaxEnds(entries);
  assert.deepEqual(findActiveIntervalIndexes(entries, prefix, 2), [1]);
});

test("caps localhost polling backoff", () => {
  assert.equal(pollingRetryDelay(1), 650);
  assert.equal(pollingRetryDelay(2), 1300);
  assert.equal(pollingRetryDelay(4), 5000);
  assert.equal(pollingRetryDelay(100), 5000);
});
