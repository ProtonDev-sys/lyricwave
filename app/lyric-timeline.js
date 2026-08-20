/**
 * Compute the first flat-word index for every lyric line in one pass.
 *
 * @param {Array<{words: unknown[]}>} lines
 * @returns {number[]}
 */
export function buildLineWordOffsets(lines) {
  const offsets = new Array(lines.length);
  let offset = 0;
  for (let index = 0; index < lines.length; index += 1) {
    offsets[index] = offset;
    offset += lines[index].words.length;
  }
  return offsets;
}

/**
 * Prefix maximums let an interval query stop as soon as no earlier interval can
 * still overlap the requested time. Lyric intervals are short, so lookup is
 * effectively O(log n + active overlaps) rather than a full timeline scan.
 *
 * @param {Array<{end: number}>} entries
 * @returns {number[]}
 */
export function buildPrefixMaxEnds(entries) {
  const prefixMaxEnds = new Array(entries.length);
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < entries.length; index += 1) {
    maximum = Math.max(maximum, Number(entries[index].end));
    prefixMaxEnds[index] = maximum;
  }
  return prefixMaxEnds;
}

/**
 * Return the final entry whose start is not later than the requested time.
 * Entries must be sorted by start ascending.
 *
 * @param {Array<{start: number}>} entries
 * @param {number} time
 */
export function findLastStartedIndex(entries, time) {
  let low = 0;
  let high = entries.length - 1;
  let candidate = -1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (entries[middle].start <= time) {
      candidate = middle;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return candidate;
}

/**
 * Return indexes of every interval active at `time`.
 *
 * @param {Array<{start: number, end: number}>} entries
 * @param {number[]} prefixMaxEnds
 * @param {number} time
 */
export function findActiveIntervalIndexes(entries, prefixMaxEnds, time) {
  const active = [];
  let index = findLastStartedIndex(entries, time);
  while (index >= 0 && prefixMaxEnds[index] > time) {
    const entry = entries[index];
    if (entry.start <= time && time < entry.end) active.push(index);
    index -= 1;
  }
  active.reverse();
  return active;
}

/**
 * Back off polling after a transient localhost interruption without abandoning
 * an expensive GPU job. The delay is capped so recovery remains responsive.
 *
 * @param {number} failureCount
 * @param {number} [baseDelay]
 * @param {number} [maximumDelay]
 */
export function pollingRetryDelay(failureCount, baseDelay = 650, maximumDelay = 5_000) {
  const exponent = Math.max(0, Math.min(8, Math.trunc(failureCount) - 1));
  return Math.min(maximumDelay, baseDelay * 2 ** exponent);
}
