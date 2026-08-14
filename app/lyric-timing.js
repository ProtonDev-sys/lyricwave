/**
 * Return the visible fill for a word at an audio-clock timestamp.
 *
 * Character-level CTC spans become acoustic landmarks for a continuous sweep.
 * A backend pause marker is the only thing that deliberately stops that sweep,
 * so the UI stays fluid through held notes without losing split-syllable timing.
 * Older jobs and languages without CTC timing fall back to a linear sweep.
 *
 * @param {{start: number, end: number, timing?: Array<{start: number, end: number, fill: number, pause_before?: boolean}>}} word
 * @param {number} displayTime
 */
export function wordFillAt(word, displayTime) {
  if (displayTime <= word.start) return 0;
  if (displayTime >= word.end) return 1;

  const acousticTiming = word.timing
    ?.filter(
      (unit) =>
        Number.isFinite(unit.start) &&
        Number.isFinite(unit.end) &&
        Number.isFinite(unit.fill) &&
        unit.end > unit.start,
    )
    .sort((left, right) => left.end - right.end);
  if (!acousticTiming?.length) {
    return Math.max(
      0,
      Math.min(1, (displayTime - word.start) / Math.max(0.05, word.end - word.start)),
    );
  }

  let previousTime = word.start;
  let completedFill = 0;
  for (const unit of acousticTiming) {
    const nextFill = Math.max(completedFill, Math.min(1, unit.fill));
    const landmarkTime = Math.max(previousTime + 0.001, Math.min(word.end, unit.end));
    if (displayTime <= landmarkTime) {
      let sweepStart = previousTime;
      if (unit.pause_before) {
        // Hold the completed syllable through an acoustically silent gap, then
        // give the resumed syllable enough time to move rather than flash on.
        sweepStart = Math.max(previousTime, Math.min(unit.start, landmarkTime) - 0.065);
      }
      if (displayTime <= sweepStart) return completedFill;
      const progress = Math.max(
        0,
        Math.min(1, (displayTime - sweepStart) / Math.max(0.016, landmarkTime - sweepStart)),
      );
      const easedProgress = unit.pause_before
        ? progress * progress * (3 - 2 * progress)
        : progress;
      return completedFill + (nextFill - completedFill) * easedProgress;
    }
    previousTime = landmarkTime;
    completedFill = nextFill;
  }
  return completedFill;
}
