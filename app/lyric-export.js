export const LYRIC_TIMING_SCHEMA = "lyricwave.word-timings.v1";

/**
 * Build a stable, versioned lyric timing export without mutating UI state.
 *
 * @param {{
 *   title: string,
 *   artist: string,
 *   duration: number,
 *   lines: Array<object>,
 *   processing?: object | null,
 *   exportedAt?: string,
 * }} input
 */
export function buildTimingExport({
  title,
  artist,
  duration,
  lines,
  processing = null,
  exportedAt = new Date().toISOString(),
}) {
  return {
    schema: LYRIC_TIMING_SCHEMA,
    title,
    artist,
    duration,
    generatedOnDevice: true,
    exportedAt,
    processing,
    lines,
  };
}
