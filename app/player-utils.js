/** Use the conventional "Artist - Title" filename form, without guessing metadata. */
export function parseTrackName(fileName) {
  const clean = fileName.replace(/\.[^.]+$/, "").replace(/_+/g, " ").trim();
  const pieces = clean.split(/\s+[-–—]\s+/);
  if (pieces.length > 1) return { artist: pieces[0], title: pieces.slice(1).join(" — ") };
  return { artist: "Local audio", title: clean || "Untitled track" };
}

/** Positive offset delays the lyrics. Exported acoustic timings stay unmodified. */
export function lyricClockTime(audioTime, offsetMs = 0) {
  const time = Number.isFinite(audioTime) ? audioTime : 0;
  const offset = Number.isFinite(offsetMs) ? Math.max(-2000, Math.min(2000, offsetMs)) : 0;
  return Math.max(0, time - offset / 1000);
}

export function formatLrcTimestamp(seconds) {
  // Round before splitting minutes so 59.999 becomes 01:00.00, never 00:60.00.
  const ticks = Math.round(Math.max(0, Number.isFinite(seconds) ? seconds : 0) * 100);
  const minutes = String(Math.floor(ticks / 6000)).padStart(2, "0");
  const remainder = String(Math.floor(ticks / 100) % 60).padStart(2, "0");
  return `[${minutes}:${remainder}.${String(ticks % 100).padStart(2, "0")}]`;
}

export function buildLrc(lines, offsetMs = 0) {
  const offset = Number.isFinite(offsetMs) ? Math.max(-2000, Math.min(2000, offsetMs)) / 1000 : 0;
  return lines.map((line) =>
    `${formatLrcTimestamp(line.start + offset)}${line.words.map((word) => word.text).join(" ")}`,
  ).join("\n");
}
