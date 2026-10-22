from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return text.replace(old, new, 1)


def patch_page() -> None:
    path = ROOT / "app" / "page.tsx"
    text = path.read_text(encoding="utf-8")
    if 'className="control-panel"' in text and 'No audio selected' in text:
        return

    icon_start = text.index('import {\n', text.index('from "react";') + len('from "react";'))
    icon_end = text.index('} from "lucide-react";', icon_start) + len('} from "lucide-react";')
    text = text[:icon_start] + '''import {
  Check,
  Download,
  Headphones,
  LoaderCircle,
  Maximize2,
  Pause,
  Play,
  RotateCcw,
  SkipBack,
  SkipForward,
  Upload,
  Volume1,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";''' + text[icon_end:]
    text = text.replace('const [language, setLanguage] = useState("english");', 'const [language, setLanguage] = useState("auto");')
    text = text.replace('const [quality, setQuality] = useState<"accurate" | "fast">("fast");', 'const [quality, setQuality] = useState<"accurate" | "fast">("accurate");')
    text = text.replace('  const [showPrivacy, setShowPrivacy] = useState(false);\n', '')
    text = text.replace('useState("Ready when you are")', 'useState("Idle")')
    text = text.replace('setStatus("Ready when you are");', 'setStatus("Idle");')

    steps_start = text.index('  const steps = [')
    steps_end = text.index('  const renderVolumeIcon', steps_start)
    text = text[:steps_start] + '''  const steps = [
    { label: "Upload", rank: 1 },
    { label: "Separate", rank: 2 },
    { label: "Transcribe", rank: 3 },
  ];

''' + text[steps_end:]

    return_start = text.index('  return (', text.index('  const renderVolumeIcon'))
    text = text[:return_start] + '''  return (
    <main className={`app-shell ${file ? "has-track" : "is-empty"}`}>
      <header className="site-header">
        <button className="brand" type="button" onClick={reset} aria-label="Reset lyricwave">
          lyricwave
        </button>
        <div className={`engine-status engine-${engineState}`} title="Local engine at 127.0.0.1:8008">
          <span aria-hidden="true" />
          {engineState === "online"
            ? "engine ready"
            : engineState === "checking"
              ? "checking engine"
              : "engine offline"}
        </div>
      </header>

      <div className="workspace">
        <aside className="control-panel">
          <section className="control-section">
            <div className="section-heading">
              <span>input</span>
              {file && (
                <button type="button" className="text-button" onClick={reset}>
                  clear
                </button>
              )}
            </div>
            <div
              className={`drop-zone ${dragging ? "is-dragging" : ""}`}
              onDragEnter={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={(event) => {
                if (!event.currentTarget.contains(event.relatedTarget as Node)) setDragging(false);
              }}
              onDrop={handleDrop}
            >
              <div className="file-copy">
                <strong>{file ? track.title : "No audio selected"}</strong>
                <span>
                  {file
                    ? `${formatFileSize(file.size)} · ${duration ? formatTime(duration) : "reading"}`
                    : "MP3, WAV, FLAC, M4A, AAC, OGG or WebM · 500 MB max"}
                </span>
              </div>
              <button className="primary-button" type="button" onClick={() => inputRef.current?.click()}>
                <Upload size={14} /> {file ? "Replace" : "Choose audio"}
              </button>
            </div>
          </section>

          <section className="control-section settings-section">
            <label className="select-control">
              <span>language</span>
              <select value={language} onChange={(event) => setLanguage(event.target.value)}>
                {languageOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <fieldset className="model-control">
              <legend>model</legend>
              <button
                type="button"
                className={quality === "fast" ? "is-active" : ""}
                onClick={() => setQuality("fast")}
              >
                <strong>Fast</strong>
                <span>large-v3-turbo</span>
              </button>
              <button
                type="button"
                className={quality === "accurate" ? "is-active" : ""}
                onClick={() => setQuality("accurate")}
              >
                <strong>Best</strong>
                <span>large-v3</span>
              </button>
            </fieldset>
          </section>

          {file && (
            <section className="control-section progress-section" aria-live="polite">
              <div className="section-heading">
                <span>status</span>
                <strong>{Math.round(progress)}%</strong>
              </div>
              <div
                className="progress-track"
                role="progressbar"
                aria-label="Processing progress"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(progress)}
              >
                <span style={{ width: `${progress}%` }} />
              </div>
              <p className="status-copy">{stage === "error" ? error : status}</p>
              <ol className="steps-list">
                {steps.map((step) => {
                  const complete = stageRank > step.rank;
                  const active = stageRank === step.rank;
                  return (
                    <li key={step.label} className={`${complete ? "is-complete" : ""} ${active ? "is-active" : ""}`}>
                      <span className="step-state">
                        {complete ? <Check size={12} /> : active ? <LoaderCircle className="spin" size={12} /> : null}
                      </span>
                      {step.label}
                    </li>
                  );
                })}
              </ol>
              {engineDetail && <p className="engine-detail">{engineDetail}</p>}
              {stage === "error" && (
                <button className="secondary-button" type="button" onClick={retry}>
                  <RotateCcw size={13} /> Retry
                </button>
              )}
              {stage === "complete" && (
                <div className="export-row">
                  <button type="button" onClick={() => exportLyrics("lrc")}>
                    <Download size={13} /> LRC
                  </button>
                  <button type="button" onClick={() => exportLyrics("json")}>
                    <Download size={13} /> JSON
                  </button>
                </div>
              )}
            </section>
          )}
        </aside>

        <section className="lyrics-stage" ref={lyricsRef} aria-label="Lyrics">
          <div className="lyrics-toolbar">
            <div>
              <span>lyrics</span>
              {stage === "complete" && <small>{wordTimeline.length} words · {lines.length} lines</small>}
            </div>
            <button type="button" onClick={requestFullscreen} aria-label="Toggle fullscreen lyrics">
              <Maximize2 size={14} />
            </button>
          </div>
          <div className={`lyrics-scroll ${stage === "complete" ? "has-lyrics" : ""}`} ref={lyricsScrollRef}>
            {stage === "complete" ? (
              <LyricsLines
                lines={lines}
                lineWordOffsets={lineWordOffsets}
                lineRefs={lineRefs}
                wordRefs={wordRefs}
                directScrollLineRef={directScrollLineRef}
                onSeek={seekTo}
                onScrollToLine={scrollToFocusedLine}
              />
            ) : (
              <div className="empty-state">
                <strong>
                  {stage === "uploading"
                    ? "Uploading"
                    : stage === "separating"
                      ? "Separating vocals"
                      : stage === "transcribing"
                        ? "Transcribing"
                        : stage === "error"
                          ? "Processing failed"
                          : "No lyrics"}
                </strong>
                <span>{stage === "error" ? error : status}</span>
              </div>
            )}
          </div>
        </section>
      </div>

      {file && (
        <footer className="player">
          <div className="player-track">
            <strong>{track.title}</strong>
            <span>{track.artist}</span>
          </div>
          <div className="transport">
            <div className="transport-buttons">
              <button type="button" onClick={() => seekTo(currentTime - 10)} aria-label="Back 10 seconds">
                <SkipBack size={16} />
              </button>
              <button className="play-button" type="button" onClick={() => void togglePlayback()} aria-label={isPlaying ? "Pause" : "Play"}>
                {isPlaying ? <Pause size={17} /> : <Play size={17} />}
              </button>
              <button type="button" onClick={() => seekTo(currentTime + 10)} aria-label="Forward 10 seconds">
                <SkipForward size={16} />
              </button>
            </div>
            <div className="timeline-row">
              <span>{formatTime(currentTime)}</span>
              <input
                className="timeline"
                type="range"
                min="0"
                max={duration || 0}
                step="0.01"
                value={Math.min(currentTime, duration || 0)}
                onChange={(event) => seekTo(Number(event.target.value))}
                aria-label="Seek through song"
                style={{ "--seek": `${duration ? (currentTime / duration) * 100 : 0}%` } as CSSProperties}
              />
              <span>{formatTime(duration)}</span>
            </div>
          </div>
          <div className="player-tools">
            <div className="source-switch" role="group" aria-label="Playback source">
              <button type="button" className={playbackMode === "mix" ? "is-active" : ""} onClick={() => changePlaybackMode("mix")}>Mix</button>
              <button
                type="button"
                className={playbackMode === "vocals" ? "is-active" : ""}
                onClick={() => changePlaybackMode("vocals")}
                disabled={!vocalUrl}
              >
                <Headphones size={12} /> Vocal
              </button>
            </div>
            <div className="volume-control">
              <button type="button" onClick={toggleMute} aria-label={isMuted ? "Unmute" : "Mute"}>
                {renderVolumeIcon()}
              </button>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={isMuted ? 0 : volume}
                onChange={(event) => updateVolume(Number(event.target.value))}
                aria-label="Volume"
                style={{ "--volume": `${(isMuted ? 0 : volume) * 100}%` } as CSSProperties}
              />
            </div>
          </div>
        </footer>
      )}

      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept="audio/*,.mp3,.wav,.flac,.m4a,.aac,.ogg,.webm"
        onChange={handleFileInput}
      />
      {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
      <audio
        ref={audioRef}
        preload="metadata"
        onLoadedMetadata={(event) => {
          setDuration(event.currentTarget.duration || 0);
          event.currentTarget.volume = volume;
        }}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
        onTimeUpdate={(event) => {
          if (!isPlaying) {
            setCurrentTime(event.currentTarget.currentTime);
            syncLyricsAt(event.currentTarget.currentTime, true);
          }
        }}
      />
      {error && stage !== "error" && (
        <div className="toast" role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => setError("")} aria-label="Dismiss message">
            <X size={14} />
          </button>
        </div>
      )}
    </main>
  );
}
'''
    path.write_text(text, encoding="utf-8")


def write_css() -> None:
    (ROOT / "app" / "globals.css").write_text(
        ''':root {
  --background: #f2f1ec;
  --surface: #f8f7f2;
  --ink: #111111;
  --muted: #686864;
  --line: #c9c8c1;
  --line-strong: #1b1b1b;
  --active: #2856d8;
  --danger: #b3261e;
}

* { box-sizing: border-box; }
html, body { margin: 0; min-height: 100%; background: var(--background); }
body { color: var(--ink); font-family: var(--font-geist-sans, Arial), sans-serif; line-height: 1.35; }
button, input, select { font: inherit; }
button { color: inherit; }
button:focus-visible, input:focus-visible, select:focus-visible { outline: 2px solid var(--active); outline-offset: 2px; }

.app-shell { min-height: 100dvh; display: grid; grid-template-rows: 52px minmax(0, 1fr) auto; background: var(--background); }
.site-header { display: flex; align-items: center; justify-content: space-between; padding: 0 18px; border-bottom: 1px solid var(--line-strong); }
.brand { padding: 0; border: 0; background: none; font-size: 15px; font-weight: 700; letter-spacing: -0.03em; cursor: pointer; }
.engine-status { display: inline-flex; align-items: center; gap: 7px; color: var(--muted); font-family: var(--font-geist-mono, monospace); font-size: 11px; }
.engine-status > span { width: 7px; height: 7px; border: 1px solid currentColor; border-radius: 50%; }
.engine-online > span { background: #21824a; border-color: #21824a; }
.engine-offline { color: var(--danger); }
.workspace { min-height: 0; display: grid; grid-template-columns: 330px minmax(0, 1fr); }
.control-panel { min-height: 0; overflow: auto; border-right: 1px solid var(--line-strong); background: var(--surface); }
.control-section { padding: 18px; border-bottom: 1px solid var(--line); }
.section-heading { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; font-family: var(--font-geist-mono, monospace); font-size: 11px; text-transform: lowercase; }
.section-heading > strong { font-size: 11px; font-weight: 500; }
.text-button { padding: 0; border: 0; background: none; color: var(--muted); font-size: 11px; cursor: pointer; text-decoration: underline; text-underline-offset: 3px; }
.drop-zone { min-height: 128px; display: flex; flex-direction: column; align-items: flex-start; justify-content: space-between; gap: 20px; padding: 14px; border: 1px dashed #8f8e88; background: var(--background); }
.drop-zone.is-dragging { border-style: solid; border-color: var(--active); background: #edf1ff; }
.file-copy { min-width: 0; display: grid; gap: 5px; }
.file-copy strong { overflow-wrap: anywhere; font-size: 14px; font-weight: 600; }
.file-copy span { color: var(--muted); font-size: 11px; }
.primary-button, .secondary-button { min-height: 32px; display: inline-flex; align-items: center; justify-content: center; gap: 7px; padding: 0 12px; border: 1px solid var(--line-strong); border-radius: 0; background: var(--ink); color: #fff; font-size: 12px; font-weight: 600; cursor: pointer; }
.primary-button:hover, .secondary-button:hover { background: #2e2e2e; }
.secondary-button { background: transparent; color: var(--ink); }
.settings-section { display: grid; gap: 18px; }
.select-control { display: grid; grid-template-columns: 80px minmax(0, 1fr); align-items: center; gap: 10px; }
.select-control > span, .model-control legend { color: var(--muted); font-family: var(--font-geist-mono, monospace); font-size: 11px; }
.select-control select { width: 100%; height: 34px; padding: 0 9px; border: 1px solid var(--line); border-radius: 0; background: var(--background); color: var(--ink); font-size: 12px; }
.model-control { display: grid; grid-template-columns: 80px 1fr 1fr; gap: 8px; min-width: 0; margin: 0; padding: 0; border: 0; }
.model-control legend { float: left; width: 80px; align-self: center; }
.model-control button { min-width: 0; display: grid; gap: 2px; padding: 8px 9px; border: 1px solid var(--line); border-radius: 0; background: transparent; text-align: left; cursor: pointer; }
.model-control button.is-active { border-color: var(--line-strong); background: var(--ink); color: #fff; }
.model-control strong { font-size: 11px; }
.model-control span { overflow: hidden; color: var(--muted); font-family: var(--font-geist-mono, monospace); font-size: 9px; text-overflow: ellipsis; }
.model-control .is-active span { color: #cfcfcf; }
.progress-track { height: 4px; overflow: hidden; background: #d9d8d1; }
.progress-track > span { display: block; height: 100%; background: var(--ink); transition: width 180ms linear; }
.status-copy { min-height: 32px; margin: 10px 0 12px; color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
.steps-list { display: grid; gap: 6px; margin: 0; padding: 0; list-style: none; }
.steps-list li { display: grid; grid-template-columns: 18px 1fr; align-items: center; color: #9a9994; font-size: 11px; }
.steps-list li.is-active { color: var(--ink); font-weight: 600; }
.steps-list li.is-complete { color: #3f6d4d; }
.step-state { width: 14px; height: 14px; display: grid; place-items: center; }
.engine-detail { margin: 12px 0 0; color: var(--muted); font-family: var(--font-geist-mono, monospace); font-size: 9px; overflow-wrap: anywhere; }
.export-row { display: flex; gap: 7px; margin-top: 14px; }
.export-row button { display: inline-flex; align-items: center; gap: 6px; height: 30px; padding: 0 9px; border: 1px solid var(--line); background: transparent; font-size: 11px; cursor: pointer; }
.lyrics-stage { min-width: 0; min-height: 0; display: grid; grid-template-rows: 42px minmax(0, 1fr); background: var(--background); }
.lyrics-toolbar { display: flex; align-items: center; justify-content: space-between; padding: 0 14px; border-bottom: 1px solid var(--line); font-family: var(--font-geist-mono, monospace); font-size: 11px; }
.lyrics-toolbar > div { display: flex; align-items: center; gap: 12px; }
.lyrics-toolbar small { color: var(--muted); font-size: 9px; }
.lyrics-toolbar button { width: 28px; height: 28px; display: grid; place-items: center; border: 0; background: transparent; cursor: pointer; }
.lyrics-scroll { min-height: 0; overflow: auto; position: relative; scrollbar-width: thin; }
.empty-state { position: absolute; inset: 0; display: grid; place-content: center; gap: 6px; padding: 24px; text-align: center; }
.empty-state strong { font-size: 15px; font-weight: 600; }
.empty-state span { max-width: 440px; color: var(--muted); font-size: 11px; }
.lyrics-lines { padding: var(--lyrics-top-space, 34vh) 6vw var(--lyrics-bottom-space, 50vh); }
.lyric-line { display: block; width: 100%; margin: 0; padding: 5px 0; border: 0; background: transparent; color: #b4b3ae; font-size: clamp(25px, 3.5vw, 54px); font-weight: 650; line-height: 1.08; letter-spacing: -0.045em; text-align: left; cursor: pointer; transition: color 120ms linear, opacity 120ms linear; }
.lyric-line.is-past { color: #9a9994; }
.lyric-line.is-active { color: var(--ink); }
.lyric-line.is-adlib { padding-left: 12%; font-size: clamp(16px, 2vw, 30px); font-weight: 500; }
.lyric-word { position: relative; display: inline-block; margin-right: 0.22em; color: inherit; }
.lyric-word-label { color: inherit; }
.lyric-word.is-active .lyric-word-label { color: transparent; }
.lyric-word.is-active { background: linear-gradient(90deg, var(--ink) var(--word-fill, 0%), #b4b3ae var(--word-fill, 0%)); background-clip: text; -webkit-background-clip: text; color: transparent; }
.lyric-word.is-past { color: var(--ink); }
.player { display: grid; grid-template-columns: minmax(150px, 1fr) minmax(320px, 2fr) minmax(200px, 1fr); align-items: center; gap: 18px; min-height: 76px; padding: 10px 18px; border-top: 1px solid var(--line-strong); background: var(--surface); }
.player-track { min-width: 0; display: grid; gap: 2px; }
.player-track strong, .player-track span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.player-track strong { font-size: 12px; }
.player-track span { color: var(--muted); font-size: 10px; }
.transport { display: grid; gap: 7px; }
.transport-buttons { display: flex; justify-content: center; align-items: center; gap: 5px; }
.transport-buttons button, .volume-control button { width: 30px; height: 30px; display: grid; place-items: center; border: 0; background: transparent; cursor: pointer; }
.transport-buttons .play-button { border: 1px solid var(--line-strong); background: var(--ink); color: #fff; }
.timeline-row { display: grid; grid-template-columns: 36px minmax(0, 1fr) 36px; align-items: center; gap: 8px; color: var(--muted); font-family: var(--font-geist-mono, monospace); font-size: 9px; }
.timeline-row > span:last-child { text-align: right; }
input[type="range"] { appearance: none; height: 3px; margin: 0; background: linear-gradient(90deg, var(--ink) var(--seek, var(--volume, 0%)), #d4d3cc var(--seek, var(--volume, 0%))); cursor: pointer; }
input[type="range"]::-webkit-slider-thumb { appearance: none; width: 9px; height: 9px; border: 1px solid var(--ink); border-radius: 50%; background: var(--surface); }
.player-tools { display: flex; justify-content: flex-end; align-items: center; gap: 14px; }
.source-switch { display: flex; border: 1px solid var(--line); }
.source-switch button { height: 28px; display: inline-flex; align-items: center; gap: 5px; padding: 0 9px; border: 0; background: transparent; font-size: 10px; cursor: pointer; }
.source-switch button + button { border-left: 1px solid var(--line); }
.source-switch button.is-active { background: var(--ink); color: #fff; }
.source-switch button:disabled { opacity: 0.35; cursor: default; }
.volume-control { display: grid; grid-template-columns: 28px 76px; align-items: center; }
.toast { position: fixed; right: 16px; bottom: 92px; display: flex; align-items: center; gap: 12px; max-width: 440px; padding: 11px 12px; border: 1px solid var(--line-strong); background: var(--surface); box-shadow: 4px 4px 0 var(--ink); font-size: 11px; }
.toast button { display: grid; place-items: center; padding: 0; border: 0; background: none; cursor: pointer; }
.spin { animation: spin 0.85s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.visually-hidden { position: fixed; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
@media (max-width: 800px) {
  .app-shell { grid-template-rows: 48px auto auto; min-height: 100dvh; }
  .workspace { display: block; }
  .control-panel { border-right: 0; border-bottom: 1px solid var(--line-strong); overflow: visible; }
  .control-section { padding: 14px; }
  .app-shell.is-empty .lyrics-stage { min-height: 320px; }
  .lyrics-stage { min-height: 55vh; }
  .model-control { grid-template-columns: 68px 1fr 1fr; }
  .model-control legend { width: 68px; }
  .player { position: sticky; bottom: 0; grid-template-columns: 1fr; gap: 8px; padding: 10px 14px; }
  .player-track { display: none; }
  .player-tools { justify-content: space-between; }
  .lyrics-lines { padding-left: 18px; padding-right: 18px; }
  .lyric-line { font-size: clamp(24px, 9vw, 40px); }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; }
}
''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    patch_page()
    write_css()
