"use client";

import {
  type CSSProperties,
  type ChangeEvent,
  type DragEvent,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  AudioLines,
  Check,
  Download,
  FileAudio,
  Headphones,
  LoaderCircle,
  LockKeyhole,
  Maximize2,
  Mic2,
  Pause,
  Play,
  RotateCcw,
  SkipBack,
  SkipForward,
  UploadCloud,
  Volume1,
  Volume2,
  VolumeX,
  Waves,
  X,
} from "lucide-react";
import {
  buildLineWordOffsets,
  buildPrefixMaxEnds,
  findActiveIntervalIndexes,
  findLastStartedIndex,
  pollingRetryDelay,
} from "./lyric-timeline.js";
import {
  LocalEngineRequestError,
  withRefreshedRequestToken,
} from "./local-engine.js";
import { wordFillAt } from "./lyric-timing.js";
type ProcessingStage =
  | "idle"
  | "uploading"
  | "separating"
  | "transcribing"
  | "complete"
  | "error";

type TimedWord = {
  text: string;
  start: number;
  end: number;
  timing?: Array<{
    start: number;
    end: number;
    fill: number;
    pause_before?: boolean;
  }>;
  kind?: "lead" | "adlib";
  phrase?: number;
};

type LyricLine = {
  id: string;
  start: number;
  end: number;
  words: TimedWord[];
  kind?: "lead" | "adlib";
};

type BackendStage =
  | "queued"
  | "separating"
  | "transcribing"
  | "complete"
  | "error"
  | "cancelled";

type BackendJob = {
  id: string;
  stage: BackendStage;
  progress: number;
  status: string;
  error?: string;
  duration: number;
  device?: string;
  vocal_url?: string | null;
  words?: TimedWord[];
  lines?: LyricLine[];
  separation_model?: string;
  transcription_model?: string;
};

type BackendHealth = {
  ok: boolean;
  ready: boolean;
  cuda: boolean;
  device: string;
  ffmpeg: boolean;
  ffprobe?: boolean;
  demucs: boolean;
  transformers: boolean;
  request_token: string;
  pending_jobs?: number;
  queue_capacity?: number;
};

const ACCEPTED_EXTENSIONS = ["mp3", "wav", "flac", "m4a", "aac", "ogg", "webm"];
const MAX_FILE_SIZE = 500 * 1024 * 1024;
const MAX_CONSECUTIVE_POLLING_FAILURES = 6;
const LYRIC_LOOKAHEAD_SECONDS = 0.055;
const PLAYER_CLOCK_INTERVAL_MS = 100;
const LOCAL_API_URL =
  process.env.NEXT_PUBLIC_LYRICWAVE_API_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8008";
const LOCAL_REQUEST_TOKEN_HEADER = "X-Lyricwave-Token";
const ENGINE_OFFLINE_MESSAGE =
  "The local engine is offline. Run `npm run setup:engine` once, then start the app with `npm run dev`.";

const languageOptions = [
  { value: "auto", label: "Auto-detect" },
  { value: "english", label: "English" },
  { value: "spanish", label: "Spanish" },
  { value: "french", label: "French" },
  { value: "german", label: "German" },
  { value: "italian", label: "Italian" },
  { value: "portuguese", label: "Portuguese" },
  { value: "japanese", label: "Japanese" },
  { value: "korean", label: "Korean" },
];

function formatTime(value: number) {
  if (!Number.isFinite(value) || value < 0) return "0:00";
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function formatFileSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(bytes > 10 * 1024 * 1024 ? 0 : 1)} MB`;
}

function parseTrackName(fileName: string) {
  const clean = fileName.replace(/\.[^.]+$/, "").replace(/[_]+/g, " ").trim();
  const pieces = clean.split(/\s+-\s+/);
  if (pieces.length > 1) {
    return { title: pieces[0], artist: pieces.slice(1).join(" — ") };
  }
  return { artist: "Local audio", title: clean || "Untitled track" };
}

function isAudioFile(file: File) {
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  return ACCEPTED_EXTENSIONS.includes(extension);
}

function groupIntoLines(words: TimedWord[]): LyricLine[] {
  if (!words.length) return [];
  const groups: Array<{ kind: "lead" | "adlib"; words: TimedWord[] }> = [];

  for (const kind of ["lead", "adlib"] as const) {
    const stream = words
      .filter((word) => (word.kind ?? "lead") === kind)
      .sort((left, right) => left.start - right.start);
    let current: TimedWord[] = [];
    const maxWords = kind === "adlib" ? 7 : 12;
    const maxDuration = kind === "adlib" ? 5.2 : 7.2;
    const maximumGap = kind === "adlib" ? 0.9 : 0.52;

    for (const word of stream) {
      const previous = current[current.length - 1];
      const gap = previous ? word.start - previous.end : 0;
      const lineDuration = current.length ? word.end - current[0].start : 0;
      const phraseChanged = previous && word.phrase !== previous.phrase;
      const punctuationBreak = previous
        ? /[.!?…]$/.test(previous.text) && current.length >= 3
        : false;
      const phraseBreak =
        phraseChanged && (gap > 0.06 || current.length >= 4 || lineDuration >= 2.5);
      const shouldBreak =
        current.length > 0 &&
        (gap > maximumGap ||
          current.length >= maxWords ||
          lineDuration > maxDuration ||
          punctuationBreak ||
          phraseBreak);

      if (shouldBreak) {
        groups.push({ kind, words: current });
        current = [];
      }
      current.push(word);
    }
    if (current.length) groups.push({ kind, words: current });
  }
  groups.sort((left, right) => left.words[0].start - right.words[0].start);

  return groups.map(({ kind, words: lineWords }, index) => ({
    id: `line-${index}-${lineWords[0].start.toFixed(2)}`,
    start: lineWords[0].start,
    end: lineWords[lineWords.length - 1].end,
    words: lineWords,
    kind,
  }));
}

function downloadBlob(blob: Blob, fileName: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function apiMessage(payload: unknown, fallback: string) {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
  }
  return fallback;
}

async function readEngineHealth() {
  const response = await fetch(`${LOCAL_API_URL}/api/health`, {
    cache: "no-store",
    signal: AbortSignal.timeout(5_000),
  });
  const payload = (await response.json()) as BackendHealth | { detail?: string };
  if (!response.ok) {
    throw new Error(apiMessage(payload, `Local engine returned ${response.status}.`));
  }
  const health = payload as BackendHealth;
  if (!health.request_token) {
    throw new Error("The local engine did not return a request token.");
  }
  return health;
}

async function cancelLocalJob(jobId: string, requestToken: string) {
  const response = await fetch(`${LOCAL_API_URL}/api/jobs/${jobId}`, {
    method: "DELETE",
    keepalive: true,
    headers: { [LOCAL_REQUEST_TOKEN_HEADER]: requestToken },
  });
  if (response.ok) return;

  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    // A plain-text proxy error still carries a useful HTTP status.
  }
  throw new LocalEngineRequestError(
    apiMessage(payload, `The local engine returned ${response.status}.`),
    response.status,
  );
}

class RetryablePollingError extends Error {}

function isRetryablePollingError(error: unknown) {
  return (
    error instanceof RetryablePollingError ||
    error instanceof SyntaxError ||
    error instanceof TypeError ||
    (error instanceof DOMException &&
      ["AbortError", "NetworkError", "TimeoutError"].includes(error.name))
  );
}

async function fetchJobStatus(jobId: string) {
  const response = await fetch(`${LOCAL_API_URL}/api/jobs/${jobId}`, {
    cache: "no-store",
    signal: AbortSignal.timeout(10_000),
  });
  const payload = (await response.json()) as BackendJob | { detail?: string };
  if (response.status === 408 || response.status === 429 || response.status >= 500) {
    throw new RetryablePollingError(
      apiMessage(payload, `The local engine returned ${response.status}.`),
    );
  }
  if (!response.ok) {
    throw new Error(apiMessage(payload, `The local engine returned ${response.status}.`));
  }
  return payload as BackendJob;
}

function uploadToLocalEngine(
  file: File,
  language: string,
  quality: "accurate" | "fast",
  requestToken: string,
  onProgress: (percent: number) => void,
  register: (request: XMLHttpRequest) => void,
) {
  return new Promise<BackendJob>((resolve, reject) => {
    const request = new XMLHttpRequest();
    register(request);
    request.open("POST", `${LOCAL_API_URL}/api/jobs`);
    request.setRequestHeader(LOCAL_REQUEST_TOKEN_HEADER, requestToken);
    request.timeout = 60_000;
    request.responseType = "json";
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress((event.loaded / event.total) * 5);
    };
    request.onerror = () => reject(new Error("Could not reach the local engine at 127.0.0.1:8008."));
    request.ontimeout = () => reject(new Error("The local engine did not answer within 60 seconds."));
    request.onabort = () => reject(new DOMException("Upload stopped", "AbortError"));
    request.onload = () => {
      const payload = request.response as unknown;
      if (request.status >= 200 && request.status < 300) {
        resolve(payload as BackendJob);
        return;
      }
      reject(
        new LocalEngineRequestError(
          apiMessage(payload, `The local engine returned ${request.status}.`),
          request.status,
        ),
      );
    };
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("language", language);
    form.append("quality", quality);
    request.send(form);
  });
}

function wait(milliseconds: number) {
  return new Promise<void>((resolve) => window.setTimeout(resolve, milliseconds));
}

type LyricsLinesProps = {
  lines: LyricLine[];
  lineWordOffsets: number[];
  lineRefs: { current: Array<HTMLButtonElement | null> };
  wordRefs: { current: Array<HTMLSpanElement | null> };
  directScrollLineRef: { current: number | null };
  onSeek: (time: number) => void;
  onScrollToLine: (lineIndex: number) => void;
};

const LyricsLines = memo(function LyricsLines({
  lines,
  lineWordOffsets,
  lineRefs,
  wordRefs,
  directScrollLineRef,
  onSeek,
  onScrollToLine,
}: LyricsLinesProps) {
  return (
    <div className="lyrics-lines">
      {lines.map((line, lineIndex) => (
        <button
          className={`lyric-line ${(line.kind ?? "lead") === "adlib" ? "is-adlib" : ""}`}
          key={line.id}
          type="button"
          ref={(element) => {
            lineRefs.current[lineIndex] = element;
          }}
          onMouseDown={(event) => {
            if (event.detail > 0) event.preventDefault();
          }}
          onClick={(event) => {
            if (event.detail > 0) event.currentTarget.blur();
            const wordElement = (event.target as HTMLElement).closest<HTMLElement>(
              "[data-word-start]",
            );
            const wordStart = Number(wordElement?.dataset.wordStart);
            const seekTime = Number.isFinite(wordStart) ? wordStart : line.start;
            directScrollLineRef.current = lineIndex;
            onSeek(seekTime);
            onScrollToLine(lineIndex);
          }}
          aria-label={`Seek to line at ${formatTime(line.start)}: ${line.words
            .map((word) => word.text)
            .join(" ")}`}
        >
          {line.words.map((word, wordIndex) => (
            <span
              key={`${word.start}-${wordIndex}`}
              className="lyric-word"
              ref={(element) => {
                wordRefs.current[lineWordOffsets[lineIndex] + wordIndex] = element;
              }}
              title={`Jump to ${formatTime(word.start)}`}
              data-word-start={word.start}
              data-timing={word.timing?.length ? "acoustic" : "linear"}
            >
              <span className="lyric-word-label">{word.text}</span>
            </span>
          ))}
        </button>
      ))}
    </div>
  );
});

export default function Home() {
  const audioRef = useRef<HTMLAudioElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const lyricsRef = useRef<HTMLElement>(null);
  const lyricsScrollRef = useRef<HTMLDivElement>(null);
  const lineRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const wordRefs = useRef<Array<HTMLSpanElement | null>>([]);
  const directScrollLineRef = useRef<number | null>(null);
  const objectUrlsRef = useRef<string[]>([]);
  const runIdRef = useRef(0);
  const fileRef = useRef<File | null>(null);
  const activeJobRef = useRef<string | null>(null);
  const uploadRequestRef = useRef<XMLHttpRequest | null>(null);
  const requestTokenRef = useRef("");
  const activeWordIndexesRef = useRef<number[]>([]);
  const lastLyricTimeRef = useRef(0);
  const focusedLineIndexRef = useRef(-1);
  const activeLineIndexesRef = useRef<number[]>([]);

  const [file, setFile] = useState<File | null>(null);
  const [track, setTrack] = useState({ title: "No song loaded", artist: "Choose local audio" });
  const [stage, setStage] = useState<ProcessingStage>("idle");
  const [status, setStatus] = useState("Ready when you are");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [lines, setLines] = useState<LyricLine[]>([]);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.82);
  const [isMuted, setIsMuted] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [language, setLanguage] = useState("english");
  const [quality, setQuality] = useState<"accurate" | "fast">("fast");
  const [originalUrl, setOriginalUrl] = useState("");
  const [vocalUrl, setVocalUrl] = useState("");
  const [playbackMode, setPlaybackMode] = useState<"mix" | "vocals">("mix");
  const [showPrivacy, setShowPrivacy] = useState(false);
  const [engineState, setEngineState] = useState<"checking" | "online" | "offline">(
    "checking",
  );
  const [engineHealth, setEngineHealth] = useState<BackendHealth | null>(null);
  const [engineDetail, setEngineDetail] = useState("");
  const [focusedLineIndex, setFocusedLineIndex] = useState(-1);

  const lineWordOffsets = useMemo(() => buildLineWordOffsets(lines), [lines]);

  const wordTimeline = useMemo(
    () =>
      lines
        .flatMap((line, lineIndex) =>
          line.words.map((word, wordIndex) => ({
            ...word,
            lineIndex,
            refIndex: lineWordOffsets[lineIndex] + wordIndex,
          })),
        )
        .sort((left, right) => left.start - right.start || left.end - right.end),
    [lineWordOffsets, lines],
  );
  const wordPrefixMaxEnds = useMemo(
    () => buildPrefixMaxEnds(wordTimeline),
    [wordTimeline],
  );
  const lineTimeline = useMemo(
    () =>
      lines
        .map((line, lineIndex) => ({
          start: line.start - 0.03,
          end: line.end + 0.24,
          lineIndex,
          kind: line.kind ?? "lead",
        }))
        .sort((left, right) => left.start - right.start || left.end - right.end),
    [lines],
  );
  const linePrefixMaxEnds = useMemo(
    () => buildPrefixMaxEnds(lineTimeline),
    [lineTimeline],
  );

  const syncLyricsAt = useCallback(
    (audioTime: number, force = false) => {
      const displayTime = Math.max(0, audioTime + LYRIC_LOOKAHEAD_SECONDS);
      const jumped = force || Math.abs(displayTime - lastLyricTimeRef.current) > 0.5;
      const candidate = findLastStartedIndex(wordTimeline, displayTime);
      const nextActive = findActiveIntervalIndexes(
        wordTimeline,
        wordPrefixMaxEnds,
        displayTime,
      );
      const nextActiveSet = new Set(nextActive);
      const previousActive = activeWordIndexesRef.current;

      const resetWord = (index: number) => {
        const word = wordTimeline[index];
        const element = word ? wordRefs.current[word.refIndex] : null;
        if (!element || !word) return;
        element.classList.remove("is-active");
        const past = displayTime >= word.end;
        element.classList.toggle("is-past", past);
        element.style.setProperty("--word-fill", past ? "100%" : "0%");
      };

      if (jumped) {
        wordTimeline.forEach((_, index) => resetWord(index));
      } else {
        previousActive
          .filter((index) => !nextActiveSet.has(index))
          .forEach((index) => resetWord(index));
      }

      nextActive.forEach((index) => {
        const word = wordTimeline[index];
        const element = word ? wordRefs.current[word.refIndex] : null;
        if (element && word) {
          const progress = wordFillAt(word, displayTime);
          element.classList.add("is-active");
          element.classList.remove("is-past");
          element.style.setProperty("--word-fill", `${progress * 100}%`);
        }
      });
      if (!nextActive.length && candidate >= 0) resetWord(candidate);
      activeWordIndexesRef.current = nextActive;

      const activeLineTimelineIndexes = findActiveIntervalIndexes(
        lineTimeline,
        linePrefixMaxEnds,
        displayTime,
      );
      const liveLines = activeLineTimelineIndexes.map(
        (timelineIndex) => lineTimeline[timelineIndex].lineIndex,
      );
      const liveLineSet = new Set(liveLines);
      const previousLiveLines = activeLineIndexesRef.current;
      const resetLine = (lineIndex: number) => {
        const element = lineRefs.current[lineIndex];
        const line = lines[lineIndex];
        if (!element || !line) return;
        element.classList.remove("is-active");
        element.classList.toggle("is-past", displayTime > line.end);
      };

      if (jumped) {
        lines.forEach((_, lineIndex) => resetLine(lineIndex));
      } else {
        previousLiveLines
          .filter((lineIndex) => !liveLineSet.has(lineIndex))
          .forEach((lineIndex) => resetLine(lineIndex));
      }
      liveLines.forEach((lineIndex) => {
        const element = lineRefs.current[lineIndex];
        if (!element) return;
        element.classList.add("is-active");
        element.classList.remove("is-past");
      });
      activeLineIndexesRef.current = liveLines;
      lastLyricTimeRef.current = displayTime;

      const focusedWordIndex =
        nextActive.find((index) => (wordTimeline[index].kind ?? "lead") === "lead") ??
        nextActive[0];
      let nextFocused =
        focusedWordIndex === undefined ? -1 : wordTimeline[focusedWordIndex].lineIndex;
      if (nextFocused < 0 && liveLines.length) {
        nextFocused =
          liveLines.find((lineIndex) => (lines[lineIndex].kind ?? "lead") === "lead") ??
          liveLines[0];
      }
      if (nextFocused < 0) {
        const lastStartedLine = findLastStartedIndex(lineTimeline, displayTime);
        nextFocused =
          lastStartedLine >= 0 ? lineTimeline[lastStartedLine].lineIndex : -1;
      }
      if (nextFocused !== focusedLineIndexRef.current) {
        focusedLineIndexRef.current = nextFocused;
        setFocusedLineIndex(nextFocused);
      }
    },
    [linePrefixMaxEnds, lineTimeline, lines, wordPrefixMaxEnds, wordTimeline],
  );

  const clearObjectUrls = useCallback(() => {
    objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    objectUrlsRef.current = [];
  }, []);

  useEffect(() => {
    fileRef.current = file;
    if (!file || !originalUrl || objectUrlsRef.current.includes(originalUrl)) return;

    const replacementUrl = URL.createObjectURL(file);
    objectUrlsRef.current.push(replacementUrl);
    setOriginalUrl(replacementUrl);
    if (audioRef.current && playbackMode === "mix") {
      audioRef.current.src = replacementUrl;
      audioRef.current.load();
    }
  }, [file, originalUrl, playbackMode]);

  const checkEngine = useCallback(async () => {
    setEngineState("checking");
    try {
      const health = await readEngineHealth();
      requestTokenRef.current = health.request_token;
      setEngineHealth(health);
      setEngineState(health.ready ? "online" : "offline");
      return health;
    } catch {
      requestTokenRef.current = "";
      setEngineHealth(null);
      setEngineState("offline");
      return null;
    }
  }, []);

  useEffect(() => {
    const checkTimer = window.setTimeout(() => void checkEngine(), 0);
    return () => window.clearTimeout(checkTimer);
  }, [checkEngine]);

  useEffect(() => {
    const refreshWhenVisible = () => {
      if (document.visibilityState === "visible") void checkEngine();
    };
    window.addEventListener("focus", refreshWhenVisible);
    document.addEventListener("visibilitychange", refreshWhenVisible);
    return () => {
      window.removeEventListener("focus", refreshWhenVisible);
      document.removeEventListener("visibilitychange", refreshWhenVisible);
    };
  }, [checkEngine]);

  const cancelJobById = useCallback(async (jobId: string) => {
    await withRefreshedRequestToken({
      token: requestTokenRef.current,
      refreshToken: async () => {
        const health = await readEngineHealth();
        requestTokenRef.current = health.request_token;
        return health.request_token;
      },
      request: (requestToken) => cancelLocalJob(jobId, requestToken),
    });
  }, []);

  const cancelCurrentJob = useCallback(() => {
    uploadRequestRef.current?.abort();
    uploadRequestRef.current = null;
    const jobId = activeJobRef.current;
    activeJobRef.current = null;
    if (jobId) void cancelJobById(jobId).catch(() => {});
  }, [cancelJobById]);

  useEffect(() => cancelCurrentJob, [cancelCurrentJob]);

  useEffect(() => {
    if (!isPlaying) {
      syncLyricsAt(audioRef.current?.currentTime ?? 0, true);
      return;
    }
    let lastPlayerClockUpdate = 0;
    let animationFrame = 0;
    const update = (frameTime: number) => {
      const audioTime = audioRef.current?.currentTime ?? 0;
      syncLyricsAt(audioTime);
      if (frameTime - lastPlayerClockUpdate >= PLAYER_CLOCK_INTERVAL_MS) {
        setCurrentTime(audioTime);
        lastPlayerClockUpdate = frameTime;
      }
      animationFrame = window.requestAnimationFrame(update);
    };
    animationFrame = window.requestAnimationFrame(update);
    return () => window.cancelAnimationFrame(animationFrame);
  }, [isPlaying, syncLyricsAt]);

  useEffect(() => {
    const container = lyricsScrollRef.current;
    if (!container) return;
    const updateSafeSpace = () => {
      const height = container.clientHeight;
      container.style.setProperty("--lyrics-top-space", `${Math.max(150, height * 0.38)}px`);
      container.style.setProperty(
        "--lyrics-bottom-space",
        `${Math.max(210, height * 0.62)}px`,
      );
    };
    updateSafeSpace();
    const observer = new ResizeObserver(updateSafeSpace);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const scrollToFocusedLine = useCallback((lineIndex: number) => {
    const container = lyricsScrollRef.current;
    const line = lineRefs.current[lineIndex];
    if (!container || !line) return;

    const containerRect = container.getBoundingClientRect();
    const lineRect = line.getBoundingClientRect();
    const lineCenter =
      container.scrollTop + lineRect.top - containerRect.top + lineRect.height / 2;
    const readingRail = Math.max(104, container.clientHeight * 0.38);
    const maximumScroll = Math.max(0, container.scrollHeight - container.clientHeight);
    const targetScroll = Math.max(0, Math.min(maximumScroll, lineCenter - readingRail));
    const startScroll = container.scrollTop;
    const distance = targetScroll - startScroll;
    if (Math.abs(distance) < 1) return;
    container.scrollTo({
      top: targetScroll,
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth",
    });
  }, []);

  useEffect(() => {
    if (focusedLineIndex < 0) return;
    if (directScrollLineRef.current === focusedLineIndex) {
      directScrollLineRef.current = null;
      return;
    }
    directScrollLineRef.current = null;
    scrollToFocusedLine(focusedLineIndex);
  }, [focusedLineIndex, scrollToFocusedLine]);

  useEffect(() => {
    const container = lyricsScrollRef.current;
    if (!container) return;
    const stopTrackingAnimation = () => {
      container.scrollTo({ top: container.scrollTop, behavior: "auto" });
    };
    container.addEventListener("wheel", stopTrackingAnimation, { passive: true });
    container.addEventListener("touchstart", stopTrackingAnimation, { passive: true });
    container.addEventListener("pointerdown", stopTrackingAnimation, { passive: true });
    return () => {
      stopTrackingAnimation();
      container.removeEventListener("wheel", stopTrackingAnimation);
      container.removeEventListener("touchstart", stopTrackingAnimation);
      container.removeEventListener("pointerdown", stopTrackingAnimation);
    };
  }, []);

  const togglePlayback = useCallback(async () => {
    const audio = audioRef.current;
    if (!audio || !originalUrl) return;
    if (audio.paused) {
      try {
        await audio.play();
        setError("");
      } catch {
        setError("Playback was blocked by the browser. Press play once more.");
      }
    } else {
      audio.pause();
    }
  }, [originalUrl]);

  const seekTo = useCallback(
    (time: number) => {
      const audio = audioRef.current;
      if (!audio) return;
      const nextTime = Math.max(0, Math.min(time, audio.duration || 0));
      audio.currentTime = nextTime;
      setCurrentTime(nextTime);
      syncLyricsAt(nextTime, true);
    },
    [syncLyricsAt],
  );

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.matches("input, select, textarea, button")) return;
      if (event.code === "Space") {
        event.preventDefault();
        void togglePlayback();
      }
      if (event.code === "ArrowLeft") seekTo(currentTime - 5);
      if (event.code === "ArrowRight") seekTo(currentTime + 5);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [currentTime, seekTo, togglePlayback]);

  const processTrack = useCallback(
    async (nextFile: File, runId: number) => {
      const stillCurrent = () => runIdRef.current === runId;
      setError("");
      setLines([]);
      setEngineDetail("");
      setStage("uploading");
      setStatus("Sending the song to the local GPU engine");
      setProgress(1);

      try {
        const currentHealth =
          engineState === "online" && requestTokenRef.current
            ? engineHealth
            : await checkEngine();
        const engineReady =
          (engineState === "online" && Boolean(requestTokenRef.current)) ||
          Boolean(currentHealth?.ready);
        if (!engineReady) throw new Error(ENGINE_OFFLINE_MESSAGE);

        const created = await withRefreshedRequestToken({
          token: requestTokenRef.current,
          refreshToken: async () => {
            const health = await checkEngine();
            if (!health?.ready) throw new Error(ENGINE_OFFLINE_MESSAGE);
            return health.request_token;
          },
          request: (requestToken) =>
            uploadToLocalEngine(
              nextFile,
              language,
              quality,
              requestToken,
              (uploadProgress) => {
                if (stillCurrent()) setProgress(Math.max(1, uploadProgress));
              },
              (request) => {
                uploadRequestRef.current = request;
              },
            ),
        });
        uploadRequestRef.current = null;
        if (!stillCurrent()) {
          void cancelJobById(created.id).catch(() => {});
          return;
        }
        activeJobRef.current = created.id;

        let pollingFailures = 0;
        while (stillCurrent()) {
          let job: BackendJob;
          try {
            job = await fetchJobStatus(created.id);
          } catch (pollingError) {
            if (!stillCurrent()) return;
            if (!isRetryablePollingError(pollingError)) throw pollingError;
            pollingFailures += 1;
            if (pollingFailures >= MAX_CONSECUTIVE_POLLING_FAILURES) {
              throw pollingError;
            }
            const retryDelay = pollingRetryDelay(pollingFailures);
            setStatus(
              `Local engine connection interrupted · retrying in ${Math.ceil(
                retryDelay / 1000,
              )}s`,
            );
            await wait(retryDelay);
            continue;
          }

          if (!stillCurrent()) return;
          pollingFailures = 0;
          setProgress(job.progress);
          setStatus(job.status);
          if (job.duration > 0) setDuration(job.duration);
          if (job.device) {
            setEngineDetail(
              `${job.device} · ${job.separation_model ?? "Demucs"} · ${job.transcription_model ?? "Whisper"}`,
            );
          }
          if (job.vocal_url) setVocalUrl(`${LOCAL_API_URL}${job.vocal_url}`);

          if (job.stage === "queued") setStage("uploading");
          if (job.stage === "separating") setStage("separating");
          if (job.stage === "transcribing") setStage("transcribing");
          if (job.stage === "error") {
            throw new Error(job.error || "The local model stopped unexpectedly.");
          }
          if (job.stage === "cancelled") {
            activeJobRef.current = null;
            return;
          }
          if (job.stage === "complete") {
            const timedWords = job.words ?? [];
            const lyricLines = job.lines?.length ? job.lines : groupIntoLines(timedWords);
            if (!lyricLines.length) {
              throw new Error("Whisper did not return any timed lyric lines.");
            }
            setLines(lyricLines);
            setProgress(100);
            setStatus(`${timedWords.length} words synced locally`);
            setStage("complete");
            activeJobRef.current = null;
            return;
          }
          await wait(650);
        }
      } catch (processingError) {
        if (!stillCurrent()) return;
        if (processingError instanceof DOMException && processingError.name === "AbortError") return;
        cancelCurrentJob();
        console.error(processingError);
        const message =
          processingError instanceof Error
            ? processingError.message
            : "The audio could not be processed on this device.";
        setError(message);
        setStatus("Processing stopped");
        setStage("error");
      }
    },
    [
      cancelCurrentJob,
      cancelJobById,
      checkEngine,
      engineHealth,
      engineState,
      language,
      quality,
    ],
  );

  const loadFile = useCallback(
    (nextFile: File) => {
      if (!isAudioFile(nextFile)) {
        setError("Choose an MP3, WAV, FLAC, M4A, AAC, OGG, or WebM audio file.");
        return;
      }
      if (nextFile.size > MAX_FILE_SIZE) {
        setError("That file is over 500 MB. Choose a smaller audio file for local processing.");
        return;
      }

      cancelCurrentJob();
      runIdRef.current += 1;
      const runId = runIdRef.current;
      audioRef.current?.pause();
      clearObjectUrls();
      const nextOriginalUrl = URL.createObjectURL(nextFile);
      objectUrlsRef.current.push(nextOriginalUrl);
      const parsed = parseTrackName(nextFile.name);
      fileRef.current = nextFile;
      setFile(nextFile);
      setTrack(parsed);
      setOriginalUrl(nextOriginalUrl);
      setVocalUrl("");
      setPlaybackMode("mix");
      setCurrentTime(0);
      setDuration(0);
      setIsPlaying(false);
      activeWordIndexesRef.current = [];
      lastLyricTimeRef.current = 0;
      focusedLineIndexRef.current = -1;
      activeLineIndexesRef.current = [];
      lineRefs.current = [];
      wordRefs.current = [];
      setFocusedLineIndex(-1);
      setError("");

      if (audioRef.current) {
        audioRef.current.src = nextOriginalUrl;
        audioRef.current.load();
      }
      void processTrack(nextFile, runId);
    },
    [cancelCurrentJob, clearObjectUrls, processTrack],
  );

  const handleFileInput = (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0];
    if (nextFile) loadFile(nextFile);
    event.target.value = "";
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const nextFile = event.dataTransfer.files?.[0];
    if (nextFile) loadFile(nextFile);
  };

  const retry = () => {
    if (!fileRef.current) return;
    cancelCurrentJob();
    runIdRef.current += 1;
    void processTrack(fileRef.current, runIdRef.current);
  };

  const reset = () => {
    cancelCurrentJob();
    runIdRef.current += 1;
    audioRef.current?.pause();
    if (audioRef.current) {
      audioRef.current.removeAttribute("src");
      audioRef.current.load();
    }
    clearObjectUrls();
    fileRef.current = null;
    setFile(null);
    setTrack({ title: "No song loaded", artist: "Choose local audio" });
    setStage("idle");
    setStatus("Ready when you are");
    setProgress(0);
    setLines([]);
    setOriginalUrl("");
    setVocalUrl("");
    setCurrentTime(0);
    setDuration(0);
    setIsPlaying(false);
    activeWordIndexesRef.current = [];
    lastLyricTimeRef.current = 0;
    focusedLineIndexRef.current = -1;
    activeLineIndexesRef.current = [];
    lineRefs.current = [];
    wordRefs.current = [];
    setFocusedLineIndex(-1);
    setPlaybackMode("mix");
    setEngineDetail("");
    setError("");
  };

  const changePlaybackMode = (nextMode: "mix" | "vocals") => {
    if (nextMode === "vocals" && !vocalUrl) return;
    const audio = audioRef.current;
    if (!audio || nextMode === playbackMode) return;
    const previousMode = playbackMode;
    const previousUrl = previousMode === "vocals" ? vocalUrl : originalUrl;
    const time = audio.currentTime;
    const shouldResume = !audio.paused;

    const restorePositionAndPlayback = async () => {
      audio.currentTime = Math.min(time, audio.duration || time);
      setCurrentTime(audio.currentTime);
      syncLyricsAt(audio.currentTime, true);
      if (shouldResume) {
        try {
          await audio.play();
        } catch {
          setError("Playback was blocked after switching sources. Press play to continue.");
        }
      }
    };
    const handleLoadError = () => {
      audio.removeEventListener("loadedmetadata", handleLoadedMetadata);
      setPlaybackMode(previousMode);
      setError("The selected playback source could not be loaded. Restored the previous source.");
      if (previousUrl) {
        audio.addEventListener("loadedmetadata", restorePositionAndPlayback, { once: true });
        audio.src = previousUrl;
        audio.load();
      }
    };
    const handleLoadedMetadata = async () => {
      audio.removeEventListener("error", handleLoadError);
      await restorePositionAndPlayback();
    };

    audio.pause();
    audio.addEventListener("loadedmetadata", handleLoadedMetadata, { once: true });
    audio.addEventListener("error", handleLoadError, { once: true });
    audio.src = nextMode === "vocals" ? vocalUrl : originalUrl;
    audio.load();
    setPlaybackMode(nextMode);
  };

  const toggleMute = () => {
    const audio = audioRef.current;
    if (!audio) return;
    const nextMuted = !isMuted;
    audio.muted = nextMuted;
    setIsMuted(nextMuted);
  };

  const updateVolume = (nextVolume: number) => {
    setVolume(nextVolume);
    setIsMuted(false);
    if (audioRef.current) {
      audioRef.current.volume = nextVolume;
      audioRef.current.muted = false;
    }
  };

  const requestFullscreen = async () => {
    if (!lyricsRef.current) return;
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await lyricsRef.current.requestFullscreen();
    } catch {
      setError("Fullscreen is unavailable in this browser window.");
    }
  };

  const exportLyrics = (kind: "json" | "lrc") => {
    if (!lines.length) return;
    const safeTitle = track.title.replace(/[^a-z0-9-_]+/gi, "-").replace(/^-|-$/g, "");
    if (kind === "json") {
      const payload = {
        title: track.title,
        artist: track.artist,
        duration,
        generatedOnDevice: true,
        lines,
      };
      downloadBlob(
        new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
        `${safeTitle || "lyrics"}-word-timings.json`,
      );
      return;
    }
    const lrc = lines
      .map((line) => {
        const minutes = Math.floor(line.start / 60)
          .toString()
          .padStart(2, "0");
        const seconds = (line.start % 60).toFixed(2).padStart(5, "0");
        return `[${minutes}:${seconds}]${line.words.map((word) => word.text).join(" ")}`;
      })
      .join("\n");
    downloadBlob(new Blob([lrc], { type: "text/plain" }), `${safeTitle || "lyrics"}.lrc`);
  };

  const stageRank = { idle: 0, uploading: 1, separating: 2, transcribing: 3, complete: 4, error: -1 }[
    stage
  ];

  const steps = [
    { label: "Open the mix", detail: "Localhost handoff", rank: 1, icon: FileAudio },
    {
      label: "Find the vocal",
      detail: quality === "accurate" ? "HTDemucs fine-tuned" : "HTDemucs fast pass",
      rank: 2,
      icon: Mic2,
    },
    {
      label: "Sync every word",
      detail: quality === "accurate" ? "Whisper large-v3" : "Whisper large-v3 turbo",
      rank: 3,
      icon: AudioLines,
    },
  ];

  const renderVolumeIcon = () => {
    if (isMuted || volume === 0) return <VolumeX size={17} />;
    if (volume < 0.45) return <Volume1 size={17} />;
    return <Volume2 size={17} />;
  };

  return (
    <main className={`app-shell ${file ? "has-track" : "is-empty"}`}>
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <div className="grain" />

      <header className="site-header">
        <button className="brand" type="button" onClick={reset} aria-label="Reset lyricwave">
          <span className="brand-wave" aria-hidden="true">
            {[10, 18, 26, 18, 10].map((height, index) => (
              <i key={`${height}-${index}`} style={{ height }} />
            ))}
          </span>
          <span>lyricwave</span>
        </button>
        <div className="header-actions">
          <button
            className={`privacy-pill engine-${engineState}`}
            type="button"
            onClick={() => setShowPrivacy((visible) => !visible)}
            aria-expanded={showPrivacy}
          >
            <LockKeyhole size={13} />
            {engineState === "online"
              ? "Local GPU ready"
              : engineState === "checking"
                ? "Checking local engine"
                : "Local engine offline"}
          </button>
          {showPrivacy && (
            <div className="privacy-popover" role="status">
              <strong>Your song stays here.</strong>
              <span>
                The interface sends audio only to 127.0.0.1 on this PC. Demucs and Whisper run on
                {engineHealth?.device ? ` ${engineHealth.device}` : " your local hardware"}.
              </span>
            </div>
          )}
          {file && (
            <button className="header-icon" type="button" onClick={reset} aria-label="Load another song">
              <RotateCcw size={16} />
            </button>
          )}
        </div>
      </header>

      <div className="workspace">
        <aside className={`song-panel ${file ? "has-file" : ""}`}>
          {!file ? (
            <>
              <div className="intro-copy">
                <span className="eyebrow">PRIVATE · LOCAL · YOURS</span>
                <h1>
                  Turn any song into <span>live lyrics.</span>
                </h1>
                <p>
                  Drop a track. Lyricwave separates the vocal and syncs every word without
                  sending the song away from this machine.
                </p>
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
                <div className="upload-icon">
                  <UploadCloud size={25} strokeWidth={1.8} />
                </div>
                <strong>{dragging ? "Let it drop" : "Drop your song here"}</strong>
                <span>MP3, WAV, FLAC, M4A · up to 500 MB</span>
                <button className="primary-button" type="button" onClick={() => inputRef.current?.click()}>
                  Choose a song
                </button>
              </div>

              <label className="language-control">
                <span>Lyrics language</span>
                <select value={language} onChange={(event) => setLanguage(event.target.value)}>
                  {languageOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="language-control mode-control">
                <span>Processing</span>
                <select
                  value={quality}
                  onChange={(event) => setQuality(event.target.value as "accurate" | "fast")}
                >
                  <option value="fast">Fast · turbo</option>
                  <option value="accurate">Accurate · large-v3</option>
                </select>
              </label>

              <p className="model-note">
                {engineState === "offline"
                  ? "Local engine offline. Run npm run setup:engine once, then npm run dev."
                  : quality === "fast"
                    ? "Fast is the best starting point. Accurate adds a slower second pass."
                    : "Accurate uses the full local models and keeps downloads cached for later runs."}
              </p>
            </>
          ) : (
            <>
              <div className="track-card">
                <div className="track-art" aria-hidden="true">
                  <div className="track-art-orbit" />
                  <Waves size={44} strokeWidth={1.3} />
                </div>
                <div className="track-heading">
                  <span className="eyebrow">NOW PLAYING</span>
                  <h1>{track.title}</h1>
                  <p>{track.artist}</p>
                  <span className="file-meta">
                    {formatFileSize(file.size)} · {duration ? formatTime(duration) : "reading…"}
                  </span>
                </div>
              </div>

              <div
                className="pipeline-card"
                aria-live="polite"
                aria-busy={stage !== "complete" && stage !== "error"}
              >
                <div className="pipeline-topline">
                  <span>{stage === "complete" ? "Ready to play" : stage === "error" ? "Needs attention" : "Making lyrics"}</span>
                  <strong>{Math.round(progress)}%</strong>
                </div>
                <div
                  className="progress-track"
                  role="progressbar"
                  aria-label="Lyric processing progress"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(progress)}
                >
                  <span aria-hidden="true" style={{ width: `${progress}%` }} />
                </div>
                <p className="status-copy">{status}</p>

                {stage === "complete" ? (
                  <div className="completion-summary">
                    <span className="completion-icon">
                      <Check size={15} strokeWidth={2.6} />
                    </span>
                    <span>
                      <strong>Playback ready</strong>
                      <small>
                        {wordTimeline.length} words · {lines.length} lines · synced on this PC
                      </small>
                    </span>
                  </div>
                ) : (
                  <div className="steps-list">
                    {steps.map((step) => {
                      const StepIcon = step.icon;
                      const complete = stageRank > step.rank;
                      const active = stageRank === step.rank;
                      return (
                        <div
                          key={step.label}
                          className={`process-step ${complete ? "is-complete" : ""} ${active ? "is-active" : ""}`}
                        >
                          <span className="step-icon">
                            {complete ? (
                              <Check size={14} strokeWidth={2.5} />
                            ) : active ? (
                              <LoaderCircle className="spin" size={15} />
                            ) : (
                              <StepIcon size={15} />
                            )}
                          </span>
                          <span>
                            <strong>{step.label}</strong>
                            <small>{step.detail}</small>
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}

                {engineDetail && <p className="engine-detail">{engineDetail}</p>}
                {stage === "error" && (
                  <div className="error-card">
                    <p>{error}</p>
                    <button type="button" onClick={retry}>
                      <RotateCcw size={14} /> Try again
                    </button>
                  </div>
                )}
              </div>

              {stage === "complete" && (
                <div className="export-row">
                  <button type="button" onClick={() => exportLyrics("lrc")}>
                    <Download size={14} /> LRC
                  </button>
                  <button type="button" onClick={() => exportLyrics("json")}>
                    <Download size={14} /> Word JSON
                  </button>
                </div>
              )}

              <button className="replace-button" type="button" onClick={() => inputRef.current?.click()}>
                <UploadCloud size={14} /> Replace audio
              </button>
            </>
          )}
        </aside>

        <section className="lyrics-stage" ref={lyricsRef} aria-label="Live lyrics">
          <div className="lyrics-toolbar">
            <div className="lyrics-status">
              <span className={`live-dot ${isPlaying ? "is-live" : ""}`} />
              <span>{stage === "complete" ? "LIVE LYRICS" : "LYRICS ROOM"}</span>
              {stage === "complete" && (
                <span className="lyrics-meta">
                  {wordTimeline.length} words · {lines.length} lines
                </span>
              )}
            </div>
            <button type="button" onClick={requestFullscreen} aria-label="Toggle fullscreen lyrics">
              <Maximize2 size={15} />
            </button>
          </div>

          <div
            className={`lyrics-scroll ${stage === "complete" ? "has-lyrics" : ""}`}
            ref={lyricsScrollRef}
          >
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
            ) : stage === "uploading" ? (
              <div className="holding-lyrics processing-lyrics">
                <p>Passing the song</p>
                <p>
                  to the <em>GPU.</em>
                </p>
                <span>The player is already usable while the local engine starts.</span>
              </div>
            ) : stage === "separating" ? (
              <div className="holding-lyrics processing-lyrics">
                <p>Pulling the voice</p>
                <p>
                  out of the <em>mix.</em>
                </p>
                <span>Fine-tuned Demucs is separating the vocal on this PC.</span>
              </div>
            ) : stage === "transcribing" ? (
              <div className="holding-lyrics processing-lyrics">
                <p>Listening for</p>
                <p>
                  every <em>word.</em>
                </p>
                <span>Whisper is adding an exact timestamp to each recognized word.</span>
              </div>
            ) : stage === "error" ? (
              <div className="holding-lyrics error-lyrics">
                <p>That one went</p>
                <p>
                  a little <em>quiet.</em>
                </p>
                <span>Try the file again or choose another track.</span>
              </div>
            ) : (
              <div className="holding-lyrics">
                <p>Drop a song.</p>
                <p>
                  We’ll find the <em>voice</em>
                </p>
                <p>and light up every word.</p>
                <span>Click any finished word to jump to its exact timestamp.</span>
              </div>
            )}
          </div>
        </section>
      </div>

      <footer className={`player ${file ? "is-loaded" : ""}`}>
        <div className="player-track">
          <div className="mini-art">
            <AudioLines size={19} />
          </div>
          <div>
            <strong>{track.title}</strong>
            <span>{track.artist}</span>
          </div>
        </div>

        <div className="transport">
          <div className="transport-buttons">
            <button type="button" onClick={() => seekTo(currentTime - 10)} disabled={!file} aria-label="Back 10 seconds">
              <SkipBack size={17} fill="currentColor" />
            </button>
            <button className="play-button" type="button" onClick={() => void togglePlayback()} disabled={!file} aria-label={isPlaying ? "Pause" : "Play"}>
              {isPlaying ? <Pause size={19} fill="currentColor" /> : <Play size={19} fill="currentColor" />}
            </button>
            <button type="button" onClick={() => seekTo(currentTime + 10)} disabled={!file} aria-label="Forward 10 seconds">
              <SkipForward size={17} fill="currentColor" />
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
              disabled={!file}
              aria-label="Seek through song"
              style={{ "--seek": `${duration ? (currentTime / duration) * 100 : 0}%` } as CSSProperties}
            />
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        <div className="player-tools">
          <div className="source-switch" role="group" aria-label="Playback source">
            <button
              type="button"
              className={playbackMode === "mix" ? "is-active" : ""}
              onClick={() => changePlaybackMode("mix")}
              disabled={!file}
            >
              Mix
            </button>
            <button
              type="button"
              className={playbackMode === "vocals" ? "is-active" : ""}
              onClick={() => changePlaybackMode("vocals")}
              disabled={!vocalUrl}
              title={vocalUrl ? "Listen to isolated vocals" : "Available after vocal isolation"}
            >
              <Headphones size={12} /> Vocal
            </button>
          </div>
          <div className="volume-control">
            <button type="button" onClick={toggleMute} disabled={!file} aria-label={isMuted ? "Unmute" : "Mute"}>
              {renderVolumeIcon()}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={isMuted ? 0 : volume}
              onChange={(event) => updateVolume(Number(event.target.value))}
              disabled={!file}
              aria-label="Volume"
              style={{ "--volume": `${(isMuted ? 0 : volume) * 100}%` } as CSSProperties}
            />
          </div>
        </div>
      </footer>

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
            <X size={15} />
          </button>
        </div>
      )}
    </main>
  );
}
