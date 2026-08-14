"use client";

import {
  type CSSProperties,
  type ChangeEvent,
  type DragEvent,
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
import type { DemucsConfig } from "@browserai/browserai/demucs";

type ProcessingStage =
  | "idle"
  | "decoding"
  | "separating"
  | "transcribing"
  | "complete"
  | "error";

type TimedWord = {
  text: string;
  start: number;
  end: number;
};

type LyricLine = {
  id: string;
  start: number;
  end: number;
  words: TimedWord[];
};

type WhisperChunk = {
  text?: string;
  timestamp?: [number | null, number | null];
};

type WhisperOutput = {
  text?: string;
  chunks?: WhisperChunk[];
};

type WhisperTranscriber = (
  audio: Float32Array,
  options: Record<string, unknown>,
) => Promise<WhisperOutput>;

type ProgressInfo = {
  status?: string;
  progress?: number;
  file?: string;
};

const ACCEPTED_EXTENSIONS = ["mp3", "wav", "flac", "m4a", "aac", "ogg", "webm"];
const MAX_FILE_SIZE = 500 * 1024 * 1024;

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

let whisperPromise: Promise<WhisperTranscriber> | null = null;

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
    return { artist: pieces[0], title: pieces.slice(1).join(" — ") };
  }
  return { artist: "Local audio", title: clean || "Untitled track" };
}

function isAudioFile(file: File) {
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  return file.type.startsWith("audio/") || ACCEPTED_EXTENSIONS.includes(extension);
}

function normalizeWords(chunks: WhisperChunk[]) {
  const words: TimedWord[] = [];

  for (const chunk of chunks) {
    const text = chunk.text?.trim();
    const start = chunk.timestamp?.[0];
    const end = chunk.timestamp?.[1];
    if (!text || start == null || end == null || end < start) continue;

    const pieces = text.split(/\s+/).filter(Boolean);
    const duration = Math.max(0.08, end - start);
    pieces.forEach((piece, index) => {
      const pieceStart = start + (duration * index) / pieces.length;
      const pieceEnd = start + (duration * (index + 1)) / pieces.length;
      words.push({ text: piece, start: pieceStart, end: pieceEnd });
    });
  }

  return words;
}

function groupIntoLines(words: TimedWord[]): LyricLine[] {
  if (!words.length) return [];
  const groups: TimedWord[][] = [];
  let current: TimedWord[] = [];

  for (const word of words) {
    const previous = current[current.length - 1];
    const gap = previous ? word.start - previous.end : 0;
    const lineDuration = current.length ? word.end - current[0].start : 0;
    const punctuationBreak = previous ? /[.!?…]$/.test(previous.text) && current.length >= 4 : false;
    const shouldBreak =
      current.length > 0 &&
      (gap > 0.95 || current.length >= 9 || lineDuration > 6.5 || punctuationBreak);

    if (shouldBreak) {
      groups.push(current);
      current = [];
    }
    current.push(word);
  }
  if (current.length) groups.push(current);

  return groups.map((lineWords, index) => ({
    id: `line-${index}-${lineWords[0].start.toFixed(2)}`,
    start: lineWords[0].start,
    end: lineWords[lineWords.length - 1].end,
    words: lineWords,
  }));
}

async function voiceFocusFallback(buffer: AudioBuffer) {
  const sampleRate = buffer.sampleRate;
  const mono = new OfflineAudioContext(1, buffer.length, sampleRate);
  const monoBuffer = mono.createBuffer(1, buffer.length, sampleRate);
  const output = monoBuffer.getChannelData(0);
  const left = buffer.getChannelData(0);
  const right = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : left;

  for (let index = 0; index < output.length; index += 1) {
    output[index] = (left[index] + right[index]) * 0.5;
  }

  const source = mono.createBufferSource();
  const highPass = mono.createBiquadFilter();
  const lowPass = mono.createBiquadFilter();
  const compressor = mono.createDynamicsCompressor();
  source.buffer = monoBuffer;
  highPass.type = "highpass";
  highPass.frequency.value = 110;
  lowPass.type = "lowpass";
  lowPass.frequency.value = 9200;
  compressor.threshold.value = -28;
  compressor.knee.value = 18;
  compressor.ratio.value = 4;
  compressor.attack.value = 0.01;
  compressor.release.value = 0.2;
  source.connect(highPass).connect(lowPass).connect(compressor).connect(mono.destination);
  source.start();
  return mono.startRendering();
}

async function resampleToWhisper(buffer: AudioBuffer) {
  const sampleRate = 16000;
  const length = Math.ceil(buffer.duration * sampleRate);
  const offline = new OfflineAudioContext(1, length, sampleRate);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0).slice();
}

function audioBufferToWav(buffer: AudioBuffer) {
  const channels = Math.min(2, buffer.numberOfChannels);
  const sampleRate = buffer.sampleRate;
  const bytesPerSample = 2;
  const frameCount = buffer.length;
  const dataSize = frameCount * channels * bytesPerSample;
  const arrayBuffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(arrayBuffer);

  const writeString = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  const channelData = Array.from({ length: channels }, (_, index) =>
    buffer.getChannelData(index),
  );
  let offset = 44;
  for (let frame = 0; frame < frameCount; frame += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const sample = Math.max(-1, Math.min(1, channelData[channel][frame]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += bytesPerSample;
    }
  }

  return new Blob([arrayBuffer], { type: "audio/wav" });
}

async function getWhisper(
  supportsWebGPU: boolean,
  onProgress: (info: ProgressInfo) => void,
) {
  if (!whisperPromise) {
    whisperPromise = import("@huggingface/transformers")
      .then(async ({ pipeline }) => {
        const transcriber = await pipeline(
          "automatic-speech-recognition",
          "onnx-community/whisper-base_timestamped",
          {
            device: supportsWebGPU ? "webgpu" : "wasm",
            dtype: supportsWebGPU ? "fp16" : "q8",
            progress_callback: onProgress,
          },
        );
        return transcriber as unknown as WhisperTranscriber;
      })
      .catch((error) => {
        whisperPromise = null;
        throw error;
      });
  }
  return whisperPromise;
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

export default function Home() {
  const audioRef = useRef<HTMLAudioElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const lyricsRef = useRef<HTMLElement>(null);
  const lineRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const objectUrlsRef = useRef<string[]>([]);
  const runIdRef = useRef(0);
  const fileRef = useRef<File | null>(null);

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
  const [language, setLanguage] = useState("auto");
  const [originalUrl, setOriginalUrl] = useState("");
  const [vocalUrl, setVocalUrl] = useState("");
  const [playbackMode, setPlaybackMode] = useState<"mix" | "vocals">("mix");
  const [usedFallback, setUsedFallback] = useState(false);
  const [showPrivacy, setShowPrivacy] = useState(false);

  const supportsWebGPU = useMemo(
    () => typeof navigator !== "undefined" && "gpu" in navigator,
    [],
  );

  const activeLineIndex = useMemo(() => {
    if (!lines.length) return -1;
    const exact = lines.findIndex(
      (line) => currentTime >= line.start - 0.12 && currentTime <= line.end + 0.45,
    );
    if (exact >= 0) return exact;
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      if (currentTime >= lines[index].start) return index;
    }
    return 0;
  }, [currentTime, lines]);

  const clearObjectUrls = useCallback(() => {
    objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    objectUrlsRef.current = [];
  }, []);

  useEffect(() => clearObjectUrls, [clearObjectUrls]);

  useEffect(() => {
    if (!isPlaying) return;
    let frame = 0;
    const update = () => {
      if (audioRef.current) setCurrentTime(audioRef.current.currentTime);
      frame = window.requestAnimationFrame(update);
    };
    frame = window.requestAnimationFrame(update);
    return () => window.cancelAnimationFrame(frame);
  }, [isPlaying]);

  useEffect(() => {
    if (activeLineIndex < 0 || !isPlaying) return;
    lineRefs.current[activeLineIndex]?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }, [activeLineIndex, isPlaying]);

  const togglePlayback = useCallback(async () => {
    const audio = audioRef.current;
    if (!audio || !originalUrl) return;
    if (audio.paused) {
      try {
        await audio.play();
      } catch {
        setError("Playback was blocked by the browser. Press play once more.");
      }
    } else {
      audio.pause();
    }
  }, [originalUrl]);

  const seekTo = useCallback((time: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    const nextTime = Math.max(0, Math.min(time, audio.duration || 0));
    audio.currentTime = nextTime;
    setCurrentTime(nextTime);
  }, []);

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
      let vocalBuffer: AudioBuffer | null = null;
      setError("");
      setLines([]);
      setUsedFallback(false);
      setStage("decoding");
      setStatus("Opening the local audio");
      setProgress(4);

      try {
        const context = new AudioContext();
        const bytes = await nextFile.arrayBuffer();
        const decoded = await context.decodeAudioData(bytes);
        await context.close();
        if (!stillCurrent()) return;
        setDuration(decoded.duration);
        setProgress(10);

        setStage("separating");
        setStatus("Loading the vocal-isolation model");

        try {
          const { DemucsEngine, demucsModels } = await import(
            "@browserai/browserai/demucs"
          );
          const engine = new DemucsEngine();
          await engine.loadModel(demucsModels.htdemucs as DemucsConfig, {
            executionProviders: supportsWebGPU ? ["webgpu", "wasm"] : ["wasm"],
            onProgress: (info: { progress?: number }) => {
              if (!stillCurrent()) return;
              const modelProgress = Math.max(0, Math.min(100, info.progress ?? 0));
              setProgress(10 + modelProgress * 0.2);
              setStatus(`Loading vocal isolation · ${Math.round(modelProgress)}%`);
            },
          });
          if (!stillCurrent()) {
            engine.dispose();
            return;
          }
          setStatus("Separating the vocal stem");
          const separated = await engine.separate(decoded, {
            shifts: 1,
            overlap: 0.25,
            onProgress: ({ percent }: { percent: number }) => {
              if (!stillCurrent()) return;
              setProgress(30 + Math.max(0, Math.min(100, percent)) * 0.34);
              setStatus(`Separating vocals · ${Math.round(percent)}%`);
            },
          });
          vocalBuffer = separated.sources.vocals;
          engine.dispose();
        } catch (separationError) {
          console.warn("Demucs unavailable; using voice-focus fallback", separationError);
          if (!stillCurrent()) return;
          setUsedFallback(true);
          setStatus("Using a lighter vocal focus on this device");
          vocalBuffer = await voiceFocusFallback(decoded);
        }

        if (!stillCurrent() || !vocalBuffer) return;
        setProgress(66);
        const vocalBlob = audioBufferToWav(vocalBuffer);
        const nextVocalUrl = URL.createObjectURL(vocalBlob);
        objectUrlsRef.current.push(nextVocalUrl);
        setVocalUrl(nextVocalUrl);

        setStage("transcribing");
        setStatus("Preparing the vocal for Whisper");
        const waveform = await resampleToWhisper(vocalBuffer);
        if (!stillCurrent()) return;
        setProgress(70);

        const transcriber = await getWhisper(supportsWebGPU, (info) => {
          if (!stillCurrent() || info.status !== "progress") return;
          const modelProgress = Math.max(0, Math.min(100, info.progress ?? 0));
          setProgress(70 + modelProgress * 0.14);
          setStatus(`Loading Whisper · ${Math.round(modelProgress)}%`);
        });

        if (!stillCurrent()) return;
        setStatus("Listening for every word and its timing");
        setProgress(85);
        const slowProgress = window.setInterval(() => {
          if (!stillCurrent()) return;
          setProgress((value) => Math.min(96, value + 0.25));
        }, 1000);

        let result: WhisperOutput;
        try {
          result = await transcriber(waveform, {
            return_timestamps: "word",
            chunk_length_s: 30,
            stride_length_s: 5,
            task: "transcribe",
            ...(language === "auto" ? {} : { language }),
          });
        } finally {
          window.clearInterval(slowProgress);
        }

        if (!stillCurrent()) return;
        const timedWords = normalizeWords(result.chunks ?? []);
        const lyricLines = groupIntoLines(timedWords);
        if (!lyricLines.length) {
          throw new Error(
            "Whisper could not find clear sung words in this file. Try a track with a more present vocal.",
          );
        }

        setLines(lyricLines);
        setProgress(100);
        setStatus(`${timedWords.length} words synced on this device`);
        setStage("complete");
      } catch (processingError) {
        if (!stillCurrent()) return;
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
    [language, supportsWebGPU],
  );

  const loadFile = useCallback(
    (nextFile: File) => {
      if (!isAudioFile(nextFile)) {
        setError("Choose an MP3, WAV, FLAC, M4A, AAC, OGG, or WebM audio file.");
        return;
      }
      if (nextFile.size > MAX_FILE_SIZE) {
        setError("That file is over 500 MB. Choose a smaller audio file for browser processing.");
        return;
      }

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
      setError("");

      if (audioRef.current) {
        audioRef.current.src = nextOriginalUrl;
        audioRef.current.load();
      }
      void processTrack(nextFile, runId);
    },
    [clearObjectUrls, processTrack],
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
    runIdRef.current += 1;
    void processTrack(fileRef.current, runIdRef.current);
  };

  const reset = () => {
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
    setPlaybackMode("mix");
    setError("");
  };

  const changePlaybackMode = (nextMode: "mix" | "vocals") => {
    if (nextMode === "vocals" && !vocalUrl) return;
    const audio = audioRef.current;
    if (!audio || nextMode === playbackMode) return;
    const time = audio.currentTime;
    const shouldResume = !audio.paused;
    audio.pause();
    audio.src = nextMode === "vocals" ? vocalUrl : originalUrl;
    audio.load();
    audio.addEventListener(
      "loadedmetadata",
      () => {
        audio.currentTime = Math.min(time, audio.duration || time);
        setCurrentTime(audio.currentTime);
        if (shouldResume) void audio.play();
      },
      { once: true },
    );
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

  const stageRank = { idle: 0, decoding: 1, separating: 2, transcribing: 3, complete: 4, error: -1 }[
    stage
  ];

  const steps = [
    { label: "Open the mix", detail: "Decode locally", rank: 1, icon: FileAudio },
    { label: "Find the vocal", detail: "HTDemucs separation", rank: 2, icon: Mic2 },
    { label: "Sync every word", detail: "Whisper timestamps", rank: 3, icon: AudioLines },
  ];

  const renderVolumeIcon = () => {
    if (isMuted || volume === 0) return <VolumeX size={17} />;
    if (volume < 0.45) return <Volume1 size={17} />;
    return <Volume2 size={17} />;
  };

  return (
    <main className="app-shell">
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
            className="privacy-pill"
            type="button"
            onClick={() => setShowPrivacy((visible) => !visible)}
            aria-expanded={showPrivacy}
          >
            <LockKeyhole size={13} />
            100% on-device
          </button>
          {showPrivacy && (
            <div className="privacy-popover" role="status">
              <strong>Your song stays here.</strong>
              <span>Only the AI model files are downloaded. Your audio never leaves this browser.</span>
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
                <span className="eyebrow">PRIVATE KARAOKE ENGINE</span>
                <h1>Turn any song into live lyrics.</h1>
                <p>
                  Drop in a track. Vocal isolation and word-level transcription happen right on
                  your device.
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
                  Choose audio
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

              <p className="model-note">
                First run downloads about 430 MB of AI models. They are reused by your browser.
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
                  <span className="eyebrow">NOW IN THE ROOM</span>
                  <h1>{track.title}</h1>
                  <p>{track.artist}</p>
                  <span className="file-meta">
                    {formatFileSize(file.size)} · {duration ? formatTime(duration) : "reading…"}
                  </span>
                </div>
              </div>

              <div className="pipeline-card" aria-live="polite">
                <div className="pipeline-topline">
                  <span>{stage === "complete" ? "Ready to play" : stage === "error" ? "Needs attention" : "Making lyrics"}</span>
                  <strong>{Math.round(progress)}%</strong>
                </div>
                <div className="progress-track" aria-hidden="true">
                  <span style={{ width: `${progress}%` }} />
                </div>
                <p className="status-copy">{status}</p>

                <div className="steps-list">
                  {steps.map((step) => {
                    const StepIcon = step.icon;
                    const complete = stageRank > step.rank || stage === "complete";
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

                {usedFallback && (
                  <p className="fallback-note">
                    Full stem separation was not available, so this run used a lighter vocal-focus
                    filter.
                  </p>
                )}
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
            <div>
              <span className={`live-dot ${isPlaying ? "is-live" : ""}`} />
              <span>{stage === "complete" ? "LIVE LYRICS" : "LYRICS ROOM"}</span>
            </div>
            <button type="button" onClick={requestFullscreen} aria-label="Toggle fullscreen lyrics">
              <Maximize2 size={15} />
            </button>
          </div>

          <div className={`lyrics-scroll ${stage === "complete" ? "has-lyrics" : ""}`}>
            {stage === "complete" ? (
              <div className="lyrics-lines">
                {lines.map((line, lineIndex) => (
                  <button
                    className={`lyric-line ${lineIndex === activeLineIndex ? "is-active" : ""} ${lineIndex < activeLineIndex ? "is-past" : ""}`}
                    key={line.id}
                    type="button"
                    ref={(element) => {
                      lineRefs.current[lineIndex] = element;
                    }}
                    onClick={() => seekTo(line.start)}
                    aria-label={`Seek to ${formatTime(line.start)}: ${line.words.map((word) => word.text).join(" ")}`}
                  >
                    {line.words.map((word, wordIndex) => {
                      const wordDuration = Math.max(0.05, word.end - word.start);
                      const fill = Math.max(
                        0,
                        Math.min(1, (currentTime - word.start) / wordDuration),
                      );
                      const wordActive = currentTime >= word.start && currentTime < word.end;
                      const wordPast = currentTime >= word.end;
                      return (
                        <span
                          key={`${word.start}-${wordIndex}`}
                          className={`lyric-word ${wordActive ? "is-active" : ""} ${wordPast ? "is-past" : ""}`}
                          style={{ "--word-fill": `${fill * 100}%` } as CSSProperties}
                        >
                          {word.text}
                        </span>
                      );
                    })}
                  </button>
                ))}
              </div>
            ) : stage === "separating" ? (
              <div className="holding-lyrics processing-lyrics">
                <p>Pulling the voice</p>
                <p>
                  out of the <em>mix.</em>
                </p>
                <span>Keep this tab open while your device listens.</span>
              </div>
            ) : stage === "transcribing" ? (
              <div className="holding-lyrics processing-lyrics">
                <p>Listening for</p>
                <p>
                  every <em>word.</em>
                </p>
                <span>The first pass takes a little longer; the next one is faster.</span>
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
                <span>Click any finished line to seek straight to it.</span>
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
          <div className="source-switch" aria-label="Playback source">
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
          if (!isPlaying) setCurrentTime(event.currentTarget.currentTime);
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
