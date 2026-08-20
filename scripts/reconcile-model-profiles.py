from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} block in {path}, found {count}.")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_pipeline() -> None:
    path = ROOT / "backend" / "inference_pipeline.py"
    replace_once(
        path,
        "def _align_segment_words(\n"
        "    job: JobState,\n"
        "    segment: dict[str, Any],\n"
        "    raw_words: list[dict[str, Any]],\n"
        ") -> tuple[list[dict[str, Any]], float]:\n"
        "    if alignment_backend(job.quality) == \"qwen3\":\n",
        "def _align_segment_words(\n"
        "    job: JobState,\n"
        "    segment: dict[str, Any],\n"
        "    raw_words: list[dict[str, Any]],\n"
        ") -> tuple[list[dict[str, Any]], float]:\n"
        "    configured_backend = alignment_backend(job.quality)\n"
        "    if configured_backend == \"none\":\n"
        "        return raw_words, 0.0\n"
        "    if configured_backend == \"qwen3\":\n",
        "alignment disable route",
    )
    replace_once(
        path,
        "            use_english_alignment = _segment_uses_english_alignment(\n"
        "                job.language,\n"
        "                segment.get(\"language\"),\n"
        "            )\n"
        "            use_qwen_alignment = (\n"
        "                alignment_backend(job.quality) == \"qwen3\"\n",
        "            configured_alignment = alignment_backend(job.quality)\n"
        "            use_english_alignment = (\n"
        "                configured_alignment != \"none\"\n"
        "                and _segment_uses_english_alignment(\n"
        "                    job.language,\n"
        "                    segment.get(\"language\"),\n"
        "                )\n"
        "            )\n"
        "            use_qwen_alignment = (\n"
        "                configured_alignment == \"qwen3\"\n",
        "alignment selection route",
    )


def patch_model_runtime() -> None:
    path = ROOT / "backend" / "model_runtime.py"
    replace_once(
        path,
        "                transcription_backend=requested_backend,\n"
        "                transcription_model=requested_model_id.split(\"/\")[-1],\n",
        "                transcription_backend=requested_backend,\n"
        "                transcription_model=requested_model_id.split(\"/\")[-1],\n"
        "                transcription_model_id=requested_model_id,\n",
        "cached ASR metadata",
    )
    replace_once(
        path,
        "            transcription_backend=actual_backend,\n"
        "            transcription_model=actual_model_id.split(\"/\")[-1],\n",
        "            transcription_backend=actual_backend,\n"
        "            transcription_model=actual_model_id.split(\"/\")[-1],\n"
        "            transcription_model_id=actual_model_id,\n",
        "loaded ASR metadata",
    )


def patch_qwen_alignment() -> None:
    path = ROOT / "backend" / "qwen_alignment.py"
    replace_once(
        path,
        "        job.update(alignment_model=model_id.split(\"/\")[-1])\n",
        "        job.update(\n"
        "            alignment_model=model_id.split(\"/\")[-1],\n"
        "            alignment_model_id=model_id,\n"
        "        )\n",
        "Qwen alignment metadata",
    )


def patch_ctc_alignment() -> None:
    path = ROOT / "backend" / "ctc_alignment.py"
    replace_once(
        path,
        "def _alignment_model_id(quality: str) -> str:\n"
        "    mode = \"accurate\" if quality == \"accurate\" else \"fast\"\n"
        "    mode_override = os.environ.get(f\"LYRICWAVE_{mode.upper()}_ALIGNER_MODEL\", \"\").strip()\n"
        "    shared_override = os.environ.get(\"LYRICWAVE_ALIGNER_MODEL\", \"\").strip()\n"
        "    return mode_override or shared_override or _DEFAULT_ALIGNMENT_MODELS[mode]\n\n\n",
        "def alignment_model_id(quality: str) -> str:\n"
        "    mode = \"accurate\" if quality == \"accurate\" else \"fast\"\n"
        "    mode_override = os.environ.get(f\"LYRICWAVE_{mode.upper()}_ALIGNER_MODEL\", \"\").strip()\n"
        "    shared_override = os.environ.get(\"LYRICWAVE_ALIGNER_MODEL\", \"\").strip()\n"
        "    return mode_override or shared_override or _DEFAULT_ALIGNMENT_MODELS[mode]\n\n\n"
        "_alignment_model_id = alignment_model_id\n\n\n",
        "public CTC model ID",
    )
    replace_once(
        path,
        "    requested = _alignment_model_id(mode)\n",
        "    requested = alignment_model_id(mode)\n",
        "CTC candidate request",
    )
    replace_once(
        path,
        "    requested_model_id = _alignment_model_id(job.quality)\n",
        "    requested_model_id = alignment_model_id(job.quality)\n",
        "CTC load request",
    )
    replace_once(
        path,
        "        job.update(alignment_model=model_id.split(\"/\")[-1])\n",
        "        job.update(\n"
        "            alignment_model=model_id.split(\"/\")[-1],\n"
        "            alignment_model_id=model_id,\n"
        "        )\n",
        "CTC actual metadata",
    )


def patch_job_state() -> None:
    path = ROOT / "backend" / "job_state.py"
    replace_once(
        path,
        "    transcription_backend: str = \"\"\n"
        "    transcription_model: str = \"\"\n"
        "    alignment_model: str = \"\"\n",
        "    transcription_backend: str = \"\"\n"
        "    transcription_model: str = \"\"\n"
        "    transcription_model_id: str = \"\"\n"
        "    alignment_model: str = \"\"\n"
        "    alignment_model_id: str = \"\"\n",
        "job model fields",
    )
    replace_once(
        path,
        "                    \"transcription_backend\": self.transcription_backend,\n"
        "                    \"transcription_model\": self.transcription_model,\n"
        "                    \"alignment_model\": self.alignment_model,\n",
        "                    \"transcription_backend\": self.transcription_backend,\n"
        "                    \"transcription_model\": self.transcription_model,\n"
        "                    \"transcription_model_id\": self.transcription_model_id,\n"
        "                    \"alignment_model\": self.alignment_model,\n"
        "                    \"alignment_model_id\": self.alignment_model_id,\n",
        "progress model metadata",
    )
    replace_once(
        path,
        "        with self.lock:\n"
        "            configured_alignment = alignment_model_id(self.quality)\n"
        "            payload: dict[str, Any] = {\n",
        "        with self.lock:\n"
        "            configured_transcription = asr_model_id(self.quality)\n"
        "            configured_alignment = alignment_model_id(self.quality)\n"
        "            actual_transcription_id = self.transcription_model_id or configured_transcription\n"
        "            actual_alignment_id = self.alignment_model_id or configured_alignment\n"
        "            payload: dict[str, Any] = {\n",
        "public configured metadata",
    )
    replace_once(
        path,
        "                \"duration\": self.duration,\n"
        "                \"quality\": self.quality,\n"
        "                \"device\": self.device,\n"
        "                \"separation_model\": demucs_model_name(self.quality),\n"
        "                \"transcription_backend\": self.transcription_backend or asr_backend(self.quality),\n"
        "                \"transcription_model\": self.transcription_model or asr_model_id(self.quality).split(\"/\")[-1],\n"
        "                \"alignment_model\": self.alignment_model or configured_alignment.split(\"/\")[-1],\n"
        "                \"created_at\": self.created_at,\n",
        "                \"duration\": self.duration,\n"
        "                \"quality\": self.quality,\n"
        "                \"language\": self.language,\n"
        "                \"device\": self.device,\n"
        "                \"separation_model\": demucs_model_name(self.quality),\n"
        "                \"transcription_backend\": self.transcription_backend or asr_backend(self.quality),\n"
        "                \"transcription_model\": self.transcription_model or actual_transcription_id.split(\"/\")[-1],\n"
        "                \"transcription_model_id\": actual_transcription_id,\n"
        "                \"alignment_model\": self.alignment_model or actual_alignment_id.split(\"/\")[-1],\n"
        "                \"alignment_model_id\": actual_alignment_id,\n"
        "                \"alignment_model_requested\": configured_alignment,\n"
        "                \"created_at\": self.created_at,\n",
        "public model metadata",
    )


def patch_worker() -> None:
    path = ROOT / "backend" / "inference_worker.py"
    replace_once(
        path,
        "        \"transcription_backend\": job.transcription_backend,\n"
        "        \"transcription_model\": job.transcription_model,\n"
        "        \"alignment_model\": job.alignment_model,\n",
        "        \"transcription_backend\": job.transcription_backend,\n"
        "        \"transcription_model\": job.transcription_model,\n"
        "        \"transcription_model_id\": job.transcription_model_id,\n"
        "        \"alignment_model\": job.alignment_model,\n"
        "        \"alignment_model_id\": job.alignment_model_id,\n",
        "worker result metadata",
    )


def patch_server() -> None:
    path = ROOT / "backend" / "server.py"
    replace_once(
        path,
        "_MUTATING_METHODS = frozenset({\"POST\", \"PUT\", \"PATCH\", \"DELETE\"})\n",
        "_MUTATING_METHODS = frozenset({\"POST\", \"PUT\", \"PATCH\", \"DELETE\"})\n"
        "_WORKER_MODEL_FIELDS = (\n"
        "    \"transcription_backend\",\n"
        "    \"transcription_model\",\n"
        "    \"transcription_model_id\",\n"
        "    \"alignment_model\",\n"
        "    \"alignment_model_id\",\n"
        ")\n",
        "worker model fields",
    )
    replace_once(
        path,
        "def _release_models() -> None:\n",
        "def _worker_model_updates(payload: object) -> dict[str, str]:\n"
        "    if not isinstance(payload, dict):\n"
        "        return {}\n"
        "    updates: dict[str, str] = {}\n"
        "    for key in _WORKER_MODEL_FIELDS:\n"
        "        value = payload.get(key)\n"
        "        if isinstance(value, str) and value.strip():\n"
        "            updates[key] = value.strip()\n"
        "    return updates\n\n\n"
        "def _release_models() -> None:\n",
        "worker metadata parser",
    )
    replace_once(
        path,
        "            device=str(metadata.get(\"device\", \"Local GPU\")),\n"
        "        )\n",
        "            device=str(metadata.get(\"device\", \"Local GPU\")),\n"
        "            transcription_backend=str(result.get(\"transcription_backend\") or \"\"),\n"
        "            transcription_model=str(result.get(\"transcription_model\") or \"\"),\n"
        "            transcription_model_id=str(\n"
        "                result.get(\"transcription_model_id\")\n"
        "                or result.get(\"transcription_model\")\n"
        "                or \"\"\n"
        "            ),\n"
        "            alignment_model=str(result.get(\"alignment_model\") or \"\"),\n"
        "            alignment_model_id=str(\n"
        "                result.get(\"alignment_model_id\")\n"
        "                or result.get(\"alignment_model\")\n"
        "                or \"\"\n"
        "            ),\n"
        "        )\n",
        "restored model metadata",
    )
    replace_once(
        path,
        "            raise RuntimeError(\n"
        "                \"Whisper could not find clear sung words. Try Accurate mode and \"\n"
        "                \"set the lyrics language explicitly.\"\n"
        "            )\n",
        "            raise RuntimeError(\n"
        "                \"The selected transcription profile could not find clear sung words. \"\n"
        "                \"Try Best quality and set the lyrics language explicitly.\"\n"
        "            )\n",
        "generic empty transcript error",
    )
    replace_once(
        path,
        "                    updates = {\n"
        "                        key: progress[key]\n"
        "                        for key in (\"stage\", \"progress\", \"status\", \"error\")\n"
        "                        if key in progress\n"
        "                    }\n",
        "                    updates = {\n"
        "                        key: progress[key]\n"
        "                        for key in (\"stage\", \"progress\", \"status\", \"error\")\n"
        "                        if key in progress\n"
        "                    }\n"
        "                    updates.update(_worker_model_updates(progress))\n",
        "progress model metadata",
    )
    replace_once(
        path,
        "    if return_code != 0 or not result.get(\"ok\"):\n"
        "        raise RuntimeError(\n",
        "    model_updates = _worker_model_updates(result)\n"
        "    if model_updates:\n"
        "        job.update(**model_updates)\n\n"
        "    if return_code != 0 or not result.get(\"ok\"):\n"
        "        raise RuntimeError(\n",
        "result model metadata",
    )


def patch_page() -> None:
    path = ROOT / "app" / "page.tsx"
    replace_once(
        path,
        "} from \"./local-engine.js\";\n"
        "import { wordFillAt } from \"./lyric-timing.js\";\n",
        "} from \"./local-engine.js\";\n"
        "import { buildTimingExport } from \"./lyric-export.js\";\n"
        "import { wordFillAt } from \"./lyric-timing.js\";\n",
        "timing export import",
    )
    replace_once(
        path,
        "  words?: TimedWord[];\n"
        "  lines?: LyricLine[];\n"
        "  separation_model?: string;\n"
        "  transcription_model?: string;\n"
        "  alignment_model?: string;\n"
        "};\n",
        "  words?: TimedWord[];\n"
        "  lines?: LyricLine[];\n"
        "  quality?: QualityPreset;\n"
        "  language?: string;\n"
        "  created_at?: string;\n"
        "  separation_model?: string;\n"
        "  transcription_backend?: string;\n"
        "  transcription_model?: string;\n"
        "  transcription_model_id?: string;\n"
        "  alignment_model?: string;\n"
        "  alignment_model_id?: string;\n"
        "  alignment_model_requested?: string;\n"
        "};\n",
        "backend job model metadata",
    )
    replace_once(
        path,
        "type BackendHealth = {\n",
        "type ProcessingMetadata = {\n"
        "  jobId: string;\n"
        "  createdAt: string | null;\n"
        "  quality: QualityPreset;\n"
        "  languageSetting: string;\n"
        "  device: string | null;\n"
        "  separationModel: string | null;\n"
        "  transcriptionBackend: string | null;\n"
        "  transcriptionModel: string | null;\n"
        "  alignmentModel: string | null;\n"
        "  alignmentModelRequested: string | null;\n"
        "};\n\n"
        "type BackendHealth = {\n",
        "processing metadata type",
    )
    replace_once(
        path,
        "  const [engineDetail, setEngineDetail] = useState(\"\");\n"
        "  const [focusedLineIndex, setFocusedLineIndex] = useState(-1);\n",
        "  const [engineDetail, setEngineDetail] = useState(\"\");\n"
        "  const [processingMetadata, setProcessingMetadata] = useState<ProcessingMetadata | null>(\n"
        "    null,\n"
        "  );\n"
        "  const [focusedLineIndex, setFocusedLineIndex] = useState(-1);\n",
        "processing metadata state",
    )
    replace_once(
        path,
        "      setLines([]);\n"
        "      setEngineDetail(\"\");\n"
        "      setStage(\"uploading\");\n",
        "      setLines([]);\n"
        "      setEngineDetail(\"\");\n"
        "      setProcessingMetadata(null);\n"
        "      setStage(\"uploading\");\n",
        "processing metadata reset",
    )
    replace_once(
        path,
        "            setLines(lyricLines);\n"
        "            setProgress(100);\n",
        "            setProcessingMetadata({\n"
        "              jobId: job.id,\n"
        "              createdAt: job.created_at ?? null,\n"
        "              quality: job.quality ?? quality,\n"
        "              languageSetting: job.language ?? language,\n"
        "              device: job.device ?? null,\n"
        "              separationModel: job.separation_model ?? null,\n"
        "              transcriptionBackend: job.transcription_backend ?? null,\n"
        "              transcriptionModel:\n"
        "                job.transcription_model_id ?? job.transcription_model ?? null,\n"
        "              alignmentModel:\n"
        "                job.alignment_model_id ?? job.alignment_model ?? null,\n"
        "              alignmentModelRequested: job.alignment_model_requested ?? null,\n"
        "            });\n"
        "            setLines(lyricLines);\n"
        "            setProgress(100);\n",
        "completed processing metadata",
    )
    replace_once(
        path,
        "      setVocalUrl(\"\");\n"
        "      setPlaybackMode(\"mix\");\n",
        "      setVocalUrl(\"\");\n"
        "      setProcessingMetadata(null);\n"
        "      setPlaybackMode(\"mix\");\n",
        "new file metadata reset",
    )
    replace_once(
        path,
        "    setLines([]);\n"
        "    setOriginalUrl(\"\");\n",
        "    setLines([]);\n"
        "    setProcessingMetadata(null);\n"
        "    setOriginalUrl(\"\");\n",
        "reset metadata",
    )
    replace_once(
        path,
        "    if (kind === \"json\") {\n"
        "      const payload = {\n"
        "        title: track.title,\n"
        "        artist: track.artist,\n"
        "        duration,\n"
        "        generatedOnDevice: true,\n"
        "        lines,\n"
        "      };\n",
        "    if (kind === \"json\") {\n"
        "      const payload = buildTimingExport({\n"
        "        title: track.title,\n"
        "        artist: track.artist,\n"
        "        duration,\n"
        "        processing: processingMetadata,\n"
        "        lines,\n"
        "      });\n",
        "timing JSON export",
    )
    for old, new, label in (
        ("aria-label=\"Load another song\"", "aria-label=\"Load another audio file\"", "header reset label"),
        ("Choose a song", "Choose audio", "empty-state action"),
        ("aria-label=\"Live lyrics\"", "aria-label=\"Lyrics\"", "lyrics region label"),
    ):
        replace_once(path, old, new, label)


def patch_readme() -> None:
    path = ROOT / "README.md"
    replace_once(
        path,
        "Reference and candidate files may use lyricwave's exported `lines` structure or a\n"
        "top-level `words` array. Text-only references still produce WER/CER; timing metrics are\n"
        "calculated only for exact aligned words where both files contain start and end times.\n",
        "Reference and candidate files may use lyricwave's exported `lines` structure or a\n"
        "top-level `words` array. Timing exports include the selected profile, language, device,\n"
        "Demucs model, actual transcription backend/model, actual alignment model, and requested\n"
        "alignment checkpoint. Text-only references still produce WER/CER; timing metrics are\n"
        "calculated only for exact aligned words where both files contain start and end times.\n",
        "export provenance documentation",
    )


def main() -> None:
    patch_pipeline()
    patch_model_runtime()
    patch_qwen_alignment()
    patch_ctc_alignment()
    patch_job_state()
    patch_worker()
    patch_server()
    patch_page()
    patch_readme()


if __name__ == "__main__":
    main()
