from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return text.replace(old, new, 1)


def patch_frontend_tests() -> None:
    path = ROOT / "tests" / "rendered-html.test.mjs"
    text = path.read_text(encoding="utf-8")
    text = text.replace("assert.match(html, /Turn any song into live lyrics/);", "assert.match(html, /No audio selected/);")
    text = text.replace("assert.match(html, /Drop your song here/);", "assert.match(html, /Choose audio/);")
    marker = "  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);\n"
    additions = (
        "  assert.doesNotMatch(html, /PRIVATE KARAOKE ENGINE|Turn any song|light up every word|LYRICS ROOM/i);\n"
        "  assert.match(html, /large-v3-turbo/);\n"
        "  assert.match(html, />Best</);\n"
    )
    if additions not in text:
        text = replace_once(text, marker, marker + additions, "minimal UI assertions")
    backend_marker = '  const backend = await readFile(new URL("../backend/server.py", import.meta.url), "utf8");\n'
    audio_read = '  const audioInput = await readFile(new URL("../backend/audio_input.py", import.meta.url), "utf8");\n'
    if audio_read not in text:
        text = replace_once(text, backend_marker, backend_marker + audio_read, "audio module read")
    inference_marker = "  assert.match(backend, /backend\\.inference_worker/);\n"
    audio_assertions = (
        "  assert.match(audioInput, /stdin=subprocess\\.DEVNULL/);\n"
        "  assert.doesNotMatch(audioInput, /\\\"ffprobe\\\",\\s*\\\"-nostdin\\\"/);\n"
        "  assert.match(audioInput, /\\\"0:a:0\\\"/);\n"
    )
    if audio_assertions not in text:
        text = replace_once(text, inference_marker, inference_marker + audio_assertions, "audio assertions")
    path.write_text(text, encoding="utf-8")


def patch_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace("Fast is the recommended default. It uses:", "Fast prioritizes turnaround time. It uses:")
    text = text.replace("Accurate uses:", "Best quality uses:")
    needle = (
        "The engine validates processing mode, supported language, file extension, decoded\n"
        "audio duration, maximum size, and maximum duration before queuing GPU work."
    )
    replacement = (
        "The engine validates processing mode, supported language, file extension, decoded\n"
        "audio duration, maximum size, and maximum duration before queuing GPU work. FFprobe\n"
        "uses redirected stdin rather than the non-portable `-nostdin` flag. Before Demucs\n"
        "runs, the first audio stream is decoded to an audio-only lossless FLAC, so OGG/Vorbis\n"
        "files with embedded cover art do not reach model loaders as mixed-media containers."
    )
    if needle in text:
        text = text.replace(needle, replacement, 1)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    patch_frontend_tests()
    patch_readme()
