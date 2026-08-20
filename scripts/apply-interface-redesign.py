from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label}, found {count}.")
    return text.replace(old, new, 1)


def patch_page() -> None:
    path = ROOT / "app" / "page.tsx"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '    <main className="app-shell">',
        '    <main className={`app-shell ${file ? "has-track" : "is-empty"}`}>',
        "app shell state class",
    )
    text = replace_once(
        text,
        '<span className="eyebrow">PRIVATE KARAOKE ENGINE</span>',
        '<span className="eyebrow">PRIVATE · LOCAL · YOURS</span>',
        "hero eyebrow",
    )
    text = replace_once(
        text,
        "                <h1>Turn any song into live lyrics.</h1>",
        "                <h1>\n"
        "                  Turn any song into <span>live lyrics.</span>\n"
        "                </h1>",
        "hero heading",
    )
    text = replace_once(
        text,
        "                <p>\n"
        "                  Drop in a track. Vocal isolation and word-level transcription happen right on\n"
        "                  your device.\n"
        "                </p>",
        "                <p>\n"
        "                  Drop a track. Lyricwave separates the vocal and syncs every word without\n"
        "                  sending the song away from this machine.\n"
        "                </p>",
        "hero description",
    )
    text = replace_once(text, "                  Choose audio", "                  Choose a song", "upload CTA")
    text = replace_once(
        text,
        '                    ? "Recommended for testing. Accurate mode adds a slower separation and transcription pass."',
        '                    ? "Fast is the best starting point. Accurate adds a slower second pass."',
        "fast mode note",
    )
    text = replace_once(
        text,
        '                    : "Accurate mode uses the RTX GPU and caches several GB of model files after its first run."}',
        '                    : "Accurate uses the full local models and keeps downloads cached for later runs."}',
        "accurate mode note",
    )
    text = replace_once(text, "NOW IN THE ROOM", "NOW PLAYING", "loaded track eyebrow")
    path.write_text(text, encoding="utf-8")


def patch_frontend_test() -> None:
    path = ROOT / "tests" / "rendered-html.test.mjs"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "  assert.match(html, /Turn any song into live lyrics/);",
        "  assert.match(html, /Turn any song into/);\n"
        "  assert.match(html, /live lyrics\\./);",
        "rendered hero assertion",
    )
    text = replace_once(
        text,
        '  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");\n'
        '  const timeline = await readFile(new URL("../app/lyric-timeline.js", import.meta.url), "utf8");',
        '  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");\n'
        '  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");\n'
        '  const timeline = await readFile(new URL("../app/lyric-timeline.js", import.meta.url), "utf8");',
        "stylesheet fixture",
    )
    text = replace_once(
        text,
        "  assert.match(page, /URL\\.createObjectURL/);",
        "  assert.match(page, /URL\\.createObjectURL/);\n"
        '  assert.match(page, /file \\? "has-track" : "is-empty"/);\n'
        "  assert.match(styles, /\\.app-shell\\.is-empty \\.lyrics-stage/);\n"
        "  assert.match(styles, /\\.intro-copy h1 span/);\n"
        "  assert.doesNotMatch(styles, /float-glow|spin-orbit/);",
        "minimal interface architecture assertions",
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    patch_page()
    patch_frontend_test()
