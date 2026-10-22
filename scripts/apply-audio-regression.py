from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return text.replace(old, new, 1)


def create_audio_input() -> None:
    path = ROOT / "backend" / "audio_input.py"
    path.write_text(
        '''from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any


class AudioInputError(ValueError):
    """Raised when FFmpeg cannot identify or prepare a usable audio stream."""


def _last_error_line(stderr: str) -> str:
    return next((line.strip() for line in reversed(stderr.splitlines()) if line.strip()), "")


def _run(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
    except FileNotFoundError as error:
        raise AudioInputError(f"{command[0]} is not installed or is not on PATH.") from error
    except subprocess.TimeoutExpired as error:
        raise AudioInputError(f"{command[0]} timed out while reading the selected file.") from error
    except subprocess.CalledProcessError as error:
        detail = _last_error_line(error.stderr or "")
        message = "The selected file does not contain a decodable audio stream."
        if detail:
            message = f"{message} FFmpeg: {detail[:240]}"
        raise AudioInputError(message) from error


def probe_audio(path: Path) -> dict[str, Any]:
    """Return validated FFprobe metadata for the first audio stream.

    `-nostdin` is an ffmpeg option, not a portable ffprobe option. Supplying
    DEVNULL as stdin prevents interaction without making valid OGG/Vorbis files
    fail on ffprobe builds that reject that flag.
    """

    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index,codec_name,channels,sample_rate,duration:format=duration,format_name",
            "-of",
            "json",
            str(path),
        ],
        timeout=30,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise AudioInputError("FFprobe returned invalid metadata for the selected file.") from error

    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise AudioInputError("The selected file does not contain an audio stream.")
    return payload


def probe_audio_duration(path: Path) -> float:
    payload = probe_audio(path)
    stream = payload["streams"][0]
    raw_duration = stream.get("duration") or payload.get("format", {}).get("duration")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError) as error:
        raise AudioInputError("The selected audio duration could not be read.") from error
    if not math.isfinite(duration) or duration <= 0:
        raise AudioInputError("The selected audio duration is not valid.")
    return duration


def prepare_demucs_input(source_path: Path, work_dir: Path) -> Path:
    """Decode the first audio stream to an audio-only lossless FLAC for Demucs."""

    probe_audio(source_path)
    prepared = work_dir / "demucs-input.flac"
    prepared.unlink(missing_ok=True)
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-c:a",
            "flac",
            "-compression_level",
            "5",
            str(prepared),
        ],
        timeout=20 * 60,
    )
    if not prepared.is_file() or prepared.stat().st_size == 0:
        raise AudioInputError("FFmpeg did not create a usable audio stream for processing.")
    probe_audio_duration(prepared)
    return prepared
''',
        encoding="utf-8",
    )


def patch_server() -> None:
    path = ROOT / "backend" / "server.py"
    text = path.read_text(encoding="utf-8")
    if "from backend.audio_input import" not in text:
        text = replace_once(
            text,
            "from backend.ctc_alignment import release_alignment_model\n",
            "from backend.audio_input import AudioInputError, prepare_demucs_input, probe_audio_duration\n"
            "from backend.ctc_alignment import release_alignment_model\n",
            "audio input import",
        )

    old_probe = '''def _probe_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-nostdin",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    duration = float(completed.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Audio duration is not valid.")
    return duration
'''
    if old_probe in text:
        text = text.replace(old_probe, '''def _probe_duration(path: Path) -> float:
    return probe_audio_duration(path)
''', 1)

    old_error = '''        except (OSError, ValueError, subprocess.SubprocessError) as error:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail="The selected file could not be decoded as supported audio.",
            ) from error
'''
    if old_error in text:
        text = text.replace(
            old_error,
            '''        except AudioInputError as error:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(status_code=422, detail=str(error)) from error
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(
                status_code=422,
                detail="No decodable audio stream was found in the selected file.",
            ) from error
''',
            1,
        )

    old_start = '''def _separate_vocals(job: JobState) -> Path:
    model_name = demucs_model_name(job.quality)
    pass_count = demucs_pass_count(job.quality)
    output_root = job.work_dir / "separated"
    job.update(
        stage="separating",
        progress=6.0,
        status=f"Loading {model_name} on the GPU",
    )

    command = [
'''
    if old_start in text:
        text = text.replace(
            old_start,
            '''def _separate_vocals(job: JobState) -> Path:
    model_name = demucs_model_name(job.quality)
    pass_count = demucs_pass_count(job.quality)
    output_root = job.work_dir / "separated"
    job.update(
        stage="separating",
        progress=6.0,
        status="Preparing audio",
    )
    demucs_input = prepare_demucs_input(job.source_path, job.work_dir)
    _check_cancelled(job)
    job.update(status=f"Loading {model_name} on the GPU")

    command = [
''',
            1,
        )
    if "        str(job.source_path),\n    ]\n" in text:
        text = text.replace("        str(job.source_path),\n    ]\n", "        str(demucs_input),\n    ]\n", 1)
    path.write_text(text, encoding="utf-8")


def create_tests() -> None:
    path = ROOT / "backend" / "test_audio_input.py"
    path.write_text(
        '''from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.audio_input import AudioInputError, prepare_demucs_input, probe_audio_duration


class AudioInputTest(unittest.TestCase):
    def test_probe_uses_devnull_without_the_invalid_ffprobe_nostdin_flag(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({
                "streams": [{"index": 0, "codec_name": "vorbis", "duration": "2.5"}],
                "format": {"duration": "2.5", "format_name": "ogg"},
            }),
            stderr="",
        )
        with patch("backend.audio_input.subprocess.run", return_value=completed) as run:
            duration = probe_audio_duration(Path("song.ogg"))
        self.assertEqual(duration, 2.5)
        command = run.call_args.args[0]
        self.assertNotIn("-nostdin", command)
        self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertIn("a:0", command)

    def test_missing_audio_stream_has_a_specific_error(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"streams": [], "format": {}}', stderr=""
        )
        with patch("backend.audio_input.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(AudioInputError, "does not contain an audio stream"):
                probe_audio_duration(Path("cover-only.ogg"))

    def test_preparation_selects_only_audio_and_outputs_lossless_flac(self) -> None:
        probe_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({
                "streams": [{"index": 0, "codec_name": "vorbis", "duration": "1.0"}],
                "format": {"duration": "1.0", "format_name": "ogg"},
            }),
            stderr="",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "song.ogg"
            source.write_bytes(b"OggS fixture")

            def fake_run(command, **kwargs):
                if command[0] == "ffmpeg":
                    Path(command[-1]).write_bytes(b"fLaC prepared")
                    return subprocess.CompletedProcess(command, 0, "", "")
                return probe_result

            with patch("backend.audio_input.subprocess.run", side_effect=fake_run) as run:
                prepared = prepare_demucs_input(source, root)

            self.assertEqual(prepared.name, "demucs-input.flac")
            ffmpeg_command = next(call.args[0] for call in run.call_args_list if call.args[0][0] == "ffmpeg")
            self.assertIn("0:a:0", ffmpeg_command)
            self.assertIn("-vn", ffmpeg_command)
            self.assertIn("flac", ffmpeg_command)


if __name__ == "__main__":
    unittest.main()
''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    create_audio_input()
    patch_server()
    create_tests()
