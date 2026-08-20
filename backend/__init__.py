"""Local lyricwave inference service and backend test discovery entrypoint."""

from __future__ import annotations

import unittest
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent


def load_tests(
    loader: unittest.TestLoader,
    _: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Load every backend test module for ``python -m unittest backend``."""

    suite = unittest.TestSuite()
    for test_path in sorted(_PACKAGE_ROOT.glob(pattern or "test*.py")):
        suite.addTests(loader.loadTestsFromName(f"{__name__}.{test_path.stem}"))
    return suite
