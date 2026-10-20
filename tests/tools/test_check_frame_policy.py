"""simACE keeps pandas inside the ADR 0015 allowlist.

The checker sat unwired for weeks and three violations landed unnoticed;
running it here keeps it from rotting again.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from check_frame_policy import check

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pandas_stays_inside_the_allowlist() -> None:
    violations, _, _ = check(REPO_ROOT)
    assert violations == []
