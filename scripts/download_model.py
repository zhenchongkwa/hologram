"""Fetch the MediaPipe hand landmark model.

The Tasks API needs its model bundle on disk. It's a few megabytes, so it is
downloaded once here rather than committed to the repo.

    python scripts/download_model.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DESTINATION = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def _progress(count: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(count * block_size, total)
    pct = done * 100 // total
    sys.stdout.write(f"\r  {pct:3d}%  {done / 1024:,.0f} / {total / 1024:,.0f} KB")
    sys.stdout.flush()


def main() -> int:
    if DESTINATION.exists():
        size = DESTINATION.stat().st_size
        print(f"Model already present: {DESTINATION} ({size / 1024:,.0f} KB)")
        return 0

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand_landmarker.task\n  from {URL}")

    # Download beside the target and rename at the end, so an interrupted run
    # can't leave a half-written file that looks valid to the app.
    partial = DESTINATION.with_suffix(".part")
    try:
        urllib.request.urlretrieve(URL, partial, reporthook=_progress)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        print(f"\nDownload failed: {exc}")
        return 1

    partial.replace(DESTINATION)
    print(f"\nSaved to {DESTINATION} ({DESTINATION.stat().st_size / 1024:,.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
