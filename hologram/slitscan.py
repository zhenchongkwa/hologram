"""Time slit-scan - every column of the picture from a different moment.

A ring of recent frames is kept, and each output column is read from a different
one, so a movement sweeping across the frame is smeared out into a record of
when it happened rather than where it is. Your hand drags the scan across, which
is what makes it a thing you play with rather than a filter.

Two measurements shape it. A 30-frame history at half resolution is 20.7 MB,
which is affordable, and the column gather over it costs 1.82 ms - against 4.4
ms to scale the result back up. So the expensive half is not the effect but the
upscale, and the gather is arranged to add as little as possible to it: no
transpose, no freshly allocated output, no per-pixel index array.

The trick that gets it there is that the age map runs along x only. Every column
asking for the same frame is a run of adjacent columns, so the gather is a
handful of slice copies out of the ring - plain memcpy - rather than a fancy
index over half a million pixels.

Before the ring has filled it holds copies of the first frame, so the mode opens
looking normal and dissolves into the effect over its first second, instead of
starting as a black screen.
"""

from __future__ import annotations

import numpy as np

import cv2


class SlitScan:
    """A ring of recent frames, read back one column at a time.

    The buffer is allocated once, at the first frame's working size, and reused
    - so the per-frame cost is the gather and nothing else. A change of frame
    size or history length rebuilds it, which only happens on a config reload.
    """

    def __init__(self, length: int = 30, scale: float = 0.5) -> None:
        self.length = max(int(length), 2)
        self.scale = float(np.clip(scale, 0.1, 1.0))
        self._ring: np.ndarray | None = None
        self._next = 0          # slot the next frame goes into
        self._out: np.ndarray | None = None

    @property
    def frames(self) -> int:
        return 0 if self._ring is None else len(self._ring)

    def reset(self) -> None:
        """Drop the history. The next frame refills it with copies of itself."""
        self._ring = None
        self._next = 0

    def working_size(self, height: int, width: int) -> tuple[int, int]:
        return (max(int(round(height * self.scale)), 8),
                max(int(round(width * self.scale)), 8))

    def push(self, small: np.ndarray) -> None:
        """Add one working-size frame to the ring."""
        if self._ring is None or self._ring.shape[1:] != small.shape:
            # Filled with copies of this frame rather than left black, so the
            # mode opens on a normal picture and dissolves into the effect over
            # its first second. Starting black reads as a bug.
            self._ring = np.repeat(small[None], self.length, axis=0).copy()
            self._out = np.empty_like(small)
            self._next = 0
        self._ring[self._next] = small
        self._next = (self._next + 1) % len(self._ring)

    def gather(self, ages: np.ndarray) -> np.ndarray:
        """Read one column per entry of ``ages``, into the reused output buffer.

        ``ages`` is per column, in frames back from the newest: 0 is now, 1 is
        the frame before it, and so on.

        Columns wanting the same frame are copied as one slice. The age map runs
        along x and moves smoothly, so those runs are long - a 30-frame history
        over 640 columns is about 30 memcpys, against a fancy index that would
        build a 640x360 index array every frame and gather through it.
        """
        if self._ring is None or self._out is None:
            raise RuntimeError("push a frame before gathering")

        count = len(self._ring)
        newest = (self._next - 1) % count
        slots = (newest - np.clip(np.asarray(ages), 0, count - 1).astype(np.int64)) % count

        out = self._out
        # Run boundaries: where the slot changes from one column to the next.
        edges = np.flatnonzero(np.diff(slots)) + 1
        start = 0
        for end in (*edges, len(slots)):
            out[:, start:end] = self._ring[slots[start]][:, start:end]
            start = end
        return out


def age_map(width: int, length: int, centre: float, spread: float,
            direction: int = 1) -> np.ndarray:
    """How far back each column reads, in frames.

    ``centre`` is where the newest moment sits, as a fraction of the width -
    that is the column your hand is over. Columns either side of it read
    further and further back, wrapping round the history, so the scan is a
    band of *now* that you drag across a picture made of *then*.
    """
    length = max(int(length), 2)
    columns = np.arange(int(width), dtype=np.float32) / max(int(width) - 1, 1)
    offset = (columns - float(centre)) * float(direction) * float(spread)
    # Scaled and wrapped by `length`, not `length - 1`. Both off by one, the
    # oldest frame in the ring was never read at all - a 30-frame history that
    # only ever showed 29 of them, and paid for the thirtieth.
    return np.mod(offset * length, length).astype(np.int64)


def render(frame: np.ndarray, scan: SlitScan, centre: float, spread: float,
           direction: int = 1) -> None:
    """Push this frame into the ring and draw the scan back out. In place."""
    height, width = frame.shape[:2]
    small_h, small_w = scan.working_size(height, width)

    if (small_w, small_h) == (width, height):
        small = frame.copy()
    else:
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    scan.push(small)

    ages = age_map(small_w, scan.frames, centre, spread, direction)
    scanned = scan.gather(ages)

    if scanned.shape[:2] == (height, width):
        frame[:] = scanned
    else:
        cv2.resize(scanned, (width, height), dst=frame, interpolation=cv2.INTER_LINEAR)
