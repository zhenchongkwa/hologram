"""On-screen overlay: the effect label, readouts, and the hand skeleton."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import HudConfig
from .tracking import CONNECTIONS, Hand

_FONT = cv2.FONT_HERSHEY_SIMPLEX

HELP_LINES = (
    "n       next mode - shape, cube, slit-scan",
    "b       jump straight to cube mode, and back",
    "palms   open both hands in cube mode to grow the rose",
    "1 palm  one open hand alone - the rose, no box",
    "2 fing  index + middle, box down - a red orb",
    "g       bloom - bright areas bleed light",
    "tap     thumb to index on BOTH hands - next effect",
    "rotate  counter-rotate your hands to twist it into an X",
    "[ / ]   previous / next effect",
    "1-9     pick effect directly",
    "p       one shape per hand (shape mode)",
    "k / h   skeleton / overlay",
    "m       mirror",
    "s / r   screenshot / record",
    "c       reload config.toml",
    "q       quit",
)


@dataclass
class HudState:
    """What the overlay draws.

    ``readout`` is the mode's own words, written by the mode. This used to be a
    ``cube`` boolean plus every mode's fields flat - twist and crossed for one,
    bloom and solo for the other, each stale in the other's branch - and the
    draw below had to pick. A mode that describes itself costs hud.py nothing.
    """

    effect: str
    fps: float
    hands: int
    mode: str
    readout: str
    orb: bool
    recording: bool
    show_help: bool
    message: str = ""


def _text(frame, label, origin, color, scale=0.6, weight=1):
    """Draw text with a dark outline so it stays readable over any background."""
    cv2.putText(frame, label, origin, _FONT, scale, (0, 0, 0), weight + 3, cv2.LINE_AA)
    cv2.putText(frame, label, origin, _FONT, scale, color, weight, cv2.LINE_AA)


def draw_skeleton(frame: np.ndarray, hands: list[Hand], color) -> None:
    color = tuple(int(c) for c in color)
    for hand in hands:
        points = np.rint(hand.points).astype(int)
        for start, end in CONNECTIONS:
            cv2.line(frame, tuple(points[start]), tuple(points[end]), color, 1, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, tuple(point), 3, color, -1, cv2.LINE_AA)


def draw(frame: np.ndarray, state: HudState, cfg: HudConfig) -> None:
    color = tuple(int(c) for c in cfg.color)
    height = frame.shape[0]

    # The effect name, top-left, as in the original video.
    _text(frame, f"Effect: {state.effect}", (18, 40), color, scale=0.85, weight=2)

    readout = f"{state.fps:5.1f} fps   hands {state.hands}   {state.mode}"
    if state.readout:
        readout += f"   {state.readout}"
    if state.orb:
        readout += "   ORB"
    _text(frame, readout, (18, 68), color, scale=0.5)

    if state.recording:
        cv2.circle(frame, (28, 92), 7, (60, 60, 255), -1, cv2.LINE_AA)
        _text(frame, "REC", (44, 98), (60, 60, 255), scale=0.55, weight=2)

    if state.message:
        _text(frame, state.message, (18, height - 24), color, scale=0.55)

    if state.show_help:
        top = 128
        for index, line in enumerate(HELP_LINES):
            _text(frame, line, (18, top + index * 22), color, scale=0.45)
    else:
        _text(frame, "? for keys", (18, 118), color, scale=0.42)
