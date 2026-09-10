"""Hand tracking built on the MediaPipe Tasks HandLandmarker.

The TikTok this is based on uses the legacy ``mp.solutions.hands`` API, which is
deprecated. This uses the Tasks API instead, which is the supported path and
costs one downloaded model file (see scripts/download_model.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import TrackingConfig

# Landmark indices in the 21-point MediaPipe hand model.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20

# Bones, for the skeleton overlay.
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


@dataclass
class Hand:
    """One detected hand, in pixel coordinates."""

    points: np.ndarray  # (21, 2) float32
    label: str          # "Left" or "Right", from the viewer's point of view
    score: float

    @property
    def scale(self) -> float:
        """Wrist-to-middle-knuckle length.

        Every gesture measurement is divided by this so that leaning towards or
        away from the camera doesn't change what a gesture means.
        """
        span = float(np.linalg.norm(self.points[MIDDLE_MCP] - self.points[WRIST]))
        return max(span, 1e-6)


class Ema:
    """Exponential moving average, for scalars or vectors."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.value: float | np.ndarray | None = None

    def update(self, sample):
        if isinstance(sample, np.ndarray):
            sample = sample.astype(np.float32)
        if self.value is None:
            self.value = sample
        else:
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class HandTracker:
    """Thin wrapper over HandLandmarker in VIDEO mode."""

    def __init__(
        self,
        cfg: TrackingConfig,
        model_path: str | Path | None = None,
        mirrored: bool = True,
    ) -> None:
        # Imported here so that importing this module (for the offline tests)
        # doesn't require mediapipe to be installed.
        import mediapipe as mp

        path = Path(model_path) if model_path else DEFAULT_MODEL
        if not path.exists():
            raise FileNotFoundError(
                f"Hand landmark model not found at {path}.\n"
                "Download it first:  python scripts/download_model.py"
            )

        self._mp = mp
        # Public so the app can keep it in step with the mirror toggle.
        self.mirrored = mirrored
        self._timestamp = 0

        base = mp.tasks.BaseOptions(model_asset_path=str(path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=cfg.detection_confidence,
            min_hand_presence_confidence=cfg.presence_confidence,
            min_tracking_confidence=cfg.tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> list[Hand]:
        """Detect hands in a BGR frame and return them in pixel coordinates."""
        import cv2

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        # detect_for_video requires strictly increasing timestamps.
        if timestamp_ms is None or timestamp_ms <= self._timestamp:
            timestamp_ms = self._timestamp + 1
        self._timestamp = timestamp_ms

        result = self._landmarker.detect_for_video(image, timestamp_ms)

        hands: list[Hand] = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            points = np.array(
                [(lm.x * width, lm.y * height) for lm in landmarks], dtype=np.float32
            )
            category = handedness[0] if handedness else None
            label = category.category_name if category else "Unknown"
            score = float(category.score) if category else 0.0
            # We feed the tracker a mirrored frame for the selfie view, which
            # flips what MediaPipe considers left and right. Undo that so the
            # label matches the hand the user is actually holding up.
            if self.mirrored:
                label = {"Left": "Right", "Right": "Left"}.get(label, label)
            hands.append(Hand(points=points, label=label, score=score))
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
