"""The shape the effect is drawn inside, and how it follows your fingertips.

With two hands the shape is a quad through four points: each hand's thumb tip
and index tip. Move your fingers and the shape stretches and skews with them -
it is not a rigid rectangle, so it has no angle, length or thickness. The four
points are smoothed individually, which is also why none of the angle-wrapping
care that a rotating rectangle needs applies here.

With one hand, the thumb-to-index span becomes an ellipse instead, since two
points cannot enclose anything on their own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import effects
from .config import ShapeConfig, TrackingConfig
from .effects import EffectContext
from .tracking import INDEX_TIP, THUMB_TIP, Ema, Hand


@dataclass
class Shape:
    """A polygon to draw the effect inside."""

    points: np.ndarray   # (N, 2) float32, in drawing order
    anchors: np.ndarray  # the fingertips it was built from, for the markers

    def polygon(self) -> np.ndarray:
        return np.rint(self.points).astype(np.int32)

    @property
    def crossed(self) -> bool:
        """True once the shape has twisted far enough to cross into an X."""
        if len(self.points) != 4:
            return False
        # Corner order is thumb_a, index_a, index_b, thumb_b, so the rails are
        # edge 1->2 (index to index) and edge 3->0 (thumb to thumb).
        return _segments_cross(self.points[1], self.points[2],
                               self.points[3], self.points[0])

    @property
    def size(self) -> float:
        """Largest extent in pixels - how open the shape is."""
        if len(self.points) == 0:
            return 0.0
        extent = self.points.max(axis=0) - self.points.min(axis=0)
        return float(extent.max())


def ribbon_quad(thumb_a, index_a, thumb_b, index_b) -> np.ndarray:
    """Corner order that lets the shape twist.

    Corners are joined by which fingertip they are, so the two long edges are
    rails - index to index, and thumb to thumb. Counter-rotate your hands and
    one rail is dragged across the other, pinching the shape shut and crossing
    it into an X, exactly the way a twisted ribbon behaves.

    Sorting the corners by angle about their centre instead would always give a
    simple convex polygon. That sounds safer and is what an earlier version did,
    but it quietly untwists the shape - the crossing can never appear.
    """
    return np.array([thumb_a, index_a, index_b, thumb_b], dtype=np.float32)


def _segments_cross(p1, p2, p3, p4) -> bool:
    """Whether segment p1-p2 crosses p3-p4, by orientation sign."""
    def side(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = side(p3, p4, p1), side(p3, p4, p2)
    d3, d4 = side(p1, p2, p3), side(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _expand(points: np.ndarray, factor: float) -> np.ndarray:
    """Push the outline out past the fingertips a little."""
    if factor <= 0:
        return points
    centre = points.mean(axis=0)
    return centre + (points - centre) * (1.0 + factor)


class ShapeBuilder:
    """Turns tracked hands into a smoothed shape."""

    def __init__(self, shape_cfg: ShapeConfig, tracking_cfg: TrackingConfig) -> None:
        self.shape = shape_cfg
        self.tracking = tracking_cfg
        self._points: dict[str, Ema] = {}

    def reconfigure(self, shape_cfg: ShapeConfig, tracking_cfg: TrackingConfig) -> None:
        self.shape = shape_cfg
        self.tracking = tracking_cfg
        self._points.clear()

    def build(self, hands: list[Hand], per_hand: bool = False) -> list[Shape]:
        if not hands:
            self._points.clear()
            return []

        live: set[str] = set()
        keyed = self._keyed(hands)

        if per_hand or len(hands) == 1:
            shapes = []
            for key, hand in keyed:
                thumb = self._smooth(f"{key}:thumb", hand.points[THUMB_TIP], live)
                index = self._smooth(f"{key}:index", hand.points[INDEX_TIP], live)
                lens = self._ellipse(thumb, index)
                if lens is not None:
                    shapes.append(lens)
            self._prune(live)
            return shapes

        tips = []
        for key, hand in keyed[:2]:
            tips.append(self._smooth(f"{key}:thumb", hand.points[THUMB_TIP], live))
            tips.append(self._smooth(f"{key}:index", hand.points[INDEX_TIP], live))
        self._prune(live)

        quad = ribbon_quad(*tips)
        anchors = np.array(tips, dtype=np.float32)
        return [Shape(points=_expand(quad, self.shape.expand), anchors=anchors)]

    @staticmethod
    def _keyed(hands: list[Hand]) -> list[tuple[str, Hand]]:
        """Give each hand a key that stays the same from frame to frame.

        Keyed by handedness plus how many hands already carried that label,
        because MediaPipe occasionally labels both hands the same and sharing
        smoothers between two hands drags each towards the other.
        """
        seen: dict[str, int] = {}
        keyed = []
        for hand in hands:
            index = seen.get(hand.label, 0)
            seen[hand.label] = index + 1
            keyed.append((f"{hand.label}#{index}", hand))
        return keyed

    def _smooth(self, key: str, value: np.ndarray, live: set[str]) -> np.ndarray:
        live.add(key)
        ema = self._points.get(key)
        if ema is None:
            ema = self._points[key] = Ema(self.tracking.point_smoothing)
        return np.asarray(ema.update(np.asarray(value, dtype=np.float32)), dtype=np.float32)

    def _prune(self, live: set[str]) -> None:
        for key in set(self._points) - live:
            del self._points[key]

    def _ellipse(self, thumb: np.ndarray, index: np.ndarray) -> Shape | None:
        """One hand: the thumb-to-index span becomes the long axis of an ellipse."""
        axis = index - thumb
        span = float(np.linalg.norm(axis))
        if span < 2.0:
            return None

        centre = (thumb + index) / 2.0
        major = max(span / 2.0 * (1.0 + self.shape.expand), 1.0)
        minor = max(major * self.shape.ellipse_ratio, 1.0)
        angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
        points = cv2.ellipse2Poly(
            (int(round(centre[0])), int(round(centre[1]))),
            (int(round(major)), int(round(minor))),
            int(round(angle)), 0, 360, 15,
        )
        return Shape(
            points=points.astype(np.float32),
            anchors=np.array([thumb, index], dtype=np.float32),
        )


def render(
    frame: np.ndarray,
    shape: Shape,
    effect_name: str,
    ctx: EffectContext,
    params: dict | None = None,
    cfg: ShapeConfig | None = None,
) -> bool:
    """Apply the effect inside ``shape`` and draw its outline. Modifies ``frame``.

    Returns False when the shape is closed up or entirely off-screen.
    """
    cfg = cfg or ShapeConfig()
    if shape.size < cfg.min_size:
        return False

    polygon = shape.polygon()
    if not effects.apply_in_polygon(frame, polygon, effect_name, ctx, params):
        return False

    _draw_outline(frame, polygon, shape.anchors, cfg)
    return True


def _draw_outline(
    frame: np.ndarray, polygon: np.ndarray, anchors: np.ndarray, cfg: ShapeConfig
) -> None:
    color = tuple(int(c) for c in cfg.border_color)

    if cfg.glow and cfg.glow_strength > 0:
        # Blur the outline and add it back, which reads as light spilling off
        # the edge. Confined to a padded bounding box: blurring the whole frame
        # every frame is far too slow at 720p.
        pad = 18
        height, width = frame.shape[:2]
        x, y, box_w, box_h = cv2.boundingRect(polygon)
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + box_w + pad, width), min(y + box_h + pad, height)
        if x1 > x0 and y1 > y0:
            region = frame[y0:y1, x0:x1]
            halo = np.zeros_like(region)
            cv2.polylines(
                halo, [polygon - np.array([x0, y0], dtype=np.int32)], True, color,
                max(cfg.border_thickness * 3, 3), cv2.LINE_AA,
            )
            halo = effects.soft_blur(halo, 9.0)
            cv2.addWeighted(region, 1.0, halo, cfg.glow_strength, 0.0, dst=region)

    cv2.polylines(frame, [polygon], True, color, max(cfg.border_thickness, 1), cv2.LINE_AA)

    if cfg.show_points:
        for point in np.rint(anchors).astype(int):
            cv2.circle(frame, tuple(point), 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(point), 9, color, 1, cv2.LINE_AA)
