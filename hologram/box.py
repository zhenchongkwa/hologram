"""Cube mode: a 3D box held between your hands, with a rose inside it.

Your hands set where the box is, how big it is and which way it faces; the
current effect is applied to whatever the box covers, so it reads as a window
cut through the picture. Open both palms and a rose blooms inside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import effects, scene3d
from .config import BoxConfig, TrackingConfig
from .effects import EffectContext
from .gestures import relative_twist, solo_palm, wants_rose
from .scene3d import Mesh
from .tracking import MIDDLE_TIP, WRIST, Ema, Hand

FINGERTIPS = (4, 8, 12, 16, 20)


@dataclass
class BoxPose:
    centre: np.ndarray  # (2,) pixels
    size: float         # half-extent in pixels
    yaw: float
    pitch: float
    roll: float
    bloom: float        # 0 closed, 1 fully open
    # One open palm on its own: the rose alone, resting on the hand, with no
    # box, no rays and no effect window.
    solo: bool = False


def _hand_centre(hand: Hand) -> np.ndarray:
    return hand.points.mean(axis=0)


class BoxView:
    """Tracks the box's pose across frames and draws it."""

    def __init__(self, cfg: BoxConfig, tracking: TrackingConfig) -> None:
        self.cfg = cfg
        self._centre = Ema(tracking.point_smoothing)
        self._size = Ema(tracking.point_smoothing)
        self._solo_centre = Ema(tracking.point_smoothing)
        self._solo_size = Ema(tracking.point_smoothing)
        self._yaw = Ema(cfg.turn_smoothing)
        self._pitch = Ema(cfg.turn_smoothing)
        self._roll = Ema(cfg.turn_smoothing)
        self._bloom = 0.0
        self._spin = 0.0
        self._last_time: float | None = None
        # Rose meshes are cached against a coarse bloom step. Regrowing 36
        # petals every frame is pure waste when the value barely moves, and the
        # step is finer than the eye can follow anyway.
        self._rose_cache: dict[int, Mesh] = {}
        self._lut = scene3d.glow_lut(cfg.rose_core, cfg.rose_edge)
        self._cloud: np.ndarray | None = None
        self._cloud_loaded = False

    def reconfigure(self, cfg: BoxConfig, tracking: TrackingConfig) -> None:
        """Take fresh settings without losing where the box already is.

        Re-running __init__ - which is what this used to do - threw away the
        smoothed pose, the spin and a half-open rose, so reloading the config
        mid-gesture made the box jump and the flower snap shut.
        """
        self.cfg = cfg
        for ema in (self._centre, self._size, self._solo_centre, self._solo_size):
            ema.alpha = float(np.clip(tracking.point_smoothing, 0.01, 1.0))
        for ema in (self._yaw, self._pitch, self._roll):
            ema.alpha = float(np.clip(cfg.turn_smoothing, 0.01, 1.0))
        # These are caches of settings, so they are the part that must go.
        self._rose_cache.clear()
        self._lut = scene3d.glow_lut(cfg.rose_core, cfg.rose_edge)
        self._cloud = None
        self._cloud_loaded = False

    # ---------- pose ----------

    def update(self, hands: list[Hand], now: float) -> BoxPose | None:
        dt = 0.0 if self._last_time is None else max(min(now - self._last_time, 0.25), 0.0)
        self._last_time = now

        target = 1.0 if wants_rose(hands) else 0.0
        rate = max(self.cfg.bloom_speed, 0.01) * dt
        self._bloom += float(np.clip(target - self._bloom, -rate, rate))
        self._bloom = float(np.clip(self._bloom, 0.0, 1.0))
        self._spin = (self._spin + self.cfg.spin * dt) % (2 * math.pi)

        if len(hands) < 2:
            self._centre.reset()
            self._size.reset()
            if solo_palm(hands):
                return self._solo_pose(hands[0])
            self._solo_centre.reset()
            self._solo_size.reset()
            return None
        self._solo_centre.reset()
        self._solo_size.reset()

        first, second = _hand_centre(hands[0]), _hand_centre(hands[1])
        delta = second - first
        span = float(np.linalg.norm(delta))
        if span < 1e-3:
            return None

        centre = np.asarray(self._centre.update((first + second) / 2.0), dtype=np.float32)
        size = float(self._size.update(span * self.cfg.size))

        # Turn your hands like a wheel to spin it; raise one hand to tip it.
        yaw = float(self._yaw.update(relative_twist(hands) * self.cfg.turn_gain))
        pitch = float(self._pitch.update(float(delta[1]) / span * self.cfg.tilt_gain))
        roll = float(self._roll.update(math.atan2(float(delta[1]), float(delta[0]))))

        return BoxPose(centre=centre, size=size, yaw=yaw, pitch=pitch,
                       roll=roll, bloom=self._bloom)

    def _solo_pose(self, hand: Hand) -> BoxPose:
        """The rose alone, hovering above one open hand.

        Sat directly over the middle of the hand and clear of its top edge,
        in screen terms rather than along the hand's own axis. Offsetting along
        the wrist-to-fingertip vector instead swings the rose off to one side
        as soon as the hand tilts, and puts it in front of the hand rather than
        above it; this keeps it centred and on top however the hand is held.
        """
        middle_x = float(hand.points[:, 0].mean())
        top_y = float(hand.points[:, 1].min())
        target = np.array(
            [middle_x, top_y - hand.scale * self.cfg.solo_lift], dtype=np.float32
        )
        centre = np.asarray(self._solo_centre.update(target), dtype=np.float32)
        size = float(self._solo_size.update(hand.scale * self.cfg.solo_size))
        return BoxPose(centre=centre, size=size, yaw=self._spin, pitch=0.0,
                       roll=0.0, bloom=self._bloom, solo=True)

    # ---------- drawing ----------

    def _points(self) -> np.ndarray | None:
        """The imported rose point cloud, or None if it has not been built.

        Loaded once and shuffled, so that showing a prefix of it during the
        bloom reveals the whole flower evenly. The file keeps its points in
        surface order, and a prefix of that would fill in one contiguous patch
        of petal instead.
        """
        if not self._cloud_loaded:
            self._cloud_loaded = True
            path = Path(self.cfg.rose_points_file)
            if not path.is_absolute():
                path = Path(__file__).resolve().parent.parent / path
            if path.exists():
                cloud = np.load(path).astype(np.float32)
                order = np.random.default_rng(11).permutation(len(cloud))
                self._cloud = cloud[order]
        return self._cloud

    def _rose(self, bloom: float) -> Mesh:
        step = max(int(round(bloom * self.cfg.bloom_steps)), 0)
        mesh = self._rose_cache.get(step)
        if mesh is None:
            if len(self._rose_cache) > self.cfg.bloom_steps + 2:
                self._rose_cache.clear()
            mesh = self._rose_cache[step] = scene3d.make_rose(
                bloom=step / max(self.cfg.bloom_steps, 1),
                whorls=self.cfg.whorls,
                rows=self.cfg.rose_rows,
                cols=self.cfg.rose_cols,
            )
        return mesh

    def render(
        self,
        frame: np.ndarray,
        pose: BoxPose,
        hands: list[Hand],
        effect_name: str,
        ctx: EffectContext,
        params: dict | None = None,
    ) -> None:
        cfg = self.cfg
        rot = scene3d.rotation(pose.yaw, pose.pitch, pose.roll)

        # A single open palm shows the rose on its own - no box, so no
        # wireframe, no rays and no effect window either.
        points = None
        if not pose.solo:
            cube = scene3d.make_cube()
            points, _ = scene3d.project(cube.vertices, rot, pose.centre, pose.size,
                                        cfg.perspective)

            if cfg.effect_inside:
                effects.apply_in_polygon(
                    frame, scene3d.silhouette(points), effect_name, ctx, params
                )

            if cfg.rays and hands:
                self._draw_rays(frame, points, hands)

        if pose.bloom > 0.01:
            # The rose only partly follows the box. Turned all the way with it
            # a flat rosette goes edge-on and disappears to a sliver, so it
            # keeps most of its face to the camera and turns just enough to
            # show it is a solid object sitting inside the box.
            rose_rot = scene3d.rotation(
                pose.yaw * cfg.rose_follow + self._spin,
                pose.pitch * cfg.rose_follow,
                pose.roll * cfg.rose_follow,
            )
            rose_size = pose.size * cfg.rose_scale * (0.45 + 0.55 * pose.bloom)
            # Fading in with the bloom stops the rose popping into existence
            # the instant the gesture registers.
            fade = cfg.rose_glow * (0.35 + 0.65 * pose.bloom)

            cloud = self._points()
            if cloud is not None:
                # Reveal more of the cloud as it opens, so the rose assembles
                # itself out of the air rather than just scaling up.
                shown = max(int(len(cloud) * (0.12 + 0.88 * pose.bloom)), 1)
                pts, cam = scene3d.project(
                    cloud[:shown], rose_rot, pose.centre, rose_size, cfg.perspective
                )
                scene3d.draw_points(
                    frame, pts, cam[:, 2], self._lut,
                    size=cfg.rose_point_size,
                    supersample=cfg.rose_supersample,
                    bloom=cfg.rose_bloom_glow,
                    strength=fade,
                    gain=cfg.rose_point_gain,
                )
            else:
                # No imported model - fall back to the generated rose, so the
                # app still works from a fresh checkout.
                rose = self._rose(pose.bloom)
                rose_pts, rose_cam = scene3d.project(
                    rose.vertices, rose_rot, pose.centre, rose_size, cfg.perspective
                )
                curves = [rose_pts[index] for index in rose.curves]
                depths = np.array(
                    [float(rose_cam[index, 2].mean()) for index in rose.curves],
                    dtype=np.float32,
                )
                scene3d.draw_glow(
                    frame, curves, depths, self._lut,
                    thickness=cfg.rose_line,
                    supersample=cfg.rose_supersample,
                    bloom=cfg.rose_bloom_glow,
                    strength=fade,
                )

        if points is not None:
            scene3d.draw_wireframe(frame, points, cube.edges, cfg.colour,
                                   cfg.thickness, cfg.glow)

    def _draw_rays(self, frame: np.ndarray, points: np.ndarray, hands: list[Hand]) -> None:
        """Thin lines from the fingertips to the nearest corner of the box."""
        colour = tuple(int(c) for c in self.cfg.colour)
        corners = np.rint(points).astype(np.int32)

        overlay = frame.copy()
        for hand in hands:
            for index in FINGERTIPS:
                tip = np.rint(hand.points[index]).astype(np.int32)
                nearest = corners[int(np.argmin(np.linalg.norm(corners - tip, axis=1)))]
                cv2.line(overlay, tuple(tip), tuple(nearest), colour, 1, cv2.LINE_AA)

        opacity = float(np.clip(self.cfg.ray_opacity, 0.0, 1.0))
        cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0.0, dst=frame)
