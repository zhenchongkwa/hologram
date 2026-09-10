"""The red orb: hold up two fingers and it appears off their ends.

A sphere of red energy - Gojo's Red - with a hot core, a defined rim, and great
circles turning around it.

It is real 3D, drawn through the same pipeline as the rose: geometry in unit
space, rotated and projected by ``scene3d.project``, then splatted or stroked
into a supersampled buffer, bloomed, pushed through a colour ramp and added to
the frame.

What makes it read as a ball is not the renderer's near-to-far weighting - on a
symmetric sphere every sightline holds a matched near and far point, so that
weighting cancels to under a tenth across the whole disc. It is the radial
density of the cloud, burning at the middle and falling off to a rim, and the
great circles around it, whose ellipses and front-to-back shading are what put
the thing in space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import scene3d
from .config import OrbConfig, TrackingConfig
from .gestures import orb_hand
from .tracking import INDEX_PIP, INDEX_TIP, MIDDLE_TIP, Ema, Hand

# The radius, in pixels, that the configured point count is quoted at. See
# render() for what both of these are for.
GAIN_RADIUS = 64.0


@dataclass
class OrbPose:
    centre: np.ndarray  # (2,) pixels
    radius: float       # the sphere's radius in pixels
    fade: float         # 0 gone, 1 fully present
    spin: float         # radians


class OrbView:
    """Tracks the orb across frames and draws it."""

    def __init__(self, cfg: OrbConfig, tracking: TrackingConfig) -> None:
        self.cfg = cfg
        self._centre = Ema(tracking.point_smoothing)
        self._radius = Ema(tracking.point_smoothing)
        self._fade = 0.0
        self._spin = 0.0
        self._last_time: float | None = None
        self._lut = scene3d.glow_lut(cfg.core_colour, cfg.edge_colour)
        # The rings burn through a ramp of their own - see render().
        self._ring_lut = scene3d.glow_lut(cfg.core_colour, cfg.ring_edge_colour)

    def reconfigure(self, cfg: OrbConfig, tracking: TrackingConfig) -> None:
        """Take fresh settings without dropping a lit orb mid-glow."""
        self.cfg = cfg
        for ema in (self._centre, self._radius):
            ema.alpha = float(np.clip(tracking.point_smoothing, 0.01, 1.0))
        self._lut = scene3d.glow_lut(cfg.core_colour, cfg.edge_colour)
        self._ring_lut = scene3d.glow_lut(cfg.core_colour, cfg.ring_edge_colour)

    def update(self, hands: list[Hand], now: float) -> OrbPose | None:
        dt = 0.0 if self._last_time is None else max(min(now - self._last_time, 0.25), 0.0)
        self._last_time = now
        self._spin = (self._spin + self.cfg.spin * dt) % (2 * math.pi)

        hand = orb_hand(hands) if self.cfg.enabled else None
        target = 1.0 if hand is not None else 0.0
        rate = max(self.cfg.fade_speed, 0.01) * dt
        self._fade += float(np.clip(target - self._fade, -rate, rate))
        self._fade = float(np.clip(self._fade, 0.0, 1.0))

        if hand is None:
            if self._fade <= 0.01:
                self._centre.reset()
                self._radius.reset()
                return None
            # Still fading out - hold the last position rather than snapping
            # the orb to the middle of the frame on the way down.
            if self._centre.value is None:
                return None
            return OrbPose(np.asarray(self._centre.value, dtype=np.float32),
                           float(self._radius.value or 0.0), self._fade, self._spin)

        # Centred between the two raised fingers rather than on one of them.
        tip = (hand.points[INDEX_TIP] + hand.points[MIDDLE_TIP]) / 2.0
        along = tip - hand.points[INDEX_PIP]
        length = float(np.linalg.norm(along))
        if length > 1e-6:
            # Sat off the ends of the fingers rather than on the nails.
            tip = tip + along / length * (hand.scale * self.cfg.reach)

        centre = np.asarray(self._centre.update(tip.astype(np.float32)), dtype=np.float32)
        radius = float(self._radius.update(hand.scale * self.cfg.size))
        return OrbPose(centre=centre, radius=radius, fade=self._fade, spin=self._spin)

    def render(self, frame: np.ndarray, pose: OrbPose) -> bool:
        cfg = self.cfg
        if pose.fade <= 0.01 or pose.radius < 1.0:
            return False
        if not np.all(np.isfinite(pose.centre)):
            return False

        # Turned about two axes rather than one, so the rings sweep instead of
        # sliding sideways. The sphere has no visible front of its own - the
        # rings are what show that it is turning at all.
        rot = scene3d.rotation(pose.spin, pose.spin * 0.45, 0.0)
        strength = float(np.clip(cfg.strength * pose.fade, 0.0, 4.0))

        # Everything here costs the square of the radius - the buffer both
        # renderers build is the orb's own bounding box - so a hand right up
        # against the lens would otherwise cost four times a frame's budget on
        # its own. Past this the orb stops growing, which also keeps it from
        # swallowing the picture.
        radius = min(float(pose.radius), float(cfg.max_radius))

        # Points per pixel, not points, is what the renderer turns into
        # brightness - it accumulates. A fixed cloud in a small ball piles
        # every point onto the same few pixels and burns out white; spread over
        # a large one it thins to isolated specks. So the count follows the
        # projected area and the density stays put, which is what keeps the orb
        # looking the same whether your hand is near the camera or far from it.
        # Raising the gain instead does not work: it only saturates each point.
        # The cloud is built big enough for a ball at the size cap and no
        # bigger, then a prefix of it is drawn - it is shuffled, so any prefix
        # is still a whole sphere.
        headroom = max(int(math.ceil((cfg.max_radius / GAIN_RADIUS) ** 2)), 1)
        cloud = scene3d.sphere_cloud(cfg.points * headroom, cfg.shell, cfg.core_bias)
        shown = int(np.clip(cfg.points * (radius / GAIN_RADIUS) ** 2, 64, len(cloud)))
        points, camera = scene3d.project(
            cloud[:shown], rot, pose.centre, radius, cfg.perspective
        )
        drew = scene3d.draw_points(
            frame, points, camera[:, 2], self._lut,
            size=cfg.point_size,
            supersample=cfg.supersample,
            bloom=cfg.bloom,
            strength=strength,
            gain=cfg.point_gain,
        )

        arcs = scene3d.ring_arcs(cfg.rings, cfg.ring_scale, cfg.ring_arcs)
        if arcs:
            # Projected in one call and split afterwards: one matrix multiply
            # for the whole armillary rather than one per arc.
            lengths = [len(arc) for arc in arcs]
            ring_points, ring_camera = scene3d.project(
                np.concatenate(arcs), rot, pose.centre, radius, cfg.perspective
            )
            curves, depths, start = [], [], 0
            for length in lengths:
                curves.append(ring_points[start:start + length])
                depths.append(float(ring_camera[start:start + length, 2].mean()))
                start += length

            # The rings take their own ramp. The glow renderer's nearest depth
            # band peaks near 222, and on the core ramp that lands in white -
            # which is why the flat version drew its ring at 135 by hand. A red
            # edge colour keeps them red however hot the near side burns.
            drew = scene3d.draw_glow(
                frame, curves, np.asarray(depths, dtype=np.float32), self._ring_lut,
                thickness=cfg.line,
                supersample=cfg.supersample,
                bloom=cfg.bloom,
                strength=strength * cfg.ring_strength,
            ) or drew
        return drew
