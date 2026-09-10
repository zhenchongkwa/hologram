"""Modes: what the app makes of your hands.

A mode owns one whole look - the state it carries between frames, what it draws,
the keys that belong to it, and the line it writes in the readout. The app keeps
an ordered list of them and cycles with ``n``; it knows nothing about what any
one of them does.

This replaces a single ``cube_mode`` boolean. Two modes fit in a boolean, but
eleven places had quietly come to depend on "not cube means shape" - the orb's
gate, the bloom's gate, the readout, four scalars that went stale on a switch,
four lines of help text. A list of objects has no second value to overload, and
it is as easy to take a mode out of as to put one in.
"""

from __future__ import annotations

import math

import numpy as np

from . import slitscan
from . import shape as shape_module
from .box import BoxView
from .config import Config
from .effects import EffectContext
from .gestures import relative_twist
from .shape import ShapeBuilder
from .tracking import Ema, Hand


def orb_visible(wants_orb: bool, blocked: bool) -> bool:
    """Whether the orb draws this frame.

    Two questions kept apart: does this mode carry the orb at all, and is
    something inside it standing in the orb's way right now. Only the box
    blocks - it fills the space between your hands, and the orb on top of it is
    two effects fighting over the same picture. The rose alone on one open palm
    is not the box, so the orb still shows beside that.
    """
    return wants_orb and not blocked


class Mode:
    """One look. Subclasses fill in what they draw and what they say."""

    name = "mode"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # Set by the app from [modes] in the config, not chosen here: whether
        # this mode is one the two-finger orb appears in, and whether the
        # full-frame bloom pass runs after it.
        self.wants_orb = False
        self.wants_bloom = True

    # ---------- per frame ----------

    def update(self, hands: list[Hand], now: float) -> None:
        """Advance this mode's own state. Draws nothing."""

    def render(self, frame: np.ndarray, hands: list[Hand], effect: str,
               ctx: EffectContext, params: dict) -> None:
        """Draw into the frame. Modifies it."""

    def readout(self) -> str:
        """This mode's part of the HUD line - the app supplies fps and hands."""
        return ""

    @property
    def blocks_orb(self) -> bool:
        """True while something in this mode is in the orb's way."""
        return False

    # ---------- lifecycle ----------

    def handle_key(self, key: int) -> str | None:
        """Take a keypress that belongs to this mode.

        Returns the message to flash, or None if the key was not ours - the app
        then tries its own bindings. Keeping mode keys here is what stops the
        app from growing a branch per mode in ``_handle_key``.
        """
        return None

    def reset(self) -> None:
        """Drop anything held from frame to frame. Called on a mode switch, so
        one mode's trail or history never bleeds into the next."""

    def reconfigure(self, cfg: Config) -> None:
        """Take fresh settings without dropping animation state mid-gesture."""
        self.cfg = cfg


class ShapeMode(Mode):
    """The original: a band drawn through four fingertips, effect inside."""

    name = "shape"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.builder = ShapeBuilder(cfg.shape, cfg.tracking)
        self.per_hand = False
        self._shapes: list = []
        self._twist = 0.0
        self._size = 0.0
        self._crossed = False

    def update(self, hands: list[Hand], now: float) -> None:
        self._twist = relative_twist(hands)
        self._shapes = self.builder.build(hands, per_hand=self.per_hand)
        self._crossed = any(item.crossed for item in self._shapes)
        if self._shapes:
            self._size = self._shapes[0].size

    def render(self, frame, hands, effect, ctx, params) -> None:
        for item in self._shapes:
            shape_module.render(frame, item, effect, ctx, params, self.cfg.shape)

    def readout(self) -> str:
        twist = f"{math.degrees(self._twist):+4.0f}" + (" X" if self._crossed else "  ")
        span = "per-hand" if self.per_hand else "paired"
        return f"{span}   shape {self._size:3.0f}px   twist {twist}"

    def handle_key(self, key: int) -> str | None:
        if key == ord("p"):
            self.per_hand = not self.per_hand
            return "one shape per hand" if self.per_hand else "shape across both hands"
        return None

    def reset(self) -> None:
        self._shapes = []
        self._size = 0.0
        self._crossed = False

    def reconfigure(self, cfg: Config) -> None:
        super().reconfigure(cfg)
        self.builder.reconfigure(cfg.shape, cfg.tracking)


class CubeMode(Mode):
    """A 3D box held between your hands, with a rose blooming inside it."""

    name = "cube"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.view = BoxView(cfg.box, cfg.tracking)
        self._pose = None

    def update(self, hands: list[Hand], now: float) -> None:
        self._pose = self.view.update(hands, now)

    def render(self, frame, hands, effect, ctx, params) -> None:
        if self._pose is not None:
            self.view.render(frame, self._pose, hands, effect, ctx, params)

    def readout(self) -> str:
        pose = self._pose
        if pose is None:
            return "no box"
        rose = "open" if pose.bloom > 0.98 else f"{pose.bloom:.0%}"
        where = "on palm" if pose.solo else f"box {pose.size * 2.0:3.0f}px"
        return f"{where}   rose {rose}"

    @property
    def blocks_orb(self) -> bool:
        # The rose alone on one open palm is not the box, and shares the screen
        # with the orb happily.
        return self._pose is not None and not self._pose.solo

    def reset(self) -> None:
        self._pose = None

    def reconfigure(self, cfg: Config) -> None:
        super().reconfigure(cfg)
        self.view.reconfigure(cfg.box, cfg.tracking)


def _hand_centre(hand: Hand) -> np.ndarray:
    return hand.points.mean(axis=0)


class SlitScanMode(Mode):
    """Every column of the picture from a different moment, dragged by your hand.

    The heaviest of the modes and the only one holding a big buffer, so it is
    the one to turn down first if the budget will not take it: halving
    ``[slitscan] scale`` quarters both the memory and the gather.
    """

    name = "slitscan"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.wants_bloom = False
        self.scan = slitscan.SlitScan(cfg.slitscan.length, cfg.slitscan.scale)
        self._centre = Ema(cfg.slitscan.smoothing)
        self._sweep = 0.0
        self._last_time: float | None = None
        self._following = False
        # The hand's x is judged against the frame, and update runs before the
        # first render has seen one.
        self._width = float(cfg.camera.width)

    def update(self, hands: list[Hand], now: float) -> None:
        cfg = self.cfg.slitscan
        dt = 0.0 if self._last_time is None else max(min(now - self._last_time, 0.25), 0.0)
        self._last_time = now
        self._sweep = (self._sweep + cfg.sweep * dt) % 1.0

        self._following = bool(cfg.follow and hands)
        if self._following:
            # The hand nearest the camera leads, so bringing one forward takes
            # the scan from the other rather than the two fighting over it.
            hand = max(hands, key=lambda item: item.scale)
            self._centre.update(float(_hand_centre(hand)[0]))

    @property
    def centre(self) -> float:
        """Where the newest moment sits, as a fraction of the width."""
        if not self._following or self._centre.value is None:
            return self._sweep
        return float(np.clip(float(self._centre.value) / max(self._width, 1.0), 0.0, 1.0))

    def render(self, frame, hands, effect, ctx, params) -> None:
        cfg = self.cfg.slitscan
        self._width = float(frame.shape[1])
        slitscan.render(frame, self.scan, self.centre, cfg.spread, cfg.direction)

    def readout(self) -> str:
        where = "hand" if self._following else "sweep"
        return f"{self.scan.frames:2d} frames   {where} {self.centre:.2f}"

    def reset(self) -> None:
        # The history is the one thing that must not survive a mode switch:
        # coming back to this mode should not open on a second of some other
        # mode's output.
        self.scan.reset()
        self._centre.reset()
        self._last_time = None
        self._following = False

    def reconfigure(self, cfg: Config) -> None:
        super().reconfigure(cfg)
        self._centre.alpha = float(np.clip(cfg.slitscan.smoothing, 0.01, 1.0))
        # Length and scale are the buffer's own shape, so changing either has
        # to rebuild it - which push() does on the next frame.
        if (self.scan.length != max(int(cfg.slitscan.length), 2)
                or abs(self.scan.scale - cfg.slitscan.scale) > 1e-6):
            self.scan = slitscan.SlitScan(cfg.slitscan.length, cfg.slitscan.scale)


#: Every mode the app can be in, in the order ``n`` walks them.
REGISTRY: dict[str, type[Mode]] = {
    ShapeMode.name: ShapeMode,
    CubeMode.name: CubeMode,
    SlitScanMode.name: SlitScanMode,
}


def build(cfg: Config) -> list[Mode]:
    """Make one of every mode named in the config, in that order.

    Unknown names are skipped rather than raising: an old config naming a mode
    that no longer exists should still start the app.
    """
    names = [name for name in cfg.modes.order if name in REGISTRY]
    names += [name for name in REGISTRY if name not in names]

    modes = []
    for name in names:
        mode = REGISTRY[name](cfg)
        mode.wants_orb = name in cfg.modes.orb
        mode.wants_bloom = name in cfg.modes.bloom
        modes.append(mode)
    return modes
