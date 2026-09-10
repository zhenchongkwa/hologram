"""Camera loop, keyboard handling, and capture."""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import config as config_module
from . import effects, hud, modes as modes_module
from .config import Config
from .effects import EffectContext
from .gestures import EffectSwitcher
from .modes import orb_visible
from .orb import OrbView
from .tracking import HandTracker

WINDOW = "Hologram Glitch Band"

# DSHOW first: the default MSMF backend on Windows can take several seconds to
# open a webcam, and sometimes refuses to set a resolution at all.
_BACKENDS = ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY"))


class CameraError(RuntimeError):
    pass


def open_camera(index: int, width: int, height: int, fps: int):
    """Open a webcam, trying each backend in turn."""
    attempts = []
    for backend, name in _BACKENDS:
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            ok, _ = capture.read()
            if ok:
                return capture, name
            capture.release()
            attempts.append(f"{name} (opened but returned no frames)")
        else:
            capture.release()
            attempts.append(f"{name} (would not open)")

    raise CameraError(
        f"Could not read from camera index {index}. Tried: " + ", ".join(attempts) + ".\n"
        "Close anything else using the webcam (Teams, Zoom, the Camera app), or try "
        "another index with --camera 1."
    )


class FrameGrabber:
    """Pulls frames on a background thread.

    ``capture.read()`` blocks for a full frame interval - 33 ms on a 30 fps
    camera - and doing that inline meant the camera wait and the effect work
    added up instead of overlapping. Here the camera fills a one-frame slot
    while the main loop renders the previous frame.

    Only the newest frame is kept. If rendering falls behind, older frames are
    dropped rather than queued, so the picture stays live instead of drifting
    further behind real time.
    """

    def __init__(self, capture) -> None:
        self._capture = capture
        self._condition = threading.Condition()
        self._frame = None
        self._running = True
        self.failed = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "FrameGrabber":
        self._thread.start()
        return self

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                with self._condition:
                    self.failed = True
                    self._condition.notify_all()
                return
            with self._condition:
                self._frame = frame
                self._condition.notify()

    def read(self, timeout: float = 2.0):
        """Wait for the next fresh frame. Returns None on timeout or failure."""
        with self._condition:
            if self._frame is None and not self.failed:
                self._condition.wait(timeout)
            if self.failed:
                return None
            frame, self._frame = self._frame, None
            return frame

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class HologramApp:
    def __init__(self, cfg: Config, camera_index: int | None = None) -> None:
        self.cfg = cfg
        self.camera_index = camera_index if camera_index is not None else cfg.camera.index

        self.effect_names = self._effect_order(cfg)
        self.effect_index = self._default_index(cfg, self.effect_names)

        self.mirror = cfg.camera.mirror
        self.show_hud = cfg.hud.enabled
        self.show_skeleton = cfg.hud.skeleton
        self.show_help = cfg.hud.show_help

        self.ctx = EffectContext()
        self.modes = modes_module.build(cfg)
        self.mode_index = self._mode_index(cfg.modes.start)
        # The orb is not a mode - it is a gesture that rides along with the
        # modes named in [modes] orb.
        self.orb = OrbView(cfg.orb, cfg.tracking)
        self.switcher = EffectSwitcher(cfg.gestures)

        self._writer: cv2.VideoWriter | None = None
        # Saving is deferred to the next frame, where a clean pre-overlay image
        # is available.
        self._pending_screenshot = False
        self._frame_times: deque[float] = deque(maxlen=30)
        self._message = ""
        self._message_until = 0.0
        self._orb = False
        self._bloomed = False

    # ---------- configuration ----------

    @staticmethod
    def _effect_order(cfg: Config) -> list[str]:
        """Config order, minus anything that isn't a real effect."""
        available = set(effects.names())
        ordered = [name for name in cfg.effects.order if name in available]
        ordered += [name for name in effects.names() if name not in ordered]
        return ordered

    @staticmethod
    def _default_index(cfg: Config, names: list[str]) -> int:
        try:
            return names.index(cfg.effects.default)
        except ValueError:
            return 0

    def reload_config(self) -> None:
        if not self.cfg.source:
            self._notify("no config file to reload")
            return
        try:
            fresh = config_module.load(self.cfg.source)
        except Exception as exc:
            self._notify(f"config error: {exc}")
            return

        current = self.effect_names[self.effect_index]
        self.cfg = fresh
        self.effect_names = self._effect_order(fresh)
        self.effect_index = (
            self.effect_names.index(current) if current in self.effect_names else 0
        )
        # Reconfigured, never rebuilt: throwing the views away mid-gesture
        # dropped a lit orb and snapped a half-open rose shut.
        for mode in self.modes:
            mode.wants_orb = mode.name in fresh.modes.orb
            mode.wants_bloom = mode.name in fresh.modes.bloom
            mode.reconfigure(fresh)
        self.orb.reconfigure(fresh.orb, fresh.tracking)
        self.switcher = EffectSwitcher(fresh.gestures)
        self._notify("config reloaded")

    # ---------- helpers ----------

    @property
    def effect(self) -> str:
        return self.effect_names[self.effect_index]

    @property
    def mode(self) -> modes_module.Mode:
        return self.modes[self.mode_index]

    def _mode_index(self, name: str) -> int:
        for index, mode in enumerate(self.modes):
            if mode.name == name:
                return index
        return 0

    def step_mode(self, delta: int) -> None:
        """Walk the mode list. Modelled on step_effect, for the same reason:
        a list with an index has no second value to overload."""
        self.mode.reset()
        self.mode_index = (self.mode_index + delta) % len(self.modes)
        self.mode.reset()
        # One mode's trail must not bleed into the next.
        self.ctx.clear()
        self._notify(f"{self.mode.name} mode")

    def go_to_mode(self, name: str) -> None:
        index = self._mode_index(name)
        if index != self.mode_index:
            self.step_mode(index - self.mode_index)

    def step_effect(self, delta: int) -> None:
        self.effect_index = (self.effect_index + delta) % len(self.effect_names)
        # Trails from the old effect would otherwise bleed into the new one.
        self.ctx.clear()
        self._notify(self.effect)

    def _notify(self, text: str, seconds: float = 1.6) -> None:
        self._message = text
        self._message_until = time.monotonic() + seconds

    def _capture_dir(self) -> Path:
        root = Path(self.cfg.source).parent if self.cfg.source else Path.cwd()
        path = root / self.cfg.capture.dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_screenshot(self, frame: np.ndarray) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = self._capture_dir() / f"hologram-{stamp}.png"
        cv2.imwrite(str(target), frame)
        self._notify(f"saved {target.name}")

    def _toggle_recording(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._notify("recording stopped")
            return

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = self._capture_dir() / f"hologram-{stamp}.mp4"
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(target), fourcc, float(self.cfg.capture.video_fps), (width, height)
        )
        if not writer.isOpened():
            self._notify("could not start recording")
            return
        self._writer = writer
        self._notify(f"recording to {target.name}")

    # ---------- main loop ----------

    def run(self) -> int:
        capture, backend = open_camera(
            self.camera_index,
            self.cfg.camera.width,
            self.cfg.camera.height,
            self.cfg.camera.fps,
        )
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera {self.camera_index} open via {backend} at {width}x{height}.")
        print("Raise both hands - the shape follows your thumbs and index fingers.")
        print("Tap thumb to index on both hands to change effect. q to quit, ? for keys.")

        tracker = HandTracker(self.cfg.tracking, mirrored=self.mirror)
        grabber = FrameGrabber(capture).start()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, width, height)

        try:
            while True:
                frame = grabber.read()
                if frame is None:
                    if grabber.failed:
                        print("Camera stopped returning frames.")
                        return 1
                    # Just a slow frame. Keep servicing the window so the app
                    # stays responsive instead of appearing to hang.
                    if self._handle_key(cv2.waitKey(1) & 0xFF, None):
                        return 0
                    continue

                # Keep handedness labels correct if the mirror was toggled.
                tracker.mirrored = self.mirror
                if self.mirror:
                    frame = cv2.flip(frame, 1)

                now = time.monotonic()
                self._frame_times.append(now)
                self.ctx.frame_index += 1

                hands = tracker.process(frame, int(now * 1000))

                step = self.switcher.update(hands, now)
                if step:
                    self.step_effect(step)

                params = self.cfg.effects.for_effect(self.effect)

                mode = self.mode
                mode.update(hands, now)
                mode.render(frame, hands, self.effect, self.ctx, params)

                # Suppressed, it is still updated with no hands so it eases
                # out rather than vanishing mid-glow when the box comes up or
                # you switch modes. Drawn before the skeleton, so the hand still
                # reads over the glow.
                show_orb = orb_visible(mode.wants_orb, mode.blocks_orb)
                orb_pose = self.orb.update(hands if show_orb else [], now)
                self._orb = show_orb and orb_pose is not None and orb_pose.fade > 0.01
                if orb_pose is not None:
                    self.orb.render(frame, orb_pose)

                if self.show_skeleton:
                    hud.draw_skeleton(frame, hands, self.cfg.hud.color)

                # Bloom last, over the finished picture, so the glow bleeds out
                # of everything drawn - but before the captures and the HUD, so
                # recordings get it and the readouts stay crisp.
                if mode.wants_bloom:
                    self._bloomed = effects.bloom_pass(frame, self.cfg.bloom)
                else:
                    self._bloomed = False

                # Both captures happen before the overlay is drawn, so what gets
                # saved is the effect on its own rather than the readouts.
                if self._writer is not None:
                    self._writer.write(frame)
                if self._pending_screenshot:
                    self._pending_screenshot = False
                    self._save_screenshot(frame)

                if self.show_hud:
                    hud.draw(frame, self._hud_state(hands, now), self.cfg.hud)

                cv2.imshow(WINDOW, frame)

                if self._handle_key(cv2.waitKey(1) & 0xFF, frame):
                    return 0

                # Closing the window with the X button should also exit.
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return 0
        finally:
            grabber.stop()
            if self._writer is not None:
                self._writer.release()
            tracker.close()
            capture.release()
            cv2.destroyAllWindows()

    def _hud_state(self, hands, now: float) -> hud.HudState:
        fps = 0.0
        if len(self._frame_times) > 1:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                fps = (len(self._frame_times) - 1) / elapsed
        return hud.HudState(
            effect=self.effect,
            fps=fps,
            hands=len(hands),
            mode=self.mode.name,
            readout=self.mode.readout(),
            orb=self._orb,
            recording=self._writer is not None,
            show_help=self.show_help,
            message=self._message if now < self._message_until else "",
        )

    def _handle_key(self, key: int, frame: np.ndarray | None) -> bool:
        """Act on a keypress. Returns True when the app should exit.

        ``frame`` is None when called while waiting on a slow camera frame; the
        two actions that need one are simply skipped.
        """
        if key in (ord("q"), 27):
            return True
        if key == 255:  # no key pressed
            return False

        # The mode gets first refusal, so a key that belongs to one look never
        # becomes a branch in here.
        message = self.mode.handle_key(key)
        if message is not None:
            self.ctx.clear()
            self._notify(message)
            return False

        if key == ord("["):
            self.step_effect(-1)
        elif key == ord("]"):
            self.step_effect(1)
        elif ord("1") <= key <= ord("9"):
            index = key - ord("1")
            if index < len(self.effect_names):
                self.effect_index = index
                self.ctx.clear()
                self._notify(self.effect)
        elif key == ord("h"):
            self.show_hud = not self.show_hud
        elif key == ord("k"):
            self.show_skeleton = not self.show_skeleton
        elif key in (ord("?"), ord("/")):
            self.show_help = not self.show_help
        elif key == ord("m"):
            self.mirror = not self.mirror
            self._notify(f"mirror {'on' if self.mirror else 'off'}")
        elif key == ord("g"):
            self.cfg.bloom.enabled = not self.cfg.bloom.enabled
            self._notify(f"bloom {'on' if self.cfg.bloom.enabled else 'off'}")
        elif key == ord("n"):
            self.step_mode(1)
        elif key == ord("b"):
            # Kept as a direct jump: it is in the README, both launchers, and
            # everyone's fingers. From cube it goes back where it came from.
            self.go_to_mode("shape" if self.mode.name == "cube" else "cube")
        elif key == ord("s"):
            self._pending_screenshot = True
        elif key == ord("r") and frame is not None:
            self._toggle_recording(frame)
        elif key == ord("c"):
            self.reload_config()
        return False

    # ---------- diagnostics ----------

    def selftest(self, frames: int = 30) -> int:
        """Open the camera, grab some frames, and report. No window, no hands."""
        capture, backend = open_camera(
            self.camera_index,
            self.cfg.camera.width,
            self.cfg.camera.height,
            self.cfg.camera.fps,
        )
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"backend      : {backend}")
            print(f"resolution   : {width}x{height}")
            print(f"reported fps : {capture.get(cv2.CAP_PROP_FPS):.1f}")

            tracker = HandTracker(self.cfg.tracking, mirrored=self.mirror)
            grabbed, seen = 0, 0
            try:
                # The first few frames carry model warm-up, which would drag the
                # measurement well below what the app actually sustains.
                for _ in range(5):
                    ok, frame = capture.read()
                    if ok:
                        tracker.process(frame)

                start = time.monotonic()
                for _ in range(frames):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    grabbed += 1
                    seen = max(seen, len(tracker.process(frame)))
                elapsed = time.monotonic() - start
            finally:
                tracker.close()

            rate = grabbed / elapsed if elapsed > 0 else 0.0
            print(f"frames       : {grabbed}/{frames} in {elapsed:.2f}s")
            # Said plainly because this path reads frames inline, while the app
            # reads them on a thread and runs faster than this number suggests.
            print(f"probe rate   : {rate:.1f} fps (camera + detection, unthreaded)")
            print(f"max hands    : {seen}")
            print(f"effects      : {', '.join(self.effect_names)}")
            return 0 if grabbed == frames else 1
        finally:
            capture.release()
