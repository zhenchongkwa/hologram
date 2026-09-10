"""Settings loading.

Everything tunable lives in config.toml so the look can be changed without
touching code. Unknown keys are ignored and missing ones fall back to the
dataclass defaults, which keeps an old or hand-edited config working.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BGR = tuple[int, int, int]

DEFAULT_ORDER = [
    "risograph",
    "cyanotype",
    "quadtree",
    "vhs",
    "ascii",
    "halftone",
    "thermal",
    "dither",
]


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True


@dataclass
class TrackingConfig:
    max_hands: int = 2
    detection_confidence: float = 0.6
    presence_confidence: float = 0.5
    tracking_confidence: float = 0.5
    # EMA factor for the tracked fingertips: higher follows your fingers faster,
    # lower is smoother but laggier.
    point_smoothing: float = 0.5


@dataclass
class ShapeConfig:
    # How far the shape reaches past the fingertips, as a fraction of its size.
    expand: float = 0.12
    # Shapes smaller than this many pixels across are skipped, so the effect
    # doesn't flicker when your fingers close.
    min_size: int = 28
    # Per-hand mode turns one thumb-to-index span into an ellipse this fraction
    # as wide as it is long.
    ellipse_ratio: float = 0.62
    border_thickness: int = 2
    border_color: BGR = (0, 255, 0)
    glow: bool = True
    glow_strength: float = 0.55
    # Dots on the fingertips currently driving the shape.
    show_points: bool = True


@dataclass
class BoxConfig:
    """Cube mode - the 3D box held between your hands, with a rose inside."""

    # Box half-size as a fraction of the distance between your hands. In the
    # clip the box spans a bit over a third of the hand span, so the half-size
    # is around 0.19 - it hangs between the hands rather than filling the gap.
    size: float = 0.19
    # Focal length as a multiple of the box size: larger flattens the
    # perspective, smaller exaggerates it.
    perspective: float = 2.2
    colour: BGR = (0, 255, 0)
    thickness: int = 2
    glow: float = 0.55
    # How far turning your hands turns the box, and tipping them tips it.
    turn_gain: float = 1.6
    tilt_gain: float = 1.1
    turn_smoothing: float = 0.25
    # Faint lines from your fingertips to the nearest corner.
    rays: bool = True
    ray_opacity: float = 0.45
    # Apply the current effect to whatever the box covers.
    effect_inside: bool = True
    # The rose. Drawn as glowing strokes, never filled - filled flat-shaded
    # polygons are what made the earlier version look faceted and dead.
    # Petals per whorl, innermost ring first.
    whorls: BGR = (3, 5, 8)
    rose_rows: int = 10
    rose_cols: int = 5
    # The glow ramp: black to core to edge, in BGR. Deep crimson burning out to
    # a hot warm red at the densest part of the cloud.
    rose_core: BGR = (18, 12, 125)
    rose_edge: BGR = (72, 98, 255)
    # One open palm on its own shows the rose with no box. Size is a multiple
    # of that hand's own scale, so it holds its proportion at any distance.
    #
    # Lift is the gap above the topmost point of the hand, in hand-scale units.
    # Measured up the screen rather than along the hand's own axis: an axis
    # offset swings the rose off to one side the moment the hand tilts.
    solo_size: float = 1.25
    solo_lift: float = 1.1
    # Point cloud imported from a real rose model by scripts/import_rose.py.
    # When the file is missing the generated rose is drawn instead, so a fresh
    # checkout still works.
    rose_points_file: str = "models/rose_points.npy"
    # Brightness per splat. The cloud is sparse enough that most pixels catch
    # under one point, so this has to be high for a single hit to register;
    # dense parts saturate and burn out, which is the look.
    rose_point_size: int = 2
    rose_point_gain: float = 105.0
    rose_line: int = 2
    rose_supersample: int = 2
    rose_glow: float = 1.0
    rose_bloom_glow: float = 1.7
    rose_scale: float = 0.95
    rose_follow: float = 0.35   # how much it turns with the box
    spin: float = 0.35          # its own idle spin, radians per second
    bloom_speed: float = 2.2    # how fast it opens, per second
    bloom_steps: int = 24       # cached mesh steps between bud and full bloom


@dataclass
class OrbConfig:
    """The red orb summoned by pointing your index finger."""

    enabled: bool = True
    # Which modes the orb appears in is set by [modes] orb, not here. It stays
    # down while the box itself is up whatever that says, and that part is not
    # tunable: the box fills the space between your hands, and the orb on top
    # of it is two effects fighting over the same picture. The rose alone on
    # one open palm is not the box, so the orb still appears beside that.
    # Core radius and stand-off distance, both as multiples of that hand's own
    # scale, so the orb keeps its proportion at any distance from the camera.
    size: float = 0.62
    reach: float = 0.55
    # The orb's radius in pixels is capped here. Both renderers build a buffer
    # the size of the orb's own bounding box, so the cost goes as the square of
    # this - a hand up against the lens would cost a whole frame budget.
    max_radius: float = 96.0
    # The sphere is a point cloud and the renderer accumulates, so the radial
    # distribution of the points is the look. core_bias above 1/3 pulls density
    # into the middle - that is the hot core - and shell is the fraction of the
    # points spent on the rim that gives it a silhouette.
    points: int = 14000
    core_bias: float = 1.4
    shell: float = 0.40
    point_size: int = 3
    point_gain: float = 150.0
    perspective: float = 2.4
    # Great circles turning around it. ring_scale is a multiple of the sphere's
    # own radius, so 1.0 would sit exactly on its surface. Each ring is cut into
    # arcs, because the glow renderer shades one depth per curve: whole, a ring
    # would come out flat instead of passing behind the ball.
    rings: int = 3
    ring_scale: float = 1.9
    ring_arcs: int = 8
    line: int = 1
    # Held below the sphere's own brightness on purpose: left equal, the rings
    # read as the subject and the ball as their backdrop, which is backwards.
    ring_strength: float = 0.42
    # BGR, black to core to edge - the same shape of ramp the rose uses.
    core_colour: BGR = (24, 20, 190)
    edge_colour: BGR = (210, 225, 255)
    # The rings burn through their own ramp. The glow renderer's nearest depth
    # band peaks near 222, which on the ramp above comes out white; a red edge
    # keeps the rings red however hot the near side gets.
    ring_edge_colour: BGR = (40, 45, 250)
    supersample: int = 2
    bloom: float = 2.0
    strength: float = 1.0
    fade_speed: float = 4.0
    spin: float = 0.8  # radians per second, for the cross


@dataclass
class BloomConfig:
    """A bloom pass over the finished frame - bright areas bleed light."""

    enabled: bool = True
    # Which modes it runs after is set by [modes] bloom, not here.
    # Conditioning before the highlights are picked out, as TouchDesigner's
    # Bloom TOP does: lift the black point, then brightness, then gamma.
    pre_black: float = 0.08
    pre_brightness: float = 1.0
    pre_gamma: float = 1.0
    # Only pixels brighter than this bloom, and how hard the cut is. Measured
    # across a sweep: at 0.62 the pass is barely visible, and by 0.30 the whole
    # background lifts into haze. 0.45 glows the bright parts and leaves the
    # rest alone.
    threshold: float = 0.45
    soft_knee: bool = True
    # The glow is blurred at several radii between these and averaged, which is
    # what gives a tight core inside a wide halo instead of one flat smear.
    min_radius: float = 3.0
    max_radius: float = 22.0
    steps: int = 3
    intensity: float = 1.1
    # Built at this fraction of the frame. Bloom is blur - there is no detail
    # in it to lose, and full size costs several times as much.
    scale: float = 0.5


@dataclass
class SlitScanConfig:
    """Time slit-scan - every column of the picture from a different moment."""

    # Frames of history. At half resolution 30 frames of 720p is 20.7 MB and
    # covers one second, which is about as far back as the effect still reads
    # as motion rather than as an unrelated picture.
    length: int = 30
    # How much of the history is spread across the frame's width. At 1.0 the
    # whole second spans the picture exactly once; higher wraps it several
    # times into stacked bands of past.
    spread: float = 1.0
    direction: int = 1  # -1 sweeps the other way
    # Whether the newest moment follows your hand across the frame. With this
    # off it sweeps by itself at `sweep` widths per second, which is the older
    # and more hands-off version of the effect.
    follow: bool = True
    sweep: float = 0.25
    # How fast the scan slides to your hand. Taken raw it snaps between columns
    # as the tracker jitters, and the whole picture shears with it.
    smoothing: float = 0.20
    # Fraction of the frame the history is kept at. This sets the memory as
    # well as the speed: the ring is length x this squared.
    scale: float = 0.5


DEFAULT_MODES = ["shape", "cube", "slitscan"]


@dataclass
class ModesConfig:
    """Which looks the app cycles through, and what rides along with each.

    The orb and the bloom used to be gated by a boolean apiece on the two
    modes that existed - `orb.cube_only` and `bloom.shape_only`. Naming the
    modes instead means adding a mode does not mean inverting a flag.
    """

    order: list[str] = field(default_factory=lambda: list(DEFAULT_MODES))
    start: str = "shape"
    # Modes the two-finger orb appears in.
    orb: list[str] = field(default_factory=lambda: ["cube"])
    # Modes the full-frame bloom pass runs after. Modes that already fill the
    # picture with light leave it off.
    bloom: list[str] = field(default_factory=lambda: ["shape"])


@dataclass
class GestureConfig:
    enabled: bool = True
    # Thumb-to-index distance, over hand size, below which a tap registers.
    tap_threshold: float = 0.38
    # Seconds before another tap is accepted.
    cooldown: float = 0.7


@dataclass
class EffectConfig:
    default: str = "risograph"
    order: list[str] = field(default_factory=lambda: list(DEFAULT_ORDER))
    # Per-effect settings, keyed by effect name. Left as plain dicts so adding
    # an effect never means adding a dataclass here too.
    params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def for_effect(self, name: str) -> dict[str, Any]:
        return self.params.get(name, {})


@dataclass
class HudConfig:
    enabled: bool = True
    skeleton: bool = True
    color: BGR = (0, 255, 0)
    show_help: bool = False


@dataclass
class CaptureConfig:
    dir: str = "captures"
    video_fps: int = 30


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    box: BoxConfig = field(default_factory=BoxConfig)
    orb: OrbConfig = field(default_factory=OrbConfig)
    bloom: BloomConfig = field(default_factory=BloomConfig)
    slitscan: SlitScanConfig = field(default_factory=SlitScanConfig)
    modes: ModesConfig = field(default_factory=ModesConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    effects: EffectConfig = field(default_factory=EffectConfig)
    hud: HudConfig = field(default_factory=HudConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    source: Path | None = None


def _coerce(value: Any, target: Any) -> Any:
    """Convert a TOML value to match the shape of the dataclass default."""
    if isinstance(target, tuple) and isinstance(value, list):
        return tuple(int(v) for v in value)
    if isinstance(target, bool):
        return bool(value)
    if isinstance(target, int) and not isinstance(target, bool):
        return int(value)
    if isinstance(target, float):
        return float(value)
    return value


def _build(cls: type, data: dict[str, Any]):
    """Build a dataclass from a dict, ignoring keys the dataclass doesn't have."""
    defaults = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        spec = defaults.get(key)
        if spec is None:
            continue
        if spec.default is not dataclasses.MISSING:
            value = _coerce(value, spec.default)
        kwargs[key] = value
    return cls(**kwargs)


def _build_effects(data: dict[str, Any]) -> EffectConfig:
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    cfg = _build(EffectConfig, scalars)
    cfg.params = {k: v for k, v in data.items() if isinstance(v, dict)}
    return cfg


def load(path: str | Path | None = None) -> Config:
    """Load config.toml, falling back to defaults when it is absent."""
    cfg = Config()
    if path is None:
        return cfg

    path = Path(path)
    if not path.exists():
        return cfg

    # Read the bytes and strip a leading BOM before parsing. tomllib reads in
    # binary and rejects a byte-order mark outright, and plenty of Windows
    # editors - Notepad, and PowerShell's own -Encoding utf8 - add one when
    # they save. Failing to open the config over an invisible character is a
    # miserable way to lose an evening.
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    raw = tomllib.loads(data.decode("utf-8", errors="replace"))

    cfg.camera = _build(CameraConfig, raw.get("camera", {}))
    cfg.tracking = _build(TrackingConfig, raw.get("tracking", {}))
    cfg.shape = _build(ShapeConfig, raw.get("shape", {}))
    cfg.box = _build(BoxConfig, raw.get("box", {}))
    cfg.orb = _build(OrbConfig, raw.get("orb", {}))
    cfg.bloom = _build(BloomConfig, raw.get("bloom", {}))
    cfg.slitscan = _build(SlitScanConfig, raw.get("slitscan", {}))
    cfg.modes = _build(ModesConfig, raw.get("modes", {}))
    cfg.gestures = _build(GestureConfig, raw.get("gestures", {}))
    cfg.effects = _build_effects(raw.get("effects", {}))
    cfg.hud = _build(HudConfig, raw.get("hud", {}))
    cfg.capture = _build(CaptureConfig, raw.get("capture", {}))
    cfg.source = path
    return cfg
