"""Camera-free checks for the shape geometry, effects, and gesture logic.

Run with:  python tests/test_offline.py

Deliberately dependency-free (no pytest) so it runs in the same environment as
the app itself. mediapipe is never imported here: HandTracker imports it lazily,
so everything below works on synthetic landmarks instead of a live camera.
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hologram import effects, hud, modes, scene3d, slitscan  # noqa: E402
from hologram.app import FrameGrabber, HologramApp, orb_visible  # noqa: E402
from hologram.box import BoxView  # noqa: E402
from hologram.config import (  # noqa: E402
    BloomConfig,
    BoxConfig,
    Config,
    GestureConfig,
    HudConfig,
    OrbConfig,
    ShapeConfig,
    TrackingConfig,
)
from hologram.orb import OrbPose, OrbView  # noqa: E402
from hologram.effects import EffectContext  # noqa: E402
from hologram.gestures import (  # noqa: E402
    EffectSwitcher,
    axis_angle,
    extended_fingers,
    is_open_palm,
    is_tapping,
    is_two_fingers,
    orb_hand,
    palms_open,
    pinch,
    relative_twist,
)
from hologram.shape import Shape, ShapeBuilder, render, ribbon_quad  # noqa: E402
from hologram.tracking import (  # noqa: E402
    INDEX_TIP,
    MIDDLE_TIP,
    THUMB_TIP,
    WRIST,
    Hand,
)

# A plausible right hand, palm to camera, fingers up. Units are multiples of
# hand scale, so landmark 9 sits exactly 1.0 from the wrist.
_OPEN_HAND = np.array([
    (0.00, 0.00),    # 0  wrist
    (-0.30, -0.20),  # 1  thumb cmc
    (-0.50, -0.40),  # 2  thumb mcp
    (-0.65, -0.55),  # 3  thumb ip
    (-0.80, -0.70),  # 4  thumb tip
    (-0.25, -0.90),  # 5  index mcp
    (-0.28, -1.25),  # 6  index pip
    (-0.30, -1.45),  # 7  index dip
    (-0.30, -1.65),  # 8  index tip
    (0.00, -1.00),   # 9  middle mcp
    (0.00, -1.35),   # 10 middle pip
    (0.00, -1.60),   # 11 middle dip
    (0.00, -1.80),   # 12 middle tip
    (0.25, -0.95),   # 13 ring mcp
    (0.28, -1.30),   # 14 ring pip
    (0.30, -1.50),   # 15 ring dip
    (0.30, -1.70),   # 16 ring tip
    (0.50, -0.80),   # 17 pinky mcp
    (0.55, -1.05),   # 18 pinky pip
    (0.60, -1.20),   # 19 pinky dip
    (0.60, -1.40),   # 20 pinky tip
], dtype=np.float32)


# Curled fingers and a thumb tucked across the palm.
_CURLED = {
    4: (0.10, -0.60),
    8: (-0.26, -0.95),
    12: (0.00, -1.00),
    16: (0.26, -0.95),
    20: (0.50, -0.80),
}


# Curling the ring and pinky tips back leaves index and middle out - the two
# fingers the orb gesture looks for.
_TWO_FINGERS = {16: (0.26, -0.95), 20: (0.50, -0.80)}


def make_hand(center=(320, 240), scale=80.0, pinch_gap=None, label="Right",
              thumb=None, index=None, closed=False, two_fingers=False) -> Hand:
    """Build a synthetic hand.

    ``pinch_gap`` sets the thumb-to-index distance in hand-scale units.
    ``thumb`` and ``index`` place those two tips at exact pixel positions, for
    tests that care where the shape's corners land. ``closed`` curls the
    fingers into a fist.
    """
    points = _OPEN_HAND.copy()
    if closed:
        for landmark, position in _CURLED.items():
            points[landmark] = position
    if two_fingers:
        for landmark, position in _TWO_FINGERS.items():
            points[landmark] = position
    if pinch_gap is not None:
        tip = points[INDEX_TIP]
        direction = points[THUMB_TIP] - tip
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            points[THUMB_TIP] = tip + direction / norm * pinch_gap
    points = points * scale + np.asarray(center, dtype=np.float32)
    if thumb is not None:
        points[THUMB_TIP] = np.asarray(thumb, dtype=np.float32)
    if index is not None:
        points[INDEX_TIP] = np.asarray(index, dtype=np.float32)
    return Hand(points=points.astype(np.float32), label=label, score=0.99)


def _builder():
    return ShapeBuilder(ShapeConfig(), TrackingConfig())


def _settle(builder, hands, per_hand=False, frames=25):
    """Run enough frames for the fingertip smoothing to converge."""
    shapes = []
    for _ in range(frames):
        shapes = builder.build(hands, per_hand=per_hand)
    return shapes


# --------------------------------------------------------------------------
# shape geometry
# --------------------------------------------------------------------------

def test_shape_is_built_from_the_four_fingertips():
    left = make_hand(label="Left", thumb=(200, 300), index=(220, 180))
    right = make_hand(label="Right", thumb=(460, 300), index=(440, 180))
    shapes = _settle(_builder(), [left, right])

    assert len(shapes) == 1
    shape = shapes[0]
    assert len(shape.anchors) == 4, shape.anchors
    assert len(shape.points) == 4

    # Every anchor should sit on one of the four tracked tips.
    wanted = [(200, 300), (220, 180), (460, 300), (440, 180)]
    for target in wanted:
        assert min(np.linalg.norm(shape.anchors - np.array(target), axis=1)) < 3.0, (
            f"no anchor near {target}: {shape.anchors.tolist()}"
        )


def test_corners_are_joined_by_fingertip_not_by_angle():
    """The rails are index-to-index and thumb-to-thumb.

    Sorting corners by angle about the centre would always give a simple convex
    polygon and quietly untwist the shape, so the ordering is by which fingertip
    each corner is.
    """
    thumb_a, index_a = (100, 300), (100, 100)
    thumb_b, index_b = (500, 300), (500, 100)
    quad = ribbon_quad(thumb_a, index_a, thumb_b, index_b)
    assert [tuple(p) for p in quad] == [thumb_a, index_a, index_b, thumb_b]


def test_counter_rotating_the_hands_crosses_the_shape_into_an_x():
    """The whole point: twisting must be able to fold the shape over itself."""
    thumb_a, index_a = (100, 300), (100, 100)

    flat = _quad(ribbon_quad(thumb_a, index_a, (500, 300), (500, 100)))
    assert not flat.crossed, "shape should be open when the hands agree"

    # Right hand rotated half a turn: its thumb and index swap ends.
    twisted = _quad(ribbon_quad(thumb_a, index_a, (500, 100), (500, 300)))
    assert twisted.crossed, "shape should cross into an X when counter-rotated"


def test_the_x_is_filled_as_two_lobes_not_as_one_block():
    """fillPoly's even-odd rule is what renders the crossing.

    fillConvexPoly would fill the whole hull and erase the twist entirely.
    """
    cfg = ShapeConfig(glow=False, border_thickness=1, show_points=False, expand=0.0)
    frame = _flat_frame()
    before = frame.copy()
    twisted = _quad(ribbon_quad((120, 380), (120, 120), (520, 120), (520, 380)))
    assert twisted.crossed

    assert render(frame, twisted, "cyanotype", EffectContext(), {}, cfg) is True

    changed = np.any(frame != before, axis=2)
    # The two lobes are filled...
    assert changed[250, 160], "left lobe not filled"
    assert changed[250, 480], "right lobe not filled"
    # ...and the wedges above and below the crossing are not.
    assert not changed[150, 320], "gap above the crossing was filled"
    assert not changed[350, 320], "gap below the crossing was filled"

    # A crossed shape covers about half of what the same corners cover untwisted.
    flat = _quad(ribbon_quad((120, 380), (120, 120), (520, 380), (520, 120)))
    other = _flat_frame()
    render(other, flat, "cyanotype", EffectContext(), {}, cfg)
    crossed_area = int(changed.sum())
    flat_area = int(np.any(other != before, axis=2).sum())
    assert crossed_area < flat_area * 0.65, (crossed_area, flat_area)


def test_the_twisted_end_pinches_to_a_point_at_the_crossover():
    """Turning one hand through the crossover closes that end to a point.

    Only the end you are turning collapses - the other stays as wide as your
    other hand is holding it - so the shape passes through a triangle, not
    through nothing. Measured on the filled mask at the twisting end, because
    contourArea returns the signed shoelace value once a polygon crosses itself
    and stops describing the area covered.
    """
    def twisting_end_height(offset):
        quad = ribbon_quad((120, 380), (120, 120), (520, 250 + offset), (520, 250 - offset))
        mask = np.zeros((500, 640), np.uint8)
        cv2.fillPoly(mask, [np.rint(quad).astype(np.int32)], 255)
        return int((mask[:, 505:518].max(axis=1) > 0).sum())

    assert twisting_end_height(0) < 20, twisting_end_height(0)
    assert twisting_end_height(130) > 200, "should be open before the crossover"
    assert twisting_end_height(-130) > 200, "should reopen crossed, on the far side"

    # And the far end is unaffected throughout - it is the other hand's.
    def far_end_height(offset):
        quad = ribbon_quad((120, 380), (120, 120), (520, 250 + offset), (520, 250 - offset))
        mask = np.zeros((500, 640), np.uint8)
        cv2.fillPoly(mask, [np.rint(quad).astype(np.int32)], 255)
        return int((mask[:, 122:135].max(axis=1) > 0).sum())

    assert far_end_height(0) > 200, far_end_height(0)


def test_shape_tracks_the_fingertips_as_they_move():
    builder = _builder()
    hands = [make_hand(label="Left", thumb=(200, 300), index=(220, 180)),
             make_hand(label="Right", thumb=(460, 300), index=(440, 180))]
    _settle(builder, hands)

    moved = [make_hand(label="Left", thumb=(200, 300), index=(220, 180)),
             make_hand(label="Right", thumb=(600, 300), index=(580, 180))]
    after = _settle(builder, moved)[0]
    assert after.anchors[:, 0].max() > 560, after.anchors.tolist()


def test_one_hand_makes_an_ellipse_across_the_pinch():
    shapes = _settle(_builder(), [make_hand(thumb=(300, 260), index=(400, 260))])
    assert len(shapes) == 1
    shape = shapes[0]
    assert len(shape.points) > 4, "expected an ellipse, not a quad"
    # Long axis spans the thumb-to-index gap, centred between them.
    assert 340 < shape.points[:, 0].mean() < 360, shape.points[:, 0].mean()
    assert shape.size >= 100


def test_per_hand_mode_makes_one_shape_each():
    hands = [make_hand(label="Left", thumb=(150, 300), index=(250, 300)),
             make_hand(label="Right", thumb=(450, 300), index=(550, 300))]
    assert len(_settle(_builder(), hands, per_hand=True)) == 2
    assert _builder().build([]) == []


def test_duplicate_handedness_labels_get_separate_smoothers():
    """MediaPipe sometimes labels both hands the same; they must not share state.

    A shared smoother is updated once per hand per frame, so each shape gets
    dragged towards the other hand instead of tracking its own.
    """
    hands = [make_hand(label="Right", thumb=(150, 300), index=(250, 300)),
             make_hand(label="Right", thumb=(700, 300), index=(800, 300))]
    shapes = _settle(_builder(), hands, per_hand=True)
    assert len(shapes) == 2
    centres = sorted(float(s.points[:, 0].mean()) for s in shapes)
    assert abs(centres[0] - 200) < 25, centres
    assert abs(centres[1] - 750) < 25, centres


def test_stale_smoothers_are_dropped_when_a_hand_leaves():
    builder = _builder()
    builder.build([make_hand(label="Left"), make_hand(label="Right")])
    builder.build([make_hand(label="Left")])
    assert set(builder._points) == {"Left#0:thumb", "Left#0:index"}, set(builder._points)


# --------------------------------------------------------------------------
# compositing
# --------------------------------------------------------------------------

def _flat_frame(value=90, size=(480, 640)):
    return np.full((size[0], size[1], 3), value, dtype=np.uint8)


def _quad(points):
    points = np.asarray(points, dtype=np.float32)
    return Shape(points=points, anchors=points)


def _point_clear_of(polygon, margin=6.0, inside_hull=False, bounds=(480, 640)):
    """Find a pixel comfortably outside ``polygon`` but inside its bounding box.

    The margin matters: the outline is stroked anti-aliased along the edge, so a
    pixel merely one step outside the polygon is legitimately painted.
    """
    x, y, w, h = cv2.boundingRect(polygon)
    hull = cv2.convexHull(polygon)
    for py in range(max(y, 0), min(y + h, bounds[0]), 2):
        for px in range(max(x, 0), min(x + w, bounds[1]), 2):
            point = (float(px), float(py))
            if cv2.pointPolygonTest(polygon, point, True) >= -margin:
                continue
            if inside_hull and cv2.pointPolygonTest(hull, point, True) <= margin:
                continue
            return (py, px)
    return None


def test_render_touches_inside_only():
    cfg = ShapeConfig(glow=False, border_thickness=1, show_points=False)
    frame = _flat_frame()
    before = frame.copy()
    shape = _quad([[200, 150], [430, 190], [420, 330], [210, 300]])

    assert render(frame, shape, "cyanotype", EffectContext(), {}, cfg) is True
    assert frame.shape == before.shape and frame.dtype == before.dtype
    assert frame[240, 320].tolist() != before[240, 320].tolist()

    for y, x in ((2, 2), (2, 637), (477, 2), (477, 637)):
        assert frame[y, x].tolist() == before[y, x].tolist(), f"corner {(y, x)} changed"

    # A point inside the bounding box but outside the quad must be untouched -
    # this is what proves the mask is the polygon, not the box.
    probe = _point_clear_of(shape.polygon())
    assert probe, "could not find a probe point outside the quad"
    assert frame[probe].tolist() == before[probe].tolist(), f"leaked outside the quad at {probe}"


def test_render_fills_a_concave_shape_correctly():
    """Bending a finger in can make the quad concave.

    fillConvexPoly would quietly fill the missing wedge; fillPoly must not.
    """
    cfg = ShapeConfig(glow=False, border_thickness=1, show_points=False, expand=0.0)
    frame = _flat_frame()
    before = frame.copy()
    # Fourth point pushed deep inside, leaving a notch on the right.
    shape = _quad([[150, 120], [500, 120], [150, 400], [300, 260]])

    assert render(frame, shape, "cyanotype", EffectContext(), {}, cfg) is True

    # A pixel in the notch: outside the polygon, but well inside its hull.
    notch = _point_clear_of(shape.polygon(), inside_hull=True)
    assert notch, "test shape was not actually concave"
    assert frame[notch].tolist() == before[notch].tolist(), (
        f"the concave notch at {notch} was filled - looks like convex filling"
    )


def test_closed_fingers_skip_the_effect():
    """Pinching shut must not leave a flickering speck on screen."""
    frame = _flat_frame()
    before = frame.copy()
    tiny = _quad([[320, 240], [324, 241], [325, 245], [321, 244]])
    assert render(frame, tiny, "cyanotype", EffectContext(), {}, ShapeConfig(min_size=28)) is False
    assert np.array_equal(frame, before)


def test_render_offscreen_shape_is_a_no_op():
    frame = _flat_frame()
    before = frame.copy()
    shape = _quad([[-900, -900], [-700, -880], [-710, -700], [-880, -720]])
    assert render(frame, shape, "vhs", EffectContext(), {}, ShapeConfig()) is False
    assert np.array_equal(frame, before)


def test_render_survives_partial_and_huge_shapes():
    cfg = ShapeConfig()
    frame = _flat_frame()
    render(frame, _quad([[-200, 100], [300, 130], [290, 380], [-210, 350]]),
           "risograph", EffectContext(), {}, cfg)
    render(frame, _quad([[-5000, -5000], [5000, -5000], [5000, 5000], [-5000, 5000]]),
           "cyanotype", EffectContext(), {}, cfg)


# --------------------------------------------------------------------------
# effects
# --------------------------------------------------------------------------

def test_every_effect_preserves_shape_and_dtype():
    rng = np.random.default_rng(7)
    roi = rng.integers(0, 256, size=(120, 200, 3), dtype=np.uint8)
    for name in effects.names():
        out = effects.apply(name, roi, EffectContext(), {})
        assert out.shape == roi.shape, f"{name} changed shape to {out.shape}"
        assert out.dtype == np.uint8, f"{name} returned {out.dtype}"


def test_effects_do_not_mutate_their_input():
    """render() still needs the original pixels to composite through the mask."""
    rng = np.random.default_rng(11)
    roi = rng.integers(0, 256, size=(80, 140, 3), dtype=np.uint8)
    for name in effects.names():
        original = roi.copy()
        effects.apply(name, roi, EffectContext(), {})
        assert np.array_equal(roi, original), f"{name} modified the input in place"


def test_effects_handle_tiny_and_empty_regions():
    ctx = EffectContext()
    for name in effects.names():
        effects.apply(name, np.zeros((1, 1, 3), np.uint8), ctx, {})
        effects.apply(name, np.zeros((0, 0, 3), np.uint8), ctx, {})


def _bloom_scene(size=(200, 320)):
    """A dim background with one bright patch, so bloom has something to bite."""
    frame = np.full((*size, 3), 40, np.uint8)
    cv2.circle(frame, (size[1] // 2, size[0] // 2), 30, (250, 250, 250), -1)
    return frame


def test_bloom_only_adds_light():
    """It is added to the frame, so nothing may come out darker."""
    frame = _bloom_scene()
    before = frame.copy()
    assert effects.bloom_pass(frame, BloomConfig()) is True
    assert np.all(frame >= before), "bloom darkened part of the frame"
    assert int(frame.sum()) > int(before.sum())


def test_bloom_glows_the_bright_parts_and_leaves_the_dark_alone():
    frame = _bloom_scene()
    before = frame.copy()
    effects.bloom_pass(frame, BloomConfig())

    # Just outside the bright circle it should have bled some light...
    assert int(frame[100, 205].sum()) > int(before[100, 205].sum()) + 3
    # ...and a far corner of the dim background should be untouched.
    assert int(frame[4, 4].sum()) <= int(before[4, 4].sum()) + 2


def test_bloom_threshold_controls_how_much_blooms():
    scene = _bloom_scene()
    changes = []
    for threshold in (0.30, 0.45, 0.62, 0.99):
        frame = scene.copy()
        effects.bloom_pass(frame, BloomConfig(threshold=threshold))
        changes.append(float(np.abs(frame.astype(int) - scene.astype(int)).mean()))

    assert changes == sorted(changes, reverse=True), changes
    assert changes[-1] < 0.5, "nothing should bloom at a threshold of 0.99"


def test_bloom_is_a_no_op_when_off():
    for cfg in (BloomConfig(enabled=False), BloomConfig(intensity=0.0)):
        frame = _bloom_scene()
        before = frame.copy()
        assert effects.bloom_pass(frame, cfg) is False
        assert np.array_equal(frame, before)


def test_bloom_curve_is_tabulated_once():
    """The whole highlight curve is a function of one pixel's brightness.

    Doing it as float maths per pixel instead measured 48 ms a frame against
    7.9 ms through a lookup table.
    """
    effects._BLOOM_LUTS.clear()
    cfg = BloomConfig()
    frame = _bloom_scene()
    effects.bloom_pass(frame, cfg)
    assert len(effects._BLOOM_LUTS) == 1
    effects.bloom_pass(frame, cfg)
    assert len(effects._BLOOM_LUTS) == 1, "the table was rebuilt"


def test_bloom_survives_a_tiny_frame():
    for size in ((2, 2), (1, 40), (8, 8)):
        frame = np.full((*size, 3), 200, np.uint8)
        effects.bloom_pass(frame, BloomConfig())


def test_vhs_smears_the_last_frame_into_this_one():
    """Tape lag: what was on the head is still there when the next field lands."""
    quiet = {"smear": 0.8, "bands": 0, "band": 0, "noise": 0.0,
             "chroma_bleed": 0, "chroma_lag": 0, "scanline_strength": 0.0}
    ctx = EffectContext()
    black = np.zeros((40, 60, 3), np.uint8)
    white = np.full((40, 60, 3), 255, np.uint8)
    effects.apply("vhs", white, ctx, quiet)
    faded = effects.apply("vhs", black, ctx, quiet)
    assert faded.mean() > 100, faded.mean()


def test_vhs_bleeds_colour_sideways_but_not_brightness():
    """Chroma is recorded at a fraction of the luma bandwidth, so it spreads.

    That one artefact is what says VHS - a picture where the colour has left the
    edges it belongs to while the edges themselves stay sharp.
    """
    quiet = {"smear": 0.0, "bands": 0, "band": 0, "noise": 0.0, "chroma_lag": 0,
             "scanline_strength": 0.0, "chroma_bleed": 21, "saturation": 1.0}
    roi = np.full((40, 120, 3), 110, np.uint8)
    roi[:, 58:64] = (40, 40, 230)          # a narrow red stripe on flat grey
    out = effects.apply("vhs", roi, EffectContext(), quiet)

    def chroma(image, column):
        pixel = image[20, column].astype(int)
        return abs(pixel[2] - pixel[0])    # red against blue, 0 when neutral

    # Ten pixels clear of the stripe: inside the chroma blur, outside the stripe.
    assert chroma(roi, 70) == 0, "the stripe should not reach this far to begin with"
    assert chroma(out, 70) > 20, chroma(out, 70)
    # The luma edge itself has not moved with it.
    luma = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(int)
    assert abs(int(luma[20, 70]) - 110) < 12, luma[20, 70]


def test_cyanotype_puts_everything_on_the_blue_ramp():
    """The iron salts only ever go blue, whatever colour went in."""
    for colour in ((30, 30, 200), (30, 200, 30), (200, 30, 30)):
        roi = np.full((24, 32, 3), colour, np.uint8)
        out = effects.apply("cyanotype", roi, EffectContext(), {"grain": 0.0})
        assert out[..., 0].mean() > out[..., 2].mean() + 40, (colour, out[12, 16])

    dark = effects.apply("cyanotype", np.full((8, 8, 3), 20, np.uint8),
                         EffectContext(), {"grain": 0.0})
    light = effects.apply("cyanotype", np.full((8, 8, 3), 230, np.uint8),
                          EffectContext(), {"grain": 0.0})
    assert light.mean() > dark.mean() + 60, (dark.mean(), light.mean())


def test_risograph_leaves_paper_showing_where_the_picture_is_bright():
    """Ink is laid onto paper, so highlights are bare paper rather than white."""
    paper = (238, 240, 240)
    params = {"grain": 0.0, "paper": paper, "ink_a": (0, 255, 0), "ink_b": (190, 40, 220)}
    white = effects.apply("risograph", np.full((60, 60, 3), 255, np.uint8),
                          EffectContext(), params)
    black = effects.apply("risograph", np.zeros((60, 60, 3), np.uint8),
                          EffectContext(), params)

    assert abs(float(white.mean()) - float(np.mean(paper))) < 2.0, white.mean()
    assert black.mean() < white.mean() - 60, (black.mean(), white.mean())
    # Under the dark ink the green pass dominates: that is the ink colour, not
    # a darkening of the picture.
    assert black[..., 1].mean() > black[..., 2].mean() + 40, black.reshape(-1, 3).mean(axis=0)


def test_halftone_dots_grow_with_the_light():
    """A screen is only a screen if coverage tracks the tone it stands for."""
    covered = []
    for level in (40, 128, 220):
        roi = np.full((48, 48, 3), level, np.uint8)
        out = effects.apply("halftone", roi, EffectContext(),
                            {"dot": 6, "ink": (0, 255, 0), "paper": (0, 0, 0)})
        covered.append(float((out[..., 1] > 128).mean()))
    assert covered[0] < covered[1] < covered[2], covered
    assert covered[0] < 0.35 and covered[2] > 0.65, covered


def test_dither_lands_on_two_colours_and_nothing_between():
    """Ordered dither is a compare: every pixel is ink or it is paper."""
    roi = np.full((64, 64, 3), 128, np.uint8)
    out = effects.apply("dither", roi, EffectContext(),
                        {"levels": 2, "ink": (0, 255, 0), "paper": (0, 0, 0)})
    colours = np.unique(out.reshape(-1, 3), axis=0)
    assert len(colours) == 2, colours
    assert {tuple(c) for c in colours} == {(0, 0, 0), (0, 255, 0)}, colours


def test_quadtree_keeps_big_blocks_where_there_is_nothing_to_see():
    """The whole point: detail is spent where the picture earns it."""
    roi = np.full((128, 256, 3), 90, np.uint8)
    # Structure, not noise: pixel noise averages away inside a block, so it is
    # detail at the block scale that a quadtree is supposed to chase.
    squares = (np.indices((128, 128)) // 12).sum(axis=0) % 2
    roi[:, 128:] = np.where(squares[..., None].astype(bool), 240, 15)
    out = effects.apply("quadtree", roi, EffectContext(), {"grid": False})

    flat = out[8:120, 8:120]
    busy = out[8:120, 136:248]
    assert float(flat.std()) < 1.0, flat.std()
    assert abs(float(flat.mean()) - 90.0) < 1.0, flat.mean()
    assert float(busy.std()) > 60.0, busy.std()


def test_ascii_writes_nothing_where_the_picture_is_black():
    """The darkest cell picks the first glyph in the ramp, which is a space."""
    dark = effects.apply("ascii", np.zeros((60, 80, 3), np.uint8), EffectContext(),
                         {"cell": 8, "ramp": " .:-=+*#%@", "tint": (0, 255, 0)})
    assert int(dark.max()) == 0, dark.max()

    light = effects.apply("ascii", np.full((60, 80, 3), 255, np.uint8), EffectContext(),
                          {"cell": 8, "ramp": " .:-=+*#%@", "tint": (0, 255, 0)})
    assert (light[..., 1] > 0).mean() > 0.15, (light[..., 1] > 0).mean()
    assert light[..., 0].max() == 0, "the tint is the only colour drawn"


def test_screens_and_atlases_are_built_once():
    """Rebuilding either per frame is the difference between 0.1 ms and 10."""
    first = effects._screen("bayer", 8, 40, 60)
    assert effects._screen("bayer", 8, 40, 60) is first
    assert effects._screen("clustered", 8, 40, 60) is not first

    atlas = effects._glyph_atlas(" .#", 6, 9)
    assert effects._glyph_atlas(" .#", 6, 9) is atlas
    assert atlas.shape == (3, 9, 6)


# --------------------------------------------------------------------------
# gestures
# --------------------------------------------------------------------------

def test_pinch_is_distance_invariant():
    """A hand twice as far away should read the same pinch."""
    near = make_hand(scale=120.0, pinch_gap=0.8)
    far = make_hand(scale=60.0, pinch_gap=0.8)
    assert abs(pinch(near) - pinch(far)) < 1e-5


def test_tap_is_detected_only_when_fingers_meet():
    cfg = GestureConfig(tap_threshold=0.38)
    assert is_tapping(make_hand(pinch_gap=0.05), cfg)
    assert not is_tapping(make_hand(pinch_gap=1.2), cfg)


def _tap_pair():
    return [make_hand(pinch_gap=0.05, label="Left"), make_hand(pinch_gap=0.05, label="Right")]


def _open_pair():
    return [make_hand(pinch_gap=1.2, label="Left"), make_hand(pinch_gap=1.2, label="Right")]


def test_tapping_both_hands_advances_the_effect():
    switcher = EffectSwitcher(GestureConfig(cooldown=0.7))
    assert switcher.update(_open_pair(), 0.0) == 0
    assert switcher.update(_tap_pair(), 0.1) == 1


def test_holding_the_tap_only_advances_once():
    """Otherwise pinching shut would race through the whole effect list."""
    switcher = EffectSwitcher(GestureConfig(cooldown=0.7))
    assert switcher.update(_tap_pair(), 0.0) == 1
    for t in (0.1, 0.5, 1.0, 2.0, 5.0):
        assert switcher.update(_tap_pair(), t) == 0, f"fired again while held at {t}"
    # Releasing and tapping again does advance.
    switcher.update(_open_pair(), 5.1)
    assert switcher.update(_tap_pair(), 5.2) == 1


def test_a_second_tap_inside_the_cooldown_is_ignored():
    switcher = EffectSwitcher(GestureConfig(cooldown=0.7))
    assert switcher.update(_tap_pair(), 0.0) == 1
    switcher.update(_open_pair(), 0.1)
    assert switcher.update(_tap_pair(), 0.2) == 0, "fired inside the cooldown"
    switcher.update(_open_pair(), 0.8)
    assert switcher.update(_tap_pair(), 0.9) == 1


def test_one_hand_tapping_does_nothing():
    """The gesture is both hands, so a single pinch must not switch effects."""
    switcher = EffectSwitcher(GestureConfig(cooldown=0.0))
    mixed = [make_hand(pinch_gap=0.05, label="Left"), make_hand(pinch_gap=1.2, label="Right")]
    for t in range(6):
        assert switcher.update(mixed, float(t)) == 0
    assert switcher.update([make_hand(pinch_gap=0.05)], 9.0) == 0


def test_open_hands_never_switch():
    switcher = EffectSwitcher(GestureConfig(cooldown=0.0))
    for t in range(10):
        assert switcher.update(_open_pair(), float(t)) == 0


def test_gestures_can_be_disabled():
    switcher = EffectSwitcher(GestureConfig(enabled=False, cooldown=0.0))
    assert switcher.update(_tap_pair(), 5.0) == 0


# --------------------------------------------------------------------------
# twist
# --------------------------------------------------------------------------

def hand_at_angle(degrees, center=(320, 240), gap=100.0, label="Right") -> Hand:
    """A hand whose thumb-to-index axis sits at a given angle."""
    radians = math.radians(degrees)
    half = np.array([math.cos(radians), math.sin(radians)], dtype=np.float32) * (gap / 2)
    centre = np.asarray(center, dtype=np.float32)
    return make_hand(center=center, label=label, thumb=centre - half, index=centre + half)


def test_axis_angle_reads_the_hand_rotation():
    assert abs(math.degrees(axis_angle(hand_at_angle(0))) - 0) < 1e-3
    assert abs(math.degrees(axis_angle(hand_at_angle(35))) - 35) < 1e-3


def test_relative_twist_measures_one_hand_against_the_other():
    left = hand_at_angle(10, center=(200, 240), label="Left")
    right = hand_at_angle(70, center=(440, 240), label="Right")
    assert abs(math.degrees(relative_twist([left, right])) - 60) < 1e-3

    # Rotating both hands together is not a twist.
    both = [hand_at_angle(40, center=(200, 240), label="Left"),
            hand_at_angle(40, center=(440, 240), label="Right")]
    assert abs(relative_twist(both)) < 1e-6

    # It needs two hands to mean anything.
    assert relative_twist([left]) == 0.0
    assert relative_twist([]) == 0.0


def test_twisting_the_hands_crosses_the_built_shape():
    """End to end: counter-rotate through the builder and the shape crosses."""
    builder = _builder()
    left = hand_at_angle(90, center=(200, 240), gap=200, label="Left")

    aligned = _settle(builder, [left, hand_at_angle(90, center=(440, 240), gap=200,
                                                    label="Right")])[0]
    assert not aligned.crossed

    builder = _builder()
    opposed = _settle(builder, [left, hand_at_angle(-90, center=(440, 240), gap=200,
                                                    label="Right")])[0]
    assert opposed.crossed, "counter-rotated hands should cross the shape"


# --------------------------------------------------------------------------
# 3D: projection, meshes, the box and the rose
# --------------------------------------------------------------------------

def test_rotation_matrices_are_orthonormal():
    for angles in ((0, 0, 0), (0.5, -0.3, 1.2), (math.pi, 0.7, -2.0)):
        rot = scene3d.rotation(*angles)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-5), angles
        assert abs(float(np.linalg.det(rot)) - 1.0) < 1e-5, angles


def test_projection_puts_points_where_expected():
    centre = (320.0, 240.0)
    rot = scene3d.rotation(0, 0, 0)
    points, _ = scene3d.project(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32), rot, centre, 100.0
    )
    # The origin lands on the centre, and a vertex on the z=0 plane lands at
    # its true size - one focal length away is where the scale is 1:1.
    assert np.allclose(points[0], centre, atol=1e-3)
    assert abs(points[1][0] - (centre[0] + 100.0)) < 1e-2, points[1]
    assert abs(points[2][1] - (centre[1] + 100.0)) < 1e-2, points[2]


def test_further_away_projects_towards_the_centre():
    centre = (320.0, 240.0)
    rot = scene3d.rotation(0, 0, 0)
    near, _ = scene3d.project(np.array([[1, 0, -0.5]], np.float32), rot, centre, 100.0)
    far, _ = scene3d.project(np.array([[1, 0, 2.0]], np.float32), rot, centre, 100.0)
    assert near[0][0] > far[0][0] > centre[0], (near, far)


def test_a_vertex_behind_the_camera_does_not_blow_up():
    """Without a near plane this divides by zero and flings the point to infinity."""
    centre = (320.0, 240.0)
    rot = scene3d.rotation(0, 0, 0)
    points, _ = scene3d.project(
        np.array([[1, 1, -50.0], [0, 0, -2.2]], np.float32), rot, centre, 100.0
    )
    assert np.all(np.isfinite(points)), points


def test_cube_mesh_is_a_cube():
    cube = scene3d.make_cube()
    assert cube.vertices.shape == (8, 3)
    assert cube.edges.shape == (12, 2)
    for a, b in cube.edges:
        delta = np.abs(cube.vertices[a] - cube.vertices[b])
        # Every edge runs along exactly one axis, corner to corner.
        assert sorted(delta.tolist()) == [0.0, 0.0, 2.0], (a, b, delta)


def test_silhouette_encloses_every_projected_vertex():
    """The mask polygon must not clip a corner of the box off.

    Tested against the rounded coordinates, because that is what the hull is
    built from. Comparing the raw floats instead reports corners up to half a
    pixel outside - which is just the rounding, not a hole in the mask.
    """
    for angles in ((0.6, 0.3, 0.2), (0.0, 0.0, 0.0), (2.1, -0.8, 1.4)):
        cube = scene3d.make_cube()
        points, _ = scene3d.project(cube.vertices, scene3d.rotation(*angles),
                                    (320.0, 240.0), 90.0)
        hull = scene3d.silhouette(points)
        for point in np.rint(points):
            inside = cv2.pointPolygonTest(hull, (float(point[0]), float(point[1])), False)
            assert inside >= 0, (angles, point.tolist())

        # And the raw floats are outside by rounding at most, never by more.
        worst = min(
            cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), True) for p in points
        )
        assert worst > -1.0, (angles, worst)


def test_glow_ramp_starts_black_and_brightens():
    """The glow is added to the frame, so zero intensity must add nothing."""
    lut = scene3d.glow_lut((120, 20, 90), (245, 190, 255))
    assert lut.shape == (256, 1, 3)
    assert lut[0].tolist() == [[0, 0, 0]], lut[0]
    brightness = lut.reshape(256, 3).astype(int).sum(axis=1)
    assert brightness[-1] > brightness[128] > brightness[8]


def test_glow_draws_inside_its_own_box_only():
    frame = np.zeros((300, 400, 3), np.uint8)
    lut = scene3d.glow_lut((120, 20, 90), (245, 190, 255))
    curve = np.array([[180, 140], [220, 140], [220, 170], [180, 170], [180, 140]], np.float32)

    assert scene3d.draw_glow(frame, [curve], np.array([1.0], np.float32), lut) is True
    # On the stroke itself. The curve is an outline, so its interior is empty -
    # sampling the middle of the rectangle tests nothing.
    assert frame[140, 200].sum() > 0, "the stroke itself was not drawn"
    # Far outside the curve's bounding box plus its bloom padding.
    for y, x in ((5, 5), (5, 395), (295, 5), (295, 395)):
        assert frame[y, x].sum() == 0, f"glow leaked to {(y, x)}"


def test_nearer_curves_glow_brighter_than_further_ones():
    """Depth has to show, or every petal reads at the same distance."""
    lut = scene3d.glow_lut((120, 20, 90), (245, 190, 255))
    curve = np.array([[150, 140], [250, 140], [250, 180], [150, 180], [150, 140]], np.float32)
    other = curve + np.array([0, 60], np.float32)

    frame = np.zeros((300, 400, 3), np.uint8)
    # Same shapes, one near and one far.
    scene3d.draw_glow(frame, [curve, other], np.array([5.0, 1.0], np.float32), lut)
    far_peak = int(frame[130:190].max())
    near_peak = int(frame[190:250].max())
    assert near_peak > far_peak * 1.3, (near_peak, far_peak)


def test_glow_survives_degenerate_input():
    frame = np.zeros((200, 200, 3), np.uint8)
    lut = scene3d.glow_lut((120, 20, 90), (245, 190, 255))
    before = frame.copy()

    assert scene3d.draw_glow(frame, [], np.array([], np.float32), lut) is False
    offscreen = np.array([[-900, -900], [-880, -880]], np.float32)
    assert scene3d.draw_glow(frame, [offscreen], np.array([1.0], np.float32), lut) is False
    broken = np.array([[np.nan, 10], [20, 30]], np.float32)
    assert scene3d.draw_glow(frame, [broken], np.array([1.0], np.float32), lut) is False
    assert np.array_equal(frame, before)


def test_rose_is_finite_and_opens_as_it_blooms():
    extents = []
    for bloom in (0.0, 0.25, 0.5, 0.75, 1.0):
        rose = scene3d.make_rose(bloom=bloom, whorls=(3, 5))
        assert np.all(np.isfinite(rose.vertices)), bloom
        assert rose.curves and len(rose.curves) == 8
        # Width across the bloom's face, which is what "opening" means here.
        extents.append(float(np.ptp(rose.vertices[:, 0])))

    assert extents == sorted(extents), extents
    # A bud is markedly tighter than an open flower. Scaling each mesh by its
    # own extent instead of a fixed one would flatten this to nothing.
    assert extents[-1] > extents[0] * 1.8, extents


def test_rose_has_one_closed_outline_per_petal():
    """Outlines only - ribs on every petal turned the flower into a wire ball."""
    rose = scene3d.make_rose(bloom=1.0, whorls=(3, 5, 8), rows=10, cols=5)
    assert len(rose.curves) == 16
    for curve in rose.curves:
        assert curve[0] == curve[-1], "outline is not closed"
        # Boundary of a rows x cols grid, plus the repeated closing point.
        assert len(curve) == 2 * (10 + 5) - 4 + 1
        assert curve.max() < len(rose.vertices)


# --------------------------------------------------------------------------
# the imported point cloud
# --------------------------------------------------------------------------

def _lut():
    return scene3d.glow_lut((120, 20, 90), (245, 190, 255))


def test_points_splat_inside_their_own_box_only():
    frame = np.zeros((300, 400, 3), np.uint8)
    rng = np.random.default_rng(3)
    cloud = np.stack([rng.uniform(180, 240, 500), rng.uniform(120, 180, 500)], axis=1)
    depth = np.full(500, 2.0, np.float32)

    assert scene3d.draw_points(frame, cloud.astype(np.float32), depth, _lut()) is True
    assert frame[120:180, 180:240].sum() > 0, "nothing was splatted"
    for y, x in ((5, 5), (5, 395), (295, 5), (295, 395)):
        assert frame[y, x].sum() == 0, f"cloud leaked to {(y, x)}"


def test_stacked_points_burn_brighter_than_scattered_ones():
    """Accumulating is the point - density is what makes the form show."""
    lut = _lut()
    depth = np.full(300, 2.0, np.float32)

    packed = np.full((300, 2), 200.0, np.float32)
    scattered = np.stack([
        np.linspace(150, 250, 300), np.linspace(150, 250, 300)
    ], axis=1).astype(np.float32)

    a, b = np.zeros((300, 400, 3), np.uint8), np.zeros((300, 400, 3), np.uint8)
    scene3d.draw_points(a, packed, depth, lut, gain=20.0)
    scene3d.draw_points(b, scattered, depth, lut, gain=20.0)
    assert int(a.max()) > int(b.max()) * 1.5, (int(a.max()), int(b.max()))


def test_nearer_points_glow_brighter_than_further_ones():
    """Depth has to show, or the cloud reads as a flat scatter.

    Kept well below saturation on purpose: stack enough points on one pixel and
    both clusters clamp at 255, and the comparison proves nothing.
    """
    lut = _lut()
    near = np.tile(np.array([[150.0, 120.0]], np.float32), (20, 1))
    far = np.tile(np.array([[150.0, 200.0]], np.float32), (20, 1))
    cloud = np.concatenate([near, far])
    depth = np.concatenate([np.full(20, 1.0), np.full(20, 9.0)]).astype(np.float32)

    frame = np.zeros((300, 400, 3), np.uint8)
    scene3d.draw_points(frame, cloud, depth, lut, gain=20.0, bloom=0.0)
    near_peak, far_peak = int(frame[110:135].max()), int(frame[190:215].max())
    assert near_peak < 255 and far_peak < 255, "saturated - test proves nothing"
    assert near_peak > far_peak * 1.5, (near_peak, far_peak)


def test_draw_points_survives_degenerate_input():
    frame = np.zeros((200, 200, 3), np.uint8)
    before = frame.copy()
    lut = _lut()

    empty = np.zeros((0, 2), np.float32)
    assert scene3d.draw_points(frame, empty, np.zeros(0, np.float32), lut) is False
    off = np.full((10, 2), -900.0, np.float32)
    assert scene3d.draw_points(frame, off, np.ones(10, np.float32), lut) is False
    broken = np.full((10, 2), np.nan, np.float32)
    assert scene3d.draw_points(frame, broken, np.ones(10, np.float32), lut) is False
    assert np.array_equal(frame, before)


def _load_importer():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / "import_rose.py"
    spec = importlib.util.spec_from_file_location("import_rose", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importer_drops_the_grass_and_keeps_the_plant():
    """The model is a whole scene - three grass planes must not come along."""
    importer = _load_importer()
    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / "rose.obj"
        obj.write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
            "v 0 0 1\nv 1 0 1\nv 0 1 1\n"
            "v 0 0 2\nv 1 0 2\nv 0 1 2\n"
            "g pCube1\nusemtl grassMat\nf 1 2 3\n"
            "g pCylinder1\nusemtl stemMat\nf 4 5 6\n"
            "g Mesh\nusemtl roseMat\nf 7 8 9\n"
        )
        obj.with_suffix(".mtl").write_text(
            "newmtl grassMat\nmap_Kd grass_texture225.jpg\n"
            "newmtl stemMat\nmap_Kd texture-green-paper-pattern.jpg\n"
            "newmtl roseMat\nmap_Kd wildtextures-leather-Campo-rose.jpg\n"
        )

        vertices, groups, materials = importer.parse_obj(obj)
        assert len(vertices) == 9
        labels = importer.classify(groups, materials,
                                   obj.with_suffix(".mtl").read_text())
        assert labels["pCube1"] == "grass"
        assert labels["pCylinder1"] == "stem"
        assert labels["Mesh"] == "flower"


def test_importer_handles_negative_face_indices():
    """OBJ allows indices counting back from the end of the vertex list."""
    importer = _load_importer()
    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / "m.obj"
        obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\ng Mesh\nusemtl m\nf -3 -2 -1\n")
        _, groups, _ = importer.parse_obj(obj)
        assert groups["Mesh"] == {0, 1, 2}


# --------------------------------------------------------------------------
# open palms, and the box
# --------------------------------------------------------------------------

def test_open_palms_are_told_apart_from_fists():
    assert extended_fingers(make_hand()) == 5
    assert is_open_palm(make_hand())
    assert extended_fingers(make_hand(closed=True)) == 0
    assert not is_open_palm(make_hand(closed=True))

    assert palms_open([make_hand(label="Left"), make_hand(label="Right")])
    assert not palms_open([make_hand(label="Left"), make_hand(closed=True, label="Right")])
    assert not palms_open([make_hand()]), "one hand is not both palms"
    assert not palms_open([])


def _box(**overrides):
    return BoxView(BoxConfig(**overrides), TrackingConfig())


def test_box_pose_sits_between_the_hands():
    box = _box()
    hands = [make_hand(center=(200, 240), label="Left"),
             make_hand(center=(600, 240), label="Right")]
    pose = None
    for frame in range(25):
        pose = box.update(hands, frame / 30.0)

    assert pose is not None
    assert abs(float(pose.centre[0]) - 400) < 30, pose.centre.tolist()
    assert pose.size > 0


def test_box_needs_two_hands():
    box = _box()
    assert box.update([], 0.1) is None
    # A single hand that is not open shows nothing either.
    assert box.update([make_hand(closed=True)], 0.0) is None


def test_one_open_palm_gives_a_solo_pose():
    """One open hand alone means the rose on the palm, with no box."""
    box = _box()
    hand = make_hand(center=(400, 300))
    pose = None
    for frame in range(25):
        pose = box.update([hand], frame / 30.0)

    assert pose is not None and pose.solo
    assert pose.bloom > 0.9, pose.bloom
    assert pose.size > 0
    # Above the top of the hand...
    assert pose.centre[1] < float(hand.points[:, 1].min()), "rose is not clear of the hand"
    # ...and centred over the middle of it, not off to one side. Offsetting
    # along the hand's own axis instead drifts it sideways as the hand tilts.
    assert abs(float(pose.centre[0]) - float(hand.points[:, 0].mean())) < 4.0


def test_solo_rose_stays_centred_when_the_hand_tilts():
    """Screen-space placement, so tilting must not swing the rose sideways."""
    offsets = []
    for angle in (0.0, 0.5, -0.6):
        box = _box()
        base = make_hand(center=(400, 300))
        centre = base.points.mean(axis=0)
        rot = np.array([[math.cos(angle), -math.sin(angle)],
                        [math.sin(angle), math.cos(angle)]], np.float32)
        tilted = Hand(points=((base.points - centre) @ rot.T + centre).astype(np.float32),
                      label="Right", score=1.0)
        pose = None
        for step in range(25):
            pose = box.update([tilted], step / 30.0)
        offsets.append(float(pose.centre[0]) - float(tilted.points[:, 0].mean()))

    assert max(abs(o) for o in offsets) < 4.0, offsets


def test_two_hands_are_never_a_solo_pose():
    box = _box()
    hands = [make_hand(center=(250, 300), label="Left"),
             make_hand(center=(550, 300), label="Right")]
    pose = None
    for frame in range(25):
        pose = box.update(hands, frame / 30.0)
    assert pose is not None and not pose.solo


def test_solo_pose_draws_no_box():
    """The wireframe, rays and effect window all belong to the box."""
    boxed = _box()
    solo = _box()
    hands = [make_hand(center=(250, 300), label="Left"),
             make_hand(center=(550, 300), label="Right")]
    one = [make_hand(center=(400, 300))]

    frames = {}
    for name, view, given in (("box", boxed, hands), ("solo", solo, one)):
        frame = np.zeros((600, 800, 3), np.uint8)
        pose = None
        for step in range(20):
            pose = view.update(given, step / 30.0)
        # Rose closed, so anything drawn has to be the box itself.
        pose.bloom = 0.0
        view.render(frame, pose, given, "cyanotype", EffectContext(), {})
        frames[name] = int(frame.sum())

    assert frames["box"] > 0, "the box drew nothing"
    assert frames["solo"] == 0, "something from the box was drawn in solo mode"


def test_box_size_follows_the_distance_between_the_hands():
    def settled(gap):
        box = _box()
        pose = None
        hands = [make_hand(center=(400 - gap / 2, 240), label="Left"),
                 make_hand(center=(400 + gap / 2, 240), label="Right")]
        for frame in range(30):
            pose = box.update(hands, frame / 30.0)
        return pose.size

    assert settled(500) > settled(250) * 1.5


def test_rose_blooms_open_on_palms_and_closes_again():
    box = _box(bloom_speed=4.0)
    open_hands = [make_hand(center=(250, 240), label="Left"),
                  make_hand(center=(550, 240), label="Right")]
    closed_hands = [make_hand(center=(250, 240), closed=True, label="Left"),
                    make_hand(center=(550, 240), closed=True, label="Right")]

    pose = None
    for frame in range(40):
        pose = box.update(open_hands, frame / 30.0)
    assert pose.bloom > 0.95, pose.bloom

    # It eases shut rather than snapping - a dropped detection should not make
    # the rose flicker.
    first_step = None
    for frame in range(40, 80):
        previous = pose.bloom
        pose = box.update(closed_hands, frame / 30.0)
        if first_step is None:
            first_step = previous - pose.bloom
    assert pose.bloom < 0.05, pose.bloom
    assert 0 < first_step < 0.25, first_step


def test_rose_meshes_are_cached_between_frames():
    """Regrowing 36 petals every frame is waste when the bloom barely moves."""
    box = _box()
    box._rose(1.0)
    first = box._rose(1.0)
    assert box._rose(1.0) is first
    # A coarse step apart still reuses, a big change does not.
    assert box._rose(1.0 - 1e-4) is first
    assert box._rose(0.2) is not first


# --------------------------------------------------------------------------
# pointing, and the orb
# --------------------------------------------------------------------------

def test_two_fingers_is_told_apart_from_the_other_poses():
    assert is_two_fingers(make_hand(two_fingers=True))
    assert not is_two_fingers(make_hand()), "an open palm is not two fingers"
    assert not is_two_fingers(make_hand(closed=True)), "a fist is not two fingers"

    # One finger is not enough - index and middle must both be out.
    index_only = make_hand(two_fingers=True)
    index_only.points[12] = np.array([320.0, 240.0], np.float32)  # middle curled in
    assert not is_two_fingers(index_only)


def test_two_fingers_ignores_the_thumb():
    """People hold this sign with the thumb tucked or cocked and mean the same."""
    tucked = make_hand(two_fingers=True)
    tucked.points[THUMB_TIP] = tucked.points[9]  # thumb folded across the palm
    assert is_two_fingers(tucked)
    assert is_two_fingers(make_hand(two_fingers=True))


def test_orb_hand_is_picked_out_of_several():
    open_hand = make_hand(center=(200, 240), label="Left")
    signer = make_hand(center=(500, 240), label="Right", two_fingers=True)
    assert orb_hand([open_hand, signer]) is signer
    assert orb_hand([open_hand]) is None
    assert orb_hand([]) is None


def _orb(**overrides):
    return OrbView(OrbConfig(**overrides), TrackingConfig())


def test_orb_appears_for_the_two_finger_sign_and_fades_out():
    orb = _orb(fade_speed=5.0)
    pointer = [make_hand(center=(400, 300), two_fingers=True)]

    pose = None
    for step in range(25):
        pose = orb.update(pointer, step / 30.0)
    assert pose is not None and pose.fade > 0.95, pose

    # Easing out, not snapping - the orb should not blink off.
    first_drop = None
    for step in range(25, 60):
        previous = pose.fade
        pose = orb.update([make_hand(center=(400, 300))], step / 30.0)
        if first_drop is None:
            first_drop = previous - (pose.fade if pose else 0.0)
        if pose is None:
            break
    assert first_drop is not None and 0 < first_drop < 0.4, first_drop


def test_orb_sits_off_the_ends_of_both_fingers():
    orb = _orb()
    hand = make_hand(center=(400, 300), two_fingers=True)
    pose = None
    for step in range(25):
        pose = orb.update([hand], step / 30.0)

    # Beyond the fingertips, and centred between them rather than on one.
    tips = hand.points[[INDEX_TIP, MIDDLE_TIP]]
    to_tips = np.linalg.norm(tips - hand.points[WRIST], axis=1).max()
    to_orb = float(np.linalg.norm(pose.centre - hand.points[WRIST]))
    assert to_orb > to_tips, (to_orb, to_tips)
    assert min(tips[:, 0]) - 5 <= pose.centre[0] <= max(tips[:, 0]) + 5
    assert pose.radius > 0


def test_orb_draws_inside_its_own_box_only():
    orb = _orb()
    frame = np.zeros((400, 600, 3), np.uint8)
    pose = OrbPose(centre=np.array([300.0, 200.0], np.float32), radius=18.0,
                   fade=1.0, spin=0.0)
    assert orb.render(frame, pose) is True
    assert frame[200, 300].sum() > 0, "nothing was drawn"
    for y, x in ((3, 3), (3, 596), (396, 3), (396, 596)):
        assert frame[y, x].sum() == 0, f"orb leaked to {(y, x)}"


def test_orb_core_is_hottest_at_its_centre():
    orb = _orb(rings=0, bloom=0.0)
    frame = np.zeros((400, 600, 3), np.uint8)
    pose = OrbPose(np.array([300.0, 200.0], np.float32), 24.0, 1.0, 0.0)
    orb.render(frame, pose)
    assert int(frame[200, 300].sum()) > int(frame[200, 340].sum())


def test_orb_draws_nothing_when_faded_out_or_degenerate():
    orb = _orb()
    frame = np.zeros((300, 300, 3), np.uint8)
    before = frame.copy()
    centre = np.array([150.0, 150.0], np.float32)

    assert orb.render(frame, OrbPose(centre, 20.0, 0.0, 0.0)) is False
    assert orb.render(frame, OrbPose(centre, 0.0, 1.0, 0.0)) is False
    assert orb.render(frame, OrbPose(np.array([-900.0, -900.0], np.float32),
                                     20.0, 1.0, 0.0)) is False
    assert orb.render(frame, OrbPose(np.array([np.nan, 10.0], np.float32),
                                     20.0, 1.0, 0.0)) is False
    assert np.array_equal(frame, before)


def test_orb_eases_out_rather_than_vanishing_when_suppressed():
    """Leaving cube mode hides it by feeding it no hands, not by dropping it.

    Cutting the render instead would snap a lit orb off mid-glow.
    """
    orb = _orb(fade_speed=4.0)
    signer = [make_hand(center=(400, 300), two_fingers=True)]
    pose = None
    for step in range(25):
        pose = orb.update(signer, step / 30.0)
    assert pose.fade > 0.95

    # What the app does while the orb is suppressed.
    faded = orb.update([], 25 / 30.0)
    assert faded is not None, "it should still be easing out, not gone"
    assert 0.0 < faded.fade < pose.fade, faded.fade


def test_the_box_keeps_the_orb_down():
    """Two questions, kept apart: does this mode carry the orb, and is
    something in it standing in the orb's way right now.

    The box and the orb fill the same part of the picture, so they are never on
    together - and that outranks the mode's own wish for the orb. The rose alone
    on one open palm is not the box, so the orb sits beside that one.
    """
    # A mode that carries the orb, with something blocking: down regardless.
    assert orb_visible(True, blocked=True) is False
    # The same mode with nothing in the way.
    assert orb_visible(True, blocked=False) is True
    # A mode that does not carry it at all stays down either way.
    assert orb_visible(False, blocked=False) is False
    assert orb_visible(False, blocked=True) is False


def test_a_solo_rose_pose_still_lets_the_orb_through():
    """The block keys off a real box, not merely off having a pose."""
    mode = modes.CubeMode(Config())
    mode.wants_orb = True

    for step in range(12):
        mode.update([make_hand(center=(360, 300))], step / 30.0)
    assert mode._pose is not None and mode._pose.solo
    assert mode.blocks_orb is False
    assert orb_visible(mode.wants_orb, mode.blocks_orb) is True

    two = [make_hand(center=(260, 300)), make_hand(center=(620, 300))]
    for step in range(12):
        mode.update(two, (12 + step) / 30.0)
    assert mode._pose is not None and not mode._pose.solo
    assert mode.blocks_orb is True
    assert orb_visible(mode.wants_orb, mode.blocks_orb) is False


def _ring_samples(orb_cfg, spin, radius=70.0):
    """Render the orb and read the frame where its nearest and furthest ring
    points landed. Both are sampled clear of the ball, so what comes back is
    ring and not sphere."""
    view = OrbView(orb_cfg, TrackingConfig())
    frame = np.zeros((500, 500, 3), np.uint8)
    centre = np.array([250.0, 250.0], np.float32)
    assert view.render(frame, OrbPose(centre, radius, 1.0, spin)) is True

    arcs = scene3d.ring_arcs(orb_cfg.rings, orb_cfg.ring_scale, orb_cfg.ring_arcs)
    points, camera = scene3d.project(
        np.concatenate(arcs), scene3d.rotation(spin, spin * 0.45, 0.0),
        centre, radius, orb_cfg.perspective)
    clear = np.linalg.norm(points - centre, axis=1) > radius * 1.3

    def patch(point):
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        return frame[y - 1:y + 2, x - 1:x + 2].reshape(-1, 3)

    return (patch(points[clear][np.argmin(camera[clear, 2])]),
            patch(points[clear][np.argmax(camera[clear, 2])]))


def test_orb_rings_pass_behind_the_ball():
    """Each ring is handed to the glow renderer in arcs, not whole.

    That renderer shades one depth per curve, so a ring given to it in one
    piece comes out flat at its mean depth and the orb loses the only cue that
    says the rings go round it rather than sit on it. Nothing else in the
    render would notice, which is why it is pinned here.
    """
    for spin in (0.3, 0.9, 1.7, 2.6):
        near, far = _ring_samples(OrbConfig(), spin)
        assert near.sum() > 2.0 * far.sum(), (spin, near.sum(), far.sum())


def test_orb_rings_stay_red_however_hot_they_burn():
    """The rings have a colour ramp of their own, and need one.

    The glow renderer's nearest depth band burns at about 222 of 255. On the
    sphere's ramp, which runs out to near-white for its core, that lands pale -
    so the rings get a ramp that tops out red instead.
    """
    near, _ = _ring_samples(OrbConfig(), 0.9)
    red, blue = float(near[:, 2].max()), float(near[:, 0].max()) + 1.0
    assert red > 3.0 * blue, (red, blue)

    # And the sphere's own ramp really would wash them out - otherwise the
    # second ramp is complexity nobody can see the point of.
    pale, _ = _ring_samples(OrbConfig(ring_edge_colour=OrbConfig().edge_colour), 0.9)
    assert float(pale[:, 2].max()) < 2.0 * (float(pale[:, 0].max()) + 1.0)


def test_orb_geometry_is_built_once():
    """Regenerating the cloud every frame would cost more than drawing it."""
    cloud = scene3d.sphere_cloud()
    assert scene3d.sphere_cloud() is cloud
    assert cloud.dtype == np.float32

    arcs = scene3d.ring_arcs()
    assert scene3d.ring_arcs() is arcs


def test_orb_cloud_is_a_sphere_weighted_to_its_middle():
    """Brightness is density, so the radial distribution is the whole look."""
    cloud = scene3d.sphere_cloud(count=8000, shell=0.4, core_bias=1.4)
    radius = np.linalg.norm(cloud, axis=1)
    assert float(radius.max()) <= 1.0001, radius.max()

    # Denser in the middle than an even fill of the volume would be, which is
    # what gives it a burning core instead of a uniform ball.
    assert float((radius < 0.5).mean()) > 0.5 ** 3 * 2, float((radius < 0.5).mean())
    # And a rim: the shell puts a spike of points at the surface.
    assert float((radius > 0.98).mean()) > 0.3, float((radius > 0.98).mean())


def test_orb_cloud_is_a_ball_in_depth_not_a_disc():
    """It has to occupy the depth it appears to, not just look round.

    The perspective bulge of the near hemisphere and the depth banding the
    rings are shaded by both read from this axis, and a cloud flattened to a
    disc would still pass every other test here.
    """
    cloud = scene3d.sphere_cloud()
    _, camera = scene3d.project(cloud, scene3d.rotation(0.4, 0.2, 0.0),
                                (300.0, 200.0), 60.0, 2.4)
    assert float(np.ptp(camera[:, 2])) > 100.0, np.ptp(camera[:, 2])


def test_orb_rings_are_projected_circles_not_drawn_ones():
    """Turned edge-on a great circle has to flatten, which a cv2.circle cannot."""
    arcs = scene3d.ring_arcs(rings=1, radius=1.45, arcs=8)
    circle = np.concatenate(arcs)

    flat, _ = scene3d.project(circle, scene3d.rotation(0.0, 0.0, 0.0),
                              (200.0, 200.0), 50.0)
    edge, _ = scene3d.project(circle, scene3d.rotation(0.0, 1.35, 0.0),
                              (200.0, 200.0), 50.0)

    assert np.ptp(flat[:, 1]) > 0.7 * np.ptp(flat[:, 0]), "face-on it should be round"
    assert np.ptp(edge[:, 1]) < 0.4 * np.ptp(edge[:, 0]), (np.ptp(edge[:, 1]),
                                                           np.ptp(edge[:, 0]))


# --------------------------------------------------------------------------
# camera thread
# --------------------------------------------------------------------------

class _FakeCapture:
    """Stands in for cv2.VideoCapture, with a settable frame interval."""

    def __init__(self, interval=0.01, fail_after=None):
        self.interval = interval
        self.fail_after = fail_after
        self.count = 0

    def read(self):
        time.sleep(self.interval)
        self.count += 1
        if self.fail_after is not None and self.count > self.fail_after:
            return False, None
        frame = np.full((8, 8, 3), self.count % 256, dtype=np.uint8)
        return True, frame


def test_grabber_hands_over_frames():
    grabber = FrameGrabber(_FakeCapture(interval=0.005)).start()
    try:
        seen = [grabber.read(timeout=1.0) for _ in range(3)]
        assert all(f is not None for f in seen), seen
    finally:
        grabber.stop()


def test_grabber_drops_stale_frames_instead_of_queueing():
    """A slow consumer must get the newest frame, not a growing backlog."""
    grabber = FrameGrabber(_FakeCapture(interval=0.005)).start()
    try:
        first = grabber.read(timeout=1.0)
        assert first is not None
        time.sleep(0.15)  # let several frames go by unread
        latest = grabber.read(timeout=1.0)
        assert int(latest[0, 0, 0]) != (int(first[0, 0, 0]) + 1) % 256, (
            "reader appears to be queueing frames rather than dropping them"
        )
    finally:
        grabber.stop()


def test_grabber_reports_camera_failure():
    grabber = FrameGrabber(_FakeCapture(interval=0.002, fail_after=2)).start()
    try:
        for _ in range(20):
            if grabber.read(timeout=0.5) is None:
                break
            time.sleep(0.01)
        assert grabber.failed, "failure was not reported to the reader"
        assert grabber.read(timeout=0.2) is None
    finally:
        grabber.stop()


# --------------------------------------------------------------------------
# the mode system
#
# None of this had any cover before: the app's mode branch, its keybindings and
# its readout were all untested, so nothing would have caught the refactor
# breaking any of them. Since that refactor is exactly what widened a boolean
# into a list, it is the moment to put them under test.
# --------------------------------------------------------------------------

def _app() -> HologramApp:
    """The real app object, minus the camera. Constructing it is most of what
    these tests are checking - it wires the modes, the orb and the effects."""
    return HologramApp(Config())


def _tile(size=(240, 320)) -> np.ndarray:
    """Something with structure in it, so a mode that warps has an edge to
    bite on and a mode that writes nothing can be told from one that does."""
    height, width = size
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    tile = np.stack([
        120 + 70 * np.sin(xs / 21.0),
        105 + 60 * np.cos(ys / 17.0),
        135 + 55 * np.sin((xs + ys) / 27.0),
    ], axis=-1)
    return np.clip(tile, 0, 255).astype(np.uint8)


def test_cycling_the_modes_comes_back_to_where_it_started():
    app = _app()
    assert len(app.modes) > 1, "a list of one mode is a boolean again"
    start = app.mode.name
    seen = []
    for _ in range(len(app.modes)):
        seen.append(app.mode.name)
        app.step_mode(1)
    assert app.mode.name == start
    assert len(set(seen)) == len(app.modes), f"a mode was visited twice: {seen}"

    # And backwards, which is the same arithmetic with the other sign.
    for _ in range(len(app.modes)):
        app.step_mode(-1)
    assert app.mode.name == start


def test_every_mode_survives_no_hands_one_hand_and_two():
    """The whole point of the list is that the app stops knowing what a mode is,
    so every mode has to cope with every pose on its own."""
    poses = {
        "none": [],
        "one": [make_hand(center=(160, 140), scale=34.0)],
        "two": [make_hand(center=(90, 140), scale=30.0),
                make_hand(center=(230, 140), scale=30.0)],
    }
    ctx = EffectContext()
    for mode in _app().modes:
        for pose, hands in poses.items():
            frame = _tile()
            # Several frames: the modes that ease, fade or keep a history only
            # reach their interesting state after a few.
            for step in range(14):
                mode.update(hands, step / 30.0)
                mode.render(frame, hands, "halftone", ctx, {})
            where = f"{mode.name}/{pose}"
            assert frame.dtype == np.uint8, f"{where} changed the frame's type"
            assert frame.shape == (240, 320, 3), f"{where} changed its shape"
            assert np.isfinite(frame).all(), f"{where} wrote non-finite pixels"
            assert isinstance(mode.readout(), str), f"{where} readout is not text"
            mode.reset()


def test_a_mode_switch_clears_what_the_last_one_was_holding():
    """State that outlives a switch is the bug the boolean version had: four
    scalars written in one branch and read in the other."""
    app = _app()
    hands = [make_hand(center=(120, 140), scale=30.0),
             make_hand(center=(220, 140), scale=30.0)]
    scan = next(m for m in app.modes if m.name == "slitscan")
    for step in range(6):
        scan.update(hands, step / 30.0)
        scan.render(_tile(), hands, "halftone", EffectContext(), {})
    assert scan.scan.frames > 0, "nothing was recorded to begin with"
    scan.reset()
    assert scan.scan.frames == 0, "a second of the old mode survived the switch"


def test_n_walks_the_modes_and_b_still_jumps_to_the_cube():
    """`b` predates the mode list and is in the README, both launchers and
    everyone's fingers, so it stays a direct jump rather than becoming an
    index into a list that may grow."""
    app = _app()
    names = [m.name for m in app.modes]
    assert app.mode.name == names[0]
    for expected in names[1:] + names[:1]:
        app._handle_key(ord("n"), None)
        assert app.mode.name == expected, f"n went to {app.mode.name}, not {expected}"

    # Back at the start of the list, wherever that is.
    assert app.mode.name == names[0] == "shape"
    app._handle_key(ord("b"), None)
    assert app.mode.name == "cube", "b should jump straight to cube mode"
    app._handle_key(ord("b"), None)
    assert app.mode.name == "shape", "b should come back out of cube mode"

    assert app._handle_key(ord("q"), None) is True
    assert app._handle_key(27, None) is True
    assert app._handle_key(255, None) is False, "no key pressed is not an exit"


def test_a_mode_takes_its_own_keys_before_the_app_does():
    """`p` belongs to shape mode. Keeping it there is what stops _handle_key
    growing a branch per mode."""
    app = _app()
    shape = next(m for m in app.modes if m.name == "shape")
    assert app.mode is shape
    before = shape.per_hand
    app._handle_key(ord("p"), None)
    assert shape.per_hand is not before

    # In another mode the same key is simply not claimed, and nothing happens.
    app.go_to_mode("slitscan")
    settled = shape.per_hand
    app._handle_key(ord("p"), None)
    assert shape.per_hand is settled


def test_the_hud_takes_the_line_the_mode_wrote():
    """hud.py used to branch on `cube` and carry every mode's fields flat. It
    now draws whatever the mode says, so adding a mode costs it nothing."""
    app = _app()
    state = app._hud_state([], 0.0)
    assert state.mode == app.mode.name
    assert state.readout == app.mode.readout()

    frame = _tile()
    hud.draw(frame, state, HudConfig())
    assert frame.max() > 0
    # Every mode's line has to be drawable, whatever it says.
    for mode in app.modes:
        hud.draw(_tile(), hud.HudState(
            effect="halftone", fps=30.0, hands=0, mode=mode.name,
            readout=mode.readout(), orb=False, recording=False, show_help=True,
        ), HudConfig())


def test_an_unknown_mode_in_the_config_is_skipped_not_fatal():
    """An old config naming a mode that no longer exists should still start."""
    cfg = Config()
    cfg.modes.order = ["cube", "no-such-mode", "shape"]
    built = [m.name for m in modes.build(cfg)]
    assert "no-such-mode" not in built
    assert built[:2] == ["cube", "shape"], built
    # And a mode left out of the order is still available, appended after.
    assert set(built) == set(modes.REGISTRY)


def test_the_orb_and_the_bloom_are_named_modes_not_flags():
    cfg = Config()
    cfg.modes.orb = ["slitscan"]
    cfg.modes.bloom = ["cube"]
    built = {m.name: m for m in modes.build(cfg)}
    assert built["slitscan"].wants_orb and not built["cube"].wants_orb
    assert built["cube"].wants_bloom and not built["slitscan"].wants_bloom


def test_a_flat_object_is_lit_at_the_front_not_the_back():
    """draw_glow bands by depth. With no depth spread the old arithmetic put
    every curve in the *back* band, drawing the whole thing at the dimmest
    level - and float noise around zero scattered the curves across bands at
    random, so one closed curve came out as broken arcs."""
    turn = np.linspace(0, 2 * np.pi, 40, dtype=np.float32)
    ring = np.stack([np.cos(turn), np.sin(turn)], axis=1) * 40.0 + 100.0
    lut = scene3d.glow_lut((140, 60, 20), (255, 220, 150))

    flat = np.zeros((200, 200, 3), np.uint8)
    scene3d.draw_glow(flat, [ring.astype(np.float32)], np.zeros(1, np.float32),
                      lut, thickness=2)
    assert flat.max() > 120, f"a flat ring came out dim: {flat.max()}"

    # Curves at genuinely different depths still grade, near over far.
    graded = np.zeros((200, 200, 3), np.uint8)
    scene3d.draw_glow(graded, [ring.astype(np.float32), (ring * 0.6).astype(np.float32)],
                      np.array([0.0, 1.0], np.float32), lut, thickness=2)
    assert graded.max() > 120



# --------------------------------------------------------------------------
# slit-scan
# --------------------------------------------------------------------------

def test_the_slit_scan_history_is_bounded():
    """It is the only mode holding a big buffer, so an unbounded one is the
    difference between 20 MB and all of memory."""
    scan = slitscan.SlitScan(length=8, scale=0.5)
    for step in range(200):
        scan.push(np.full((30, 40, 3), step % 256, np.uint8))
    assert scan.frames == 8
    assert scan._ring.nbytes == 8 * 30 * 40 * 3


def test_the_slit_scan_opens_on_a_normal_picture_not_a_black_one():
    """The ring starts full of copies of the first frame, so the mode dissolves
    into the effect over its first second instead of flashing black."""
    mode = modes.SlitScanMode(Config())
    frame = _tile()
    before = frame.copy()
    mode.update([], 0.0)
    mode.render(frame, [], "halftone", EffectContext(), {})
    assert frame.max() > 0, "it opened black"
    # Half res down and back up, so not pixel-identical - but the same picture.
    assert np.abs(frame.astype(int) - before.astype(int)).mean() < 12


def test_each_slit_scan_column_comes_from_its_own_moment():
    """The claim the whole mode rests on. Fed frames that are flat and
    distinct, a correct scan comes back as bands, one per moment."""
    scan = slitscan.SlitScan(length=10, scale=1.0)
    for step in range(10):
        scan.push(np.full((20, 60, 3), 20 + step * 20, np.uint8))

    ages = slitscan.age_map(60, 10, centre=0.0, spread=1.0, direction=1)
    assert ages.min() == 0 and ages.max() == 9, (ages.min(), ages.max())

    out = scan.gather(ages)
    assert len(np.unique(out)) == 10, f"only {len(np.unique(out))} moments showed"
    # Column 0 asks for the newest frame, which is the last one pushed.
    assert out[0, 0, 0] == 200


def test_the_gather_reuses_its_output_buffer():
    """Allocating a fresh frame per gather, and transposing into it, is what
    the run-slice arrangement exists to avoid."""
    scan = slitscan.SlitScan(length=6, scale=1.0)
    scan.push(np.zeros((20, 40, 3), np.uint8))
    ages = slitscan.age_map(40, 6, 0.5, 1.0)
    assert scan.gather(ages) is scan.gather(ages)


def test_the_scan_follows_the_nearest_hand():
    mode = modes.SlitScanMode(Config())
    mode._width = 320.0
    # The nearer hand - the one with the larger scale - leads, so bringing one
    # forward takes the scan rather than the two fighting over it.
    hands = [make_hand(center=(40, 140), scale=20.0),
             make_hand(center=(280, 140), scale=44.0)]
    for step in range(60):
        mode.update(hands, step / 30.0)
    assert mode.centre > 0.6, mode.centre

    mode.update([], 2.5)
    assert not mode._following, "with no hands it should sweep by itself"



# --------------------------------------------------------------------------

def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  pass  {name}")
        except Exception:
            failures.append(name)
            print(f"  FAIL  {name}")
            traceback.print_exc()

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("failed: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
