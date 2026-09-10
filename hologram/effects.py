"""The effects applied inside the shape.

Each effect takes the cropped region and returns a new image of the same size;
none of them modify the input in place, because the caller still needs the
original pixels to composite through the shape's mask.

Everything is vectorised numpy/OpenCV. A per-pixel Python loop here would drop
the frame rate through the floor.

Three of these - risograph, halftone and dither - are the same machine wearing
different clothes: a small threshold matrix is tiled across the region and one
uint8 compare turns the picture into ink. Screens, glyph atlases and colour
ramps are functions of a size or a setting and never of the picture, so each is
built once and reused for the life of the process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

Params = dict[str, Any]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_MAX_CACHE = 32


# ---------------------------------------------------------------------------
# Colour ramps. Colourising through a lookup table keeps the whole path in
# uint8; doing it as float multiplication cost roughly twenty times as much.
# ---------------------------------------------------------------------------
_LUT_CACHE: dict[tuple, np.ndarray] = {}


def _ramp_lut(stops, contrast: float = 1.0) -> np.ndarray:
    """A 256-entry BGR ramp through evenly spaced colour stops.

    ``contrast`` pivots the input around mid grey before the ramp is sampled,
    so a print's tone curve and its ink colours collapse into one table.
    """
    key = (tuple(tuple(int(c) for c in stop) for stop in stops), round(float(contrast), 3))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        if len(_LUT_CACHE) > _MAX_CACHE:
            _LUT_CACHE.clear()
        value = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        if contrast != 1.0:
            value = np.clip((value - 0.5) * contrast + 0.5, 0.0, 1.0)
        positions = np.linspace(0.0, 1.0, len(key[0]), dtype=np.float32)
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for channel in range(3):
            column = [stop[channel] for stop in key[0]]
            lut[:, 0, channel] = np.interp(value, positions, column).clip(0, 255).astype(np.uint8)
        _LUT_CACHE[key] = lut
    return lut


def _tint_lut(tint) -> np.ndarray:
    """Black up to ``tint`` - the ramp a single-channel image is coloured through."""
    return _ramp_lut(((0, 0, 0), tint))


def _over(paper, ink) -> tuple[int, int, int]:
    """One ink multiplied onto paper: the colour a printed dot actually lands."""
    return tuple(int(p) * int(i) // 255 for p, i in zip(paper, ink))


# ---------------------------------------------------------------------------
# Threshold screens. Both are ordered dithers; the only difference is where in
# the cell the low thresholds sit, which is the whole difference between a
# printer's dot and a spray of pixels.
# ---------------------------------------------------------------------------
_SCREEN_CACHE: dict[tuple, np.ndarray] = {}


def _bayer_cell(size: int) -> np.ndarray:
    """The recursive Bayer matrix, as thresholds spread evenly over 0-255.

    Doubles until it covers the size asked for, so it is always a power of two.
    """
    cell = np.zeros((1, 1), dtype=np.float32)
    while cell.shape[0] < size:
        cell = np.block([[4 * cell, 4 * cell + 2], [4 * cell + 3, 4 * cell + 1]])
    return (cell + 0.5) / cell.size * 255.0


def _clustered_cell(size: int) -> np.ndarray:
    """A printer's screen: thresholds rise outwards from the centre of the cell.

    Ranking the cell by a spot function rather than writing the matrix out by
    hand means any size works, and the dots grow round from the middle and meet
    at the corners the way a real screen does.
    """
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    spot = (-np.cos(2 * np.pi * (x + 0.5) / size)
            - np.cos(2 * np.pi * (y + 0.5) / size))
    order = np.argsort(spot, axis=None).argsort().reshape(size, size)
    return (order + 0.5) / order.size * 255.0


def _screen(kind: str, size: int, height: int, width: int, phase=(0, 0)) -> np.ndarray:
    """A threshold screen tiled to cover the region, built once per size.

    ``phase`` rolls the cell before tiling, which is how two ink passes are kept
    from landing their dots on top of each other and beating into moire.
    """
    key = (kind, size, height, width, tuple(phase))
    tile = _SCREEN_CACHE.get(key)
    if tile is None:
        if len(_SCREEN_CACHE) > _MAX_CACHE:
            _SCREEN_CACHE.clear()
        cell = _bayer_cell(size) if kind == "bayer" else _clustered_cell(size)
        if any(phase):
            cell = np.roll(np.roll(cell, phase[0], axis=0), phase[1], axis=1)
        cell = np.clip(cell, 0, 255).astype(np.uint8)
        repeats = (-(-height // cell.shape[0]), -(-width // cell.shape[1]))
        tile = np.ascontiguousarray(np.tile(cell, repeats)[:height, :width])
        _SCREEN_CACHE[key] = tile
    return tile


_GRID_CACHE: dict[tuple, np.ndarray] = {}


def _grid_lines(block: int, height: int, width: int) -> np.ndarray:
    """The top and left edge of every block at one size."""
    key = (block, height, width)
    lines = _GRID_CACHE.get(key)
    if lines is None:
        if len(_GRID_CACHE) > _MAX_CACHE:
            _GRID_CACHE.clear()
        lines = np.zeros((height, width), np.uint8)
        lines[::block, :] = 255
        lines[:, ::block] = 255
        _GRID_CACHE[key] = lines
    return lines


_DENSITY_CACHE: dict[tuple, np.ndarray] = {}


def _density_lut(kind: str, gain: float) -> np.ndarray:
    """How much ink one pass lays down at each brightness.

    ``shadow`` carries the picture and ``midtone`` is the accent pass: weighting
    the second towards the middle of the range makes the two inks overlap in a
    band rather than one simply sitting under the other.
    """
    key = (kind, round(float(gain), 3))
    lut = _DENSITY_CACHE.get(key)
    if lut is None:
        if len(_DENSITY_CACHE) > _MAX_CACHE:
            _DENSITY_CACHE.clear()
        value = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        # sin(pi) lands a hair below zero in floating point, and a negative
        # base under a fractional power is a NaN, so clamp before the power.
        bump = np.clip(np.sin(np.pi * value), 0.0, 1.0)
        curve = (1.0 - value) ** 0.85 if kind == "shadow" else bump ** 1.4
        lut = _DENSITY_CACHE[key] = np.clip(curve * gain * 255.0, 0, 255).astype(np.uint8)
    return lut


_QUANT_CACHE: dict[int, np.ndarray] = {}


def _quantise_lut(levels: int) -> np.ndarray:
    """Snap to ``levels`` evenly spaced tones."""
    lut = _QUANT_CACHE.get(levels)
    if lut is None:
        value = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        snapped = np.rint(value * (levels - 1)) / (levels - 1)
        lut = _QUANT_CACHE[levels] = np.clip(snapped * 255.0, 0, 255).astype(np.uint8)
    return lut


_ATLAS_CACHE: dict[tuple, np.ndarray] = {}


def _glyph_atlas(ramp: str, cell_w: int, cell_h: int) -> np.ndarray:
    """Every character in the ramp drawn once into its own tile."""
    key = (ramp, cell_w, cell_h)
    atlas = _ATLAS_CACHE.get(key)
    if atlas is None:
        if len(_ATLAS_CACHE) > 8:
            _ATLAS_CACHE.clear()
        scale = max(cell_h / 30.0, 0.2)
        thickness = 2 if cell_h >= 20 else 1
        atlas = np.zeros((len(ramp), cell_h, cell_w), np.uint8)
        for index, char in enumerate(ramp):
            (text_w, text_h), _ = cv2.getTextSize(char, _FONT, scale, thickness)
            origin = ((cell_w - text_w) // 2, (cell_h + text_h) // 2)
            cv2.putText(atlas[index], char, origin, _FONT, scale, 255, thickness, cv2.LINE_AA)
        _ATLAS_CACHE[key] = atlas
    return atlas


_SOLID_CACHE: dict[tuple, np.ndarray] = {}


def _solid(shape, colour) -> np.ndarray:
    """A block of one colour, for masked fills through cv2.copyTo."""
    key = (tuple(shape), tuple(int(c) for c in colour))
    block = _SOLID_CACHE.get(key)
    if block is None:
        if len(_SOLID_CACHE) > 8:
            _SOLID_CACHE.clear()
        block = _SOLID_CACHE[key] = np.full(key[0], key[1], np.uint8)
    return block


def soft_blur(image: np.ndarray, sigma: float, downscale: int = 4) -> np.ndarray:
    """Approximate a wide blur by blurring a shrunken copy and scaling it back.

    A soft glow has no high-frequency detail to lose, so this looks the same as
    blurring at full resolution and costs a fraction as much.
    """
    height, width = image.shape[:2]
    small_w, small_h = max(width // downscale, 1), max(height // downscale, 1)
    if small_w < 2 or small_h < 2:
        return cv2.GaussianBlur(image, (0, 0), sigma)
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    small = cv2.GaussianBlur(small, (0, 0), max(sigma / downscale, 0.6))
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


@dataclass
class EffectContext:
    """State carried between frames, for effects that need history."""

    frame_index: int = 0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    state: dict[str, Any] = field(default_factory=dict)

    def clear(self) -> None:
        self.state.clear()


def _scanlines(image: np.ndarray, gap: int, strength: float) -> np.ndarray:
    """Darken every ``gap``-th row, the CRT look. Modifies the image given."""
    if gap > 1 and strength > 0:
        image[::gap] = (image[::gap].astype(np.float32) * (1.0 - strength)).astype(np.uint8)
    return image


def _grain(gray: np.ndarray, ctx: EffectContext, amount: float) -> np.ndarray:
    """Add noise to a single-channel image.

    A third of the random numbers a three-channel grain would need, for the same
    look once the picture is coloured through a ramp.
    """
    if amount <= 0:
        return gray
    amplitude = max(int(amount * 255), 1)
    noise = ctx.rng.integers(-amplitude, amplitude + 1, size=gray.shape, dtype=np.int16)
    return np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# The effects
# ---------------------------------------------------------------------------
def _risograph(roi: np.ndarray, ctx: EffectContext, params: Params) -> np.ndarray:
    """Two spot inks screened onto paper, slightly out of register.

    A riso print is not a filter over a picture: it is two passes of solid ink,
    each screened into dots, on paper that shows through. Compositing it as
    multiplies over the paper colour is what gives the ink its flat, faintly
    wrong colour - adding the layers instead just lights the picture up.
    """
    height, width = roi.shape[:2]
    gray = _grain(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), ctx, float(params.get("grain", 0.10)))

    dot = max(int(params.get("dot", 6)), 2)
    paper = tuple(params.get("paper", (238, 240, 240)))
    ink_a = tuple(params.get("ink_a", (0, 255, 0)))
    ink_b = tuple(params.get("ink_b", (190, 40, 220)))

    first = cv2.compare(
        cv2.LUT(gray, _density_lut("shadow", float(params.get("ink_a_gain", 0.80)))),
        _screen("clustered", dot, height, width), cv2.CMP_GT)
    second = cv2.compare(
        cv2.LUT(gray, _density_lut("midtone", float(params.get("ink_b_gain", 0.55)))),
        _screen("clustered", dot, height, width, (dot // 2, dot // 3)), cv2.CMP_GT)

    # Misregistration: the second pass lands a pixel or two off, the way a
    # second run through the drum does.
    offset = params.get("offset", (2, 1))
    if any(offset):
        second = np.roll(np.roll(second, int(offset[0]), axis=0), int(offset[1]), axis=1)

    # The paper is baked into the first pass's table, so only the second layer
    # needs a multiply of its own.
    out = cv2.applyColorMap(first, _ramp_lut((paper, _over(paper, ink_a))))
    return cv2.multiply(out, cv2.applyColorMap(second, _ramp_lut(((255, 255, 255), ink_b))),
                        scale=1.0 / 255.0)


def _cyanotype(roi: np.ndarray, ctx: EffectContext, params: Params) -> np.ndarray:
    """The blueprint: one Prussian-blue ramp, hard contrast, paper highlights.

    A cyanotype has no colour of its own to keep - the iron salts only ever go
    blue - so the whole effect is a tone curve into a single ramp, which is one
    table lookup per pixel.
    """
    gray = _grain(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), ctx, float(params.get("grain", 0.05)))
    stops = (tuple(params.get("shadow", (92, 46, 16))),
             tuple(params.get("mid", (200, 120, 40))),
             tuple(params.get("paper", (238, 244, 244))))
    return cv2.applyColorMap(gray, _ramp_lut(stops, float(params.get("contrast", 1.35))))


def _quadtree(roi: np.ndarray, ctx: EffectContext, params: Params) -> np.ndarray:
    """Flat blocks that only keep splitting where there is detail to justify it.

    Worked coarse to fine: a block that is already uniform enough settles and
    takes its own average colour, and only what is left carries on down. The
    variance test is made at the block resolution rather than the pixel one, so
    the only full-size work is copying the colour that wins.
    """
    height, width = roi.shape[:2]
    levels = max(int(params.get("levels", 4)), 1)
    min_block = max(int(params.get("min_block", 8)), 1)
    limit = float(params.get("threshold", 26.0)) ** 2  # given as a standard deviation
    grid = bool(params.get("grid", True))

    if height < 2 or width < 2:
        return roi.copy()          # nothing to subdivide

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    squares = cv2.multiply(gray, gray)

    out = np.empty_like(roi)
    active = np.full((height, width), 255, np.uint8)
    level_map = np.zeros((height, width), np.uint8)
    blocks = [min_block * (1 << (levels - 1 - step)) for step in range(levels)]

    for step, block in enumerate(blocks):
        small = (max(width // block, 1), max(height // block, 1))
        colour = cv2.resize(cv2.resize(roi, small, interpolation=cv2.INTER_AREA),
                            (width, height), interpolation=cv2.INTER_NEAREST)
        if step == levels - 1:
            settle = active
        else:
            mean = cv2.resize(gray, small, interpolation=cv2.INTER_AREA)
            mean_sq = cv2.resize(squares, small, interpolation=cv2.INTER_AREA)
            variance = cv2.subtract(mean_sq, cv2.multiply(mean, mean))
            # Compared in numpy, not cv2.compare: this array is one value per
            # block, and a coarse level over a small region is a single value,
            # where OpenCV cannot tell an array from a scalar and throws.
            flat = np.where(variance <= limit, np.uint8(255), np.uint8(0))
            settle = cv2.bitwise_and(
                active, cv2.resize(flat, (width, height), interpolation=cv2.INTER_NEAREST))
        cv2.copyTo(colour, settle, out)
        if grid:
            cv2.copyTo(_solid((height, width), (step,)), settle, level_map)
        active = cv2.subtract(active, settle)

    if grid:
        # A block's edge is the grid at its own size, kept only where a block of
        # that size actually settled - which is what makes the lines follow the
        # subdivision instead of drawing one even mesh over everything.
        edges = np.zeros((height, width), np.uint8)
        for step, block in enumerate(blocks):
            cv2.bitwise_or(edges, cv2.bitwise_and(_grid_lines(block, height, width),
                                                  cv2.compare(level_map, step, cv2.CMP_EQ)),
                           dst=edges)
        cv2.copyTo(_solid(roi.shape, params.get("grid_colour", (0, 255, 0))), edges, out)
    return out


def _vhs(roi: np.ndarray, ctx: EffectContext, params: Params) -> np.ndarray:
    """Worn tape: chroma smeared sideways, bands sliding, a head crossing.

    Colour is recorded at a fraction of the luma bandwidth on tape, so it
    spreads sideways and lags behind the edges it belongs to. That artefact
    says VHS on its own - the jitter and the noise only sell it.
    """
    height, width = roi.shape[:2]

    smear = float(np.clip(params.get("smear", 0.28), 0.0, 0.95))
    previous = ctx.state.get("vhs")
    if previous is not None and smear > 0:
        # The shape changes size as the hands move, so the last field has to be
        # stretched to fit before it can be blended.
        if previous.shape != roi.shape:
            previous = cv2.resize(previous, (width, height))
        out = cv2.addWeighted(roi, 1.0 - smear, previous, smear, 0.0)
    else:
        out = roi.copy()
    ctx.state["vhs"] = out.copy()

    bleed = max(int(params.get("chroma_bleed", 18)), 0)
    lag = int(params.get("chroma_lag", 4))
    if width > 2 and (bleed > 1 or lag):
        ycc = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
        chroma = np.ascontiguousarray(ycc[:, :, 1:])
        if bleed > 1:
            chroma = cv2.blur(chroma, (min(bleed, width), 1))
        if lag:
            chroma = np.roll(chroma, lag, axis=1)
        # Blurring colour together also drains it, so the tape's oversaturated
        # look has to be put back afterwards. Weighted rather than scaled, so a
        # channel pushed past the ends clips instead of folding back.
        saturation = float(params.get("saturation", 2.0))
        if saturation != 1.0:
            chroma = cv2.addWeighted(chroma, saturation, chroma, 0.0,
                                     128.0 * (1.0 - saturation))
        ycc[:, :, 1:] = chroma
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

    bands = int(params.get("bands", 10))
    jitter = int(params.get("jitter", 5))
    if bands > 0 and jitter and height > bands:
        bounds = np.linspace(0, height, bands + 1).astype(int)
        shifts = ctx.rng.integers(-jitter, jitter + 1, size=bands)
        for index in range(bands):
            top, bottom, shift = int(bounds[index]), int(bounds[index + 1]), int(shifts[index])
            if bottom > top and shift:
                out[top:bottom] = np.roll(out[top:bottom], shift, axis=1)

    # The head-switching band, drifting down the picture frame after frame.
    band = int(params.get("band", 26))
    if band > 0 and height > band:
        speed = float(params.get("band_speed", 0.03))
        top = max(int((ctx.frame_index * speed * height) % (height + band)) - band, 0)
        bottom = min(top + band, height)
        if bottom > top:
            strip = np.roll(out[top:bottom], max(jitter, 1) * 3, axis=1)
            out[top:bottom] = cv2.convertScaleAbs(strip, alpha=1.2, beta=16)

    noise = float(params.get("noise", 0.07))
    if noise > 0:
        amplitude = max(int(noise * 255), 1)
        speckle = ctx.rng.integers(0, amplitude + 1, size=(height, width), dtype=np.uint8)
        cv2.add(out, cv2.cvtColor(speckle, cv2.COLOR_GRAY2BGR), dst=out)

    return _scanlines(out, int(params.get("scanline_gap", 3)),
                      float(params.get("scanline_strength", 0.35)))


def _ascii(roi: np.ndarray, ctx: EffectContext, params: Params) -> np.ndarray:
    """The picture retyped as characters, the fullest glyph for the lightest cell.

    Built by one gather: every cell is replaced by its glyph in a single index
    into the atlas, where drawing the characters one at a time would be tens of
    thousands of putText calls a second.
    """
    height, width = roi.shape[:2]
    cell_w = max(int(params.get("cell", 10)), 2)
    cell_h = max(int(cell_w * float(params.get("aspect", 1.6))), 2)
    cols, rows = max(width // cell_w, 1), max(height // cell_h, 1)

    small = cv2.resize(roi, (cols, rows), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    ramp = str(params.get("ramp", " .:-=+*#%@")) or " ."
    atlas = _glyph_atlas(ramp, cell_w, cell_h)

    index = gray.astype(np.uint16) * (len(ramp) - 1) // 255
    canvas = np.ascontiguousarray(
        atlas[index].transpose(0, 2, 1, 3).reshape(rows * cell_h, cols * cell_w))

    tint = params.get("tint")
    if tint:
        glyphs = cv2.applyColorMap(canvas, _tint_lut(tint))
    else:
        # Otherwise each glyph takes the colour of the cell it stands for.
        colour = cv2.resize(small, (cols * cell_w, rows * cell_h),
                            interpolation=cv2.INTER_NEAREST)
        glyphs = cv2.multiply(colour, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR),
                              scale=1.0 / 255.0)

    # The grid rarely divides the region exactly. Padding leaves at most one
    # cell of black along two edges; resizing to fit would stretch every glyph.
    out = np.zeros_like(roi)
    rendered = glyphs[:height, :width]
    out[:rendered.shape[0], :rendered.shape[1]] = rendered
    return out


def _halftone(roi: np.ndarray, ctx: EffectContext, params: Params) -> np.ndarray:
    """A printer's dot screen: one compare against a clustered threshold cell."""
    height, width = roi.shape[:2]
    dot = max(int(params.get("dot", 6)), 2)

    if params.get("colour", False):
        # One screen per channel, each at its own phase - the separations of a
        # colour print are screened at different angles for the same reason.
        phases = ((0, 0), (dot // 2, dot // 3), (dot // 3, dot // 2))
        channels = [cv2.compare(plane, _screen("clustered", dot, height, width, phase), cv2.CMP_GT)
                    for plane, phase in zip(cv2.split(roi), phases)]
        return cv2.merge(channels)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Positive: the brighter the picture, the bigger the dot. Inverted, it is
    # ink on paper - dark subject, heavy ink.
    density = gray if params.get("positive", True) else cv2.bitwise_not(gray)
    dots = cv2.compare(density, _screen("clustered", dot, height, width), cv2.CMP_GT)
    return cv2.applyColorMap(dots, _ramp_lut((tuple(params.get("paper", (0, 0, 0))),
                                              tuple(params.get("ink", (0, 255, 0))))))


def _thermal(roi: np.ndarray, ctx: EffectContext, params: Params) -> np.ndarray:
    name = str(params.get("colormap", "JET")).upper()
    colormap = getattr(cv2, f"COLORMAP_{name}", cv2.COLORMAP_JET)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, colormap)


def _dither(roi: np.ndarray, ctx: EffectContext, params: Params) -> np.ndarray:
    """Ordered dither against a Bayer matrix - few tones, scattered pixels.

    The same compare halftone makes, against a screen whose low thresholds are
    spread across the cell instead of clustered at its middle. That one
    difference turns dots into the pixel spray of a 1-bit display.
    """
    height, width = roi.shape[:2]
    levels = max(int(params.get("levels", 2)), 2)
    screen = _screen("bayer", max(int(params.get("matrix", 8)), 2), height, width)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ink = _ramp_lut((tuple(params.get("paper", (0, 0, 0))), tuple(params.get("ink", (0, 255, 0)))))

    if levels == 2:
        return cv2.applyColorMap(cv2.compare(gray, screen, cv2.CMP_GT), ink)

    # More than two tones: offset each pixel by its own threshold, then snap to
    # the nearest level. The offset is what scatters the banding into a pattern.
    step = 255.0 / (levels - 1)
    spread = np.clip(gray.astype(np.int16)
                     + ((screen.astype(np.int16) - 127) * step / 255.0).astype(np.int16),
                     0, 255).astype(np.uint8)
    return cv2.applyColorMap(cv2.LUT(spread, _quantise_lut(levels)), ink)


REGISTRY: dict[str, Callable[[np.ndarray, EffectContext, Params], np.ndarray]] = {
    "risograph": _risograph,
    "cyanotype": _cyanotype,
    "quadtree": _quadtree,
    "vhs": _vhs,
    "ascii": _ascii,
    "halftone": _halftone,
    "thermal": _thermal,
    "dither": _dither,
}


def names() -> list[str]:
    return list(REGISTRY)


def apply(name: str, roi: np.ndarray, ctx: EffectContext, params: Params | None = None) -> np.ndarray:
    """Run one effect over a cropped region, returning a new image."""
    effect = REGISTRY.get(name)
    if effect is None or roi.size == 0:
        return roi
    return effect(roi, ctx, params or {})


# The whole highlight curve - black point, brightness, gamma, threshold and the
# soft knee - collapses into one 256-entry table, because every one of those is
# a function of a single pixel's brightness and nothing else.
_BLOOM_LUTS: dict[tuple, np.ndarray] = {}


def _bloom_lut(black, brightness, gamma, threshold, soft_knee) -> np.ndarray:
    key = (round(black, 4), round(brightness, 4), round(gamma, 4),
           round(threshold, 4), bool(soft_knee))
    lut = _BLOOM_LUTS.get(key)
    if lut is None:
        if len(_BLOOM_LUTS) > 32:
            _BLOOM_LUTS.clear()
        value = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        black = float(np.clip(black, 0.0, 0.95))
        if black > 0:
            value = np.clip((value - black) / (1.0 - black), 0.0, 1.0)
        if brightness != 1.0:
            value = np.clip(value * brightness, 0.0, 1.0)
        if gamma != 1.0:
            value = np.power(value, max(gamma, 1e-3))

        threshold = float(np.clip(threshold, 0.0, 0.99))
        keep = np.clip((value - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
        if soft_knee:
            # Smoothstep, so highlights ease into blooming rather than
            # switching on at a hard edge that crawls as the picture moves.
            keep = keep * keep * (3.0 - 2.0 * keep)
        lut = _BLOOM_LUTS[key] = np.clip(keep * 255.0, 0, 255).astype(np.uint8)
    return lut


def bloom_pass(frame: np.ndarray, cfg) -> bool:
    """Bleed light out of the bright parts of a finished frame.

    The same shape as TouchDesigner's Bloom TOP: condition the image, pick out
    what is brighter than a threshold, blur that at a spread of radii, and add
    it back. Modifies ``frame``.

    Blurring at several radii and averaging is what separates it from a plain
    glow - a single radius gives one flat smear, where a tight blur inside a
    wide one reads as a hot core with a halo around it.

    Built at half size, because bloom is blur: there is no detail in it to lose
    and full resolution costs several times as much.
    """
    if not cfg.enabled or cfg.intensity <= 0:
        return False

    height, width = frame.shape[:2]
    small_w = max(int(width * cfg.scale), 2)
    small_h = max(int(height * cfg.scale), 2)
    if small_w < 4 or small_h < 4:
        return False

    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Perceptual luminance rather than a channel maximum, so a saturated red
    # does not bloom as hard as a white of the same value. One table lookup
    # then does the entire highlight curve.
    luma = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    keep = cv2.LUT(luma, _bloom_lut(cfg.pre_black, cfg.pre_brightness,
                                    cfg.pre_gamma, cfg.threshold, cfg.soft_knee))
    bright = cv2.multiply(small, cv2.cvtColor(keep, cv2.COLOR_GRAY2BGR),
                          scale=1.0 / 255.0)

    steps = max(int(cfg.steps), 1)
    radii = (np.linspace(cfg.min_radius, cfg.max_radius, steps)
             if steps > 1 else [cfg.max_radius])
    share = 1.0 / len(radii)
    glow = np.zeros_like(bright)
    for radius in radii:
        glow = cv2.addWeighted(glow, 1.0, soft_blur(bright, float(radius)), share, 0.0)

    # Scaled while it is still small, then enlarged. The other way round put
    # the multiply and the clamp on four times as many pixels.
    glow = cv2.convertScaleAbs(glow, alpha=float(cfg.intensity))
    glow = cv2.resize(glow, (width, height), interpolation=cv2.INTER_LINEAR)
    cv2.add(frame, glow, dst=frame)
    return True


def apply_in_polygon(
    frame: np.ndarray,
    polygon: np.ndarray,
    name: str,
    ctx: EffectContext,
    params: Params | None = None,
) -> bool:
    """Run an effect on ``frame``, confined to ``polygon``. Modifies the frame.

    Shared by the fingertip shape and the 3D cube so there is one masking path
    rather than two that can drift apart. Returns False when the polygon lies
    entirely off-screen.

    Work is confined to the polygon's bounding box, so the cost scales with the
    shape rather than the frame.
    """
    height, width = frame.shape[:2]
    polygon = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
    if len(polygon) < 3:
        return False

    x, y, box_w, box_h = cv2.boundingRect(polygon)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + box_w, width), min(y + box_h, height)
    if x1 <= x0 or y1 <= y0:
        return False

    roi = frame[y0:y1, x0:x1]
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    # fillPoly, not fillConvexPoly. The fingertip quad goes concave when a
    # finger bends in, and crosses itself entirely once twisted past the pinch;
    # fillPoly's even-odd rule renders both, filling a crossed shape as the two
    # lobes of an X. The convex version fills the whole hull instead.
    cv2.fillPoly(mask, [polygon - np.array([x0, y0], dtype=np.int32)], 255)

    treated = apply(name, roi, ctx, params)
    if treated.shape == roi.shape:
        # roi is a view, so this writes straight into the frame. cv2.copyTo is
        # about a hundred times faster here than a boolean numpy assignment.
        cv2.copyTo(treated, mask, roi)
    return True
