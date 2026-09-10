"""A very small software 3D renderer.

Enough to float a wireframe cube between your hands and grow a rose inside it,
built on numpy and cv2 alone - the app already owns its draw loop, so this is a
few matrix multiplies and some polygon fills rather than a 3D engine.

The camera looks down +Z. Meshes are modelled in a roughly unit-sized space and
scaled to pixels at projection time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# Nothing is allowed closer than this to the camera. A vertex swinging behind
# the eye would otherwise divide by zero and fling its projection to infinity.
NEAR = 1e-3


@dataclass
class Mesh:
    vertices: np.ndarray                      # (N, 3) float32
    edges: np.ndarray | None = None           # (M, 2) int32
    # Polylines through the vertices, for the glow renderer. A closed loop is
    # stored with its first index repeated at the end, so everything can be
    # stroked with one call and no per-curve closed/open bookkeeping.
    curves: list[np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float32)
        if self.edges is not None:
            self.edges = np.asarray(self.edges, dtype=np.int32)


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Rotation matrix, applied roll then pitch then yaw."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    return (ry @ rx @ rz).astype(np.float32)


def project(
    vertices: np.ndarray,
    rot: np.ndarray,
    centre,
    size: float,
    perspective: float = 2.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate, scale to pixels, and project.

    ``perspective`` is the focal length as a multiple of ``size``: larger
    flattens the foreshortening, smaller exaggerates it. The object sits one
    focal length away, so a vertex on the z=0 plane lands at its true size.

    Returns the 2D points and the camera-space points (needed for depth sorting
    and face normals).
    """
    scaled = (np.asarray(vertices, dtype=np.float32) @ rot.T) * float(size)
    focal = max(float(perspective) * float(size), 1.0)

    depth = np.maximum(scaled[:, 2] + focal, NEAR)
    factor = focal / depth
    points = np.stack(
        [centre[0] + scaled[:, 0] * factor, centre[1] + scaled[:, 1] * factor], axis=1
    )
    return points.astype(np.float32), scaled.astype(np.float32)


def silhouette(points: np.ndarray) -> np.ndarray:
    """Outline of a projected mesh - the polygon to mask an effect to."""
    return cv2.convexHull(np.rint(points).astype(np.int32))


def draw_wireframe(
    frame: np.ndarray,
    points: np.ndarray,
    edges: np.ndarray,
    colour,
    thickness: int = 2,
    glow: float = 0.0,
) -> None:
    colour = tuple(int(c) for c in colour)
    lines = np.rint(points).astype(np.int32)

    if glow > 0:
        # Same trick as the shape outline: blur a copy of the lines and add it
        # back, confined to their bounding box.
        from . import effects

        x, y, w, h = cv2.boundingRect(lines)
        pad = 16
        fh, fw = frame.shape[:2]
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, fw), min(y + h + pad, fh)
        if x1 > x0 and y1 > y0:
            region = frame[y0:y1, x0:x1]
            halo = np.zeros_like(region)
            offset = np.array([x0, y0], dtype=np.int32)
            for a, b in edges:
                cv2.line(halo, tuple(lines[a] - offset), tuple(lines[b] - offset),
                         colour, max(thickness * 3, 3), cv2.LINE_AA)
            halo = effects.soft_blur(halo, 8.0)
            cv2.addWeighted(region, 1.0, halo, glow, 0.0, dst=region)

    for a, b in edges:
        cv2.line(frame, tuple(lines[a]), tuple(lines[b]), colour,
                 max(thickness, 1), cv2.LINE_AA)


# Intensity-to-colour ramps for the glow, built once per colour pair. The same
# lookup-table trick the print effects use: colourising through a LUT keeps
# the whole path in uint8, where float colour maths costs many times more.
_GLOW_LUTS: dict[tuple, np.ndarray] = {}


def glow_lut(core, edge) -> np.ndarray:
    """Black to ``core`` to ``edge`` - the palette a stroke burns through.

    It must start at black: the glow is added to the frame, so zero intensity
    has to contribute nothing at all.
    """
    key = (tuple(int(c) for c in core), tuple(int(c) for c in edge))
    lut = _GLOW_LUTS.get(key)
    if lut is None:
        core_c = np.asarray(key[0], dtype=np.float32)
        edge_c = np.asarray(key[1], dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None]
        low = ramp / 0.55 * core_c
        high = core_c + (ramp - 0.55) / 0.45 * (edge_c - core_c)
        blended = np.where(ramp < 0.55, low, high)
        lut = _GLOW_LUTS[key] = np.clip(blended, 0, 255).astype(np.uint8).reshape(256, 1, 3)
    return lut


def draw_points(
    frame: np.ndarray,
    points: np.ndarray,
    depth: np.ndarray,
    lut: np.ndarray,
    size: int = 1,
    supersample: int = 2,
    bloom: float = 1.6,
    strength: float = 1.0,
    gain: float = 165.0,
) -> bool:
    """Splat a point cloud as glowing particles and add it to the frame.

    Points are accumulated with ``np.bincount`` over flattened pixel indices -
    one vectorised pass for the whole cloud. Drawing tens of thousands of
    points one ``cv2.circle`` at a time would be hopeless in Python, and
    ``np.add.at`` is far slower than bincount for the same job.

    Accumulating rather than overwriting is the point: where the model's
    surface is dense the splats stack and burn brighter, so the form shows
    through the cloud without any surface being drawn.
    """
    if len(points) == 0:
        return False
    if not np.all(np.isfinite(points)):
        return False

    height, width = frame.shape[:2]
    pad = 8 + int(bloom * 6)
    x, y, box_w, box_h = cv2.boundingRect(np.rint(points).astype(np.int32))
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + box_w + pad, width), min(y + box_h + pad, height)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False

    scale = max(int(supersample), 1)
    region_w, region_h = x1 - x0, y1 - y0
    buf_w, buf_h = region_w * scale, region_h * scale

    xs = np.rint((points[:, 0] - x0) * scale).astype(np.int64)
    ys = np.rint((points[:, 1] - y0) * scale).astype(np.int64)
    inside = (xs >= 0) & (xs < buf_w) & (ys >= 0) & (ys < buf_h)
    if not inside.any():
        return False

    depth = np.asarray(depth, dtype=np.float32)
    near, far = float(depth.min()), float(depth.max())
    nearness = (far - depth) / max(far - near, 1e-6)
    weight = (0.22 + 0.78 * nearness ** 1.5).astype(np.float32)

    flat = ys[inside] * buf_w + xs[inside]
    canvas = np.bincount(flat, weights=weight[inside], minlength=buf_w * buf_h)
    # bincount hands back float64, and scaling it in numpy would walk that
    # buffer three times over - a multiply, a clip and a cast, each allocating
    # megabytes. convertScaleAbs does all three in one pass, and every point of
    # this renderer's cost past the splat itself is in walking this buffer.
    canvas = cv2.convertScaleAbs(canvas.reshape(buf_h, buf_w), alpha=gain)

    if size > 1:
        canvas = cv2.dilate(canvas, np.ones((size, size), np.uint8))

    small = cv2.resize(canvas, (region_w, region_h), interpolation=cv2.INTER_AREA)
    if bloom > 0:
        from . import effects

        small = cv2.addWeighted(small, 1.0, effects.soft_blur(small, bloom * 4.0), 0.75, 0.0)

    coloured = cv2.applyColorMap(small, lut)
    if strength != 1.0:
        coloured = cv2.convertScaleAbs(coloured, alpha=float(np.clip(strength, 0.0, 4.0)))

    region = frame[y0:y1, x0:x1]
    cv2.add(region, coloured, dst=region)
    return True


def draw_glow(
    frame: np.ndarray,
    curves: list[np.ndarray],
    depths: np.ndarray,
    lut: np.ndarray,
    thickness: int = 1,
    supersample: int = 2,
    bloom: float = 1.7,
    strength: float = 1.0,
    bands: int = 5,
) -> bool:
    """Stroke curves as glowing light and add them to the frame.

    Nothing is filled. Filled flat-shaded polygons are what made the earlier
    rose look chopped up: one colour per face gives hard facet steps and seams
    that no amount of tuning removes. Strokes have no interior to facet.

    Curves are stroked into a small supersampled buffer and scaled back down,
    which is where the smooth edges come from, then bloomed and pushed through
    a colour ramp. They are grouped into depth bands and drawn far-to-near with
    rising brightness - lighting front and back petals identically is most of
    why a flower reads as noise rather than as a solid object.
    """
    if not curves:
        return False

    height, width = frame.shape[:2]
    stacked = np.concatenate(curves, axis=0)
    if not np.all(np.isfinite(stacked)):
        return False

    pad = 10 + int(bloom * 6)
    x, y, box_w, box_h = cv2.boundingRect(np.rint(stacked).astype(np.int32))
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + box_w + pad, width), min(y + box_h + pad, height)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False

    scale = max(int(supersample), 1)
    region_w, region_h = x1 - x0, y1 - y0
    origin = np.array([x0, y0], dtype=np.float32)
    canvas = np.zeros((region_h * scale, region_w * scale), dtype=np.uint8)

    depths = np.asarray(depths, dtype=np.float32)
    near, far = float(depths.min()), float(depths.max())
    span = far - near
    if span < 1e-3:
        # A flat object square to the camera - a ring, a mesh seen edge-on -
        # has no depth to sort by, and dividing by the epsilon below put every
        # curve in the *back* band, drawing the whole thing at the dimmest
        # level. Worse, float noise around zero scattered the curves across
        # bands at random, so a single closed curve came out as broken arcs.
        # Nothing is behind anything here, so everything is at the front.
        band_of = np.full(len(depths), bands - 1, dtype=int)
    else:
        # Far to near, so nearer strokes land on top of dimmer ones.
        band_of = np.clip(((far - depths) / span * bands).astype(int), 0, bands - 1)

    for band in range(bands):
        members = [c for c, b in zip(curves, band_of) if b == band]
        if not members:
            continue
        polys = [np.rint((c - origin) * scale).astype(np.int32) for c in members]
        # A steep ramp, not a gentle one. Without real occlusion every petal's
        # whole outline shows through every other, and the flower turns into a
        # tangle of overlapping loops; dropping the back ranks to near-nothing
        # is what stands in for petals hiding behind each other.
        nearness = band / max(bands - 1, 1)
        level = int(22 + 200 * nearness ** 1.6)
        # Nearer strokes are drawn heavier as well as brighter. Weight is what
        # separates foreground from background in line art; brightness alone
        # leaves every petal reading at the same distance.
        weight = max(int(round(thickness * scale * (0.7 + 0.8 * nearness))), 1)
        layer = np.zeros_like(canvas)
        cv2.polylines(layer, polys, False, level, weight, cv2.LINE_AA)
        cv2.max(canvas, layer, dst=canvas)

    small = cv2.resize(canvas, (region_w, region_h), interpolation=cv2.INTER_AREA)
    if bloom > 0:
        from . import effects

        # Restrained: the strokes already overlap densely at the heart, and a
        # heavy bloom on top saturates the middle into a featureless blob.
        small = cv2.addWeighted(small, 1.0, effects.soft_blur(small, bloom * 4.0), 0.40, 0.0)

    coloured = cv2.applyColorMap(small, lut)
    if strength != 1.0:
        coloured = cv2.convertScaleAbs(coloured, alpha=float(np.clip(strength, 0.0, 4.0)))

    region = frame[y0:y1, x0:x1]
    cv2.add(region, coloured, dst=region)
    return True


# --------------------------------------------------------------------------
# meshes
# --------------------------------------------------------------------------

_CUBE_VERTS = np.array([
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
], dtype=np.float32)

_CUBE_EDGES = np.array([
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
], dtype=np.int32)

def make_cube() -> Mesh:
    """The box. Wireframe only - its faces are never filled."""
    return Mesh(vertices=_CUBE_VERTS.copy(), edges=_CUBE_EDGES.copy())


GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def _petal(rows: int, cols: int, cup: float, recurve: float) -> np.ndarray:
    """One petal as an (rows*cols, 3) grid, pointing +Y, cupping in +Z.

    The width profile widens fast off the base and stays broad almost to the
    top before rounding off. A simple ``sin(pi*u)`` taper gives a pointed leaf
    instead, and a flower built from those reads as a spiky burr rather than a
    rose.
    """
    u = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]   # along
    v = np.linspace(-1.0, 1.0, cols, dtype=np.float32)[None, :]  # across

    width = 0.72 * np.sqrt(np.maximum(1.0 - (1.0 - u) ** 2, 0.0)) \
        * np.power(np.maximum(1.0 - u ** 6, 0.0), 0.40)

    x = v * width
    y = np.repeat(u, cols, axis=1)
    # Cupped across its width, deepening up the petal, with the tip curling
    # back on itself the way an open rose petal does.
    z = cup * (v ** 2) * (0.20 + 0.80 * u) - recurve * (u ** 3)

    return np.stack([x, y, np.broadcast_to(z, x.shape)], axis=-1).reshape(-1, 3)


def _grid_outline(rows: int, cols: int, offset: int) -> np.ndarray:
    """The boundary of a petal grid, as one closed loop."""
    def at(r: int, c: int) -> int:
        return offset + r * cols + c

    loop = [at(0, c) for c in range(cols)]
    loop += [at(r, cols - 1) for r in range(1, rows)]
    loop += [at(rows - 1, c) for c in range(cols - 2, -1, -1)]
    loop += [at(r, 0) for r in range(rows - 2, 0, -1)]
    loop.append(loop[0])
    return np.asarray(loop, dtype=np.int32)


def _grid_ribs(rows: int, cols: int, offset: int, count: int = 2) -> list[np.ndarray]:
    """Lines up the length of a petal.

    An outline on its own reads as an empty hoop; the ribs are what make a
    stroked petal look like a curved surface.
    """
    if cols < 3 or count < 1:
        return []
    columns = np.unique(np.linspace(1, cols - 2, count).astype(int))
    return [
        np.asarray([offset + r * cols + int(c) for r in range(rows)], dtype=np.int32)
        for c in columns
    ]


# The extent a fully open rose reaches, measured once. Every bloom value is
# scaled by this same number, so the flower grows as it opens. Normalising each
# mesh by its own extent instead - the obvious thing - makes a tight bud fill
# the frame and a full bloom shrink, which is backwards.
_ROSE_REACH: dict[tuple, float] = {}


# Petals per whorl, innermost first. Real roses are layered in rings, and a
# single evenly-spread spiral - which is what this used to be - scatters petals
# into a noisy silhouette that does not read as a rose at a glance.
WHORLS = (3, 5, 8)


def _grow_rose(bloom, whorls, rows, cols):
    per_petal = rows * cols
    chunks, curves = [], []
    rings = len(whorls)
    petal_index = 0

    for ring, count in enumerate(whorls):
        base_t = ring / max(rings - 1, 1)
        for j in range(count):
            # A touch of variation inside a ring, so the layers do not look
            # stamped, without breaking the layered structure.
            t = float(np.clip(base_t + 0.05 * math.sin(j * 2.4), 0.0, 1.0))
            phi = j * (2 * math.pi / count) + ring * GOLDEN_ANGLE

            # Outer whorls carry much larger petals, sitting further out and
            # falling right back to make the flat rim of the rosette. Narrow
            # these spreads and the rings pile up into a shapeless mass.
            scale = 0.26 + 0.90 * t
            # Closed, the petals furl inward over the heart. The angle is
            # bounded: measured across a sweep, past about -26 degrees they
            # fold so far they splay out the other side, and the "tight" bud
            # comes out wider than a half-open flower.
            lean = (math.radians(-26) * (1.0 - bloom)
                    + (0.05 + 0.95 * t) * bloom * math.radians(85))
            radius = (0.02 + 0.52 * t) * (0.35 + 0.65 * bloom)
            drop = -0.12 * t * bloom

            # Petals cup much harder while closed, which is what stops a bud
            # looking spiky - curled around the axis they hide their tips.
            cup = (0.70 - 0.44 * t) * (1.0 + 0.8 * (1.0 - bloom))
            petal = _petal(rows, cols, cup=cup, recurve=0.10 + 0.17 * t) * scale
            petal = petal @ rotation(0.0, 0.0, 0.26 * t).T
            petal = petal @ rotation(0.0, lean, 0.0).T
            petal = petal + np.array([0.0, drop, radius], dtype=np.float32)
            petal = petal @ rotation(phi, 0.0, 0.0).T

            # Outlines only. Ribs on every petal tripled the stroke count and
            # turned the flower into a ball of wire - a line drawing of a rose
            # is its petal edges and nothing else.
            offset = petal_index * per_petal
            curves.append(_grid_outline(rows, cols, offset))
            chunks.append(petal)
            petal_index += 1

    vertices = np.concatenate(chunks, axis=0)
    # Modelled opening up the +Y axis; tip it over so you look into the flower.
    vertices = vertices @ rotation(0.0, -math.pi / 2, 0.0).T
    return vertices, curves


def make_rose(
    bloom: float = 1.0,
    whorls=WHORLS,
    rows: int = 10,
    cols: int = 5,
) -> Mesh:
    """A rose grown from parametric petals, as curves to be stroked as light.

    Petals sit in concentric whorls, each ring larger, further out and leaning
    further back, offset by the golden angle so the rings do not line up.
    ``bloom`` from 0 to 1 opens the lean, taking the flower from tight bud to
    fully open.

    Returns curves rather than faces: the glow renderer strokes them, and
    nothing is filled.
    """
    bloom = float(np.clip(bloom, 0.0, 1.0))
    whorls = tuple(int(n) for n in whorls)

    key = (whorls, rows, cols)
    if key not in _ROSE_REACH:
        open_verts, _ = _grow_rose(1.0, whorls, rows, cols)
        _ROSE_REACH[key] = float(np.abs(open_verts).max()) or 1.0

    vertices, curves = _grow_rose(bloom, whorls, rows, cols)
    return Mesh(vertices=vertices / _ROSE_REACH[key], curves=curves)


# --------------------------------------------------------------------------
# the orb
# --------------------------------------------------------------------------
_ORB_CLOUDS: dict[tuple, np.ndarray] = {}


def sphere_cloud(count: int = 5200, shell: float = 0.30,
                 core_bias: float = 1.1, seed: int = 17) -> np.ndarray:
    """Points filling a unit sphere, weighted to the middle, with a clean rim.

    The point renderer accumulates, so brightness is density and the radial
    distribution *is* the look. Two parts:

    - a volume fill at ``r = u ** core_bias``. At 1/3 that is uniform through
      the volume; anything above pulls mass into the centre, which is the hot
      core burning inside the ball.
    - a shell at r = 1 for the rim, on a golden-angle lattice rather than random
      directions. The silhouette is the one place clumping would show.

    Cached: it is a function of its arguments and never of the picture.
    """
    key = (int(count), round(float(shell), 3), round(float(core_bias), 3), int(seed))
    cloud = _ORB_CLOUDS.get(key)
    if cloud is None:
        if len(_ORB_CLOUDS) > 8:
            _ORB_CLOUDS.clear()
        count = max(int(count), 8)
        rim = int(np.clip(shell, 0.0, 0.95) * count)
        core = max(count - rim, 1)

        rng = np.random.default_rng(seed)
        # Gaussian directions normalise to a uniform spread over the sphere,
        # where sampling angles directly bunches points at the poles.
        direction = rng.normal(size=(core, 3)).astype(np.float32)
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
        radius = np.power(rng.random(core, dtype=np.float32),
                          max(float(core_bias), 1e-3))[:, None]
        points = direction * radius

        if rim > 0:
            index = np.arange(rim, dtype=np.float32) + 0.5
            z = 1.0 - 2.0 * index / rim
            ring = np.sqrt(np.maximum(1.0 - z * z, 0.0))
            angle = GOLDEN_ANGLE * index
            points = np.concatenate([
                points,
                np.stack([ring * np.cos(angle), ring * np.sin(angle), z], axis=1),
            ]).astype(np.float32)

        # Shuffled, so a prefix of the cloud is still a whole sphere - the orb
        # draws more or fewer points with its size, and taking them in order
        # would hand back a ball of core with no rim.
        rng.shuffle(points)
        cloud = _ORB_CLOUDS[key] = np.ascontiguousarray(points, dtype=np.float32)
    return cloud


_RING_ARCS: dict[tuple, list[np.ndarray]] = {}


def ring_arcs(rings: int = 3, radius: float = 1.45, arcs: int = 8,
              segments: int = 96) -> list[np.ndarray]:
    """Great circles about a unit sphere, cut into arcs for depth shading.

    The glow renderer takes one depth per curve, so a whole ring handed over in
    one piece shades flat at its mean depth. Cut into arcs, its near side burns
    and its far side drops away, and the ring reads as passing behind the ball -
    which is the only occlusion an additive pipeline can have.

    Rings are spaced by the golden angle, the same trick that keeps the rose's
    whorls from lining up.
    """
    key = (int(rings), round(float(radius), 4), int(arcs), int(segments))
    pieces = _RING_ARCS.get(key)
    if pieces is None:
        if len(_RING_ARCS) > 8:
            _RING_ARCS.clear()
        pieces = []
        arcs = max(int(arcs), 1)
        step = max(int(segments) // arcs, 2)
        for index in range(max(int(rings), 0)):
            turn = np.linspace(0.0, 2 * np.pi, step * arcs + 1, dtype=np.float32)
            circle = np.stack([np.cos(turn), np.sin(turn), np.zeros_like(turn)],
                              axis=1) * float(radius)
            tilt = rotation(GOLDEN_ANGLE * index, np.pi * index / max(int(rings), 1), 0.0)
            circle = circle @ tilt.T
            # Arcs overlap by a point so the ring has no gaps at the seams.
            pieces.extend(circle[part * step:(part + 1) * step + 1]
                          for part in range(arcs))
        _RING_ARCS[key] = pieces
    return pieces
