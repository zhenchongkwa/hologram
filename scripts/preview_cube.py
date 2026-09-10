"""Render every mode to a contact sheet, with no camera.

Lets the look be checked and tuned from a file instead of by standing in front
of the webcam - the loop the rose, the effects and the orb were all tuned in.

    python scripts/preview_cube.py
    python scripts/preview_cube.py --petals 30 --bloom 0.6
    python scripts/preview_cube.py --modes-only
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hologram import scene3d  # noqa: E402

TILE = (360, 360)
EDGE = (0, 255, 0)

# A plausible open hand, in multiples of hand scale. The solo-palm pose is only
# offered for a hand that reads as open, so a ring of dummy points will not do -
# it has to have real finger geometry.
OPEN_HAND = np.array([
    (0.00, 0.00), (-0.30, -0.20), (-0.50, -0.40), (-0.65, -0.55), (-0.80, -0.70),
    (-0.25, -0.90), (-0.28, -1.25), (-0.30, -1.45), (-0.30, -1.65),
    (0.00, -1.00), (0.00, -1.35), (0.00, -1.60), (0.00, -1.80),
    (0.25, -0.95), (0.28, -1.30), (0.30, -1.50), (0.30, -1.70),
    (0.50, -0.80), (0.55, -1.05), (0.60, -1.20), (0.60, -1.40),
], dtype=np.float32)


# Curling the ring and pinky tips back leaves index and middle out - the two
# fingers the orb gesture looks for.
CURLED_TIPS = {16: (0.26, -0.95), 20: (0.50, -0.80)}


def hand_at(centre, scale=52.0, two_fingers=False):
    points = OPEN_HAND.copy()
    if two_fingers:
        for landmark, position in CURLED_TIPS.items():
            points[landmark] = position
    return points * scale + np.asarray(centre, dtype=np.float32)


def backdrop() -> np.ndarray:
    """A soft gradient, so shading and glow have something to sit on."""
    height, width = TILE
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    tile = np.stack([
        40 + 40 * (ys / height),
        30 + 25 * (xs / width),
        28 + 20 * (1 - ys / height),
    ], axis=-1)
    return np.clip(tile, 0, 255).astype(np.uint8)


def scene(size):
    """Something with structure, so a masked effect has an edge to bite on."""
    height, width = size
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    tile = np.stack([
        120 + 70 * np.sin(xs / 70.0),
        105 + 60 * np.cos(ys / 55.0),
        135 + 55 * np.sin((xs + ys) / 90.0),
    ], axis=-1)
    tile = np.clip(tile, 0, 255).astype(np.uint8)
    cv2.circle(tile, (width // 2, height // 2 - 20), 62, (214, 205, 196), -1, cv2.LINE_AA)
    for offset in range(0, width, 30):
        cv2.line(tile, (offset, 0), (offset - 60, height), (232, 232, 232), 1, cv2.LINE_AA)
    return tile


def draw_rose(tile, rose, rot, centre, size, lut, strength=1.0):
    points, cam = scene3d.project(rose.vertices, rot, centre, size)
    curves = [points[i] for i in rose.curves]
    depths = np.array([float(cam[i, 2].mean()) for i in rose.curves], dtype=np.float32)
    scene3d.draw_glow(tile, curves, depths, lut, strength=strength)
    return len(curves)


def label(tile, text):
    cv2.putText(tile, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(tile, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, EDGE, 1, cv2.LINE_AA)



def hands_at(*spec):
    """Synthetic hands from (x fraction, y fraction, scale) triples."""
    from hologram.tracking import Hand

    return [Hand(points=hand_at((TILE[1] * x, TILE[0] * y), scale),
                 label="Left" if index == 0 else "Right", score=1.0)
            for index, (x, y, scale) in enumerate(spec)]


def mode_rows():
    """A row for the slit-scan, which the rows above cannot show.

    The cube's own rows are whole simulated frames already; this one needs
    something they do not, which is time - the effect does not exist in a
    single frame.
    """
    from hologram import config as config_module, modes as modes_module
    from hologram.effects import EffectContext

    cfg = config_module.load(ROOT / "config.toml")
    tiles = []

    # Slit-scan needs time, so this row is one run sampled as it fills: a white
    # disc crossing the frame with a hand tracking it.
    mode = modes_module.SlitScanMode(cfg)
    ctx = EffectContext()
    wanted = {0: "first frame", 8: "filling", 20: "most of a second", 44: "settled"}
    for step in range(45):
        tile = scene(TILE)
        across = int(TILE[1] * (0.12 + 0.76 * step / 44.0))
        cv2.circle(tile, (across, int(TILE[0] * 0.42)), 40, (250, 250, 250), -1, cv2.LINE_AA)
        hands = [h for h in hands_at((across / TILE[1], 0.80, 44))]
        mode.update(hands, step / 30.0)
        mode.render(tile, hands, "halftone", ctx, {})
        if step in wanted:
            label(tile, f"slit {wanted[step]}: {mode.readout()}")
            tiles.append(tile)

    return tiles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", type=Path, default=ROOT / "captures" / "cube.png")
    parser.add_argument("--petals", type=int, default=24)
    parser.add_argument("--bloom", type=float, default=None,
                        help="fix the bloom instead of sweeping it")
    parser.add_argument("--modes-only", action="store_true",
                        help="just the one-row-per-mode sheet, which is quicker")
    args = parser.parse_args()

    centre = (TILE[1] / 2, TILE[0] / 2)
    size = TILE[0] * 0.22
    cube = scene3d.make_cube()
    tiles = []

    if args.modes_only:
        tiles.extend(mode_rows())
        return write(tiles, args.out)

    # Row 1 - the cube turning.
    for index, (yaw, pitch) in enumerate(
        [(0.0, 0.0), (0.6, 0.3), (1.1, -0.35), (2.4, 0.55)]
    ):
        tile = backdrop()
        rot = scene3d.rotation(yaw, pitch, 0.0)
        points, _ = scene3d.project(cube.vertices, rot, centre, size)
        scene3d.draw_wireframe(tile, points, cube.edges, EDGE, 2, glow=0.5)
        label(tile, f"cube yaw {math.degrees(yaw):.0f} pitch {math.degrees(pitch):.0f}")
        tiles.append(tile)

    # Row 2 - the rose opening.
    from hologram import config as cfg_module

    box_cfg = cfg_module.load(ROOT / "config.toml").box
    lut = scene3d.glow_lut(box_cfg.rose_core, box_cfg.rose_edge)
    cloud_path = ROOT / "models" / "rose_points.npy"
    cloud = np.load(cloud_path) if cloud_path.exists() else None
    if cloud is not None:
        cloud = cloud[np.random.default_rng(11).permutation(len(cloud))]

    blooms = [args.bloom] * 4 if args.bloom is not None else [0.0, 0.35, 0.7, 1.0]
    for bloom in blooms:
        tile = backdrop()
        rot = scene3d.rotation(0.18, 0.12, 0.0)
        if cloud is not None:
            shown = max(int(len(cloud) * (0.12 + 0.88 * bloom)), 1)
            pts, cam = scene3d.project(cloud[:shown], rot, centre, size * 1.35)
            scene3d.draw_points(tile, pts, cam[:, 2], lut, size=2)
            label(tile, f"cloud bloom {bloom:.2f}  {shown} pts")
        else:
            drawn = draw_rose(tile, scene3d.make_rose(bloom=bloom), rot,
                              centre, size * 1.35, lut)
            label(tile, f"rose bloom {bloom:.2f}  {drawn} curves")
        tiles.append(tile)

    # Row 3 - rose inside the cube, plus the silhouette the effect masks to.
    for index, yaw in enumerate([0.0, 0.7, 1.4, 2.2]):
        tile = backdrop()
        rot = scene3d.rotation(yaw, 0.3, 0.0)
        cube_pts, _ = scene3d.project(cube.vertices, rot, centre, size)

        if index == 3:
            hull = scene3d.silhouette(cube_pts)
            shaded = tile.copy()
            cv2.fillPoly(shaded, [hull], (90, 70, 60))
            cv2.addWeighted(tile, 0.45, shaded, 0.55, 0, dst=tile)

        if cloud is not None:
            pts, cam = scene3d.project(cloud, rot, centre, size * 0.95)
            scene3d.draw_points(tile, pts, cam[:, 2], lut, size=2)
        else:
            draw_rose(tile, scene3d.make_rose(bloom=1.0), rot, centre, size * 0.95, lut)
        scene3d.draw_wireframe(tile, cube_pts, cube.edges, EDGE, 2, glow=0.5)
        label(tile, "silhouette mask" if index == 3 else f"rose in cube {index + 1}")
        tiles.append(tile)

    # Row 4 - a whole cube-mode frame through the real BoxView path: effect
    # masked to the silhouette, fingertip rays, rose, wireframe.
    from hologram import config as config_module
    from hologram.box import BoxView
    from hologram.effects import EffectContext
    from hologram.tracking import Hand

    cfg = config_module.load(ROOT / "config.toml")
    left = hand_at((TILE[1] * 0.20, TILE[0] * 0.74))
    right = hand_at((TILE[1] * 0.80, TILE[0] * 0.74))
    hands = [Hand(points=left, label="Left", score=1.0),
             Hand(points=right, label="Right", score=1.0)]

    for effect, bloom in (("halftone", 0.0), ("halftone", 1.0)):
        tile = scene(TILE)
        box = BoxView(cfg.box, cfg.tracking)
        pose = None
        for step in range(12):
            pose = box.update(hands, step / 30.0)
        pose.bloom = bloom
        box.render(tile, pose, hands, effect, EffectContext(),
                   cfg.effects.for_effect(effect))
        label(tile, f"cube mode: {effect}" + ("  + rose" if bloom else ""))
        tiles.append(tile)

    # One open palm on its own: the rose hovering above the hand, no box. The
    # skeleton is drawn so the offset can actually be judged.
    from hologram import hud

    tile = scene(TILE)
    solo_hand = [Hand(points=hand_at((TILE[1] * 0.5, TILE[0] * 0.92), 58.0),
                      label="Right", score=1.0)]
    box = BoxView(cfg.box, cfg.tracking)
    pose = None
    for step in range(20):
        pose = box.update(solo_hand, step / 30.0)
    if pose is not None:
        box.render(tile, pose, solo_hand, "halftone", EffectContext(),
                   cfg.effects.for_effect("halftone"))
    hud.draw_skeleton(tile, solo_hand, cfg.hud.color)
    label(tile, "one open palm: rose above the hand")
    tiles.append(tile)

    # Pointing index finger: the red orb off the fingertip.
    from hologram.orb import OrbPose, OrbView

    tile = scene(TILE)
    point_hand = [Hand(points=hand_at((TILE[1] * 0.42, TILE[0] * 0.95), 62.0,
                                      two_fingers=True),
                       label="Right", score=1.0)]
    orb = OrbView(cfg.orb, cfg.tracking)
    pose = None
    for step in range(20):
        pose = orb.update(point_hand, step / 30.0)
    if pose is not None:
        orb.render(tile, pose)
    hud.draw_skeleton(tile, point_hand, cfg.hud.color)
    label(tile, "two fingers: red orb" if pose else "two fingers: NOT DETECTED")
    tiles.append(tile)

    # The orb through a turn. It is 3D - a point cloud with great circles round
    # it - and a still of one angle says nothing about that; four do.
    for turn in (0.0, math.pi / 3, 2 * math.pi / 3, math.pi):
        tile = backdrop()
        orb.render(tile, OrbPose(np.array([TILE[1] / 2.0, TILE[0] / 2.0], np.float32),
                                 TILE[0] * 0.19, 1.0, turn))
        label(tile, f"orb spin {math.degrees(turn):.0f}")
        tiles.append(tile)

    tiles.extend(mode_rows())
    return write(tiles, args.out)


def write(tiles, out: Path) -> int:
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    sheet = np.vstack(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"Wrote {out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
