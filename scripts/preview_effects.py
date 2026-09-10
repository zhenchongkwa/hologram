"""Render every effect onto a synthetic scene and write a contact sheet.

Useful for tuning config.toml without standing in front of the camera.

    python scripts/preview_effects.py [-o captures/effects.png]
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

from hologram import config as config_module, effects  # noqa: E402
from hologram import shape as shape_module  # noqa: E402
from hologram.effects import EffectContext  # noqa: E402
from hologram.shape import Shape, ribbon_quad  # noqa: E402

TILE = (360, 640)  # height, width


def build_scene() -> np.ndarray:
    """A stand-in for a webcam frame: gradient, shapes, and fine detail."""
    height, width = TILE
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]

    scene = np.zeros((height, width, 3), np.float32)
    scene[:, :, 0] = 90 + 120 * y            # blue rises down the frame
    scene[:, :, 1] = 60 + 90 * x             # green rises to the right
    scene[:, :, 2] = 140 * (1 - y) + 40 * x  # red falls down
    frame = np.clip(scene, 0, 255).astype(np.uint8)

    # A head-and-shoulders silhouette, so the effects have an edge to bite on.
    cv2.circle(frame, (width // 2, height // 2 - 30), 78, (215, 205, 195), -1, cv2.LINE_AA)
    cv2.ellipse(frame, (width // 2, height + 40), (170, 130), 0, 180, 360,
                (150, 140, 135), -1, cv2.LINE_AA)
    cv2.circle(frame, (width // 2 - 28, height // 2 - 42), 9, (60, 55, 50), -1, cv2.LINE_AA)
    cv2.circle(frame, (width // 2 + 28, height // 2 - 42), 9, (60, 55, 50), -1, cv2.LINE_AA)

    # Fine stripes, to show what each effect does to high-frequency detail.
    for offset in range(0, width, 26):
        cv2.line(frame, (offset, 0), (offset - 60, height), (235, 235, 235), 1, cv2.LINE_AA)
    return frame


def label(tile: np.ndarray, text: str, color) -> None:
    cv2.putText(tile, text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(tile, text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", type=Path, default=ROOT / "captures" / "effects.png")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--twist", type=float, default=0.0,
                        help="degrees to counter-rotate the right hand; 180 gives a full X")
    args = parser.parse_args()

    cfg = config_module.load(ROOT / "config.toml")
    scene = build_scene()
    hud_color = tuple(int(c) for c in cfg.hud.color)

    names = ["source"] + effects.names()
    tiles = []
    for name in names:
        tile = scene.copy()
        if name != "source":
            params = cfg.effects.for_effect(name)
            ctx = EffectContext()

            def shape_at(offset: float) -> Shape:
                """Roughly where four fingertips of two raised hands would land.

                The right hand is rotated by --twist about its own midpoint, the
                way turning that wrist would, so the rails cross at 180.
                """
                left_x = TILE[1] * 0.13 + offset
                thumb_a = (left_x, TILE[0] * 0.78)
                index_a = (left_x, TILE[0] * 0.22)

                right_x, middle = TILE[1] * 0.87 + offset, TILE[0] * 0.5
                reach = TILE[0] * 0.28
                turn = math.radians(args.twist)
                dx, dy = math.sin(turn) * reach, math.cos(turn) * reach
                thumb_b = (right_x + dx, middle + dy)
                index_b = (right_x - dx, middle - dy)

                tips = [thumb_a, index_a, thumb_b, index_b]
                return Shape(
                    points=ribbon_quad(*tips), anchors=np.array(tips, dtype=np.float32)
                )

            # Effects that build up over time - ghost in particular - show
            # nothing on a single frame, so sweep the shape in first on throwaway
            # copies to give them some history.
            for step in range(3, 0, -1):
                shape_module.render(
                    scene.copy(), shape_at(-14.0 * step), name, ctx, params, cfg.shape
                )
            shape_module.render(tile, shape_at(0.0), name, ctx, params, cfg.shape)
        label(tile, name, hud_color)
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (40, 40, 40), 1)
        tiles.append(tile)

    columns = max(args.columns, 1)
    while len(tiles) % columns:
        tiles.append(np.zeros_like(scene))
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    sheet = np.vstack(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), sheet)
    print(f"Wrote {args.out}  ({sheet.shape[1]}x{sheet.shape[0]}, {len(names)} tiles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
