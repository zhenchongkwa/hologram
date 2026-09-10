"""Turn a rose .obj into a point cloud the app can load instantly.

The source model is a whole scene - flower, leaves, a long stem and three grass
backdrop planes - and 13 MB of text with about 80,000 faces. Rasterising that
many faces in Python is hopeless, but its vertices make an excellent point
cloud, which is what the app draws.

Parsing happens once, here, and the result is saved as a small .npy. Reading
13 MB of text on every start would add seconds to launch for a fixed answer.

    python scripts/import_rose.py "C:/path/to/rose.obj"
    python scripts/import_rose.py "..." --part flower --points 20000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Groups are matched by material, not by name, so the same rules survive a
# model whose objects are named differently. The grass planes are a backdrop
# and are always dropped.
GRASS_HINTS = ("grass",)
STEM_HINTS = ("paper", "green")


def parse_obj(path: Path):
    """Read vertices and, for each group, which vertex indices it uses."""
    vertices: list[tuple[float, float, float]] = []
    groups: dict[str, set[int]] = {}
    materials: dict[str, str] = {}
    current = "default"

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("g "):
                current = line.split()[1] if len(line.split()) > 1 else "default"
                groups.setdefault(current, set())
            elif line.startswith("usemtl "):
                materials[current] = line.split()[1]
            elif line.startswith("f "):
                used = groups.setdefault(current, set())
                for token in line.split()[1:]:
                    index = token.split("/")[0]
                    if index:
                        value = int(index)
                        # OBJ allows negative indices, counting back from the
                        # end of the vertex list so far.
                        used.add(value - 1 if value > 0 else len(vertices) + value)

    return np.asarray(vertices, dtype=np.float32), groups, materials


def classify(groups, materials, mtl_text: str) -> dict[str, str]:
    """Label each group flower / stem / grass from its material's texture."""
    texture_of: dict[str, str] = {}
    material = None
    for line in mtl_text.splitlines():
        line = line.strip()
        if line.startswith("newmtl "):
            material = line.split()[1]
        elif line.startswith("map_Kd ") and material:
            texture_of[material] = line.split(maxsplit=1)[1].lower()

    labels = {}
    for group in groups:
        texture = texture_of.get(materials.get(group, ""), "")
        if any(hint in texture for hint in GRASS_HINTS):
            labels[group] = "grass"
        elif any(hint in texture for hint in STEM_HINTS):
            labels[group] = "stem"
        else:
            labels[group] = "flower"
    return labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("obj", type=Path, help="path to rose.obj")
    parser.add_argument("--part", choices=("plant", "flower"), default="plant",
                        help="'plant' keeps flower, leaves and stem; 'flower' drops the stem")
    parser.add_argument("--points", type=int, default=24000)
    parser.add_argument("-o", "--out", type=Path,
                        default=ROOT / "models" / "rose_points.npy")
    args = parser.parse_args()

    if not args.obj.exists():
        print(f"No such file: {args.obj}", file=sys.stderr)
        return 1

    vertices, groups, materials = parse_obj(args.obj)
    mtl = args.obj.with_suffix(".mtl")
    labels = classify(groups, materials, mtl.read_text(errors="replace") if mtl.exists() else "")

    keep_labels = {"flower", "stem"} if args.part == "plant" else {"flower"}
    keep: set[int] = set()
    print(f"{'group':<14}{'label':<9}{'verts':>8}")
    for group, used in groups.items():
        label = labels.get(group, "flower")
        print(f"{group:<14}{label:<9}{len(used):>8}"
              + ("" if label in keep_labels else "   (dropped)"))
        if label in keep_labels:
            keep |= used

    def gather(labels_wanted):
        picked = set()
        for group, used in groups.items():
            if labels.get(group, "flower") in labels_wanted:
                picked |= used
        return np.fromiter(sorted(i for i in picked if 0 <= i < len(vertices)),
                           dtype=np.int64)

    flower = vertices[gather({"flower"})]
    if len(flower) == 0:
        print("No flower found - check the material hints.", file=sys.stderr)
        return 1

    # The stem is kept whole while the flower is strided down. The flower has
    # sixty times the stem's vertices packed into a fifth of the height, so
    # sampling them together leaves the stem too sparse to see at all.
    stem = vertices[gather({"stem"})] if args.part == "plant" else np.zeros((0, 3), np.float32)
    budget = max(args.points - len(stem), 1)
    if len(flower) > budget:
        # Evenly strided rather than random: vertex order follows the surface,
        # so a stride samples it uniformly where a random draw clumps.
        step = len(flower) / budget
        flower = flower[(np.arange(budget) * step).astype(np.int64)]

    # Framed on the flower, not on the whole plant. The model is 86 units tall
    # and 68 of that is bare stem, so normalising the lot shrinks the bloom to
    # a dot. Scaling to the flower instead fills the box with it and lets the
    # stem simply run out of the bottom.
    centre = flower.mean(axis=0)
    reach = float(np.abs(flower - centre).max()) or 1.0
    points = np.concatenate([flower, stem], axis=0) if len(stem) else flower
    points = (points - centre) / reach
    # The model is Y-up; screen Y runs down.
    points[:, 1] *= -1.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, points.astype(np.float32))
    span = points.max(axis=0) - points.min(axis=0)
    print(f"\nSaved {len(points):,} points to {args.out}")
    print(f"extent  x {span[0]:.2f}  y {span[1]:.2f}  z {span[2]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
