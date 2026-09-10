"""Hologram Glitch Band - hand-tracked video effects.

    python run.py                 start it
    python run.py --selftest      check the camera without opening a window
    python run.py --list-effects  show the effect names
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Glitch whatever falls inside a shape drawn by your fingertips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--camera", type=int, help="camera index (default: from config)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="path to config.toml")
    parser.add_argument("--effect", help="effect to start on")
    parser.add_argument("--width", type=int, help="capture width")
    parser.add_argument("--height", type=int, help="capture height")
    parser.add_argument("--no-mirror", action="store_true", help="don't flip the image")
    parser.add_argument("--selftest", action="store_true", help="probe the camera and exit")
    parser.add_argument("--list-effects", action="store_true", help="list effects and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from hologram import config as config_module
    from hologram.app import CameraError, HologramApp

    if args.list_effects:
        from hologram import effects

        print("\n".join(effects.names()))
        return 0

    cfg = config_module.load(args.config)
    if args.width:
        cfg.camera.width = args.width
    if args.height:
        cfg.camera.height = args.height
    if args.no_mirror:
        cfg.camera.mirror = False
    if args.effect:
        cfg.effects.default = args.effect

    app = HologramApp(cfg, camera_index=args.camera)

    if args.effect and app.effect != args.effect:
        print(f"Unknown effect {args.effect!r}. Available: {', '.join(app.effect_names)}")
        return 2

    try:
        return app.selftest() if args.selftest else app.run()
    except KeyboardInterrupt:
        return 0
    except (CameraError, FileNotFoundError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
