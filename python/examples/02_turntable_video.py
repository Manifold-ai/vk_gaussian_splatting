#!/usr/bin/env python3
"""Render a turntable orbit video around a 3DGS .ply scene.

Usage:
    python 02_turntable_video.py --ply /path/to/point_cloud.ply [--out turntable.mp4]

Places N orbit keyframes on a circle around the scene center and renders a
closed-loop ("cyclic") camera path in one headless run, then assembles the
MP4 with imageio-ffmpeg. Requires a Vulkan-capable GPU and the renderer
executable (repo _bin/Release, or set $VKGS_BIN).
"""

import argparse
import math
import os
import sys

from vkgs.camera import Camera
from vkgs.constants import Pipeline
from vkgs.project import Scene
from vkgs.runner import find_executable
from vkgs.video import render_video


def orbit_keyframes(count, radius, height, center=(0.0, 0.0, 0.0), fov=60.0):
    """Evenly spaced cameras on a circle around ``center``, looking inward."""
    cams = []
    for k in range(count):
        angle = 2.0 * math.pi * k / count
        eye = (
            center[0] + radius * math.cos(angle),
            center[1] + height,
            center[2] + radius * math.sin(angle),
        )
        cams.append(Camera(eye=eye, ctr=tuple(center), up=(0.0, 1.0, 0.0), fov=fov))
    return cams


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", required=True, help="path to a 3DGS .ply/.spz/.splat file")
    parser.add_argument("--out", default="turntable.mp4", help="output video path (default: turntable.mp4)")
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    parser.add_argument("--keyframes", type=int, default=8, help="orbit keyframe count (default: 8)")
    parser.add_argument("--frames-between", type=int, default=30, help="interpolated frames per keyframe segment")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--spp", type=int, default=32, help="frames accumulated per video frame")
    parser.add_argument("--radius", type=float, default=3.0, help="orbit radius")
    parser.add_argument("--height", type=float, default=1.0, help="camera height above the orbit center")
    parser.add_argument("--center", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--mode",
        default="cyclic",
        choices=["cyclic", "spline", "linear"],
        help="trajectory interpolation (cyclic closes the orbit loop)",
    )
    parser.add_argument(
        "--pipeline",
        type=int,
        default=int(Pipeline.MESH),
        choices=[int(p) for p in Pipeline],
        help="0=vert 1=mesh 2=rtx 3=hybrid 4=3dgut 5=hybrid-3dgut",
    )
    parser.add_argument("--gpu", type=int, default=None, help="Vulkan physical device index")
    parser.add_argument("--keep-frames", action="store_true", help="keep the per-frame PNGs")
    parser.add_argument("--workdir", default=None, help="frame/work directory (default: temp dir)")
    args = parser.parse_args()

    try:
        find_executable()
    except FileNotFoundError as err:
        print(f"Renderer executable not available:\n{err}", file=sys.stderr)
        return 1

    if not os.path.isfile(args.ply):
        print(f"Splat file not found: {args.ply}", file=sys.stderr)
        return 1

    scene = Scene()
    scene.renderer.pipeline = args.pipeline
    scene.add_splats(args.ply)

    keyframes = orbit_keyframes(args.keyframes, args.radius, args.height, center=args.center)

    try:
        out = render_video(
            scene,
            keyframes,
            args.out,
            mode=args.mode,
            frames_between=args.frames_between,
            fps=args.fps,
            spp=args.spp,
            size=tuple(args.size),
            workdir=args.workdir,
            keep_frames=args.keep_frames,
            gpu=args.gpu,
        )
    except ImportError as err:
        print(f"Video assembly unavailable:\n{err}", file=sys.stderr)
        return 1
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
