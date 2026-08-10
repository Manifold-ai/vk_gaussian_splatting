#!/usr/bin/env python3
"""Build a scene in Python and render it headless with vk_gaussian_splatting.

Usage:
    python 01_build_and_render.py --ply /path/to/point_cloud.ply [--out out/]

The renderer executable is located automatically (repo _bin/Release, or set
$VKGS_BIN). Requires a Vulkan-capable GPU.
"""

import argparse
import os
import sys

from vkgs.camera import Camera
from vkgs.constants import LightType, Pipeline
from vkgs.facade import render_scene
from vkgs.project import Scene
from vkgs.runner import find_executable


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", required=True, help="path to a 3DGS .ply/.spz/.splat file")
    parser.add_argument("--out", default="out", help="output directory (default: out/)")
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    parser.add_argument("--spp", type=int, default=32, help="frames accumulated per camera")
    parser.add_argument(
        "--pipeline",
        type=int,
        default=int(Pipeline.MESH),
        choices=[int(p) for p in Pipeline],
        help="0=vert 1=mesh 2=rtx 3=hybrid 4=3dgut 5=hybrid-3dgut",
    )
    parser.add_argument("--gpu", type=int, default=None, help="Vulkan physical device index")
    args = parser.parse_args()

    try:
        exe = find_executable()
    except FileNotFoundError as err:
        print(f"Renderer executable not available:\n{err}", file=sys.stderr)
        return 1

    if not os.path.isfile(args.ply):
        print(f"Splat file not found: {args.ply}", file=sys.stderr)
        return 1

    # ---- Build the scene ---------------------------------------------------
    scene = Scene()
    scene.renderer.pipeline = args.pipeline
    scene.add_splats(args.ply)
    # A point light only shows up with lighting enabled (path-traced pipelines
    # give the full effect; raster pipelines approximate it).
    scene.renderer.lighting_enabled = True
    scene.add_light(LightType.POINT, translation=(2.0, 3.0, 1.0), intensity=40.0, radius=0.1)

    cameras = [
        Camera(eye=(1.7, 1.5, 1.7), ctr=(0.0, 0.0, 0.0), fov=60.0),
        Camera(eye=(-1.7, 1.5, 1.7), ctr=(0.0, 0.0, 0.0), fov=60.0),
        Camera(eye=(0.0, 2.5, -2.2), ctr=(0.0, 0.0, 0.0), fov=50.0),
    ]

    # ---- Render ------------------------------------------------------------
    print(f"Rendering {len(cameras)} views with {exe}")
    result = render_scene(
        scene,
        cameras,
        size=tuple(args.size),
        spp=args.spp,
        buffers=("main", "depth"),
        out_dir=args.out,
        gpu=args.gpu,
    )

    # ---- Read back ---------------------------------------------------------
    for i in range(len(cameras)):
        rgb = result.image(camera=i, buffer="main")
        print(f"cam{i}: {result.path(i, 'main')}  shape={rgb.shape}  mean={rgb.mean():.1f}")

    frame = next((s for s in result.sequences if s.name == "cam0"), None)
    if frame and "Frame" in frame.timers:
        print(f"cam0 frame time: {frame.timers['Frame']['gpu_ms']:.2f} ms GPU")
    print(f"Log: {result.log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
