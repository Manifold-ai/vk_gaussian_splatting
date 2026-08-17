#!/usr/bin/env python3
"""[MESH-BLACK-DIAG] Headless reproduction of the "full black screen when a .glb
is added to a gaussian scene" bug.

For a matrix of {pipeline} x {splats-only, splats+glb} it renders the ``main``
AOV, reports which combinations come out all-black, and then prints the C++
``[MESH-BLACK-DIAG]`` / error / validation lines captured from the renderer log.
The expected symptom (per the bug report): the *splats-only* rows render fine
while the *splats+glb* rows go fully black — meaning adding a mesh breaks the
whole frame, not just mesh visibility.

Prerequisites
-------------
* A built ``vk_gaussian_splatting`` executable. Set ``$VKGS_BIN`` or build into
  ``_bin/{Release,Debug}`` / ``build*/`` (see docs/getting-started.md).
* A test ``.ply`` gaussian scene. ``flowers_1.ply`` (the app's default scene) is
  auto-discovered; override with ``$VKGS_TEST_PLY``.
* A test ``.glb`` (defaults to ``~/vkgs-test-models/vkgs-test-models/model_997.glb``;
  override with argv[1] or ``$VKGS_TEST_GLB``).

Usage
-----
    cd python && uv run python examples/repro_glb_black.py [path/to/model.glb]

Paste the full console output (the per-row PASS/BLACK table + the captured
``[MESH-BLACK-DIAG]`` lines) back into the chat.
"""

import glob
import os
import sys

import numpy as np

from vkgs.camera import Camera
from vkgs.constants import Pipeline
from vkgs.facade import render_scene
from vkgs.project import Scene
from vkgs.runner import find_executable

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Pipelines to exercise. MESH (pure raster) and HYBRID (raster + RTX mesh color)
# are the two the interactive app and the AOV/product path use. Edit freely.
PIPELINES = [Pipeline.MESH, Pipeline.HYBRID]

# Markers to surface from the renderer log after each run.
LOG_MARKERS = ("[MESH-BLACK-DIAG]", "LOGE", "Failed to create", "VUID-", "Validation",
               "VK_ERROR", "DEVICE_LOST", "error:", "Error")


def find_test_ply():
    env = os.environ.get("VKGS_TEST_PLY")
    if env and os.path.isfile(env):
        return env
    for pattern in (
        os.path.join(REPO_ROOT, "_bin", "**", "flowers_1.ply"),
        os.path.join(REPO_ROOT, "downloaded_resources", "**", "flowers_1.ply"),
        os.path.join(REPO_ROOT, "_bin", "**", "*.ply"),
        os.path.join(REPO_ROOT, "downloaded_resources", "**", "*.ply"),
    ):
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]
    return None


def find_test_glb(argv):
    if len(argv) > 1 and os.path.isfile(argv[1]):
        return os.path.abspath(argv[1])
    env = os.environ.get("VKGS_TEST_GLB")
    if env and os.path.isfile(env):
        return env
    default = os.path.expanduser("~/vkgs-test-models/vkgs-test-models/model_997.glb")
    return default if os.path.isfile(default) else None


def print_log_markers(log_path):
    """Print the interesting lines (diagnostics/errors/validation) from a log."""
    if not log_path or not os.path.isfile(log_path):
        print(f"    (no log at {log_path})")
        return
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    hits = [ln for ln in lines if any(m in ln for m in LOG_MARKERS)]
    if not hits:
        print(f"    (no diagnostic/error markers in {log_path})")
        return
    print(f"    --- markers from {log_path} ---")
    for ln in hits:
        print(f"    {ln}")


def render_once(ply, glb, pipeline, with_glb, out_dir):
    """Render one main-AOV frame; return (is_black, max_rgb, log_path) or raise."""
    scene = Scene()
    scene.renderer.pipeline = pipeline
    # Mirror the backend: needSurfaceInfo() must be on for the surface/FTB path
    # (and for the AOV writes). This is exactly the config the product pipeline uses.
    scene.renderer.force_surfel = True
    scene.add_splats(ply)
    if with_glb:
        # Placement is irrelevant to the bug: if the pipeline is healthy the
        # splats stay visible even with a mis-placed mesh. The bug blacks the
        # whole frame. Origin + unit scale keeps it simple.
        scene.add_mesh(glb, position=(0.0, 0.0, 0.0), scale=1.0)
    preset = scene.add_camera_preset(Camera(eye=(1.7, 1.5, 1.7), ctr=(0, 0, 0)))

    result = render_scene(
        scene,
        [preset],
        size=(320, 240),
        spp=8,
        buffers=("main",),
        out_dir=out_dir,
        image_format="raw",          # RGBA32F -> exact black detection
        extra_args=("--validation", "1"),  # bool[1]: needs an explicit value -> surface Vulkan validation errors
    )
    img = result.image(0, "main")     # (H, W, 4) float32
    rgb_max = float(img[..., :3].max())
    is_black = rgb_max < 1e-3
    return is_black, rgb_max, result.log_path


def main():
    ply = find_test_ply()
    glb = find_test_glb(sys.argv)

    try:
        exe = find_executable()
    except FileNotFoundError:
        exe = None

    print("=" * 78)
    print("[MESH-BLACK-DIAG] GLB-black headless reproduction")
    print("=" * 78)
    print(f"executable : {exe}")
    print(f"ply        : {ply}")
    print(f"glb        : {glb}")
    print()

    if exe is None:
        print("ERROR: no built executable found. Build first, e.g.:")
        print("    cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j")
        print("then re-run (or set $VKGS_BIN to the binary).")
        return 2
    if ply is None:
        print("ERROR: no test .ply found. Set $VKGS_TEST_PLY to any gaussian .ply.")
        return 2
    if glb is None:
        print("ERROR: no test .glb found. Pass one as argv[1] or set $VKGS_TEST_GLB.")
        return 2

    out_root = os.path.join(REPO_ROOT, "_repro_glb_black")
    rows = []       # (pipeline, with_glb, status, max_rgb, log_path)
    for pipeline in PIPELINES:
        for with_glb in (False, True):
            tag = f"{Pipeline(pipeline).name.lower()}_{'glb' if with_glb else 'splatsonly'}"
            out_dir = os.path.join(out_root, tag)
            print(f"--- rendering {tag} ...")
            try:
                is_black, rgb_max, log_path = render_once(ply, glb, pipeline, with_glb, out_dir)
                status = "BLACK" if is_black else "ok"
                rows.append((pipeline, with_glb, status, rgb_max, log_path))
            except Exception as exc:  # RunError etc. — keep going, report per row
                rows.append((pipeline, with_glb, f"ERROR: {exc}", float("nan"),
                             os.path.join(out_dir, "render.log")))

    print()
    print("=" * 78)
    print("SUMMARY  (expect: splats-only=ok, +glb=BLACK reproduces the bug)")
    print("=" * 78)
    print(f"{'pipeline':<10} {'+glb':<6} {'status':<10} {'max(rgb)':>10}")
    for pipeline, with_glb, status, rgb_max, _ in rows:
        name = Pipeline(pipeline).name.lower()
        print(f"{name:<10} {('yes' if with_glb else 'no'):<6} {status:<10} {rgb_max:>10.5f}")

    print()
    print("=" * 78)
    print("CAPTURED RENDERER DIAGNOSTICS (per row)")
    print("=" * 78)
    for pipeline, with_glb, status, _, log_path in rows:
        name = Pipeline(pipeline).name.lower()
        print(f"[{name} +glb={'yes' if with_glb else 'no'}  status={status}]")
        print_log_markers(log_path)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
