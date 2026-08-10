"""Example 03: render a 3dgrut-trained model in VKGS, with a glass sphere.

Workflow
--------
1. (in your 3dgrut conda env) Export the trained checkpoint to .ply —
   VKGS loads .ply directly; .pt/.ingp checkpoints are not auto-converted.
   See :func:`export_ply_from_3dgrut` below, or run this script with
   ``--checkpoint ckpt_last.pt`` inside an environment where 3dgrut imports.
2. Build a .vkgs scene around the exported .ply: a refractive glass sphere
   (procedural mesh from vkgs.geometry + preset from vkgs.materials) and a
   point light, rendered with the hybrid raster+RTX pipeline so the mesh
   gets real refraction rays through the splats.
3. Render headless and save the framebuffers.

Coordinate systems: 3dgrut keeps the PLY's native frame while VKGS converts
splats to a right-up-back world on load; the world change of basis is
diag(1, -1, -1). Use ``vkgs.camera.Camera.from_threedgrut_world`` /
``from_kaolin`` to carry camera poses over from 3dgrut scripts.

Usage
-----
    # PLY already at hand:
    python 03_from_3dgrut_ply.py --ply garden.ply --out out/

    # or export from a 3dgrut checkpoint first (needs the 3dgrut env):
    python 03_from_3dgrut_ply.py --checkpoint runs/.../ckpt_last.pt --out out/
"""

import argparse
import glob
import os
import sys

from vkgs.camera import Camera
from vkgs.constants import LightType, Pipeline, ShadowsMode
from vkgs.geometry import ensure_procedural
from vkgs.project import Scene
from vkgs import materials


def export_ply_from_3dgrut(checkpoint_path: str, ply_path: str) -> str:
    """Export a 3dgrut .pt checkpoint to .ply (run inside the 3dgrut env).

    Mirrors the loading path of the 3dgrut playground
    (threedgrut_playground/engine.py:1420-1426) and uses
    ``MixtureOfGaussians.export_ply(path)``
    (threedgrut/model/model.py:931-934).
    """
    try:
        import torch
        from threedgrut.model.model import MixtureOfGaussians
    except ImportError as exc:
        raise SystemExit(
            f"3dgrut is not importable here ({exc}).\n"
            "Run the export inside your 3dgrut conda environment:\n"
            "    import torch\n"
            "    from threedgrut.model.model import MixtureOfGaussians\n"
            "    ckpt = torch.load('ckpt_last.pt', weights_only=False)\n"
            "    model = MixtureOfGaussians(ckpt['config'])\n"
            "    model.init_from_checkpoint(ckpt, setup_optimizer=False)\n"
            "    model.export_ply('exported.ply')\n"
            "then rerun this script with --ply exported.ply"
        )

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = MixtureOfGaussians(checkpoint["config"])
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.export_ply(ply_path)
    print(f"exported {ply_path}")
    return ply_path


def build_scene(ply_path: str, out_dir: str, sphere_pos, sphere_scale: float, spp: int) -> str:
    """Assemble splats + glass sphere + point light and save the .vkgs."""
    scene = Scene()

    # Hybrid pipeline: 3DGS raster primary rays + RTX secondary rays, so the
    # glass sphere refracts the splats. Stochastic effects (glass, soft
    # shadows) need temporal accumulation: render sequences must run at least
    # `spp` frames (handled by RenderScript below).
    scene.renderer.pipeline = Pipeline.HYBRID
    scene.renderer.lighting_enabled = True
    scene.renderer.shadows_mode = ShadowsMode.SOFT

    scene.add_splats(ply_path)

    # Procedural UV sphere (cached .obj) with the clear glass preset. Presets
    # named after the 3dgrut playground registry are available too, e.g.
    # materials.blue_glass(), materials.diamond(), materials.jade().
    sphere_obj = ensure_procedural("sphere", subdivisions=64)
    scene.add_mesh(
        sphere_obj,
        name="glass sphere",
        position=sphere_pos,
        scale=sphere_scale,
        materials=[materials.glass(ior=1.5)],
    )

    scene.add_light(
        LightType.POINT,
        translation=(sphere_pos[0] + 1.5, sphere_pos[1] + 2.0, sphere_pos[2] + 1.0),
        intensity=40.0,
        radius=0.1,
    )

    # Camera preset 0. Replace with poses converted from your 3dgrut/kaolin
    # cameras, e.g. Camera.from_threedgrut_world(c2w, fov=45) or
    # Camera.from_kaolin(kaolin_cam).
    cam = Camera(eye=(2.5, 1.2, 2.5), ctr=sphere_pos, fov=60.0)
    preset = scene.add_camera_preset(cam)
    scene.set_camera(cam)
    assert preset == 0

    project = scene.save(os.path.join(out_dir, "from_3dgrut.vkgs"))
    print(f"wrote {project}")
    return project


def render(project: str, out_dir: str, size, spp: int) -> None:
    """Render camera preset 0 headless; falls back to printing the manual
    command if no vk_gaussian_splatting executable is found."""
    from vkgs.runner import HeadlessRunner, find_executable
    from vkgs.sequence import RenderScript

    script = RenderScript(frames=spp)
    script.load_block()
    stem = script.capture("glass sphere view", camera_preset=0, out_stem=os.path.join(out_dir, "view0"))
    cfg = script.write(os.path.join(out_dir, "from_3dgrut.cfg"))
    print(f"wrote {cfg}")

    try:
        find_executable()
    except FileNotFoundError as exc:
        print(f"\nno renderer executable found ({exc});")
        print("build the app, then either rerun this script or run manually:")
        print(
            f'  vk_gaussian_splatting --size {size[0]} {size[1]} --benchmark 1 --headless 1 '
            f'--sequencefile "{cfg}" --inputProject "{project}" --loadDefaultScene 0'
        )
        return

    runner = HeadlessRunner()
    result = runner.run(
        cfg,
        project=project,
        size=size,
        expected_outputs=[stem + "_main.png"],
    )
    # --saveImage appends a buffer postfix per dumped buffer (_main, _normal,
    # _depth, ...); pick files by postfix, never by numeric buffer index.
    print(f"render finished in {result.duration_s:.1f}s; outputs:")
    for path in sorted(glob.glob(stem + "_*.png")):
        print(f"  {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", help="splat .ply exported from 3dgrut (or any 3DGS ply)")
    parser.add_argument("--checkpoint", help="3dgrut .pt checkpoint to export first (needs 3dgrut importable)")
    parser.add_argument("--out", default="out", help="output directory (default: out)")
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    parser.add_argument("--spp", type=int, default=64, help="temporal samples for the stochastic hybrid pipeline")
    parser.add_argument("--sphere-pos", type=float, nargs=3, default=(0.0, 0.5, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--sphere-scale", type=float, default=0.4)
    args = parser.parse_args(argv)

    if not args.ply and not args.checkpoint:
        parser.error("provide --ply or --checkpoint")

    os.makedirs(args.out, exist_ok=True)
    ply = args.ply
    if ply is None:
        ply = export_ply_from_3dgrut(args.checkpoint, os.path.join(args.out, "exported.ply"))
    if not os.path.isfile(ply):
        sys.exit(f"ply not found: {ply}")

    project = build_scene(ply, args.out, tuple(args.sphere_pos), args.sphere_scale, args.spp)
    render(project, args.out, tuple(args.size), args.spp)


if __name__ == "__main__":
    main()
