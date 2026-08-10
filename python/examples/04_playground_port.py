"""Example 04: the 3dgrut headless playground notebook, ported to VKGS.

This script mirrors threedgrut_playground/headless.ipynb cell-for-cell using
the ``vkgs.compat`` shim. Diff-notes below mark every place the API deviates
from Engine3DGRUT; everything else is copy-paste compatible.

Usage
-----
    python 04_playground_port.py --ply garden.ply --out out/
    python 04_playground_port.py --ply garden.ply --assets ./assets \\
        --geometry Armadillo --size 512 512

Requires a built vk_gaussian_splatting executable (or $VKGS_BIN); without
one the script writes the .vkgs/.cfg pair and prints the manual command.
"""

import argparse
import os
import sys

import numpy as np

# DIFF-NOTE [imports]: the 3dgrut notebook imports torch/kaolin and
#   `from threedgrut_playground.engine import Engine3DGRUT, OptixPrimitiveTypes`.
# The shim needs neither torch nor kaolin (numpy in, numpy out; pass
# return_torch=True for torch tensors if torch is installed).
from vkgs.compat import EngineVKGS, OptixPrimitiveTypes
from vkgs.camera import Camera


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", required=True, help="splat .ply (export .pt checkpoints with model.export_ply first)")
    parser.add_argument("--assets", default=None, help="mesh assets folder (.obj/.glb), like the 3dgrut assets dir")
    parser.add_argument("--geometry", default="Sphere", help="geometry to insert (default: procedural Sphere)")
    parser.add_argument("--out", default="out", help="output directory")
    parser.add_argument("--size", type=int, nargs=2, default=(512, 512), metavar=("W", "H"))
    parser.add_argument("--gpu", type=int, default=None, help="unused; kept for CLI symmetry")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    # ---------------------------------------------------------- Setup Engine
    # DIFF-NOTE [ctor]: Engine3DGRUT(gs_object, mesh_assets_folder,
    #   default_config) loads the model onto the GPU here. EngineVKGS only
    #   records state; the renderer runs as a subprocess at render() time.
    #   default_config is unnecessary (and warns if passed) because VKGS
    #   consumes the .ply directly; .pt/.ingp raise with export instructions.
    engine = EngineVKGS(
        gs_object=args.ply,
        mesh_assets_folder=args.assets,
        out_dir=args.out,
        size=tuple(args.size),
    )

    # ------------------------------------------------- Configure the engine
    # Identical to the notebook's configuration cell.
    engine.camera_type = "Pinhole"
    engine.camera_fov = 60.0
    engine.use_spp = True
    # DIFF-NOTE [antialiasing]: accepted verbatim, but maps to the temporal
    #   accumulation frame count (8 frames here); a CompatWarning reminds you
    #   the per-sample jitter pattern differs from 3dgrut's MSAA grid.
    engine.antialiasing_mode = "8x MSAA"

    # Remove initial glass sphere from scene (same idiom as the notebook —
    # EngineVKGS also seeds a glass Sphere on construction).
    for mesh_name in list(engine.primitives.objects.keys()):
        engine.primitives.remove_primitive(mesh_name)

    # Add a glass primitive to the scene.
    # DIFF-NOTE [device]: the device argument is accepted and ignored — the
    #   mesh lives in the renderer subprocess, not in torch CUDA memory.
    #   'Quad'/'Sphere' are always available (procedural .obj); other names
    #   come from the --assets folder registry (capitalized file stems).
    engine.primitives.add_primitive(
        geometry_type=args.geometry,
        primitive_type=OptixPrimitiveTypes.GLASS,
        device="cuda",
    )

    # Optional: a soft point light (same Light dataclass fields as 3dgrut).
    engine.add_light(light_type=2, position=(0.0, -2.0, -1.0), intensity=30.0, angular_radius=0.02)

    # -------------------------------------------------- Render single image
    # DIFF-NOTE [camera]: the notebook builds a kaolin camera
    #   (kaolin.render.easy_render.default_camera(512).cuda() + move/rotate).
    #   The shim accepts kaolin cameras directly when kaolin is installed;
    #   here we use a plain vkgs Camera (or pass a 4x4 3dgrut camera-to-world
    #   matrix). Note the vkgs Camera is in the VKGS world frame (RUB);
    #   kaolin/matrix inputs are converted with diag(1,-1,-1) automatically.
    camera = Camera(eye=(2.5, 1.5, 2.5), ctr=(0.0, 0.0, 0.0), fov=60.0)

    try:
        # Render a full quality frame — identical call and return contract:
        # {'rgb': (1,H,W,3) in [0,1], 'opacity': (1,H,W,1), 'rgb_buffer': rgb}
        framebuffer = engine.render(camera)
    except FileNotFoundError as exc:
        # DIFF-NOTE [execution]: rendering needs the vk_gaussian_splatting
        #   binary. The lazy scene state can still be serialized without it.
        scene = engine._build_scene()
        project = scene.save(os.path.join(args.out, "playground_port.vkgs"))
        print(f"no renderer executable found:\n  {exc}\n")
        print(f"scene written to {project}; build the app or set $VKGS_BIN, then rerun.")
        return 1

    # DIFF-NOTE [buffers]: numpy arrays by default (use return_torch=True for
    #   torch); 'opacity' is all-ones until the RTX alpha patch + .raw
    #   readback land (raster pipelines carry real coverage alpha).
    rgba = np.concatenate([framebuffer["rgb"], framebuffer["opacity"]], axis=-1)
    out_png = os.path.join(args.out, "playground_port.png")
    try:
        import imageio.v3 as iio

        iio.imwrite(out_png, (np.clip(rgba[0], 0.0, 1.0) * 255).astype(np.uint8))
        print(f"wrote {out_png}")
    except ImportError:
        print("imageio not installed; skipping PNG export (pip install 'vkgs[image]')")

    # ------------------------------------------- Interactive-style rendering
    # DIFF-NOTE [progressive]: render_pass accumulates *nothing* between
    #   calls — the subprocess renders all samples in one go, so
    #   is_first_pass is only honored as "re-render if dirty", and
    #   has_progressive_effects_to_render() is always False. The notebook's
    #   refinement loop therefore degenerates to a cache hit:
    framebuffer = engine.render_pass(camera, is_first_pass=engine.is_dirty(camera))
    while engine.has_progressive_effects_to_render():  # never loops
        framebuffer = engine.render_pass(camera, is_first_pass=False)
    assert framebuffer["rgb"] is engine.last_state["rgb"]  # served from cache
    print("render_pass served the cached frame (no second subprocess run).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
