# Python Scripting

The `vkgs` Python package (in `python/`) lets you build scenes as `.vkgs`
project files, drive the renderer headlessly through generated `.cfg`
benchmark sequences, and read rendered images back as numpy arrays — no GUI
interaction required. It also ships a compatibility shim for porting
[3dgrut](https://github.com/nv-tlabs/3dgrut) playground scripts.

## Installation

The package is pure Python (numpy only for the core); it works with a locally
built executable or a prebuilt release binary.

```bash
pip install -e python/            # core
pip install -e "python/[image]"   # + imageio (png/hdr readback)
pip install -e "python/[video]"   # + imageio-ffmpeg (mp4 assembly)
pip install -e "python/[dev]"     # + pytest
```

The renderer executable is discovered in this order: the `executable=`
argument, the `VKGS_BIN` environment variable, `_bin/Release/` and
`_bin/Debug/` in the repository, then `build*/` directories.

## Building and rendering a scene

```python
from vkgs import Scene, Camera, Pipeline, LightType, EnvMode, materials, render_scene

scene = Scene()
scene.renderer.pipeline = Pipeline.HYBRID       # raster primary + RTX secondary
scene.renderer.lighting_enabled = True
scene.renderer.rtx_max_bounces = 3

scene.add_splats("garden.ply")                  # .ply/.spz/.splat, dedup by path
scene.add_mesh("assets/teapot.obj",
               position=(0, 0.5, 0),
               materials=[materials.glass(ior=1.5)])
scene.add_light(LightType.POINT, translation=(2, 3, 1), intensity=40, radius=0.1)
scene.set_environment(EnvMode.HDR, hdr_file="env/studio.hdr", ibl_intensity=1.2)

cams = [Camera(eye=(3, 1.5, z), ctr=(0, 0.5, 0), fov=50) for z in (2, 0, -2)]
idx = [scene.add_camera_preset(c) for c in cams]

result = render_scene(scene, cameras=idx, size=(1920, 1080), spp=64,
                      buffers=["main", "depth"], out_dir="out/")
rgb = result.image(camera=0)                     # numpy array
depth = result.image(camera=0, buffer="depth")
```

`Scene.save("scene.vkgs")` / `Scene.load("scene.vkgs")` round-trip the same
project format the application reads and writes (format version 7), so scenes
composed in the GUI can be post-processed in Python and vice versa.

Under the hood `render_scene`:

1. writes the scene to `out_dir/scene.vkgs` (camera presets included),
2. generates a `.cfg` sequence file — one *render* sequence
   (`--activateCameraPreset N`, `--sequenceframes spp`) followed by one *save*
   sequence (`--saveImage`) per camera (the save captures the previous
   sequence's converged frame),
3. runs `vk_gaussian_splatting --headless 1 --benchmark 1 --inputProject ...
   --sequencefile ...` as a subprocess and checks the log for errors,
4. maps the buffer-postfixed output files (`*_main.png`, `*_depth.png`, ...)
   back to numpy arrays.

Timing/profiler data parsed from the run log is available via
`result.sequences`.

## Camera utilities

`vkgs.Camera` mirrors the application camera (eye/ctr/up, vertical fov in
degrees, per-camera depth of field). Conversions are provided from 4x4 view
matrices, COLMAP/INRIA `cameras.json` entries (`Camera.from_colmap` — unlike
the in-app `--loadCameraPresets` importer, this preserves the fov), kaolin
cameras (`Camera.from_kaolin`), and 3dgrut-world poses
(`Camera.from_threedgrut_world`, which applies the `diag(1,-1,-1)` world
change of basis between the two renderers).

## Video rendering

```python
from vkgs import render_video

render_video(scene, keyframes=cams, out="orbit.mp4",
             mode="spline", frames_between=60, fps=30, spp=32)
```

All interpolated frames become camera presets in a single project file and are
rendered in one process launch. `save_trajectory`/`load_trajectory` interop
with 3dgrut `VideoRecorder` trajectory files.

## Porting 3dgrut playground scripts

`vkgs.compat.EngineVKGS` mirrors the `Engine3DGRUT` surface so that headless
playground scripts port with minimal edits:

```python
from vkgs.compat import EngineVKGS, OptixPrimitiveTypes

engine = EngineVKGS(gs_object="model.ply", mesh_assets_folder="./assets")
engine.camera_fov = 60.0
engine.primitives.add_primitive("Sphere", OptixPrimitiveTypes.GLASS)
engine.add_light(light_type=1, direction=(0, -1, -1), intensity=3.0)

frame = engine.render(camera)        # {'rgb': (1,H,W,3), 'opacity': (1,H,W,1)}
```

Key differences from the CUDA/OptiX playground (each raises or emits a
`CompatWarning` with a workaround):

- **Execution model**: scene mutations are collected lazily; every `render()`
  serializes the scene and launches a headless render (seconds, not
  milliseconds). Use `render_many()` to amortize one launch over many cameras.
- **Not available** (renderer-level gaps): custom per-pixel rays
  (`raygen`/`RayPack`), true progressive `render_pass` refinement, shadow
  catchers, rectangular area lights, live `scene_mog` tensor access, extra
  AOVs (`hits_count`), and the OptiX denoiser (DLSS Ray Reconstruction is the
  VKGS analog in `USE_DLSS` builds).
- **Approximated**: MIRROR/GLASS map to PBR metallic/transmission factors,
  soft-shadow radii, MSAA/Sobol antialiasing (temporal accumulation instead),
  point-light falloff. Tonemapping and gamma are applied in Python on the HDR
  readback using 3dgrut's own formulas, so those match pixel-exactly.
- **`.pt`/`.ingp` checkpoints** are not loaded directly: export a PLY first in
  your 3dgrut environment (`model.export_ply(...)` — see
  `python/examples/03_from_3dgrut_ply.py`).

## Benchmark-script extras

Two parameters were added for scripting workflows (usable in any `.cfg`):

- `--saveProject out.vkgs` — saves the current scene state to a project file
  (e.g. to convert CLI-composed scenes into projects, or for round-trip
  testing). Like `--saveImage`, it executes at sequence start, capturing the
  state as of the previous sequence's end.
- RTX pipelines (2/3/5) now write accumulated coverage (`1 - transmittance`)
  into the alpha channel of the main color buffer instead of constant 1.0,
  so `--saveImage` PNG/RAW dumps carry a usable opacity channel, matching the
  raster pipelines.

## Examples and tests

- `python/examples/01_build_and_render.py` — scene building + multi-camera render
- `python/examples/02_turntable_video.py` — orbit video
- `python/examples/03_from_3dgrut_ply.py` — 3dgrut checkpoint → PLY → VKGS render
- `python/examples/04_playground_port.py` — ported playground script

```bash
python -m pytest python/tests -m "not gpu"   # CPU-only unit tests
python -m pytest python/tests -m gpu         # smoke tests (needs built exe + dataset)
```
