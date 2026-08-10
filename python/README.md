# vkgs — Python scripting layer for vk_gaussian_splatting

Build scenes programmatically, serialize them to `.vkgs` project files, drive
the renderer headlessly through generated `.cfg` benchmark sequences, and read
rendered images back as numpy arrays. Includes a 3dgrut-playground
compatibility shim (`vkgs.compat.EngineVKGS`) for porting scripts between the
two renderers.

```bash
pip install -e python/          # core (numpy only)
pip install -e "python/[image]" # + imageio for png/hdr readback
pip install -e "python/[video]" # + mp4 assembly
pip install -e "python/[dev]"   # + pytest
```

```python
from vkgs import Scene, Camera, Pipeline, LightType, materials, render_scene

scene = Scene()
scene.renderer.pipeline = Pipeline.HYBRID
scene.renderer.lighting_enabled = True
scene.add_splats("garden.ply")
scene.add_mesh("teapot.obj", position=(0, 0.5, 0), materials=[materials.glass()])
scene.add_light(LightType.POINT, translation=(2, 3, 1), intensity=40)
idx = scene.add_camera_preset(Camera(eye=(3, 1.5, 2), ctr=(0, 0.5, 0)))

result = render_scene(scene, cameras=[idx], size=(1920, 1080), spp=64, out_dir="out")
rgb = result.image(camera=0)
```

The renderer executable is located automatically (`_bin/Release/`, `$VKGS_BIN`,
or `executable=` argument). See `docs/python-scripting.md` at the repository
root for the full guide, `examples/` for runnable scripts, and run tests with
`python -m pytest python/tests -m "not gpu"`.
