"""vkgs — Python scripting layer for vk_gaussian_splatting.

Build scenes programmatically, serialize them to .vkgs project files, drive
the renderer headlessly through generated .cfg benchmark sequences, and read
rendered images back as numpy arrays.

Quick start::

    from vkgs import Scene, Camera, Pipeline, LightType, materials, render_scene

    scene = Scene()
    scene.renderer.pipeline = Pipeline.HYBRID
    scene.renderer.lighting_enabled = True
    scene.add_splats("garden.ply")
    scene.add_mesh("teapot.obj", position=(0, 0.5, 0), materials=[materials.glass()])
    scene.add_light(LightType.POINT, translation=(2, 3, 1), intensity=40)
    idx = scene.add_camera_preset(Camera(eye=(3, 1.5, 2), ctr=(0, 0.5, 0)))

    result = render_scene(scene, cameras=[idx], size=(1920, 1080), spp=64, out_dir="out")
    rgb = result.image(camera=0)  # numpy array

For porting 3dgrut playground scripts, see :mod:`vkgs.compat` (EngineVKGS).
"""

from . import geometry, materials
from .camera import Camera, load_inria_cameras, save_inria_cameras
from .constants import (
    AttenuationMode,
    BillboardBoundingMode,
    CameraModel,
    ColorBufferFormat,
    DofMode,
    EnvMode,
    Format,
    KernelDegree,
    LightType,
    Pipeline,
    ShadowsMode,
    SortingMethod,
    Storage,
    TemporalSamplingMode,
    VkColorFormat,
    Visualize,
)
from .facade import RenderResult, render_scene
from .images import find_outputs, load_image, resolve_outputs
from .project import (
    Environment,
    LightAsset,
    LightInstance,
    Material,
    MeshAsset,
    MeshInstance,
    RendererSettings,
    Scene,
    SplatInstance,
    SplatSet,
    SplatsGlobals,
    Tonemapping,
)
from .runner import HeadlessRunner, RunError, RunResult, find_executable
from .sequence import RenderScript
from .video import interpolate_cameras, load_trajectory, render_video, save_trajectory

__version__ = "0.1.0"

__all__ = [
    "AttenuationMode",
    "BillboardBoundingMode",
    "Camera",
    "CameraModel",
    "ColorBufferFormat",
    "DofMode",
    "EnvMode",
    "Environment",
    "Format",
    "HeadlessRunner",
    "KernelDegree",
    "LightAsset",
    "LightInstance",
    "LightType",
    "Material",
    "MeshAsset",
    "MeshInstance",
    "Pipeline",
    "RenderResult",
    "RenderScript",
    "RendererSettings",
    "RunError",
    "RunResult",
    "Scene",
    "ShadowsMode",
    "SortingMethod",
    "SplatInstance",
    "SplatSet",
    "SplatsGlobals",
    "Storage",
    "TemporalSamplingMode",
    "Tonemapping",
    "VkColorFormat",
    "Visualize",
    "find_executable",
    "find_outputs",
    "geometry",
    "interpolate_cameras",
    "load_image",
    "load_inria_cameras",
    "load_trajectory",
    "materials",
    "render_scene",
    "render_video",
    "resolve_outputs",
    "save_inria_cameras",
    "save_trajectory",
    "__version__",
]
