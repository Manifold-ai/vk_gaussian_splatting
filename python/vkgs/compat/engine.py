"""EngineVKGS: a 3dgrut-playground-compatible facade over the VKGS
headless renderer.

Mirrors the outward surface of ``Engine3DGRUT``
(threedgrut_playground/engine.py:895-1667) on a lazy execution model: every
setter only mutates in-memory state; ``render(camera)`` serializes the
scene to .vkgs + .cfg, runs one headless subprocess
(vkgs.facade.render_scene, image_format='raw') and returns
``{'rgb': (1, H, W, 3) float32 in [0, 1], 'opacity': (1, H, W, 1),
'rgb_buffer': alias of rgb}`` (plus 'depth' when ``return_depth=True``) —
numpy by default, torch-wrapped with ``return_torch=True``.

Tonemapping/gamma are applied in Python on the HDR readback using 3dgrut's
exact formulas (vkgs.compat.tonemap), so 'None'/'Reinhard'/'Filmic' match
pixel-for-pixel modulo the renderer's own output differences.

Opacity note: ``opacity`` is real per-pixel coverage, read from the 4th
channel of the ``.raw`` RGBA32F main-buffer dump. Both pipelines write it:
RTX as ``1 - transmittance`` (shaders/threedgrt_raytrace.rgen.slang), raster
via premultiplied alpha. (A ``.hdr`` readback would drop alpha and yield
all-ones; the compat engine uses ``.raw`` precisely to keep coverage.)

Progressive-rendering surface (render_pass/is_dirty/...) is ported to the
batch model: ``render_pass`` performs one full-quality render and then
serves cached buffers until the state or camera changes (plan gap A2 — the
subprocess accumulates all samples inside a single run, so per-pass
refinement cannot cross the process boundary).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import facade, images
from ..camera import Camera
from ..constants import CameraModel, DofMode, EnvMode, Pipeline, ShadowsMode
from ..geometry import ensure_procedural
from ..project import Scene
from . import tonemap as tonemap_mod
from .convert import (
    Light,
    OptixPrimitiveTypes,
    camera_to_vkgs,
    light_to_vkgs,
    primitive_type_to_material,
    warn_compat,
)
from .primitives import PrimitivesVKGS, ply_scene_extent

# Pipelines where the camera DoF parameters take effect
# (shaders/cameras.h.slang callers; raster primaries ignore them).
_DOF_PIPELINES = {Pipeline.RTX, Pipeline.MESH_3DGUT, Pipeline.HYBRID_3DGUT}

# Pipelines where the GS shadow mask takes effect (RTX-traced splats only,
# src/parameters.h:185-188): RTX(2) / HYBRID(3) / HYBRID_3DGUT(5).
_GS_SHADOW_MASK_PIPELINES = {Pipeline.RTX, Pipeline.HYBRID, Pipeline.HYBRID_3DGUT}

# Pipelines that populate the mesh-instance-id AOV: the hybrid raygen writes it
# (shader gate HYBRID_ENABLED && NEED_SURFACE_INFO), i.e. HYBRID(3) / HYBRID_3DGUT(5).
# Pure RTX(2) and the raster pipelines leave the AOV at the sentinel.
_INSTANCE_ID_PIPELINES = {Pipeline.HYBRID, Pipeline.HYBRID_3DGUT}

_EXPORT_PLY_HINT = (
    ".pt/.ingp checkpoints are not auto-converted (project decision B1). "
    "Export a .ply inside your 3dgrut environment first:\n"
    "    import torch\n"
    "    from threedgrut.model.model import MixtureOfGaussians\n"
    "    ckpt = torch.load('ckpt_last.pt', weights_only=False)\n"
    "    model = MixtureOfGaussians(ckpt['config'])\n"
    "    model.init_from_checkpoint(ckpt, setup_optimizer=False)\n"
    "    model.export_ply('exported.ply')\n"
    "then pass the .ply to EngineVKGS."
)


class DepthOfFieldVKGS:
    """State mirror of threedgrut_playground.utils.depth_of_field.DepthOfField.

    ``aperture_size``/``focus_z`` map onto the VKGS camera aperture /
    focusDist; ``spp`` becomes the number of accumulated frames
    (sequenceframes) of the render run. Accumulation counters exist only
    for script portability — sampling happens inside the renderer.
    """

    def __init__(self, spp: int = 64, aperture_size: float = 0.1, focus_z: float = 1.0):
        self.aperture_size = aperture_size
        self.focus_z = focus_z
        self.spp = spp
        self.spp_accumulated_for_frame = 1

    def reset_accumulation(self) -> None:
        self.spp_accumulated_for_frame = 1

    def has_more_to_accumulate(self) -> bool:
        return False  # samples accumulate inside the subprocess run


class SPPVKGS:
    """State mirror of threedgrut_playground.utils.spp.SPP.

    ``spp`` maps to accumulated frames of the render run; ``mode`` is kept
    for portability but the renderer uses its own temporal jitter sequence
    (plan gap C3: converged results are comparable, per-sample patterns are
    not reproduced).
    """

    def __init__(self, mode: str = "msaa", spp: int = 4, batch_size: int = 1, device=None):
        self.mode = mode
        self.spp = spp
        self.device = device
        self.batch_size = 1
        self.spp_accumulated_for_frame = 1

    def reset_accumulation(self) -> None:
        self.spp_accumulated_for_frame = self.batch_size

    def has_more_to_accumulate(self) -> bool:
        return False


class EnvironmentVKGS:
    """State mirror of threedgrut_playground.utils.environment.Environment.

    ``set_env`` selects 'Model-Background'/'Black'/'White' or an .hdr from
    the envmap assets folder; ``tonemapper``/``exposure`` are applied in
    Python on the HDR readback (pixel-faithful port of environment.py:
    134-163); ``ibl_intensity`` scales the VKGS IBL. VKGS has no learned
    background model, so 'Model-Background' and 'Black' both mean "no
    environment" (black); 'White' is unsupported and warns.
    """

    FIXED_ENVMAP_OPTIONS = ["Model-Background", "Black", "White"]
    TONEMAPPER_OPTIONS = list(tonemap_mod.TONEMAPPER_OPTIONS)

    def __init__(self, folder: Optional[str] = None):
        self.folder = folder
        self.current_name = "Model-Background"
        self.tonemapper = "None"
        self.ibl_intensity = 1.0
        self.exposure = 0.0
        self._envmap_offset = [0.0, 0.0]
        self.available_envmaps = list(self.FIXED_ENVMAP_OPTIONS)
        if folder is not None:
            self.available_envmaps += sorted(f for f in os.listdir(folder) if f.lower().endswith(".hdr"))

    def set_env(self, env_name: Optional[str] = None) -> None:
        if env_name is None:
            env_name = "Model-Background"
        if env_name not in self.available_envmaps:
            raise ValueError(f"Environment map {self.folder}{os.path.sep}{env_name} not found.")
        if env_name == "White":
            warn_compat(
                "'White' environment is not supported by VKGS (no constant-color "
                "env mode); rendering with no environment instead. Workaround: "
                "author a constant white .hdr and select it from the envmap folder."
            )
        self.current_name = env_name

    @property
    def envmap_offset(self):
        return self._envmap_offset

    @envmap_offset.setter
    def envmap_offset(self, value) -> None:
        warn_compat(
            "envmap_offset is only approximated: the azimuth (phi) component "
            "maps to the VKGS IBL Y rotation, the polar (theta) component to an "
            "X rotation, which is not the same spherical shift for off-axis "
            "content (plan gap C8)."
        )
        self._envmap_offset = [float(value[0]), float(value[1])]

    def is_ignore_envmap(self) -> bool:
        return self.current_name == "Model-Background"

    def hdr_path(self) -> Optional[str]:
        if self.current_name in self.FIXED_ENVMAP_OPTIONS:
            return None
        if self.folder is None:
            return os.path.abspath(self.current_name)
        return os.path.abspath(os.path.join(self.folder, self.current_name))

    def fingerprint(self) -> tuple:
        return (
            self.current_name,
            self.tonemapper,
            float(self.ibl_intensity),
            float(self.exposure),
            tuple(self._envmap_offset),
        )


class EngineVKGS:
    """Engine3DGRUT-compatible interface to VKGS headless rendering.

    Positional arguments mirror ``Engine3DGRUT(gs_object,
    mesh_assets_folder, default_config, envmap_assets_folder)``; the
    keyword-only extras configure the VKGS side:

    - ``executable``: renderer binary (default: vkgs.runner.find_executable
      search at render time, so construction works without a build);
    - ``out_dir``: where .vkgs/.cfg/images/logs are written (default: a
      fresh temp directory);
    - ``size``: output (W, H) when the camera carries no resolution (kaolin
      cameras use their own width/height);
    - ``return_torch``: wrap render() results with torch.from_numpy;
    - ``pipeline``: vkgs.constants.Pipeline (default HYBRID: 3DGS raster
      primaries + RTX secondaries, the closest match to the playground's
      hybrid tracer; use RTX/HYBRID_3DGUT for DoF support).

    Like Engine3DGRUT, construction seeds the scene with one glass Sphere
    primitive (engine.py:1024-1026).
    """

    AVAILABLE_CAMERAS = ["Pinhole", "Fisheye"]
    ANTIALIASING_MODES = ["4x MSAA", "8x MSAA", "16x MSAA", "Quasi-Random (Sobol)"]

    def __init__(
        self,
        gs_object: str,
        mesh_assets_folder: Optional[str] = None,
        default_config: Optional[str] = None,
        envmap_assets_folder: Optional[str] = None,
        *,
        executable: Optional[str] = None,
        out_dir: Optional[str] = None,
        size: Tuple[int, int] = (1920, 1080),
        return_torch: bool = False,
        pipeline: int = Pipeline.HYBRID,
        return_depth: bool = False,
        extra_args: Sequence[str] = (),
    ):
        ext = os.path.splitext(gs_object)[1].lower()
        if ext in (".pt", ".ingp"):
            raise ValueError(f"cannot load {gs_object!r}: {_EXPORT_PLY_HINT}")
        if ext not in (".ply", ".spz", ".splat"):
            raise ValueError(f"unknown object type: {gs_object} (expected .ply/.spz/.splat)")
        if default_config is not None:
            warn_compat(
                "default_config is ignored: VKGS renders the splat file directly "
                "and needs no 3dgrut hydra config"
            )

        self.gs_object = os.path.abspath(gs_object)
        self.scene_name = os.path.splitext(os.path.basename(gs_object))[0]
        self.executable = executable
        self.out_dir = os.path.abspath(out_dir) if out_dir else tempfile.mkdtemp(prefix="vkgs_playground_")
        self.size = (int(size[0]), int(size[1]))
        self.return_torch = bool(return_torch)
        self.pipeline = Pipeline(int(pipeline))
        # When True, render_many also requests the depth buffer and returns it
        # under a "depth" key (NDC [0,1]); off by default to skip the extra I/O.
        self.return_depth = bool(return_depth)
        # Extra CLI args appended verbatim to every render subprocess command
        # (e.g. ["--spirvCacheDir", "/var/cache/vkgs"]).
        self.extra_args = tuple(extra_args)
        # Not applicable in the subprocess model; kept for script portability
        # (3dgrut scripts pass engine.device into add_primitive, which we ignore).
        self.device = None
        self.scene_mog = None  # plan gap A5: no in-process tensor access
        self.frame_id = 0

        # -- Outward-facing settings mirroring Engine3DGRUT (engine.py:992-1018)
        self._camera_type = "Pinhole"
        self.camera_fov = 45.0
        self.use_depth_of_field = False
        self.depth_of_field = DepthOfFieldVKGS(aperture_size=0.01, focus_z=1.0)
        self.use_spp = True
        self._antialiasing_mode = "4x MSAA"
        self.spp = SPPVKGS(mode="msaa", spp=4)
        self.gamma_correction = 1.0
        self.max_pbr_bounces = 15
        self._shadow_min = 0.0
        self._shadow_spp = 128
        self._use_optix_denoiser = False
        self.disable_gaussian_tracing = False

        # Extra VKGS knobs applied verbatim onto scene.renderer at flush,
        # e.g. engine.renderer_overrides["sorting_method"] = 3.
        self.renderer_overrides: Dict[str, object] = {}

        self.environment = EnvironmentVKGS(envmap_assets_folder)

        if ext == ".ply":
            scene_extent = ply_scene_extent(self.gs_object)
        else:
            warn_compat(
                f"no cheap position scan for {ext} files; primitive autoscale "
                "uses the default scene scale (1, 1, 1)"
            )
            scene_extent = (1.0, 1.0, 1.0)
        self.primitives = PrimitivesVKGS(mesh_assets_folder, scene_extent=scene_extent)
        # Engine3DGRUT seeds the scene with a glass sphere (engine.py:1024-1026)
        self.primitives.add_primitive(geometry_type="Sphere", primitive_type=OptixPrimitiveTypes.GLASS)

        self.lights: List[Light] = []
        self._lights_dirty = True
        self.is_materials_dirty = False
        self.last_state: dict = dict(camera=None, rgb=None, opacity=None, rgb_buffer=None)
        self._last_fingerprint: Optional[str] = None
        self._render_index = 0

    # ------------------------------------------------------------ properties

    @property
    def camera_type(self) -> str:
        return self._camera_type

    @camera_type.setter
    def camera_type(self, value: str) -> None:
        if value not in self.AVAILABLE_CAMERAS:
            raise ValueError(f"Unknown camera type: {value}")
        self._camera_type = value

    @property
    def antialiasing_mode(self) -> str:
        return self._antialiasing_mode

    @antialiasing_mode.setter
    def antialiasing_mode(self, value: str) -> None:
        """'4x/8x/16x MSAA' set spp to 4/8/16; 'Quasi-Random (Sobol)' keeps
        the current spp count. Either way VKGS accumulates temporally with
        its own jitter sequence, so per-sample patterns differ from 3dgrut
        MSAA/Sobol (converged images are comparable, plan gap C3)."""
        if value not in self.ANTIALIASING_MODES:
            raise ValueError(f"unknown antialiasing mode: {value!r}; expected one of {self.ANTIALIASING_MODES}")
        warn_compat(
            "antialiasing_mode maps to the accumulated frame count only; VKGS "
            "temporal jitter is not sample-for-sample identical to 3dgrut "
            "MSAA/Sobol patterns"
        )
        if value.endswith("MSAA"):
            self.spp.mode = "msaa"
            self.spp.spp = int(value.split("x")[0])
        else:
            self.spp.mode = "low_discrepancy_seq"
        self._antialiasing_mode = value

    @property
    def max_pbr_bounces(self) -> int:
        return self._max_pbr_bounces

    @max_pbr_bounces.setter
    def max_pbr_bounces(self, value: int) -> None:
        self._max_pbr_bounces = int(value)  # -> renderer.rtx_max_bounces at flush

    @property
    def use_optix_denoiser(self) -> bool:
        return self._use_optix_denoiser

    @use_optix_denoiser.setter
    def use_optix_denoiser(self, value: bool) -> None:
        if value:
            warn_compat(
                "use_optix_denoiser is ignored: VKGS has no OptiX denoiser; "
                "DLSS-RR is the only denoiser and needs a USE_DLSS build + RTX "
                "hardware (plan gap A7). Raise spp for noise-free stochastic "
                "renders instead."
            )
        self._use_optix_denoiser = bool(value)

    @property
    def shadow_min(self) -> float:
        return self._shadow_min

    @shadow_min.setter
    def shadow_min(self, value: float) -> None:
        if float(value) != 0.0:
            warn_compat(
                "shadow_min is unsupported (it belongs to the 3dgrut "
                "shadow-catcher path, plan gap A8); approximate global shadow "
                "strength via renderer_overrides "
                "['particle_shadow_transmittance_threshold' / 'shadows_mode']."
            )
        self._shadow_min = float(value)

    @property
    def shadow_spp(self) -> int:
        return self._shadow_spp

    @shadow_spp.setter
    def shadow_spp(self, value: int) -> None:
        if int(value) != 128:
            warn_compat(
                "shadow_spp has no VKGS equivalent (plan gap A8): soft shadows "
                "converge over the accumulated frames (spp) instead."
            )
        self._shadow_spp = int(value)

    # ---------------------------------------------------------------- lights

    def add_light(self, light: Optional[Light] = None, **kwargs) -> int:
        """Append a light, returning its index (mirrors engine.py:1044-1048).
        AREA lights emit a CompatWarning at flush (emissive-quad approximation)."""
        self.lights.append(light if light is not None else Light(**kwargs))
        self._lights_dirty = True
        return len(self.lights) - 1

    def update_light(self, index: int, **kwargs) -> None:
        light = self.lights[index]
        for key, value in kwargs.items():
            if not hasattr(light, key):
                raise AttributeError(f"Light has no field {key!r}")
            setattr(light, key, value)
        self._lights_dirty = True

    def remove_light(self, index: int) -> None:
        del self.lights[index]
        self._lights_dirty = True

    def clear_lights(self) -> None:
        self.lights.clear()
        self._lights_dirty = True

    # ------------------------------------- shadow catcher / unsupported API

    def load_shadow_catcher(self, path: str, name: Optional[str] = None) -> str:
        """Import a mesh as a shadow catcher (engine.py:629-709 with
        primitive_type=SHADOW_CATCHER). Mapped to the GS shadow mask: the
        catcher mesh is never rendered, its presence only requests
        renderer.gs_shadow_mask + per-light shadow_only copies at flush
        (CompatWarning explains the approximation). Returns the primitive
        name."""
        return self.primitives.load_external_primitive(path, OptixPrimitiveTypes.SHADOW_CATCHER, name=name)

    def raygen(self, camera, use_spp: bool = False):
        raise NotImplementedError(
            "raygen()/RayPack are unsupported: VKGS generates rays in-shader "
            "from the camera model, there is no ray-buffer input path (plan gap "
            "A1). Workaround: stick to Pinhole/Fisheye cameras; for exotic "
            "projections render cube faces and resample in Python."
        )

    # ------------------------------------------------------------- stubs

    def rebuild_bvh(self, scene_mog=None) -> None:
        """No-op: acceleration structures are built inside the render
        subprocess."""

    def invalidate_materials_on_gpu(self) -> None:
        self.is_materials_dirty = True

    def set_primitive_material(self, name: str, material=None) -> None:
        """Override (or, with ``material=None``, clear) the material of one
        primitive instance. The override wins over the primitive_type preset at
        flush, so e.g. ``materials.flat((1, 1, 1))`` gives a white-model
        reference render and ``materials.flat((0, 0, 0))`` a black occluder.
        Native Scene already supports per-instance materials; this exposes it
        through the compat shim."""
        self.primitives.set_primitive_material(name, material)
        self.is_materials_dirty = True

    def get_scene_bounds(self) -> Tuple[float, float, float]:
        """Per-axis extent (max - min) of the splat cloud in the PLY-native
        frame — NOT AABB corners and NOT a metric center. ``(1, 1, 1)``
        fallback for non-.ply inputs (.spz/.splat have no cheap position
        scan). Useful for framing a default camera."""
        return self.primitives.scene_extent

    def has_progressive_effects_to_render(self) -> bool:
        """Always False: render()/render_pass() runs accumulate every sample
        inside one subprocess call (plan gap A2)."""
        return False

    # ------------------------------------------------------------ scene flush

    @property
    def _shadow_catcher_requested(self) -> bool:
        """Lazy state derived from the primitives manager: True when any
        visible SHADOW_CATCHER primitive exists. At flush this turns into
        renderer.gs_shadow_mask + one shadow_only copy per analytic light
        (the catcher mesh itself never enters the scene)."""
        return self.primitives.enabled and any(
            prim.primitive_type == OptixPrimitiveTypes.SHADOW_CATCHER and prim.show
            for prim in self.primitives.objects.values()
        )

    def _build_scene(self) -> Scene:
        """Serialize the lazy state into a vkgs.project.Scene (called on
        every render; also stable API for tests)."""
        scene = Scene()
        scene.renderer.pipeline = int(self.pipeline)
        scene.renderer.rtx_max_bounces = int(self.max_pbr_bounces)

        scene.add_splats(self.gs_object, show=not self.disable_gaussian_tracing)

        show_meshes = self.primitives.enabled
        for prim in self.primitives.objects.values():
            if prim.primitive_type == OptixPrimitiveTypes.SHADOW_CATCHER:
                continue  # never rendered; only requests the GS shadow mask
            # A per-instance override wins over the primitive_type preset — and
            # applies even to PBR/NONE, whose preset is None.
            if prim.material_override is not None:
                material = prim.material_override
            else:
                material = primitive_type_to_material(prim.primitive_type, ior=prim.refractive_index)
            scene.add_mesh(
                prim.path,
                name=prim.name,
                position=prim.position,
                rotation=prim.rotation,
                scale=prim.scale,
                materials=[material] if material is not None else None,
                show=show_meshes and prim.show and prim.primitive_type != OptixPrimitiveTypes.NONE,
            )

        any_light = False
        any_soft = False
        quad_path = None
        light_specs = []  # analytic-light specs, for the shadow-mask copies
        for light in self.lights:
            spec = light_to_vkgs(light)
            if spec["kind"] == "skip":
                continue
            if spec["kind"] == "emissive_quad":
                if quad_path is None:
                    quad_path = ensure_procedural("quad")
                scene.add_mesh(
                    quad_path,
                    name="area light quad",
                    position=spec["position"],
                    rotation=spec["rotation"],
                    scale=spec["scale"],
                    materials=[spec["material"]],
                )
                continue
            any_light = True
            any_soft = any_soft or spec["soft"]
            light_specs.append(spec)
            scene.add_light(
                spec["type"],
                color=spec["color"],
                intensity=spec["intensity"],
                translation=spec["translation"],
                rotation=spec["rotation"],
                radius=spec["radius"] if spec["radius"] > 0 else 1.0,
                # cast_on_gs: one light both illuminates and casts onto the GS
                # shadow mask (no shadow_only twin needed); ignored by the
                # renderer when shadow_only is set (which it never is here).
                cast_on_gs=spec.get("cast_on_gs", False),
            )
        if any_light:
            # 3dgrut always traces shadows when lights exist; soft when any
            # light has angular_radius > 0 (radius from the tan() heuristic).
            scene.renderer.lighting_enabled = True
            scene.renderer.shadows_mode = int(ShadowsMode.SOFT if any_soft else ShadowsMode.HARD)
            if any(spec.get("cast_on_gs") for spec in light_specs):
                # cast_on_gs lights darken the splat emissive via the GS shadow
                # mask, so enable it (RTX pipelines only). One flag replaces the
                # shadow_only twin path below.
                scene.renderer.gs_shadow_mask = True
                if self.pipeline not in _GS_SHADOW_MASK_PIPELINES:
                    warn_compat(
                        "cast_on_gs lights only cast onto the GS shadow mask in the "
                        f"RTX-traced pipelines {sorted(int(p) for p in _GS_SHADOW_MASK_PIPELINES)}; "
                        f"current pipeline {int(self.pipeline)} ignores it."
                    )

        if self._shadow_catcher_requested:
            # SHADOW_CATCHER -> GS shadow mask (see SHADOW_CATCHER_WORKAROUND):
            # every analytic light gets a shadow_only twin that only darkens
            # the mask; the original keeps illuminating the mesh/shaded set.
            # (Future: LightAsset.cast_on_gs=True lets one light do both, which
            # could replace the twin — kept as-is to preserve the hard-mask twin.)
            scene.renderer.gs_shadow_mask = True
            if self.pipeline not in _GS_SHADOW_MASK_PIPELINES:
                warn_compat(
                    f"the GS shadow mask only takes effect in the RTX-traced "
                    f"pipelines {sorted(int(p) for p in _GS_SHADOW_MASK_PIPELINES)}; "
                    f"current pipeline {int(self.pipeline)} ignores it. Set "
                    "engine.pipeline = vkgs.constants.Pipeline.RTX (or HYBRID)."
                )
            for spec in light_specs:
                scene.add_light(
                    spec["type"],
                    name="shadow mask light",
                    color=spec["color"],
                    intensity=spec["intensity"],
                    translation=spec["translation"],
                    rotation=spec["rotation"],
                    # radius=0 -> hard, noise-free mask shadow; soft lights
                    # keep the tan() radius and converge over the spp frames.
                    radius=spec["radius"] if spec["soft"] else 0.0,
                    shadow_only=True,
                )

        env = self.environment
        hdr = env.hdr_path()
        if hdr is not None:
            theta, phi = env.envmap_offset
            scene.set_environment(
                EnvMode.HDR,
                hdr_file=hdr,
                ibl_intensity=env.ibl_intensity,
                ibl_rotation=(float(np.degrees(theta)), float(np.degrees(phi)), 0.0),
            )
        # Fixed options keep EnvMode.NONE (black / model background, see
        # EnvironmentVKGS). In-app tonemapping stays off: tonemap+gamma are
        # applied in Python on the HDR readback.

        for attr, value in self.renderer_overrides.items():
            if not hasattr(scene.renderer, attr):
                raise AttributeError(f"renderer_overrides: RendererSettings has no attribute {attr!r}")
            setattr(scene.renderer, attr, value)
        return scene

    def _state_fingerprint(self) -> str:
        return repr(
            (
                int(self.pipeline),
                self._camera_type,
                float(self.camera_fov),
                bool(self.use_depth_of_field),
                (float(self.depth_of_field.aperture_size), float(self.depth_of_field.focus_z), int(self.depth_of_field.spp)),
                bool(self.use_spp),
                (self.spp.mode, int(self.spp.spp)),
                float(self.gamma_correction),
                int(self.max_pbr_bounces),
                bool(self.disable_gaussian_tracing),
                tuple(self.size),
                self.environment.fingerprint(),
                self.primitives.fingerprint(),
                tuple(tuple(light.to_row()) for light in self.lights),
                tuple(sorted((k, repr(v)) for k, v in self.renderer_overrides.items())),
            )
        )

    # ---------------------------------------------------------------- camera

    def _prepare_camera(self, camera) -> Tuple[Camera, Optional[Tuple[int, int]]]:
        model = CameraModel.FISHEYE if self._camera_type == "Fisheye" else CameraModel.PINHOLE
        cam, size_hint = camera_to_vkgs(camera, fov=self.camera_fov, model=model)
        if self.use_depth_of_field:
            if self.pipeline not in _DOF_PIPELINES:
                warn_compat(
                    f"depth of field only takes effect in pipelines "
                    f"{sorted(int(p) for p in _DOF_PIPELINES)} (RTX / 3DGUT); current "
                    f"pipeline {int(self.pipeline)} ignores it. Set "
                    "engine.pipeline = vkgs.constants.Pipeline.RTX."
                )
            cam = replace(
                cam,
                dof_mode=int(DofMode.FIXED_FOCUS),
                focus_dist=float(self.depth_of_field.focus_z),
                aperture=float(self.depth_of_field.aperture_size),
            )
        return cam, size_hint

    def _frames(self) -> int:
        frames = 1
        if self.use_spp:
            frames = max(frames, int(self.spp.spp))
        if self.use_depth_of_field:
            frames = max(frames, int(self.depth_of_field.spp))
        return frames

    # ------------------------------------------------------------- rendering

    def render(self, camera) -> Dict[str, np.ndarray]:
        """Full-quality render (engine.py:1320-1349). ``camera`` is a vkgs
        Camera, a kaolin Camera, or a 4x4 3dgrut camera-to-world matrix."""
        return self.render_many([camera])[-1]

    def render_pass(self, camera, is_first_pass: bool) -> Dict[str, np.ndarray]:
        """Batch-model port of the progressive API: one full-quality render,
        then cached buffers until is_dirty(camera) (plan gap A2 —
        ``is_first_pass`` cannot restart in-process accumulation)."""
        if self.is_dirty(camera):
            return self.render(camera)
        return {
            "rgb": self.last_state["rgb"],
            "opacity": self.last_state["opacity"],
            "rgb_buffer": self.last_state["rgb_buffer"],
        }

    def render_many(self, cameras: Sequence) -> List[Dict[str, np.ndarray]]:
        """Render several cameras in a single subprocess run (one render+save
        capture pair per camera); returns one buffer dict per camera."""
        prepared = [self._prepare_camera(camera) for camera in cameras]
        cams = [cam for cam, _ in prepared]
        hints = [hint for _, hint in prepared if hint is not None]
        size = hints[0] if hints else self.size
        if any(hint != size for hint in hints):
            warn_compat(
                f"cameras carry different resolutions {sorted(set(hints))}; a "
                f"single run renders at one --size, using {size}"
            )

        scene = self._build_scene()
        scene.set_camera(cams[0])

        out_dir = os.path.join(self.out_dir, f"render_{self._render_index:04d}")
        self._render_index += 1
        buffers = ["main"] + (["depth"] if self.return_depth else [])
        result = facade.render_scene(
            scene,
            cams,
            size=size,
            spp=self._frames(),
            buffers=buffers,
            out_dir=out_dir,
            image_format="raw",  # RGBA32F -> real alpha coverage in channel 3
            executable=self.executable,
            extra_args=self.extra_args,
        )

        buffers: List[Dict[str, np.ndarray]] = []
        for position in range(len(cams)):
            buffers.append(self._postprocess(result, position))

        self._cache_last_state(cams[-1], buffers[-1])
        self.frame_id += self._frames()
        return buffers

    # ------------------------------------------------- per-product AOV masks

    def _mesh_instance_order(self) -> List[str]:
        """Primitive names in the order they become mesh instances in the built
        scene — which equals the C++ TLAS InstanceID and the value written into
        the instance-id AOV. Mirrors _build_scene's mesh add order: every
        non-SHADOW_CATCHER primitive (visible or not), in insertion order.
        Area-light quads (added afterwards) get later ids and are not products."""
        return [
            prim.name
            for prim in self.primitives.objects.values()
            if prim.primitive_type != OptixPrimitiveTypes.SHADOW_CATCHER
        ]

    def render_masks(self, camera, names: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        """Per-product visibility masks from ONE render via the instance-id AOV
        (no per-product paint-white/paint-black passes). Returns
        ``{name: bool (H, W)}`` — True where that product's mesh instance is the
        visible primary surface.

        Requires an RTX-traced hybrid pipeline (HYBRID / HYBRID_3DGUT) with
        surface info; other pipelines leave the AOV at the sentinel and yield
        empty masks (a CompatWarning is emitted). ``names`` selects a subset
        (default: every mesh primitive)."""
        order = self._mesh_instance_order()
        id_of = {name: index for index, name in enumerate(order)}
        want = list(order) if names is None else list(names)
        unknown = [n for n in want if n not in id_of]
        if unknown:
            raise KeyError(f"unknown primitive(s) {unknown}; known: {sorted(id_of)}")
        if self.pipeline not in _INSTANCE_ID_PIPELINES:
            warn_compat(
                f"render_masks needs an RTX-traced hybrid pipeline "
                f"{sorted(int(p) for p in _INSTANCE_ID_PIPELINES)} to populate the "
                f"instance-id AOV; current pipeline {int(self.pipeline)} leaves the "
                "sentinel (all masks empty). Set engine.pipeline = "
                "vkgs.constants.Pipeline.HYBRID."
            )

        cam, size_hint = self._prepare_camera(camera)
        size = size_hint if size_hint is not None else self.size
        scene = self._build_scene()
        scene.set_camera(cam)
        out_dir = os.path.join(self.out_dir, f"masks_{self._render_index:04d}")
        self._render_index += 1
        result = facade.render_scene(
            scene,
            [cam],
            size=size,
            spp=self._frames(),
            buffers=["main", "instance_id"],
            out_dir=out_dir,
            image_format="raw",
            executable=self.executable,
            extra_args=self.extra_args,
        )
        aov = images.load_raw_uint(result.path(0, "instance_id"))  # (H, W) uint32
        return {name: (aov == id_of[name]) for name in want}

    def _postprocess(self, result, position: int) -> Dict[str, np.ndarray]:
        image = result.image(position, "main")
        rgb = tonemap_mod.tonemap(
            image[..., :3], tonemapper=self.environment.tonemapper, exposure=self.environment.exposure
        )
        rgb = tonemap_mod.apply_gamma(rgb, self.gamma_correction)
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)[np.newaxis]  # (1, H, W, 3)

        if image.ndim == 3 and image.shape[2] == 4:  # .png/.raw readback path
            alpha = image[..., 3:4].astype(np.float32)
            if alpha.max() > 1.0:
                alpha = alpha / 255.0
            opacity = alpha[np.newaxis]
        else:  # .hdr carries no alpha; see the module docstring opacity note
            opacity = np.ones(rgb.shape[:3] + (1,), dtype=np.float32)

        rb = {"rgb": rgb, "opacity": opacity, "rgb_buffer": rgb}
        if self.return_depth:
            try:
                depth_img = result.image(position, "depth")
            except KeyError:
                warn_compat("return_depth=True but no depth buffer was saved; skipping depth")
            else:
                # Channel R = depth in NDC [0,1] (NOT metric world distance —
                # un-project with the camera projection for view-space Z);
                # channel G = transmittance.
                rb["depth"] = depth_img[..., 0:1].astype(np.float32)[np.newaxis]  # (1, H, W, 1)
                if depth_img.ndim == 3 and depth_img.shape[2] >= 2:
                    rb["depth_transmittance"] = depth_img[..., 1:2].astype(np.float32)[np.newaxis]

        if self.return_torch:
            try:
                import torch

                rb = {key: torch.from_numpy(value) for key, value in rb.items()}
            except ImportError:
                warn_compat("return_torch=True but torch is not importable; returning numpy arrays")
        return rb

    def _cache_last_state(self, cam: Camera, rb: Dict[str, np.ndarray]) -> None:
        self.last_state["camera"] = cam.to_view_matrix()
        self.last_state["rgb"] = rb["rgb"]
        self.last_state["opacity"] = rb["opacity"]
        self.last_state["rgb_buffer"] = rb["rgb_buffer"]
        self.last_state["canvas_size"] = [rb["rgb"].shape[1], rb["rgb"].shape[2]]
        self._last_fingerprint = self._state_fingerprint()
        self.is_materials_dirty = False
        self._lights_dirty = False
        self.spp.spp_accumulated_for_frame = int(self.spp.spp) + 1
        self.depth_of_field.spp_accumulated_for_frame = int(self.depth_of_field.spp) + 1

    # ----------------------------------------------------------- dirty state

    def did_camera_change(self, camera) -> bool:
        cached = self.last_state.get("camera")
        if cached is None:
            return False
        cam, _ = self._prepare_camera(camera)
        return not np.array_equal(cached, cam.to_view_matrix())

    def has_cached_buffers(self) -> bool:
        return self.last_state.get("rgb") is not None and self.last_state.get("opacity") is not None

    def is_dirty(self, camera) -> bool:
        if self.is_materials_dirty or self._lights_dirty:
            return True
        if not self.has_cached_buffers():
            return True
        if self.did_camera_change(camera):
            return True
        return self._state_fingerprint() != self._last_fingerprint
