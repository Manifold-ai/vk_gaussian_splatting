"""In-memory scene model and .vkgs (version 7) project file round-trip.

The dataclass tree mirrors the JSON emitted by src/vkgs_project_writer.cpp
one-to-one; ``Scene.save`` produces files the C++ reader
(src/vkgs_project_reader.cpp) loads, and ``Scene.load`` reads files written by
either side.

Contract notes (pinned against the C++ sources):
- The reader tolerates missing optional keys but hard-requires
  ``splatSets[].id/.path``, ``splats[].splatSetId``, ``meshAssets[].id/.path``,
  ``meshInstances.items[].meshAssetId``, ``lights.assets[].id`` and
  ``lights.instances[].assetId`` — a missing one throws inside loadProject's
  catch-all and the whole project silently fails to load.
- Asset paths are stored relative to the project file's directory.
- Instance rotations are Euler angles in degrees (GLM quat convention:
  R = Rz@Ry@Rx, see vkgs.camera euler helpers).
- Splat instance transforms are applied only when position AND rotation AND
  scale are all present (reader.cpp:372) — we always emit all three.
- ``lights`` must contain both "assets" and "instances" arrays (v3+ branch).
"""

from __future__ import annotations

import json
import os
import posixpath
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .camera import Camera
from .constants import (
    MIN_SUPPORTED_PROJECT_VERSION,
    PROJECT_FILE_VERSION,
    AttenuationMode,
    BillboardBoundingMode,
    CameraModel,
    EnvMode,
    Format,
    LightType,
    Pipeline,
    Storage,
    VkColorFormat,
)

Vec3 = Tuple[float, float, float]


def _v3(v: Sequence[float]) -> List[float]:
    return [float(v[0]), float(v[1]), float(v[2])]


def _t3(v: Sequence[float]) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


def _get(item: dict, key: str, current):
    """LOAD1-style guarded read."""
    return item[key] if key in item else current


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------


@dataclass
class Material:
    """PBR metallic-roughness material (shaders/shading.h Material factors).

    Defaults match the shader-side struct. ``name`` is only serialized for
    mesh materials. The extended fields (specular*/clearcoat*) are only
    serialized for splat materials, mirroring the C++ writer.
    """

    base_color: Vec3 = (0.7, 0.7, 0.7)
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: Vec3 = (0.0, 0.0, 0.0)
    emissive_strength: float = 1.0
    max_bounces: int = 3
    ior: float = 1.5
    transmission: float = 0.0
    opacity: float = 1.0
    specular_factor: float = 1.0
    specular_color_factor: Vec3 = (1.0, 1.0, 1.0)
    clearcoat_factor: float = 0.0
    clearcoat_roughness: float = 0.0
    name: Optional[str] = None

    @classmethod
    def splat_default(cls) -> "Material":
        """Defaults the reader applies to splat instance materials before
        parsing (vkgs_project_reader.cpp:384-388)."""
        return cls(
            base_color=(0.0, 0.0, 0.0),
            emissive=(1.0, 1.0, 1.0),
            max_bounces=0,
            specular_factor=0.0,
            specular_color_factor=(0.0, 0.0, 0.0),
        )

    def to_json(self, extended: bool) -> dict:
        item = {
            "baseColor": _v3(self.base_color),
            "metallic": float(self.metallic),
            "roughness": float(self.roughness),
            "emissive": _v3(self.emissive),
            "emissiveStrength": float(self.emissive_strength),
            "maxBounces": int(self.max_bounces),
            "ior": float(self.ior),
            "transmission": float(self.transmission),
            "opacity": float(self.opacity),
        }
        if extended:
            item["specularFactor"] = float(self.specular_factor)
            item["specularColorFactor"] = _v3(self.specular_color_factor)
            item["clearcoatFactor"] = float(self.clearcoat_factor)
            item["clearcoatRoughness"] = float(self.clearcoat_roughness)
        elif self.name is not None:
            item["name"] = self.name
        return item

    @classmethod
    def from_json(cls, item: dict, base: Optional["Material"] = None) -> "Material":
        mat = base if base is not None else cls()
        mat.base_color = _t3(_get(item, "baseColor", mat.base_color))
        mat.metallic = float(_get(item, "metallic", mat.metallic))
        mat.roughness = float(_get(item, "roughness", mat.roughness))
        mat.emissive = _t3(_get(item, "emissive", mat.emissive))
        mat.emissive_strength = float(_get(item, "emissiveStrength", mat.emissive_strength))
        mat.max_bounces = int(_get(item, "maxBounces", mat.max_bounces))
        mat.ior = float(_get(item, "ior", mat.ior))
        mat.transmission = float(_get(item, "transmission", mat.transmission))
        mat.opacity = float(_get(item, "opacity", mat.opacity))
        mat.specular_factor = float(_get(item, "specularFactor", mat.specular_factor))
        mat.specular_color_factor = _t3(_get(item, "specularColorFactor", mat.specular_color_factor))
        mat.clearcoat_factor = float(_get(item, "clearcoatFactor", mat.clearcoat_factor))
        mat.clearcoat_roughness = float(_get(item, "clearcoatRoughness", mat.clearcoat_roughness))
        mat.name = item.get("name", mat.name)
        return mat


# --------------------------------------------------------------------------
# Splats
# --------------------------------------------------------------------------


@dataclass
class SplatSet:
    """Splat asset: one loaded .ply/.spz/.splat file (splatSets[] entry)."""

    id: int
    path: str  # absolute path in memory; relativized on save
    storage: int = Storage.BUFFERS
    sh_format: int = Format.UINT8
    rgba_format: int = Format.UINT8


@dataclass
class SplatInstance:
    """splats[] entry referencing a SplatSet by id."""

    splat_set_id: int
    name: str = ""
    show: bool = True
    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Vec3 = (0.0, 0.0, 0.0)  # Euler XYZ degrees
    scale: Vec3 = (1.0, 1.0, 1.0)
    material: Material = field(default_factory=Material.splat_default)

    def to_json(self) -> dict:
        return {
            "splatSetId": int(self.splat_set_id),
            "name": self.name,
            "show": bool(self.show),
            "position": _v3(self.position),
            "rotation": _v3(self.rotation),
            "scale": _v3(self.scale),
            "material": self.material.to_json(extended=True),
        }


# --------------------------------------------------------------------------
# Meshes
# --------------------------------------------------------------------------


@dataclass
class MeshAsset:
    """meshAssets[] entry: one .glb/.gltf/.obj file."""

    id: int
    path: str


@dataclass
class MeshInstance:
    """meshInstances.items[] entry.

    ``materials`` overrides the mesh file's material slots in order; None (or
    a shorter list) keeps the remaining slots as authored in the file.
    """

    mesh_asset_id: int
    name: str = ""
    show: bool = True
    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Vec3 = (0.0, 0.0, 0.0)  # Euler XYZ degrees
    scale: Vec3 = (1.0, 1.0, 1.0)
    materials: List[Material] = field(default_factory=list)

    def to_json(self) -> dict:
        item = {
            "meshAssetId": int(self.mesh_asset_id),
            "name": self.name,
            "show": bool(self.show),
            "position": _v3(self.position),
            "rotation": _v3(self.rotation),
            "scale": _v3(self.scale),
            "materials": [m.to_json(extended=False) for m in self.materials],
        }
        return item


# --------------------------------------------------------------------------
# Lights
# --------------------------------------------------------------------------


@dataclass
class LightAsset:
    """lights.assets[] entry (shared by instances). Defaults match
    LightSourceVk (src/light_manager_vk.h:48)."""

    id: int
    type: int = LightType.POINT
    color: Vec3 = (1.0, 1.0, 1.0)
    intensity: float = 100.0
    range: float = 10.0
    inner_cone_angle: float = 30.0
    outer_cone_angle: float = 45.0
    attenuation_mode: int = AttenuationMode.QUADRATIC
    radius: float = 1.0
    enabled: bool = True
    # gs-shadow light: only darkens the GS shadow mask (renderer.gs_shadow_mask,
    # RTX pipelines), never illuminates (src/light_manager_vk.h:60)
    shadow_only: bool = False
    # both light and shadow: illuminates normally AND also casts onto the GS shadow
    # mask (renderer.gs_shadow_mask); orthogonal to shadow_only (which turns lighting off)
    cast_on_gs: bool = False

    def to_json(self) -> dict:
        return {
            "id": int(self.id),
            "type": int(self.type),
            "color": _v3(self.color),
            "intensity": float(self.intensity),
            "range": float(self.range),
            "innerConeAngle": float(self.inner_cone_angle),
            "outerConeAngle": float(self.outer_cone_angle),
            "attenuationMode": int(self.attenuation_mode),
            "radius": float(self.radius),
            # LightSourceVk.enabled / .shadowOnly / .castOnGs are ints (0/1)
            "enabled": int(bool(self.enabled)),
            "shadowOnly": int(bool(self.shadow_only)),
            "castOnGs": int(bool(self.cast_on_gs)),
        }


@dataclass
class LightInstance:
    """lights.instances[] entry. Note: uses translation/rotation keys (not
    position), unlike splat/mesh instances."""

    asset_id: int
    name: str = ""
    translation: Vec3 = (0.0, 2.0, 0.0)
    # Euler degrees; light direction = R @ (0,0,-1) (light_manager_vk.cpp:465)
    rotation: Vec3 = (0.0, 0.0, 0.0)

    def to_json(self) -> dict:
        return {
            "assetId": int(self.asset_id),
            "name": self.name,
            "translation": _v3(self.translation),
            "rotation": _v3(self.rotation),
        }


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


@dataclass
class SkyAndSun:
    """Procedural sky parameters (environment.skyAndSun)."""

    sun_direction: Vec3 = (0.0, 0.7071068, 0.7071067)
    sun_disk_scale: float = 1.0
    sun_disk_intensity: float = 1.0
    sun_glow_intensity: float = 2.5
    haze: float = 0.1
    redblueshift: float = 0.1
    saturation: float = 1.0
    horizon_height: float = 0.0
    ground_color: Vec3 = (0.4, 0.4, 0.4)
    horizon_blur: float = 0.3
    night_color: Vec3 = (0.0, 0.0, 0.01)

    def to_json(self) -> dict:
        return {
            "sunDirection": _v3(self.sun_direction),
            "sunDiskScale": float(self.sun_disk_scale),
            "sunDiskIntensity": float(self.sun_disk_intensity),
            "sunGlowIntensity": float(self.sun_glow_intensity),
            "haze": float(self.haze),
            "redblueshift": float(self.redblueshift),
            "saturation": float(self.saturation),
            "horizonHeight": float(self.horizon_height),
            "groundColor": _v3(self.ground_color),
            "horizonBlur": float(self.horizon_blur),
            "nightColor": _v3(self.night_color),
        }

    @classmethod
    def from_json(cls, item: dict) -> "SkyAndSun":
        s = cls()
        s.sun_direction = _t3(_get(item, "sunDirection", s.sun_direction))
        s.sun_disk_scale = float(_get(item, "sunDiskScale", s.sun_disk_scale))
        s.sun_disk_intensity = float(_get(item, "sunDiskIntensity", s.sun_disk_intensity))
        s.sun_glow_intensity = float(_get(item, "sunGlowIntensity", s.sun_glow_intensity))
        s.haze = float(_get(item, "haze", s.haze))
        s.redblueshift = float(_get(item, "redblueshift", s.redblueshift))
        s.saturation = float(_get(item, "saturation", s.saturation))
        s.horizon_height = float(_get(item, "horizonHeight", s.horizon_height))
        s.ground_color = _t3(_get(item, "groundColor", s.ground_color))
        s.horizon_blur = float(_get(item, "horizonBlur", s.horizon_blur))
        s.night_color = _t3(_get(item, "nightColor", s.night_color))
        return s


@dataclass
class Environment:
    """environment section. ``hdr_file`` is absolute in memory, relativized
    on save. The C++ reader only loads the HDR when mode == HDR."""

    mode: int = EnvMode.NONE
    enabled: bool = True
    resolution: Tuple[int, int] = (2048, 1024)
    sky_and_sun: SkyAndSun = field(default_factory=SkyAndSun)
    hdr_file: str = ""
    ibl_intensity: float = 1.0
    ibl_rotation: Vec3 = (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# Renderer / globals / settings / tonemapping
# --------------------------------------------------------------------------


@dataclass
class RendererSettings:
    """renderer section. Field defaults match the C++ parameter defaults
    (src/parameters.h, shaders/shaderio.h FrameInfo)."""

    vsync: bool = False
    pipeline: int = Pipeline.MESH
    max_sh_degree: int = 3
    opacity_gaussian_disabled: bool = False
    show_sh_only: bool = False
    visualize: int = 0
    wireframe: bool = False
    cpu_lazy_sort: bool = True
    dist_shader_workgroup_size: int = 256
    fragment_barycentric: bool = False
    frustum_culling: int = 1  # FRUSTUM_CULLING_AT_DIST
    size_culling: int = 0
    size_culling_min_pixels: float = 1.0
    mesh_shader_workgroup_size: int = 32
    point_cloud_mode_enabled: bool = False
    sorting_method: int = 0  # SORTING_GPU_SYNC_RADIX
    temporal_sampling: bool = False
    temporal_samples_count: int = 1000
    kernel_adaptive_clamping: bool = True
    kernel_degree: int = 2
    kernel_min_response: float = 0.0113
    particle_samples_per_pass: int = 18
    rtx_trace_strategy: int = 2  # RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT
    normal_method: int = 0
    thin_particle_threshold: float = 1e-6
    firefly_clamp_threshold: float = 13.0
    # DLSS (only meaningful for USE_DLSS builds; omitted from JSON when None)
    dlss_min_radiance_threshold: Optional[float] = None
    dlss_enabled: Optional[bool] = None
    dlss_size_mode: Optional[int] = None
    lighting_enabled: bool = False
    shadows_mode: int = 0
    # GS shadow mask (src/parameters.h:185-188): analytic-light shadows cast
    # onto the splat emissive; RTX pipelines (2/3/5) only, driven by
    # shadow_only lights. gs_shadow_mask_min is the shadow floor (0 = black,
    # 1 = no visible shadow); gs_shadow_mask_from_particles lets particles
    # occlude the mask rays too (mesh instances are always occluders).
    gs_shadow_mask: bool = False
    gs_shadow_mask_min: float = 0.2
    gs_shadow_mask_from_particles: bool = False
    force_surfel: bool = False
    color_format: int = VkColorFormat.R32G32B32A32_SFLOAT  # raw VkFormat value
    rtx_max_bounces: int = 3
    rtx_secondary_ray_offset: float = 0.001
    splat_set_composite_transmittance: float = 0.1
    temporal_sampling_mode: int = 0  # TEMPORAL_SAMPLING_AUTO
    alpha_cull_threshold: float = 1.0 / 255.0
    splat_scale: float = 1.0
    alpha_clamp: float = 0.99
    min_transmittance: float = 0.01
    max_passes: int = 200
    particle_depth: int = 1  # PARTICLE_DEPTH_ELLIPSOID
    billboard_frustum_culling: bool = True
    shorten_ray: bool = True
    quantize_mesh_payload: bool = True
    particle_shadow_offset: float = 0.2
    particle_shadow_transmittance_threshold: float = 0.8
    particle_shadow_color_strength: float = 0.0
    particle_emissive_ao_enabled: bool = False
    particle_emissive_ao_radius: float = 0.05
    particle_emissive_ao_strength: float = 1.0
    depth_iso_threshold_rtx: float = 0.7
    depth_iso_threshold: float = 0.7
    covariance_dilation: float = 0.3
    ms_antialiasing: bool = False
    quantize_normals: bool = True
    ftb_sync_mode: int = 0
    extent_projection: int = 1  # EXTENT_CONIC

    # (python attr, json key, type coercion)
    _FIELDS = [
        ("vsync", "vsync", bool),
        ("pipeline", "pipeline", int),
        ("max_sh_degree", "maxShDegree", int),
        ("opacity_gaussian_disabled", "opacityGaussianDisabled", bool),
        ("show_sh_only", "showShOnly", bool),
        ("visualize", "visualize", int),
        ("wireframe", "wireframe", bool),
        ("cpu_lazy_sort", "cpuLazySort", bool),
        ("dist_shader_workgroup_size", "distShaderWorkgroupSize", int),
        ("fragment_barycentric", "fragmentBarycentric", bool),
        ("frustum_culling", "frustumCulling", int),
        ("size_culling", "sizeCulling", int),
        ("size_culling_min_pixels", "sizeCullingMinPixels", float),
        ("mesh_shader_workgroup_size", "meshShaderWorkgroupSize", int),
        ("point_cloud_mode_enabled", "pointCloudModeEnabled", bool),
        ("sorting_method", "sortingMethod", int),
        ("temporal_sampling", "temporalSampling", bool),
        ("temporal_samples_count", "temporalSamplesCount", int),
        ("kernel_adaptive_clamping", "kernelAdaptiveClamping", bool),
        ("kernel_degree", "kernelDegree", int),
        ("kernel_min_response", "kernelMinResponse", float),
        ("particle_samples_per_pass", "particleSamplesPerPass", int),
        ("rtx_trace_strategy", "rtxTraceStrategy", int),
        ("normal_method", "normalMethod", int),
        ("thin_particle_threshold", "thinParticleThreshold", float),
        ("firefly_clamp_threshold", "fireflyClampThreshold", float),
        ("dlss_min_radiance_threshold", "dlssMinRadianceThreshold", float),
        ("dlss_enabled", "dlssEnabled", bool),
        ("dlss_size_mode", "dlssSizeMode", int),
        ("lighting_enabled", "lightingEnabled", bool),
        ("shadows_mode", "shadowsMode", int),
        ("gs_shadow_mask", "gsShadowMask", bool),
        ("gs_shadow_mask_min", "gsShadowMaskMin", float),
        ("gs_shadow_mask_from_particles", "gsShadowMaskFromParticles", bool),
        ("force_surfel", "forceSurfel", bool),
        ("color_format", "colorFormat", int),
        ("rtx_max_bounces", "rtxMaxBounces", int),
        ("rtx_secondary_ray_offset", "rtxSecondaryRayOffset", float),
        ("splat_set_composite_transmittance", "splatSetCompositeTransmittance", float),
        ("temporal_sampling_mode", "temporalSamplingMode", int),
        ("alpha_cull_threshold", "alphaCullThreshold", float),
        ("splat_scale", "splatScale", float),
        ("alpha_clamp", "alphaClamp", float),
        ("min_transmittance", "minTransmittance", float),
        ("max_passes", "maxPasses", int),
        ("particle_depth", "particleDepth", int),
        ("billboard_frustum_culling", "billboardFrustumCulling", bool),
        ("shorten_ray", "shortenRay", bool),
        ("quantize_mesh_payload", "quantizeMeshPayload", bool),
        ("particle_shadow_offset", "particleShadowOffset", float),
        ("particle_shadow_transmittance_threshold", "particleShadowTransmittanceThreshold", float),
        ("particle_shadow_color_strength", "particleShadowColorStrength", float),
        ("particle_emissive_ao_enabled", "particleEmissiveAoEnabled", bool),
        ("particle_emissive_ao_radius", "particleEmissiveAoRadius", float),
        ("particle_emissive_ao_strength", "particleEmissiveAoStrength", float),
        ("depth_iso_threshold_rtx", "depthIsoThresholdRTX", float),
        ("depth_iso_threshold", "depthIsoThreshold", float),
        ("covariance_dilation", "covarianceDilation", float),
        ("ms_antialiasing", "msAntialiasing", bool),
        ("quantize_normals", "quantizeNormals", bool),
        ("ftb_sync_mode", "ftbSyncMode", int),
        ("extent_projection", "extentProjection", int),
    ]

    def to_json(self) -> dict:
        item = {}
        for attr, key, cast in self._FIELDS:
            value = getattr(self, attr)
            if value is None:
                continue
            item[key] = cast(value)
        return item

    @classmethod
    def from_json(cls, item: dict) -> "RendererSettings":
        rs = cls()
        for attr, key, cast in cls._FIELDS:
            if key in item:
                setattr(rs, attr, cast(item[key]))
        return rs


@dataclass
class SplatsGlobals:
    """splatsGlobals section (prmData / prmRtxData defaults)."""

    sh_format: int = Format.UINT8
    rgba_format: int = Format.UINT8
    compress_blas: bool = True
    use_aabbs: bool = False
    use_spheres: bool = False
    use_tlas_instances: bool = True
    billboard_bounding_mode: int = BillboardBoundingMode.FITTED

    def to_json(self) -> dict:
        return {
            "shFormat": int(self.sh_format),
            "rgbaFormat": int(self.rgba_format),
            "compressBlas": bool(self.compress_blas),
            "useAABBs": bool(self.use_aabbs),
            "useSpheres": bool(self.use_spheres),
            "useTlasInstances": bool(self.use_tlas_instances),
            "billboardBoundingMode": int(self.billboard_bounding_mode),
        }

    @classmethod
    def from_json(cls, item: dict) -> "SplatsGlobals":
        g = cls()
        g.sh_format = int(_get(item, "shFormat", g.sh_format))
        g.rgba_format = int(_get(item, "rgbaFormat", g.rgba_format))
        g.compress_blas = bool(_get(item, "compressBlas", g.compress_blas))
        g.use_aabbs = bool(_get(item, "useAABBs", g.use_aabbs))
        g.use_spheres = bool(_get(item, "useSpheres", g.use_spheres))
        g.use_tlas_instances = bool(_get(item, "useTlasInstances", g.use_tlas_instances))
        g.billboard_bounding_mode = int(_get(item, "billboardBoundingMode", g.billboard_bounding_mode))
        return g


@dataclass
class Settings:
    """settings section (UI/navigation preferences; harmless headless)."""

    navigation_mode: int = 0
    navigation_speed: float = 1.0
    navigation_transition: float = 0.5
    auto_play_presets: bool = False
    helpers_show: bool = False
    snap_enabled: bool = False
    snap_translate: float = 0.1
    snap_rotate: float = 5.0
    snap_scale: float = 0.1
    grid_show: bool = False
    light_proxies_show: bool = False
    summary_overlay_show: bool = True

    def to_json(self) -> dict:
        return {
            "navigation": {
                "mode": int(self.navigation_mode),
                "speed": float(self.navigation_speed),
                "transition": float(self.navigation_transition),
                "autoPlay": bool(self.auto_play_presets),
            },
            "transformHelpers": {
                "show": bool(self.helpers_show),
                "snapEnabled": bool(self.snap_enabled),
                "snapTranslate": float(self.snap_translate),
                "snapRotate": float(self.snap_rotate),
                "snapScale": float(self.snap_scale),
            },
            "grid": {"show": bool(self.grid_show)},
            "lightProxies": {"show": bool(self.light_proxies_show)},
            "summaryOverlay": {"show": bool(self.summary_overlay_show)},
        }

    @classmethod
    def from_json(cls, item: dict) -> "Settings":
        s = cls()
        nav = item.get("navigation", {})
        s.navigation_mode = int(_get(nav, "mode", s.navigation_mode))
        s.navigation_speed = float(_get(nav, "speed", s.navigation_speed))
        s.navigation_transition = float(_get(nav, "transition", s.navigation_transition))
        s.auto_play_presets = bool(_get(nav, "autoPlay", s.auto_play_presets))
        th = item.get("transformHelpers", {})
        s.helpers_show = bool(_get(th, "show", s.helpers_show))
        s.snap_enabled = bool(_get(th, "snapEnabled", s.snap_enabled))
        s.snap_translate = float(_get(th, "snapTranslate", s.snap_translate))
        s.snap_rotate = float(_get(th, "snapRotate", s.snap_rotate))
        s.snap_scale = float(_get(th, "snapScale", s.snap_scale))
        s.grid_show = bool(_get(item.get("grid", {}), "show", s.grid_show))
        s.light_proxies_show = bool(_get(item.get("lightProxies", {}), "show", s.light_proxies_show))
        s.summary_overlay_show = bool(_get(item.get("summaryOverlay", {}), "show", s.summary_overlay_show))
        return s


@dataclass
class Tonemapping:
    """tonemapping section (shaderio::TonemapperData; int flags stay ints)."""

    is_active: bool = False
    method: int = 0
    exposure: float = 1.0
    temperature: float = 6500.0
    tint: float = 0.0
    contrast: float = 1.0
    brightness: float = 1.0
    saturation: float = 1.0
    vignette: float = 0.0
    vibrance: float = 0.0
    shadow_bias: float = 0.0
    midtone_bias: float = 0.0
    highlight_bias: float = 0.0
    cool_color: Vec3 = (1.0, 1.0, 1.0)
    warm_color: Vec3 = (1.0, 1.0, 1.0)
    split_balance: float = 0.0
    auto_exposure: bool = False
    auto_exposure_speed: float = 5.0
    ev_min_value: float = -5.0
    ev_max_value: float = 10.0
    enable_center_metering: bool = False
    center_metering_size: float = 0.5
    average_mode: int = 1
    dither: bool = True

    def to_json(self) -> dict:
        return {
            "isActive": int(bool(self.is_active)),
            "method": int(self.method),
            "exposure": float(self.exposure),
            "temperature": float(self.temperature),
            "tint": float(self.tint),
            "contrast": float(self.contrast),
            "brightness": float(self.brightness),
            "saturation": float(self.saturation),
            "vignette": float(self.vignette),
            "vibrance": float(self.vibrance),
            "shadowBias": float(self.shadow_bias),
            "midtoneBias": float(self.midtone_bias),
            "highlightBias": float(self.highlight_bias),
            "coolColor": _v3(self.cool_color),
            "warmColor": _v3(self.warm_color),
            "splitBalance": float(self.split_balance),
            "autoExposure": int(bool(self.auto_exposure)),
            "autoExposureSpeed": float(self.auto_exposure_speed),
            "evMinValue": float(self.ev_min_value),
            "evMaxValue": float(self.ev_max_value),
            "enableCenterMetering": int(bool(self.enable_center_metering)),
            "centerMeteringSize": float(self.center_metering_size),
            "averageMode": int(self.average_mode),
            "dither": int(bool(self.dither)),
        }

    @classmethod
    def from_json(cls, item: dict) -> "Tonemapping":
        t = cls()
        t.is_active = bool(_get(item, "isActive", t.is_active))
        t.method = int(_get(item, "method", t.method))
        t.exposure = float(_get(item, "exposure", t.exposure))
        t.temperature = float(_get(item, "temperature", t.temperature))
        t.tint = float(_get(item, "tint", t.tint))
        t.contrast = float(_get(item, "contrast", t.contrast))
        t.brightness = float(_get(item, "brightness", t.brightness))
        t.saturation = float(_get(item, "saturation", t.saturation))
        t.vignette = float(_get(item, "vignette", t.vignette))
        t.vibrance = float(_get(item, "vibrance", t.vibrance))
        t.shadow_bias = float(_get(item, "shadowBias", t.shadow_bias))
        t.midtone_bias = float(_get(item, "midtoneBias", t.midtone_bias))
        t.highlight_bias = float(_get(item, "highlightBias", t.highlight_bias))
        t.cool_color = _t3(_get(item, "coolColor", t.cool_color))
        t.warm_color = _t3(_get(item, "warmColor", t.warm_color))
        t.split_balance = float(_get(item, "splitBalance", t.split_balance))
        t.auto_exposure = bool(_get(item, "autoExposure", t.auto_exposure))
        t.auto_exposure_speed = float(_get(item, "autoExposureSpeed", t.auto_exposure_speed))
        t.ev_min_value = float(_get(item, "evMinValue", t.ev_min_value))
        t.ev_max_value = float(_get(item, "evMaxValue", t.ev_max_value))
        t.enable_center_metering = bool(_get(item, "enableCenterMetering", t.enable_center_metering))
        t.center_metering_size = float(_get(item, "centerMeteringSize", t.center_metering_size))
        t.average_mode = int(_get(item, "averageMode", t.average_mode))
        t.dither = bool(_get(item, "dither", t.dither))
        return t


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------


def _relativize(project_dir: str, path: str) -> str:
    """Path stored in the file: relative to the project dir, POSIX separators
    (mirrors getRelativePath in vkgs_project_writer.cpp)."""
    if not path:
        return ""
    try:
        rel = os.path.relpath(os.path.abspath(path), project_dir)
    except ValueError:
        # Different drive on Windows: keep absolute
        return path.replace(os.sep, "/")
    return rel.replace(os.sep, "/")


def _absolutize(project_dir: str, rel: str) -> str:
    if not rel:
        return ""
    rel_os = rel.replace("\\", "/")
    if posixpath.isabs(rel_os) or os.path.isabs(rel_os):
        return os.path.normpath(rel_os)
    return os.path.normpath(os.path.join(project_dir, rel_os))


class Scene:
    """Scriptable scene: build with add_* methods, then ``save()`` to .vkgs.

    Camera preset indices returned by :meth:`add_camera_preset` map directly
    to ``--activateCameraPreset`` after the project loads (the C++ reader
    clears presets and recreates them from the ``cameras`` array in order).
    """

    def __init__(self) -> None:
        self.renderer = RendererSettings()
        self.splats_globals = SplatsGlobals()
        self.splat_sets: List[SplatSet] = []
        self.splat_instances: List[SplatInstance] = []
        self.mesh_assets: List[MeshAsset] = []
        self.mesh_instances: List[MeshInstance] = []
        self.light_assets: List[LightAsset] = []
        self.light_instances: List[LightInstance] = []
        self.camera: Camera = Camera()
        self.camera_presets: List[Camera] = []
        self.environment = Environment()
        self.settings = Settings()
        self.tonemapping = Tonemapping()

    # ------------------------------------------------------------- splats

    def add_splats(
        self,
        path: str,
        *,
        name: Optional[str] = None,
        position: Vec3 = (0.0, 0.0, 0.0),
        rotation: Vec3 = (0.0, 0.0, 0.0),
        scale: Union[float, Vec3] = (1.0, 1.0, 1.0),
        material: Optional[Material] = None,
        show: bool = True,
        sh_format: Optional[int] = None,
        rgba_format: Optional[int] = None,
    ) -> SplatInstance:
        """Add a splat file instance. The asset is deduplicated by absolute
        path, so adding the same file twice shares one GPU copy."""
        abspath = os.path.abspath(path)
        asset = next((s for s in self.splat_sets if s.path == abspath), None)
        if asset is None:
            asset = SplatSet(
                id=len(self.splat_sets),
                path=abspath,
                sh_format=self.splats_globals.sh_format if sh_format is None else int(sh_format),
                rgba_format=self.splats_globals.rgba_format if rgba_format is None else int(rgba_format),
            )
            self.splat_sets.append(asset)

        instance = SplatInstance(
            splat_set_id=asset.id,
            name=name or f"Splat set {len(self.splat_instances)} - {os.path.basename(path)}",
            show=show,
            position=_t3(position),
            rotation=_t3(rotation),
            scale=_t3((scale, scale, scale)) if isinstance(scale, (int, float)) else _t3(scale),
            material=material or Material.splat_default(),
        )
        self.splat_instances.append(instance)
        return instance

    # ------------------------------------------------------------- meshes

    def add_mesh(
        self,
        path: str,
        *,
        name: Optional[str] = None,
        position: Vec3 = (0.0, 0.0, 0.0),
        rotation: Vec3 = (0.0, 0.0, 0.0),
        scale: Union[float, Vec3] = (1.0, 1.0, 1.0),
        materials: Optional[Sequence[Material]] = None,
        show: bool = True,
    ) -> MeshInstance:
        """Add a mesh (.glb/.gltf/.obj) instance; asset deduplicated by path.

        ``materials`` overrides the file's material slots in order. An empty /
        omitted list keeps the materials authored in the mesh file.
        """
        abspath = os.path.abspath(path)
        asset = next((m for m in self.mesh_assets if m.path == abspath), None)
        if asset is None:
            asset = MeshAsset(id=len(self.mesh_assets), path=abspath)
            self.mesh_assets.append(asset)

        instance = MeshInstance(
            mesh_asset_id=asset.id,
            name=name or f"Model {len(self.mesh_instances)}",
            show=show,
            position=_t3(position),
            rotation=_t3(rotation),
            scale=_t3((scale, scale, scale)) if isinstance(scale, (int, float)) else _t3(scale),
            materials=list(materials) if materials else [],
        )
        self.mesh_instances.append(instance)
        return instance

    # ------------------------------------------------------------- lights

    def add_light(
        self,
        type: int = LightType.POINT,
        *,
        name: Optional[str] = None,
        color: Vec3 = (1.0, 1.0, 1.0),
        intensity: float = 100.0,
        translation: Vec3 = (0.0, 2.0, 0.0),
        rotation: Vec3 = (0.0, 0.0, 0.0),
        range: float = 10.0,
        radius: float = 1.0,
        inner_cone_angle: float = 30.0,
        outer_cone_angle: float = 45.0,
        attenuation_mode: int = AttenuationMode.QUADRATIC,
        enabled: bool = True,
        shadow_only: bool = False,
        cast_on_gs: bool = False,
    ) -> LightInstance:
        """Create a light (asset + one instance) and return the instance.

        ``shadow_only=True`` makes a gs-shadow light: it only darkens the GS
        shadow mask (renderer.gs_shadow_mask, RTX pipelines 2/3/5) and never
        illuminates anything; use radius=0 for noise-free hard mask shadows.
        ``cast_on_gs=True`` keeps the light illuminating normally but ALSO casts
        onto the GS shadow mask (one light that both shadows meshes and darkens
        splats); requires renderer.gs_shadow_mask on. Ignored when shadow_only.
        """
        asset = LightAsset(
            id=len(self.light_assets),
            type=int(type),
            color=_t3(color),
            intensity=float(intensity),
            range=float(range),
            inner_cone_angle=float(inner_cone_angle),
            outer_cone_angle=float(outer_cone_angle),
            attenuation_mode=int(attenuation_mode),
            radius=float(radius),
            enabled=enabled,
            shadow_only=bool(shadow_only),
            cast_on_gs=bool(cast_on_gs),
        )
        self.light_assets.append(asset)
        return self.add_light_instance(asset, name=name, translation=translation, rotation=rotation)

    def add_light_instance(
        self,
        asset: LightAsset,
        *,
        name: Optional[str] = None,
        translation: Vec3 = (0.0, 2.0, 0.0),
        rotation: Vec3 = (0.0, 0.0, 0.0),
    ) -> LightInstance:
        """Add another instance sharing an existing light asset."""
        instance = LightInstance(
            asset_id=asset.id,
            name=name or f"Light {len(self.light_instances)}",
            translation=_t3(translation),
            rotation=_t3(rotation),
        )
        self.light_instances.append(instance)
        return instance

    # -------------------------------------------------- environment / misc

    def set_environment(
        self,
        mode: int = EnvMode.SKY,
        *,
        hdr_file: str = "",
        ibl_intensity: float = 1.0,
        ibl_rotation: Vec3 = (0.0, 0.0, 0.0),
        sun_direction: Optional[Vec3] = None,
        **sky_params,
    ) -> Environment:
        env = self.environment
        env.mode = int(mode)
        env.enabled = True
        env.hdr_file = os.path.abspath(hdr_file) if hdr_file else ""
        env.ibl_intensity = float(ibl_intensity)
        env.ibl_rotation = _t3(ibl_rotation)
        if sun_direction is not None:
            env.sky_and_sun.sun_direction = _t3(sun_direction)
        for key, value in sky_params.items():
            if not hasattr(env.sky_and_sun, key):
                raise TypeError(f"unknown sky parameter: {key}")
            setattr(env.sky_and_sun, key, value)
        return env

    def set_tonemapping(self, active: bool = True, **kwargs) -> Tonemapping:
        self.tonemapping.is_active = active
        for key, value in kwargs.items():
            if not hasattr(self.tonemapping, key):
                raise TypeError(f"unknown tonemapping parameter: {key}")
            setattr(self.tonemapping, key, value)
        return self.tonemapping

    def set_camera(self, camera: Camera) -> None:
        self.camera = camera

    def add_camera_preset(self, camera: Camera) -> int:
        """Append a camera preset; returns its --activateCameraPreset index."""
        self.camera_presets.append(camera)
        return len(self.camera_presets) - 1

    # ------------------------------------------------------------ save/load

    def to_json(self, project_dir: str) -> dict:
        """Serialize to the .vkgs JSON tree; asset paths become relative to
        ``project_dir``."""
        data: dict = {"version": PROJECT_FILE_VERSION}

        data["renderer"] = self.renderer.to_json()
        data["camera"] = self.camera.to_json()
        data["cameras"] = [c.to_json() for c in self.camera_presets]

        data["lights"] = {
            "nextNamingNumber": len(self.light_instances) + 1,
            "assets": [a.to_json() for a in self.light_assets],
            "instances": [i.to_json() for i in self.light_instances],
        }

        data["splatsGlobals"] = self.splats_globals.to_json()

        data["splatSets"] = [
            {
                "id": int(s.id),
                "path": _relativize(project_dir, s.path),
                "storage": int(s.storage),
                "shFormat": int(s.sh_format),
                "rgbaFormat": int(s.rgba_format),
            }
            for s in self.splat_sets
        ]
        data["splats"] = [i.to_json() for i in self.splat_instances]

        data["meshAssets"] = [
            {"id": int(m.id), "path": _relativize(project_dir, m.path)} for m in self.mesh_assets
        ]
        data["meshInstances"] = {
            "nextNamingNumber": len(self.mesh_instances) + 1,
            "items": [i.to_json() for i in self.mesh_instances],
        }

        env = self.environment
        data["environment"] = {
            "mode": int(env.mode),
            "enabled": bool(env.enabled),
            "resolution": [int(env.resolution[0]), int(env.resolution[1])],
            "skyAndSun": env.sky_and_sun.to_json(),
            "ibl": {
                "file": _relativize(project_dir, env.hdr_file) if env.hdr_file else "",
                "intensity": float(env.ibl_intensity),
                "rotation": _v3(env.ibl_rotation),
            },
        }

        data["settings"] = self.settings.to_json()
        data["tonemapping"] = self.tonemapping.to_json()
        return data

    def save(self, path: str) -> str:
        """Write the .vkgs project file and return its absolute path."""
        abspath = os.path.abspath(path)
        if not abspath.lower().endswith(".vkgs"):
            abspath = os.path.splitext(abspath)[0] + ".vkgs"
        project_dir = os.path.dirname(abspath)
        os.makedirs(project_dir, exist_ok=True)
        data = self.to_json(project_dir)
        with open(abspath, "w", encoding="utf-8") as f:
            # nlohmann::json emits alphabetically sorted keys with 4-space
            # indent; match it so C++/Python outputs diff cleanly.
            json.dump(data, f, indent=4, sort_keys=True)
            f.write("\n")
        return abspath

    @classmethod
    def load(cls, path: str) -> "Scene":
        """Parse a .vkgs project file (version >= 5) back into a Scene."""
        abspath = os.path.abspath(path)
        project_dir = os.path.dirname(abspath)
        with open(abspath, "r", encoding="utf-8") as f:
            data = json.load(f)

        version = int(data.get("version", 0))
        if version < MIN_SUPPORTED_PROJECT_VERSION:
            raise ValueError(
                f"{path}: project version {version} < {MIN_SUPPORTED_PROJECT_VERSION}; "
                "open and re-save it with the vk_gaussian_splatting app to migrate."
            )

        scene = cls()

        if "renderer" in data:
            scene.renderer = RendererSettings.from_json(data["renderer"])
        if "camera" in data:
            scene.camera = Camera.from_json(data["camera"])
        for item in data.get("cameras", []):
            scene.camera_presets.append(Camera.from_json(item))

        for item in data.get("splatSets", []):
            scene.splat_sets.append(
                SplatSet(
                    id=int(item["id"]),
                    path=_absolutize(project_dir, item["path"]),
                    storage=int(item.get("storage", Storage.BUFFERS)),
                    sh_format=int(item.get("shFormat", item.get("format", Format.UINT8))),
                    rgba_format=int(item.get("rgbaFormat", Format.UINT8)),
                )
            )
        for item in data.get("splats", []):
            inst = SplatInstance(splat_set_id=int(item["splatSetId"]))
            inst.name = item.get("name", "")
            inst.show = bool(item.get("show", True))
            inst.position = _t3(item.get("position", inst.position))
            inst.rotation = _t3(item.get("rotation", inst.rotation))
            inst.scale = _t3(item.get("scale", inst.scale))
            if "material" in item:
                inst.material = Material.from_json(item["material"], base=Material.splat_default())
            scene.splat_instances.append(inst)

        for item in data.get("meshAssets", []):
            scene.mesh_assets.append(MeshAsset(id=int(item["id"]), path=_absolutize(project_dir, item["path"])))
        mesh_instances = data.get("meshInstances", {})
        mesh_items = mesh_instances.get("items", []) if isinstance(mesh_instances, dict) else mesh_instances
        for item in mesh_items:
            inst = MeshInstance(mesh_asset_id=int(item["meshAssetId"]))
            inst.name = item.get("name", "")
            inst.show = bool(item.get("show", True))
            inst.position = _t3(item.get("position", inst.position))
            inst.rotation = _t3(item.get("rotation", inst.rotation))
            inst.scale = _t3(item.get("scale", inst.scale))
            inst.materials = [Material.from_json(m) for m in item.get("materials", [])]
            scene.mesh_instances.append(inst)

        lights = data.get("lights", {})
        if isinstance(lights, dict):
            for item in lights.get("assets", []):
                asset = LightAsset(id=int(item["id"]))
                asset.type = int(_get(item, "type", asset.type))
                asset.color = _t3(_get(item, "color", asset.color))
                asset.intensity = float(_get(item, "intensity", asset.intensity))
                asset.range = float(_get(item, "range", asset.range))
                asset.inner_cone_angle = float(_get(item, "innerConeAngle", asset.inner_cone_angle))
                asset.outer_cone_angle = float(_get(item, "outerConeAngle", asset.outer_cone_angle))
                asset.attenuation_mode = int(_get(item, "attenuationMode", asset.attenuation_mode))
                asset.radius = float(_get(item, "radius", asset.radius))
                asset.enabled = bool(_get(item, "enabled", asset.enabled))
                asset.shadow_only = bool(_get(item, "shadowOnly", asset.shadow_only))
                asset.cast_on_gs = bool(_get(item, "castOnGs", asset.cast_on_gs))
                scene.light_assets.append(asset)
            for item in lights.get("instances", []):
                inst = LightInstance(asset_id=int(item["assetId"]))
                inst.name = item.get("name", "")
                inst.translation = _t3(item.get("translation", inst.translation))
                inst.rotation = _t3(item.get("rotation", inst.rotation))
                scene.light_instances.append(inst)

        if "environment" in data:
            env_item = data["environment"]
            env = scene.environment
            env.mode = int(_get(env_item, "mode", env.mode))
            env.enabled = bool(_get(env_item, "enabled", env.enabled))
            if "resolution" in env_item:
                env.resolution = (int(env_item["resolution"][0]), int(env_item["resolution"][1]))
            if "skyAndSun" in env_item:
                env.sky_and_sun = SkyAndSun.from_json(env_item["skyAndSun"])
            ibl = env_item.get("ibl", {})
            if ibl.get("file"):
                env.hdr_file = _absolutize(project_dir, ibl["file"])
            env.ibl_intensity = float(_get(ibl, "intensity", env.ibl_intensity))
            if "rotation" in ibl:
                env.ibl_rotation = _t3(ibl["rotation"])

        if "settings" in data:
            scene.settings = Settings.from_json(data["settings"])
        if "tonemapping" in data:
            scene.tonemapping = Tonemapping.from_json(data["tonemapping"])

        return scene
