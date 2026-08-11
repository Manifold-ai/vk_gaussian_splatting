"""3dgrut playground -> VKGS conversions: enums, cameras, materials,
lights and object transforms.

Coordinate systems: 3dgrut works in the PLY-native world frame, VKGS
converts to right-up-back on load; the world change of basis is
``vkgs.camera.THREEDGRUT_TO_VKGS`` = diag(1, -1, -1) (a proper rotation,
so cross products and handedness are preserved).

The mirrored enums/dataclasses copy the 3dgrut playground definitions
(threedgrut_playground/engine.py): OptixPrimitiveTypes :196-209, LightType
:120-131, Light :134-179.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np

from .. import materials
from ..camera import THREEDGRUT_TO_VKGS, Camera, decompose_trs
from ..constants import LightType as VkgsLightType
from ..project import Material

Vec3 = Tuple[float, float, float]

# Heuristic reference distance (world units) used to turn a 3dgrut angular
# radius (radians, as seen from the shading point) into the world-space
# emitter radius VKGS soft shadows expect: radius = tan(angular_radius) * D.
# Exact conversion depends on the light-to-shading-point distance, which is
# per-pixel; D=10 matches typical splat-scene scales. Documented
# approximation (plan gap C2).
SOFT_SHADOW_REFERENCE_DISTANCE = 10.0

# 3dgrut Primitives.DEFAULT_REFRACTIVE_INDEX (engine.py:346)
DEFAULT_REFRACTIVE_INDEX = 1.33


class CompatWarning(UserWarning):
    """Emitted whenever the compat shim approximates or ignores a 3dgrut
    playground feature; the message always includes the workaround."""


def warn_compat(message: str, stacklevel: int = 3) -> None:
    warnings.warn(message, CompatWarning, stacklevel=stacklevel)


class OptixPrimitiveTypes(IntEnum):
    """Mirror of threedgrut_playground.engine.OptixPrimitiveTypes (:196-209)."""

    NONE = 0
    MIRROR = 1
    GLASS = 2
    DIFFUSE = 3
    PBR = 4
    SHADOW_CATCHER = 5

    @classmethod
    def names(cls):
        # Index must equal the enum value: names()[t.value] == display name
        return ["None", "Mirror", "Glass", "Diffuse Mesh", "PBR Mesh", "Shadow Catcher"]


class LightType(IntEnum):
    """Mirror of threedgrut_playground.engine.LightType (:120-131).

    NOTE: these are the *3dgrut* numeric values; the VKGS project file uses
    vkgs.constants.LightType (0=directional, 1=point, 2=spot) instead.
    """

    NONE = 0
    DIRECTIONAL = 1
    POINT = 2
    AREA = 3

    @classmethod
    def names(cls):
        return ["None", "Directional", "Point", "Area"]


@dataclass
class Light:
    """Mirror of threedgrut_playground.engine.Light (:134-179).

    ``direction`` is a unit vector pointing FROM the shading point TO the
    light (3dgrut contract R6), in the 3dgrut world frame.
    """

    light_type: int = int(LightType.DIRECTIONAL)
    direction: Vec3 = (0.0, -1.0, 0.0)
    color: Vec3 = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    angular_radius: float = 0.0  # radians; 0 = hard shadow
    position: Vec3 = (0.0, 0.0, 0.0)
    tangent_u: Vec3 = (0.0, 0.0, 0.0)  # AREA half-edge U (world)
    tangent_v: Vec3 = (0.0, 0.0, 0.0)  # AREA half-edge V (world)
    # both light and shadow: illuminate normally AND cast onto the GS shadow
    # mask (renderer.gs_shadow_mask, RTX pipelines) — no shadow_only twin needed
    cast_on_gs: bool = False

    def to_row(self) -> List[float]:
        """3dgrut contract-C8 18-column row (engine.py:150-179); used here
        only as a stable fingerprint of the light state."""
        d = np.asarray(self.direction, dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n > 1e-8:
            d = d / n
        return [
            float(self.light_type),
            *(float(c) for c in d),
            *(float(c) for c in self.color),
            float(self.intensity),
            float(self.angular_radius),
            *(float(c) for c in self.position),
            *(float(c) for c in self.tangent_u),
            *(float(c) for c in self.tangent_v),
            float(self.cast_on_gs),
        ]


# --------------------------------------------------------------------------
# numpy helpers
# --------------------------------------------------------------------------


def _to_numpy(value) -> np.ndarray:
    """torch tensor / array-like -> float64 numpy array."""
    if hasattr(value, "detach"):  # torch.Tensor without importing torch
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _v3(v) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


# --------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------


def camera_to_vkgs(camera, fov: float = 45.0, model: Optional[int] = None) -> Tuple[Camera, Optional[Tuple[int, int]]]:
    """Convert a render() camera argument into a vkgs Camera.

    Accepts:
    - a vkgs Camera: passed through as a copy (its own fov/model win);
    - a kaolin Camera (detected by module name, no kaolin import needed
      here): converted via Camera.from_kaolin, size hint = (width, height);
    - a 4x4 camera-to-world matrix (numpy or torch) expressed in the 3dgrut
      world frame with OpenGL right/up/backward camera axes, converted via
      Camera.from_threedgrut_world using ``fov`` (vertical degrees).

    ``model`` (vkgs.constants.CameraModel) is applied to the kaolin/matrix
    branches only; an explicit vkgs Camera keeps its authored model.
    Returns (camera, size_hint) where size_hint is (W, H) or None.
    """
    if isinstance(camera, Camera):
        return replace(camera), None

    if any(klass.__module__.startswith("kaolin") for klass in type(camera).__mro__):
        cam = Camera.from_kaolin(camera)
        if model is not None:
            cam.model = int(model)
        return cam, (int(camera.width), int(camera.height))

    try:
        matrix = _to_numpy(camera)
    except (TypeError, ValueError):
        matrix = None
    if matrix is None or matrix.shape != (4, 4):
        raise TypeError(
            f"unsupported camera argument {type(camera).__name__}: expected a vkgs Camera, "
            "a kaolin Camera, or a 4x4 camera-to-world matrix"
        )
    cam = Camera.from_threedgrut_world(matrix, fov=float(fov))
    if model is not None:
        cam.model = int(model)
    return cam, None


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------

SHADOW_CATCHER_WORKAROUND = (
    "SHADOW_CATCHER primitives are mapped to the VKGS GS shadow mask "
    "approximation (renderer.gs_shadow_mask, RTX pipelines): analytic-light "
    "shadows fall directly onto the GS surfels and the catcher mesh itself "
    "is never rendered. For the usual ground-plane catcher coincident with "
    "GS geometry the result is equivalent or better (no proxy-mesh seams); "
    "a catcher floating in empty space casts no shadow, since there are no "
    "surfels beneath it to darken."
)


def primitive_type_to_material(primitive_type, ior: Optional[float] = None, **overrides) -> Optional[Material]:
    """Map a 3dgrut OptixPrimitiveTypes value to a vkgs Material.

    - MIRROR -> materials.mirror() (metallic=1, roughness=0)
    - GLASS -> materials.glass(ior=1.33) (3dgrut DEFAULT_REFRACTIVE_INDEX)
    - DIFFUSE -> materials.diffuse()
    - PBR / NONE -> None (PBR keeps the mesh file's authored glTF materials;
      NONE primitives are hidden by the caller)
    - SHADOW_CATCHER -> CompatWarning + None (the catcher mesh is never
      rendered; it only requests the GS shadow mask, see
      SHADOW_CATCHER_WORKAROUND and EngineVKGS._build_scene)

    ``overrides`` are forwarded to the material factory.
    """
    primitive_type = OptixPrimitiveTypes(int(primitive_type))
    if primitive_type == OptixPrimitiveTypes.SHADOW_CATCHER:
        warn_compat(SHADOW_CATCHER_WORKAROUND)
        return None
    if primitive_type in (OptixPrimitiveTypes.PBR, OptixPrimitiveTypes.NONE):
        return None
    if primitive_type == OptixPrimitiveTypes.MIRROR:
        return materials.mirror(**overrides)
    if primitive_type == OptixPrimitiveTypes.GLASS:
        return materials.glass(ior=DEFAULT_REFRACTIVE_INDEX if ior is None else float(ior), **overrides)
    return materials.diffuse(**overrides)  # DIFFUSE


# --------------------------------------------------------------------------
# Lights
# --------------------------------------------------------------------------


def direction_to_euler_deg(direction) -> Vec3:
    """Euler XYZ degrees (GLM R = Rz@Ry@Rx) whose rotated -Z axis equals
    ``direction`` (VKGS light direction = R @ (0,0,-1),
    src/light_manager_vk.cpp:465). Roll (z) is left at 0.
    """
    d = np.asarray(direction, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        raise ValueError("light direction must be non-zero")
    d = d / n
    # With z=0: R @ (0,0,-1) = (-sin(y)cos(x), sin(x), -cos(y)cos(x))
    x = math.asin(np.clip(d[1], -1.0, 1.0))
    cx = math.cos(x)
    y = 0.0 if abs(cx) < 1e-9 else math.atan2(-d[0], -d[2])
    return (math.degrees(x), math.degrees(y), 0.0)


AREA_LIGHT_WORKAROUND = (
    "AREA lights have no VKGS equivalent (only directional/point/spot, plan "
    "gap A4); approximating with an emissive quad mesh sized from "
    "tangent_u/tangent_v. Emissive meshes only light the scene through the "
    "path-traced pipelines (RTX/HYBRID/HYBRID_3DGUT) and converge noisier "
    "than an analytic light; alternatively use a POINT light with a radius."
)


def area_light_to_quad(light: Light) -> dict:
    """Approximate a 3dgrut AREA light as an emissive unit quad instance.

    The rect is centered at ``position`` with half-edges tangent_u/tangent_v
    (engine.py:126,146-148). The returned position/rotation/scale place the
    vkgs.geometry unit quad (XZ plane, +Y normal, edge 1) so that local X
    spans 2*|tangent_u|, local Z spans 2*|tangent_v| and +Y faces the
    emission normal (tangent_v x tangent_u). tangent_v is orthogonalized
    against tangent_u if the input rect is sheared. Emits a CompatWarning
    (see AREA_LIGHT_WORKAROUND).
    """
    warn_compat(AREA_LIGHT_WORKAROUND)
    M = THREEDGRUT_TO_VKGS
    u = M @ np.asarray(light.tangent_u, dtype=np.float64)
    v = M @ np.asarray(light.tangent_v, dtype=np.float64)
    p = M @ np.asarray(light.position, dtype=np.float64)
    lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if lu < 1e-9 or lv < 1e-9:
        raise ValueError("AREA light needs non-zero tangent_u and tangent_v half-edges")
    u_hat = u / lu
    v_ortho = v - float(np.dot(v, u_hat)) * u_hat  # drop shear
    if float(np.linalg.norm(v_ortho)) < 1e-9:
        raise ValueError("AREA light tangents are parallel")
    v_hat = v_ortho / np.linalg.norm(v_ortho)
    n_hat = np.cross(v_hat, u_hat)  # +Y of the local quad frame (see geometry)

    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([u_hat, n_hat, v_hat]) @ np.diag([2.0 * lu, 1.0, 2.0 * lv])
    matrix[:3, 3] = p
    position, rotation, scale = decompose_trs(matrix)
    material = Material(
        name="area_light",
        base_color=(0.0, 0.0, 0.0),
        metallic=0.0,
        roughness=1.0,
        emissive=_v3(light.color),
        emissive_strength=float(light.intensity),
        max_bounces=0,
    )
    return {"kind": "emissive_quad", "position": position, "rotation": rotation, "scale": scale, "material": material}


def light_to_vkgs(light: Light) -> dict:
    """Convert one 3dgrut Light into VKGS scene data.

    Returns a dict with key "kind":
    - "skip": LightType.NONE (contributes nothing on either side);
    - "light": fields for Scene.add_light — type (vkgs LightType), color,
      intensity, translation, rotation, radius, soft (True when
      angular_radius > 0, i.e. the scene needs shadows_mode=SOFT);
    - "emissive_quad": see :func:`area_light_to_quad` (AREA, CompatWarning).

    Directions/positions are moved into the VKGS world with
    THREEDGRUT_TO_VKGS. The 3dgrut ``direction`` points TO the light, so a
    DIRECTIONAL light's VKGS emission direction is -direction; its instance
    rotation is derived with :func:`direction_to_euler_deg`. angular_radius
    becomes a world radius via tan(angular_radius) *
    SOFT_SHADOW_REFERENCE_DISTANCE (documented approximation). Intensity and
    point-light attenuation semantics differ between the engines (plan gap
    C7): values are passed through unscaled.
    """
    ltype = LightType(int(light.light_type))
    if ltype == LightType.NONE:
        return {"kind": "skip"}
    if ltype == LightType.AREA:
        return area_light_to_quad(light)

    M = THREEDGRUT_TO_VKGS
    soft = float(light.angular_radius) > 0.0
    radius = math.tan(float(light.angular_radius)) * SOFT_SHADOW_REFERENCE_DISTANCE if soft else 0.0

    if ltype == LightType.DIRECTIONAL:
        emit_dir = M @ (-np.asarray(light.direction, dtype=np.float64))
        return {
            "kind": "light",
            "type": VkgsLightType.DIRECTIONAL,
            "color": _v3(light.color),
            "intensity": float(light.intensity),
            "translation": (0.0, 0.0, 0.0),
            "rotation": direction_to_euler_deg(emit_dir),
            "radius": radius,
            "soft": soft,
            "cast_on_gs": bool(light.cast_on_gs),
        }

    # POINT: position only (3dgrut ignores direction for point lights)
    return {
        "kind": "light",
        "type": VkgsLightType.POINT,
        "color": _v3(light.color),
        "intensity": float(light.intensity),
        "translation": _v3(M @ np.asarray(light.position, dtype=np.float64)),
        "rotation": (0.0, 0.0, 0.0),
        "radius": radius,
        "soft": soft,
        "cast_on_gs": bool(light.cast_on_gs),
    }


# --------------------------------------------------------------------------
# Object transforms
# --------------------------------------------------------------------------


def transform_to_vkgs_trs(matrix) -> Tuple[Vec3, Vec3, Vec3]:
    """Convert a 3dgrut ObjectTransform model matrix (4x4 local->world in
    the 3dgrut frame, torch or numpy) into VKGS instance fields.

    The mesh file's local vertices are identical on both sides, so the VKGS
    placement matrix is M_h @ T (M_h = homogeneous THREEDGRUT_TO_VKGS),
    decomposed into (position, rotation Euler XYZ degrees, scale) with
    vkgs.camera.decompose_trs.
    """
    T = _to_numpy(matrix)
    if T.shape != (4, 4):
        raise ValueError(f"expected a 4x4 transform, got shape {T.shape}")
    M_h = np.eye(4)
    M_h[:3, :3] = THREEDGRUT_TO_VKGS
    return decompose_trs(M_h @ T)
