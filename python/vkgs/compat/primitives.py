"""PrimitivesVKGS: lazy mesh-primitive manager mirroring the 3dgrut
playground ``Primitives`` class (threedgrut_playground/engine.py:328-887).

Where 3dgrut keeps torch geometry buffers on the GPU, this manager only
records *which* mesh file goes where with what material type; the VKGS
renderer loads the files itself when the engine flushes the scene. The
autoscale heuristic replicates ``set_mesh_scale_to_scene`` (engine.py:
293-325) using file-level bounding boxes instead of loaded vertices.
"""

from __future__ import annotations

import copy
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .. import geometry, materials
from .convert import (
    DEFAULT_REFRACTIVE_INDEX,
    SHADOW_CATCHER_WORKAROUND,
    OptixPrimitiveTypes,
    transform_to_vkgs_trs,
    warn_compat,
)

Vec3 = Tuple[float, float, float]


# --------------------------------------------------------------------------
# File-level extent scanners (for the autoscale heuristic only)
# --------------------------------------------------------------------------

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def ply_scene_extent(path: str, max_samples: int = 65536) -> Vec3:
    """Per-axis extent (max - min) of a splat PLY's vertex positions.

    Minimal plyfile-free parser: reads the header, then samples up to
    ``max_samples`` evenly strided vertices' x/y/z via numpy memmap (binary)
    or line scanning (ascii). Only fixed-size vertex properties are
    supported (which covers 3DGS PLYs). On any failure — including .spz /
    .splat inputs, which have no cheap position scan — a CompatWarning is
    emitted and the 3dgrut default scene scale (1, 1, 1) is returned
    (Primitives(scene_scale=None), engine.py:388-391).
    """
    try:
        return _ply_scene_extent(path, max_samples)
    except Exception as exc:
        warn_compat(
            f"could not read splat extent from {path!r} ({exc}); primitive "
            "autoscale falls back to the 3dgrut default scene scale (1, 1, 1)"
        )
        return (1.0, 1.0, 1.0)


def _ply_scene_extent(path: str, max_samples: int) -> Vec3:
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        fmt = None
        count = 0
        fields = []  # (name, numpy type) of the vertex element, in order
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unterminated PLY header")
            tokens = line.decode("ascii", "replace").split()
            if not tokens or tokens[0] == "comment":
                continue
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                in_vertex = tokens[1] == "vertex"
                if in_vertex:
                    count = int(tokens[2])
            elif tokens[0] == "property" and in_vertex:
                if tokens[1] == "list":
                    raise ValueError("list property in vertex element")
                fields.append((tokens[-1], _PLY_TYPES[tokens[1]]))
            elif tokens[0] == "end_header":
                break
        data_offset = f.tell()

    names = [name for name, _ in fields]
    if count == 0 or any(axis not in names for axis in ("x", "y", "z")):
        raise ValueError("no x/y/z vertex properties")
    stride = max(1, count // max_samples)

    if fmt == "ascii":
        xyz_cols = [names.index(axis) for axis in ("x", "y", "z")]
        samples = []
        with open(path, "r", encoding="ascii", errors="replace") as f:
            f.seek(data_offset)
            for i in range(count):
                line = f.readline().split()
                if i % stride == 0:
                    samples.append([float(line[c]) for c in xyz_cols])
        positions = np.asarray(samples)
    else:
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(name, endian + kind) for name, kind in fields])
        vertices = np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=(count,))
        positions = np.stack(
            [vertices["x"][::stride], vertices["y"][::stride], vertices["z"][::stride]], axis=1
        ).astype(np.float64)

    extent = positions.max(axis=0) - positions.min(axis=0)
    return (float(extent[0]), float(extent[1]), float(extent[2]))


def _obj_extent(path: str) -> Vec3:
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                v = np.array([float(t) for t in line.split()[1:4]])
                lo = np.minimum(lo, v)
                hi = np.maximum(hi, v)
    if not np.all(np.isfinite(lo)):
        raise ValueError("no vertices found")
    return tuple(float(c) for c in hi - lo)


def _gltf_extent(path: str) -> Vec3:
    """Extent from the POSITION accessors' min/max (required by the glTF
    spec). Node transforms are ignored — heuristic only."""
    if path.lower().endswith(".glb"):
        with open(path, "rb") as f:
            magic, _, _ = struct.unpack("<4sII", f.read(12))
            if magic != b"glTF":
                raise ValueError("not a GLB file")
            chunk_len, chunk_type = struct.unpack("<II", f.read(8))
            if chunk_type != 0x4E4F534A:  # 'JSON'
                raise ValueError("first GLB chunk is not JSON")
            doc = json.loads(f.read(chunk_len))
    else:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            index = prim.get("attributes", {}).get("POSITION")
            if index is None:
                continue
            accessor = doc["accessors"][index]
            lo = np.minimum(lo, np.asarray(accessor["min"][:3], dtype=np.float64))
            hi = np.maximum(hi, np.asarray(accessor["max"][:3], dtype=np.float64))
    if not np.all(np.isfinite(lo)):
        raise ValueError("no POSITION accessors with min/max")
    return tuple(float(c) for c in hi - lo)


def mesh_file_extent(path: str) -> Vec3:
    """Bounding-box extent of a mesh file (.obj/.glb/.gltf). Falls back to
    (1, 1, 1) with a CompatWarning when the file cannot be parsed."""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".obj":
            return _obj_extent(path)
        if ext in (".glb", ".gltf"):
            return _gltf_extent(path)
        raise ValueError(f"unsupported mesh extension {ext!r}")
    except Exception as exc:
        warn_compat(f"could not read mesh extent from {path!r} ({exc}); autoscale assumes unit size")
        return (1.0, 1.0, 1.0)


def autoscale_factor(scene_extent: Vec3, mesh_extent: Vec3, scale_of_new_mesh_to_small_scene: float = 0.5) -> float:
    """Replicates set_mesh_scale_to_scene (engine.py:293-325): normalize the
    mesh to unit size, then for small scenes (max extent <= 5.0) scale to a
    fraction of the scene size. Returns the uniform scale factor."""
    factor = 1.0 / max(1e-12, max(mesh_extent))
    scene_max = max(scene_extent)
    if scene_max > 5.0:  # Don't scale for large scenes
        return factor
    return factor * scale_of_new_mesh_to_small_scene * scene_max


# --------------------------------------------------------------------------
# Primitive records
# --------------------------------------------------------------------------


@dataclass
class PrimitiveVKGS:
    """Lazy record of one mesh primitive (the compat stand-in for 3dgrut's
    OptixPrimitive). Transform fields are already in VKGS conventions
    (position / Euler XYZ degrees / scale, VKGS world frame)."""

    name: str
    geometry_type: str
    path: str  # absolute mesh file path fed to Scene.add_mesh
    primitive_type: OptixPrimitiveTypes = OptixPrimitiveTypes.PBR
    refractive_index: float = DEFAULT_REFRACTIVE_INDEX
    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Vec3 = (0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)
    show: bool = True

    def set_transform(self, matrix) -> None:
        """Set position/rotation/scale from a 3dgrut ObjectTransform model
        matrix (4x4 torch/numpy in the 3dgrut world frame)."""
        self.position, self.rotation, self.scale = transform_to_vkgs_trs(matrix)

    def fingerprint(self) -> tuple:
        return (
            self.name,
            self.geometry_type,
            self.path,
            int(self.primitive_type),
            float(self.refractive_index),
            tuple(self.position),
            tuple(self.rotation),
            tuple(self.scale),
            bool(self.show),
        )


class PrimitivesVKGS:
    """Mirror of threedgrut_playground.engine.Primitives on lazy VKGS state.

    The asset registry replicates register_available_assets (engine.py:
    402-422): every .obj/.glb/.gltf in ``mesh_assets_folder`` is registered
    under its capitalized file stem; procedural shapes are added on top.
    3dgrut only has a procedural 'Quad'; this manager also provides a
    procedural 'Sphere' (unless the assets folder ships one), because the
    stock playground scene expects a Sphere asset.
    """

    SUPPORTED_MESH_EXTENSIONS = [".obj", ".glb", ".gltf"]
    DEFAULT_REFRACTIVE_INDEX = DEFAULT_REFRACTIVE_INDEX

    def __init__(
        self,
        mesh_assets_folder: Optional[str] = None,
        scene_extent: Optional[Vec3] = None,
        cache_dir: Optional[str] = None,
    ):
        self.assets: Dict[str, Optional[str]] = self.register_available_assets(mesh_assets_folder)
        self.objects: Dict[str, PrimitiveVKGS] = {}
        self.instance_counter: Dict[str, int] = {}
        self.enabled: bool = True
        self.dirty: bool = True
        # 3dgrut default scene scale when none is given (engine.py:388-391)
        self.scene_extent: Vec3 = (1.0, 1.0, 1.0) if scene_extent is None else tuple(scene_extent)
        self.cache_dir = cache_dir
        # Factor-only stand-ins for 3dgrut register_default_materials
        # (PBRMaterial instances there, vkgs Materials here; the texture-based
        # 'checkboard' preset is not representable, see vkgs.materials).
        self.registered_materials = {name: factory() for name, factory in materials.PRESETS.items()}
        # 3dgrut toggles with no VKGS effect; kept so ported scripts run.
        self.use_smooth_normals: bool = True
        self.disable_pbr_textures: bool = False

    def register_available_assets(self, assets_folder: Optional[str]) -> Dict[str, Optional[str]]:
        available: Dict[str, Optional[str]] = {}
        if assets_folder:
            for asset in sorted(os.listdir(assets_folder)):
                if Path(asset).suffix.lower() in self.SUPPORTED_MESH_EXTENSIONS:
                    available[Path(asset).stem.capitalize()] = os.path.abspath(os.path.join(assets_folder, asset))
        # Procedural shapes map to None; 'Quad' always procedural (3dgrut
        # overwrites folder collisions too), 'Sphere' only when not shipped.
        available["Quad"] = None
        available.setdefault("Sphere", None)
        return available

    # ------------------------------------------------------------- add/remove

    def _next_name(self, geometry_type: str) -> str:
        self.instance_counter[geometry_type] = self.instance_counter.get(geometry_type, 0) + 1
        return f"{geometry_type} {self.instance_counter[geometry_type]}"

    def _resolve_path(self, geometry_type: str) -> str:
        if geometry_type not in self.assets:
            raise KeyError(
                f"unknown geometry_type {geometry_type!r}; available: {sorted(self.assets)} "
                "(assets are registered by capitalized file stem, engine.py:402-422)"
            )
        path = self.assets[geometry_type]
        if path is None:  # procedural
            path = geometry.ensure_procedural(geometry_type, cache_dir=self.cache_dir)
        return path

    def add_primitive(self, geometry_type: str, primitive_type, device=None) -> str:
        """Create a primitive from a registered geometry type with automatic
        scene-relative scaling (engine.py:563-627). ``device`` is accepted
        for signature parity and ignored. Returns the generated name
        ("{geometry_type} {count}"; 3dgrut returns None here)."""
        primitive_type = _coerce_primitive_type(primitive_type)
        path = self._resolve_path(geometry_type)
        name = self._next_name(geometry_type)
        factor = autoscale_factor(self.scene_extent, mesh_file_extent(path))
        self.objects[name] = PrimitiveVKGS(
            name=name,
            geometry_type=geometry_type,
            path=path,
            primitive_type=primitive_type,
            scale=(factor, factor, factor),
        )
        self.dirty = True
        return name

    def load_external_primitive(self, path: str, primitive_type, device=None, name: Optional[str] = None) -> str:
        """Import a mesh from an arbitrary path, as-authored: no recenter,
        no autoscale (engine.py:629-709). Returns the primitive name."""
        primitive_type = _coerce_primitive_type(primitive_type)
        abspath = os.path.abspath(path)
        if Path(abspath).suffix.lower() not in self.SUPPORTED_MESH_EXTENSIONS:
            raise ValueError(f"unsupported mesh file {path!r}; expected one of {self.SUPPORTED_MESH_EXTENSIONS}")
        geometry_type = Path(abspath).stem.capitalize()
        if name is None:
            name = self._next_name(geometry_type)
        self.objects[name] = PrimitiveVKGS(
            name=name, geometry_type=geometry_type, path=abspath, primitive_type=primitive_type
        )
        self.dirty = True
        return name

    def remove_primitive(self, name: str) -> None:
        del self.objects[name]
        self.dirty = True

    def duplicate_primitive(self, name: str) -> str:
        prim = self.objects[name]
        new_name = self._next_name(prim.geometry_type)
        clone = copy.deepcopy(prim)
        clone.name = new_name
        self.objects[new_name] = clone
        self.dirty = True
        return new_name

    # --------------------------------------------------------------- queries

    def has_visible_objects(self) -> bool:
        return any(p.primitive_type != OptixPrimitiveTypes.NONE and p.show for p in self.objects.values())

    def rebuild_bvh_if_needed(self, force: bool = False, rebuild: bool = True) -> None:
        """No-op: the VKGS renderer builds its BLAS/TLAS inside the render
        subprocess. Kept for script portability."""
        self.dirty = False

    def fingerprint(self) -> tuple:
        return (
            bool(self.enabled),
            tuple(prim.fingerprint() for prim in self.objects.values()),
        )


def _coerce_primitive_type(primitive_type) -> OptixPrimitiveTypes:
    """Coerce to the enum; SHADOW_CATCHER warns at add time (not at flush)
    that it maps to the GS shadow mask approximation (the record is kept so
    scripts can transform/remove it, but it never becomes a scene mesh —
    EngineVKGS._build_scene turns it into renderer.gs_shadow_mask +
    shadow_only light copies instead)."""
    primitive_type = OptixPrimitiveTypes(int(primitive_type))
    if primitive_type == OptixPrimitiveTypes.SHADOW_CATCHER:
        warn_compat(SHADOW_CATCHER_WORKAROUND)
    return primitive_type
