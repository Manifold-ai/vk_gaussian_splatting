"""Camera model and pose conversions.

The VKGS camera is stored as eye / ctr (look-at point) / up + vertical fov in
degrees (src/camera_set.h). This module converts between that representation
and 4x4 view matrices, COLMAP/INRIA camera files, and (optionally) kaolin
cameras used by 3dgrut.

Coordinate systems
------------------
VKGS converts every splat input (PLY/SPZ/SPLAT, COLMAP-style RDF) to RUB on
load, while 3dgrut keeps the PLY's native frame. The world-frame change of
basis between the two is M = diag(1, -1, -1) (a 180-degree rotation about X);
see ``THREEDGRUT_TO_VKGS`` and :func:`from_threedgrut_world`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .constants import CameraModel, DofMode

Vec3 = Tuple[float, float, float]

# World-frame change of basis: 3dgrut world (PLY-native, COLMAP RDF) -> VKGS
# world (RUB). Its own inverse.
THREEDGRUT_TO_VKGS = np.diag([1.0, -1.0, -1.0])


def _v3(v: Sequence[float]) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


@dataclass
class Camera:
    """One VKGS camera (active camera or preset).

    Field names/defaults mirror the C++ ``Camera`` struct (src/camera_set.h:44).
    ``fov`` is the vertical field of view in degrees.
    """

    model: int = CameraModel.PINHOLE
    eye: Vec3 = (1.7, 1.5, 1.7)
    ctr: Vec3 = (0.0, 0.0, 0.0)
    up: Vec3 = (0.0, 1.0, 0.0)
    fov: float = 60.0
    clip: Tuple[float, float] = (0.1, 2000.0)
    dof_mode: int = DofMode.DISABLED
    focus_dist: float = 1.3
    aperture: float = 0.001

    # ---------------------------------------------------------------- JSON

    def to_json(self) -> dict:
        """Serialize with the exact keys of vkgs_project_writer.cpp."""
        return {
            "model": int(self.model),
            "ctr": [float(c) for c in self.ctr],
            "eye": [float(c) for c in self.eye],
            "up": [float(c) for c in self.up],
            "fov": float(self.fov),
            "clip": [float(self.clip[0]), float(self.clip[1])],
            "dofMode": int(self.dof_mode),
            "focusDist": float(self.focus_dist),
            "aperture": float(self.aperture),
        }

    @classmethod
    def from_json(cls, item: dict) -> "Camera":
        cam = cls()
        if "model" in item:
            cam.model = int(item["model"])
        if "ctr" in item:
            cam.ctr = _v3(item["ctr"])
        if "eye" in item:
            cam.eye = _v3(item["eye"])
        if "up" in item:
            cam.up = _v3(item["up"])
        if "fov" in item:
            cam.fov = float(item["fov"])
        if "clip" in item:
            cam.clip = (float(item["clip"][0]), float(item["clip"][1]))
        # Backward compat: old files use bool "dofEnabled" (reader.cpp:709-712)
        if "dofEnabled" in item:
            cam.dof_mode = int(item["dofEnabled"])
        if "dofMode" in item:
            cam.dof_mode = int(item["dofMode"])
        if "focusDist" in item:
            cam.focus_dist = float(item["focusDist"])
        if "aperture" in item:
            cam.aperture = float(item["aperture"])
        return cam

    # ------------------------------------------------------- view matrices

    def to_view_matrix(self) -> np.ndarray:
        """Right-handed lookAt view matrix (world -> camera, camera looks -Z)."""
        eye = np.asarray(self.eye, dtype=np.float64)
        ctr = np.asarray(self.ctr, dtype=np.float64)
        up = np.asarray(self.up, dtype=np.float64)

        f = ctr - eye
        f = f / np.linalg.norm(f)
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)

        view = np.eye(4)
        view[0, :3] = s
        view[1, :3] = u
        view[2, :3] = -f
        view[:3, 3] = -view[:3, :3] @ eye
        return view

    @classmethod
    def from_view_matrix(cls, view: np.ndarray, fov: float = 60.0, focus: float = 1.0, **kwargs) -> "Camera":
        """Build a camera from a right-handed world->camera view matrix.

        ``focus`` sets the eye->ctr distance (only affects the navigation
        pivot, not the rendered image).
        """
        view = np.asarray(view, dtype=np.float64)
        rot = view[:3, :3]
        eye = -rot.T @ view[:3, 3]
        fwd = -rot[2]  # camera looks down -Z: forward = -(third row)
        up = rot[1]
        return cls(eye=_v3(eye), ctr=_v3(eye + focus * fwd), up=_v3(up), fov=fov, **kwargs)

    def to_camera_to_world(self) -> np.ndarray:
        """4x4 camera->world matrix (inverse of the view matrix)."""
        view = self.to_view_matrix()
        c2w = np.eye(4)
        rot = view[:3, :3]
        c2w[:3, :3] = rot.T
        c2w[:3, 3] = -rot.T @ view[:3, 3]
        return c2w

    # ------------------------------------------------ COLMAP / INRIA input

    @classmethod
    def from_colmap(
        cls,
        position: Sequence[float],
        rotation: Sequence[Sequence[float]],
        fy: Optional[float] = None,
        height: Optional[int] = None,
        **kwargs,
    ) -> "Camera":
        """Convert an INRIA/COLMAP ``cameras.json`` entry to a VKGS camera.

        Replicates the RDF->RUB math of ``importCamerasINRIA``
        (src/camera_set.h:254-266) exactly, and — unlike the C++ importer,
        which discards fx/fy — sets the vertical fov from ``fy``/``height``
        when provided.
        """
        R = np.asarray(rotation, dtype=np.float64)
        M = R.copy()
        M[0, 1] = -R[0, 1]
        M[1, 2] = -R[1, 2]
        M[2, 2] = -R[2, 2]

        up = M @ np.array([0.0, 1.0, 0.0])
        up = up / np.linalg.norm(up)
        at = M @ np.array([0.0, 0.0, 1.0])
        at = at / np.linalg.norm(at)

        eye = np.array([position[0], -position[1], -position[2]], dtype=np.float64)

        cam = cls(eye=_v3(eye), ctr=_v3(eye + at), up=_v3(up), **kwargs)
        if fy is not None and height is not None:
            cam.fov = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
        return cam

    # --------------------------------------------------- 3dgrut / kaolin

    @classmethod
    def from_threedgrut_world(cls, camera_to_world: np.ndarray, fov: float = 60.0, focus: float = 1.0, **kwargs) -> "Camera":
        """Convert a camera->world pose expressed in the 3dgrut world frame
        (PLY-native / COLMAP RDF) into a VKGS camera (RUB world).

        The camera-space convention is OpenGL-style right/up/backward
        (camera looks down -Z), which is what kaolin's inv_view_matrix
        returns.
        """
        C = np.asarray(camera_to_world, dtype=np.float64)
        M = THREEDGRUT_TO_VKGS
        eye_w = C[:3, 3]
        fwd_w = -C[:3, 2]
        up_w = C[:3, 1]
        return cls(
            eye=_v3(M @ eye_w),
            ctr=_v3(M @ (eye_w + focus * fwd_w)),
            up=_v3(M @ up_w),
            fov=fov,
            **kwargs,
        )

    @classmethod
    def from_kaolin(cls, kaolin_camera, focus: float = 1.0, **kwargs) -> "Camera":
        """Convert a kaolin ``Camera`` (as used by 3dgrut playground scripts).

        Requires torch/kaolin importable. Handles batched cameras by taking
        index 0.
        """
        import torch  # noqa: F401  (kaolin cameras hold torch tensors)

        c2w = kaolin_camera.inv_view_matrix()
        if c2w.dim() == 3:
            c2w = c2w[0]
        c2w = c2w.detach().cpu().numpy().astype(np.float64)

        fov = float(kaolin_camera.fov(in_degrees=True))
        # kaolin fov may be horizontal; convert to the vertical fov VKGS expects
        fov_dir = getattr(kaolin_camera.intrinsics, "fov_direction", None)
        if fov_dir is not None and getattr(fov_dir, "name", "").lower().startswith("horizontal"):
            aspect = float(kaolin_camera.width) / float(kaolin_camera.height)
            fov = math.degrees(2.0 * math.atan(math.tan(math.radians(fov) / 2.0) / aspect))

        near = float(getattr(kaolin_camera, "near", 0.1)) or 0.1
        far = float(getattr(kaolin_camera, "far", 2000.0)) or 2000.0
        kwargs.setdefault("clip", (max(near, 1e-3), far))
        return cls.from_threedgrut_world(c2w, fov=fov, focus=focus, **kwargs)


# ------------------------------------------------------------- Euler helpers
# VKGS instance rotations are Euler angles in degrees, applied as
# R = mat4_cast(quat(radians(xyz))) (src/utilities.h:170). GLM's
# quat(eulerAngles) composes q = qz * qy * qx, so the matrix is
# R = Rz @ Ry @ Rx (extrinsic X-then-Y-then-Z, i.e. yaw-pitch-roll ZYX).


def rotation_matrix_from_euler_deg(euler_deg: Sequence[float]) -> np.ndarray:
    """R = Rz @ Ry @ Rx for Euler angles in degrees (GLM quat convention)."""
    x, y, z = (math.radians(a) for a in euler_deg)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def euler_deg_from_rotation_matrix(R: np.ndarray) -> Vec3:
    """Inverse of :func:`rotation_matrix_from_euler_deg` (R = Rz @ Ry @ Rx)."""
    R = np.asarray(R, dtype=np.float64)
    sy = float(np.clip(-R[2, 0], -1.0, 1.0))
    y = math.asin(sy)
    if abs(sy) < 1.0 - 1e-9:
        x = math.atan2(R[2, 1], R[2, 2])
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: fold z into x
        x = math.atan2(-R[0, 1], R[1, 1])
        z = 0.0
    return (math.degrees(x), math.degrees(y), math.degrees(z))


def decompose_trs(matrix: np.ndarray) -> Tuple[Vec3, Vec3, Vec3]:
    """Decompose a 4x4 T*R*S matrix into VKGS instance fields
    (position, rotation Euler XYZ degrees, scale).

    Assumes no shear (which T*R*S composition guarantees). Negative
    determinants fold the flip into the Z scale.
    """
    m = np.asarray(matrix, dtype=np.float64)
    position = _v3(m[:3, 3])
    rs = m[:3, :3]
    scale = np.linalg.norm(rs, axis=0)
    if np.linalg.det(rs) < 0:
        scale[2] = -scale[2]
    rot = rs / scale[np.newaxis, :]
    return position, euler_deg_from_rotation_matrix(rot), _v3(scale)


# ------------------------------------------------------------ INRIA files


def load_inria_cameras(path: str) -> List[Camera]:
    """Load an INRIA-format cameras.json into VKGS cameras (with correct fov,
    unlike the in-app --loadCameraPresets importer which fixes fov at 60)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cams = []
    for item in data:
        cams.append(
            Camera.from_colmap(
                position=item["position"],
                rotation=item["rotation"],
                fy=item.get("fy"),
                height=item.get("height"),
            )
        )
    return cams


def save_inria_cameras(path: str, cameras: Iterable[Camera], width: int = 1920, height: int = 1080) -> None:
    """Write cameras to an INRIA-format json consumable by --loadCameraPresets.

    Inverse of the import math in src/camera_set.h:254-266. fx/fy are derived
    from each camera's vertical fov (the in-app importer ignores them).
    """
    out = []
    for i, cam in enumerate(cameras):
        eye = np.asarray(cam.eye, dtype=np.float64)
        ctr = np.asarray(cam.ctr, dtype=np.float64)
        up = np.asarray(cam.up, dtype=np.float64)

        at = ctr - eye
        at = at / np.linalg.norm(at)
        # Re-orthogonalize the frame: side = at x up, corrected up = side x at
        side = np.cross(at, up)
        side = side / np.linalg.norm(side)
        up_o = np.cross(side, at)

        # Invert: col0 = side (unsigned), col1/col2 sign pattern of the importer
        M = np.stack([side, up_o, at], axis=1)
        R = M.copy()
        R[0, 1] = -M[0, 1]
        R[1, 2] = -M[1, 2]
        R[2, 2] = -M[2, 2]

        fy = height / (2.0 * math.tan(math.radians(cam.fov) / 2.0))
        fx = fy  # square pixels
        out.append(
            {
                "id": i,
                "img_name": f"camera_{i:05d}",
                "width": width,
                "height": height,
                "position": [float(eye[0]), float(-eye[1]), float(-eye[2])],
                "rotation": [[float(R[r, c]) for c in range(3)] for r in range(3)],
                "fx": fx,
                "fy": fy,
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
