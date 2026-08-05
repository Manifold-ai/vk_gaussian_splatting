import math

import numpy as np
import pytest

from vkgs.camera import (
    Camera,
    THREEDGRUT_TO_VKGS,
    decompose_trs,
    euler_deg_from_rotation_matrix,
    load_inria_cameras,
    rotation_matrix_from_euler_deg,
    save_inria_cameras,
)


def test_view_matrix_roundtrip():
    cam = Camera(eye=(3.0, 1.5, -2.0), ctr=(0.2, 0.5, 0.1), up=(0.0, 1.0, 0.0), fov=50.0)
    view = cam.to_view_matrix()
    focus = float(np.linalg.norm(np.subtract(cam.ctr, cam.eye)))
    back = Camera.from_view_matrix(view, fov=cam.fov, focus=focus)

    assert np.allclose(back.eye, cam.eye, atol=1e-9)
    assert np.allclose(back.ctr, cam.ctr, atol=1e-9)
    # up is re-orthogonalized but must span the same plane orientation
    view2 = back.to_view_matrix()
    assert np.allclose(view, view2, atol=1e-9)


def test_view_matrix_properties():
    cam = Camera(eye=(1, 2, 3), ctr=(0, 0, 0))
    V = cam.to_view_matrix()
    R = V[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)  # orthonormal
    # eye maps to origin
    assert np.allclose(V @ np.array([1, 2, 3, 1.0]), [0, 0, 0, 1], atol=1e-12)
    # looking down -Z: ctr maps to negative z
    p = V @ np.array([0, 0, 0, 1.0])
    assert p[2] < 0


def test_colmap_matches_cpp_importer_math():
    """Replicates importCamerasINRIA (src/camera_set.h:254-266) by hand for a
    non-trivial rotation and compares."""
    # Rotation: 30 deg about Y composed with 20 deg about X (row-major world)
    ry, rx = math.radians(30), math.radians(20)
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    R = (Ry @ Rx)
    position = [1.0, 2.0, 3.0]

    # Hand-computed C++ math
    M = R.copy()
    M[0, 1] = -R[0, 1]
    M[1, 2] = -R[1, 2]
    M[2, 2] = -R[2, 2]
    up = M @ [0, 1, 0]
    up /= np.linalg.norm(up)
    at = M @ [0, 0, 1]
    at /= np.linalg.norm(at)
    eye = np.array([position[0], -position[1], -position[2]])

    cam = Camera.from_colmap(position, R.tolist(), fy=1000.0, height=1080)
    assert np.allclose(cam.eye, eye, atol=1e-12)
    assert np.allclose(cam.up, up, atol=1e-12)
    assert np.allclose(np.subtract(cam.ctr, cam.eye), at, atol=1e-12)
    assert cam.fov == pytest.approx(math.degrees(2 * math.atan(1080 / 2000.0)))


def test_inria_json_roundtrip(tmp_path):
    cams = [
        Camera(eye=(1.0, 2.0, 3.0), ctr=(0.0, 0.0, 0.0), fov=55.0),
        Camera(eye=(-2.0, 0.5, 1.0), ctr=(0.3, 0.2, -0.5), fov=70.0),
    ]
    path = str(tmp_path / "cameras.json")
    save_inria_cameras(path, cams, width=1920, height=1080)
    loaded = load_inria_cameras(path)

    for orig, back in zip(cams, loaded):
        assert np.allclose(back.eye, orig.eye, atol=1e-9)
        assert back.fov == pytest.approx(orig.fov, abs=1e-9)
        # direction and up preserved (ctr distance is normalized to 1)
        d0 = np.subtract(orig.ctr, orig.eye)
        d0 = d0 / np.linalg.norm(d0)
        d1 = np.subtract(back.ctr, back.eye)
        assert np.allclose(d1, d0, atol=1e-9)


def test_euler_roundtrip():
    for angles in [(10, 20, 30), (-45, 60, 5), (0, 0, 0), (90, 10, -20)]:
        R = rotation_matrix_from_euler_deg(angles)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        back = euler_deg_from_rotation_matrix(R)
        R2 = rotation_matrix_from_euler_deg(back)
        assert np.allclose(R, R2, atol=1e-9)


def test_euler_matches_glm_quat_convention():
    """R must equal mat3_cast(quat(radians(xyz))) i.e. the GLM component
    formula (utilities.h computeTransform)."""
    angles = (25.0, -40.0, 65.0)
    x, y, z = (math.radians(a) / 2 for a in angles)
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    # GLM quat-from-euler components
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    # quat -> matrix
    R_quat = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )
    R = rotation_matrix_from_euler_deg(angles)
    assert np.allclose(R, R_quat, atol=1e-12)


def test_decompose_trs():
    t = (1.0, -2.0, 3.0)
    r = (30.0, 45.0, -10.0)
    s = (2.0, 0.5, 1.5)
    R = rotation_matrix_from_euler_deg(r)
    m = np.eye(4)
    m[:3, :3] = R * np.array(s)[np.newaxis, :]
    m[:3, 3] = t
    pos, rot, scale = decompose_trs(m)
    assert np.allclose(pos, t, atol=1e-12)
    assert np.allclose(scale, s, atol=1e-12)
    assert np.allclose(rotation_matrix_from_euler_deg(rot), R, atol=1e-9)


def test_threedgrut_world_conversion():
    """A camera at +Z looking at origin in the 3dgrut (RDF-world) frame must
    land at -Z (flipped Y/Z) in the VKGS RUB world."""
    M = THREEDGRUT_TO_VKGS
    # camera-to-world: identity rotation (right=+X, up=+Y, backward=+Z), eye at (0, 1, 5)
    c2w = np.eye(4)
    c2w[:3, 3] = [0.0, 1.0, 5.0]
    cam = Camera.from_threedgrut_world(c2w, fov=45.0)
    assert np.allclose(cam.eye, M @ [0, 1, 5], atol=1e-12)
    # forward in 3dgrut world = -Z -> in VKGS world = +Z
    fwd = np.subtract(cam.ctr, cam.eye)
    assert np.allclose(fwd, M @ [0, 0, -1], atol=1e-12)
    assert np.allclose(cam.up, M @ [0, 1, 0], atol=1e-12)
