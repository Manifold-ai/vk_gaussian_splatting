"""CPU tests for vkgs.video: trajectory interpolation and .npy round-trip."""

import math
import os

import numpy as np
import pytest

from vkgs.camera import THREEDGRUT_TO_VKGS, Camera
from vkgs.video import interpolate_cameras, load_trajectory, save_trajectory

_M4 = np.eye(4)
_M4[:3, :3] = THREEDGRUT_TO_VKGS


def _keyframes(n=4):
    """Distinct keyframes on a horizontal arc (up perpendicular to view)."""
    cams = []
    for k in range(n):
        angle = 2.0 * math.pi * k / n
        cams.append(
            Camera(
                eye=(3.0 * math.cos(angle), 1.0, 3.0 * math.sin(angle)),
                ctr=(0.1 * k, 1.0, 0.2),
                up=(0.0, 1.0, 0.0),
                fov=40.0 + 10.0 * k,
                clip=(0.1 + 0.01 * k, 100.0 + k),
                focus_dist=1.0 + 0.5 * k,
                aperture=0.001 * (k + 1),
            )
        )
    return cams


# ------------------------------------------------------------- interpolation


@pytest.mark.parametrize("mode", ["linear", "spline"])
def test_open_mode_frame_count(mode):
    keyframes = _keyframes(5)
    frames = interpolate_cameras(keyframes, frames_between=12, mode=mode)
    assert len(frames) == (len(keyframes) - 1) * 12 + 1


def test_cyclic_frame_count():
    keyframes = _keyframes(4)
    frames = interpolate_cameras(keyframes, frames_between=10, mode="cyclic")
    assert len(frames) == len(keyframes) * 10 + 1


@pytest.mark.parametrize("mode", ["linear", "spline"])
def test_passes_through_keyframes_at_knots(mode):
    keyframes = _keyframes(4)
    fb = 8
    frames = interpolate_cameras(keyframes, frames_between=fb, mode=mode)
    for k, kf in enumerate(keyframes):
        frame = frames[k * fb]
        np.testing.assert_allclose(frame.eye, kf.eye, atol=1e-9)
        np.testing.assert_allclose(frame.ctr, kf.ctr, atol=1e-9)
        np.testing.assert_allclose(frame.fov, kf.fov, atol=1e-9)
        np.testing.assert_allclose(frame.aperture, kf.aperture, atol=1e-12)
        np.testing.assert_allclose(frame.focus_dist, kf.focus_dist, atol=1e-9)
        np.testing.assert_allclose(frame.clip, kf.clip, atol=1e-9)
        up = np.asarray(kf.up) / np.linalg.norm(kf.up)
        np.testing.assert_allclose(frame.up, up, atol=1e-9)


def test_cyclic_closes_loop():
    keyframes = _keyframes(4)
    fb = 10
    frames = interpolate_cameras(keyframes, frames_between=fb, mode="cyclic")
    # Passes through every keyframe at knots and returns to keyframe 0.
    for k, kf in enumerate(keyframes):
        np.testing.assert_allclose(frames[k * fb].eye, kf.eye, atol=1e-9)
        np.testing.assert_allclose(frames[k * fb].ctr, kf.ctr, atol=1e-9)
    np.testing.assert_allclose(frames[-1].eye, keyframes[0].eye, atol=1e-9)
    np.testing.assert_allclose(frames[-1].ctr, keyframes[0].ctr, atol=1e-9)


@pytest.mark.parametrize("mode", ["linear", "spline", "cyclic"])
def test_up_stays_normalized(mode):
    keyframes = _keyframes(4)
    # Deliberately non-unit, tilted up vectors
    for k, kf in enumerate(keyframes):
        kf.up = (0.3 * (k % 2), 2.0 + k, 0.1)
    frames = interpolate_cameras(keyframes, frames_between=7, mode=mode)
    for frame in frames:
        assert abs(np.linalg.norm(frame.up) - 1.0) < 1e-9


def test_fov_lerps_monotonically():
    keyframes = _keyframes(4)  # fov 40, 50, 60, 70: increasing
    for mode in ("linear", "spline"):
        fovs = [c.fov for c in interpolate_cameras(keyframes, frames_between=9, mode=mode)]
        assert fovs[0] == pytest.approx(40.0)
        assert fovs[-1] == pytest.approx(70.0)
        assert all(b - a > -1e-12 for a, b in zip(fovs, fovs[1:]))


def test_spline_continuity():
    keyframes = _keyframes(5)
    fb = 20
    frames = interpolate_cameras(keyframes, frames_between=fb, mode="spline")
    eyes = np.array([f.eye for f in frames])
    steps = np.linalg.norm(np.diff(eyes, axis=0), axis=1)
    # Neighboring keyframes are ~3.5 apart; per-frame steps must stay small
    # (no jumps), i.e. well below the keyframe spacing.
    assert steps.max() < 3.5 / fb * 4.0


def test_threedgrut_mode_aliases():
    keyframes = _keyframes(4)
    reference = interpolate_cameras(keyframes, frames_between=5, mode="spline")
    for alias in ("path_spline", "path_smooth", "smooth", "catmull_rom"):
        frames = interpolate_cameras(keyframes, frames_between=5, mode=alias)
        assert [f.eye for f in frames] == [f.eye for f in reference]


def test_invalid_arguments():
    keyframes = _keyframes(4)
    with pytest.raises(ValueError, match="unknown mode"):
        interpolate_cameras(keyframes, mode="bezier")
    with pytest.raises(ValueError, match="at least 2"):
        interpolate_cameras(_keyframes(1), mode="linear")
    with pytest.raises(ValueError, match="at least 3"):
        interpolate_cameras(_keyframes(2), mode="cyclic")
    with pytest.raises(ValueError, match="frames_between"):
        interpolate_cameras(keyframes, frames_between=0)


# ---------------------------------------------------------------- trajectory


def test_trajectory_npy_roundtrip(tmp_path):
    keyframes = _keyframes(4)
    keyframes[2].dof_mode = 1
    keyframes[2].model = 1
    path = save_trajectory(str(tmp_path / "cameras.npy"), keyframes)
    assert os.path.isfile(path)

    loaded = load_trajectory(path)
    assert len(loaded) == len(keyframes)
    for cam, ref in zip(loaded, keyframes):
        np.testing.assert_allclose(cam.eye, ref.eye, atol=1e-9)
        np.testing.assert_allclose(cam.ctr, ref.ctr, atol=1e-9)
        # up is orthonormalized through the view matrix; the reference ups
        # here are unit and perpendicular to the view direction already
        np.testing.assert_allclose(cam.up, ref.up, atol=1e-9)
        assert cam.fov == pytest.approx(ref.fov)
        assert cam.clip == pytest.approx(ref.clip)
        assert cam.focus_dist == pytest.approx(ref.focus_dist)
        assert cam.aperture == pytest.approx(ref.aperture)
        assert cam.model == ref.model
        assert cam.dof_mode == ref.dof_mode


def test_trajectory_stores_threedgrut_frame_poses(tmp_path):
    """The stored c2w must be in the 3dgrut world frame (diag(1,-1,-1) away
    from VKGS), matching Camera.from_threedgrut_world."""
    cam = Camera(eye=(1.0, 2.0, 3.0), ctr=(0.0, 2.0, 0.0), up=(0.0, 1.0, 0.0))
    path = save_trajectory(str(tmp_path / "one.npy"), [cam])
    record = np.load(path)[0]
    c2w = record["c2w"]
    np.testing.assert_allclose(c2w, _M4 @ cam.to_camera_to_world(), atol=1e-12)
    # eye in the stored frame has y/z negated
    np.testing.assert_allclose(c2w[:3, 3], (1.0, -2.0, -3.0), atol=1e-12)


def test_load_plain_c2w_stack(tmp_path):
    """A bare (N, 4, 4) float stack of 3dgrut-frame camera-to-world matrices
    is accepted (fov defaults to 60)."""
    cams = _keyframes(3)
    stack = np.stack([_M4 @ c.to_camera_to_world() for c in cams])
    path = str(tmp_path / "stack.npy")
    np.save(path, stack)

    loaded = load_trajectory(path)
    assert len(loaded) == 3
    for cam, ref in zip(loaded, cams):
        np.testing.assert_allclose(cam.eye, ref.eye, atol=1e-9)
        assert cam.fov == 60.0


def test_save_empty_trajectory_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        save_trajectory(str(tmp_path / "empty.npy"), [])


def test_load_unrecognized_file(tmp_path):
    bad = tmp_path / "garbage.npy"
    bad.write_bytes(b"not a trajectory at all")
    with pytest.raises((ValueError, ImportError)):
        load_trajectory(str(bad))

    wrong_shape = tmp_path / "wrong.npy"
    np.save(str(wrong_shape), np.zeros((5, 3)))
    with pytest.raises(ValueError, match="unrecognized trajectory layout"):
        load_trajectory(str(wrong_shape))
