"""Camera trajectory interpolation and headless video rendering.

Pure-Python reimplementation of the 3dgrut playground ``VideoRecorder``
(threedgrut_playground/utils/video_out.py) on top of the vkgs batch model:
keyframes are interpolated into per-frame cameras, every frame becomes a
camera preset in one .vkgs project, a single headless run captures all frames
(render+save sequence pair per frame, see vkgs.sequence), and the resulting
PNGs are assembled into an MP4 with imageio-ffmpeg.

Interpolation modes (mirroring VideoRecorder.MODES, video_out.py:33):
- "linear":  straight lerp of eye/ctr between consecutive keyframes.
- "spline":  Catmull-Rom on eye/ctr with mirrored phantom endpoints, so the
  path passes through every keyframe (video_out.py 'path_spline' /
  interpolate_camera_on_spline_path). 3dgrut's 'path_smooth' mode (polynomial
  least-squares smoothstep, interpolated_cameras.py:_smoothstep) is mapped to
  this mode: both are smooth paths through the keyframes, we simply keep one
  spline implementation instead of the 7th-order smoothstep.
- "cyclic":  closed Catmull-Rom loop with wraparound control points; the
  path returns to keyframe 0 (video_out.py render_continuous_trajectory,
  which fits a periodic bspline with splprep(per=1)).

In every mode ``up`` is nlerp'd (normalized lerp) and fov / aperture /
focus_dist / clip are lerp'd per segment.
"""

from __future__ import annotations

import copy
import math
import os
import tempfile
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import images
from .camera import THREEDGRUT_TO_VKGS, Camera

# User-facing mode -> canonical mode. The path_* names are accepted for
# 3dgrut VideoRecorder script compatibility.
_MODE_ALIASES = {
    "linear": "linear",
    "spline": "spline",
    "smooth": "spline",
    "catmull_rom": "spline",
    "path_spline": "spline",  # 3dgrut name
    "path_smooth": "spline",  # 3dgrut name (polynomial smoothstep -> spline)
    "cyclic": "cyclic",
}

# 4x4 homogeneous world change of basis 3dgrut <-> VKGS (self-inverse).
_M4 = np.eye(4)
_M4[:3, :3] = THREEDGRUT_TO_VKGS


# --------------------------------------------------------------- interpolation


def _catmull_rom(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    """Uniform Catmull-Rom; passes through p1 at t=0 and p2 at t=1
    (same polynomial as 3dgrut interpolated_cameras.py:_catmull_rom)."""
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
    )


def _lerp(a, b, t: float):
    return a * (1.0 - t) + b * t


def _nlerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Normalized lerp for up vectors; falls back to normalize(a) when the
    interpolant degenerates (a ~ -b)."""
    v = _lerp(a, b, t)
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        v = a
        norm = np.linalg.norm(v)
    return v / norm


def _control_points(points: np.ndarray, seg: int, cyclic: bool):
    """Catmull-Rom control quad (p0..p3) for segment seg -> seg+1."""
    n = len(points)
    if cyclic:
        return (
            points[(seg - 1) % n],
            points[seg % n],
            points[(seg + 1) % n],
            points[(seg + 2) % n],
        )
    p1, p2 = points[seg], points[seg + 1]
    # Mirrored phantom endpoints keep the open curve through the first/last
    # keyframes without needing >= 4 keyframes.
    p0 = points[seg - 1] if seg > 0 else 2.0 * points[0] - points[1]
    p3 = points[seg + 2] if seg + 2 < n else 2.0 * points[n - 1] - points[n - 2]
    return p0, p1, p2, p3


def interpolate_cameras(
    keyframes: Sequence[Camera],
    frames_between: int = 30,
    mode: str = "spline",
) -> List[Camera]:
    """Interpolate keyframe cameras into a dense per-frame trajectory.

    - ``mode``: "spline" (Catmull-Rom, default), "linear", or "cyclic"
      (closed loop). 3dgrut names "path_spline"/"path_smooth" are accepted
      aliases of "spline" (see module docstring).
    - Open modes return ``(len(keyframes)-1) * frames_between + 1`` cameras
      and pass exactly through every keyframe at frame ``k * frames_between``.
    - "cyclic" returns ``len(keyframes) * frames_between + 1`` cameras; the
      last frame equals keyframe 0 (the loop closes).

    eye/ctr follow the path; up is nlerp'd (always unit length in the
    output); fov/aperture/focus_dist/clip lerp per segment; discrete fields
    (model, dof_mode) snap to the nearer keyframe.
    """
    try:
        canonical = _MODE_ALIASES[mode]
    except KeyError:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(_MODE_ALIASES)}") from None
    keyframes = list(keyframes)
    n = len(keyframes)
    cyclic = canonical == "cyclic"
    if cyclic and n < 3:
        raise ValueError(f"cyclic mode needs at least 3 keyframes, got {n}")
    if n < 2:
        raise ValueError(f"interpolation needs at least 2 keyframes, got {n}")
    frames_between = int(frames_between)
    if frames_between < 1:
        raise ValueError("frames_between must be >= 1")

    eyes = np.array([k.eye for k in keyframes], dtype=np.float64)
    ctrs = np.array([k.ctr for k in keyframes], dtype=np.float64)
    ups = np.array([k.up for k in keyframes], dtype=np.float64)

    def frame_at(seg: int, t: float) -> Camera:
        a = keyframes[seg % n]
        b = keyframes[(seg + 1) % n]
        if canonical == "linear":
            eye = _lerp(eyes[seg], eyes[(seg + 1) % n], t)
            ctr = _lerp(ctrs[seg], ctrs[(seg + 1) % n], t)
        else:
            eye = _catmull_rom(*_control_points(eyes, seg, cyclic), t)
            ctr = _catmull_rom(*_control_points(ctrs, seg, cyclic), t)
        up = _nlerp(ups[seg % n], ups[(seg + 1) % n], t)
        nearest = a if t < 0.5 else b
        return Camera(
            model=nearest.model,
            eye=(float(eye[0]), float(eye[1]), float(eye[2])),
            ctr=(float(ctr[0]), float(ctr[1]), float(ctr[2])),
            up=(float(up[0]), float(up[1]), float(up[2])),
            fov=float(_lerp(a.fov, b.fov, t)),
            clip=(float(_lerp(a.clip[0], b.clip[0], t)), float(_lerp(a.clip[1], b.clip[1], t))),
            dof_mode=nearest.dof_mode,
            focus_dist=float(_lerp(a.focus_dist, b.focus_dist, t)),
            aperture=float(_lerp(a.aperture, b.aperture, t)),
        )

    segments = n if cyclic else n - 1
    frames = [
        frame_at(seg, j / frames_between)
        for seg in range(segments)
        for j in range(frames_between)
    ]
    frames.append(frame_at(segments - 1, 1.0))  # last keyframe (kf 0 if cyclic)
    return frames


# --------------------------------------------------------------- trajectories

# Our .npy layout: one structured record per keyframe. "c2w" is the
# kaolin-convention camera-to-world matrix (camera looks down -Z, columns =
# right/up/backward) expressed in the 3dgrut world frame (PLY-native), i.e.
# poses convert through THREEDGRUT_TO_VKGS on both save and load, matching
# vkgs.camera.Camera.from_threedgrut_world.
_TRAJECTORY_DTYPE = np.dtype(
    [
        ("c2w", "<f8", (4, 4)),
        ("fov", "<f8"),
        ("focus", "<f8"),  # eye->ctr distance, restores ctr exactly
        ("near", "<f8"),
        ("far", "<f8"),
        ("model", "<i4"),
        ("dof_mode", "<i4"),
        ("focus_dist", "<f8"),
        ("aperture", "<f8"),
    ]
)


def save_trajectory(path: str, keyframes: Sequence[Camera]) -> str:
    """Save keyframes as a .npy trajectory (pure numpy, no pickle).

    Poses are stored as kaolin-convention camera-to-world matrices in the
    3dgrut world frame (see _TRAJECTORY_DTYPE) so they interoperate with
    kaolin/3dgrut tooling. Note that 3dgrut's own VideoRecorder.save_trajectory
    is ``torch.save`` of kaolin Camera objects (video_out.py:92-93) despite the
    default "cameras.npy" name; :func:`load_trajectory` reads both formats.
    """
    keyframes = list(keyframes)
    if not keyframes:
        raise ValueError("cannot save an empty trajectory")
    records = np.zeros(len(keyframes), dtype=_TRAJECTORY_DTYPE)
    for record, cam in zip(records, keyframes):
        record["c2w"] = _M4 @ cam.to_camera_to_world()  # VKGS -> 3dgrut world
        record["fov"] = cam.fov
        record["focus"] = np.linalg.norm(np.subtract(cam.ctr, cam.eye))
        record["near"], record["far"] = cam.clip
        record["model"] = int(cam.model)
        record["dof_mode"] = int(cam.dof_mode)
        record["focus_dist"] = cam.focus_dist
        record["aperture"] = cam.aperture
    abspath = os.path.abspath(path)
    os.makedirs(os.path.dirname(abspath), exist_ok=True)
    np.save(abspath, records)
    return abspath


def load_trajectory(path: str) -> List[Camera]:
    """Load a trajectory saved by :func:`save_trajectory` or by 3dgrut's
    VideoRecorder.

    Accepted layouts:
    - our structured .npy (see save_trajectory);
    - a plain (N, 4, 4) float array of kaolin-convention camera-to-world
      matrices in the 3dgrut world frame (fov defaults to 60);
    - a ``torch.save``d list of kaolin Cameras, which is what 3dgrut's
      VideoRecorder writes (video_out.py:92-96) — requires torch + kaolin.
    """
    abspath = os.path.abspath(path)
    loaded = None
    try:
        loaded = np.load(abspath, allow_pickle=False)
    except (ValueError, OSError):
        loaded = None  # pickled / not a numpy file: try the torch path
    if isinstance(loaded, np.lib.npyio.NpzFile):
        # torch.save zip archives parse as "npz" files; not a numpy trajectory
        loaded.close()
        loaded = None
    if loaded is not None:
        return _cameras_from_array(abspath, loaded)
    return _cameras_from_torch(abspath)


def _cameras_from_array(path: str, data: np.ndarray) -> List[Camera]:
    if data.dtype.names and "c2w" in data.dtype.names:
        names = data.dtype.names
        cams = []
        for record in data:
            cam = Camera.from_threedgrut_world(
                np.asarray(record["c2w"], dtype=np.float64),
                fov=float(record["fov"]),
                focus=float(record["focus"]) if "focus" in names else 1.0,
            )
            if "near" in names and "far" in names:
                cam.clip = (float(record["near"]), float(record["far"]))
            if "model" in names:
                cam.model = int(record["model"])
            if "dof_mode" in names:
                cam.dof_mode = int(record["dof_mode"])
            if "focus_dist" in names:
                cam.focus_dist = float(record["focus_dist"])
            if "aperture" in names:
                cam.aperture = float(record["aperture"])
            cams.append(cam)
        return cams
    if data.ndim == 3 and data.shape[1:] == (4, 4):
        return [Camera.from_threedgrut_world(np.asarray(c2w, dtype=np.float64)) for c2w in data]
    raise ValueError(
        f"{path}: unrecognized trajectory layout {data.dtype} shape {data.shape}; "
        "expected the vkgs structured dtype or an (N, 4, 4) camera-to-world stack"
    )


def _cameras_from_torch(path: str) -> List[Camera]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            f"{path} is not a numpy trajectory; reading 3dgrut VideoRecorder "
            "trajectories (torch.save of kaolin cameras) requires torch + kaolin"
        ) from exc
    try:
        trajectory = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"{path}: not a numpy trajectory and torch.load failed: {exc}") from exc
    return [Camera.from_kaolin(cam) for cam in trajectory]


# --------------------------------------------------------------------- video


def render_video(
    scene,
    keyframes: Sequence[Camera],
    out: str = "out.mp4",
    *,
    mode: str = "spline",
    frames_between: int = 30,
    fps: int = 30,
    spp: int = 32,
    size: Tuple[int, int] = (1280, 720),
    executable: Optional[str] = None,
    workdir: Optional[str] = None,
    keep_frames: bool = False,
    gpu: Optional[int] = None,
) -> str:
    """Render a camera trajectory through ``scene`` into an MP4.

    Interpolates ``keyframes`` (see :func:`interpolate_cameras`), adds every
    frame as a camera preset on a deep copy of the scene (the caller's scene
    is never touched), renders all frames in ONE headless run (a render+save
    sequence pair per frame), then assembles the PNGs with imageio-ffmpeg.

    - ``spp``: frames accumulated per video frame (sequenceframes); keep
      >= sample count for stochastic pipelines/DoF.
    - ``workdir``: where frame PNGs / scene.vkgs / render.cfg / render.log
      go; a temp directory when None.
    - ``keep_frames``: keep the per-frame PNGs and project files (the log is
      always kept).

    Returns the absolute path of the written MP4. Widths/heights divisible
    by 16 avoid an imageio macro-block resize.
    """
    try:
        import imageio.v2 as iio
        import imageio_ffmpeg  # noqa: F401  (registers the ffmpeg writer)
    except ImportError as exc:
        raise ImportError(
            "render_video requires imageio and imageio-ffmpeg (pip install 'vkgs[video]')"
        ) from exc

    from .facade import render_scene  # local import to avoid a cycle

    frames = interpolate_cameras(keyframes, frames_between=frames_between, mode=mode)

    out_path = os.path.abspath(out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="vkgs_video_")
    workdir = os.path.abspath(workdir)

    scene = copy.deepcopy(scene)
    indices = [scene.add_camera_preset(cam) for cam in frames]

    total = len(frames)
    print(f"render_video: rendering {total} frames at {size[0]}x{size[1]}, spp={spp} (workdir: {workdir})")
    result = render_scene(
        scene,
        indices,
        size=size,
        spp=spp,
        buffers=("main",),
        out_dir=workdir,
        executable=executable,
        gpu=gpu,
        keep_files=True,
        # generous per-frame budget; headless exits when the sequencer ends
        timeout=max(1800.0, 10.0 * total),
    )

    print(f"render_video: encoding {total} frames -> {out_path}")
    step = max(1, math.ceil(total / 10))  # progress every ~10%
    writer = iio.get_writer(out_path, fps=int(fps))
    try:
        for i in range(total):
            frame = np.asarray(iio.imread(result.path(i, "main")))
            writer.append_data(frame[..., :3])
            if (i + 1) % step == 0 or i + 1 == total:
                print(f"render_video: {i + 1}/{total} frames ({100 * (i + 1) // total}%)")
    finally:
        writer.close()

    if not keep_frames:
        for i in range(total):
            main_path = result.path(i, "main")
            stem = main_path[: main_path.rfind("_main")]
            for buffer_path in images.find_outputs(stem).values():
                _remove_quiet(buffer_path)
        _remove_quiet(os.path.join(workdir, "scene.vkgs"))
        _remove_quiet(os.path.join(workdir, "render.cfg"))
    print(f"render_video: done ({out_path}; log: {result.log_path})")
    return out_path


def _remove_quiet(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
