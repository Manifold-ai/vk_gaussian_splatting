"""High-level one-call rendering API: Scene -> images.

``render_scene`` serializes the scene to .vkgs, generates a paired
render/save .cfg (see vkgs.sequence for the saveImage timing contract), runs
one headless subprocess and returns a :class:`RenderResult` that lazily loads
the saved buffers as numpy arrays.
"""

from __future__ import annotations

import copy
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import images
from .camera import Camera
from .constants import BUFFER_POSTFIXES, ColorBufferFormat, DofMode, Pipeline, SortingMethod
from .project import Scene
from .runner import HeadlessRunner, RunResult
from .sequence import RenderScript

# Minimum frames for the load/settle block: lets async scene loading, shader
# builds and sorting warm up before the first activateCameraPreset.
_MIN_SETTLE_FRAMES = 5

# Pipelines whose output is stochastic and converges via temporal
# accumulation (RTX/hybrids); raster pipelines only need it for
# sortStrategy 3 or DoF.
_STOCHASTIC_PIPELINES = {Pipeline.RTX, Pipeline.HYBRID, Pipeline.MESH_3DGUT, Pipeline.HYBRID_3DGUT}


class RenderResult:
    """Outputs of one render_scene call.

    ``camera`` arguments are positions in the ``cameras`` list passed to
    :func:`render_scene` (0-based), ``buffer`` names are BUFFER_POSTFIXES
    keys ("main", "depth", ...).
    """

    def __init__(self, paths: Dict[int, Dict[str, str]], run: RunResult, out_dir: str):
        self._paths = paths
        self._cache: Dict[Tuple[int, str], np.ndarray] = {}
        self.run = run
        self.out_dir = out_dir

    @property
    def sequences(self):
        """Parsed per-sequence timer data (see vkgs.runner.SequenceInfo)."""
        return self.run.sequences

    @property
    def log_path(self) -> str:
        return self.run.log_path

    def path(self, camera: int = 0, buffer: str = "main") -> str:
        try:
            return self._paths[camera][buffer]
        except KeyError:
            available = {cam: sorted(bufs) for cam, bufs in self._paths.items()}
            raise KeyError(f"no output for camera={camera} buffer={buffer!r}; available: {available}") from None

    def image(self, camera: int = 0, buffer: str = "main") -> np.ndarray:
        """Load (and cache) one buffer as a numpy array."""
        key = (camera, buffer)
        if key not in self._cache:
            self._cache[key] = images.load_image(self.path(camera, buffer))
        return self._cache[key]


def _resolve_cameras(scene: Scene, cameras) -> Tuple[Scene, List[int]]:
    """Map the cameras argument (preset indices and/or Camera objects) to
    preset indices, appending Camera objects to a shallow copy of the scene
    (the caller's scene is never mutated)."""
    if isinstance(cameras, (int, Camera)):
        cameras = [cameras]
    cameras = list(cameras)
    if not cameras:
        raise ValueError("cameras must contain at least one preset index or Camera")

    if any(isinstance(cam, Camera) for cam in cameras):
        scene = copy.copy(scene)
        scene.camera_presets = list(scene.camera_presets)

    indices: List[int] = []
    for cam in cameras:
        if isinstance(cam, Camera):
            indices.append(scene.add_camera_preset(cam))
        else:
            index = int(cam)
            if not 0 <= index < len(scene.camera_presets):
                raise ValueError(
                    f"camera preset index {index} out of range [0, {len(scene.camera_presets)}); "
                    "pass Camera objects or add presets with scene.add_camera_preset first"
                )
            indices.append(index)
    return scene, indices


def render_scene(
    scene: Scene,
    cameras: Union[int, Camera, Sequence[Union[int, Camera]]],
    *,
    size: Tuple[int, int] = (1920, 1080),
    spp: int = 64,
    buffers: Sequence[str] = ("main",),
    out_dir: str,
    hdr: bool = False,
    executable: Optional[str] = None,
    gpu: Optional[int] = None,
    keep_files: bool = True,
    timeout: float = 1800,
    extra_args: Sequence[str] = (),
) -> RenderResult:
    """Render ``scene`` from each camera and return the images.

    - ``cameras``: camera preset indices (as returned by
      ``scene.add_camera_preset``) and/or Camera objects (appended to a copy
      of the scene automatically).
    - ``spp``: frames accumulated per camera (sequenceframes). Stochastic /
      temporal pipelines need this >= the desired sample count to converge.
    - ``buffers``: which saved buffers to expect/read ("main", "depth",
      "normal", "aux1", and "ldr" when tonemapping is active).
    - ``hdr``: save .hdr (float32 RGB, rendered into an RGBA32F color
      buffer) instead of .png (RGBA8).
    - ``keep_files``: when False, images are loaded eagerly and the
      intermediate files (images, .vkgs, .cfg) are deleted; the log is kept.
    """
    buffers = list(buffers)
    unknown = [b for b in buffers if b not in BUFFER_POSTFIXES]
    if unknown:
        raise ValueError(f"unknown buffer(s) {unknown}; expected {sorted(BUFFER_POSTFIXES)}")
    if "ldr" in buffers and not scene.tonemapping.is_active:
        raise ValueError("buffer 'ldr' only exists when scene.tonemapping.is_active is set")
    if "comparison" in buffers:
        raise ValueError("buffer 'comparison' requires the interactive image-compare mode; not available here")

    spp = int(spp)
    if spp < 1:
        raise ValueError("spp must be >= 1")

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    scene, indices = _resolve_cameras(scene, cameras)
    _warn_if_underconverged(scene, indices, spp)
    vkgs_path = scene.save(os.path.join(out_dir, "scene.vkgs"))

    ext = ".hdr" if hdr else ".png"
    script = RenderScript(frames=spp, averages=min(spp, 32))
    # Load/settle block: the project loads during this sequence; camera
    # presets only become available after it. colorBufferFormat 2 keeps full
    # float32 precision for .hdr readback.
    script.load_block(
        frames=max(spp, _MIN_SETTLE_FRAMES),
        colorBufferFormat=ColorBufferFormat.RGBA32F if hdr else None,
    )

    expected: List[str] = []
    stems: Dict[int, str] = {}
    for position, preset in enumerate(indices):
        stem = script.capture(
            f"cam{position}",
            preset,
            os.path.join(out_dir, f"cam{position}"),
            frames=spp,
            buffers=buffers,
            hdr=hdr,
        )
        stems[position] = stem
        expected += [stem + BUFFER_POSTFIXES[b] + ext for b in buffers]

    cfg_path = script.write(os.path.join(out_dir, "render.cfg"))

    runner = HeadlessRunner(executable)
    run = runner.run(
        cfg_path,
        project=vkgs_path,
        size=size,
        gpu=gpu,
        timeout=timeout,
        log_path=os.path.join(out_dir, "render.log"),
        expected_outputs=expected,
        extra_args=extra_args,
    )

    # Map every buffer file actually present (robust to conditional buffers),
    # not just the requested ones.
    paths = {position: images.find_outputs(stem) for position, stem in stems.items()}
    result = RenderResult(paths, run, out_dir)

    if not keep_files:
        for position in stems:
            for buffer in buffers:
                result.image(position, buffer)  # eager-load into the cache
        for files in paths.values():
            for path in files.values():
                _remove_quiet(path)
        _remove_quiet(vkgs_path)
        _remove_quiet(cfg_path)

    return result


def _warn_if_underconverged(scene: Scene, indices: Sequence[int], spp: int) -> None:
    """Warn when the scene renders stochastically but spp is too low to
    converge (stochastic pipelines/sorting and DoF accumulate over frames;
    note DoF only takes effect in pipelines 2/4/5)."""
    stochastic = (
        scene.renderer.pipeline in _STOCHASTIC_PIPELINES
        or scene.renderer.sorting_method == SortingMethod.STOCHASTIC_SPLAT
        or any(scene.camera_presets[i].dof_mode != DofMode.DISABLED for i in indices)
    )
    if stochastic and spp < 8:
        warnings.warn(
            f"scene renders stochastically but spp={spp}; expect noise "
            "(sequenceframes must cover the desired sample count)",
            stacklevel=3,
        )


def _remove_quiet(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
