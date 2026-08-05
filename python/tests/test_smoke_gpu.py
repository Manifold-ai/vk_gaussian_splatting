"""GPU smoke tests: launch the real renderer headless (pytest -m gpu).

Skipped automatically when the executable is not built or no test splat file
is found. Set $VKGS_TEST_PLY to point at any .ply to force a specific asset.
"""

import glob
import os

import numpy as np
import pytest

from vkgs.camera import Camera
from vkgs.constants import Pipeline
from vkgs.facade import render_scene
from vkgs.images import find_outputs
from vkgs.project import Scene
from vkgs.runner import HeadlessRunner, find_executable
from vkgs.sequence import RenderScript

pytestmark = pytest.mark.gpu

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _find_executable_or_none():
    try:
        return find_executable()
    except FileNotFoundError:
        return None


def _find_test_ply():
    env = os.environ.get("VKGS_TEST_PLY")
    if env and os.path.isfile(env):
        return env
    # flowers_1.ply is the bundled default scene the app downloads next to the
    # binary; fall back to any .ply under the usual resource dirs.
    for pattern in (
        os.path.join(REPO_ROOT, "_bin", "**", "flowers_1.ply"),
        os.path.join(REPO_ROOT, "downloaded_resources", "**", "flowers_1.ply"),
        os.path.join(REPO_ROOT, "_bin", "**", "*.ply"),
        os.path.join(REPO_ROOT, "downloaded_resources", "**", "*.ply"),
    ):
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]
    return None


EXECUTABLE = _find_executable_or_none()
TEST_PLY = _find_test_ply()

needs_gpu_assets = pytest.mark.skipif(
    EXECUTABLE is None or TEST_PLY is None,
    reason="requires a built vk_gaussian_splatting executable and a test .ply "
    "(set $VKGS_BIN / $VKGS_TEST_PLY)",
)


def make_scene():
    scene = Scene()
    scene.renderer.pipeline = Pipeline.MESH
    scene.add_splats(TEST_PLY)
    return scene


@needs_gpu_assets
def test_render_scene_two_presets(tmp_path):
    scene = make_scene()
    presets = [
        scene.add_camera_preset(Camera(eye=(1.7, 1.5, 1.7), ctr=(0, 0, 0))),
        scene.add_camera_preset(Camera(eye=(-1.7, 1.5, 1.7), ctr=(0, 0, 0))),
    ]

    result = render_scene(
        scene,
        presets,
        size=(640, 480),
        spp=8,
        buffers=("main",),
        out_dir=str(tmp_path / "out"),
    )

    for camera in range(2):
        path = result.path(camera, "main")
        assert os.path.isfile(path) and path.endswith("_main.png")
        image = result.image(camera, "main")
        assert image.shape[0] == 480 and image.shape[1] == 640
        assert image.max() > 0, "rendered image is all black"

    # The two viewpoints must produce different images
    a = result.image(0, "main").astype(np.float64)
    b = result.image(1, "main").astype(np.float64)
    assert np.abs(a - b).mean() > 0.5

    # Sequencer ran the expected blocks: load + (render, save) per camera
    names = [s.name for s in result.sequences]
    assert "cam0" in names and "cam1" in names


@needs_gpu_assets
def test_saveimage_pairing_semantics(tmp_path):
    """Empirically verify the render/save pairing: saveImage captures the
    framebuffer from the END of the PREVIOUS sequence, so two capture pairs
    with visually distinct settings (maxShDegree 0 vs 3) must yield two
    different _main images. If the save landed in its own (renamed-settings)
    sequence instead, both images would show the same state."""
    scene = make_scene()
    preset = scene.add_camera_preset(Camera(eye=(1.7, 1.5, 1.7), ctr=(0, 0, 0)))
    out_dir = tmp_path / "pairing"
    out_dir.mkdir()
    vkgs_path = scene.save(str(out_dir / "scene.vkgs"))

    script = RenderScript(frames=8, averages=4)
    script.load_block(frames=32)
    stem0 = script.capture("sh0", preset, str(out_dir / "sh0"), max_sh_degree=0)
    stem3 = script.capture("sh3", preset, str(out_dir / "sh3"), max_sh_degree=3)
    cfg = script.write(str(out_dir / "pairing.cfg"))

    runner = HeadlessRunner(str(EXECUTABLE))
    runner.run(
        cfg,
        project=vkgs_path,
        size=(640, 480),
        expected_outputs=[stem0 + "_main.png", stem3 + "_main.png"],
    )

    outputs0 = find_outputs(stem0)
    outputs3 = find_outputs(stem3)
    assert "main" in outputs0 and "main" in outputs3

    from vkgs.images import load_image

    sh0 = load_image(outputs0["main"]).astype(np.float64)
    sh3 = load_image(outputs3["main"]).astype(np.float64)
    assert sh0.shape == sh3.shape
    assert sh0.max() > 0 and sh3.max() > 0
    # SH degree 0 drops all view-dependent color: images must differ measurably
    assert np.abs(sh0 - sh3).mean() > 0.1
