"""Tests for vkgs.facade format selection (no GPU/executable needed).

The headless subprocess is stubbed out (HeadlessRunner.run), so these assert
on the generated .cfg text and the predicted output paths only.
"""

import pytest

from vkgs import facade
from vkgs.camera import Camera
from vkgs.facade import _resolve_ext
from vkgs.project import Scene


def test_resolve_ext():
    # image_format wins and is normalized (lowercase, leading dot)
    assert _resolve_ext(False, "raw") == ".raw"
    assert _resolve_ext(False, ".raw") == ".raw"
    assert _resolve_ext(False, "RAW") == ".raw"
    assert _resolve_ext(True, "png") == ".png"  # explicit format beats hdr=True
    # legacy hdr bool fallback when image_format is None
    assert _resolve_ext(True, None) == ".hdr"
    assert _resolve_ext(False, None) == ".png"
    with pytest.raises(ValueError, match="image_format"):
        _resolve_ext(False, "tiff")


def _stub_runner(monkeypatch):
    """Replace the whole HeadlessRunner (its __init__ resolves the executable
    eagerly) with a stub that captures the written .cfg and the expected-output
    list, and returns a dummy run object."""
    captured = {}

    class _StubRunner:
        def __init__(self, executable=None):
            pass

        def run(self, cfg_path, **kw):
            with open(cfg_path) as f:
                captured["cfg"] = f.read()
            captured["expected"] = list(kw.get("expected_outputs", []))
            return object()

    monkeypatch.setattr(facade, "HeadlessRunner", _StubRunner)
    return captured


def test_render_scene_raw_uses_float_colorbuffer_and_ext(tmp_path, monkeypatch):
    captured = _stub_runner(monkeypatch)
    facade.render_scene(
        Scene(),
        Camera(eye=(0, 0, 3)),
        size=(8, 8),
        spp=4,
        buffers=["main"],
        out_dir=str(tmp_path / "o"),
        image_format="raw",
    )
    assert "--colorBufferFormat 2" in captured["cfg"]  # RGBA32F for float readback
    assert captured["expected"]
    assert all(p.endswith("_main.raw") for p in captured["expected"])


def test_render_scene_png_default_no_float_colorbuffer(tmp_path, monkeypatch):
    captured = _stub_runner(monkeypatch)
    facade.render_scene(
        Scene(),
        Camera(eye=(0, 0, 3)),
        size=(8, 8),
        spp=4,
        buffers=["main"],
        out_dir=str(tmp_path / "o"),
    )
    assert "--colorBufferFormat" not in captured["cfg"]
    assert all(p.endswith("_main.png") for p in captured["expected"])
