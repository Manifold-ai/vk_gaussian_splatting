"""Tests for vkgs.facade format selection (no GPU/executable needed).

The headless subprocess is stubbed out (HeadlessRunner.run), so these assert
on the generated .cfg text and the predicted output paths only.
"""

import os
import struct

import numpy as np
import pytest

from vkgs import facade, images
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


class _FileStubRun:
    log_path = ""
    sequences = []
    warnings = []
    log_text = ""


def _stub_runner_files(monkeypatch):
    """Stub runner that also creates each expected output file plus one extra
    unrequested buffer (_aux1 per stem), to exercise selective reclaim / logs."""
    captured = {}

    class _StubRunner:
        def __init__(self, executable=None):
            pass

        def run(self, cfg_path, **kw):
            captured["expected"] = list(kw.get("expected_outputs", []))
            captured["log_path"] = kw.get("log_path")
            for p in captured["expected"]:
                open(p, "wb").close()
                base, ext = os.path.splitext(p)
                stem = base.rsplit("_", 1)[0]  # ".../cam0_main" -> ".../cam0"
                open(stem + "_aux1" + ext, "wb").close()  # unrequested dump
            return _FileStubRun()

    monkeypatch.setattr(facade, "HeadlessRunner", _StubRunner)
    return captured


def test_keep_buffers_reclaims_all_but_requested(tmp_path, monkeypatch):
    _stub_runner_files(monkeypatch)
    out = tmp_path / "o"
    result = facade.render_scene(
        Scene(), Camera(eye=(0, 0, 3)), size=(8, 8), spp=4,
        buffers=["main", "depth"], out_dir=str(out), image_format="raw",
        keep_buffers=["main"], log=False,  # log=False so the tmp log self-cleans
    )
    assert os.path.isfile(result.path(0, "main"))          # requested + kept
    with pytest.raises(KeyError):
        result.path(0, "depth")                            # requested but not kept
    assert not os.path.exists(out / "cam0_depth.raw")      # deleted from disk
    assert not os.path.exists(out / "cam0_aux1.raw")       # unrequested dump gone
    assert not os.path.exists(out / "scene.vkgs")
    assert not os.path.exists(out / "render.cfg")


def test_log_goes_to_render_logs_dir_and_is_kept(tmp_path, monkeypatch):
    captured = _stub_runner_files(monkeypatch)
    out = tmp_path / "o"
    facade.render_scene(
        Scene(), Camera(eye=(0, 0, 3)), size=(8, 8), spp=4,
        buffers=["main"], out_dir=str(out), image_format="raw", log=True,
    )
    log_path = captured["log_path"]
    assert os.path.basename(os.path.dirname(log_path)) == "vkgs_render_logs"
    assert os.path.isfile(log_path)                        # kept when log=True
    assert not os.path.exists(out / "render.log")          # never in out_dir
    os.remove(log_path)                                    # don't leak into $TMPDIR


def test_log_false_deletes_log_after_parsing(tmp_path, monkeypatch):
    captured = _stub_runner_files(monkeypatch)
    facade.render_scene(
        Scene(), Camera(eye=(0, 0, 3)), size=(8, 8), spp=4,
        buffers=["main"], out_dir=str(tmp_path / "o"), image_format="raw", log=False,
    )
    assert not os.path.exists(captured["log_path"])        # captured, then removed


def test_load_raw_uint_roundtrip(tmp_path):
    # header {w, h, channels=1, bytesPerChannel=4} + row-major uint32 payload
    w, h = 3, 2
    vals = np.array([[10, 0xFFFFFFFF, 7], [0, 0xFFFFFFFF, 2]], dtype=np.uint32)
    path = tmp_path / "cam0_instance_id.raw"
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", w, h, 1, 4))
        f.write(vals.tobytes())
    got = images.load_raw_uint(str(path))
    assert got.shape == (2, 3) and got.dtype == np.uint32
    assert np.array_equal(got, vals)  # sentinel 0xFFFFFFFF preserved, not float-reinterpreted
