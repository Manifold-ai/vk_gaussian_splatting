import os
import re

import pytest

from vkgs.sequence import RenderScript, _cfg_name


def blocks_of(text):
    """Split cfg text into (name, [param lines]) blocks."""
    blocks = []
    name = None
    lines = []
    for line in text.splitlines():
        match = re.match(r'^SEQUENCE "(.*)"$', line)
        if match:
            if name is not None:
                blocks.append((name, lines))
            name, lines = match.group(1), []
        elif line.strip() and not line.startswith("#"):
            assert name is not None, f"param line before first SEQUENCE: {line}"
            lines.append(line)
    if name is not None:
        blocks.append((name, lines))
    return blocks


def test_load_block_is_first_and_lazy(tmp_path):
    script = RenderScript(frames=16)
    script.capture("cam0", 0, str(tmp_path / "cam0"))
    blocks = blocks_of(script.text())
    assert blocks[0][0] == "Load scene and settle"
    assert "--sequenceframes 16" in blocks[0][1]
    # activateCameraPreset must never appear in the load block (presets only
    # exist after the project finishes loading)
    assert not any("activateCameraPreset" in line for line in blocks[0][1])


def test_explicit_load_block_must_be_first():
    script = RenderScript()
    script.sequence("something")
    with pytest.raises(RuntimeError, match="first"):
        script.load_block()


def test_capture_emits_render_save_pair(tmp_path):
    script = RenderScript(frames=64, averages=32)
    script.load_block(frames=128)
    script.capture("cam0", 3, str(tmp_path / "shot"), frames=64)
    blocks = blocks_of(script.text())

    assert [name for name, _ in blocks] == ["Load scene and settle", "cam0", "cam0 save"]

    render = blocks[1][1]
    save = blocks[2][1]

    # Render sequence: activates the preset, accumulates frames, does NOT save
    assert "--activateCameraPreset 3" in render
    assert "--sequenceframes 64" in render
    assert not any("saveImage" in line for line in render)

    # Save sequence: 1 frame, saveImageBuffer -1 BEFORE saveImage
    assert "--sequenceframes 1" in save
    buffer_idx = next(i for i, l in enumerate(save) if l.startswith("--saveImageBuffer"))
    image_idx = next(i for i, l in enumerate(save) if l.startswith("--saveImage "))
    assert save[buffer_idx] == "--saveImageBuffer -1"
    assert buffer_idx < image_idx


def test_capture_paths_absolute_and_quoted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = RenderScript()
    script.capture("cam0", 0, "relative/stem")  # relative on purpose
    text = script.text()
    match = re.search(r'--saveImage "([^"]+)"', text)
    assert match, text
    assert os.path.isabs(match.group(1))
    assert match.group(1).endswith(".png")


def test_capture_hdr_extension(tmp_path):
    script = RenderScript()
    script.capture("cam0", 0, str(tmp_path / "shot"), hdr=True)
    assert re.search(r'--saveImage "[^"]+\.hdr"', script.text())


def test_capture_rejects_unknown_buffer(tmp_path):
    script = RenderScript()
    with pytest.raises(ValueError, match="unknown buffer"):
        script.capture("cam0", 0, str(tmp_path / "shot"), buffers=["main", "bogus"])


def test_no_screenshot_ever(tmp_path):
    script = RenderScript()
    with pytest.raises(ValueError, match="screenshot"):
        script.sequence("bad", screenshot="x.png")
    script.capture("cam0", 0, str(tmp_path / "shot"))
    assert "--screenshot" not in script.text()


def test_param_name_mapping():
    assert _cfg_name("max_sh_degree") == "maxShDegree"
    assert _cfg_name("sh_format") == "shformat"
    assert _cfg_name("rgba_format") == "rgbaformat"
    assert _cfg_name("lighting_enabled") == "lightingEnabled"
    assert _cfg_name("shadows_mode") == "shadowMode"
    assert _cfg_name("sorting_method") == "sortStrategy"
    assert _cfg_name("rtx_trace_strategy") == "rtxSortStrategy"
    assert _cfg_name("use_aabbs") == "useAABBs"
    assert _cfg_name("color_buffer_format") == "colorBufferFormat"
    assert _cfg_name("sequence_frames") == "sequenceframes"
    # exact spellings pass through
    assert _cfg_name("maxShDegree") == "maxShDegree"
    assert _cfg_name("updateData") == "updateData"


def test_value_formatting():
    script = RenderScript(frames=8, averages=4)
    script.load_block(
        pipeline=2,
        lighting_enabled=True,
        wireframe=False,
        kernel_min_response=0.0113,
        update_data=True,
    )
    (_, lines), = blocks_of(script.text())
    assert "--pipeline 2" in lines
    assert "--lightingEnabled 1" in lines
    assert "--wireframe 0" in lines
    assert "--kernelMinResponse 0.0113" in lines
    assert "--updateData" in lines  # flag param: bare, no value


def test_flag_false_and_none_omitted():
    script = RenderScript()
    script.load_block(update_data=False, colorBufferFormat=None)
    (_, lines), = blocks_of(script.text())
    assert not any("updateData" in line for line in lines)
    assert not any("colorBufferFormat" in line for line in lines)


def test_every_block_restates_sequencer_params(tmp_path):
    """Blocks must not rely on sequenceframes persisting across sequences."""
    script = RenderScript(frames=10, averages=5, reset_frames=2)
    script.load_block()
    script.sequence("s1")
    script.capture("cam0", 0, str(tmp_path / "s"))
    for _, lines in blocks_of(script.text()):
        assert any(l.startswith("--sequenceframes ") for l in lines)
        assert any(l.startswith("--sequenceaverages ") for l in lines)
        assert any(l.startswith("--sequenceresetframes ") for l in lines)


def test_snapshot(tmp_path):
    """Full-text snapshot of a two-camera script."""
    out = tmp_path / "out"
    script = RenderScript(frames=4, averages=2)
    script.load_block(frames=8, colorBufferFormat=2)
    script.capture("cam0", 0, str(out / "cam0"), frames=4)
    script.capture("cam1", 1, str(out / "cam1"), frames=4)

    expected = f"""# Generated by vkgs.sequence.RenderScript

SEQUENCE "Load scene and settle"
--sequenceframes 8
--sequenceaverages 2
--sequenceresetframes 0
--colorBufferFormat 2

SEQUENCE "cam0"
--sequenceframes 4
--sequenceaverages 2
--sequenceresetframes 0
--activateCameraPreset 0

SEQUENCE "cam0 save"
--sequenceframes 1
--sequenceaverages 1
--sequenceresetframes 0
--saveImageBuffer -1
--saveImage "{out / 'cam0'}.png"

SEQUENCE "cam1"
--sequenceframes 4
--sequenceaverages 2
--sequenceresetframes 0
--activateCameraPreset 1

SEQUENCE "cam1 save"
--sequenceframes 1
--sequenceaverages 1
--sequenceresetframes 0
--saveImageBuffer -1
--saveImage "{out / 'cam1'}.png"
"""
    assert script.text() == expected


def test_write(tmp_path):
    script = RenderScript()
    script.load_block()
    path = script.write(str(tmp_path / "sub" / "run.cfg"))
    assert os.path.isfile(path)
    with open(path) as f:
        assert f.read() == script.text()


def test_empty_script_raises():
    with pytest.raises(RuntimeError, match="empty"):
        RenderScript().text()
