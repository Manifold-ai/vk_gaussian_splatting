import json
import math
import os

import pytest

from vkgs.camera import Camera
from vkgs.constants import EnvMode, Format, LightType, Pipeline
from vkgs.project import Material, Scene

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAMPLES = os.path.join(REPO_ROOT, "samples")

REQUIRED_LOADER_KEYS = {
    # A missing one of these throws inside the C++ loadProject catch-all and
    # the whole project silently fails to load (vkgs_project_reader.cpp).
    "splatSets": ["id", "path"],
    "splats": ["splatSetId"],
    "meshAssets": ["id", "path"],
    "lights.assets": ["id"],
    "lights.instances": ["assetId"],
}


def build_scene(tmp_path):
    scene = Scene()
    scene.renderer.pipeline = Pipeline.HYBRID
    scene.renderer.lighting_enabled = True
    scene.renderer.gs_shadow_mask = True
    scene.renderer.gs_shadow_mask_min = 0.35
    scene.renderer.gs_shadow_mask_from_particles = True
    scene.renderer.force_surfel = True
    scene.add_splats(str(tmp_path / "garden.ply"), position=(1, 2, 3), rotation=(0, 90, 0), scale=2.0)
    scene.add_splats(str(tmp_path / "garden.ply"), name="copy")  # dedup asset
    scene.add_mesh(
        str(tmp_path / "teapot.obj"),
        position=(0, 0.5, 0),
        materials=[Material(name="glass", transmission=1.0, ior=1.5, roughness=0.0)],
    )
    scene.add_light(LightType.POINT, translation=(2, 3, 1), intensity=40.0, radius=0.1)
    scene.add_light(LightType.DIRECTIONAL, rotation=(45, 0, 0), intensity=2.0)
    # gs-shadow light: only darkens the GS shadow mask, radius=0 = hard shadow
    scene.add_light(LightType.DIRECTIONAL, rotation=(60, 0, 0), radius=0.0, shadow_only=True)
    scene.set_environment(EnvMode.HDR, hdr_file=str(tmp_path / "env.hdr"), ibl_intensity=1.2)
    scene.set_tonemapping(True, exposure=1.5, method=2)
    scene.set_camera(Camera(eye=(3, 1.5, 2), ctr=(0, 0.5, 0)))
    for z in (2.0, 0.0, -2.0):
        scene.add_camera_preset(Camera(eye=(3, 1.5, z), ctr=(0, 0.5, 0), fov=50.0))
    return scene


def test_roundtrip_equality(tmp_path):
    scene = build_scene(tmp_path)
    path = scene.save(str(tmp_path / "scene.vkgs"))

    loaded = Scene.load(path)
    resaved = loaded.save(str(tmp_path / "scene2.vkgs"))

    with open(path) as f:
        first = json.load(f)
    with open(resaved) as f:
        second = json.load(f)
    assert first == second


def test_asset_dedup(tmp_path):
    scene = build_scene(tmp_path)
    assert len(scene.splat_sets) == 1
    assert len(scene.splat_instances) == 2
    assert scene.splat_instances[1].splat_set_id == 0


def test_required_fields_present(tmp_path):
    scene = build_scene(tmp_path)
    data = scene.to_json(str(tmp_path))

    assert data["version"] == 7
    for entry in data["splatSets"]:
        for key in REQUIRED_LOADER_KEYS["splatSets"]:
            assert key in entry
    for entry in data["splats"]:
        for key in REQUIRED_LOADER_KEYS["splats"]:
            assert key in entry
    for entry in data["meshAssets"]:
        for key in REQUIRED_LOADER_KEYS["meshAssets"]:
            assert key in entry
    for entry in data["meshInstances"]["items"]:
        assert "meshAssetId" in entry
    # v3+ lights branch requires both arrays even when empty
    assert "assets" in data["lights"] and "instances" in data["lights"]
    for entry in data["lights"]["assets"]:
        assert "id" in entry
    for entry in data["lights"]["instances"]:
        assert "assetId" in entry
    # Splat transforms only applied when all three keys present (reader.cpp:372)
    for entry in data["splats"]:
        assert {"position", "rotation", "scale"} <= set(entry)


def test_paths_relative_posix(tmp_path):
    scene = Scene()
    sub = tmp_path / "assets" / "deep"
    scene.add_splats(str(sub / "model.ply"))
    scene.save(str(tmp_path / "proj" / "scene.vkgs"))
    with open(tmp_path / "proj" / "scene.vkgs") as f:
        data = json.load(f)
    assert data["splatSets"][0]["path"] == "../assets/deep/model.ply"

    loaded = Scene.load(str(tmp_path / "proj" / "scene.vkgs"))
    assert loaded.splat_sets[0].path == str(sub / "model.ply")


def test_empty_scene_sections(tmp_path):
    data = Scene().to_json(str(tmp_path))
    # All top-level sections the C++ writer emits are present
    for key in [
        "version", "renderer", "camera", "cameras", "lights", "splatsGlobals",
        "splatSets", "splats", "meshAssets", "meshInstances", "environment",
        "settings", "tonemapping",
    ]:
        assert key in data
    assert data["lights"]["assets"] == []
    assert data["meshInstances"]["items"] == []


@pytest.mark.parametrize(
    "sample",
    [
        "3dgs_winter_house_objects_on_stove_lighting.vkgs",
        "3dgs_large_city_sky_environment_lighting.vkgs",
        "3dgs_winter_house_chessboard_light_from_splats.vkgs",
    ],
)
def test_golden_sample_roundtrip(sample, tmp_path):
    """Load a C++-written sample project, re-save, reload: the modeled fields
    must survive unchanged (numeric tolerance for float32 storage)."""
    path = os.path.join(SAMPLES, sample)
    if not os.path.exists(path):
        pytest.skip("samples not present")

    scene = Scene.load(path)
    resaved = scene.save(str(tmp_path / "resaved.vkgs"))
    scene2 = Scene.load(resaved)

    with open(path) as f:
        original = json.load(f)

    # Structural checks against the original file
    assert len(scene.splat_sets) == len(original.get("splatSets", []))
    assert len(scene.splat_instances) == len(original.get("splats", []))
    assert len(scene.mesh_assets) == len(original.get("meshAssets", []))
    assert len(scene.light_assets) == len(original.get("lights", {}).get("assets", []))
    assert len(scene.camera_presets) == len(original.get("cameras", []))
    assert scene.renderer.pipeline == original["renderer"]["pipeline"]

    # Python-side round-trip must be exact
    assert scene.to_json(str(tmp_path)) == scene2.to_json(str(tmp_path))

    # Spot-check float fidelity vs original
    cam0 = original["cameras"][0]
    assert scene.camera_presets[0].eye == pytest.approx(tuple(cam0["eye"]), abs=1e-6)
    assert scene.camera_presets[0].fov == pytest.approx(cam0["fov"], abs=1e-6)


def test_gs_shadow_mask_serialization(tmp_path):
    """The v7-optional GS shadow mask keys must use the exact C++ names
    (vkgs_project_writer.cpp:156-159,271) and survive save/load."""
    scene = build_scene(tmp_path)
    data = scene.to_json(str(tmp_path))

    renderer = data["renderer"]
    assert renderer["gsShadowMask"] is True
    assert renderer["gsShadowMaskMin"] == pytest.approx(0.35)
    assert renderer["gsShadowMaskFromParticles"] is True
    assert renderer["forceSurfel"] is True

    # shadowOnly is an int 0/1 like enabled (LightSourceVk convention)
    assert [a["shadowOnly"] for a in data["lights"]["assets"]] == [0, 0, 1]
    assert data["lights"]["assets"][2]["radius"] == 0.0

    loaded = Scene.load(scene.save(str(tmp_path / "mask.vkgs")))
    assert loaded.renderer.gs_shadow_mask is True
    assert loaded.renderer.gs_shadow_mask_min == pytest.approx(0.35)
    assert loaded.renderer.gs_shadow_mask_from_particles is True
    assert loaded.renderer.force_surfel is True
    assert [a.shadow_only for a in loaded.light_assets] == [False, False, True]


def test_gs_shadow_mask_defaults_omit_nothing(tmp_path):
    """Defaults mirror src/parameters.h:185-188; the keys are always written
    (the C++ reader treats them as optional via LOAD1)."""
    renderer = Scene().to_json(str(tmp_path))["renderer"]
    assert renderer["gsShadowMask"] is False
    assert renderer["gsShadowMaskMin"] == pytest.approx(0.2)
    assert renderer["gsShadowMaskFromParticles"] is False
    assert renderer["forceSurfel"] is False


def test_version_gate(tmp_path):
    bad = tmp_path / "old.vkgs"
    bad.write_text(json.dumps({"version": 3, "splats": []}))
    with pytest.raises(ValueError, match="version 3"):
        Scene.load(str(bad))
