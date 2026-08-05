"""Tests for vkgs.compat (3dgrut playground shim) — no GPU/executable needed."""

import json
import math

import numpy as np
import pytest

from vkgs.camera import THREEDGRUT_TO_VKGS, Camera, rotation_matrix_from_euler_deg
from vkgs.constants import CameraModel, DofMode, LightType as VkgsLightType, Pipeline, ShadowsMode
from vkgs.compat import (
    CompatWarning,
    EngineVKGS,
    Light,
    LightType,
    OptixPrimitiveTypes,
    camera_to_vkgs,
    direction_to_euler_deg,
    light_to_vkgs,
    primitive_type_to_material,
    transform_to_vkgs_trs,
)
from vkgs.compat.convert import SOFT_SHADOW_REFERENCE_DISTANCE, area_light_to_quad
from vkgs.compat.primitives import PrimitivesVKGS, autoscale_factor, mesh_file_extent, ply_scene_extent
from vkgs.compat.tonemap import filmic_curve, tonemap


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def write_toy_ply(path, points):
    """Minimal binary_little_endian splat-style PLY with x/y/z floats."""
    points = np.asarray(points, dtype="<f4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    ).encode("ascii")
    with open(path, "wb") as f:
        f.write(header + points.tobytes())
    return str(path)


@pytest.fixture
def toy_ply(tmp_path):
    # extent (2, 2, 2): "small scene" branch of the autoscale heuristic
    corners = [[x, y, z] for x in (0, 2) for y in (0, 2) for z in (0, 2)]
    return write_toy_ply(tmp_path / "toy.ply", corners)


@pytest.fixture
def engine(toy_ply, tmp_path):
    return EngineVKGS(toy_ply, out_dir=str(tmp_path / "out"))


# --------------------------------------------------------------------------
# Enum / constructor mirroring
# --------------------------------------------------------------------------


def test_optix_primitive_types_mirror_3dgrut():
    # engine.py:196-209: names()[t.value] == display name
    assert [int(t) for t in OptixPrimitiveTypes] == [0, 1, 2, 3, 4, 5]
    assert OptixPrimitiveTypes.names()[OptixPrimitiveTypes.SHADOW_CATCHER] == "Shadow Catcher"
    assert [int(t) for t in LightType] == [0, 1, 2, 3]  # NONE/DIRECTIONAL/POINT/AREA


def test_engine_rejects_pt_and_ingp_with_export_hint(tmp_path):
    for ext in (".pt", ".ingp"):
        with pytest.raises(ValueError, match="export_ply"):
            EngineVKGS(str(tmp_path / f"ckpt{ext}"))
    with pytest.raises(ValueError, match="Unknown object type|unknown object type"):
        EngineVKGS(str(tmp_path / "scene.xyz"))


def test_engine_seeds_glass_sphere_like_3dgrut(engine):
    # Engine3DGRUT adds a glass Sphere on construction (engine.py:1024-1026)
    assert list(engine.primitives.objects) == ["Sphere 1"]
    assert engine.primitives.objects["Sphere 1"].primitive_type == OptixPrimitiveTypes.GLASS


def test_default_config_warns_ignored(toy_ply):
    with pytest.warns(CompatWarning, match="default_config"):
        EngineVKGS(toy_ply, default_config="apps/colmap_3dgrt.yaml")


# --------------------------------------------------------------------------
# Property setters (lazy state)
# --------------------------------------------------------------------------


def test_property_setters_mutate_state(engine):
    engine.camera_type = "Fisheye"
    assert engine.camera_type == "Fisheye"
    with pytest.raises(ValueError):
        engine.camera_type = "Orthographic"

    with pytest.warns(CompatWarning):
        engine.antialiasing_mode = "8x MSAA"
    assert engine.spp.spp == 8 and engine.spp.mode == "msaa"
    with pytest.warns(CompatWarning):
        engine.antialiasing_mode = "Quasi-Random (Sobol)"
    assert engine.spp.mode == "low_discrepancy_seq"
    assert engine.spp.spp == 8  # Sobol keeps the count

    engine.max_pbr_bounces = 7
    assert engine._build_scene().renderer.rtx_max_bounces == 7

    engine.use_depth_of_field = True
    engine.depth_of_field.aperture_size = 0.02
    engine.depth_of_field.focus_z = 2.5
    engine.pipeline = Pipeline.RTX
    cam, _ = engine._prepare_camera(Camera())
    assert cam.dof_mode == DofMode.FIXED_FOCUS
    assert cam.focus_dist == pytest.approx(2.5)
    assert cam.aperture == pytest.approx(0.02)


def test_unsupported_props_warn_compat(engine):
    with pytest.warns(CompatWarning, match="denoiser"):
        engine.use_optix_denoiser = True
    with pytest.warns(CompatWarning, match="shadow_min"):
        engine.shadow_min = 0.2
    with pytest.warns(CompatWarning, match="shadow_spp"):
        engine.shadow_spp = 64
    with pytest.warns(CompatWarning, match="envmap_offset"):
        engine.environment.envmap_offset = [0.1, 0.5]
    with pytest.warns(CompatWarning, match="depth of field"):
        engine.use_depth_of_field = True
        engine.pipeline = Pipeline.HYBRID  # not a DoF pipeline
        engine._prepare_camera(Camera())


def test_fisheye_applied_to_converted_cameras_only(engine):
    engine.camera_type = "Fisheye"
    cam, _ = engine._prepare_camera(np.eye(4))
    assert cam.model == CameraModel.FISHEYE
    authored = Camera(model=CameraModel.PINHOLE)
    cam, _ = engine._prepare_camera(authored)
    assert cam.model == CameraModel.PINHOLE  # explicit vkgs Camera wins


# --------------------------------------------------------------------------
# Scene flush
# --------------------------------------------------------------------------


def test_flush_builds_glass_sphere_project(engine, tmp_path):
    scene = engine._build_scene()
    assert scene.renderer.pipeline == Pipeline.HYBRID
    assert len(scene.splat_instances) == 1 and scene.splat_instances[0].show
    assert len(scene.mesh_instances) == 1
    mat = scene.mesh_instances[0].materials[0]
    assert mat.transmission == 1.0
    assert mat.ior == pytest.approx(1.33)  # 3dgrut DEFAULT_REFRACTIVE_INDEX

    path = scene.save(str(tmp_path / "flush.vkgs"))
    data = json.loads(open(path).read())
    item = data["meshInstances"]["items"][0]
    assert item["materials"][0]["transmission"] == 1.0
    assert item["materials"][0]["ior"] == pytest.approx(1.33)
    assert data["splats"][0]["show"] is True


def test_flush_pbr_keeps_authored_materials_and_none_hides(engine):
    engine.primitives.objects["Sphere 1"].primitive_type = OptixPrimitiveTypes.PBR
    scene = engine._build_scene()
    assert scene.mesh_instances[0].materials == []  # keep glTF/OBJ materials
    engine.primitives.objects["Sphere 1"].primitive_type = OptixPrimitiveTypes.NONE
    assert engine._build_scene().mesh_instances[0].show is False
    engine.primitives.objects["Sphere 1"].primitive_type = OptixPrimitiveTypes.GLASS
    engine.primitives.enabled = False
    assert engine._build_scene().mesh_instances[0].show is False


def test_disable_gaussian_tracing_hides_splat_instance(engine):
    engine.disable_gaussian_tracing = True
    assert engine._build_scene().splat_instances[0].show is False


def test_renderer_overrides_applied_and_validated(engine):
    engine.renderer_overrides["sorting_method"] = 3
    assert engine._build_scene().renderer.sorting_method == 3
    engine.renderer_overrides["no_such_setting"] = 1
    with pytest.raises(AttributeError):
        engine._build_scene()


# --------------------------------------------------------------------------
# Lights
# --------------------------------------------------------------------------


def test_directional_light_direction_roundtrips_through_euler():
    d = np.array([0.3, -0.5, 0.2])
    d /= np.linalg.norm(d)
    spec = light_to_vkgs(Light(light_type=LightType.DIRECTIONAL, direction=tuple(d)))
    assert spec["kind"] == "light" and spec["type"] == VkgsLightType.DIRECTIONAL
    # VKGS light emits along R @ (0,0,-1); 3dgrut direction points TO the
    # light, world change of basis diag(1,-1,-1).
    emitted = rotation_matrix_from_euler_deg(spec["rotation"]) @ np.array([0.0, 0.0, -1.0])
    expected = THREEDGRUT_TO_VKGS @ (-d)
    assert np.abs(emitted - expected).max() < 1e-6


def test_direction_to_euler_handles_poles():
    for d in ((0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)):
        emitted = rotation_matrix_from_euler_deg(direction_to_euler_deg(d)) @ np.array([0.0, 0.0, -1.0])
        assert np.abs(emitted - np.asarray(d)).max() < 1e-9


def test_point_light_position_flipped():
    spec = light_to_vkgs(Light(light_type=LightType.POINT, position=(1.0, 2.0, 3.0), intensity=40.0))
    assert spec["type"] == VkgsLightType.POINT
    assert spec["translation"] == pytest.approx((1.0, -2.0, -3.0))
    assert spec["intensity"] == 40.0
    assert spec["soft"] is False


def test_soft_shadow_radius_heuristic(engine):
    theta = 0.05
    spec = light_to_vkgs(Light(light_type=LightType.POINT, position=(0, 1, 0), angular_radius=theta))
    assert spec["soft"] is True
    assert spec["radius"] == pytest.approx(math.tan(theta) * SOFT_SHADOW_REFERENCE_DISTANCE)

    engine.add_light(light_type=int(LightType.POINT), position=(0, 1, 0), angular_radius=theta)
    scene = engine._build_scene()
    assert scene.renderer.lighting_enabled is True
    assert scene.renderer.shadows_mode == ShadowsMode.SOFT
    assert scene.light_assets[0].radius == pytest.approx(math.tan(theta) * SOFT_SHADOW_REFERENCE_DISTANCE)


def test_area_light_warns_and_becomes_emissive_quad(engine):
    light = Light(
        light_type=LightType.AREA,
        position=(1.0, 2.0, 3.0),
        tangent_u=(0.5, 0.0, 0.0),
        tangent_v=(0.0, 0.0, 0.25),
        color=(1.0, 0.5, 0.2),
        intensity=8.0,
    )
    with pytest.warns(CompatWarning, match="AREA"):
        quad = area_light_to_quad(light)
    assert quad["position"] == pytest.approx((1.0, -2.0, -3.0))
    # local X spans 2*|tangent_u|, local Z spans 2*|tangent_v|
    assert sorted(np.abs(quad["scale"])) == pytest.approx([0.5, 1.0, 1.0])
    assert quad["material"].emissive == pytest.approx((1.0, 0.5, 0.2))
    assert quad["material"].emissive_strength == pytest.approx(8.0)

    engine.add_light(light)
    with pytest.warns(CompatWarning, match="AREA"):
        scene = engine._build_scene()
    quads = [m for m in scene.mesh_instances if m.name == "area light quad"]
    assert len(quads) == 1
    assert quads[0].materials[0].emissive_strength == pytest.approx(8.0)
    assert scene.light_assets == []  # no analytic light created for AREA


def test_light_crud_mirrors_engine3dgrut(engine):
    index = engine.add_light(light_type=int(LightType.POINT), position=(0, 1, 0))
    assert index == 0
    engine.update_light(index, intensity=5.0, color=(1, 0, 0))
    assert engine.lights[0].intensity == 5.0
    with pytest.raises(AttributeError):
        engine.update_light(index, not_a_field=1)
    engine.remove_light(index)
    engine.add_light(Light())
    engine.clear_lights()
    assert engine.lights == []


def test_none_light_skipped():
    assert light_to_vkgs(Light(light_type=LightType.NONE))["kind"] == "skip"


# --------------------------------------------------------------------------
# Shadow catcher / raygen (unsupported surface)
# --------------------------------------------------------------------------


def test_shadow_catcher_raises_everywhere(engine):
    with pytest.warns(CompatWarning):
        with pytest.raises(NotImplementedError, match="shadow-catcher|SHADOW_CATCHER"):
            primitive_type_to_material(OptixPrimitiveTypes.SHADOW_CATCHER)
    with pytest.warns(CompatWarning):
        with pytest.raises(NotImplementedError):
            engine.primitives.add_primitive("Sphere", OptixPrimitiveTypes.SHADOW_CATCHER)
    with pytest.warns(CompatWarning):
        with pytest.raises(NotImplementedError):
            engine.load_shadow_catcher("floor.obj")
    assert list(engine.primitives.objects) == ["Sphere 1"]  # nothing was added


def test_raygen_raises_with_workaround(engine):
    with pytest.raises(NotImplementedError, match="ray-buffer"):
        engine.raygen(None)


# --------------------------------------------------------------------------
# Materials mapping
# --------------------------------------------------------------------------


def test_primitive_type_material_mapping():
    mirror = primitive_type_to_material(OptixPrimitiveTypes.MIRROR)
    assert mirror.metallic == 1.0 and mirror.roughness == 0.0
    glass = primitive_type_to_material(OptixPrimitiveTypes.GLASS, ior=1.5)
    assert glass.transmission == 1.0 and glass.ior == 1.5
    diffuse = primitive_type_to_material(OptixPrimitiveTypes.DIFFUSE)
    assert diffuse.metallic == 0.0 and diffuse.roughness == 1.0
    assert primitive_type_to_material(OptixPrimitiveTypes.PBR) is None
    assert primitive_type_to_material(OptixPrimitiveTypes.NONE) is None


# --------------------------------------------------------------------------
# Tonemap (exact 3dgrut formulas)
# --------------------------------------------------------------------------


def test_tonemap_reinhard_matches_hand_computed():
    # Hand-computed from environment.py:148-155 for a uniform 1.0 image:
    # luminance = 0.2126+0.7152+0.0722 = 1.0; log-avg = 1+eps;
    # scaled = 0.18/(1+eps); toneL = scaled/(1+scaled); ldr = toneL/(1+eps)
    hdr = np.ones((2, 2, 3), dtype=np.float32)
    eps = 1e-6
    scaled = 0.18 / (1.0 + eps)
    expected = (scaled / (1.0 + scaled)) / (1.0 + eps)
    out = tonemap(hdr, "Reinhard")
    assert out == pytest.approx(np.full_like(hdr, expected), rel=1e-5)


def test_tonemap_filmic_and_exposure_and_none():
    hdr = np.full((1, 1, 3), 0.5, dtype=np.float32)
    # exposure applies 2**exposure before the curve (environment.py:136)
    out = tonemap(hdr, "Filmic", exposure=1.0)
    expected = filmic_curve(1.0) / filmic_curve(11.2)
    assert out == pytest.approx(np.full_like(hdr, expected), rel=1e-5)
    assert tonemap(hdr, "None", exposure=2.0) == pytest.approx(np.full_like(hdr, 2.0))
    with pytest.raises(ValueError):
        tonemap(hdr, "ACES")
    nan_in = np.full((1, 1, 3), np.nan, dtype=np.float32)
    assert np.all(tonemap(nan_in, "None") == 0.0)  # nan_to_num like 3dgrut


# --------------------------------------------------------------------------
# Transforms / cameras
# --------------------------------------------------------------------------


def test_transform_to_vkgs_trs_flips_world():
    T = np.eye(4)
    T[:3, 3] = (1.0, 2.0, 3.0)
    T[:3, :3] = np.diag([2.0, 2.0, 2.0])
    position, rotation, scale = transform_to_vkgs_trs(T)
    assert position == pytest.approx((1.0, -2.0, -3.0))
    assert scale == pytest.approx((2.0, 2.0, 2.0))
    # diag(1,-1,-1) is a 180-degree rotation about X
    R = rotation_matrix_from_euler_deg(rotation)
    assert np.allclose(R, THREEDGRUT_TO_VKGS)


def test_camera_to_vkgs_matrix_and_passthrough():
    c2w = np.eye(4)
    c2w[:3, 3] = (0.0, 0.0, 5.0)  # 3dgrut frame, looking down -Z
    cam, hint = camera_to_vkgs(c2w, fov=50.0)
    assert hint is None
    assert cam.fov == pytest.approx(50.0)
    assert cam.eye == pytest.approx((0.0, 0.0, -5.0))  # flipped world
    original = Camera(eye=(1, 2, 3), fov=33.0)
    copy_cam, _ = camera_to_vkgs(original)
    assert copy_cam == original and copy_cam is not original
    with pytest.raises(TypeError):
        camera_to_vkgs("not a camera")


# --------------------------------------------------------------------------
# Primitives manager
# --------------------------------------------------------------------------


def test_asset_registry_capitalizes_stems(tmp_path):
    (tmp_path / "teapot.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 1\nf 1 2 3\n")
    (tmp_path / "notes.txt").write_text("ignored")
    prims = PrimitivesVKGS(str(tmp_path))
    assert "Teapot" in prims.assets and prims.assets["Teapot"].endswith("teapot.obj")
    assert "Notes" not in prims.assets
    assert prims.assets["Quad"] is None and prims.assets["Sphere"] is None  # procedural
    with pytest.raises(KeyError, match="Armadillo"):
        prims.add_primitive("Armadillo", OptixPrimitiveTypes.GLASS)


def test_primitive_naming_add_remove_duplicate():
    prims = PrimitivesVKGS()
    assert prims.add_primitive("Sphere", OptixPrimitiveTypes.GLASS, device="cuda") == "Sphere 1"
    assert prims.add_primitive("Sphere", OptixPrimitiveTypes.MIRROR) == "Sphere 2"
    assert prims.duplicate_primitive("Sphere 1") == "Sphere 3"
    assert prims.objects["Sphere 3"].primitive_type == OptixPrimitiveTypes.GLASS
    prims.remove_primitive("Sphere 2")
    assert sorted(prims.objects) == ["Sphere 1", "Sphere 3"]
    assert prims.has_visible_objects()


def test_autoscale_replicates_set_mesh_scale_to_scene():
    # engine.py:293-325: 1/mesh_max, then *0.5*scene_max for scenes <= 5
    assert autoscale_factor((2, 2, 2), (2, 2, 2)) == pytest.approx(0.5)
    assert autoscale_factor((4, 1, 1), (1, 1, 1)) == pytest.approx(2.0)
    # large scenes: unit normalization only
    assert autoscale_factor((10, 1, 1), (2, 2, 2)) == pytest.approx(0.5)


def test_external_primitive_not_autoscaled(tmp_path, toy_ply):
    obj = tmp_path / "proxy.obj"
    obj.write_text("v 0 0 0\nv 4 0 0\nv 0 4 0\nf 1 2 3\n")
    prims = PrimitivesVKGS(scene_extent=(2, 2, 2))
    name = prims.load_external_primitive(str(obj), OptixPrimitiveTypes.DIFFUSE)
    assert name == "Proxy 1"
    assert prims.objects[name].scale == (1.0, 1.0, 1.0)  # as-authored (C9)


def test_set_transform_from_3dgrut_matrix():
    prims = PrimitivesVKGS()
    name = prims.add_primitive("Quad", OptixPrimitiveTypes.DIFFUSE)
    T = np.eye(4)
    T[:3, 3] = (0.0, 1.0, 0.0)
    prims.objects[name].set_transform(T)
    assert prims.objects[name].position == pytest.approx((0.0, -1.0, 0.0))


# --------------------------------------------------------------------------
# Extent scanners
# --------------------------------------------------------------------------


def test_ply_scene_extent_binary_and_ascii(tmp_path):
    binary = write_toy_ply(tmp_path / "b.ply", [[0, 0, 0], [1, 2, 4]])
    assert ply_scene_extent(binary) == pytest.approx((1.0, 2.0, 4.0))
    ascii_ply = tmp_path / "a.ply"
    ascii_ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n0 0 0\n3 1 2\n"
    )
    assert ply_scene_extent(str(ascii_ply)) == pytest.approx((3.0, 1.0, 2.0))


def test_ply_scene_extent_fallback_warns(tmp_path):
    with pytest.warns(CompatWarning, match="autoscale"):
        assert ply_scene_extent(str(tmp_path / "missing.ply")) == (1.0, 1.0, 1.0)


def test_mesh_file_extent_obj_and_gltf(tmp_path):
    obj = tmp_path / "tri.obj"
    obj.write_text("v -1 0 0\nv 1 0 0\nv 0 3 0\nf 1 2 3\n")
    assert mesh_file_extent(str(obj)) == pytest.approx((2.0, 3.0, 0.0))
    gltf = tmp_path / "box.gltf"
    gltf.write_text(json.dumps({
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"min": [-1, -2, -3], "max": [1, 2, 3]}],
    }))
    assert mesh_file_extent(str(gltf)) == pytest.approx((2.0, 4.0, 6.0))


# --------------------------------------------------------------------------
# Dirty-state / cached-buffer logic (no subprocess)
# --------------------------------------------------------------------------


def _seed_cache(engine, cam):
    rgb = np.zeros((1, 4, 4, 3), dtype=np.float32)
    opacity = np.ones((1, 4, 4, 1), dtype=np.float32)
    engine._cache_last_state(cam, {"rgb": rgb, "opacity": opacity, "rgb_buffer": rgb})


def test_dirty_state_machine(engine):
    cam = Camera(eye=(3, 1, 3))
    assert not engine.has_cached_buffers()
    assert engine.is_dirty(cam)  # no cache yet

    _seed_cache(engine, cam)
    assert engine.has_cached_buffers()
    assert not engine.is_dirty(cam)
    assert not engine.did_camera_change(cam)
    assert not engine.has_progressive_effects_to_render()

    moved = Camera(eye=(0, 0, 9))
    assert engine.did_camera_change(moved)
    assert engine.is_dirty(moved)

    engine.gamma_correction = 2.2  # any state mutation dirties
    assert engine.is_dirty(cam)
    _seed_cache(engine, cam)
    assert not engine.is_dirty(cam)

    engine.invalidate_materials_on_gpu()
    assert engine.is_dirty(cam)


def test_render_pass_serves_cache_when_clean(engine):
    cam = Camera(eye=(3, 1, 3))
    _seed_cache(engine, cam)
    # No executable involved: a clean state must return the cached buffers.
    rb = engine.render_pass(cam, is_first_pass=False)
    assert rb["rgb"] is engine.last_state["rgb"]
    assert rb["rgb_buffer"] is engine.last_state["rgb_buffer"]
    assert rb["opacity"] is engine.last_state["opacity"]


class _StubResult:
    """Duck-typed vkgs.facade.RenderResult (only .image is consumed)."""

    def __init__(self, image):
        self._image = image

    def image(self, position, buffer):
        assert buffer == "main"
        return self._image


def test_postprocess_contract_shapes_tonemap_gamma(engine):
    hdr = np.full((4, 6, 3), 4.0, dtype=np.float32)  # .hdr readback: RGB float
    engine.environment.tonemapper = "None"
    engine.environment.exposure = -1.0  # halves: 4.0 -> 2.0, clipped to 1.0
    engine.gamma_correction = 2.0
    rb = engine._postprocess(_StubResult(hdr), 0)
    assert rb["rgb"].shape == (1, 4, 6, 3) and rb["rgb"].dtype == np.float32
    assert rb["opacity"].shape == (1, 4, 6, 1)  # all-ones: hdr has no alpha
    assert np.all(rb["opacity"] == 1.0)
    assert rb["rgb_buffer"] is rb["rgb"]
    # 4.0 * 2**-1 = 2.0 -> gamma sqrt(2.0) ~ 1.414 -> clipped to 1.0
    assert np.all(rb["rgb"] == 1.0)

    engine.environment.exposure = -3.0  # 4.0 * 0.125 = 0.5 -> sqrt = 0.7071
    rb = engine._postprocess(_StubResult(hdr), 0)
    assert rb["rgb"] == pytest.approx(np.full((1, 4, 6, 3), math.sqrt(0.5)), rel=1e-6)

    rgba = np.zeros((4, 6, 4), dtype=np.float32)  # future .raw path keeps alpha
    rgba[..., 3] = 0.25
    rb = engine._postprocess(_StubResult(rgba), 0)
    assert np.all(rb["opacity"] == 0.25)
