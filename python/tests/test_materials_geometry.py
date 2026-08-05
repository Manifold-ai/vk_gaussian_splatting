"""Tests for vkgs.materials (3dgrut preset parity) and vkgs.geometry (.obj writers)."""

import math
import os

import pytest

from vkgs import geometry, materials
from vkgs.project import Material

# Names that must mirror 3dgrut register_default_materials
# (threedgrut_playground/engine.py:424-561); 'checkboard' is texture-based
# and intentionally absent.
THREEDGRUT_NAMES = [
    "solid",
    "brushed_copper",
    "rose_gold",
    "blue_plastic",
    "oak_wood",
    "black_rubber",
    "polished_marble",
    "blue_glass",
    "jade",
    "diamond",
    "ruby_red",
    "luminous_yellow",
]


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------


def test_presets_registry_complete():
    for name in THREEDGRUT_NAMES + ["glass", "mirror", "diffuse"]:
        assert name in materials.PRESETS, name
        mat = materials.PRESETS[name]()
        assert isinstance(mat, Material)
        assert mat.name == name
    assert "checkboard" not in materials.PRESETS  # texture-based, unsupported


def test_brushed_copper_matches_3dgrut():
    # engine.py:461-469
    m = materials.brushed_copper()
    assert m.metallic == 1.0
    assert m.roughness == 0.5
    assert m.base_color == pytest.approx((0.95, 0.64, 0.54))
    assert m.ior == pytest.approx(1.1)
    assert m.transmission == 0.0
    assert m.opacity == 1.0


def test_diamond_matches_3dgrut():
    # engine.py:497-505
    m = materials.diamond()
    assert m.ior == pytest.approx(2.42)
    assert m.transmission == pytest.approx(0.99)
    assert m.base_color == pytest.approx((0.98, 0.98, 0.98))
    assert m.roughness == pytest.approx(0.02)
    assert m.opacity == pytest.approx(0.2)  # diffuse_factor alpha
    assert m.max_bounces >= 3  # path-traced transmissive material


def test_luminous_yellow_matches_3dgrut():
    # engine.py:515-523
    m = materials.luminous_yellow()
    assert m.emissive == pytest.approx((0.8, 0.8, 0.4))
    assert m.emissive_strength == 1.0
    assert m.base_color == pytest.approx((0.2, 0.9, 0.3))
    assert m.roughness == pytest.approx(0.7)
    assert m.metallic == 0.0


def test_generic_helpers_semantics():
    g = materials.glass()
    assert g.transmission == 1.0 and g.ior == 1.5 and g.max_bounces >= 3
    mi = materials.mirror()
    assert mi.metallic == 1.0 and mi.roughness == 0.0 and mi.max_bounces >= 3
    d = materials.diffuse(color=(0.2, 0.3, 0.4))
    assert d.metallic == 0.0 and d.roughness == 1.0
    assert d.base_color == pytest.approx((0.2, 0.3, 0.4))


def test_overrides_forwarded():
    m = materials.brushed_copper(roughness=0.2, max_bounces=7)
    assert m.roughness == 0.2 and m.max_bounces == 7
    # non-overridden fields keep preset values
    assert m.metallic == 1.0 and m.base_color == pytest.approx((0.95, 0.64, 0.54))
    g = materials.glass(ior=1.33)  # 3dgrut DEFAULT_REFRACTIVE_INDEX parity
    assert g.ior == 1.33
    with pytest.raises(TypeError):
        materials.jade(not_a_field=1.0)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def parse_obj(path):
    """Minimal OBJ parser returning (vertices, normals, uvs, faces) where
    faces are lists of (v_idx, vt_idx, vn_idx) 0-based triplets."""
    verts, normals, uvs, faces = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] == "#":
                continue
            if parts[0] == "v":
                verts.append(tuple(float(x) for x in parts[1:4]))
            elif parts[0] == "vn":
                normals.append(tuple(float(x) for x in parts[1:4]))
            elif parts[0] == "vt":
                uvs.append(tuple(float(x) for x in parts[1:3]))
            elif parts[0] == "f":
                face = []
                for chunk in parts[1:]:
                    ids = chunk.split("/")
                    v = int(ids[0]) - 1
                    vt = int(ids[1]) - 1 if len(ids) > 1 and ids[1] else None
                    vn = int(ids[2]) - 1 if len(ids) > 2 and ids[2] else None
                    face.append((v, vt, vn))
                faces.append(face)
    return verts, normals, uvs, faces


def test_quad_obj(tmp_path):
    path = geometry.write_quad_obj(str(tmp_path / "quad.obj"), size=2.0)
    verts, normals, uvs, faces = parse_obj(path)
    assert len(verts) == 4
    assert len(faces) == 2  # two triangles
    assert all(len(f) == 3 for f in faces)
    # unit quad scaled: edge length 2 => coordinates +-1, all y == 0
    for x, y, z in verts:
        assert y == 0.0
        assert abs(x) == pytest.approx(1.0) and abs(z) == pytest.approx(1.0)
    # normals up, referenced by every face vertex
    assert normals == [(0.0, 1.0, 0.0)]
    for face in faces:
        for v, vt, vn in face:
            assert vt is not None and vn == 0
    assert len(uvs) == 4 and set(uvs) == {(0, 0), (0, 1), (1, 1), (1, 0)}
    # winding: face normal must point +Y (front face up)
    import numpy as np

    for face in faces:
        a, b, c = (np.array(verts[i[0]]) for i in face)
        assert float(np.cross(b - a, c - b)[1]) > 0


def test_sphere_obj(tmp_path):
    radius, subdivisions = 2.5, 16
    path = geometry.write_sphere_obj(str(tmp_path / "sphere.obj"), radius=radius, subdivisions=subdivisions)
    verts, normals, uvs, faces = parse_obj(path)

    sectors, stacks = subdivisions, subdivisions // 2
    assert len(verts) == 2 + (stacks - 1) * sectors
    assert len(faces) == 2 * sectors * (stacks - 1)
    assert len(normals) == len(verts) and len(uvs) == len(verts)

    # all vertices on the sphere, normals unit and radial
    for (x, y, z), (nx, ny, nz) in zip(verts, normals):
        assert math.sqrt(x * x + y * y + z * z) == pytest.approx(radius, rel=1e-6)
        assert math.sqrt(nx * nx + ny * ny + nz * nz) == pytest.approx(1.0, rel=1e-6)
        assert (nx * x + ny * y + nz * z) == pytest.approx(radius, rel=1e-6)

    # closed manifold: Euler characteristic V - E + F == 2
    edges = set()
    for face in faces:
        idx = [i[0] for i in face]
        assert len(set(idx)) == 3  # no degenerate triangles
        for k in range(3):
            edges.add(frozenset((idx[k], idx[(k + 1) % 3])))
    euler = len(verts) - len(edges) + len(faces)
    assert euler == 2

    # every edge shared by exactly two faces (watertight)
    from collections import Counter

    counts = Counter()
    for face in faces:
        idx = [i[0] for i in face]
        for k in range(3):
            counts[frozenset((idx[k], idx[(k + 1) % 3]))] += 1
    assert set(counts.values()) == {2}

    # outward winding: face normal agrees with centroid direction
    import numpy as np

    for face in faces:
        a, b, c = (np.array(verts[i[0]]) for i in face)
        n = np.cross(b - a, c - b)
        assert float(np.dot(n, (a + b + c) / 3.0)) > 0


def test_plane_alias(tmp_path):
    p1 = geometry.write_quad_obj(str(tmp_path / "a.obj"), size=3.0)
    p2 = geometry.write_plane_obj(str(tmp_path / "b.obj"), size=3.0)
    with open(p1) as f1, open(p2) as f2:
        assert f1.read() == f2.read()


def test_ensure_procedural_caches(tmp_path):
    cache = str(tmp_path / "cache")
    p1 = geometry.ensure_procedural("Sphere", cache_dir=cache, radius=0.5)
    assert os.path.isfile(p1)
    mtime = os.path.getmtime(p1)
    p2 = geometry.ensure_procedural("sphere", cache_dir=cache, radius=0.5)
    assert p1 == p2
    assert os.path.getmtime(p2) == mtime  # not rewritten
    # different params -> different file
    p3 = geometry.ensure_procedural("sphere", cache_dir=cache, radius=1.0)
    assert p3 != p1
    # quad/plane kinds work and forward size
    q = geometry.ensure_procedural("Quad", cache_dir=cache, size=4.0)
    verts, _, _, _ = parse_obj(q)
    assert max(abs(c) for v in verts for c in v) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        geometry.ensure_procedural("torus", cache_dir=cache)
    with pytest.raises(TypeError):
        geometry.ensure_procedural("quad", cache_dir=cache, radius=1.0)
