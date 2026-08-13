"""Factor-only PBR material presets, named to match the 3dgrut playground.

The named presets copy the exact factors of
``Engine3DGRUT.register_default_materials``
(3dgrut threedgrut_playground/engine.py:424-561, PBRMaterial dataclass
:97-117), translated onto :class:`vkgs.project.Material`
(shaders/shading.h factors) as follows:

===================  =========================================
3dgrut PBRMaterial   vkgs Material
===================  =========================================
diffuse_factor[:3]   base_color
diffuse_factor[3]    opacity
metallic_factor      metallic
roughness_factor     roughness
transmission_factor  transmission
ior                  ior
emissive_factor      emissive (with emissive_strength = 1.0)
===================  =========================================

Texture-based 3dgrut presets:

- ``checkboard`` (procedural 512x512 checker diffuse_map, engine.py:426-437)
  has no factor-only equivalent and is NOT provided here; approximate it with
  ``diffuse(color=(0.375, 0.375, 0.375))`` (the mean of its two squares) or
  author a textured mesh asset instead.
- ``solid`` (engine.py:439-450) technically uses a diffuse_map too, but the
  map is a constant color (130, 193, 255)/255, so the factor translation here
  is exact.

The generic helpers mirror the ``OptixPrimitiveTypes`` semantics of the
playground (engine.py:196-209): MIRROR -> :func:`mirror`, GLASS ->
:func:`glass` (3dgrut's default refractive index is 1.33, engine.py:346),
DIFFUSE -> :func:`diffuse`.

``max_bounces`` defaults are chosen for VKGS path tracing (pipelines
RTX/HYBRID/HYBRID_3DGUT): transmissive materials get enough bounces for
enter/exit refraction plus total internal reflection, metals enough for
mirror chains. Every factory forwards ``**overrides`` to
:class:`vkgs.project.Material`, so any field can be overridden, e.g.
``brushed_copper(roughness=0.2)``.
"""

from __future__ import annotations

from .project import Material

__all__ = [
    "glass",
    "mirror",
    "diffuse",
    "flat",
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
    "PRESETS",
]


def _make(overrides: dict, **defaults) -> Material:
    """Build a Material from preset defaults, letting overrides win."""
    defaults.update(overrides)
    return Material(**defaults)


# --------------------------------------------------------------------------
# Generic helpers (OptixPrimitiveTypes parity)
# --------------------------------------------------------------------------


def glass(ior: float = 1.5, transmission: float = 1.0, color=(1.0, 1.0, 1.0), roughness: float = 0.0, **overrides) -> Material:
    """Clear refractive glass (3dgrut OptixPrimitiveTypes.GLASS).

    Note: 3dgrut's GLASS primitive uses a fixed default refractive index of
    1.33 (engine.py:346 DEFAULT_REFRACTIVE_INDEX); pass ``ior=1.33`` for
    exact parity with an untouched playground glass primitive.
    """
    return _make(
        overrides,
        name="glass",
        base_color=tuple(color),
        metallic=0.0,
        roughness=roughness,
        transmission=transmission,
        ior=ior,
        max_bounces=8,
    )


def mirror(color=(1.0, 1.0, 1.0), **overrides) -> Material:
    """Perfect mirror (3dgrut OptixPrimitiveTypes.MIRROR): metallic=1, roughness=0."""
    return _make(
        overrides,
        name="mirror",
        base_color=tuple(color),
        metallic=1.0,
        roughness=0.0,
        max_bounces=4,
    )


def diffuse(color=(0.7, 0.7, 0.7), **overrides) -> Material:
    """Pure Lambertian surface (3dgrut OptixPrimitiveTypes.DIFFUSE)."""
    return _make(
        overrides,
        name="diffuse",
        base_color=tuple(color),
        metallic=0.0,
        roughness=1.0,
        max_bounces=3,
    )


def flat(color=(1.0, 1.0, 1.0), unlit: bool = True, **overrides) -> Material:
    """Flat reference material for matte ID / white-model / occlusion passes.

    ``unlit=True`` (default) renders ``color`` directly, ignoring scene
    lighting — implemented via emissive with ``base_color=0`` and
    ``max_bounces=0`` (the same trick splat materials use, project.py
    Material.splat_default). ``unlit=False`` gives a Lambertian ``color``
    surface instead. Typical use: ``flat((1, 1, 1))`` white, ``flat((0, 0, 0))``
    black, as a per-instance ``set_primitive_material`` override.
    """
    if unlit:
        return _make(
            overrides,
            name="flat",
            base_color=(0.0, 0.0, 0.0),
            emissive=tuple(color),
            emissive_strength=1.0,
            metallic=0.0,
            roughness=1.0,
            max_bounces=0,
        )
    return _make(
        overrides,
        name="flat",
        base_color=tuple(color),
        metallic=0.0,
        roughness=1.0,
        max_bounces=0,
    )


# --------------------------------------------------------------------------
# Named presets (exact factors of engine.py register_default_materials)
# --------------------------------------------------------------------------


def solid(**overrides) -> Material:
    """3dgrut 'solid' (engine.py:439-450): light blue, smooth dielectric.

    The 3dgrut original carries a constant-color diffuse_map of
    (130, 193, 255)/255 with diffuse_factor=1; the factor translation below
    is therefore exact.
    """
    return _make(
        overrides,
        name="solid",
        base_color=(130.0 / 255.0, 193.0 / 255.0, 255.0 / 255.0),
        metallic=0.0,
        roughness=0.0,
        transmission=0.0,
        ior=1.0,
        max_bounces=3,
    )


def brushed_copper(**overrides) -> Material:
    """3dgrut 'brushed_copper' (engine.py:461-469)."""
    return _make(
        overrides,
        name="brushed_copper",
        base_color=(0.95, 0.64, 0.54),
        metallic=1.0,
        roughness=0.5,
        transmission=0.0,
        ior=1.1,
        max_bounces=4,
    )


def rose_gold(**overrides) -> Material:
    """3dgrut 'rose_gold' (engine.py:506-514)."""
    return _make(
        overrides,
        name="rose_gold",
        base_color=(0.92, 0.72, 0.75),
        metallic=1.0,
        roughness=0.15,
        transmission=0.0,
        ior=1.1,
        max_bounces=4,
    )


def blue_plastic(**overrides) -> Material:
    """3dgrut 'blue_plastic' (engine.py:533-541)."""
    return _make(
        overrides,
        name="blue_plastic",
        base_color=(0.1, 0.2, 0.8),
        metallic=0.0,
        roughness=0.4,
        transmission=0.0,
        ior=1.45,
        max_bounces=3,
    )


def oak_wood(**overrides) -> Material:
    """3dgrut 'oak_wood' (engine.py:542-550)."""
    return _make(
        overrides,
        name="oak_wood",
        base_color=(0.65, 0.5, 0.35),
        metallic=0.0,
        roughness=0.75,
        transmission=0.0,
        ior=1.3,
        max_bounces=3,
    )


def black_rubber(**overrides) -> Material:
    """3dgrut 'black_rubber' (engine.py:551-559)."""
    return _make(
        overrides,
        name="black_rubber",
        base_color=(0.1, 0.1, 0.1),
        metallic=0.0,
        roughness=0.9,
        transmission=0.0,
        ior=1.5,
        max_bounces=3,
    )


def polished_marble(**overrides) -> Material:
    """3dgrut 'polished_marble' (engine.py:488-496)."""
    return _make(
        overrides,
        name="polished_marble",
        base_color=(0.9, 0.9, 0.95),
        metallic=0.0,
        roughness=0.1,
        transmission=0.0,
        ior=1.6,
        max_bounces=3,
    )


def blue_glass(**overrides) -> Material:
    """3dgrut 'blue_glass' (engine.py:470-478)."""
    return _make(
        overrides,
        name="blue_glass",
        base_color=(0.1, 0.2, 0.8),
        opacity=0.8,
        metallic=0.0,
        roughness=0.1,
        transmission=0.95,
        ior=1.52,
        max_bounces=8,
    )


def jade(**overrides) -> Material:
    """3dgrut 'jade' (engine.py:479-487)."""
    return _make(
        overrides,
        name="jade",
        base_color=(0.2, 0.8, 0.5),
        opacity=0.9,
        metallic=0.0,
        roughness=0.3,
        transmission=0.4,
        ior=1.66,
        max_bounces=6,
    )


def diamond(**overrides) -> Material:
    """3dgrut 'diamond' (engine.py:497-505)."""
    return _make(
        overrides,
        name="diamond",
        base_color=(0.98, 0.98, 0.98),
        opacity=0.2,
        metallic=0.0,
        roughness=0.02,
        transmission=0.99,
        ior=2.42,
        max_bounces=8,
    )


def ruby_red(**overrides) -> Material:
    """3dgrut 'ruby_red' (engine.py:524-532)."""
    return _make(
        overrides,
        name="ruby_red",
        base_color=(0.9, 0.1, 0.2),
        opacity=0.9,
        metallic=0.0,
        roughness=0.1,
        transmission=0.3,
        ior=1.76,
        max_bounces=6,
    )


def luminous_yellow(**overrides) -> Material:
    """3dgrut 'luminous_yellow' (engine.py:515-523): green-ish base with a
    warm yellow glow (emissive_factor (0.8, 0.8, 0.4), strength 1)."""
    return _make(
        overrides,
        name="luminous_yellow",
        base_color=(0.2, 0.9, 0.3),
        emissive=(0.8, 0.8, 0.4),
        emissive_strength=1.0,
        metallic=0.0,
        roughness=0.7,
        transmission=0.0,
        ior=1.0,
        max_bounces=3,
    )


# name -> zero-config factory; every factory still accepts **overrides.
PRESETS = {
    "glass": glass,
    "mirror": mirror,
    "diffuse": diffuse,
    "solid": solid,
    "brushed_copper": brushed_copper,
    "rose_gold": rose_gold,
    "blue_plastic": blue_plastic,
    "oak_wood": oak_wood,
    "black_rubber": black_rubber,
    "polished_marble": polished_marble,
    "blue_glass": blue_glass,
    "jade": jade,
    "diamond": diamond,
    "ruby_red": ruby_red,
    "luminous_yellow": luminous_yellow,
}
