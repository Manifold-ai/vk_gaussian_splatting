"""3dgrut playground compatibility shim for VKGS.

Drop-in mirror of the ``threedgrut_playground.engine`` scripting surface on
top of the VKGS headless renderer::

    from vkgs.compat import EngineVKGS, OptixPrimitiveTypes

    engine = EngineVKGS(gs_object="garden.ply", mesh_assets_folder="assets")
    engine.camera_type = "Pinhole"
    engine.antialiasing_mode = "8x MSAA"
    engine.primitives.add_primitive("Sphere", OptixPrimitiveTypes.GLASS)
    framebuffer = engine.render(camera)   # {'rgb', 'opacity', 'rgb_buffer'}

Every approximated or unsupported playground feature raises/warns
:class:`CompatWarning` with the workaround from the gap analysis; see
vkgs.compat.engine for the execution-model differences (lazy state +
flush-on-render subprocess instead of in-process CUDA).
"""

from .convert import (
    CompatWarning,
    Light,
    LightType,
    OptixPrimitiveTypes,
    camera_to_vkgs,
    direction_to_euler_deg,
    light_to_vkgs,
    primitive_type_to_material,
    transform_to_vkgs_trs,
)
from .engine import DepthOfFieldVKGS, EngineVKGS, EnvironmentVKGS, SPPVKGS
from .primitives import PrimitivesVKGS, PrimitiveVKGS
from .tonemap import apply_gamma, tonemap

__all__ = [
    "EngineVKGS",
    "OptixPrimitiveTypes",
    "CompatWarning",
    "Light",
    "LightType",
    "PrimitivesVKGS",
    "PrimitiveVKGS",
    "DepthOfFieldVKGS",
    "SPPVKGS",
    "EnvironmentVKGS",
    "camera_to_vkgs",
    "direction_to_euler_deg",
    "light_to_vkgs",
    "primitive_type_to_material",
    "transform_to_vkgs_trs",
    "tonemap",
    "apply_gamma",
]
