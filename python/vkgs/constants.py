"""Enumerations mirroring the C++ integer constants of vk_gaussian_splatting.

Every value is pinned against the C++ sources (shaders/shaderio.h,
shaders/shading.h, src/parameters.h). Do not renumber.
"""

from enum import IntEnum


class Pipeline(IntEnum):
    """Rendering pipeline selector (--pipeline / renderer.pipeline)."""

    VERT = 0          # 3DGS vertex-shader rasterization
    MESH = 1          # 3DGS mesh-shader rasterization (default)
    RTX = 2           # 3DGRT full ray tracing
    HYBRID = 3        # hybrid: 3DGS raster primary + RTX secondary
    MESH_3DGUT = 4    # 3DGUT rasterization
    HYBRID_3DGUT = 5  # hybrid: 3DGUT raster primary + RTX secondary


class Storage(IntEnum):
    """Splat data storage (splatSets[].storage)."""

    BUFFERS = 0
    TEXTURES = 1


class Format(IntEnum):
    """SH / RGBA data format (shFormat / rgbaFormat)."""

    FLOAT32 = 0
    FLOAT16 = 1
    UINT8 = 2


class CameraModel(IntEnum):
    PINHOLE = 0
    FISHEYE = 1


class DofMode(IntEnum):
    DISABLED = 0
    FIXED_FOCUS = 1
    AUTO_FOCUS = 2  # needs cursor position; not meaningful headless


class LightType(IntEnum):
    """shaders/shading.h LightType."""

    DIRECTIONAL = 0
    POINT = 1
    SPOT = 2


class AttenuationMode(IntEnum):
    """Point/spot light attenuation (lights.assets[].attenuationMode)."""

    NONE = 0
    LINEAR = 1
    QUADRATIC = 2
    PHYSICAL = 3


class EnvMode(IntEnum):
    """environment.mode (shaderio::EnvironmentMode)."""

    NONE = 0
    SKY = 1
    HDR = 2


class ShadowsMode(IntEnum):
    DISABLED = 0
    HARD = 1
    SOFT = 2


class SortingMethod(IntEnum):
    GPU_SYNC_RADIX = 0
    CPU_ASYNC_MONO = 1
    CPU_ASYNC_MULTI = 2
    STOCHASTIC_SPLAT = 3


class ExtentProjection(IntEnum):
    EIGEN = 0
    CONIC = 1


class KernelDegree(IntEnum):
    LINEAR = 0
    LAPLACIAN = 1
    QUADRATIC = 2
    CUBIC = 3
    TESSERACTIC = 4
    QUINTIC = 5


class RtxTraceStrategy(IntEnum):
    FULL_ANYHIT = 0
    PASS_STOCHASTIC = 1
    STOCHASTIC_ANYHIT = 2


class ParticleDepth(IntEnum):
    BILLBOARD = 0
    ELLIPSOID = 1
    MAX_DENSITY_PLANE = 2


class TemporalSamplingMode(IntEnum):
    AUTO = 0
    ENABLED = 1
    DISABLED = 2


class NormalMethod(IntEnum):
    MAX_DENSITY_PLANE = 0
    ISO_SURFACE = 1


class Visualize(IntEnum):
    """renderer.visualize AOV modes (rendered into the main buffer)."""

    FINAL = 0
    CLOCK = 1
    RAYHITS = 2
    DEPTH = 3
    DEPTH_INTEGRATED = 4
    DEPTH_FOR_DLSS = 5
    NORMAL = 6
    NORMAL_INTEGRATED = 7
    NORMAL_FOR_DLSS = 8
    DLSS_INPUT = 9
    DLSS_ALBEDO = 10
    DLSS_SPECULAR = 11
    DLSS_NORMAL = 12
    DLSS_MOTION = 13
    DLSS_DEPTH = 14
    SPLAT_ID = 15
    CLAY = 16


class BillboardBoundingMode(IntEnum):
    FITTED = 0
    UNIFORM = 1
    UNIFORM_3_4 = 2
    UNIFORM_2_3 = 3
    UNIFORM_1_2 = 4
    UNIFORM_1_3 = 5
    UNIFORM_1_4 = 6
    OPTIMAL = 7


class VkColorFormat(IntEnum):
    """Raw VkFormat values used by renderer.colorFormat.

    The .vkgs file stores the raw VkFormat enum value. The benchmark .cfg
    parameter --colorBufferFormat instead uses index 0/1/2 (see
    ColorBufferFormat below).
    """

    R8G8B8A8_UNORM = 37
    R16G16B16A16_SFLOAT = 97
    R32G32B32A32_SFLOAT = 109


class ColorBufferFormat(IntEnum):
    """--colorBufferFormat index in benchmark .cfg files."""

    RGBA8 = 0
    RGBA16F = 1
    RGBA32F = 2


# Buffer name -> filename postfix appended by --saveImage
# (src/gaussian_splatting.cpp getAllDumpableBuffers()). Buffers marked
# conditional only exist when the corresponding feature is active.
BUFFER_POSTFIXES = {
    "main": "_main",              # HDR color buffer (always)
    "aux1": "_aux1",              # temporal intermediate (always)
    "comparison": "_comparison",  # conditional: image compare active
    "normal": "_normal",          # always
    "depth": "_depth",            # always
    "ldr": "_ldr",                # conditional: tonemapping active
}

# Project file format version this package writes (src/vkgs_project_writer.cpp)
PROJECT_FILE_VERSION = 7
# Minimum version Scene.load() accepts (older formats need C++-side migration)
MIN_SUPPORTED_PROJECT_VERSION = 5
