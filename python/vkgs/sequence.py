"""Generation of benchmark .cfg sequence scripts (RenderScript).

The .cfg format consumed by ``--sequencefile`` is a list of ``SEQUENCE "name"``
headers, each followed by ``--param value`` lines (``#`` starts a comment
line). Parameters available inside sequences are the callback parameters of
GaussianSplattingUI::registerParameters (src/gaussian_splatting_ui.cpp:73-177)
plus every global CLI parameter (src/parameters.cpp).

Critical timing contract (src/gaussian_splatting_ui.cpp:104-121): the
``--saveImage`` callback runs saveBufferToFile *immediately* when the
parameter is applied at sequence start, so it captures the framebuffer from
the END of the PREVIOUS sequence. :meth:`RenderScript.capture` therefore
always emits a pair of sequences: a RENDER sequence (activateCameraPreset +
sequenceframes N) followed by a SAVE sequence (saveImageBuffer -1 + saveImage
+ sequenceframes 1). Never place saveImage inside the render sequence itself.

Other pinned rules:
- ``--screenshot`` reads the swapchain and is invalid headless
  (gaussian_splatting_ui.cpp:88); this module refuses to emit it.
- ``--saveImageBuffer`` numeric indices shift with active features (the
  dumpable-buffer vector is built conditionally); we always emit ``-1`` (all
  buffers) and select files by postfix at read time (see vkgs.images).
- Every block re-emits sequenceframes/sequenceaverages/sequenceresetframes so
  the script does not depend on sequencer state persisting across blocks.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

from .constants import BUFFER_POSTFIXES

# Exact cfg spellings, pinned against the C++ parameter registrations.
# Sequencer built-ins + UI callback parameters (gaussian_splatting_ui.cpp)
# + global parameters (parameters.cpp).
_EXACT_SPELLINGS = {
    "sequenceframes",
    "sequenceaverages",
    "sequenceresetframes",
    "updateData",
    "saveImage",
    "saveImageBuffer",
    "loadCameraPresets",
    "activateCameraPreset",
    "colorBufferFormat",
    "loadDefaultScene",
    "inputProject",
    "inputFile",
    "shformat",
    "rgbaformat",
    "useAABBs",
    "useSpheres",
    "useTlasInstances",
    "compressBlas",
    "pipeline",
    "maxShDegree",
    "lightingEnabled",
    "shadowMode",
    "sortStrategy",
    "extentProjection",
    "rtxSortStrategy",
    "rtxMaxBounces",
    "kernelDegree",
    "rtxParticleDepth",
    "rtxShortenRay",
    "rtxBillboardBounding",
}

# python_snake -> exact cfg spelling, for the names a naive snake->camelCase
# conversion would get wrong, plus aliases matching RendererSettings attrs.
_SNAKE_TO_CFG = {
    "sequence_frames": "sequenceframes",
    "sequence_averages": "sequenceaverages",
    "sequence_reset_frames": "sequenceresetframes",
    "sh_format": "shformat",
    "rgba_format": "rgbaformat",
    "use_aabbs": "useAABBs",
    "shadow_mode": "shadowMode",
    "shadows_mode": "shadowMode",
    "sort_strategy": "sortStrategy",
    "sorting_method": "sortStrategy",
    "rtx_sort_strategy": "rtxSortStrategy",
    # parameters.cpp:155 maps the cfg name "rtxSortStrategy" onto
    # prmRtx.rtxTraceStrategy, hence this alias.
    "rtx_trace_strategy": "rtxSortStrategy",
    "particle_depth": "rtxParticleDepth",
    "shorten_ray": "rtxShortenRay",
    "billboard_bounding_mode": "rtxBillboardBounding",
}

# Parameters emitted as a bare flag when True (sample cfgs use "--updateData"
# with no value) and omitted entirely when False/None.
_FLAG_PARAMS = {"updateData"}

# Parameters whose value is a file path: always absolutized (the app runs
# with cwd = executable dir, not the caller's cwd).
_PATH_PARAMS = {"saveImage", "loadCameraPresets", "inputProject", "inputFile"}

_FORBIDDEN_PARAMS = {
    # Reads the swapchain; unavailable headless (gaussian_splatting_ui.cpp:88).
    "screenshot": "use capture()/saveImage instead: --screenshot reads the "
    "swapchain and does nothing in headless mode",
}


def _cfg_name(name: str) -> str:
    """Map a python_snake (or already-exact) parameter name to its cfg spelling."""
    if name in _EXACT_SPELLINGS or name in _FORBIDDEN_PARAMS:
        return name
    if name in _SNAKE_TO_CFG:
        return _SNAKE_TO_CFG[name]
    if "_" in name:
        head, *rest = name.split("_")
        name = head + "".join(part.capitalize() for part in rest)
    return name


def _format_value(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):  # includes IntEnum
        return str(int(value))
    if isinstance(value, float):
        return format(value, ".10g")
    if isinstance(value, (str, os.PathLike)):
        return '"' + str(value) + '"'
    raise TypeError(f"unsupported cfg parameter value type: {type(value).__name__}")


class RenderScript:
    """Builder for a .cfg sequence script.

    ``frames``/``averages``/``reset_frames`` are the per-sequence defaults
    (sequenceframes / sequenceaverages / sequenceresetframes). For stochastic
    pipelines (RTX/hybrid, sortStrategy 3, DoF) ``frames`` must be at least
    the desired sample count so temporal accumulation converges.
    """

    def __init__(self, frames: int = 64, averages: int = 32, reset_frames: int = 0):
        self.frames = int(frames)
        self.averages = int(averages)
        self.reset_frames = int(reset_frames)
        self._blocks = []  # list of (name, [lines])

    # --------------------------------------------------------------- blocks

    def load_block(self, name: str = "Load scene and settle", *, frames: Optional[int] = None, **params) -> "RenderScript":
        """Emit the initial load/settle sequence (must be the first block).

        The scene given via --inputProject/--inputFile finishes loading during
        this sequence; camera presets from a .vkgs are only available after it
        completes, so activateCameraPreset must never appear here. Global
        settings that need the app initialized (e.g. colorBufferFormat) belong
        in this block.
        """
        if self._blocks:
            raise RuntimeError("load_block must be the first sequence of the script")
        return self.sequence(name, frames=frames, **params)

    def _ensure_load_block(self) -> None:
        if not self._blocks:
            self.load_block()

    def sequence(
        self,
        name: str,
        *,
        frames: Optional[int] = None,
        averages: Optional[int] = None,
        reset_frames: Optional[int] = None,
        **params,
    ) -> "RenderScript":
        """Append one SEQUENCE block.

        Parameter names are python_snake (mapped to the exact cfg spelling)
        or the exact spelling itself. Values: bool -> 1/0, int/float -> text,
        str -> quoted. Flag parameters (updateData) are emitted bare when
        True. ``None`` values are skipped.
        """
        ordered = {
            "sequenceframes": self.frames if frames is None else int(frames),
            "sequenceaverages": self.averages if averages is None else int(averages),
            "sequenceresetframes": self.reset_frames if reset_frames is None else int(reset_frames),
        }
        for key, value in params.items():
            cfg = _cfg_name(key)
            if cfg in _FORBIDDEN_PARAMS:
                raise ValueError(f"--{cfg} is not allowed: {_FORBIDDEN_PARAMS[cfg]}")
            ordered[cfg] = value

        lines = []
        for cfg, value in ordered.items():
            if value is None:
                continue
            if cfg in _FLAG_PARAMS:
                if value:
                    lines.append(f"--{cfg}")
                continue
            if cfg in _PATH_PARAMS:
                value = os.path.abspath(str(value))
            lines.append(f"--{cfg} {_format_value(value)}")

        self._blocks.append((name.replace('"', "'"), lines))
        return self

    def capture(
        self,
        name: str,
        camera_preset: int,
        out_stem: str,
        *,
        frames: Optional[int] = None,
        buffers: Optional[Sequence[str]] = None,
        ext: str = ".png",
        hdr: Optional[bool] = None,
        **params,
    ) -> str:
        """Render from a camera preset and save the framebuffers.

        Emits the render+save sequence pair required by the saveImage timing
        contract (see module docstring). ``buffers`` only validates the names
        you plan to read back (buffer *selection* happens at read time by
        filename postfix; the save sequence always uses saveImageBuffer -1).
        Extra ``params`` go into the RENDER sequence.

        ``ext`` is the save-file extension (".png"/".hdr"/".raw"); the app
        writes ``<stem>_<buffer><ext>`` files. ``hdr`` is a deprecated alias
        (``hdr=True`` -> ``.hdr``) kept for backward compatibility.

        Returns the absolute output stem.
        """
        if buffers:
            unknown = [b for b in buffers if b not in BUFFER_POSTFIXES]
            if unknown:
                raise ValueError(f"unknown buffer(s) {unknown}; expected {sorted(BUFFER_POSTFIXES)}")

        self._ensure_load_block()
        self.sequence(name, frames=frames, activateCameraPreset=int(camera_preset), **params)

        stem = os.path.abspath(out_stem)
        if hdr is not None:  # deprecated alias for ext
            ext = ".hdr" if hdr else ".png"
        if not ext.startswith("."):
            ext = "." + ext
        # Order matters: saveImageBuffer must precede saveImage (the saveImage
        # callback reads the already-applied buffer index).
        self.sequence(
            f"{name} save",
            frames=1,
            averages=1,
            reset_frames=0,
            saveImageBuffer=-1,
            saveImage=stem + ext,
        )
        return stem

    # --------------------------------------------------------------- output

    def text(self) -> str:
        """The .cfg file content."""
        if not self._blocks:
            raise RuntimeError("empty script: add a load_block/sequence/capture first")
        chunks = ["# Generated by vkgs.sequence.RenderScript\n"]
        for name, lines in self._blocks:
            chunks.append(f'SEQUENCE "{name}"\n' + "\n".join(lines) + "\n")
        return "\n".join(chunks)

    def write(self, path: str) -> str:
        """Write the .cfg file and return its absolute path."""
        abspath = os.path.abspath(path)
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        with open(abspath, "w", encoding="utf-8", newline="\n") as f:
            f.write(self.text())
        return abspath
