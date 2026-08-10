"""Read back --saveImage outputs as numpy arrays and resolve buffer files.

File formats produced by saveBufferToFile (src/gaussian_splatting_ui.cpp):
- .png/.jpg: RGBA8 (GPU blit to R8G8B8A8_UNORM, saved via stb).
- .hdr: float32, RGB only (stb_image_write drops the alpha channel).
- .raw: 16-byte header of four uint32 {width, height, channels=4,
  bytesPerChannel=4} followed by row-major float32 RGBA rows
  (saveRawImageToFile, gaussian_splatting_ui.cpp:5833-5864). This is the only
  lossless float format that keeps alpha.

The app ALWAYS appends a buffer postfix to the requested filename stem
(out.png -> out_main.png, out_normal.png, ...), and numeric --saveImageBuffer
indices shift with active features — so files are selected by postfix, never
by index.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Optional, Tuple

import numpy as np

from .constants import BUFFER_POSTFIXES

_RAW_HEADER_BYTES = 16


def load_image(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load one saved buffer file as a numpy array.

    - .png/.jpg -> uint8, RGBA kept as stored (H, W, 4) for png.
    - .hdr -> float32 (H, W, 3).
    - .raw -> float32 (H, W, 4); the 16-byte header is authoritative,
      ``size`` (W, H) is only needed for legacy headerless dumps and
      otherwise just validated.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".raw":
        return _load_raw(path, size)

    try:
        import imageio.v3 as iio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("reading .png/.jpg/.hdr requires imageio (pip install 'vkgs[image]')") from exc

    image = np.asarray(iio.imread(path))
    if ext == ".hdr":
        image = image.astype(np.float32, copy=False)
    if size is not None and (image.shape[1], image.shape[0]) != (int(size[0]), int(size[1])):
        raise ValueError(f"{path}: image is {image.shape[1]}x{image.shape[0]}, expected {size[0]}x{size[1]}")
    return image


def _load_raw(path: str, size: Optional[Tuple[int, int]]) -> np.ndarray:
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.uint32, count=4)
        if header.size == 4:
            width, height, channels, bytes_per_channel = (int(v) for v in header)
            expected = _RAW_HEADER_BYTES + width * height * channels * bytes_per_channel
            if bytes_per_channel == 4 and 1 <= channels <= 4 and expected == file_size:
                data = np.fromfile(f, dtype=np.float32, count=width * height * channels)
                image = data.reshape(height, width, channels)
                if size is not None and (width, height) != (int(size[0]), int(size[1])):
                    raise ValueError(f"{path}: raw header says {width}x{height}, expected {size[0]}x{size[1]}")
                return image
        # Headerless fallback: tightly packed float32 RGBA, size required.
        if size is None:
            raise ValueError(f"{path}: unrecognized .raw header and no size=(W, H) given")
        width, height = int(size[0]), int(size[1])
        if file_size != width * height * 4 * 4:
            raise ValueError(
                f"{path}: {file_size} bytes matches neither headered nor packed float32 RGBA at {width}x{height}"
            )
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32, count=width * height * 4)
        return data.reshape(height, width, 4)


def resolve_outputs(stem: str, ext: str, tonemapping_active: bool = False) -> Dict[str, str]:
    """Predict the buffer files a ``--saveImage <stem><ext>`` produces.

    Returns buffer name -> absolute path for the unconditional buffers (main,
    aux1, normal, depth) plus ldr when tonemapping is active. Conditional
    buffers we never enable (comparison, dlss_*) are excluded.
    """
    if not ext.startswith("."):
        ext = "." + ext
    stem = os.path.abspath(stem)
    names = ["main", "aux1", "normal", "depth"]
    if tonemapping_active:
        names.append("ldr")
    return {name: stem + BUFFER_POSTFIXES[name] + ext for name in names}


def find_outputs(stem: str) -> Dict[str, str]:
    """Map buffer name -> path for the files actually on disk for ``stem``.

    Globs ``<stem>_*.*`` and derives each buffer name from the filename
    portion between the stem and the extension (robust to conditional
    buffers like ldr/comparison/dlss_* appearing or not).
    """
    stem = os.path.abspath(stem)
    outputs: Dict[str, str] = {}
    for path in sorted(glob.glob(glob.escape(stem) + "_*.*")):
        remainder = os.path.splitext(path[len(stem):])[0]
        name = remainder.lstrip("_")
        if name:
            outputs[name] = path
    return outputs
