"""Numpy port of the 3dgrut playground tonemapping math.

For pixel-faithful compatibility, the VKGS compat shim renders to HDR
(.hdr readback, in-app tonemapper disabled) and applies 3dgrut's own
tonemap + gamma formulas in Python. The formulas below are line-for-line
ports of ``Environment.tonemap``
(3dgrut threedgrut_playground/utils/environment.py:134-163) and the gamma
step of ``Engine3DGRUT.render_pass`` (engine.py:1292-1293).

Divergence note: 3dgrut clamps the traced radiance to [0, 1] *before*
tonemapping (engine.py:1213); the VKGS readback keeps full HDR range into
the tonemapper (usually closer to the physically intended result). Final
output is clamped to [0, 1] either way.
"""

from __future__ import annotations

import numpy as np

TONEMAPPER_OPTIONS = ["None", "Reinhard", "Filmic"]

# Filmic (Uncharted 2 / "polyhaven") curve constants, environment.py:140
_FILMIC_A, _FILMIC_B, _FILMIC_C = 0.22, 0.30, 0.10
_FILMIC_D, _FILMIC_E, _FILMIC_F = 0.20, 0.01, 0.30
_FILMIC_WHITE_POINT = 11.2

# Reinhard constants, environment.py:149-150
_REINHARD_EPSILON = 1e-6
_REINHARD_KEY = 0.18


def filmic_curve(x):
    """environment.py:142-143 (elementwise)."""
    A, B, C, D, E, F = _FILMIC_A, _FILMIC_B, _FILMIC_C, _FILMIC_D, _FILMIC_E, _FILMIC_F
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F


def tonemap(hdr: np.ndarray, tonemapper: str = "None", exposure: float = 0.0) -> np.ndarray:
    """Apply 3dgrut's Environment.tonemap to an (..., 3) float image.

    ``exposure`` scales by 2**exposure first (environment.py:136). Reinhard
    is the luminance-preserving global operator keyed on the log-average
    luminance of the whole image (environment.py:148-155); Filmic is the
    Uncharted-2 curve normalized by white point 11.2 (:138-146). NaNs are
    zeroed like the original (torch.nan_to_num, :162).
    """
    hdr = np.asarray(hdr, dtype=np.float32) * np.float32(2.0 ** float(exposure))

    if tonemapper == "Filmic":
        white_scale = filmic_curve(_FILMIC_WHITE_POINT)
        ldr = filmic_curve(hdr) / white_scale
    elif tonemapper == "Reinhard":
        eps = _REINHARD_EPSILON
        luminance = 0.2126 * hdr[..., 0] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 2]
        log_average_luminance = np.exp(np.mean(np.log(eps + luminance)))
        scaled_luminance = _REINHARD_KEY / log_average_luminance * luminance
        tone_mapped_luminance = scaled_luminance / (1.0 + scaled_luminance)
        ldr = hdr * (tone_mapped_luminance / (luminance + eps))[..., np.newaxis]
    elif tonemapper == "None" or tonemapper is None:
        ldr = hdr
    else:
        raise ValueError(f"Invalid tonemapper. Must be one of {TONEMAPPER_OPTIONS}")

    return np.nan_to_num(ldr).astype(np.float32, copy=False)


def apply_gamma(rgb: np.ndarray, gamma_correction: float = 1.0) -> np.ndarray:
    """rgb ** (1 / gamma_correction), the render_pass gamma step
    (engine.py:1293). Negative inputs are clamped to 0 first."""
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, None)
    if gamma_correction != 1.0:
        rgb = np.power(rgb, np.float32(1.0 / float(gamma_correction)))
    return rgb
