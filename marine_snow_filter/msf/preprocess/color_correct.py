"""
color_correct.py - White-balance and colour correction methods for underwater images.

All public functions share the signature:
    def method(image: np.ndarray, **kwargs) -> np.ndarray
where *image* is a BGR uint8 array and the return value has the same dtype/shape.
"""

from __future__ import annotations

from typing import Callable, Dict

import cv2
import numpy as np


def grayworld(image: np.ndarray, **kwargs) -> np.ndarray:
    """Gray-world white balance.

    Normalises each BGR channel so that every channel has the same mean,
    equal to the overall mean intensity.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 input image.

    Returns
    -------
    np.ndarray
        White-balanced BGR uint8 image.
    """
    img_f = image.astype(np.float64)
    means = img_f.mean(axis=(0, 1))  # per-channel means
    global_mean = means.mean()

    scale = np.where(means > 1e-8, global_mean / means, 1.0)
    result = img_f * scale[np.newaxis, np.newaxis, :]

    return np.clip(result, 0, 255).astype(np.uint8)


def max_rgb(image: np.ndarray, **kwargs) -> np.ndarray:
    """Max-RGB white balance.

    Scales each channel so that its maximum value equals the overall
    maximum across all channels.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 input image.

    Returns
    -------
    np.ndarray
        White-balanced BGR uint8 image.
    """
    img_f = image.astype(np.float64)
    channel_max = img_f.max(axis=(0, 1))  # per-channel max
    overall_max = channel_max.max()

    scale = np.where(channel_max > 1e-8, overall_max / channel_max, 1.0)
    result = img_f * scale[np.newaxis, np.newaxis, :]

    return np.clip(result, 0, 255).astype(np.uint8)


def shades_of_gray(
    image: np.ndarray,
    *,
    p_norm: int = 6,
    **kwargs,
) -> np.ndarray:
    """Shades-of-Gray white balance (Minkowski norm generalisation).

    The standard gray-world assumption (p=1) is generalised by using the
    Minkowski p-norm of pixel intensities per channel.  Higher values of *p*
    give more weight to brighter pixels, approaching max-RGB as p -> inf.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 input image.
    p_norm : int
        Minkowski norm exponent.  p=1 is gray-world, p=inf is max-RGB.

    Returns
    -------
    np.ndarray
        White-balanced BGR uint8 image.
    """
    img_f = image.astype(np.float64)
    n_pixels = img_f.shape[0] * img_f.shape[1]

    # Minkowski p-norm per channel: (sum(x^p) / N) ^ (1/p)
    norms = np.zeros(3, dtype=np.float64)
    for c in range(3):
        norms[c] = (np.sum(img_f[:, :, c] ** p_norm) / n_pixels) ** (1.0 / p_norm)

    global_norm = norms.mean()
    scale = np.where(norms > 1e-8, global_norm / norms, 1.0)
    result = img_f * scale[np.newaxis, np.newaxis, :]

    return np.clip(result, 0, 255).astype(np.uint8)


def percentile_stretch(
    image: np.ndarray,
    *,
    low: float = 1.0,
    high: float = 99.0,
    **kwargs,
) -> np.ndarray:
    """Per-channel percentile clip and linear stretch.

    For each channel the *low*-th and *high*-th percentiles are computed;
    values are clipped to that range and linearly mapped to [0, 255].

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 input image.
    low : float
        Lower percentile (0-100).
    high : float
        Upper percentile (0-100).

    Returns
    -------
    np.ndarray
        Contrast-stretched BGR uint8 image.
    """
    result = np.zeros_like(image, dtype=np.float64)

    for c in range(3):
        ch = image[:, :, c].astype(np.float64)
        lo_val = np.percentile(ch, low)
        hi_val = np.percentile(ch, high)

        if hi_val - lo_val > 1e-8:
            result[:, :, c] = (ch - lo_val) / (hi_val - lo_val) * 255.0
        else:
            result[:, :, c] = ch

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

WB_METHODS: Dict[str, Callable[..., np.ndarray]] = {
    "grayworld": grayworld,
    "max-rgb": max_rgb,
    "shades-of-gray": shades_of_gray,
}
