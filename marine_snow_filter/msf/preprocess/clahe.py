"""
clahe.py - Contrast Limited Adaptive Histogram Equalisation for underwater images.

Applies CLAHE to the L channel in LAB colour space to enhance local contrast
without distorting colour balance.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    *,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Apply CLAHE on the L channel of the LAB colour space.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 input image.
    clip_limit : float
        Contrast clipping limit for CLAHE.
    tile_size : int
        Size of the grid tiles (both width and height).

    Returns
    -------
    np.ndarray
        Enhanced BGR uint8 image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
