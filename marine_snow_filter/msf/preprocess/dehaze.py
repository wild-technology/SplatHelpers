"""
dehaze.py - Image dehazing / underwater enhancement methods for marine snow filtering.

Implements multiple dehazing algorithms adapted for underwater imagery where
scattering and absorption (especially in the red channel) dominate.

All public functions share the signature:
    def method(image: np.ndarray, **kwargs) -> np.ndarray
where *image* is a BGR uint8 array and the return value has the same dtype/shape.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(image: np.ndarray) -> np.ndarray:
    """Convert a uint8 BGR image to float64 in [0, 1]."""
    return image.astype(np.float64) / 255.0


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Clip to [0, 1] and convert back to uint8."""
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 60,
    eps: float = 1e-3,
) -> np.ndarray:
    """Edge-preserving guided filter (He et al. 2013).

    Parameters
    ----------
    guide : np.ndarray
        Guidance image, single-channel float64.
    src : np.ndarray
        Source (filtering input), single-channel float64.
    radius : int
        Box-filter radius.
    eps : float
        Regularisation parameter.

    Returns
    -------
    np.ndarray
        Filtered output, same shape as *src*.
    """
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(guide, ddepth=-1, ksize=ksize)
    mean_p = cv2.boxFilter(src, ddepth=-1, ksize=ksize)
    corr_Ip = cv2.boxFilter(guide * src, ddepth=-1, ksize=ksize)
    var_I = cv2.boxFilter(guide * guide, ddepth=-1, ksize=ksize) - mean_I * mean_I

    a = (corr_Ip - mean_I * mean_p) / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=ksize)
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=ksize)

    return mean_a * guide + mean_b


# ---------------------------------------------------------------------------
# 1. Dark Channel Prior  (He et al. 2009, adapted for underwater)
# ---------------------------------------------------------------------------

def dcp(
    image: np.ndarray,
    *,
    patch_size: int = 15,
    omega: float = 0.85,
    t_min: float = 0.1,
    guided_radius: int = 60,
    guided_eps: float = 1e-3,
    **kwargs,
) -> np.ndarray:
    """Dehaze using the Dark Channel Prior.

    Underwater adaptation: the dark-channel contribution of each colour
    channel is weighted by its typical underwater attenuation.  Red
    attenuates most, so its dark-channel weight is increased, which better
    captures the true haze transmission in underwater scenes.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 input image.
    patch_size : int
        Side length of the local minimum patch (must be odd).
    omega : float
        Haze removal strength.  0.85 is lower than the standard 0.95 to
        avoid over-enhancement in turbid water.
    t_min : float
        Minimum transmission to prevent division-by-zero artifacts.
    guided_radius : int
        Radius for the guided-filter refinement of the transmission map.
    guided_eps : float
        Regularisation for the guided filter.

    Returns
    -------
    np.ndarray
        Dehazed BGR uint8 image.
    """
    img_f = _to_float(image)
    h, w, _ = img_f.shape

    # Ensure odd patch size
    if patch_size % 2 == 0:
        patch_size += 1
    pad = patch_size // 2

    # --- Dark channel ---
    # Underwater attenuation weights: red attenuates most, then green, blue least.
    # Weighting the per-channel contribution lets the dark channel better reflect
    # true scattering in water rather than just the missing red channel.
    attenuation_weights = np.array([0.25, 0.35, 0.40])  # B, G, R
    dark_raw = np.min(img_f, axis=2)
    weighted_channels = img_f * attenuation_weights[np.newaxis, np.newaxis, :]
    dark_weighted = np.min(weighted_channels, axis=2)
    dark_channel = cv2.erode(
        dark_weighted,
        cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size)),
    )

    # --- Atmospheric light ---
    # Use the top 0.1% brightest pixels in the dark channel, then pick the
    # brightest pixel from the *original* image at those locations.
    # This avoids bias from isolated bright marine snow particles.
    num_pixels = h * w
    num_top = max(int(num_pixels * 0.001), 1)
    dark_flat = dark_channel.ravel()
    top_indices = np.argpartition(dark_flat, -num_top)[-num_top:]
    # Among those, find the one with the highest intensity in the original image
    intensity = np.sum(img_f.reshape(-1, 3), axis=1)
    best = top_indices[np.argmax(intensity[top_indices])]
    A = img_f.reshape(-1, 3)[best]
    A = np.clip(A, 0.01, 1.0)  # safety clamp

    # --- Transmission map ---
    normalised = img_f / A[np.newaxis, np.newaxis, :]
    norm_dark = np.min(normalised, axis=2)
    norm_dark = cv2.erode(
        norm_dark,
        cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size)),
    )
    t_coarse = 1.0 - omega * norm_dark

    # Guided-filter refinement using the grey version of the image as guide
    guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    t_refined = _guided_filter(guide, t_coarse, radius=guided_radius, eps=guided_eps)
    t_refined = np.clip(t_refined, t_min, 1.0)

    # --- Scene radiance recovery ---
    t_3 = t_refined[:, :, np.newaxis]
    J = (img_f - A[np.newaxis, np.newaxis, :]) / t_3 + A[np.newaxis, np.newaxis, :]

    return _to_uint8(J)


# ---------------------------------------------------------------------------
# 2. Homomorphic filtering
# ---------------------------------------------------------------------------

def homomorphic(
    image: np.ndarray,
    *,
    gamma_low: float = 0.5,
    gamma_high: float = 1.5,
    cutoff: float = 30.0,
    **kwargs,
) -> np.ndarray:
    """Homomorphic filter for illumination/reflectance separation.

    Operates per-channel in the log-frequency domain with a Gaussian
    high-pass transfer function.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 input image.
    gamma_low : float
        Gain applied to low-frequency (illumination) components.
    gamma_high : float
        Gain applied to high-frequency (reflectance) components.
    cutoff : float
        Cutoff frequency (in pixels) of the Gaussian filter.

    Returns
    -------
    np.ndarray
        Filtered BGR uint8 image.
    """
    img_f = image.astype(np.float64) + 1.0  # avoid log(0)
    rows, cols = img_f.shape[:2]

    # Optimal DFT size
    dft_rows = cv2.getOptimalDFTSize(rows)
    dft_cols = cv2.getOptimalDFTSize(cols)

    # Distance matrix for Gaussian filter
    crow, ccol = dft_rows // 2, dft_cols // 2
    u = np.arange(dft_rows) - crow
    v = np.arange(dft_cols) - ccol
    U, V = np.meshgrid(v, u)
    D_sq = U.astype(np.float64) ** 2 + V.astype(np.float64) ** 2

    # Gaussian high-pass transfer function
    H = (gamma_high - gamma_low) * (1.0 - np.exp(-D_sq / (2.0 * cutoff ** 2))) + gamma_low

    result_channels: List[np.ndarray] = []
    for c in range(3):
        channel = img_f[:, :, c]

        # Log domain
        log_ch = np.log(channel)

        # Pad to optimal DFT size
        padded = np.zeros((dft_rows, dft_cols), dtype=np.float64)
        padded[:rows, :cols] = log_ch

        # FFT, shift, filter, shift back, IFFT
        fft = np.fft.fft2(padded)
        fft_shift = np.fft.fftshift(fft)
        filtered = fft_shift * H
        ifft = np.fft.ifft2(np.fft.ifftshift(filtered))
        result = np.exp(np.real(ifft[:rows, :cols]))

        result_channels.append(result)

    out = np.stack(result_channels, axis=2)
    # Normalise to [0, 255] using percentile-based clipping.
    # Plain min/max normalisation is highly sensitive to outliers which
    # is common in underwater imagery (a few extremely bright specks
    # pull max up and squash the rest of the histogram).
    for c in range(3):
        ch = out[:, :, c]
        lo = np.percentile(ch, 1)
        hi = np.percentile(ch, 99)
        if hi - lo > 1e-8:
            out[:, :, c] = (ch - lo) / (hi - lo) * 255.0
        else:
            out[:, :, c] = 0.0

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. Multi-Scale Retinex with Colour Restoration (MSRCR)
# ---------------------------------------------------------------------------

def retinex(
    image: np.ndarray,
    *,
    scales: Optional[List[int]] = None,
    alpha: float = 125.0,
    beta: float = 46.0,
    **kwargs,
) -> np.ndarray:
    """Multi-Scale Retinex with Colour Restoration.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 input image.
    scales : list of int or None
        Gaussian blur sigma values for each retinex scale.
        Defaults to ``[15, 80, 250]``.
    alpha : float
        Colour restoration log-domain gain.
    beta : float
        Colour restoration multiplier.

    Returns
    -------
    np.ndarray
        Enhanced BGR uint8 image.
    """
    if scales is None:
        scales = [15, 80, 250]

    img_f = image.astype(np.float64) + 1.0  # avoid log(0)
    num_scales = len(scales)
    weight = 1.0 / num_scales

    # Multi-scale retinex
    msr = np.zeros_like(img_f)
    for sigma in scales:
        # Kernel size must be odd and large enough for the sigma
        ksize = int(sigma * 6) | 1  # ensure odd
        blurred = cv2.GaussianBlur(img_f, (ksize, ksize), sigma).astype(np.float64)
        blurred = np.maximum(blurred, 1.0)
        msr += weight * (np.log(img_f) - np.log(blurred))

    # Colour restoration factor
    channel_sum = np.sum(img_f, axis=2, keepdims=True)
    channel_sum = np.maximum(channel_sum, 1.0)
    C = beta * (np.log(alpha * img_f) - np.log(channel_sum))

    # Combined
    msrcr = msr * C

    # Normalise each channel to [0, 255]
    out = np.zeros_like(msrcr)
    for c in range(3):
        ch = msrcr[:, :, c]
        mn, mx = ch.min(), ch.max()
        if mx - mn > 1e-8:
            out[:, :, c] = (ch - mn) / (mx - mn) * 255.0
        else:
            out[:, :, c] = 0.0

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. Fusion-based underwater enhancement  (Ancuti et al.)
# ---------------------------------------------------------------------------

def _laplacian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    """Build a Laplacian pyramid with *levels* levels.

    Parameters
    ----------
    image : np.ndarray
        Input image (float64, any number of channels).
    levels : int
        Number of pyramid levels.

    Returns
    -------
    list of np.ndarray
        Laplacian pyramid from finest to coarsest (last element is residual).
    """
    pyramid: List[np.ndarray] = []
    current = image.copy()
    for _ in range(levels - 1):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        pyramid.append(current - up)
        current = down
    pyramid.append(current)  # residual (low-freq)
    return pyramid


def _gaussian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    """Build a Gaussian pyramid with *levels* levels.

    Parameters
    ----------
    image : np.ndarray
        Input image (float64).
    levels : int
        Number of pyramid levels.

    Returns
    -------
    list of np.ndarray
        Gaussian pyramid from finest to coarsest.
    """
    pyramid: List[np.ndarray] = [image.copy()]
    current = image.copy()
    for _ in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def _reconstruct_laplacian(pyramid: List[np.ndarray]) -> np.ndarray:
    """Collapse a Laplacian pyramid back into an image.

    Parameters
    ----------
    pyramid : list of np.ndarray
        Laplacian pyramid (finest to coarsest, last is residual).

    Returns
    -------
    np.ndarray
        Reconstructed image.
    """
    current = pyramid[-1]
    for level in reversed(pyramid[:-1]):
        up = cv2.pyrUp(current, dstsize=(level.shape[1], level.shape[0]))
        current = up + level
    return current


def fusion(
    image: np.ndarray,
    *,
    levels: int = 5,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
    **kwargs,
) -> np.ndarray:
    """Fusion-based underwater image enhancement (Ancuti et al.).

    Creates two derived inputs -- white-balanced (gray-world) and
    CLAHE-enhanced -- computes quality weight maps, and blends them
    via multi-scale Laplacian pyramid fusion.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 input image.
    levels : int
        Number of pyramid levels for the multi-scale fusion.
    clahe_clip : float
        CLAHE clip limit for the contrast-enhanced input.
    clahe_grid : int
        CLAHE tile grid size.

    Returns
    -------
    np.ndarray
        Enhanced BGR uint8 image.
    """
    img_f = _to_float(image)

    # --- Input 1: Gray-world white balance ---
    means = img_f.mean(axis=(0, 1))
    global_mean = means.mean()
    scale = np.where(means > 1e-8, global_mean / means, 1.0)
    wb = np.clip(img_f * scale[np.newaxis, np.newaxis, :], 0, 1)

    # --- Input 2: CLAHE on L channel ---
    lab = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    clahe_obj = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
    ce = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float64) / 255.0

    inputs = [wb, ce]

    # --- Weight maps ---
    weight_maps: List[np.ndarray] = []
    for inp in inputs:
        grey = cv2.cvtColor((inp * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

        # 1) Laplacian contrast weight
        lap = np.abs(cv2.Laplacian(grey, cv2.CV_64F))

        # 2) Saturation weight
        sat = np.std(inp, axis=2)

        # 3) Exposedness weight (Gaussian centred at 0.5, sigma=0.25)
        sigma_exp = 0.25
        exposedness = np.prod(
            np.exp(-((inp - 0.5) ** 2) / (2 * sigma_exp ** 2)),
            axis=2,
        )

        W = (lap + 1e-12) * (sat + 1e-12) * (exposedness + 1e-12)
        weight_maps.append(W)

    # Normalise so weights sum to 1 at every pixel
    total = sum(weight_maps)
    total = np.maximum(total, 1e-12)
    weight_maps = [w / total for w in weight_maps]

    # --- Multi-scale fusion via Laplacian pyramid ---
    # Build Laplacian pyramids for each input and Gaussian pyramids for weights
    input_pyramids = [_laplacian_pyramid(inp, levels) for inp in inputs]
    weight_pyramids = [_gaussian_pyramid(w, levels) for w in weight_maps]

    # Blend at each level
    blended_pyramid: List[np.ndarray] = []
    for lvl in range(levels):
        blended = np.zeros_like(input_pyramids[0][lvl])
        for i in range(len(inputs)):
            w = weight_pyramids[i][lvl]
            if blended.ndim == 3:
                w = w[:, :, np.newaxis]
            blended += w * input_pyramids[i][lvl]
        blended_pyramid.append(blended)

    result = _reconstruct_laplacian(blended_pyramid)
    return _to_uint8(result)


# ---------------------------------------------------------------------------
# 5. Red Channel Prior  (underwater-specific)
# ---------------------------------------------------------------------------

def rcp(
    image: np.ndarray,
    *,
    patch_size: int = 15,
    omega: float = 0.85,
    t_min: float = 0.1,
    guided_radius: int = 60,
    guided_eps: float = 1e-3,
    **kwargs,
) -> np.ndarray:
    """Dehaze using the Red Channel Prior for underwater images.

    Unlike the generic Dark Channel Prior, the Red Channel Prior exploits the
    fact that red light attenuates most rapidly in water.  The red channel
    therefore carries direct depth / attenuation information: darker red
    means greater distance through water.  This is *meaningfully different*
    from DCP underwater because DCP treats all channels symmetrically (taking
    the per-pixel min), which in water simply re-discovers the absent red
    channel without modelling the physics.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 input image.
    patch_size : int
        Side length of the local patch for red-channel estimation.
    omega : float
        Haze removal strength.
    t_min : float
        Minimum transmission to prevent artifacts.
    guided_radius : int
        Radius for guided-filter refinement.
    guided_eps : float
        Guided filter regularisation.

    Returns
    -------
    np.ndarray
        Dehazed BGR uint8 image.
    """
    img_f = _to_float(image)
    h, w, _ = img_f.shape

    if patch_size % 2 == 0:
        patch_size += 1

    B, G, R = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    # --- Atmospheric light estimation from red channel ---
    # Use local min-filtered red channel to find the region with least
    # red attenuation (i.e. closest to camera / brightest in red).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    red_eroded = cv2.erode(R, kernel)

    num_pixels = h * w
    num_top = max(int(num_pixels * 0.001), 1)
    flat = red_eroded.ravel()
    top_indices = np.argpartition(flat, -num_top)[-num_top:]
    intensity = np.sum(img_f.reshape(-1, 3), axis=1)
    best = top_indices[np.argmax(intensity[top_indices])]
    A = img_f.reshape(-1, 3)[best]
    A = np.clip(A, 0.01, 1.0)

    A_r = A[2]  # red component of atmospheric light
    A_b = A[0]  # blue
    A_g = A[1]  # green

    # --- Transmission from red channel ---
    # Red attenuates fastest => transmission is directly related to
    # how much red is lost relative to the atmospheric light.
    red_norm = R / A_r
    red_norm_patch = cv2.erode(red_norm, kernel)
    t_red = 1.0 - omega * red_norm_patch

    # --- Transmission for blue and green ---
    # Blue and green attenuate less.  We model their transmission from
    # the complement relationship: t_c = 1 - omega * (C / A_c), but
    # combine with the red-derived estimate since red is more reliable
    # for depth estimation.
    blue_norm_patch = cv2.erode(B / A_b, kernel)
    green_norm_patch = cv2.erode(G / A_g, kernel)
    t_blue = 1.0 - omega * blue_norm_patch
    t_green = 1.0 - omega * green_norm_patch

    # Guided-filter refinement
    guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    t_red = _guided_filter(guide, t_red, radius=guided_radius, eps=guided_eps)
    t_blue = _guided_filter(guide, t_blue, radius=guided_radius, eps=guided_eps)
    t_green = _guided_filter(guide, t_green, radius=guided_radius, eps=guided_eps)

    t_red = np.clip(t_red, t_min, 1.0)
    t_blue = np.clip(t_blue, t_min, 1.0)
    t_green = np.clip(t_green, t_min, 1.0)

    # --- Recover scene radiance per-channel ---
    J_b = (B - A_b) / t_blue + A_b
    J_g = (G - A_g) / t_green + A_g
    J_r = (R - A_r) / t_red + A_r

    result = np.stack([J_b, J_g, J_r], axis=2)
    return _to_uint8(result)


# ---------------------------------------------------------------------------
# 6. Bright particle suppression
# ---------------------------------------------------------------------------

def suppress_particles(
    image: np.ndarray,
    *,
    kernel_size: int = 3,
    brightness_threshold: int = 220,
    **kwargs,
) -> np.ndarray:
    """Suppress bright isolated specks (marine snow / backscatter particles).

    Detects bright spots via thresholding on a greyscale version, dilates
    the detected regions to capture the full neighbourhood, then replaces
    only those regions with a median-filtered version of the image.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 input image.
    kernel_size : int
        Morphological and median kernel size (must be odd).
    brightness_threshold : int
        Greyscale intensity above which a pixel is considered "bright".

    Returns
    -------
    np.ndarray
        Image with bright specks suppressed.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect bright spots
    _, bright_mask = cv2.threshold(grey, brightness_threshold, 255, cv2.THRESH_BINARY)

    # Morphological opening to keep only small isolated specks
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, open_kernel)

    # Dilate to cover the neighbourhood of each speck
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size * 2 + 1, kernel_size * 2 + 1)
    )
    mask = cv2.dilate(opened, dilate_kernel)

    # Median filter the whole image
    median = cv2.medianBlur(image, kernel_size)

    # Replace only the masked (bright-speck) regions
    result = image.copy()
    result[mask > 0] = median[mask > 0]

    return result


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS: Dict[str, Callable[..., np.ndarray]] = {
    "dcp": dcp,
    "homomorphic": homomorphic,
    "retinex": retinex,
    "fusion": fusion,
    "rcp": rcp,
}
