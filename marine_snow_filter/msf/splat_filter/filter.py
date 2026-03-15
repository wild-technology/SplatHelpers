"""
filter.py - Gaussian splat filtering for marine-snow and artifact removal.

Each filter function accepts the structured vertex array (from
:func:`~msf.splat_filter.ply_io.read_splat`) and returns a boolean
*keep* mask.  :func:`apply_filters` orchestrates them with AND logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from msf.splat_filter.ply_io import (
    get_opacity,
    get_positions,
    get_rgb,
    get_scales,
)

logger = logging.getLogger("msf")


# ---------------------------------------------------------------------------
# Individual filters  (each returns a boolean *keep* mask)
# ---------------------------------------------------------------------------

def filter_by_opacity(data: np.ndarray, min_opacity: float = 0.02) -> np.ndarray:
    """Remove near-transparent gaussians (volumetric haze).

    Parameters
    ----------
    data : np.ndarray
        Structured vertex array.
    min_opacity : float
        Minimum sigmoid-activated opacity to keep.

    Returns
    -------
    np.ndarray
        Boolean keep mask of shape ``(N,)``.
    """
    opacity = get_opacity(data)
    return opacity >= min_opacity


def filter_by_scale(
    data: np.ndarray,
    max_scale: Optional[float] = None,
) -> np.ndarray:
    """Remove oversized gaussians.

    Parameters
    ----------
    data : np.ndarray
        Structured vertex array.
    max_scale : float | None
        Upper bound on the largest axis-scale.  If *None*, automatically
        computed as ``median + 3 * MAD`` of the per-gaussian max-scale
        distribution.

    Returns
    -------
    np.ndarray
        Boolean keep mask of shape ``(N,)``.
    """
    scales = get_scales(data)  # (N, 3)
    max_per_gauss = np.max(scales, axis=1)  # (N,)

    if max_scale is None:
        median = np.median(max_per_gauss)
        mad = np.median(np.abs(max_per_gauss - median))
        max_scale = median + 3.0 * mad
        logger.info("Auto scale threshold: %.6f  (median=%.6f, MAD=%.6f)",
                     max_scale, median, mad)

    return max_per_gauss <= max_scale


def filter_by_distance(
    data: np.ndarray,
    max_sigma: float = 3.0,
) -> np.ndarray:
    """Remove spatial outliers using MAD-based distance from the centroid.

    Uses the same convention as the COLMAP outlier filter:
    ``threshold = median_dist + sigma * 1.4826 * MAD``.

    Parameters
    ----------
    data : np.ndarray
        Structured vertex array.
    max_sigma : float
        Number of (robust) standard deviations for the distance threshold.

    Returns
    -------
    np.ndarray
        Boolean keep mask of shape ``(N,)``.
    """
    positions = get_positions(data)  # (N, 3)
    centroid = np.mean(positions, axis=0)
    dists = np.linalg.norm(positions - centroid, axis=1)

    median_dist = np.median(dists)
    mad = np.median(np.abs(dists - median_dist))
    threshold = median_dist + max_sigma * 1.4826 * mad

    logger.info("Distance threshold: %.4f  (median=%.4f, MAD=%.4f, sigma=%.1f)",
                threshold, median_dist, mad, max_sigma)

    return dists <= threshold


def filter_by_neutral(
    data: np.ndarray,
    v_min: float = 0.7,
    s_max: float = 0.15,
    opacity_max: float = 0.3,
) -> np.ndarray:
    """Remove bright, desaturated, semi-transparent gaussians (marine snow).

    A gaussian is flagged only when ALL THREE criteria are met:

    - HSV value > *v_min*  (bright)
    - HSV saturation < *s_max*  (near-white/grey)
    - sigmoid opacity < *opacity_max*  (semi-transparent)

    This triple criterion is the marine-snow fingerprint in underwater
    photogrammetric Gaussian splats.

    Parameters
    ----------
    data : np.ndarray
        Structured vertex array.
    v_min : float
        Minimum HSV *V* to be considered bright.
    s_max : float
        Maximum HSV *S* to be considered neutral.
    opacity_max : float
        Maximum sigmoid opacity to be considered semi-transparent.

    Returns
    -------
    np.ndarray
        Boolean keep mask of shape ``(N,)``.
    """
    rgb = get_rgb(data)  # (N, 3), [0, 1]
    opacity = get_opacity(data)  # (N,)

    # Vectorised RGB -> HSV via numpy (avoids per-pixel Python loop)
    # colorsys works scalar-only, so we use matplotlib-style conversion:
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Saturation (suppress divide-by-zero when cmax == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        saturation = np.where(cmax > 0, delta / cmax, 0.0)
    # Value
    value = cmax

    is_bright = value > v_min
    is_neutral = saturation < s_max
    is_transparent = opacity < opacity_max

    is_snow = is_bright & is_neutral & is_transparent
    removed = np.count_nonzero(is_snow)
    logger.info("Neutral filter: flagged %d gaussians as marine snow", removed)

    return ~is_snow


def filter_by_density(
    data: np.ndarray,
    radius: float = 0.5,
    min_neighbors: int = 5,
) -> np.ndarray:
    """Remove isolated gaussians with few spatial neighbours.

    Parameters
    ----------
    data : np.ndarray
        Structured vertex array.
    radius : float
        Search radius for neighbour counting.
    min_neighbors : int
        Minimum number of neighbours (exclusive of the point itself)
        required to keep a gaussian.

    Returns
    -------
    np.ndarray
        Boolean keep mask of shape ``(N,)``.
    """
    positions = get_positions(data)
    tree = cKDTree(positions)
    # query_ball_point returns list of lists; count excludes self
    counts = np.array([
        len(neighbors) - 1
        for neighbors in tree.query_ball_point(positions, radius)
    ])
    return counts >= min_neighbors


def filter_by_elongation(
    data: np.ndarray,
    max_ratio: float = 50.0,
) -> np.ndarray:
    """Remove extremely elongated (needle-like) gaussians.

    Computes ``max_scale / min_scale`` per gaussian.  Highly elongated
    gaussians typically arise from poor triangulation when particles
    (marine snow) appear in only 2-3 views from similar angles.

    Parameters
    ----------
    data : np.ndarray
        Structured vertex array.
    max_ratio : float
        Maximum allowed ratio of largest to smallest axis scale.

    Returns
    -------
    np.ndarray
        Boolean keep mask of shape ``(N,)``.
    """
    scales = get_scales(data)  # (N, 3)
    scale_max = np.max(scales, axis=1)
    scale_min = np.min(scales, axis=1)

    # Guard against division by zero (degenerate gaussians)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(scale_min > 0, scale_max / scale_min, np.inf)

    return ratio <= max_ratio


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def apply_filters(
    plydata: Any,
    data: np.ndarray,
    *,
    min_opacity: float = 0.02,
    max_scale: Optional[float] = None,
    max_distance_sigma: float = 3.0,
    neutral_filter: bool = True,
    neutral_v_min: float = 0.7,
    neutral_s_max: float = 0.15,
    neutral_opacity_max: float = 0.3,
    density_filter: bool = False,
    density_radius: float = 0.5,
    density_min_neighbors: int = 5,
    elongation_filter: bool = True,
    max_elongation_ratio: float = 50.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run all enabled filters and return the combined keep mask.

    Every individual filter produces a boolean keep mask; the final mask
    is the element-wise AND of all enabled filters.

    Parameters
    ----------
    plydata : PlyData
        Original PlyData (unused here but accepted for API symmetry).
    data : np.ndarray
        Structured vertex array from :func:`~msf.splat_filter.ply_io.read_splat`.
    min_opacity : float
        Minimum sigmoid opacity to keep.
    max_scale : float | None
        Maximum axis scale (None = auto).
    max_distance_sigma : float
        MAD-based sigma for distance outlier removal.
    neutral_filter : bool
        Enable the neutral-colour marine-snow filter.
    neutral_v_min, neutral_s_max, neutral_opacity_max : float
        Thresholds for the neutral filter.
    density_filter : bool
        Enable the spatial density filter.
    density_radius : float
        Radius for neighbour search.
    density_min_neighbors : int
        Minimum neighbour count.
    elongation_filter : bool
        Enable the elongation (needle) filter.
    max_elongation_ratio : float
        Maximum allowed scale ratio.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``(keep_mask, stats)`` where *stats* maps filter names to the
        number of gaussians each filter would individually remove.
    """
    n = len(data)
    keep = np.ones(n, dtype=bool)
    stats: Dict[str, Any] = {"total_input": n}

    # --- Opacity ---
    logger.info("Running opacity filter (min_opacity=%.4f) ...", min_opacity)
    m_opacity = filter_by_opacity(data, min_opacity=min_opacity)
    stats["opacity_removed"] = int(np.count_nonzero(~m_opacity))
    keep &= m_opacity

    # --- Scale ---
    logger.info("Running scale filter (max_scale=%s) ...", max_scale)
    m_scale = filter_by_scale(data, max_scale=max_scale)
    stats["scale_removed"] = int(np.count_nonzero(~m_scale))
    keep &= m_scale

    # --- Distance ---
    logger.info("Running distance filter (sigma=%.1f) ...", max_distance_sigma)
    m_dist = filter_by_distance(data, max_sigma=max_distance_sigma)
    stats["distance_removed"] = int(np.count_nonzero(~m_dist))
    keep &= m_dist

    # --- Neutral / marine snow ---
    if neutral_filter:
        logger.info("Running neutral-colour filter ...")
        m_neutral = filter_by_neutral(
            data,
            v_min=neutral_v_min,
            s_max=neutral_s_max,
            opacity_max=neutral_opacity_max,
        )
        stats["neutral_removed"] = int(np.count_nonzero(~m_neutral))
        keep &= m_neutral
    else:
        stats["neutral_removed"] = 0

    # --- Density ---
    if density_filter:
        logger.info("Running density filter (r=%.2f, k=%d) ...",
                     density_radius, density_min_neighbors)
        m_density = filter_by_density(
            data,
            radius=density_radius,
            min_neighbors=density_min_neighbors,
        )
        stats["density_removed"] = int(np.count_nonzero(~m_density))
        keep &= m_density
    else:
        stats["density_removed"] = 0

    # --- Elongation ---
    if elongation_filter:
        logger.info("Running elongation filter (max_ratio=%.1f) ...",
                     max_elongation_ratio)
        m_elong = filter_by_elongation(data, max_ratio=max_elongation_ratio)
        stats["elongation_removed"] = int(np.count_nonzero(~m_elong))
        keep &= m_elong
    else:
        stats["elongation_removed"] = 0

    stats["total_removed"] = int(np.count_nonzero(~keep))
    stats["total_kept"] = int(np.count_nonzero(keep))

    logger.info(
        "Filter summary: %d / %d kept  (%d removed, %.1f%%)",
        stats["total_kept"],
        n,
        stats["total_removed"],
        100.0 * stats["total_removed"] / max(n, 1),
    )

    return keep, stats


# ---------------------------------------------------------------------------
# Optional diagnostics
# ---------------------------------------------------------------------------

def save_histograms(
    data: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
) -> None:
    """Save before/after distribution histograms as PNG files.

    Generates histograms for opacity, scale, and colour distributions.

    Parameters
    ----------
    data : np.ndarray
        Structured vertex array (full, unfiltered).
    mask : np.ndarray
        Boolean keep mask.
    output_dir : Path
        Directory where PNG files will be saved.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_kept = data[mask]

    # --- Opacity histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    opacity_before = get_opacity(data)
    opacity_after = get_opacity(data_kept)
    axes[0].hist(opacity_before, bins=100, color="steelblue", alpha=0.8)
    axes[0].set_title("Opacity - Before")
    axes[0].set_xlabel("Sigmoid opacity")
    axes[1].hist(opacity_after, bins=100, color="forestgreen", alpha=0.8)
    axes[1].set_title("Opacity - After")
    axes[1].set_xlabel("Sigmoid opacity")
    fig.tight_layout()
    fig.savefig(output_dir / "hist_opacity.png", dpi=150)
    plt.close(fig)

    # --- Scale histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    scales_before = np.max(get_scales(data), axis=1)
    scales_after = np.max(get_scales(data_kept), axis=1)
    axes[0].hist(scales_before, bins=100, color="steelblue", alpha=0.8)
    axes[0].set_title("Max Scale - Before")
    axes[0].set_xlabel("exp(scale)")
    axes[1].hist(scales_after, bins=100, color="forestgreen", alpha=0.8)
    axes[1].set_title("Max Scale - After")
    axes[1].set_xlabel("exp(scale)")
    fig.tight_layout()
    fig.savefig(output_dir / "hist_scale.png", dpi=150)
    plt.close(fig)

    # --- Colour (value channel) histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    rgb_before = get_rgb(data)
    rgb_after = get_rgb(data_kept)
    val_before = np.max(rgb_before, axis=1)  # approx HSV V
    val_after = np.max(rgb_after, axis=1)
    axes[0].hist(val_before, bins=100, color="steelblue", alpha=0.8)
    axes[0].set_title("Colour Value - Before")
    axes[0].set_xlabel("Max(R, G, B)")
    axes[1].hist(val_after, bins=100, color="forestgreen", alpha=0.8)
    axes[1].set_title("Colour Value - After")
    axes[1].set_xlabel("Max(R, G, B)")
    fig.tight_layout()
    fig.savefig(output_dir / "hist_colour.png", dpi=150)
    plt.close(fig)

    logger.info("Histograms saved to %s", output_dir)
