"""
filter.py - Point cloud filters for removing marine snow from COLMAP points3D.

Each filter function accepts array data (as returned by
:func:`~msf.colmap_filter.points3d_io.read_points3d_arrays`) and returns a
boolean *keep* mask.  Filters are independently toggleable and combined via
logical AND in :func:`apply_filters`, which also reports per-filter removal
counts.

Design notes
------------
* **Track length** is the single strongest marine-snow discriminator.
  Particles drift between frames, so they are only triangulated from a very
  small number of images.
* **Colour** filtering targets the bright, desaturated appearance of
  back-scattered particles.  To avoid false positives on grey rock / silt,
  the colour filter is *by default* restricted to short-track points (track
  length <= ``max_track_for_color``).  Long-track neutral-coloured points
  are overwhelmingly real geometry.
* **Temporal adjacency** catches particles that appear in a burst of
  consecutive images despite having an adequate track length -- their
  temporal span is unnaturally small.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Individual filter functions
# ---------------------------------------------------------------------------

def filter_by_track_length(
    track_lengths: np.ndarray,
    min_length: int = 3,
) -> np.ndarray:
    """Remove points seen in fewer than *min_length* images.

    This is the strongest marine-snow discriminator: particles move between
    frames and therefore cannot be triangulated from many views.

    Parameters
    ----------
    track_lengths : np.ndarray
        ``uint64[N]`` per-point track lengths.
    min_length : int
        Minimum number of images a point must appear in.

    Returns
    -------
    np.ndarray
        ``bool[N]`` keep mask.
    """
    return track_lengths >= min_length


def filter_by_reproj_error(
    errors: np.ndarray,
    max_error: float = 1.0,
) -> np.ndarray:
    """Remove points whose reprojection error exceeds *max_error*.

    Parameters
    ----------
    errors : np.ndarray
        ``float64[N]`` reprojection errors.
    max_error : float
        Maximum allowed reprojection error (pixels).

    Returns
    -------
    np.ndarray
        ``bool[N]`` keep mask.
    """
    return errors <= max_error


def filter_by_color(
    rgb: np.ndarray,
    max_saturation: float = 0.15,
    min_brightness: int = 180,
    track_lengths: Optional[np.ndarray] = None,
    max_track_for_color: Optional[int] = 5,
) -> np.ndarray:
    """Remove bright, desaturated (near-white / grey) points.

    Marine snow particles scatter light and appear as bright, low-saturation
    blobs.  This filter flags points that are **both** bright
    (V > min_brightness / 255) **and** desaturated (S < max_saturation) in
    the HSV colour space.

    To reduce false positives on legitimately grey surfaces (rock, silt),
    the colour filter is **by default only applied to short-track points**
    (track_length <= *max_track_for_color*).  Long-track neutral points are
    almost certainly real geometry, so they are kept regardless.

    Parameters
    ----------
    rgb : np.ndarray
        ``uint8[N, 3]`` colours.
    max_saturation : float
        HSV saturation threshold in [0, 1].
    min_brightness : int
        Minimum HSV *value* (in 0-255 scale) to be considered "bright".
    track_lengths : np.ndarray | None
        If provided together with *max_track_for_color*, only short-track
        points are subject to the colour test.
    max_track_for_color : int | None
        Maximum track length for a point to be colour-filtered.  Set to
        ``None`` or ``0`` to apply the colour filter to **all** points.

    Returns
    -------
    np.ndarray
        ``bool[N]`` keep mask.
    """
    n = rgb.shape[0]
    if n == 0:
        return np.ones(0, dtype=bool)

    # Convert RGB [0-255] -> float [0-1]
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[:, 0], rgb_f[:, 1], rgb_f[:, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Saturation: S = delta / cmax  (0 when cmax == 0)
    sat = np.zeros(n, dtype=np.float32)
    nonzero = cmax > 0
    sat[nonzero] = delta[nonzero] / cmax[nonzero]

    # Value (brightness) in [0, 1]
    val = cmax

    brightness_thresh = min_brightness / 255.0
    is_snow_color = (sat < max_saturation) & (val > brightness_thresh)

    # By default restrict to short-track points
    if track_lengths is not None and max_track_for_color and max_track_for_color > 0:
        is_short_track = track_lengths <= max_track_for_color
        is_snow_color = is_snow_color & is_short_track

    return ~is_snow_color


def filter_by_distance(
    xyz: np.ndarray,
    sigma: float = 3.0,
) -> np.ndarray:
    """Remove statistical outliers by distance from the centroid.

    Uses the **median absolute deviation (MAD)** instead of standard
    deviation for robustness against heavy-tailed distributions::

        threshold = median(dist) + sigma * 1.4826 * MAD(dist)

    The constant 1.4826 makes MAD consistent with the standard deviation
    for normally distributed data.

    Parameters
    ----------
    xyz : np.ndarray
        ``float64[N, 3]`` positions.
    sigma : float
        Number of MAD-scaled deviations above the median for the cutoff.

    Returns
    -------
    np.ndarray
        ``bool[N]`` keep mask.
    """
    if xyz.shape[0] == 0:
        return np.ones(0, dtype=bool)

    centroid = np.median(xyz, axis=0)
    dists = np.linalg.norm(xyz - centroid, axis=1)
    med = np.median(dists)
    mad = np.median(np.abs(dists - med))
    threshold = med + sigma * 1.4826 * mad
    return dists <= threshold


def filter_by_density(
    xyz: np.ndarray,
    min_neighbors: int = 5,
    radius: float = 0.5,
) -> np.ndarray:
    """Remove points with too few neighbours within a given radius.

    Uses :class:`scipy.spatial.cKDTree` for efficient spatial queries.
    Disabled by default (``min_neighbors=0``).

    Parameters
    ----------
    xyz : np.ndarray
        ``float64[N, 3]`` positions.
    min_neighbors : int
        Minimum number of neighbours within *radius*.  Points with fewer
        neighbours are removed.  Set to 0 to disable.
    radius : float
        Search radius.

    Returns
    -------
    np.ndarray
        ``bool[N]`` keep mask.
    """
    n = xyz.shape[0]
    if n == 0 or min_neighbors <= 0:
        return np.ones(n, dtype=bool)

    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)
    # query_ball_point returns lists; count neighbours (subtract 1 for self)
    counts = np.array(
        tree.query_ball_point(xyz, r=radius, return_length=True),
        dtype=np.int64,
    )
    # query_ball_point includes the point itself
    counts -= 1
    return counts >= min_neighbors


def filter_by_temporal_adjacency(
    track_lengths: np.ndarray,
    tracks_flat: np.ndarray,
    max_temporal_span: int = 3,
    min_track_for_temporal: int = 3,
) -> np.ndarray:
    """Remove points whose track spans only a narrow window of image IDs.

    A marine-snow particle may be triangulated from several *consecutive*
    images (adequate track length) yet span only a tiny temporal range.
    Real scene points are typically observed across a wider range of
    viewpoints / timestamps.

    Only applied to points with ``track_length >= min_track_for_temporal``
    so as not to double-filter with the track-length filter.

    Parameters
    ----------
    track_lengths : np.ndarray
        ``uint64[N]`` per-point track lengths.
    tracks_flat : np.ndarray
        ``uint32[T, 2]`` concatenated ``(image_id, point2d_idx)`` pairs.
    max_temporal_span : int
        Maximum ``max(image_ids) - min(image_ids)`` for a point to be
        considered temporally clustered (and therefore suspicious).
    min_track_for_temporal : int
        Minimum track length for the temporal check to apply.  Points with
        shorter tracks are not filtered here (they are handled by the
        track-length filter instead).

    Returns
    -------
    np.ndarray
        ``bool[N]`` keep mask.
    """
    n = len(track_lengths)
    keep = np.ones(n, dtype=bool)

    if n == 0 or tracks_flat.shape[0] == 0:
        return keep

    # Compute offsets into tracks_flat
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(track_lengths.astype(np.int64), out=offsets[1:])

    # Only check points with sufficient track length
    candidates = np.nonzero(track_lengths >= min_track_for_temporal)[0]

    for i in candidates:
        start = offsets[i]
        end = offsets[i + 1]
        if end <= start:
            continue
        image_ids = tracks_flat[start:end, 0]
        span = int(image_ids.max()) - int(image_ids.min())
        if span <= max_temporal_span:
            keep[i] = False

    return keep


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_filters(
    points_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    min_track_length: int = 3,
    max_reproj_error: float = 1.0,
    max_color_saturation: float = 0.15,
    min_color_brightness: int = 180,
    max_track_for_color: int = 5,
    sigma_outlier: float = 3.0,
    min_neighbors: int = 0,
    neighbor_radius: float = 0.5,
    max_temporal_span: int = 3,
    min_track_for_temporal: int = 3,
    no_track_filter: bool = False,
    no_error_filter: bool = False,
    no_color_filter: bool = False,
    no_outlier_filter: bool = False,
    no_density_filter: bool = False,
    no_temporal_filter: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply all enabled filters and return a combined keep mask.

    Filters are combined with logical AND (a point must pass *all* enabled
    filters to be kept).  Per-filter removal counts are reported in the
    returned stats dict.

    Parameters
    ----------
    points_data : tuple
        ``(ids, xyz, rgb, errors, track_lengths, tracks_flat)`` as returned
        by :func:`~msf.colmap_filter.points3d_io.read_points3d_arrays`.
    min_track_length : int
        Minimum track length (default 3).
    max_reproj_error : float
        Maximum reprojection error in pixels (default 1.0).
    max_color_saturation : float
        HSV saturation threshold (default 0.15).
    min_color_brightness : int
        HSV brightness threshold on 0-255 scale (default 180).
    max_track_for_color : int
        Only colour-filter points with track length <= this value.  Set 0
        to apply to all points (default 5).
    sigma_outlier : float
        MAD sigma for statistical outlier removal (default 3.0).
    min_neighbors : int
        Minimum neighbour count for density filter.  0 = disabled (default).
    neighbor_radius : float
        Radius for neighbour search (default 0.5).
    max_temporal_span : int
        Maximum image-ID span for temporal adjacency filter (default 3).
    min_track_for_temporal : int
        Minimum track length for temporal adjacency check (default 3).
    no_track_filter : bool
        Disable track-length filter.
    no_error_filter : bool
        Disable reprojection-error filter.
    no_color_filter : bool
        Disable colour filter.
    no_outlier_filter : bool
        Disable statistical-outlier filter.
    no_density_filter : bool
        Disable density filter.
    no_temporal_filter : bool
        Disable temporal-adjacency filter.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``(keep_mask, stats)`` where *keep_mask* is ``bool[N]`` and *stats*
        is a dict with keys like ``'total'``, ``'kept'``,
        ``'removed_by_track_length'``, etc.
    """
    ids, xyz, rgb, errors, track_lengths, tracks_flat = points_data
    n = len(ids)
    combined_mask = np.ones(n, dtype=bool)
    stats: Dict[str, Any] = {"total": n}

    # Helper to apply a single filter and record stats
    def _apply(name: str, mask: np.ndarray) -> None:
        removed_by_this = int((combined_mask & ~mask).sum())
        stats[f"removed_by_{name}"] = removed_by_this
        combined_mask[:] = combined_mask & mask

    # 1. Track length
    if not no_track_filter:
        _apply("track_length", filter_by_track_length(track_lengths, min_length=min_track_length))

    # 2. Reprojection error
    if not no_error_filter:
        _apply("reproj_error", filter_by_reproj_error(errors, max_error=max_reproj_error))

    # 3. Colour
    if not no_color_filter:
        effective_max_track = max_track_for_color if max_track_for_color > 0 else None
        _apply(
            "color",
            filter_by_color(
                rgb,
                max_saturation=max_color_saturation,
                min_brightness=min_color_brightness,
                track_lengths=track_lengths,
                max_track_for_color=effective_max_track,
            ),
        )

    # 4. Statistical outlier
    if not no_outlier_filter:
        _apply("outlier", filter_by_distance(xyz, sigma=sigma_outlier))

    # 5. Density
    if not no_density_filter and min_neighbors > 0:
        _apply("density", filter_by_density(xyz, min_neighbors=min_neighbors, radius=neighbor_radius))

    # 6. Temporal adjacency
    if not no_temporal_filter:
        _apply(
            "temporal_adjacency",
            filter_by_temporal_adjacency(
                track_lengths,
                tracks_flat,
                max_temporal_span=max_temporal_span,
                min_track_for_temporal=min_track_for_temporal,
            ),
        )

    stats["kept"] = int(combined_mask.sum())
    stats["removed_total"] = n - stats["kept"]
    return combined_mask, stats
