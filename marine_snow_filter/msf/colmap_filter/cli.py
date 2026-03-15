"""
cli.py - Click CLI for the COLMAP points3D marine snow filter.

Runnable as::

    python -m msf.colmap_filter.cli --input points3D.txt --output filtered.txt

All filters are enabled by default and can be individually disabled with
``--no-*-filter`` flags.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import click

from msf.colmap_filter.filter import apply_filters
from msf.colmap_filter.points3d_io import (
    read_points3d_arrays,
    write_points3d_from_arrays,
)
from msf.common import setup_logging, write_run_log


@click.command("colmap-filter")
@click.option(
    "--input", "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to input points3D.txt or points3D.bin.",
)
@click.option(
    "--output", "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to write the filtered points3D file.",
)
@click.option(
    "--min-track-length", default=3, show_default=True, type=int,
    help="Minimum number of images a point must appear in.",
)
@click.option(
    "--max-reproj-error", default=1.0, show_default=True, type=float,
    help="Maximum reprojection error (pixels).",
)
@click.option(
    "--max-color-saturation", default=0.15, show_default=True, type=float,
    help="HSV saturation threshold for bright/desaturated filtering.",
)
@click.option(
    "--min-color-brightness", default=180, show_default=True, type=int,
    help="HSV value threshold (0-255) for brightness filtering.",
)
@click.option(
    "--max-track-for-color", default=5, show_default=True, type=int,
    help="Only apply color filter to points with track length <= this value. "
         "Set 0 to apply to all points.",
)
@click.option(
    "--sigma-outlier", default=3.0, show_default=True, type=float,
    help="MAD sigma multiplier for statistical outlier removal.",
)
@click.option(
    "--min-neighbors", default=0, show_default=True, type=int,
    help="Minimum neighbour count for density filter. 0 = disabled.",
)
@click.option(
    "--neighbor-radius", default=0.5, show_default=True, type=float,
    help="Search radius for the density filter.",
)
@click.option(
    "--max-temporal-span", default=3, show_default=True, type=int,
    help="Maximum image-ID span for temporal adjacency filter.",
)
@click.option(
    "--no-track-filter", is_flag=True, default=False,
    help="Disable the track-length filter.",
)
@click.option(
    "--no-error-filter", is_flag=True, default=False,
    help="Disable the reprojection-error filter.",
)
@click.option(
    "--no-color-filter", is_flag=True, default=False,
    help="Disable the colour filter.",
)
@click.option(
    "--no-outlier-filter", is_flag=True, default=False,
    help="Disable the statistical-outlier filter.",
)
@click.option(
    "--no-density-filter", is_flag=True, default=False,
    help="Disable the density filter.",
)
@click.option(
    "--no-temporal-filter", is_flag=True, default=False,
    help="Disable the temporal-adjacency filter.",
)
@click.option(
    "--dry-run", is_flag=True, default=False,
    help="Run filters and report stats without writing output.",
)
@click.option(
    "--verbose", is_flag=True, default=False,
    help="Enable verbose (DEBUG-level) console output.",
)
@click.option(
    "--log-dir", default=None, type=click.Path(path_type=Path),
    help="Directory for log files and run logs.",
)
def main(
    input_path: Path,
    output_path: Path,
    min_track_length: int,
    max_reproj_error: float,
    max_color_saturation: float,
    min_color_brightness: int,
    max_track_for_color: int,
    sigma_outlier: float,
    min_neighbors: int,
    neighbor_radius: float,
    max_temporal_span: int,
    no_track_filter: bool,
    no_error_filter: bool,
    no_color_filter: bool,
    no_outlier_filter: bool,
    no_density_filter: bool,
    no_temporal_filter: bool,
    dry_run: bool,
    verbose: bool,
    log_dir: Optional[Path],
) -> None:
    """Filter marine snow from COLMAP points3D files.

    Reads a points3D file (text or binary), applies a configurable chain of
    filters to remove marine-snow artifacts, and writes the cleaned result.
    """
    logger = setup_logging(verbose=verbose, log_dir=log_dir)
    t0 = time.perf_counter()

    # ---- Read ----
    logger.info("Reading %s ...", input_path)
    points_data = read_points3d_arrays(input_path)
    ids, xyz, rgb, errors, track_lengths, tracks_flat = points_data
    logger.info("Loaded %d points.", len(ids))

    # ---- Filter ----
    filter_params = dict(
        min_track_length=min_track_length,
        max_reproj_error=max_reproj_error,
        max_color_saturation=max_color_saturation,
        min_color_brightness=min_color_brightness,
        max_track_for_color=max_track_for_color,
        sigma_outlier=sigma_outlier,
        min_neighbors=min_neighbors,
        neighbor_radius=neighbor_radius,
        max_temporal_span=max_temporal_span,
        min_track_for_temporal=min_track_length,  # reuse track length threshold
        no_track_filter=no_track_filter,
        no_error_filter=no_error_filter,
        no_color_filter=no_color_filter,
        no_outlier_filter=no_outlier_filter,
        no_density_filter=no_density_filter,
        no_temporal_filter=no_temporal_filter,
    )

    logger.info("Applying filters ...")
    keep_mask, stats = apply_filters(points_data, **filter_params)

    # ---- Report ----
    logger.info("Filter results:")
    logger.info("  Total points:   %d", stats["total"])
    for key, val in stats.items():
        if key.startswith("removed_by_"):
            filter_name = key[len("removed_by_"):]
            logger.info("  Removed by %-20s %d", filter_name + ":", val)
    logger.info("  Total removed:  %d", stats["removed_total"])
    logger.info("  Points kept:    %d (%.1f%%)",
                stats["kept"],
                100.0 * stats["kept"] / max(stats["total"], 1))

    # ---- Write ----
    if dry_run:
        logger.info("Dry run -- no output written.")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing %s ...", output_path)
        write_points3d_from_arrays(
            output_path, ids, xyz, rgb, errors, track_lengths, tracks_flat,
            mask=keep_mask,
        )
        logger.info("Done. Wrote %d points.", stats["kept"])

    elapsed = time.perf_counter() - t0
    stats["elapsed_seconds"] = round(elapsed, 2)
    logger.info("Elapsed: %.1f s", elapsed)

    # ---- Run log ----
    if log_dir is not None:
        run_log_path = write_run_log(
            log_dir,
            params={**filter_params, "input": str(input_path), "output": str(output_path)},
            stats=stats,
        )
        logger.debug("Run log written to %s", run_log_path)


if __name__ == "__main__":
    main()
