"""
cli.py - Click command-line interface for the Gaussian splat filter.

Runnable as::

    python -m msf.splat_filter.cli --input scene.ply [OPTIONS]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import click

from msf.common import setup_logging, write_run_log


@click.command("splat-filter")
@click.option(
    "--input", "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the input 3DGS .ply file.",
)
@click.option(
    "--output", "output_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output .ply path. Default: <input_stem>_filtered.ply.",
)
@click.option(
    "--min-opacity",
    default=0.02,
    type=float,
    show_default=True,
    help="Minimum sigmoid opacity to keep.",
)
@click.option(
    "--max-scale",
    default=None,
    type=float,
    help="Maximum axis scale. Default: auto (median + 3*MAD).",
)
@click.option(
    "--max-distance-sigma",
    default=3.0,
    type=float,
    show_default=True,
    help="MAD-based sigma for distance outlier removal.",
)
@click.option(
    "--neutral-filter/--no-neutral-filter",
    default=True,
    show_default=True,
    help="Enable/disable the neutral-colour marine-snow filter.",
)
@click.option(
    "--neutral-v-min",
    default=0.7,
    type=float,
    show_default=True,
    help="Minimum HSV value to flag as bright.",
)
@click.option(
    "--neutral-s-max",
    default=0.15,
    type=float,
    show_default=True,
    help="Maximum HSV saturation to flag as neutral.",
)
@click.option(
    "--neutral-opacity-max",
    default=0.3,
    type=float,
    show_default=True,
    help="Maximum sigmoid opacity for neutral filter.",
)
@click.option(
    "--density-filter/--no-density-filter",
    default=False,
    show_default=True,
    help="Enable/disable the spatial density filter.",
)
@click.option(
    "--density-radius",
    default=0.5,
    type=float,
    show_default=True,
    help="Radius for neighbour counting.",
)
@click.option(
    "--density-min-neighbors",
    default=5,
    type=int,
    show_default=True,
    help="Minimum neighbours required to keep a gaussian.",
)
@click.option(
    "--elongation-filter/--no-elongation-filter",
    default=True,
    show_default=True,
    help="Enable/disable the elongation (needle) filter.",
)
@click.option(
    "--max-elongation-ratio",
    default=50.0,
    type=float,
    show_default=True,
    help="Maximum allowed max_scale/min_scale ratio.",
)
@click.option(
    "--save-histograms",
    is_flag=True,
    default=False,
    help="Save before/after distribution histograms as PNGs.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Compute statistics without writing the output file.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG) logging.",
)
@click.option(
    "--log-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Directory for log files and run logs.",
)
def main(
    input_path: Path,
    output_path: Optional[Path],
    min_opacity: float,
    max_scale: Optional[float],
    max_distance_sigma: float,
    neutral_filter: bool,
    neutral_v_min: float,
    neutral_s_max: float,
    neutral_opacity_max: float,
    density_filter: bool,
    density_radius: float,
    density_min_neighbors: int,
    elongation_filter: bool,
    max_elongation_ratio: float,
    save_histograms: bool,
    dry_run: bool,
    verbose: bool,
    log_dir: Optional[Path],
) -> None:
    """Filter a 3D Gaussian Splatting .ply file to remove marine snow and artifacts."""
    # Lazy imports so --help stays fast
    from msf.splat_filter.filter import apply_filters, save_histograms as _save_hists
    from msf.splat_filter.ply_io import read_splat, write_splat

    logger = setup_logging(verbose=verbose, log_dir=log_dir)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_filtered.ply"

    logger.info("Input:  %s", input_path)
    logger.info("Output: %s", output_path)

    # --- Read ---
    t0 = time.perf_counter()
    logger.info("Reading PLY ...")
    plydata, data = read_splat(input_path)
    logger.info("Loaded %d gaussians in %.1fs", len(data), time.perf_counter() - t0)

    # --- Filter ---
    params = dict(
        min_opacity=min_opacity,
        max_scale=max_scale,
        max_distance_sigma=max_distance_sigma,
        neutral_filter=neutral_filter,
        neutral_v_min=neutral_v_min,
        neutral_s_max=neutral_s_max,
        neutral_opacity_max=neutral_opacity_max,
        density_filter=density_filter,
        density_radius=density_radius,
        density_min_neighbors=density_min_neighbors,
        elongation_filter=elongation_filter,
        max_elongation_ratio=max_elongation_ratio,
    )

    t1 = time.perf_counter()
    keep_mask, stats = apply_filters(plydata, data, **params)
    stats["filter_time_s"] = round(time.perf_counter() - t1, 2)

    # --- Histograms ---
    if save_histograms:
        hist_dir = (log_dir or output_path.parent) / "histograms"
        logger.info("Saving histograms to %s ...", hist_dir)
        _save_hists(data, keep_mask, hist_dir)

    # --- Write ---
    if dry_run:
        logger.info("Dry run - no output file written.")
    else:
        t2 = time.perf_counter()
        logger.info("Writing filtered PLY ...")
        write_splat(output_path, plydata, keep_mask)
        stats["write_time_s"] = round(time.perf_counter() - t2, 2)
        logger.info("Wrote %d gaussians to %s", stats["total_kept"], output_path)

    # --- Run log ---
    if log_dir is not None:
        params_log = {k: str(v) if isinstance(v, Path) else v for k, v in params.items()}
        params_log["input"] = str(input_path)
        params_log["output"] = str(output_path)
        params_log["dry_run"] = dry_run
        log_path = write_run_log(log_dir, params_log, stats)
        logger.info("Run log: %s", log_path)

    # --- Summary to stdout ---
    click.echo(f"\nSummary:")
    click.echo(f"  Input gaussians:  {stats['total_input']:>10,}")
    click.echo(f"  Kept:             {stats['total_kept']:>10,}")
    click.echo(f"  Removed:          {stats['total_removed']:>10,}  "
               f"({100.0 * stats['total_removed'] / max(stats['total_input'], 1):.1f}%)")
    click.echo(f"    opacity:        {stats['opacity_removed']:>10,}")
    click.echo(f"    scale:          {stats['scale_removed']:>10,}")
    click.echo(f"    distance:       {stats['distance_removed']:>10,}")
    click.echo(f"    neutral:        {stats['neutral_removed']:>10,}")
    click.echo(f"    density:        {stats['density_removed']:>10,}")
    click.echo(f"    elongation:     {stats['elongation_removed']:>10,}")


# Allow `python -m msf.splat_filter.cli`
if __name__ == "__main__":
    main()
