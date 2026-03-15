"""
cli.py - Click CLI for the marine snow filter image preprocessing pipeline.

Pipeline order: dehaze -> colour correction -> CLAHE (each independently
disableable).  Supports parallel processing via ProcessPoolExecutor.

Usage::

    python -m msf.preprocess.cli --input-dir raw/ --output-dir enhanced/
    python -m msf.preprocess.cli --input-dir raw/ --output-dir enhanced/ --compare
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import cv2
import numpy as np
from tqdm import tqdm

from msf.common import (
    copy_metadata,
    discover_images,
    mirror_directory,
    setup_logging,
    write_run_log,
)
from msf.preprocess.clahe import apply_clahe
from msf.preprocess.color_correct import WB_METHODS, percentile_stretch
from msf.preprocess.dehaze import METHODS as DEHAZE_METHODS
from msf.preprocess.dehaze import suppress_particles

logger = logging.getLogger("msf")


# ---------------------------------------------------------------------------
# Single-image processing (must be top-level for pickling by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_single_image(
    input_dir: str,
    output_dir: str,
    rel_path: str,
    method_name: str,
    dehaze_kwargs: Dict[str, Any],
    wb_method: Optional[str],
    wb_kwargs: Dict[str, Any],
    do_stretch: bool,
    stretch_kwargs: Dict[str, Any],
    do_clahe: bool,
    clahe_kwargs: Dict[str, Any],
    do_suppress: bool,
    suppress_kwargs: Dict[str, Any],
) -> Tuple[str, bool, str]:
    """Process a single image through the full pipeline.

    This function is designed to be called from a ``ProcessPoolExecutor``.
    All arguments are plain types (no lambdas, no unpicklable objects).

    Parameters
    ----------
    input_dir : str
        Root input directory.
    output_dir : str
        Root output directory.
    rel_path : str
        Image path relative to *input_dir*.
    method_name : str
        Dehaze method key (from ``DEHAZE_METHODS``).
    dehaze_kwargs : dict
        Extra keyword arguments forwarded to the dehaze method.
    wb_method : str or None
        White-balance method key, or None to skip.
    wb_kwargs : dict
        Extra keyword arguments for the white-balance method.
    do_stretch : bool
        Whether to apply percentile stretch after white balance.
    stretch_kwargs : dict
        Keyword arguments for ``percentile_stretch``.
    do_clahe : bool
        Whether to apply CLAHE.
    clahe_kwargs : dict
        Keyword arguments for ``apply_clahe``.
    do_suppress : bool
        Whether to run bright-particle suppression *before* the pipeline.
    suppress_kwargs : dict
        Keyword arguments for ``suppress_particles``.

    Returns
    -------
    tuple[str, bool, str]
        ``(rel_path, success, message)``
    """
    try:
        src = Path(input_dir) / rel_path
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            return (rel_path, False, f"Failed to read {src}")

        # Optional particle suppression (before main pipeline)
        if do_suppress:
            image = suppress_particles(image, **suppress_kwargs)

        # Stage 1: Dehaze
        dehaze_fn = DEHAZE_METHODS[method_name]
        image = dehaze_fn(image, **dehaze_kwargs)

        # Stage 2: Colour correction / white balance
        if wb_method is not None:
            wb_fn = WB_METHODS[wb_method]
            image = wb_fn(image, **wb_kwargs)

        if do_stretch:
            image = percentile_stretch(image, **stretch_kwargs)

        # Stage 3: CLAHE
        if do_clahe:
            image = apply_clahe(image, **clahe_kwargs)

        # Write output
        dst = mirror_directory(input_dir, output_dir, rel_path)
        cv2.imwrite(str(dst), image)

        # Copy EXIF / XMP metadata
        copy_metadata(src, dst)

        return (rel_path, True, "OK")

    except Exception as exc:  # noqa: BLE001
        return (rel_path, False, str(exc))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("preprocess")
@click.option(
    "--input-dir", required=True, type=click.Path(exists=True, file_okay=False),
    help="Directory containing source images.",
)
@click.option(
    "--output-dir", required=True, type=click.Path(file_okay=False),
    help="Directory for enhanced output images.",
)
@click.option(
    "--method",
    type=click.Choice(list(DEHAZE_METHODS.keys()), case_sensitive=False),
    default="dcp",
    show_default=True,
    help="Dehaze algorithm.",
)
@click.option("--compare", is_flag=True, help="Run ALL methods; write to method-named subdirectories.")
# Dehaze parameters
@click.option("--patch-size", type=int, default=15, show_default=True, help="DCP/RCP patch size.")
@click.option("--omega", type=float, default=0.85, show_default=True, help="DCP/RCP haze removal strength.")
@click.option("--t-min", type=float, default=0.1, show_default=True, help="DCP/RCP minimum transmission.")
@click.option("--gamma-low", type=float, default=0.5, show_default=True, help="Homomorphic low-frequency gain.")
@click.option("--gamma-high", type=float, default=1.5, show_default=True, help="Homomorphic high-frequency gain.")
@click.option("--cutoff", type=float, default=30.0, show_default=True, help="Homomorphic filter cutoff.")
@click.option(
    "--retinex-scales", type=str, default="15,80,250", show_default=True,
    help="Comma-separated retinex sigma scales.",
)
# Colour correction
@click.option(
    "--wb-method",
    type=click.Choice(["grayworld", "max-rgb", "shades-of-gray", "none"], case_sensitive=False),
    default="grayworld",
    show_default=True,
    help="White-balance method ('none' to skip).",
)
@click.option("--p-norm", type=int, default=6, show_default=True, help="Shades-of-gray Minkowski exponent.")
@click.option("--stretch/--no-stretch", default=True, show_default=True, help="Apply percentile stretch.")
# CLAHE
@click.option("--clip-limit", type=float, default=2.0, show_default=True, help="CLAHE clip limit.")
@click.option("--tile-size", type=int, default=8, show_default=True, help="CLAHE tile grid size.")
@click.option("--no-clahe", is_flag=True, help="Disable CLAHE.")
# Particle suppression
@click.option("--suppress-particles", is_flag=True, help="Run bright-speck suppression before the pipeline.")
# Execution
@click.option(
    "--workers", type=int, default=max(os.cpu_count() - 1, 1),  # type: ignore[operator]
    show_default=True, help="Number of parallel workers.",
)
@click.option("--dry-run", is_flag=True, help="Discover images and print plan without processing.")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
@click.option("--log-dir", type=click.Path(file_okay=False), default=None, help="Directory for run logs.")
def main(
    input_dir: str,
    output_dir: str,
    method: str,
    compare: bool,
    patch_size: int,
    omega: float,
    t_min: float,
    gamma_low: float,
    gamma_high: float,
    cutoff: float,
    retinex_scales: str,
    wb_method: str,
    p_norm: int,
    stretch: bool,
    clip_limit: float,
    tile_size: int,
    no_clahe: bool,
    suppress_particles: bool,
    workers: int,
    dry_run: bool,
    verbose: bool,
    log_dir: Optional[str],
) -> None:
    """Preprocess underwater images: dehaze, colour-correct, and enhance contrast."""
    setup_logging(verbose=verbose, log_dir=log_dir)

    # Parse retinex scales
    scales = [int(s.strip()) for s in retinex_scales.split(",")]

    # Build dehaze kwargs (methods ignore irrelevant keys via **kwargs)
    dehaze_kwargs: Dict[str, Any] = {
        "patch_size": patch_size,
        "omega": omega,
        "t_min": t_min,
        "gamma_low": gamma_low,
        "gamma_high": gamma_high,
        "cutoff": cutoff,
        "scales": scales,
    }

    wb = wb_method.lower() if wb_method.lower() != "none" else None
    wb_kwargs: Dict[str, Any] = {"p_norm": p_norm}
    stretch_kwargs: Dict[str, Any] = {}  # uses defaults
    clahe_kwargs: Dict[str, Any] = {"clip_limit": clip_limit, "tile_size": tile_size}
    suppress_kwargs: Dict[str, Any] = {}

    do_clahe = not no_clahe

    # Discover images
    images = discover_images(input_dir)
    if not images:
        logger.warning("No images found in %s", input_dir)
        return

    logger.info("Found %d images in %s", len(images), input_dir)

    # Determine methods to run
    methods_to_run: List[str] = list(DEHAZE_METHODS.keys()) if compare else [method]

    if dry_run:
        click.echo(f"Images:  {len(images)}")
        click.echo(f"Methods: {', '.join(methods_to_run)}")
        click.echo(f"WB:      {wb or 'none'}")
        click.echo(f"Stretch: {stretch}")
        click.echo(f"CLAHE:   {do_clahe}")
        click.echo(f"Suppress:{suppress_particles}")
        click.echo(f"Workers: {workers}")
        for img in images[:10]:
            click.echo(f"  {img}")
        if len(images) > 10:
            click.echo(f"  ... and {len(images) - 10} more")
        return

    t_start = time.time()
    total_ok = 0
    total_fail = 0

    for method_name in methods_to_run:
        if compare:
            out_dir = str(Path(output_dir) / method_name)
        else:
            out_dir = output_dir

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Running method '%s' -> %s", method_name, out_dir)

        futures = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for rel in images:
                fut = pool.submit(
                    _process_single_image,
                    str(input_dir),
                    out_dir,
                    str(rel),
                    method_name,
                    dehaze_kwargs,
                    wb,
                    wb_kwargs,
                    stretch,
                    stretch_kwargs,
                    do_clahe,
                    clahe_kwargs,
                    suppress_particles,
                    suppress_kwargs,
                )
                futures.append(fut)

            desc = f"[{method_name}]"
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                rel_path, success, msg = fut.result()
                if success:
                    total_ok += 1
                else:
                    total_fail += 1
                    logger.error("FAILED %s: %s", rel_path, msg)

    elapsed = time.time() - t_start
    logger.info(
        "Done: %d succeeded, %d failed, %.1f s elapsed", total_ok, total_fail, elapsed
    )

    # Write run log
    if log_dir:
        params = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "methods": methods_to_run,
            "dehaze_kwargs": dehaze_kwargs,
            "wb_method": wb,
            "stretch": stretch,
            "clahe": do_clahe,
            "suppress_particles": suppress_particles,
            "workers": workers,
        }
        stats = {
            "images_found": len(images),
            "succeeded": total_ok,
            "failed": total_fail,
            "elapsed_s": round(elapsed, 2),
        }
        log_path = write_run_log(log_dir, params, stats)
        logger.info("Run log written to %s", log_path)


if __name__ == "__main__":
    main()
