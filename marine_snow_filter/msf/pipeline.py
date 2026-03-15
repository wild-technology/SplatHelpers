"""Pipeline orchestrator — single entry point that chains MSF modules.

Usage:
    python -m msf.pipeline --config config.yaml --stage preprocess
    python -m msf.pipeline --config config.yaml --stage colmap_filter
    python -m msf.pipeline --config config.yaml --stage splat_filter
    python -m msf.pipeline --config config.yaml --stage all
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import click

from msf.common import load_config, setup_logging, write_run_log


def _merge_global(cfg: dict, section: str) -> dict:
    """Merge global settings into a section config (section takes precedence)."""
    merged = dict(cfg.get("global", {}))
    merged.update(cfg.get(section, {}))
    return merged


def _build_cli_args(params: dict) -> List[str]:
    """Convert a flat dict of parameters into CLI-style args for click.

    Handles booleans (--flag / --no-flag), None (skip), and lists.
    Keys with underscores are converted to dashes (e.g. input_dir -> --input-dir).
    """
    args: List[str] = []
    for key, value in params.items():
        if value is None:
            continue
        cli_key = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(cli_key)
            # For false booleans, skip (click defaults handle it)
        elif isinstance(value, list):
            # Convert list to comma-separated string
            args.append(cli_key)
            args.append(",".join(str(v) for v in value))
        else:
            args.append(cli_key)
            args.append(str(value))
    return args


def _run_preprocess(cfg: dict) -> None:
    """Run the image preprocessing stage via its click CLI."""
    from msf.preprocess.cli import main as preprocess_main

    params = _merge_global(cfg, "preprocess")
    # Map config keys to CLI option names
    cli_args = []
    cli_args.extend(["--input-dir", str(params.get("input_dir", ""))])
    cli_args.extend(["--output-dir", str(params.get("output_dir", ""))])
    if params.get("method"):
        cli_args.extend(["--method", params["method"]])
    if params.get("compare"):
        cli_args.append("--compare")
    if params.get("patch_size"):
        cli_args.extend(["--patch-size", str(params["patch_size"])])
    if params.get("omega") is not None:
        cli_args.extend(["--omega", str(params["omega"])])
    if params.get("t_min") is not None:
        cli_args.extend(["--t-min", str(params["t_min"])])
    if params.get("gamma_low") is not None:
        cli_args.extend(["--gamma-low", str(params["gamma_low"])])
    if params.get("gamma_high") is not None:
        cli_args.extend(["--gamma-high", str(params["gamma_high"])])
    if params.get("cutoff") is not None:
        cli_args.extend(["--cutoff", str(params["cutoff"])])
    if params.get("retinex_scales"):
        scales = params["retinex_scales"]
        if isinstance(scales, list):
            cli_args.extend(["--retinex-scales", ",".join(str(s) for s in scales)])
        else:
            cli_args.extend(["--retinex-scales", str(scales)])
    if params.get("wb_method"):
        cli_args.extend(["--wb-method", params["wb_method"]])
    if params.get("p_norm") is not None:
        cli_args.extend(["--p-norm", str(params["p_norm"])])
    if params.get("stretch") is False:
        cli_args.append("--no-stretch")
    if params.get("clip_limit") is not None:
        cli_args.extend(["--clip-limit", str(params["clip_limit"])])
    if params.get("tile_size") is not None:
        cli_args.extend(["--tile-size", str(params["tile_size"])])
    if params.get("no_clahe"):
        cli_args.append("--no-clahe")
    if params.get("suppress_particles"):
        cli_args.append("--suppress-particles")
    if params.get("workers"):
        cli_args.extend(["--workers", str(params["workers"])])
    if params.get("dry_run"):
        cli_args.append("--dry-run")
    if params.get("verbose"):
        cli_args.append("--verbose")
    if params.get("log_dir"):
        cli_args.extend(["--log-dir", str(params["log_dir"])])

    preprocess_main(cli_args, standalone_mode=False)


def _run_colmap_filter(cfg: dict) -> None:
    """Run the COLMAP points3D filtering stage via its click CLI."""
    from msf.colmap_filter.cli import main as colmap_main

    params = _merge_global(cfg, "colmap_filter")
    cli_args = []
    cli_args.extend(["--input", str(params.get("input", ""))])
    cli_args.extend(["--output", str(params.get("output", ""))])
    for key in ["min_track_length", "max_reproj_error", "max_color_saturation",
                "min_color_brightness", "max_track_for_color", "sigma_outlier",
                "min_neighbors", "neighbor_radius", "max_temporal_span"]:
        if params.get(key) is not None:
            cli_args.extend(["--" + key.replace("_", "-"), str(params[key])])
    for flag in ["no_track_filter", "no_error_filter", "no_color_filter",
                 "no_outlier_filter", "no_density_filter", "no_temporal_filter"]:
        if params.get(flag):
            cli_args.append("--" + flag.replace("_", "-"))
    if params.get("dry_run"):
        cli_args.append("--dry-run")
    if params.get("verbose"):
        cli_args.append("--verbose")
    if params.get("log_dir"):
        cli_args.extend(["--log-dir", str(params["log_dir"])])

    colmap_main(cli_args, standalone_mode=False)


def _run_splat_filter(cfg: dict) -> None:
    """Run the Gaussian splat .ply filtering stage via its click CLI."""
    from msf.splat_filter.cli import main as splat_main

    params = _merge_global(cfg, "splat_filter")
    cli_args = []
    cli_args.extend(["--input", str(params.get("input", ""))])
    if params.get("output"):
        cli_args.extend(["--output", str(params["output"])])
    if params.get("min_opacity") is not None:
        cli_args.extend(["--min-opacity", str(params["min_opacity"])])
    if params.get("max_scale") is not None:
        cli_args.extend(["--max-scale", str(params["max_scale"])])
    if params.get("max_distance_sigma") is not None:
        cli_args.extend(["--max-distance-sigma", str(params["max_distance_sigma"])])
    if params.get("neutral_filter") is False:
        cli_args.append("--no-neutral-filter")
    for key in ["neutral_v_min", "neutral_s_max", "neutral_opacity_max"]:
        if params.get(key) is not None:
            cli_args.extend(["--" + key.replace("_", "-"), str(params[key])])
    if params.get("density_filter"):
        cli_args.append("--density-filter")
    if params.get("density_radius") is not None:
        cli_args.extend(["--density-radius", str(params["density_radius"])])
    if params.get("density_min_neighbors") is not None:
        cli_args.extend(["--density-min-neighbors", str(params["density_min_neighbors"])])
    if params.get("elongation_filter") is False:
        cli_args.append("--no-elongation-filter")
    if params.get("max_elongation_ratio") is not None:
        cli_args.extend(["--max-elongation-ratio", str(params["max_elongation_ratio"])])
    if params.get("save_histograms"):
        cli_args.append("--save-histograms")
    if params.get("dry_run"):
        cli_args.append("--dry-run")
    if params.get("verbose"):
        cli_args.append("--verbose")
    if params.get("log_dir"):
        cli_args.extend(["--log-dir", str(params["log_dir"])])

    splat_main(cli_args, standalone_mode=False)


STAGES = {
    "preprocess": _run_preprocess,
    "colmap_filter": _run_colmap_filter,
    "splat_filter": _run_splat_filter,
}


@click.command("pipeline")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True),
              help="Path to YAML configuration file.")
@click.option("--stage", required=True,
              type=click.Choice(["preprocess", "colmap_filter", "splat_filter", "all"]),
              help="Pipeline stage to run.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Compute and report without writing output files.")
@click.option("--verbose", is_flag=True, default=False,
              help="Enable DEBUG-level logging.")
@click.option("--workers", type=int, default=None,
              help="Override number of parallel workers.")
@click.option("--log-dir", type=click.Path(), default=None,
              help="Directory for structured run logs.")
def main(config_path: str, stage: str, dry_run: bool, verbose: bool,
         workers: int | None, log_dir: str | None) -> None:
    """Marine Snow Filter pipeline orchestrator.

    Chains preprocessing, COLMAP filtering, and splat filtering stages.
    Each stage can be run independently or all together with --stage all.
    """
    cfg = load_config(Path(config_path))

    # Apply CLI overrides into global section
    if "global" not in cfg:
        cfg["global"] = {}
    if dry_run:
        cfg["global"]["dry_run"] = True
    if verbose:
        cfg["global"]["verbose"] = True
    if workers is not None:
        cfg["global"]["workers"] = workers
    if log_dir is not None:
        cfg["global"]["log_dir"] = log_dir

    logger = setup_logging(
        verbose=cfg["global"].get("verbose", False),
        log_dir=cfg["global"].get("log_dir"),
    )

    if stage == "all":
        stages_to_run = ["preprocess", "colmap_filter", "splat_filter"]
    else:
        stages_to_run = [stage]

    for s in stages_to_run:
        logger.info("=" * 60)
        logger.info("Running stage: %s", s)
        logger.info("=" * 60)
        try:
            STAGES[s](cfg)
            logger.info("Stage %s completed.", s)
        except click.UsageError as e:
            logger.error("Configuration error for stage %s: %s", s, e)
            sys.exit(1)
        except Exception:
            logger.exception("Stage %s failed", s)
            sys.exit(1)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
