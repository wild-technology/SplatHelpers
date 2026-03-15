# Marine Snow Filter (msf)

Tools for removing marine snow (suspended particulate matter) from underwater photogrammetry reconstructions. Operates at three stages of the reconstruction pipeline: image pre-processing, COLMAP point cloud filtering, and 3DGS splat post-filtering.

## Setup

```bash
cd marine_snow_filter
pip install -r requirements.txt
```

Dependencies: numpy, opencv-python-headless, plyfile, scipy, tqdm, click, piexif, pyyaml, matplotlib

## Project Structure

```
marine_snow_filter/
  requirements.txt
  example_config.yaml          # Full config with documented defaults
  msf/
    __init__.py
    common.py                  # Shared: image discovery, metadata, logging, config
    pipeline.py                # Orchestrator CLI (--stage preprocess|colmap_filter|splat_filter|all)
    preprocess/
      dehaze.py                # 5 methods: dcp, homomorphic, retinex, fusion, rcp
      color_correct.py         # 3 WB methods: grayworld, max-rgb, shades-of-gray
      clahe.py                 # CLAHE on LAB L-channel
      cli.py                   # Standalone CLI + compare mode
    colmap_filter/
      points3d_io.py           # Read/write COLMAP points3D (text + binary)
      filter.py                # 6 filters: track length, reproj error, color, outlier, density, temporal
      cli.py                   # Standalone CLI
    splat_filter/
      ply_io.py                # 3DGS .ply I/O, sigmoid/SH math utilities
      filter.py                # 6 filters: opacity, scale, distance, neutral, density, elongation
      cli.py                   # Standalone CLI
  tests/
    test_math.py               # Unit tests for critical math + format parsing
```

## Running

### Standalone modules (preferred for individual stages)

```bash
# Image preprocessing — run with --dry-run first to check discovery
python -m msf.preprocess.cli \
    --input-dir /path/to/raw_images \
    --output-dir /path/to/cleaned \
    --method dcp \
    --wb-method grayworld \
    --stretch \
    --workers 8 \
    --dry-run

# Compare all dehaze methods side-by-side
python -m msf.preprocess.cli \
    --input-dir /path/to/raw_images \
    --output-dir /path/to/comparison \
    --compare \
    --workers 8

# COLMAP points3D filter
python -m msf.colmap_filter.cli \
    --input /path/to/points3D.txt \
    --output /path/to/points3D_filtered.txt \
    --min-track-length 4 \
    --dry-run

# Gaussian splat filter
python -m msf.splat_filter.cli \
    --input /path/to/splat.ply \
    --output /path/to/splat_filtered.ply \
    --save-histograms \
    --dry-run
```

### Pipeline orchestrator (config-driven)

```bash
python -m msf.pipeline --config config.yaml --stage preprocess
python -m msf.pipeline --config config.yaml --stage colmap_filter
python -m msf.pipeline --config config.yaml --stage splat_filter
python -m msf.pipeline --config config.yaml --stage all --dry-run
```

Copy `example_config.yaml` and fill in your paths. CLI flags override config values.

### Tests

```bash
python -m pytest tests/ -v
```

## Typical workflow

1. `python -m msf.preprocess.cli --compare ...` to find the best dehaze method for your dataset
2. `python -m msf.preprocess.cli --method <best> ...` to process all images
3. Run COLMAP or RealityCapture on the cleaned images (external step)
4. `python -m msf.colmap_filter.cli --input points3D.txt ...` to filter the SfM point cloud
5. Train 3DGS on the filtered point cloud (external step)
6. `python -m msf.splat_filter.cli --input splat.ply --save-histograms ...` to post-filter

## Key design decisions

- **Non-destructive**: Never overwrites input files. Always writes to new output paths.
- **Filename preservation**: Output images keep original filenames and directory structure so COLMAP/RealityCapture project references stay valid.
- **EXIF/XMP preservation**: Copies all EXIF (via piexif) and raw XMP APP1 segments. RealityCapture's XMP calibration data is preserved.
- **--dry-run everywhere**: Every module supports dry-run to preview what would happen.
- **No deep learning**: All algorithms are classical CV / analytical. Deterministic and reproducible.
- **MAD over std**: Statistical outlier filters use Median Absolute Deviation for robustness to skewed underwater distributions.

## Architecture notes for development

- All dehaze functions share signature `def method(image: np.ndarray, **kwargs) -> np.ndarray` (BGR uint8 in/out). Registered in `dehaze.METHODS` dict.
- White balance functions registered in `color_correct.WB_METHODS` dict.
- COLMAP I/O has two APIs: dict-based (`read_points3d`/`write_points3d`) for convenience and array-based (`read_points3d_arrays`/`write_points3d_from_arrays`) for filtering performance on large datasets.
- Splat opacity is stored as logit (inverse sigmoid). Use `ply_io.sigmoid()`/`inv_sigmoid()` to convert. Scale is stored as log. Use `ply_io.actual_scale()`.
- SH DC to RGB: `rgb = 0.5 + C0 * f_dc` where `C0 = 0.28209479177387814`.
- Filter functions return boolean masks (True = keep). Combined with logical AND.
- `--save-histograms` in splat filter writes before/after PNG plots for threshold tuning.

## Gotchas

- **Retinex is slow** (~30s per 9MP image due to large-sigma Gaussian blurs). Use fewer workers or smaller retinex_scales for faster iteration.
- **Density filters are expensive** (cKDTree construction on millions of points). Default disabled. Enable only when other filters are insufficient.
- **Color filter false positives**: Gray rock, silt, and metal are low-saturation. The color filter is restricted to short-track points by default (`--max-track-for-color 5`) to mitigate this.
- **piexif only handles JPEG EXIF**. For TIFF/PNG inputs, metadata copying is silently skipped.
- The splat filter's 55% removal rate on the test data is aggressive. Tune `--min-opacity`, `--max-elongation-ratio`, and `--neutral-*` thresholds with `--save-histograms` to find the right balance for your data.
