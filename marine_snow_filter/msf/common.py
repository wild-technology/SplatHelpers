"""
common.py - Shared utilities for the marine snow filter (msf) pipeline.

Provides image discovery, directory mirroring, EXIF/XMP metadata copying,
logging setup, run-log writing, and YAML config loading with CLI override
merging.  Designed for underwater photogrammetry workflows where preserving
camera-calibration metadata (e.g. RealityCapture XMP) is critical.
"""

from __future__ import annotations

import copy
import json
import logging
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import piexif
import yaml


# ---------------------------------------------------------------------------
# 1. Image discovery
# ---------------------------------------------------------------------------

def discover_images(
    input_dir: Union[str, Path],
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
) -> List[Path]:
    """Recursively find all image files under *input_dir*.

    Parameters
    ----------
    input_dir : str | Path
        Root directory to search.
    extensions : tuple of str
        Case-insensitive file extensions to include (with leading dot).

    Returns
    -------
    list[Path]
        Paths **relative to *input_dir***, sorted by name for deterministic
        ordering across platforms.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    ext_lower = {e.lower() for e in extensions}
    images: List[Path] = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in ext_lower:
            images.append(p.relative_to(input_dir))

    images.sort(key=lambda p: p.name)
    return images


# ---------------------------------------------------------------------------
# 2. Directory mirroring
# ---------------------------------------------------------------------------

def mirror_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    relative_path: Union[str, Path],
) -> Path:
    """Create output sub-directories that mirror the input structure.

    Parameters
    ----------
    input_dir : str | Path
        Original root (used only for context / validation).
    output_dir : str | Path
        Destination root directory.
    relative_path : str | Path
        Path of a file relative to *input_dir* (e.g. ``sub/img001.jpg``).

    Returns
    -------
    Path
        The full output file path (``output_dir / relative_path``), with its
        parent directories already created on disk.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    relative_path = Path(relative_path)

    dest = output_dir / relative_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


# ---------------------------------------------------------------------------
# 3. Metadata copying (EXIF + XMP)
# ---------------------------------------------------------------------------

def _read_xmp_from_jpeg(src_path: Path) -> Optional[bytes]:
    """Extract the raw XMP APP1 segment from a JPEG file.

    We scan APP1 markers (``0xFFE1``) for one whose payload begins with the
    XMP namespace URI ``http://ns.adobe.com/xap/1.0/\\x00``.  This preserves
    RealityCapture calibration data verbatim.

    Returns the full XMP packet (without the APP1 header) or *None*.
    """
    xmp_marker = b"http://ns.adobe.com/xap/1.0/\x00"
    try:
        with open(src_path, "rb") as fh:
            # Verify JPEG SOI
            if fh.read(2) != b"\xff\xd8":
                return None

            while True:
                marker = fh.read(2)
                if len(marker) < 2:
                    break
                if marker[0:1] != b"\xff":
                    break

                # Skip padding 0xFF bytes
                while marker[1:2] == b"\xff":
                    next_byte = fh.read(1)
                    if not next_byte:
                        return None
                    marker = b"\xff" + next_byte

                marker_code = marker[1]

                # SOS (Start of Scan) – no more metadata segments
                if marker_code == 0xDA:
                    break

                # Markers without a length field
                if marker_code in (0x00, 0x01) or 0xD0 <= marker_code <= 0xD7:
                    continue

                length_bytes = fh.read(2)
                if len(length_bytes) < 2:
                    break
                seg_length = struct.unpack(">H", length_bytes)[0]
                payload = fh.read(seg_length - 2)

                # APP1 marker
                if marker_code == 0xE1 and payload.startswith(xmp_marker):
                    # Return the XMP XML after the namespace+null header
                    return payload

    except OSError:
        return None

    return None


def _inject_xmp_into_jpeg(dst_path: Path, xmp_payload: bytes) -> None:
    """Write an XMP APP1 segment into an existing JPEG file.

    The segment is inserted right after the SOI marker and before any
    existing segments, which is the conventional position for XMP.
    """
    with open(dst_path, "rb") as fh:
        data = fh.read()

    if data[:2] != b"\xff\xd8":
        return  # not a valid JPEG

    seg_length = len(xmp_payload) + 2  # +2 for the length field itself
    app1_header = b"\xff\xe1" + struct.pack(">H", seg_length)

    # Insert after SOI (first 2 bytes)
    new_data = data[:2] + app1_header + xmp_payload + data[2:]

    with open(dst_path, "wb") as fh:
        fh.write(new_data)


def copy_metadata(src_path: Union[str, Path], dst_path: Union[str, Path]) -> None:
    """Copy EXIF and XMP metadata from *src_path* to *dst_path*.

    - **EXIF** is transferred with *piexif* (dump/insert).
    - **XMP** is read as a raw APP1 segment so that RealityCapture
      calibration data is preserved byte-for-byte.

    For non-JPEG files the function returns silently without error.
    If the source lacks EXIF or XMP the corresponding step is skipped.

    Parameters
    ----------
    src_path, dst_path : str | Path
        Source and destination image files.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    jpeg_exts = {".jpg", ".jpeg"}
    if src_path.suffix.lower() not in jpeg_exts or dst_path.suffix.lower() not in jpeg_exts:
        return  # silently skip non-JPEG files

    if not src_path.is_file() or not dst_path.is_file():
        return

    logger = logging.getLogger("msf")

    # --- EXIF ---
    try:
        exif_dict = piexif.load(str(src_path))
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, str(dst_path))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not copy EXIF from %s: %s", src_path, exc)

    # --- XMP ---
    xmp_payload = _read_xmp_from_jpeg(src_path)
    if xmp_payload is not None:
        try:
            _inject_xmp_into_jpeg(dst_path, xmp_payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not copy XMP from %s: %s", src_path, exc)


# ---------------------------------------------------------------------------
# 4. Logging setup
# ---------------------------------------------------------------------------

def setup_logging(
    verbose: bool,
    log_dir: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Configure Python logging for the *msf* pipeline.

    Parameters
    ----------
    verbose : bool
        If *True* the console level is ``DEBUG``; otherwise ``INFO``.
    log_dir : str | Path | None
        When provided, a ``FileHandler`` writing to
        ``<log_dir>/msf.log`` is added as well.

    Returns
    -------
    logging.Logger
        The logger named ``'msf'``.
    """
    logger = logging.getLogger("msf")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG if verbose else logging.INFO)
        console.setFormatter(fmt)
        logger.addHandler(console)

        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / "msf.log", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# 5. Run log
# ---------------------------------------------------------------------------

def write_run_log(
    log_dir: Union[str, Path],
    params: Dict[str, Any],
    stats: Dict[str, Any],
) -> Path:
    """Write a structured JSON run log.

    The file is written to ``<log_dir>/run_YYYYMMDD_HHMMSS.json`` and
    contains the UTC timestamp, the full parameter dict, and summary stats.

    Parameters
    ----------
    log_dir : str | Path
        Directory for run logs (created if necessary).
    params : dict
        Pipeline parameters to record.
    stats : dict
        Summary statistics (e.g. image count, duration).

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    filename = now.strftime("run_%Y%m%d_%H%M%S.json")
    out_path = log_dir / filename

    record = {
        "timestamp": now.isoformat(),
        "params": params,
        "stats": stats,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)

    return out_path


# ---------------------------------------------------------------------------
# 6. Config loading with CLI override merging
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *overrides* into a copy of *base*.

    Scalar values in *overrides* replace those in *base*.  Nested dicts are
    merged recursively.
    """
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    config_path: Union[str, Path],
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a YAML config file and deep-merge CLI overrides.

    Parameters
    ----------
    config_path : str | Path
        Path to a YAML configuration file.
    cli_overrides : dict | None
        Key/value pairs from the command line that take precedence over
        the file-based config.

    Returns
    -------
    dict
        The merged configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    if cli_overrides:
        config = _deep_merge(config, cli_overrides)

    return config


# ---------------------------------------------------------------------------
# 7. Self-test smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Create a fake image tree
        sub = tmp / "input" / "subdir"
        sub.mkdir(parents=True)
        (tmp / "input" / "a.jpg").write_bytes(b"fake")
        (sub / "b.png").write_bytes(b"fake")
        (tmp / "input" / "not_an_image.txt").write_bytes(b"skip me")

        # Test discover_images
        images = discover_images(tmp / "input")
        assert len(images) == 2, f"Expected 2 images, got {len(images)}"
        assert all(isinstance(p, Path) for p in images)
        # Should be relative paths
        assert not any(p.is_absolute() for p in images)

        # Test mirror_directory
        out_path = mirror_directory(
            tmp / "input", tmp / "output", Path("subdir") / "b.png"
        )
        assert out_path == tmp / "output" / "subdir" / "b.png"
        assert out_path.parent.is_dir()

        # Test setup_logging
        logger = setup_logging(verbose=True, log_dir=tmp / "logs")
        assert logger.name == "msf"
        assert (tmp / "logs" / "msf.log").exists()

        # Test write_run_log
        log_path = write_run_log(
            tmp / "logs",
            params={"threshold": 0.5},
            stats={"images_processed": 42},
        )
        assert log_path.exists()
        with open(log_path) as f:
            data = json.load(f)
        assert "timestamp" in data
        assert data["params"]["threshold"] == 0.5

        # Test load_config
        cfg_file = tmp / "config.yaml"
        cfg_file.write_text("model:\n  threshold: 0.3\noutput: results\n")
        cfg = load_config(cfg_file, cli_overrides={"model": {"threshold": 0.9}})
        assert cfg["model"]["threshold"] == 0.9
        assert cfg["output"] == "results"

        # Clean up logging handlers so the temp directory can be removed on Windows
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    print("common.py self-test passed")
