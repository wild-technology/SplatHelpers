"""
points3d_io.py - Read/write COLMAP points3D files in text and binary formats.

Supports both points3D.txt and points3D.bin produced by COLMAP's SfM pipeline.
Provides dict-based and array-based interfaces; the array interface is optimised
for batch filtering of large point clouds (tested at ~4.9 M points).

Text format (points3D.txt)
--------------------------
Header lines start with ``#``.  Each data line::

    POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX) pairs

Example::

    1 -2.13 -14.79 6.73 84 112 119 0.5 6300 0 6308 0

Binary format (points3D.bin)
----------------------------
* ``num_points`` : uint64
* Per point:
    - point3D_id : uint64
    - xyz        : 3 x float64
    - rgb        : 3 x uint8
    - error      : float64
    - track_length : uint64
    - track_length pairs of (image_id : uint32, point2d_idx : uint32)
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_format(path: Path) -> str:
    """Auto-detect COLMAP points3D format from file extension.

    Falls back to inspecting the first bytes when the extension is
    ambiguous.

    Parameters
    ----------
    path : Path
        Path to the points3D file.

    Returns
    -------
    str
        ``'txt'`` or ``'bin'``.
    """
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return "txt"
    if suffix == ".bin":
        return "bin"
    # Heuristic: text files start with '#' or a digit
    with open(path, "rb") as fh:
        first = fh.read(1)
    if first in (b"#", b"0", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8", b"9"):
        return "txt"
    return "bin"


# ---------------------------------------------------------------------------
# 1. Dict-based readers / writers
# ---------------------------------------------------------------------------

def read_points3d(path: Union[str, Path]) -> Dict[int, Dict[str, Any]]:
    """Read a COLMAP points3D file into a dict.

    Auto-detects text vs binary format from the file extension (or first
    bytes when the extension is ambiguous).

    Parameters
    ----------
    path : str | Path
        Path to ``points3D.txt`` or ``points3D.bin``.

    Returns
    -------
    dict[int, dict]
        Mapping of ``point3d_id`` to a dict with keys:

        - **xyz** (*np.ndarray float64[3]*) -- 3-D position.
        - **rgb** (*np.ndarray uint8[3]*) -- colour.
        - **error** (*float*) -- reprojection error.
        - **track** (*list[tuple[int, int]]*) -- list of
          ``(image_id, point2d_idx)`` pairs.
    """
    path = Path(path)
    fmt = _detect_format(path)
    if fmt == "txt":
        return _read_points3d_text(path)
    return _read_points3d_binary(path)


def _read_points3d_text(path: Path) -> Dict[int, Dict[str, Any]]:
    """Read points3D.txt."""
    points: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point3d_id = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            rgb = np.array([int(parts[4]), int(parts[5]), int(parts[6])], dtype=np.uint8)
            error = float(parts[7])
            # Remaining tokens are track pairs
            track: List[Tuple[int, int]] = []
            track_tokens = parts[8:]
            for i in range(0, len(track_tokens), 2):
                track.append((int(track_tokens[i]), int(track_tokens[i + 1])))
            points[point3d_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track,
            }
    return points


def _read_points3d_binary(path: Path) -> Dict[int, Dict[str, Any]]:
    """Read points3D.bin."""
    points: Dict[int, Dict[str, Any]] = {}
    with open(path, "rb") as fh:
        num_points = struct.unpack("<Q", fh.read(8))[0]
        for _ in range(num_points):
            point3d_id = struct.unpack("<Q", fh.read(8))[0]
            xyz = np.frombuffer(fh.read(24), dtype="<f8").copy()
            rgb = np.frombuffer(fh.read(3), dtype=np.uint8).copy()
            error = struct.unpack("<d", fh.read(8))[0]
            track_length = struct.unpack("<Q", fh.read(8))[0]
            track: List[Tuple[int, int]] = []
            for _ in range(track_length):
                img_id, pt2d_idx = struct.unpack("<II", fh.read(8))
                track.append((img_id, pt2d_idx))
            points[point3d_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track,
            }
    return points


def write_points3d(
    path: Union[str, Path],
    points: Dict[int, Dict[str, Any]],
    fmt: str = "auto",
) -> None:
    """Write a COLMAP points3D file from a dict.

    Parameters
    ----------
    path : str | Path
        Output file path.
    points : dict[int, dict]
        Same structure returned by :func:`read_points3d`.
    fmt : str
        ``'txt'``, ``'bin'``, or ``'auto'`` (infer from *path* extension).
    """
    path = Path(path)
    if fmt == "auto":
        fmt = "txt" if path.suffix.lower() == ".txt" else "bin"

    if fmt == "txt":
        _write_points3d_text(path, points)
    else:
        _write_points3d_binary(path, points)


def _write_points3d_text(path: Path, points: Dict[int, Dict[str, Any]]) -> None:
    """Write points3D.txt."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# 3D point list with one line of data per point:\n")
        fh.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
                 "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        fh.write(f"# Number of points: {len(points)}\n")
        for pid, p in points.items():
            xyz = p["xyz"]
            rgb = p["rgb"]
            track_str = " ".join(f"{img_id} {pt_idx}" for img_id, pt_idx in p["track"])
            line = (
                f"{pid} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                f"{rgb[0]} {rgb[1]} {rgb[2]} {p['error']:.6f}"
            )
            if track_str:
                line += " " + track_str
            fh.write(line + "\n")


def _write_points3d_binary(path: Path, points: Dict[int, Dict[str, Any]]) -> None:
    """Write points3D.bin."""
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(points)))
        for pid, p in points.items():
            fh.write(struct.pack("<Q", pid))
            fh.write(p["xyz"].astype("<f8").tobytes())
            fh.write(p["rgb"].astype(np.uint8).tobytes())
            fh.write(struct.pack("<d", p["error"]))
            track = p["track"]
            fh.write(struct.pack("<Q", len(track)))
            for img_id, pt_idx in track:
                fh.write(struct.pack("<II", img_id, pt_idx))


# ---------------------------------------------------------------------------
# 2. Array-based readers / writers (optimised for batch filtering)
# ---------------------------------------------------------------------------

def read_points3d_arrays(
    path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Optimised batch reader returning flat NumPy arrays.

    This avoids per-point Python dicts and is much faster for large clouds
    (e.g. 4.9 M points).

    Parameters
    ----------
    path : str | Path
        Path to ``points3D.txt`` or ``points3D.bin``.

    Returns
    -------
    tuple of np.ndarray
        ``(ids, xyz, rgb, errors, track_lengths, tracks_flat)``

        - **ids** -- ``uint64[N]`` point3D IDs.
        - **xyz** -- ``float64[N, 3]`` positions.
        - **rgb** -- ``uint8[N, 3]`` colours.
        - **errors** -- ``float64[N]`` reprojection errors.
        - **track_lengths** -- ``uint64[N]`` number of track entries per point.
        - **tracks_flat** -- ``uint32[T, 2]`` concatenated ``(image_id,
          point2d_idx)`` pairs.  The entries for point *i* start at offset
          ``sum(track_lengths[:i])`` and span ``track_lengths[i]`` rows.
    """
    path = Path(path)
    fmt = _detect_format(path)
    if fmt == "txt":
        return _read_points3d_arrays_text(path)
    return _read_points3d_arrays_binary(path)


def _read_points3d_arrays_text(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Array reader for text format."""
    ids_list: List[int] = []
    xyz_list: List[Tuple[float, float, float]] = []
    rgb_list: List[Tuple[int, int, int]] = []
    errors_list: List[float] = []
    track_lengths_list: List[int] = []
    tracks_flat_list: List[Tuple[int, int]] = []

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ids_list.append(int(parts[0]))
            xyz_list.append((float(parts[1]), float(parts[2]), float(parts[3])))
            rgb_list.append((int(parts[4]), int(parts[5]), int(parts[6])))
            errors_list.append(float(parts[7]))
            track_tokens = parts[8:]
            n_track = len(track_tokens) // 2
            track_lengths_list.append(n_track)
            for i in range(0, len(track_tokens), 2):
                tracks_flat_list.append((int(track_tokens[i]), int(track_tokens[i + 1])))

    ids = np.array(ids_list, dtype=np.uint64)
    xyz = np.array(xyz_list, dtype=np.float64)
    if xyz.size == 0:
        xyz = xyz.reshape(0, 3)
    rgb = np.array(rgb_list, dtype=np.uint8)
    if rgb.size == 0:
        rgb = rgb.reshape(0, 3)
    errors = np.array(errors_list, dtype=np.float64)
    track_lengths = np.array(track_lengths_list, dtype=np.uint64)
    if tracks_flat_list:
        tracks_flat = np.array(tracks_flat_list, dtype=np.uint32)
    else:
        tracks_flat = np.empty((0, 2), dtype=np.uint32)

    return ids, xyz, rgb, errors, track_lengths, tracks_flat


def _read_points3d_arrays_binary(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Array reader for binary format -- reads large chunks to minimise
    Python-level loops."""
    with open(path, "rb") as fh:
        data = fh.read()

    offset = 0
    num_points = struct.unpack_from("<Q", data, offset)[0]
    offset += 8

    ids = np.empty(num_points, dtype=np.uint64)
    xyz = np.empty((num_points, 3), dtype=np.float64)
    rgb = np.empty((num_points, 3), dtype=np.uint8)
    errors = np.empty(num_points, dtype=np.float64)
    track_lengths = np.empty(num_points, dtype=np.uint64)
    tracks_accum: List[np.ndarray] = []

    for i in range(num_points):
        ids[i] = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        xyz[i] = struct.unpack_from("<3d", data, offset)
        offset += 24
        rgb[i] = struct.unpack_from("<3B", data, offset)
        offset += 3
        errors[i] = struct.unpack_from("<d", data, offset)[0]
        offset += 8
        tl = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        track_lengths[i] = tl
        if tl > 0:
            track_bytes = tl * 8  # each pair is 2 x uint32 = 8 bytes
            track_arr = np.frombuffer(data, dtype="<u4", count=tl * 2, offset=offset).reshape(-1, 2).copy()
            tracks_accum.append(track_arr)
            offset += track_bytes

    if tracks_accum:
        tracks_flat = np.concatenate(tracks_accum, axis=0)
    else:
        tracks_flat = np.empty((0, 2), dtype=np.uint32)

    return ids, xyz, rgb, errors, track_lengths, tracks_flat


def write_points3d_from_arrays(
    path: Union[str, Path],
    ids: np.ndarray,
    xyz: np.ndarray,
    rgb: np.ndarray,
    errors: np.ndarray,
    track_lengths: np.ndarray,
    tracks_flat: np.ndarray,
    mask: np.ndarray,
    fmt: str = "auto",
) -> None:
    """Write a subset of points (selected by *mask*) from array data.

    Parameters
    ----------
    path : str | Path
        Output file path.
    ids : np.ndarray
        ``uint64[N]`` point IDs.
    xyz : np.ndarray
        ``float64[N, 3]`` positions.
    rgb : np.ndarray
        ``uint8[N, 3]`` colours.
    errors : np.ndarray
        ``float64[N]`` reprojection errors.
    track_lengths : np.ndarray
        ``uint64[N]`` per-point track lengths.
    tracks_flat : np.ndarray
        ``uint32[T, 2]`` concatenated track pairs.
    mask : np.ndarray
        ``bool[N]`` -- only points where ``mask[i]`` is *True* are written.
    fmt : str
        ``'txt'``, ``'bin'``, or ``'auto'`` (infer from extension).
    """
    path = Path(path)
    if fmt == "auto":
        fmt = "txt" if path.suffix.lower() == ".txt" else "bin"

    # Compute track offsets from track_lengths
    offsets = np.zeros(len(track_lengths) + 1, dtype=np.int64)
    np.cumsum(track_lengths.astype(np.int64), out=offsets[1:])

    if fmt == "txt":
        _write_arrays_text(path, ids, xyz, rgb, errors, track_lengths, tracks_flat, offsets, mask)
    else:
        _write_arrays_binary(path, ids, xyz, rgb, errors, track_lengths, tracks_flat, offsets, mask)


def _write_arrays_text(
    path: Path,
    ids: np.ndarray,
    xyz: np.ndarray,
    rgb: np.ndarray,
    errors: np.ndarray,
    track_lengths: np.ndarray,
    tracks_flat: np.ndarray,
    offsets: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Write filtered points as text."""
    kept_count = int(mask.sum())
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# 3D point list with one line of data per point:\n")
        fh.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
                 "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        fh.write(f"# Number of points: {kept_count}\n")

        indices = np.nonzero(mask)[0]
        for i in indices:
            p_id = int(ids[i])
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            err = errors[i]
            tl = int(track_lengths[i])
            start = int(offsets[i])
            end = start + tl

            parts = [f"{p_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {err:.6f}"]
            if tl > 0:
                track_pairs = tracks_flat[start:end]
                track_str = " ".join(f"{row[0]} {row[1]}" for row in track_pairs)
                parts.append(track_str)
            fh.write(" ".join(parts) + "\n")


def _write_arrays_binary(
    path: Path,
    ids: np.ndarray,
    xyz: np.ndarray,
    rgb: np.ndarray,
    errors: np.ndarray,
    track_lengths: np.ndarray,
    tracks_flat: np.ndarray,
    offsets: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Write filtered points as binary."""
    indices = np.nonzero(mask)[0]
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(indices)))
        for i in indices:
            fh.write(struct.pack("<Q", int(ids[i])))
            fh.write(xyz[i].astype("<f8").tobytes())
            fh.write(rgb[i].astype(np.uint8).tobytes())
            fh.write(struct.pack("<d", float(errors[i])))
            tl = int(track_lengths[i])
            fh.write(struct.pack("<Q", tl))
            if tl > 0:
                start = int(offsets[i])
                end = start + tl
                fh.write(tracks_flat[start:end].astype("<u4").tobytes())
