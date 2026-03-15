"""
ply_io.py - Read and write 3D Gaussian Splatting .ply files.

Uses the ``plyfile`` library for parsing and writing binary PLY data.
Provides helper functions for converting between raw stored values and
the actual physical quantities (opacity via sigmoid, scale via exp,
colour via spherical-harmonic DC coefficient).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from plyfile import PlyData, PlyElement

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Zeroth spherical-harmonic coefficient  (Y_0^0 = 1 / (2*sqrt(pi)))
C0: float = 0.28209479177387814


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid:  1 / (1 + exp(-x)).

    Parameters
    ----------
    x : np.ndarray
        Raw (logit-space) values.

    Returns
    -------
    np.ndarray
        Values in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def inv_sigmoid(y: np.ndarray) -> np.ndarray:
    """Inverse sigmoid (logit):  log(y / (1 - y)).

    Parameters
    ----------
    y : np.ndarray
        Probability-space values in (0, 1).

    Returns
    -------
    np.ndarray
        Logit-space values.
    """
    y = np.clip(y, 1e-7, 1.0 - 1e-7)
    return np.log(y / (1.0 - y))


def sh_dc_to_rgb(
    f_dc_0: np.ndarray,
    f_dc_1: np.ndarray,
    f_dc_2: np.ndarray,
) -> np.ndarray:
    """Convert zeroth-order SH DC coefficients to linear RGB.

    Parameters
    ----------
    f_dc_0, f_dc_1, f_dc_2 : np.ndarray
        Per-gaussian DC coefficients for R, G, B (shape ``(N,)``).

    Returns
    -------
    np.ndarray
        RGB array of shape ``(N, 3)`` clipped to [0, 1].
    """
    rgb = np.column_stack([f_dc_0, f_dc_1, f_dc_2]).astype(np.float64)
    rgb = 0.5 + C0 * rgb
    return np.clip(rgb, 0.0, 1.0)


def actual_scale(raw_scale: np.ndarray) -> np.ndarray:
    """Convert log-space scale to actual scale via exp.

    Parameters
    ----------
    raw_scale : np.ndarray
        Raw (log-space) scale values.

    Returns
    -------
    np.ndarray
        Positive scale values.
    """
    return np.exp(raw_scale.astype(np.float64))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_splat(path: Path) -> Tuple[PlyData, np.ndarray]:
    """Read a 3DGS ``.ply`` file.

    Parameters
    ----------
    path : Path
        Path to a binary-little-endian PLY file produced by a Gaussian
        Splatting trainer.

    Returns
    -------
    tuple[PlyData, np.ndarray]
        ``(plydata, vertex_data)`` where *plydata* is the full
        :class:`PlyData` object (kept for round-trip fidelity) and
        *vertex_data* is a NumPy structured array of the ``'vertex'``
        element.
    """
    path = Path(path)
    plydata = PlyData.read(str(path))
    vertex_data = plydata["vertex"].data
    return plydata, vertex_data


def write_splat(path: Path, plydata: PlyData, mask: np.ndarray) -> None:
    """Write a filtered 3DGS ``.ply`` file.

    Applies *mask* to the vertex data and writes a new PLY that preserves
    every property name, dtype, ordering, and the original binary format
    so that downstream viewers (e.g. SuperSplat, Luma) work correctly.

    Parameters
    ----------
    path : Path
        Destination file path.
    plydata : PlyData
        The original :class:`PlyData` (from :func:`read_splat`).
    mask : np.ndarray
        Boolean array of shape ``(N,)`` selecting vertices to keep.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertex = plydata["vertex"]
    filtered = vertex.data[mask]

    new_element = PlyElement.describe(filtered, "vertex")

    # Preserve text_format from the original file
    PlyData(
        [new_element],
        text=plydata.text,
        byte_order=plydata.byte_order,
    ).write(str(path))


# ---------------------------------------------------------------------------
# Property accessors
# ---------------------------------------------------------------------------

def get_positions(data: np.ndarray) -> np.ndarray:
    """Extract gaussian centre positions.

    Parameters
    ----------
    data : np.ndarray
        Structured vertex array from :func:`read_splat`.

    Returns
    -------
    np.ndarray
        Float array of shape ``(N, 3)`` with columns ``[x, y, z]``.
    """
    return np.column_stack([
        data["x"].astype(np.float64),
        data["y"].astype(np.float64),
        data["z"].astype(np.float64),
    ])


def get_opacity(data: np.ndarray) -> np.ndarray:
    """Return per-gaussian opacity in [0, 1].

    Parameters
    ----------
    data : np.ndarray
        Structured vertex array.

    Returns
    -------
    np.ndarray
        Float array of shape ``(N,)``.
    """
    return sigmoid(data["opacity"].astype(np.float64))


def get_scales(data: np.ndarray) -> np.ndarray:
    """Return actual (exponentiated) scales per gaussian.

    Parameters
    ----------
    data : np.ndarray
        Structured vertex array.

    Returns
    -------
    np.ndarray
        Float array of shape ``(N, 3)`` with columns
        ``[scale_0, scale_1, scale_2]``.
    """
    return np.column_stack([
        actual_scale(data["scale_0"].astype(np.float64)),
        actual_scale(data["scale_1"].astype(np.float64)),
        actual_scale(data["scale_2"].astype(np.float64)),
    ])


def get_rgb(data: np.ndarray) -> np.ndarray:
    """Return per-gaussian RGB colour derived from the SH DC band.

    Parameters
    ----------
    data : np.ndarray
        Structured vertex array.

    Returns
    -------
    np.ndarray
        Float array of shape ``(N, 3)`` with values in [0, 1].
    """
    return sh_dc_to_rgb(
        data["f_dc_0"].astype(np.float64),
        data["f_dc_1"].astype(np.float64),
        data["f_dc_2"].astype(np.float64),
    )
