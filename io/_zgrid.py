"""
_zgrid.py — Shared z-grid construction for GENE simulations.

Used by both geometry.py (curvature computation) and coordinates.py.
"""

import numpy as np


def build_zgrid(nz: int, n_pol: int = 1, edge_opt: float = 0) -> np.ndarray:
    """
    Construct the z (parallel) coordinate grid.

    Parameters
    ----------
    nz : int
        Number of z grid points.
    n_pol : int, optional
        Number of poloidal turns (default 1).
    edge_opt : float, optional
        Edge clustering parameter. Zero gives uniform spacing;
        nonzero applies a sinh transformation (default 0).

    Returns
    -------
    np.ndarray
        z coordinate array of shape ``(nz,)``.
    """
    z = np.linspace(-np.pi * n_pol, np.pi * n_pol, nz + 1)[:-1]

    if edge_opt != 0:
        k = np.arange(nz)
        logterm = np.log(
            edge_opt * np.pi + np.sqrt((edge_opt * np.pi) ** 2 + 1)
        ) / np.pi
        z = np.sinh((-np.pi + k * 2 * np.pi / nz) * logterm) / edge_opt

    return z
