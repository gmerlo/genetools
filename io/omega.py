"""
omega.py — parse GENE linear growth-rate output files.

These files are written by GENE for linear runs and are used here only as an
*optional cross-check* against the field-based growth rate computed in
:class:`~genetools.diagnostics.growthrate.GrowthRate`.

- ``omega<ext>``        one or more rows of ``ky  gamma  omega``
- ``eigenvalues.dat``   rows of ``gamma  omega`` (eigenvalue solver output)
"""

from pathlib import Path

import numpy as np


def _floats(line: str):
    """Return the whitespace-separated floats in *line*, or None if not numeric."""
    try:
        return [float(tok) for tok in line.split()]
    except ValueError:
        return None


def read_omega(folder, ext: str = "") -> dict | None:
    """
    Read a GENE ``omega<ext>`` file.

    Returns
    -------
    dict or None
        ``{"ky": ndarray, "gamma": ndarray, "omega": ndarray}`` or ``None`` when
        the file is absent or empty.
    """
    path = Path(folder) / f"omega{ext}"
    if not path.is_file():
        return None
    rows = []
    for line in path.read_text().splitlines():
        vals = _floats(line.strip())
        if vals and len(vals) >= 3:
            rows.append(vals[:3])
    if not rows:
        return None
    arr = np.asarray(rows, dtype=float)
    return {"ky": arr[:, 0], "gamma": arr[:, 1], "omega": arr[:, 2]}


def read_eigenvalues(folder, ext: str = "") -> dict | None:
    """
    Read a GENE ``eigenvalues<ext>`` / ``eigenvalues.dat`` file.

    Returns
    -------
    dict or None
        ``{"gamma": ndarray, "omega": ndarray}`` or ``None`` if absent/empty.
    """
    path = Path(folder) / f"eigenvalues{ext}"
    if not path.is_file():
        path = Path(folder) / "eigenvalues.dat"
    if not path.is_file():
        return None
    rows = []
    for line in path.read_text().splitlines():
        vals = _floats(line.strip())
        if vals and len(vals) >= 2:
            rows.append(vals[:2])
    if not rows:
        return None
    arr = np.asarray(rows, dtype=float)
    return {"gamma": arr[:, 0], "omega": arr[:, 1]}
