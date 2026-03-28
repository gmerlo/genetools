"""
profiles_loader.py — Equilibrium profile loader for global GENE simulations.

GENE writes per-species equilibrium profile files (``profiles_{species}_{ext}``)
containing radial profiles of temperature, density, and their logarithmic
gradients.  These files are plain text with two header lines followed by
columns of floats.

Public interface
----------------
``load_equilibrium_profiles(folder, ext, species_name)``
    Load equilibrium profiles for one species.

Example
-------
>>> from genetools.io.profiles_loader import load_equilibrium_profiles
>>> prof = load_equilibrium_profiles('/path/to/run/', '_0001', 'ions')
>>> prof['T']        # temperature profile, shape (nx,)
>>> prof['omt']      # R/L_T gradient, shape (nx,)
"""

import os
import numpy as np


def load_equilibrium_profiles(folder: str, ext: str, species_name: str) -> dict:
    """
    Load equilibrium profiles for one species from a GENE text file.

    Parameters
    ----------
    folder : str
        Run directory containing profile files.
    ext : str
        File-name suffix, e.g. ``'_0001'``.
    species_name : str
        Species name as written in the file name (e.g. ``'ions'``, ``'electrons'``).

    Returns
    -------
    dict
        Keys: ``x_o_rho_ref``, ``x_o_a``, ``T``, ``n``, ``omt``, ``omn``.
        All values are 1-D ``np.ndarray`` of shape ``(nx,)``.

    Raises
    ------
    FileNotFoundError
        If the profile file does not exist.
    """
    fname = os.path.join(folder, f"profiles_{species_name}{ext}")
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Equilibrium profile file not found: {fname}")

    data = np.loadtxt(fname, skiprows=2)

    return {
        "x_o_rho_ref": data[:, 0],
        "x_o_a": data[:, 1],
        "T": data[:, 2],
        "n": data[:, 3],
        "omt": data[:, 4],
        "omn": data[:, 5],
    }
