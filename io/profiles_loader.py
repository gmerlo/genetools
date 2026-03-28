"""
profiles_loader.py — Equilibrium profile loader for global GENE simulations.

GENE writes per-species equilibrium profile files (``profiles_{species}_{ext}``)
containing radial profiles of temperature, density, and their logarithmic
gradients.  These files are plain text with two header lines followed by
columns of floats.

Public interface
----------------
``EquilibriumProfiles(folder, ext, species_names)``
    Load equilibrium profiles for one or more species.

``load_equilibrium_profiles(folder, ext, species_name)``
    Load equilibrium profiles for one species (backward-compatible function).

Example
-------
>>> from genetools.io.profiles_loader import EquilibriumProfiles
>>> profs = EquilibriumProfiles('/path/to/run/', '_0001', ['ions', 'electrons'])
>>> profs['ions']['T']    # temperature profile, shape (nx,)
>>> profs.plot()          # plot all species
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def _load_single(folder: str, ext: str, species_name: str) -> dict:
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


class EquilibriumProfiles:
    """
    Load and plot equilibrium profiles for one or more species.

    Parameters
    ----------
    folder : str
        Run directory containing ``profiles_{species}{ext}`` files.
    ext : str
        File-name suffix, e.g. ``'_0001'``.
    species_names : list of str
        Species names to load, e.g. ``['ions', 'electrons']``.

    Examples
    --------
    >>> profs = EquilibriumProfiles('/path/to/run/', '_0001', ['ions', 'electrons'])
    >>> profs['ions']['T']      # temperature array
    >>> profs['ions']['omt']    # R/L_T gradient
    >>> profs.plot()            # 2x2 figure: T, n, R/L_T, R/L_n
    """

    def __init__(self, folder: str, ext: str, species_names: list):
        self.folder = folder
        self.ext = ext
        self.species_names = list(species_names)
        self._data = {}
        for name in self.species_names:
            self._data[name] = _load_single(folder, ext, name)

    def __getitem__(self, species_name: str) -> dict:
        """Access profiles for a species by name."""
        return self._data[species_name]

    def __contains__(self, species_name: str) -> bool:
        return species_name in self._data

    def __repr__(self) -> str:
        return (f"EquilibriumProfiles({self.species_names}, "
                f"ext='{self.ext}')")

    def keys(self):
        """Return species names."""
        return self._data.keys()

    def items(self):
        """Return (species_name, profile_dict) pairs."""
        return self._data.items()

    def plot(self, x_key: str = "x_o_a", x_label: str = "x/a") -> None:
        """
        Plot temperature, density, and their gradients for all species.

        Parameters
        ----------
        x_key : str, optional
            Key to use for the x-axis (default ``'x_o_a'``).
            Use ``'x_o_rho_ref'`` for rho_ref normalisation.
        x_label : str, optional
            Label for the x-axis (default ``'x/a'``).
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

        quantities = [
            ("T", r"$T$"),
            ("n", r"$n$"),
            ("omt", r"$R/L_T$"),
            ("omn", r"$R/L_n$"),
        ]

        for ax, (key, ylabel) in zip(axes.flat, quantities):
            for name in self.species_names:
                prof = self._data[name]
                ax.plot(prof[x_key], prof[key], label=name)
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)

        axes[1, 0].set_xlabel(x_label)
        axes[1, 1].set_xlabel(x_label)
        fig.suptitle("Equilibrium profiles")
        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------
# Backward-compatible function interface
# -----------------------------------------------------------------------

def load_equilibrium_profiles(folder: str, ext: str, species_name: str) -> dict:
    """
    Load equilibrium profiles for one species from a GENE text file.

    This is a convenience wrapper around :class:`EquilibriumProfiles` for
    loading a single species. For multiple species with plotting, use
    the class directly.

    Parameters
    ----------
    folder : str
        Run directory containing profile files.
    ext : str
        File-name suffix, e.g. ``'_0001'``.
    species_name : str
        Species name as written in the file name.

    Returns
    -------
    dict
        Keys: ``x_o_rho_ref``, ``x_o_a``, ``T``, ``n``, ``omt``, ``omn``.
    """
    return _load_single(folder, ext, species_name)
