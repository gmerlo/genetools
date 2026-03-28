"""
profiles_loader.py — Equilibrium profile loader for global GENE simulations.

GENE writes per-species equilibrium profile files (``profiles_{species}_{ext}``)
containing radial profiles of temperature, density, and their logarithmic
gradients.  These files are plain text with two header lines followed by
columns of floats.

Public interface
----------------
``EquilibriumProfiles(folder, extensions, params)``
    Load equilibrium profiles for all segments and species.
    Returns a list-like object; one entry per segment.

``load_equilibrium_profiles(folder, ext, species_name)``
    Load equilibrium profiles for one species (backward-compatible function).

Example
-------
>>> from genetools.io.profiles_loader import EquilibriumProfiles
>>> profs = EquilibriumProfiles(folder, runs, params)
>>> profs[0]['ions']['T']   # segment 0, ions temperature
>>> profs.plot()            # plots segment 0 by default
>>> profs.plot(segment=1)   # plot a different segment
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


class _SegmentProfiles:
    """Profiles for all species in a single run segment."""

    def __init__(self, folder: str, ext: str, species_names: list):
        self.ext = ext
        self.species_names = species_names
        self._data = {}
        for name in species_names:
            self._data[name] = _load_single(folder, ext, name)

    def __getitem__(self, species_name: str) -> dict:
        return self._data[species_name]

    def __contains__(self, species_name: str) -> bool:
        return species_name in self._data

    def __repr__(self) -> str:
        return f"_SegmentProfiles({self.species_names}, ext='{self.ext}')"

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()


class EquilibriumProfiles:
    """
    Load and plot equilibrium profiles for all segments and species.

    Follows the same convention as ``Geometry()`` and ``Coordinates()``:
    takes ``(folder, extensions, params)`` and returns a list-like object
    with one entry per segment.

    Parameters
    ----------
    folder : str
        Run directory containing ``profiles_{species}{ext}`` files.
    extensions : str or list of str
        File-name suffix(es), e.g. ``'_0001'`` or ``['_0001', '_0002']``.
    params : Params
        Parameter object (as returned by :class:`~genetools.io.params.Params`).

    Examples
    --------
    >>> profs = EquilibriumProfiles(folder, runs, params)
    >>> profs[0]['ions']['T']      # segment 0, ions temperature
    >>> profs[0]['ions']['omt']    # segment 0, ions R/L_T
    >>> profs.plot()               # plot segment 0
    >>> profs.plot(segment=1)      # plot segment 1
    """

    def __init__(self, folder: str, extensions, params):
        if isinstance(extensions, str):
            extensions = [extensions]

        self.folder = folder
        self.extensions = extensions
        self._segments = []
        for fn, ext in enumerate(extensions):
            p = params.get(fn)
            species_names = [sp["name"] for sp in p["species"]]
            self._segments.append(
                _SegmentProfiles(folder, ext, species_names))

    def __getitem__(self, index: int) -> _SegmentProfiles:
        """Access profiles for a segment by index."""
        return self._segments[index]

    def __len__(self) -> int:
        return len(self._segments)

    def __repr__(self) -> str:
        return (f"EquilibriumProfiles({len(self._segments)} segments, "
                f"extensions={self.extensions})")

    def plot(self, segment: int = 0,
             x_key: str = "x_o_a", x_label: str = "x/a") -> None:
        """
        Plot temperature, density, and their gradients for all species.

        Parameters
        ----------
        segment : int, optional
            Segment index to plot (default 0).
        x_key : str, optional
            Key to use for the x-axis (default ``'x_o_a'``).
            Use ``'x_o_rho_ref'`` for rho_ref normalisation.
        x_label : str, optional
            Label for the x-axis (default ``'x/a'``).
        """
        seg = self._segments[segment]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

        quantities = [
            ("T", r"$T$"),
            ("n", r"$n$"),
            ("omt", r"$R/L_T$"),
            ("omn", r"$R/L_n$"),
        ]

        for ax, (key, ylabel) in zip(axes.flat, quantities):
            for name in seg.species_names:
                prof = seg[name]
                ax.plot(prof[x_key], prof[key], label=name)
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)

        axes[1, 0].set_xlabel(x_label)
        axes[1, 1].set_xlabel(x_label)
        fig.suptitle(f"Equilibrium profiles (segment {segment})")
        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------
# Backward-compatible function interface
# -----------------------------------------------------------------------

def load_equilibrium_profiles(folder: str, ext: str, species_name: str) -> dict:
    """
    Load equilibrium profiles for one species from a GENE text file.

    This is a convenience wrapper around :func:`_load_single` for
    loading a single species. For multiple species with plotting, use
    :class:`EquilibriumProfiles`.

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
