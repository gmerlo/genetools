# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
profiles_loader.py — Equilibrium profile loader for global GENE simulations.

GENE writes per-species equilibrium profile files (``profiles_{species}{ext}``)
containing radial profiles of temperature, density, and their logarithmic
gradients.  These are plain text — a header line, a ``#<time>`` line, then
columns of floats in the order ``x/a, x/rho_ref, T, n, omt, omn`` — and, when
the run was built with HDF5 support, an equivalent ``.h5`` twin. GENE-3D writes
both; the HDF5 form is read when the text file is absent.

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

import h5py
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
    if os.path.isfile(fname):
        return _load_ascii(fname)

    # GENE-3D and GENE with write_h5 also write an HDF5 twin. GENE-3D is the
    # case that needs it: its ASCII file is rewritten (and appended to) as the
    # background profiles evolve, while the HDF5 form is a clean snapshot.
    h5name = fname + ".h5"
    if os.path.isfile(h5name):
        return _load_h5(h5name)

    raise FileNotFoundError(
        f"Equilibrium profile file not found: {fname} (nor {h5name})")


def _load_ascii(fname: str) -> dict:
    """
    Parse a ``profiles_<species>`` text file.

    The column order is ``x/a`` then ``x/rho_ref`` — that is what both
    ``gene/src/profiles.F90`` and ``gene3d-dev/src/profiles.F90`` write, and
    what the file's own header line says. Getting these two the wrong way round
    silently rescales the radial axis by ``rhostar``.

    GENE-3D appends a fresh block (separated by two blank lines) every time the
    background profiles are updated, so only the last block is returned.
    """
    with open(fname) as fh:
        lines = fh.readlines()

    blocks, current = [], []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append(current)
                current = []
            continue
        if stripped.startswith("#"):
            continue
        current.append(stripped)
    if current:
        blocks.append(current)

    if not blocks:
        raise ValueError(f"No profile data rows found in {fname}")

    data = np.array([[float(v) for v in row.split()] for row in blocks[-1]])
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < 6:
        raise ValueError(
            f"{fname} has {data.shape[1]} columns; expected at least 6 "
            "(x/a, x/rho_ref, T, n, omt, omn)")

    out = {
        "x_o_a": data[:, 0],
        "x_o_rho_ref": data[:, 1],
        "T": data[:, 2],
        "n": data[:, 3],
        "omt": data[:, 4],
        "omn": data[:, 5],
    }
    if data.shape[1] >= 7:
        # 'te' when tau is computed, 'tau' when it was read in; the header
        # says which, and either way it is the seventh column.
        out["te_or_tau"] = data[:, 6]
    return out


def _load_h5(fname: str) -> dict:
    """
    Read a ``profiles_<species>.h5`` file.

    Written by ``write_spec_profiles`` in both codes:
    ``/position/{x_o_a,x_o_rho_ref}``, ``/temp/{T,omt}``,
    ``/density/{n,omn}``, optionally ``/temp/{te,tau}`` and ``/Erad``.
    """
    def get(f, path):
        dset = f.get(path)
        return None if dset is None else np.asarray(dset[...], dtype=float)

    with h5py.File(fname, "r") as f:
        out = {
            "x_o_a": get(f, "position/x_o_a"),
            "x_o_rho_ref": get(f, "position/x_o_rho_ref"),
            "T": get(f, "temp/T"),
            "n": get(f, "density/n"),
            "omt": get(f, "temp/omt"),
            "omn": get(f, "density/omn"),
        }
        for name, key in (("temp/te", "te"), ("temp/tau", "tau"),
                          ("Erad", "Erad")):
            val = get(f, name)
            if val is not None:
                out[key] = val

    missing = [k for k, v in out.items() if v is None]
    if missing:
        raise ValueError(
            f"{fname} is missing profile dataset(s): {', '.join(missing)}")
    return out


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
