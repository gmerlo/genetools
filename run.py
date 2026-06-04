# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
run.py — the :class:`Run` facade.

A single object that wires up everything needed to post-process a GENE run
(segment discovery, parameters, geometry, coordinates, and data readers) and
exposes every diagnostic as a lazy, tab-completable attribute that returns
``xarray.Dataset`` objects and plots.

Example
-------
>>> from genetools import Run
>>> run = Run("/path/to/run")        # discovers segments, params, geometry
>>> run.nrg.plot()                   # energy/flux time traces
>>> run.spectra.plot(t=(500, 2000))  # flux spectra over a time window
>>> ds = run.spectra.data            # underlying xarray.Dataset

Continuation / restart runs are handled transparently: all segments
(``_0001``, ``_0002``, …, ``.dat``) are discovered and stitched with
:class:`~genetools.io.MultiSegmentReader` (overlap dedup, later segment wins).
The grid is assumed consistent across segments; a mismatch raises a warning
telling you to scope to a subset via ``Run(path, ext=[...])``.

Both Fortran-binary and ADIOS2 BP outputs are supported: each segment's reader
is chosen per file (binary ``field_0001`` vs BP ``field_0001.bp``), so BP runs
work without any extra arguments (the ``adios2`` package must be installed).
"""

from __future__ import annotations

import os
import warnings
from functools import cached_property
from pathlib import Path

import numpy as np

from .io import (
    Params,
    set_runs,
    Geometry,
    Coordinates,
    BinaryReader,
    BPReader,
    MultiSegmentReader,
)
from .diagnostics import (
    NrgReader,
    Spectra,
    SpectraGlobal,
    Profiles,
    Fluxes2D,
    ShearingRate,
    Contours,
)


# ---------------------------------------------------------------------------
# Time-window helpers
# ---------------------------------------------------------------------------

def _window(t):
    """Normalise *t* to ``(start, stop)`` where either bound may be ``None``."""
    if t is None:
        return None, None
    if isinstance(t, (tuple, list)):
        a, b = t
        return (None if a is None else float(a),
                None if b is None else float(b))
    return float(t), None


def _bounds(t):
    """Like :func:`_window` but with concrete float bounds for streaming."""
    a, b = _window(t)
    return (-1e30 if a is None else a, 1e30 if b is None else b)


# ---------------------------------------------------------------------------
# Run facade
# ---------------------------------------------------------------------------

class Run:
    """
    High-level handle to a single GENE run directory.

    Parameters
    ----------
    path : str or path-like
        The GENE run directory.
    ext : str or list of str, optional
        Segment suffix(es) to use (e.g. ``"_0002"`` or ``["_0002", ".dat"]``).
        ``None`` (default) auto-discovers all segments.
    """

    def __init__(self, path, ext=None):
        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"Run directory not found: {self.path}")
        # Readers build filenames as f"{folder}{type}{ext}" — needs a trailing /.
        self._folder = str(self.path).rstrip("/") + "/"

        if ext is None:
            self.extensions = set_runs(self.path)
        elif isinstance(ext, str):
            self.extensions = [ext]
        else:
            self.extensions = list(ext)

        self.params = Params(self.path, self.extensions)
        self._validate_grid()

        self._field = None
        self._field_segments = None
        self._mom = {}

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def is_local(self) -> bool:
        """Whether the run uses local (spectral-x) geometry (``x_local``)."""
        return bool(self.params.get(0)["general"].get("x_local", True))

    @property
    def species(self) -> list:
        """List of species names."""
        return [s["name"] for s in self.params.get(0)["species"]]

    @cached_property
    def geometry(self) -> list:
        """Per-segment geometry dictionaries (from :func:`Geometry`)."""
        return Geometry(self.path, self.extensions, self.params)

    @cached_property
    def coords(self) -> list:
        """Per-segment coordinate dictionaries (from :func:`Coordinates`)."""
        return Coordinates(self.path, self.extensions, self.params)

    @property
    def times(self) -> np.ndarray:
        """Merged, deduplicated time axis across all segments (field times)."""
        try:
            return self.field.read_all_times()
        except FileNotFoundError:
            return self.nrg.read_all()[0]

    # ------------------------------------------------------------------
    # Continuation / restart validation
    # ------------------------------------------------------------------

    def _validate_grid(self) -> None:
        """Warn if grid/geometry dimensions differ across segments."""
        plist = self.params.tolist()
        if len(plist) <= 1:
            return
        keys = ("nx0", "nky0", "nz0", "n_spec")
        ref = {k: plist[0]["box"].get(k) for k in keys}
        ref_local = plist[0]["general"].get("x_local", True)
        bad = []
        for i, p in enumerate(plist[1:], start=1):
            this = {k: p["box"].get(k) for k in keys}
            if this != ref or p["general"].get("x_local", True) != ref_local:
                bad.append((self.extensions[i], this))
        if bad:
            details = "; ".join(f"{ext}: {dims}" for ext, dims in bad)
            warnings.warn(
                "Run segments have inconsistent grids "
                f"(reference {self.extensions[0]}: {ref}; differing — {details}). "
                "genetools assumes a consistent grid across a continuation run; "
                "scope to a consistent subset with Run(path, ext=[...]).",
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Readers (lazy, multi-segment aware)
    # ------------------------------------------------------------------

    def _make_reader(self, file_type, ext, params_i, species=None):
        """
        Build one segment reader, auto-selecting Fortran-binary vs ADIOS2 BP.

        The file present on disk decides: binary (``field_0001``) is preferred
        when both exist (it needs no adios2); otherwise the BP form
        (``field_0001.bp`` / ``field.bp``) is used, or selected from the
        ``write_bp`` parameter when neither file is present yet. Constructing a
        :class:`BPReader` without the ``adios2`` package raises a clear
        ``ImportError``.
        """
        sp = f"_{species}" if species else ""
        binfile = f"{self._folder}{file_type}{sp}{ext}"
        if os.path.exists(binfile):
            return BinaryReader(file_type, self._folder, ext, params_i,
                                species=species)
        bp_ext = ("" if ext == ".dat" else ext) + ".bp"
        bpfile = f"{self._folder}{file_type}{sp}{bp_ext}"
        if os.path.exists(bpfile) or params_i.get("in_out", {}).get("write_bp", False):
            return BPReader(file_type, self._folder, bp_ext, params_i,
                            species=species)
        # Neither present: default to binary (errors clearly on first read).
        return BinaryReader(file_type, self._folder, ext, params_i,
                            species=species)

    @property
    def _field_segment_readers(self) -> list:
        """One field reader per segment (used by ShearingRate)."""
        if self._field_segments is None:
            self._field_segments = [
                self._make_reader("field", ext, self.params.get(i))
                for i, ext in enumerate(self.extensions)
            ]
        return self._field_segments

    @property
    def field(self):
        """Field reader spanning all segments (single or MultiSegmentReader)."""
        if self._field is None:
            readers = self._field_segment_readers
            self._field = readers[0] if len(readers) == 1 \
                else MultiSegmentReader(readers)
        return self._field

    def mom(self, species=None):
        """Moment reader for *species* (defaults to the first), all segments."""
        if species is None:
            species = self.species[0]
        if species not in self._mom:
            readers = [
                self._make_reader("mom", ext, self.params.get(i), species=species)
                for i, ext in enumerate(self.extensions)
            ]
            self._mom[species] = readers[0] if len(readers) == 1 \
                else MultiSegmentReader(readers)
        return self._mom[species]

    def _mom_list(self) -> list:
        return [self.mom(n) for n in self.species]

    def _mom_dict(self) -> dict:
        return {n: self.mom(n) for n in self.species}

    def _indices(self, reader, t):
        """Return ``(all_times, selected_indices)`` for time window *t*."""
        a, b = _bounds(t)
        times = reader.read_all_times()
        idx = np.where((times >= a) & (times <= b))[0]
        return times, idx

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @cached_property
    def nrg(self):
        return _BoundNrg(self)

    @cached_property
    def spectra(self):
        return _BoundSpectra(self)

    @cached_property
    def profiles(self):
        return _BoundProfiles(self)

    @cached_property
    def fluxes2d(self):
        return _BoundFluxes2D(self)

    @cached_property
    def shearing(self):
        return _BoundShearing(self)

    @cached_property
    def contours(self):
        return _BoundContours(self)

    def ballooning(self, ky=None, **kw):
        """Ballooning mode structure for a chosen ``ky`` (local runs only)."""
        from .diagnostics.ballooning import Ballooning
        return Ballooning(self, ky=ky, **kw)

    @cached_property
    def growthrate(self):
        from .diagnostics.growthrate import GrowthRate
        return GrowthRate(self)

    @cached_property
    def amplitude(self):
        from .diagnostics.amplitude import AmplitudeSpectra
        return AmplitudeSpectra(self)

    @cached_property
    def zonal(self):
        from .diagnostics.zonal import Zonal
        return Zonal(self)

    def __repr__(self) -> str:
        geom = "local" if self.is_local else "global"
        return (f"<Run {self.path.name!r} | {geom} | "
                f"{len(self.extensions)} segment(s) | "
                f"species={self.species}>")


# ---------------------------------------------------------------------------
# Bound diagnostic wrappers
# ---------------------------------------------------------------------------

class _BoundNrg:
    """Energy/flux time traces (``nrg`` files)."""

    def __init__(self, run: Run):
        self.run = run
        self._reader = NrgReader(run._folder, run.params.get(0),
                                 extensions=run.extensions)

    def read_all(self):
        return self._reader.read_all()

    @property
    def data(self):
        return self._reader.dataset(self.run.params.get(0))

    def plot(self, **kw):
        self._reader.plot()


class _BoundSpectra:
    """Time-averaged flux spectra; auto-dispatches local vs global."""

    def __init__(self, run: Run):
        self.run = run
        self.is_global = not run.is_local
        self._diag = (SpectraGlobal(folder=run._folder) if self.is_global
                      else Spectra(folder=run._folder))

    def compute(self, t=None):
        a, b = _bounds(t)
        r = self.run
        if self.is_global:
            self._diag.compute_and_save(r.field, r._mom_list(), r.coords[0],
                                        r.geometry[0], r.params.get(0), a, b)
        else:
            self._diag.compute_missing(r.field, r._mom_list(), r.coords[0],
                                       r.geometry[0], r.params, a, b)
        return self

    def save(self, t=None):
        return self.compute(t)

    def load(self, t=None):
        self.compute(t)
        a, b = _window(t)
        return self._diag.dataset(self.run.coords[0], self.run.params.get(0),
                                  self.run.species, a, b)

    @property
    def data(self):
        return self.load()

    def plot(self, t=None, **kw):
        a, b = _bounds(t)
        r = self.run
        if self.is_global:
            self.compute(t)
            self._diag.plot(r.coords[0], r.params.get(0), a, b, **kw)
        else:
            self._diag.plot(r.field, r._mom_list(), r.coords, r.geometry,
                            r.params, a, b)


class _BoundProfiles:
    """Flux-surface-averaged radial profiles (time-resolved)."""

    def __init__(self, run: Run):
        self.run = run
        self._diag = Profiles(folder=run._folder)

    def compute(self, t=None):
        a, b = _bounds(t)
        r = self.run
        self._diag.compute_and_save(r._mom_dict(), r.coords[0], r.geometry[0],
                                    r.params.get(0), a, b)
        return self

    def save(self, t=None):
        return self.compute(t)

    def load(self, t=None):
        self.compute(t)
        a, b = _window(t)
        return self._diag.dataset(self.run.coords[0], self.run.params.get(0),
                                  self.run.species, a, b)

    @property
    def data(self):
        return self.load()

    def plot(self, t=None, eq_profs=None, **kw):
        a, b = _bounds(t)
        self.compute(t)
        self._diag.plot(self.run.coords[0], self.run.params.get(0), a, b,
                        equilibrium_profiles=eq_profs)


class _BoundFluxes2D:
    """x-resolved transport fluxes (time-averaged)."""

    def __init__(self, run: Run):
        self.run = run
        self._diag = Fluxes2D(folder=run._folder)

    def compute(self, t=None):
        a, b = _bounds(t)
        r = self.run
        self._diag.compute_and_save(r.field, r._mom_list(), r.coords[0],
                                    r.geometry[0], r.params.get(0), a, b)
        return self

    def save(self, t=None):
        return self.compute(t)

    def load(self, t=None):
        self.compute(t)
        a, b = _window(t)
        return self._diag.dataset(self.run.coords[0], self.run.params.get(0),
                                  self.run.species, a, b)

    @property
    def data(self):
        return self.load()

    def plot(self, t=None, show_heatmaps=False, **kw):
        a, b = _bounds(t)
        self.compute(t)
        self._diag.plot(self.run.coords[0], self.run.params.get(0), a, b,
                        show_heatmaps=show_heatmaps)


class _BoundShearing:
    """ExB shearing rate / zonal electric field (time-resolved)."""

    def __init__(self, run: Run):
        self.run = run
        self._diag = ShearingRate(folder=run._folder)

    def compute(self, t=None):
        a, b = _bounds(t)
        r = self.run
        self._diag.compute_and_save(r._field_segment_readers, r.coords,
                                    r.geometry, r.params, a, b)
        return self

    def save(self, t=None):
        return self.compute(t)

    def load(self, t=None):
        self.compute(t)
        return self._diag.dataset(self.run.coords[0], self.run.params.get(0))

    @property
    def data(self):
        return self.load()

    def plot(self, t=None, **kw):
        a, b = _bounds(t)
        self.compute(t)
        self._diag.plot(self.run.coords[0], a, b)


class _BoundContours:
    """2D field/moment slice visualisation (plot-only)."""

    def __init__(self, run: Run):
        self.run = run
        self._diag = Contours()

    @property
    def data(self):
        raise NotImplementedError(
            "Contours is a visualisation diagnostic; use .plot(...) instead.")

    def plot(self, t=None, field=0, ifft=None, species=None, **kw):
        a, b = _bounds(t)
        reader = self.run.field if species is None else self.run.mom(species)
        self._diag.plot_timeseries_2d(
            reader, a, b, field=field, ifft=ifft,
            params_list=self.run.params.get(0), coords=self.run.coords[0],
            species=species, **kw)
