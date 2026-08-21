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

Fortran-binary, HDF5 and ADIOS2 BP outputs are all supported: each segment's
reader is chosen from the file actually on disk (``field_0001`` vs
``field_0001.h5`` vs ``field_0001.bp``), so no extra arguments are needed (BP
runs additionally require the ``adios2`` package).

GENE-3D runs work through the same facade. They are real-space in x *and* y and
write only HDF5, which :attr:`Run.geometry_kind` reports as ``'xy_global'``;
diagnostics dispatch on that rather than on the two-valued :attr:`Run.is_local`.
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
    H5Reader,
    MultiSegmentReader,
    load_equilibrium_profiles,
)
from .diagnostics import NrgReader


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
    def geometry_kind(self) -> str:
        """
        ``'flux_tube'``, ``'x_global'``, ``'y_global'`` or ``'xy_global'``.

        Prefer this over :attr:`is_local` when the distinction that matters is
        *which* directions are spectral — GENE-3D is global in x and y, so
        ``is_local`` alone cannot tell it apart from an x-global run.
        """
        return self.params.geometry_kind(0)

    @property
    def is_3d(self) -> bool:
        """Whether this is a GENE-3D run (real space in x *and* y)."""
        return self.params.is_3d(0)

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

    @cached_property
    def eq_profiles(self):
        """
        Background equilibrium profiles per species — ``None`` for local runs.

        Global diagnostics need the background ``n(x)`` and ``T(x)`` to build
        the density and temperature prefactors that enter the heat and momentum
        fluxes; computing those without the profiles silently assumes flat
        backgrounds, which is wrong for a global run. Loaded once from
        ``profiles_<species><ext>`` for the first segment — the same segment
        the diagnostics take their params, coords, and geometry from.

        Assign to this attribute to supply profiles from another source::

            run.eq_profiles = {"ions": {"T": ..., "n": ..., "omt": ..., "omn": ...}}

        Raises
        ------
        FileNotFoundError
            If a global run has no profile file for one or more species.
        ValueError
            If a profile's radial grid does not match the simulation grid.
        """
        if self.is_local:
            return None

        ext = self.extensions[0]
        nx = int(self.params.get(0)["box"]["nx0"])
        profiles, missing = {}, []
        for name in self.species:
            try:
                ep = load_equilibrium_profiles(self._folder, ext, name)
            except FileNotFoundError:
                missing.append(f"profiles_{name}{ext}")
                continue
            npts = np.size(ep["T"])
            if npts != nx:
                raise ValueError(
                    f"Equilibrium profile 'profiles_{name}{ext}' has {npts} "
                    f"radial points but the run grid has nx0={nx}. The profile "
                    "must be on the simulation's radial grid.")
            profiles[name] = ep

        if missing:
            raise FileNotFoundError(
                f"Global run is missing equilibrium profile file(s) in "
                f"{self._folder}: {', '.join(missing)}. These carry the "
                "background n(x) and T(x) that the heat and momentum fluxes "
                "need; without them those fluxes would assume flat profiles. "
                "Assign run.eq_profiles = {...} to supply them from elsewhere.")
        return profiles

    # ------------------------------------------------------------------
    # Continuation / restart validation
    # ------------------------------------------------------------------

    def _validate_grid(self) -> None:
        """Warn if grid/geometry dimensions differ across segments."""
        plist = self.params.tolist()
        if len(plist) <= 1:
            return
        keys = ("nx0", "nky0", "ny0", "nz0", "n_spec")
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
        Build one segment reader, auto-selecting the output format.

        The file present on disk decides. Binary (``field_0001``) wins when
        several exist, since it needs no optional package; then HDF5
        (``field_0001.h5``), then ADIOS2 BP (``field_0001.bp`` / ``field.bp``).
        With nothing on disk yet the ``write_h5``/``write_bp`` parameters
        choose, falling back to binary so the first read fails with a clear
        message naming the file it wanted. GENE-3D only ever writes HDF5, and
        its ``write_h5 = T`` lives in ``&info`` — already reconciled into
        ``in_out`` by :class:`~genetools.io.params.Params`.

        Constructing a :class:`BPReader` without the ``adios2`` package raises
        a clear ``ImportError``.
        """
        sp = f"_{species}" if species else ""
        binfile = f"{self._folder}{file_type}{sp}{ext}"
        if os.path.exists(binfile):
            return BinaryReader(file_type, self._folder, ext, params_i,
                                species=species)

        h5_ext = ext + ".h5"
        if os.path.exists(f"{self._folder}{file_type}{sp}{h5_ext}"):
            return H5Reader(file_type, self._folder, h5_ext, params_i,
                            species=species)

        bp_ext = ("" if ext == ".dat" else ext) + ".bp"
        bpfile = f"{self._folder}{file_type}{sp}{bp_ext}"
        if os.path.exists(bpfile) or params_i.get("in_out", {}).get("write_bp", False):
            return BPReader(file_type, self._folder, bp_ext, params_i,
                            species=species)

        if params_i.get("in_out", {}).get("write_h5", False):
            return H5Reader(file_type, self._folder, h5_ext, params_i,
                            species=species)

        # Nothing present: default to binary (errors clearly on first read).
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
        from .diagnostics.spectra import Spectra
        return Spectra(self)

    @cached_property
    def profiles(self):
        from .diagnostics.profiles import Profiles
        return Profiles(self)

    @cached_property
    def fluxes2d(self):
        from .diagnostics.fluxes2d import Fluxes2D
        return Fluxes2D(self)

    @cached_property
    def shearing(self):
        from .diagnostics.shearingrate import ShearingRate
        return ShearingRate(self)

    @cached_property
    def contours(self):
        """2-D field/moment slices; options are passed to ``.plot()``."""
        from .diagnostics.contours import Contours
        return Contours(self)

    def ballooning(self, ky=None, **kw):
        """Ballooning mode structure for a chosen ``ky`` (local runs only)."""
        if self.is_3d:
            raise NotImplementedError(
                "Ballooning mode structure needs a single ky mode, which a "
                "GENE-3D run does not have — it is real-space in y. Use "
                "run.planes for the field-aligned structure instead.")
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

    @cached_property
    def vexmax(self):
        from .diagnostics.vexmax import VexMax
        return VexMax(self)

    @cached_property
    def profile_diag(self):
        from .diagnostics.profile_diag import ProfileDiag
        return ProfileDiag(self)

    # ------------------------------------------------------------------
    # GENE-3D only
    #
    # These have no spectral counterpart, so they raise rather than silently
    # doing something different for a flux-tube or x-global run.
    # ------------------------------------------------------------------

    def slices(self, **kw):
        """Every 1-D and 2-D reduction of a GENE-3D snapshot."""
        from .diagnostics.slices import Slices
        return Slices(self, **kw)

    def timetraces(self, **kw):
        """Volume-averaged and ky-resolved time traces (GENE-3D)."""
        from .diagnostics.timetraces import TimeTraces
        return TimeTraces(self, **kw)

    @cached_property
    def gam(self):
        """Zonal-flow / GAM oscillation traces (GENE-3D)."""
        from .diagnostics.gam import Gam
        return Gam(self)

    @cached_property
    def chi(self):
        """Heat diffusivity against the driving gradient (GENE-3D)."""
        from .diagnostics.chi import ChiGradient
        return ChiGradient(self)

    @cached_property
    def omega(self):
        """Real-frequency view of the growth-rate fit (GENE-3D)."""
        from .diagnostics.omega import Omega
        return Omega(self)

    @cached_property
    def geometry_plots(self):
        """Geometry coefficients along cuts and planes (GENE-3D)."""
        from .diagnostics.geometry_plots import GeometryPlots
        return GeometryPlots(self)

    @cached_property
    def srcmom(self):
        """Krook source moments as radial profiles (GENE-3D)."""
        from .diagnostics.velocity import SrcMom
        return SrcMom(self)

    @cached_property
    def vsp(self):
        """Velocity-space output on the (z, v_par, mu) grid (GENE-3D)."""
        from .diagnostics.velocity import VspSlice
        return VspSlice(self)

    def planes(self, **kw):
        """Data remapped onto geometric (theta, phi) angles (GENE-3D)."""
        from .diagnostics.planes import Planes
        return Planes(self, **kw)

    def vis3d(self, **kw):
        """VTK export for external 3-D visualisation (GENE-3D)."""
        from .diagnostics.vis import Vis
        return Vis(self, **kw)


    def __repr__(self) -> str:
        geom = self.geometry_kind
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

    def plot(self, t=None):
        # nrg plots the full time series; t is accepted for a uniform facade API.
        self._reader.plot()
