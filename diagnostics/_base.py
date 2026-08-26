# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
_base.py — Base classes for diagnostics.

:class:`CachingDiagnostic` carries the HDF5-persistence machinery: time
checking, windowed loading, time averaging, and pairing field with moment
snapshots by time value.

:class:`RunDiagnostic` builds on it to give every diagnostic one uniform
surface — constructed from a :class:`~genetools.run.Run`, offering ``.data``,
``.plot(t=...)`` and ``.save()`` — and one place for the time-window handling
that would otherwise be copy-pasted into each module. Set ``cache_file`` on a
subclass to make it HDF5-backed, so its time series is streamed to disk
incrementally and read back on demand rather than being held in memory; leave it
unset for diagnostics whose results are small enough to keep in memory.
"""

import os
import numpy as np
import h5py

from genetools.compat import trapz as _trapz


class CachingDiagnostic:
    """
    Base class for diagnostics that cache results to an HDF5 file.

    Subclasses should set ``self.outfile`` in their ``__init__``.

    Provides
    --------
    _load_saved_times() → np.ndarray
    _is_already_saved(time, saved_times) → bool
    _time_average(arr, times) → np.ndarray
    _sync_field_mom_indices(fld_reader, mom_readers, t_start, t_stop, params) → (list, list)
    """

    def __init__(self, outfile: str = None, folder: str = None):
        if outfile and folder is not None and not os.path.dirname(outfile):
            self.outfile = os.path.join(folder, outfile)
        else:
            self.outfile = outfile

    # ------------------------------------------------------------------
    # Time-checking helpers
    # ------------------------------------------------------------------

    def _load_saved_times(self) -> np.ndarray:
        """Load all saved times from the HDF5 file (empty array if none)."""
        if not self.outfile or not os.path.exists(self.outfile):
            return np.array([], dtype=np.float64)
        with h5py.File(self.outfile, "r") as f:
            if "time" not in f:
                return np.array([], dtype=np.float64)
            return f["time"][...]

    @staticmethod
    def _is_already_saved(time: float, saved_times: np.ndarray) -> bool:
        """Check if *time* is in *saved_times* within relative tolerance."""
        if saved_times.size == 0:
            return False
        tol = max(1e-6, abs(time) * 1e-6)
        return bool(np.any(np.abs(saved_times - time) <= tol))

    @staticmethod
    def _select_window(time, t_start=None, t_stop=None):
        """
        Sort cached times, apply a window, and build HDF5 read indices.

        The cached HDF5 rows are not guaranteed to be in time order (windows
        may be computed out of order; restarts append later segments first), so
        loading must pair each time with its own row. h5py fancy indexing
        requires strictly increasing indices, hence the read/unsort pair.

        Returns
        -------
        times : np.ndarray
            Selected times in ascending time order.
        read_idx : np.ndarray of int
            File positions to read, strictly increasing (h5py-safe).
        unsort : np.ndarray of int
            Permutation such that ``data[..., read_idx][..., unsort]`` is in
            ascending time order, aligned with *times*.
        """
        time = np.asarray(time)
        order_t = np.argsort(time, kind="stable")
        tsorted = time[order_t]
        mask = np.ones(tsorted.size, dtype=bool)
        if t_start is not None:
            mask &= tsorted >= t_start
        if t_stop is not None:
            mask &= tsorted <= t_stop
        file_pos = order_t[mask]                  # file rows, in time order
        read_idx = np.sort(file_pos)
        unsort = np.argsort(np.argsort(file_pos, kind="stable"), kind="stable")
        return tsorted[mask], read_idx, unsort

    @staticmethod
    def _time_dtype(params) -> type:
        """
        Resolve the on-disk time-axis dtype from GENE's output precision.

        Single-precision runs store time as ``float32``, double-precision as
        ``float64`` — matching the precision GENE wrote the data with. Used
        consistently for the cached ``time`` dataset across all diagnostics.
        Time *comparisons* (dedup) are always done in float64.
        """
        prec = "double"
        if isinstance(params, dict):
            prec = str(params.get("info", {}).get("precision", "double")).lower()
        return np.float32 if prec.startswith("s") else np.float64

    # ------------------------------------------------------------------
    # Time averaging
    # ------------------------------------------------------------------

    @staticmethod
    def _time_average(arr: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Trapezoidal time average of *arr* over *times*.

        Parameters
        ----------
        arr : np.ndarray
            Array with time along axis 0.
        times : np.ndarray
            1-D time array.

        Returns
        -------
        np.ndarray
            Time-averaged array (one fewer dimension than *arr*).
        """
        dt = times[-1] - times[0]
        if dt == 0 or len(times) == 1:
            return arr[0]
        return _trapz(arr, x=times, axis=0) / dt

    # ------------------------------------------------------------------
    # Field / moment index synchronisation
    # ------------------------------------------------------------------

    def _sync_field_mom_indices(self, fld_reader, mom_readers,
                                t_start, t_stop, params):
        """
        Compute aligned field and moment iteration indices, filtering
        out already-saved timesteps.

        The two lists are paired by matching *time values*, not by position.
        Field and moment files are generally written at different cadences
        (``istep_field`` vs ``istep_mom``), and either may be truncated when a
        run is killed mid-write, so the n-th field record and the n-th moment
        record usually belong to different times. Pairing them positionally
        silently correlates a field snapshot with moments from another time,
        and crashes outright when the moment file is the shorter of the two.
        Only times present in both files, and not already cached, are returned.

        Parameters
        ----------
        fld_reader : reader
            Field reader.
        mom_readers : list of readers
            Moment readers (one per species).
        t_start, t_stop : float
            Time window.
        params : dict
            Parameter dictionary. Retained for API compatibility; the cadence
            keys are no longer needed now that pairing is by time value.

        Returns
        -------
        idx_fld, idx_mom : list of int
            Equal-length, time-aligned iteration indices.
        """
        times_fld = np.asarray(fld_reader.read_all_times(), dtype=np.float64)
        times_mom = np.asarray(mom_readers[0].read_all_times(), dtype=np.float64)

        idx_fld = np.where((times_fld >= t_start) & (times_fld <= t_stop))[0]
        idx_mom = np.where((times_mom >= t_start) & (times_mom <= t_stop))[0]
        if idx_fld.size == 0 or idx_mom.size == 0:
            return [], []

        # Moment times, sorted, for tolerant lookup by value.
        order = np.argsort(times_mom[idx_mom], kind="stable")
        mom_idx_sorted = idx_mom[order]
        mom_t_sorted = times_mom[mom_idx_sorted]
        pos = np.searchsorted(mom_t_sorted, times_fld[idx_fld])

        saved_sorted = np.sort(self._load_saved_times().astype(np.float64))

        def _nearest(sorted_times, at, value, tol):
            """Index into *sorted_times* matching *value*, or None."""
            for offset in (0, -1):
                j = int(np.clip(at + offset, 0, sorted_times.size - 1))
                if abs(sorted_times[j] - value) <= tol:
                    return j
            return None

        pair_fld, pair_mom = [], []
        for k, i_fld in enumerate(idx_fld):
            t = times_fld[i_fld]
            tol = max(1e-6, abs(t) * 1e-6)

            j = _nearest(mom_t_sorted, pos[k], t, tol)
            if j is None:
                continue          # no moment output at this time
            if saved_sorted.size:
                at = int(np.searchsorted(saved_sorted, t))
                if _nearest(saved_sorted, at, t, tol) is not None:
                    continue      # already in the cache

            pair_fld.append(int(i_fld))
            pair_mom.append(int(mom_idx_sorted[j]))

        return pair_fld, pair_mom


# ---------------------------------------------------------------------------
# Run-native diagnostic base
# ---------------------------------------------------------------------------

class RunDiagnostic(CachingDiagnostic):
    """
    Base for diagnostics constructed from a :class:`~genetools.run.Run`.

    Subclasses implement ``compute(t=None)`` and ``dataset(t=None)`` (and
    usually ``plot``); everything shared lives here — the time-window helpers,
    the in-memory result cache, and the ``.data`` / ``.save()`` surface.

    ``run`` may be ``None`` for a *detached* instance. Nothing that reads the run
    works then, but the pure computational helpers do, which is what lets them be
    tested without a run directory on disk.

    Class attributes
    ----------------
    name : str
        Filename stem used by :meth:`save`.
    cache_file : str or None
        HDF5 filename for a disk-backed diagnostic, resolved relative to the run
        directory. ``None`` keeps everything in memory.
    supported : tuple of str or None
        Geometry kinds this diagnostic handles, checked by :meth:`_require`.
        ``None`` means all of them.
    """

    name = "diagnostic"
    cache_file = None
    supported = None

    def __init__(self, run=None):
        self.run = run
        self._cache = {}
        super().__init__(self.cache_file, folder=getattr(run, "_folder", None))
        if run is not None and self.supported is not None:
            self._require(*self.supported)

    # ------------------------------------------------------------------
    # Run shortcuts
    # ------------------------------------------------------------------

    @property
    def params(self) -> dict:
        """Parameter dict of the first segment."""
        return self.run.params.get(0)

    @property
    def coord(self) -> dict:
        """Coordinate dict of the first segment."""
        return self.run.coords[0]

    @property
    def geom(self) -> dict:
        """Geometry dict of the first segment."""
        return self.run.geometry[0]

    @property
    def geometry_kind(self) -> str:
        return self.run.geometry_kind

    @property
    def is_3d(self) -> bool:
        return self.run.is_3d

    def _require(self, *kinds) -> None:
        """
        Raise unless this run's geometry is one of *kinds*.

        Refusing beats guessing: a diagnostic that has no meaning for a geometry
        should say so rather than quietly reducing the data some other way.
        """
        if self.geometry_kind not in kinds:
            raise NotImplementedError(
                f"{type(self).__name__} supports {', '.join(kinds)}; this run "
                f"is {self.geometry_kind!r}.")

    # ------------------------------------------------------------------
    # Time windows
    # ------------------------------------------------------------------

    @staticmethod
    def _window(t):
        """Normalise *t* to ``(start, stop)``, either of which may be ``None``."""
        if t is None:
            return None, None
        if isinstance(t, (tuple, list)):
            a, b = t
            return (None if a is None else float(a),
                    None if b is None else float(b))
        return float(t), None

    @classmethod
    def _bounds(cls, t):
        """Like :meth:`_window` but with concrete float bounds for streaming."""
        a, b = cls._window(t)
        return (-1e30 if a is None else a, 1e30 if b is None else b)

    @staticmethod
    def _key(t):
        """Hashable cache key for a time window."""
        return tuple(t) if isinstance(t, (tuple, list)) else t

    def _indices(self, reader, t):
        """
        Return ``(all_times, selected_indices)`` for the window *t*.

        Raises
        ------
        ValueError
            If the window selects nothing, quoting the range that is available —
            an empty window is nearly always a mistyped time, and the available
            range is the piece of information needed to fix it.
        """
        times = np.asarray(reader.read_all_times())
        a, b = self._bounds(t)
        idx = np.where((times >= a) & (times <= b))[0]
        if idx.size == 0:
            span = (f"{times[0]:.4g}..{times[-1]:.4g}" if times.size
                    else "no output at all")
            raise ValueError(
                f"{type(self).__name__}: no output in the requested time "
                f"window; available: {span}")
        return times, idx

    # ------------------------------------------------------------------
    # Uniform surface
    # ------------------------------------------------------------------

    def _common_indices(self, readers, t, tol=1e-6):
        """
        Return ``(times, {id(reader): indices})`` for the times *all* readers have.

        Per-species moment files do not always hold the same number of complete
        snapshots: output is written species by species, so a run that is still
        going — or was killed mid-write — leaves one file a snapshot short. The
        H5 reader drops those incomplete snapshots, and streaming each species
        over its own index list then yields arrays of different lengths, which
        only shows up later as ``all input arrays must have the same shape``.

        Matching by time value rather than by position also keeps a species from
        being paired with another's snapshot from a different time.

        Raises
        ------
        ValueError
            If no time is common to every reader inside the window.
        """
        readers = list(readers)
        lo, hi = self._bounds(t)
        per = []
        for reader in readers:
            times = np.asarray(reader.read_all_times(), dtype=np.float64)
            keep = np.where((times >= lo) & (times <= hi))[0]
            per.append((times, keep))

        base_times, base_keep = per[0]
        common, index_of = [], [[] for _ in readers]
        for i0 in base_keep:
            tv = base_times[i0]
            atol = max(tol, abs(tv) * tol)
            picks = [int(i0)]
            for times, keep in per[1:]:
                if keep.size == 0:
                    picks = None
                    break
                j = keep[int(np.argmin(np.abs(times[keep] - tv)))]
                if abs(times[j] - tv) > atol:
                    picks = None
                    break
                picks.append(int(j))
            if picks is None:
                continue
            common.append(tv)
            for slot, k in enumerate(picks):
                index_of[slot].append(k)

        if not common:
            # Two distinct failures: the window catches nothing at all (usually a
            # mistyped time), or the files each have output but never at the same
            # time. Say which, and quote the ranges needed to fix it.
            if all(kp.size == 0 for _, kp in per):
                spans = "; ".join(
                    f"{tm[0]:.4g}..{tm[-1]:.4g}" if tm.size else "no output"
                    for tm, _ in per)
                raise ValueError(
                    f"{type(self).__name__}: no output in the requested time "
                    f"window; available: {spans}")
            spans = "; ".join(
                f"{np.min(tm[kp]):.4g}..{np.max(tm[kp]):.4g}" if kp.size else "empty"
                for tm, kp in per)
            raise ValueError(
                f"{type(self).__name__}: no output time is common to all files "
                f"in the requested window (per-file ranges: {spans})")
        return np.asarray(common), {id(r): idx for r, idx in zip(readers, index_of)}

    def _sources(self, quantities, species=None):
        """
        Map each requested quantity to the reader that holds it.

        Returns ``[(reader, [names...]), ...]`` — grouped per reader so each file
        is streamed once however many of its variables were asked for. The field
        file wins a name collision, which is what the field/moment split gives
        anyway.
        """
        fld = self.run.field
        mom = self.run.mom(species) if species else None
        out = {}
        for name in quantities:
            if name in fld.var_names:
                out.setdefault(id(fld), (fld, []))[1].append(name)
            elif mom is not None and name in mom.var_names:
                out.setdefault(id(mom), (mom, []))[1].append(name)
            else:
                available = list(fld.var_names) + (
                    list(mom.var_names) if mom is not None else [])
                raise KeyError(
                    f"unknown quantity {name!r}; available: "
                    f"{', '.join(available)}")
        return list(out.values())

    @staticmethod
    def _t_average(da):
        """
        Trapezoidal average of a DataArray over its ``time`` dimension.

        Not ``.mean("time")``. GENE's timestep is adaptive and output happens
        every ``istep_*`` *steps*, so output times are generally unevenly spaced
        and a plain mean weights every sample equally regardless of the interval
        it stands for. On a realistically uneven axis the two differ by tens of
        percent, and a time-averaged profile or flux is usually the number being
        quoted.
        """
        t = np.asarray(da["time"], dtype=float)
        if t.size <= 1:
            return da.isel(time=0)
        span = float(t[-1] - t[0])
        if span == 0:
            return da.isel(time=0)
        return da.integrate("time") / span

    @property
    def data(self):
        """The diagnostic's :class:`xarray.Dataset` over the full time range."""
        return self.dataset()

    def save(self, t=None, path=None) -> str:
        """Write the dataset to NetCDF and return the path."""
        ds = self.dataset(t)
        out = path or str(self.run.path / f"{self.name}.nc")
        ds.to_netcdf(out)
        return out

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} {self.run.path.name!r} "
                f"| {self.geometry_kind}>")
