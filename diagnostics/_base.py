# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
_base.py — Base class for HDF5-caching diagnostics.

Provides shared logic for time-checking, loading, and time-averaging
that is used by ShearingRate, Profiles, Fluxes2D, SpectraGlobal,
and (partially) Spectra.
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

    def __init__(self, outfile: str, folder: str = None):
        if folder is not None and not os.path.dirname(outfile):
            self.outfile = os.path.join(folder, outfile)
        else:
            self.outfile = outfile

    # ------------------------------------------------------------------
    # Time-checking helpers
    # ------------------------------------------------------------------

    def _load_saved_times(self) -> np.ndarray:
        """Load all saved times from the HDF5 file (empty array if none)."""
        if not os.path.exists(self.outfile):
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
