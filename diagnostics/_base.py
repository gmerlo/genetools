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

    def __init__(self, outfile: str):
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

        Parameters
        ----------
        fld_reader : reader
            Field reader.
        mom_readers : list of readers
            Moment readers (one per species).
        t_start, t_stop : float
            Time window.
        params : dict
            Parameter dictionary.

        Returns
        -------
        idx_fld, idx_mom : list of int
            Filtered iteration indices for field and moment readers.
        """
        istep_fld = int(params["in_out"]["istep_field"])
        istep_mom = int(params["in_out"]["istep_mom"])
        L = int(np.lcm(istep_fld, istep_mom))
        stride_fld = L // istep_fld
        stride_mom = L // istep_mom

        times_fld = fld_reader.read_all_times()
        idx_fld = np.where(
            (times_fld >= t_start) & (times_fld <= t_stop)
        )[0][::stride_fld]
        times_mom = mom_readers[0].read_all_times()
        idx_mom = np.where(
            (times_mom >= t_start) & (times_mom <= t_stop)
        )[0][::stride_mom]

        saved_times = self._load_saved_times()
        if saved_times.size == 0:
            return idx_fld.tolist(), idx_mom.tolist()

        saved_sorted = np.sort(saved_times.astype(np.float64))

        def _filter(indices, all_times):
            if len(indices) == 0:
                return []
            cand = all_times[indices]
            tol = np.maximum(1e-6, np.abs(cand) * 1e-6)
            pos = np.searchsorted(saved_sorted, cand)
            found = np.zeros(len(cand), dtype=bool)
            for offset in (0, -1):
                ic = np.clip(pos + offset, 0, len(saved_sorted) - 1)
                found |= np.abs(saved_sorted[ic] - cand) <= tol
            return [i for i, f in zip(indices, found) if not f]

        return _filter(idx_fld, times_fld), _filter(idx_mom, times_mom)
