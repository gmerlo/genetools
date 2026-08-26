# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import os
import h5py
from concurrent.futures import ThreadPoolExecutor

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from genetools.compat import trapz as _trapz
from genetools.diagnostics._base import (CachingDiagnostic,
                                        RunDiagnostic)
from genetools.diagnostics import _gene3d as g3


#: Cache file per geometry. The two schemas differ in dimension order and
#: grouping, so they stay in separate files; merging them would invalidate every
#: existing cache for no visible gain.
_CACHE_FILES = {"flux_tube": "flux_spectra.h5",
                "x_global": "spectra_global.h5"}

#: HDF5 keys written before the geometries were made to agree on flux names.
#: Both caches are translated on read and migrated in place before an append.
_LEGACY_FLUX_NAMES = {
    # x-global, inside each species group
    "Qes_ky": "Q_es_ky", "Ges_ky": "Gamma_es_ky", "Pes_ky": "Pi_es_ky",
    "Qem_ky": "Q_em_ky", "Gem_ky": "Gamma_em_ky", "Pem_ky": "Pi_em_ky",
}
#: Flux-tube substitutions, applied to the flat ``{species}_{flux}_{axis}`` keys.
_LEGACY_FLUX_INFIX = {"_G_es_": "_Gamma_es_", "_G_em_": "_Gamma_em_"}


def _compute_flux_yspectra(a: np.ndarray, b: np.ndarray,
                           C_xy: np.ndarray,
                           J_norm: np.ndarray) -> np.ndarray:
    """
    Compute ky-resolved, flux-surface-averaged cross-correlation.

    Parameters
    ----------
    a, b : np.ndarray
        Complex arrays of shape ``(nx, nky, nz)``.
    C_xy : np.ndarray or float
        Metric coefficient, shape ``(nx, nz)`` or scalar.
    J_norm : np.ndarray
        Normalised Jacobian, shape ``(nx, nz)``.

    Returns
    -------
    np.ndarray
        Real array of shape ``(nx, nky)``.
    """
    nky = a.shape[1]

    # z-summed cross-correlation per ky (vectorised over ky), with Hermitian
    # weighting: factor 1 for ky=0, factor 2 for ky>0. J_norm may be (nx, nz)
    # or (nz,); insert the ky axis so it broadcasts over ky either way.
    J = np.asarray(J_norm)
    Jb = J[:, np.newaxis, :] if J.ndim == 2 else J[np.newaxis, np.newaxis, :]
    cross = np.real(np.conj(a) * b)                       # (nx, nky, nz)
    out = np.sum(cross * Jb, axis=2)                      # (nx, nky)
    if nky > 1:
        out[:, 1:] *= 2.0

    # Division by C_xy (applied after z-summation, matching MATLAB)
    C_xy_arr = np.asarray(C_xy)
    if C_xy_arr.ndim == 2:
        # (nx, nz) -> average over z to get per-x scalar
        out /= np.mean(C_xy_arr, axis=1)[:, np.newaxis]
    elif C_xy_arr.ndim == 1:
        out /= C_xy_arr[:, np.newaxis]
    else:
        out /= float(C_xy_arr)

    return out


def _ky_weighted_label(label: str) -> str:
    """Turn ``'$Q_{\\rm es}$'`` into ``'$k_y\\,Q_{\\rm es}$'``."""
    return r"$k_y\," + label.lstrip("$")


def _flux_norm_and_cmap(data: np.ndarray):
    """
    Return ``(norm, cmap)`` for a logarithmic flux panel.

    A flux that keeps one sign — the usual case for heat flux — is scaled with
    a plain :class:`LogNorm` over a sequential map, so the whole colour range
    covers the actual dynamic range. Forcing a symmetric range on such data
    would spend half the colours on values that never occur and wash the panel
    out.

    A flux that genuinely changes sign (inward particle transport,
    counter-current momentum) cannot use LogNorm at all, since it blanks the
    negative half. Those get a symmetric-log norm — logarithmic in magnitude,
    linear through zero — over a diverging map so the sign stays readable.

    ``(None, ...)`` falls back to matplotlib's linear default when there is no
    dynamic range to scale.
    """
    finite = np.asarray(data)[np.isfinite(data)]
    nonzero = finite[finite != 0]
    if nonzero.size == 0:
        return None, "viridis"

    vmax = float(np.abs(nonzero).max())
    vmin = float(np.abs(nonzero).min())

    if (nonzero < 0).any():
        # Keep the linear window below the smallest resolved magnitude, but do
        # not let it collapse to zero when the dynamic range is extreme.
        linthresh = float(max(vmin, vmax * 1e-6))
        return SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax,
                          base=10), "RdBu_r"
    if vmin == vmax:
        return None, "viridis"
    return LogNorm(vmin=vmin, vmax=vmax), "viridis"


class Spectra(RunDiagnostic):
    name = "spectra"
    cache_file = "flux_spectra.h5"

    #: GENE-3D flux moments, electrostatic then electromagnetic.
    _ES_FLUXES = ("Gamma_es", "Q_es")
    _EM_FLUXES = ("Gamma_em", "Q_em")

    _TITLES_3D = {
        "Gamma_es": r"$\Gamma_{es}$", "Gamma_em": r"$\Gamma_{em}$",
        "Q_es": r"$Q_{es}$", "Q_em": r"$Q_{em}$",
    }

    def __init__(self, run=None, outfile: str = None, folder: str = None,
                 x_avg_lims=None, buffer_frac=0.0):
        """
        Parameters
        ----------
        run : genetools.run.Run, optional
        outfile, folder : str, optional
            Override the HDF5 cache location; normally derived from the run.
        x_avg_lims : (float, float), optional
            Global geometries: radial range the ``*_ky`` reduction averages
            over, in ``x/a``. Defaults to the **whole domain** — the ``(x, ky)``
            map is the primary output and nothing is averaged away unless asked.
        buffer_frac : float
            Fraction trimmed from each radial end when *x_avg_lims* is not
            given. Zero by default; set it (0.1 is the usual choice) to keep the
            Krook buffer regions, where the fluxes are unphysical, out of the
            radial average.
        """
        self.x_avg_lims = x_avg_lims
        self.buffer_frac = buffer_frac
        self.consistency = {}
        if run is not None:
            RunDiagnostic.__init__(self, run)
            # One class, one cache per geometry: the flux-tube and x-global
            # schemas differ in dimension order and grouping, so they keep
            # separate files rather than one being misread as the other.
            default = _CACHE_FILES.get(self.geometry_kind)
            if outfile:
                self.outfile = outfile
            elif default:
                self.outfile = os.path.join(os.path.dirname(self.outfile),
                                            default)
            return
        self._legacy_init(outfile or self.cache_file, folder)

    def _legacy_init(self, outfile: str, folder: str = None):
        """
        Detached construction: HDF5 cache only, no run attached.

        Goes straight to the caching layer — ``super()`` is now
        :class:`RunDiagnostic`, whose constructor takes a run.
        """
        self.run = None
        self._cache = {}
        CachingDiagnostic.__init__(self, outfile, folder)

    # ------------------------------------------------------------------
    # Synchronisation
    # ------------------------------------------------------------------

    def sync_indices(self, fld_reader, mom_readers, t_start, t_stop, params):
        """
        Return time-aligned, not-yet-cached field and moment indices.

        Delegates to :meth:`CachingDiagnostic._sync_field_mom_indices`, which
        pairs the two files by *time value*. The previous implementation here
        derived strides from ``istep_field``/``istep_mom`` and sliced each index
        list independently, which assumes both files start at the same time and
        are complete. When they are not — different cadences, or a run killed
        mid-write — the two lists come out different lengths and the streaming
        loop exhausts the moment iterators, surfacing as
        ``RuntimeError: generator raised StopIteration``.
        """
        return self._sync_field_mom_indices(fld_reader, mom_readers,
                                           t_start, t_stop, params)

    def compute_spectra(self, fields, moments, ky3, J_norm, Bfield, params,
                        ky_weight=None):
        """
        Compute ES and EM flux spectra for all species.

        Parameters
        ----------
        fields     : list of np.ndarray  (nx, nky, nz)
        moments    : list of list        one inner list per species, 9 moment arrays each
        ky3        : np.ndarray          (1, nky, 1) broadcast array, precomputed
        J_norm     : np.ndarray          (nz,) normalised Jacobian, precomputed
        Bfield     : np.ndarray          (nz,) equilibrium B field from geometry
        params     : dict
        ky_weight  : np.ndarray          (nky,) precomputed [1,2,2,...,2]

        Returns
        -------
        list of [Q_es, Q_em, G_es, G_em] per species
        """
        species  = params["species"]
        n_fields = params["info"]["n_fields"]

        results = []
        for i_sp in range(len(species)):
            sp         = species[i_sp]
            mom        = moments[i_sp]
            n0, T0, q0 = sp['dens'], sp['temp'], sp['charge']

            G_es = self.averages(-1j*ky3*fields[0] * np.conj(mom[0])*n0, J_norm, ky_weight)
            Q_es = self.averages(-1j*ky3*fields[0] * np.conj(0.5*mom[1]+mom[2]+1.5*mom[0])*n0*T0, J_norm, ky_weight)

            if n_fields > 1:
                B_x  = 1j*ky3*fields[1]
                tmp1 = B_x * np.conj(mom[5])*n0
                tmp2 = B_x * np.conj(mom[3]+mom[4])*n0*T0
                if n_fields > 2:
                    dBpar_dy = -1j*ky3*fields[2] / Bfield[np.newaxis, np.newaxis, :]
                    tmp1 += dBpar_dy * np.conj(mom[6])*n0*T0/q0
                    tmp2 += dBpar_dy * np.conj(mom[7]+mom[8])*n0*T0**2/q0
                G_em = self.averages(tmp1, J_norm, ky_weight)
                Q_em = self.averages(tmp2, J_norm, ky_weight)
            else:
                G_em = Q_em = (None, None, None)

            results.append([Q_es, Q_em, G_es, G_em])

        return results

    @staticmethod
    def averages(flux, J_norm, ky_weight=None):
        """
        Compute kx spectrum, ky spectrum, and z profile of a flux array.

        Parameters
        ----------
        flux      : np.ndarray  (nx, nky, nz) complex
        J_norm    : np.ndarray  (nz,) normalised Jacobian, precomputed
        ky_weight : np.ndarray  (nky,) precomputed weight [1,2,2,...,2]

        Returns
        -------
        sp_kx : np.ndarray  (nx//2+1,)
        sp_ky : np.ndarray  (nky,)
        sum_z : np.ndarray  (nz,)
        """
        if flux is None:
            return (None, None, None)
        # Apply ky weight (1 for ky=0, 2 for ky>0) without copying
        W = ky_weight[np.newaxis, :, np.newaxis] if ky_weight is not None else 1.0
        J = J_norm[np.newaxis, np.newaxis, :]
        weighted = flux.real * (W * J)
        sum_z  = np.sum(weighted, axis=(0, 1))
        avg_z  = np.sum(weighted, axis=2)
        sp_ky  = np.sum(avg_z, axis=0)
        tmp    = np.sum(avg_z, axis=1)
        nx     = tmp.shape[0]
        nx2    = nx // 2 + 1
        sp_kx  = np.zeros(nx2)
        sp_kx[0] = tmp[0]
        if nx > 1:
            if nx % 2 == 1:
                sp_kx[1:nx2]   = tmp[1:nx2] + tmp[-1:nx2-1:-1]
            else:
                sp_kx[1:nx2-1] = tmp[1:nx2-1] + tmp[-1:nx2-1:-1]
                sp_kx[nx2-1]   = tmp[nx2-1]
        return sp_kx, sp_ky, sum_z

    # ------------------------------------------------------------------
    # HDF5 helpers — opened once per compute_missing call
    # ------------------------------------------------------------------

    @staticmethod
    def _init_h5(f, coords, species_names, fluxes, n_alloc=1,
                 time_dtype=np.float64):
        """Create all datasets in a newly opened HDF5 file, pre-allocated."""
        f.create_dataset("kx",   data=coords["kx"])
        f.create_dataset("ky",   data=coords["ky"])
        f.create_dataset("z",    data=coords["z"])
        f.create_dataset("time", shape=(n_alloc,), dtype=time_dtype,
                         maxshape=(None,), chunks=True)
        for i_sp, name in enumerate(species_names):
            Q_es, Q_em, G_es, G_em = fluxes[i_sp]
            for label, flux in zip(["Q_es", "Q_em", "Gamma_es", "Gamma_em"],
                                   [Q_es,    Q_em,   G_es,   G_em]):
                for axis_name, arr in zip(["kx", "ky", "z"], flux):
                    dsname = f"{name}_{label}_{axis_name}"
                    if arr is None:
                        f.create_dataset(dsname, shape=(n_alloc, 0),
                                         maxshape=(None, None), chunks=True)
                    else:
                        f.create_dataset(dsname, shape=(n_alloc, arr.size),
                                         maxshape=(None, arr.size), chunks=True)

    @staticmethod
    def _write_to_open_file(f, fluxes, species_names, time_value, row_idx):
        """Write one time step at *row_idx* in an already-open HDF5 file."""
        n_current = f["time"].shape[0]
        if row_idx >= n_current:
            new_size = row_idx + 1
            f["time"].resize((new_size,))
            for i_sp, name in enumerate(species_names):
                Q_es, Q_em, G_es, G_em = fluxes[i_sp]
                for label, flux in zip(["Q_es", "Q_em", "Gamma_es", "Gamma_em"],
                                       [Q_es,    Q_em,   G_es,   G_em]):
                    for axis_name, arr in zip(["kx", "ky", "z"], flux):
                        dsname = f"{name}_{label}_{axis_name}"
                        ds = f[dsname]
                        if arr is None:
                            ds.resize((new_size, 0))
                        else:
                            ds.resize((new_size, ds.shape[1]))

        f["time"][row_idx] = time_value   # cast to the dataset's dtype by h5py
        for i_sp, name in enumerate(species_names):
            Q_es, Q_em, G_es, G_em = fluxes[i_sp]
            for label, flux in zip(["Q_es", "Q_em", "Gamma_es", "Gamma_em"],
                                   [Q_es,    Q_em,   G_es,   G_em]):
                for axis_name, arr in zip(["kx", "ky", "z"], flux):
                    dsname = f"{name}_{label}_{axis_name}"
                    if arr is not None:
                        f[dsname][row_idx, :] = np.asarray(arr, float)

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def _migrate_cache(self) -> None:
        """
        Migrate an existing cache's legacy flux names, in place.

        Done before the "is anything missing?" check, not inside the write
        block: when every requested step is already cached the writer returns
        early, so a legacy cache would sit unmigrated until the first append —
        which is exactly when the rename becomes mandatory, and by then `time`
        has already been extended.
        """
        if not os.path.exists(self.outfile):
            return
        with h5py.File(self.outfile, "a") as f:
            self._migrate_legacy_names(f)

    def compute_missing(self, fld_reader, mom_readers, coords, geom,
                        params_list, t_start, t_stop):
        """
        Compute and cache spectra for all missing time steps.

        Parameters
        ----------
        fld_reader   : single field reader (BinaryReader or MultiSegmentReader)
        mom_readers  : list of readers, one per species
        coords       : dict from Coordinates()
        geom         : dict from Geometry()
        params_list  : Params object, list of dicts, or single dict
        t_start, t_stop : float
        """
        if hasattr(params_list, 'tolist'):
            params_list = params_list.tolist()
        elif isinstance(params_list, dict):
            params_list = [params_list]

        params        = params_list[0]
        species_names = [sp['name'] for sp in params['species']]

        # Precompute invariants once outside the time loop
        ky_arr = coords["ky"]
        ky3    = ky_arr[np.newaxis, :, np.newaxis]                # (1, nky, 1)
        J_norm = geom['Jacobian'] / np.sum(geom['Jacobian'])      # (nz,)
        Bfield = geom['Bfield']                                    # (nz,)

        # ky weight: 1 for ky=0, 2 for ky>0 (one-sided spectrum)
        ky_weight = np.ones(len(ky_arr))
        ky_weight[1:] = 2.0

        self._migrate_cache()
        idx_fld, idx_mom = self.sync_indices(fld_reader, mom_readers,
                                             t_start, t_stop, params)

        if len(idx_fld) == 0 or len(idx_mom) == 0:
            return

        n_missing = len(idx_fld)
        it_field = fld_reader.stream_selected(idx_fld)
        it_moms  = [r.stream_selected(idx_mom) for r in mom_readers]

        # Open HDF5 once for all time steps
        with h5py.File(self.outfile, "a") as hf:
            initialised = "time" in hf
            # Track write position: start after existing data
            write_idx = hf["time"].shape[0] if initialised else 0

            # Reuse a single executor for moment file reading
            n_mom_readers = len(it_moms)
            use_executor = n_mom_readers > 1
            executor = ThreadPoolExecutor(max_workers=n_mom_readers) if use_executor else None

            try:
                for tm, fields in it_field:
                    # Read all species moment files in parallel
                    if not use_executor:
                        moments_data = [next(it_moms[0])[1]]
                    else:
                        moments_data = [r[1] for r in executor.map(next, it_moms)]

                    fluxes = self.compute_spectra(fields, moments_data,
                                                  ky3, J_norm, Bfield, params,
                                                  ky_weight)

                    if not initialised:
                        self._init_h5(hf, coords, species_names, fluxes,
                                      n_alloc=n_missing,
                                      time_dtype=self._time_dtype(params))
                        initialised = True

                    self._write_to_open_file(hf, fluxes, species_names, tm,
                                             write_idx)
                    write_idx += 1
            finally:
                if executor is not None:
                    executor.shutdown(wait=False)

    def load_time_average(self, t_start=None, t_stop=None):
        if not os.path.exists(self.outfile):
            return {}
        with h5py.File(self.outfile, "r") as f:
            time, read_idx, unsort = self._select_window(
                f["time"][...], t_start, t_stop)

            flux_avg = {}
            for key in f.keys():
                if key in ("time", "kx", "ky", "z"):
                    continue
                data = f[key][read_idx][unsort]
                name = self._current_name(key)
                if len(time) <= 1:
                    flux_avg[name] = data[0] if len(time) == 1 else data
                else:
                    flux_avg[name] = _trapz(data, x=time, axis=0) / (time[-1] - time[0])
        return flux_avg

    def dataset(self, t=None, params=None, species=None, t_start=None,
                t_stop=None):
        """
        Return the flux spectra as an :class:`xarray.Dataset`.

        Called with a time window when bound to a run; the older
        ``dataset(coords, params, species, ...)`` form still works.
        """
        if self.run is not None and not isinstance(t, dict):
            if self.is_3d:
                return self._dataset_3d(t)
            self.compute(t)
            lo, hi = self._window(t)
            if self.geometry_kind == "x_global":
                return self._dataset_global(self.coord, self.params,
                                            self.run.species, lo, hi)
            return self._dataset_from_cache(self.coord, self.params,
                                            self.run.species, lo, hi)
        return self._dataset_from_cache(t, params, species, t_start, t_stop)

    def _dataset_from_cache(self, coords, params, species, t_start=None,
                            t_stop=None):
        """Return the time-averaged flux spectra as an ``xarray.Dataset``."""
        import xarray as xr
        from genetools import _xr

        raw = self.load_time_average(t_start, t_stop)
        if not raw:
            return xr.Dataset()

        def dim_of(var):                       # "Q_es_kx" -> ("kx",)
            return (var.rsplit("_", 1)[1],)

        data_vars, used = _xr.stacked_vars(raw, species, dim_of)
        candidates = {
            "kx": np.asarray(coords.get("kx_2", coords.get("kx", []))),
            "ky": np.asarray(coords.get("ky", [])),
            "z":  np.asarray(coords.get("z", [])),
        }
        return _xr.make_dataset(data_vars, candidates, species=used, params=params)

    # ------------------------------------------------------------------
    # Run-native front end
    # ------------------------------------------------------------------

    def compute(self, t=None):
        """
        Stream the data and cache the flux spectra.

        Flux tubes append kx/ky/z spectra to the HDF5 cache. x-global runs build
        ``(x, ky)`` maps into their own cache, whose schema differs enough to
        keep a separate file. GENE-3D builds the same ``(x, ky)`` maps in memory
        and cross-checks them against the fluxes the code wrote itself.
        """
        key = (self._key(t),
               tuple(self.x_avg_lims) if self.x_avg_lims else None)
        if self.is_3d:
            if key not in self._cache:
                self._cache[key] = self._compute_3d(t)
            return self._cache[key]
        lo, hi = self._bounds(t)
        r = self.run
        moms = [r.mom(n) for n in r.species]
        if self.geometry_kind == "x_global":
            self._compute_global(
                r.field, moms, self.coord, self.geom, self.params, lo, hi,
                equilibrium_profiles=r.eq_profiles)
        else:
            self.compute_missing(r.field, moms, self.coord, self.geom,
                                 r.params, lo, hi)
        return self

    def _background(self, species):
        """
        Return ``(n_0, T_0)`` normalised to this species' namelist values.

        The heat flux needs the background profiles: GENE-3D's moments are
        perturbations about ``n_0(x)`` and ``T_0(x)``, and the profile files
        store them in keV / 1e19 m^-3, so both are divided back down by the
        species factor and the reference value.
        """
        params = self.params
        spec = next(s for s in params["species"] if s["name"] == species)
        units = params["units"]
        prof = self.run.eq_profiles[species]
        T0 = (np.asarray(prof["T"], dtype=float)
              / (float(spec.get("temp", 1.0)) * float(units["Tref"])))
        n0 = (np.asarray(prof["n"], dtype=float)
              / (float(spec.get("dens", 1.0)) * float(units["nref"])))
        return n0, T0

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _compute_3d(self, t):
        """Stream field and moments, returning per-species ky spectra."""
        run = self.run
        J = self.geom["Jacobian"]
        geomfac = g3.flux_geomfac(self.geom, self.params)
        coord = self.coord
        ky = np.asarray(coord["ky"], dtype=float)
        xsl = g3.radial_slice(coord["x_o_a"], limits=self.x_avg_lims,
                             buffer_frac=self.buffer_frac)

        fld = run.field
        lo, hi = self._bounds(t)

        mom_readers = {n: run.mom(n) for n in run.species}
        # istep_field and istep_mom need not agree, so a flux built from both
        # files may only be evaluated where the two coincide. The shared
        # helper pairs them by time value, never by position.
        idx_f, idx_m = self._sync_field_mom_indices(
            fld, list(mom_readers.values()), lo, hi, self.params)
        if not idx_f:
            raise ValueError(
                "No time at which both field and moment output exist inside "
                "the requested window.")
        common = {"times": np.asarray(fld.read_all_times())[idx_f],
                  "field": idx_f}
        common.update({n: idx_m for n in mom_readers})

        i_phi = fld.index_of("phi")
        i_apar = fld.index_of("A_par") if g3.has_var(fld, "A_par") else None

        spectra = {n: {} for n in run.species}
        code_flux = {n: {} for n in run.species}
        for name in run.species:
            n0, T0 = self._background(name)
            reader = mom_readers[name]
            wanted = [v for v in self._ES_FLUXES + self._EM_FLUXES
                      if g3.has_var(reader, v)]
            acc = {v: [] for v in wanted}
            ref = {v: [] for v in wanted}

            stream_f = fld.stream_selected(common["field"])
            stream_m = reader.stream_selected(common[name])
            for (_, f_arrays), (_, m_arrays) in zip(stream_f, stream_m):
                phi = f_arrays[i_phi]
                v_E = g3.exb_velocity_ky(phi, ky, geomfac)
                b_x = (g3.flutter_velocity_ky(f_arrays[i_apar], ky, geomfac)
                       if i_apar is not None else None)

                fluxes = self._fluxes_from_moments(
                    reader, m_arrays, v_E, b_x, n0, T0, wanted)
                for v in wanted:
                    # Keep the radial axis: the (x, ky) map is the primary
                    # product and both 1-D views are reductions of it. Averaging
                    # x away here, as this used to, threw the radial structure
                    # of the spectrum out before anyone could look at it.
                    acc[v].append(g3.z_average_ky(fluxes[v], J))
                    # The reference has to be reduced identically, or the
                    # comparison measures the difference between two averaging
                    # weights rather than a normalisation error: summing the
                    # spectrum over ky is a Jacobian-weighted average over
                    # x, y and z, so the code's own flux gets the same.
                    ref[v].append(np.average(g3.pick(reader, m_arrays, v)[xsl],
                                             weights=J[xsl]))

            for v in wanted:
                spectra[name][v] = self._time_average(
                    np.asarray(acc[v]), common["times"])
                code_flux[name][v] = self._time_average(
                    np.asarray(ref[v]), common["times"])

        result = {"ky": ky, "times": common["times"], "spectra": spectra,
                  "code_flux": code_flux, "xslice": xsl,
                  "x_weights": g3.radial_weights(J)}
        self._check(result)
        return result

    @staticmethod
    def _reduce_x(spectrum, weights, xsl):
        """
        Radially average an ``(x, ky)`` map over the retained window.

        Weighted by each surface's ``sum_z J``, which makes this exactly the
        joint x-z average :func:`~genetools.diagnostics._gene3d.xz_average`
        performs in one step -- so the consistency check below still compares
        like with like.
        """
        w = np.asarray(weights)[xsl]
        return (np.asarray(spectrum)[xsl] * w).sum(axis=0) / w.sum(axis=0)

    def _fluxes_from_moments(self, reader, arrays, v_E, b_x, n0, T0, wanted):
        """
        Build the complex per-mode flux densities from one snapshot.

        Both integrands follow from GENE-3D's own moment slots rather than
        being assumed. Its heat flux is built from ``momc(5) = mat_20 + mat_01``
        after the FLR corrections; undoing the post-processing that turns those
        into the written ``T_par``/``T_per``/``n`` leaves

            n_0 (T_par/2 + T_per) + 3/2 T_0 n

        which is the integrand used here. Its electromagnetic heat flux uses
        ``momc(6) = mat_30 + mat_11``, and those are written verbatim as
        ``q_par`` and ``q_perp`` — so ``Q_em`` is an exact identity in the data
        on disk. The reference GUI omits both from its variable map and
        therefore reports ``Q_em = 0`` for every GENE-3D run.
        """
        nx = n0[:, np.newaxis, np.newaxis]
        tx = T0[:, np.newaxis, np.newaxis]

        dens = g3.to_ky(g3.pick(reader, arrays, "n"))
        t_par = g3.to_ky(g3.pick(reader, arrays, "T_par"))
        t_perp = g3.to_ky(g3.pick(reader, arrays, "T_per"))

        out = {}
        if "Gamma_es" in wanted:
            out["Gamma_es"] = np.conj(v_E) * dens
        if "Q_es" in wanted:
            integrand = (0.5 * t_par + t_perp) * nx + 1.5 * dens * tx
            out["Q_es"] = np.conj(v_E) * integrand

        if b_x is not None:
            if "Gamma_em" in wanted:
                u_par = g3.to_ky(g3.pick(reader, arrays, "u_par"))
                out["Gamma_em"] = np.conj(b_x) * u_par
            if "Q_em" in wanted:
                if g3.has_var(reader, "q_par") and g3.has_var(reader, "q_perp"):
                    q_tot = (g3.to_ky(g3.pick(reader, arrays, "q_par"))
                             + g3.to_ky(g3.pick(reader, arrays, "q_perp")))
                    out["Q_em"] = np.conj(b_x) * q_tot
                else:
                    out["Q_em"] = np.zeros_like(out.get("Gamma_em", dens))
        return out

    def _check(self, result):
        """Compare each ky-summed spectrum against the code's own flux."""
        self.consistency = {}
        for name, per in result["spectra"].items():
            for v, spectrum in per.items():
                # Reduce the map the same way the reference was reduced: over
                # the retained radial window, then summed over ky.
                reduced = self._reduce_x(spectrum, result["x_weights"],
                                         result["xslice"])
                ratio = g3.check_flux_consistency(
                    np.sum(reduced), result["code_flux"][name][v],
                    f"{name} {v}")
                self.consistency[f"{name}/{v}"] = ratio


    # ------------------------------------------------------------------
    # GENE-3D dataset and plot
    # ------------------------------------------------------------------

    def _dataset_3d(self, t):
        """
        The GENE-3D spectra as ``(x, ky)`` maps plus their two reductions.

        Same three products as the x-global path, so a script does not have to
        know which global geometry it is looking at: ``*_xky`` is the map,
        ``*_x`` the flux profile summed over ky, and ``*_ky`` the ky spectrum
        averaged over the retained radial window.
        """
        from genetools._xr import make_dataset, unit_attrs
        raw = self.compute(t)
        params = self.params
        names = list(self.run.species)
        present = [v for v in self._ES_FLUXES + self._EM_FLUXES
                   if all(v in raw["spectra"][n] for n in names)]
        xsl, weights = raw["xslice"], raw["x_weights"]

        data_vars = {}
        for v in present:
            maps = [np.real(raw["spectra"][n][v]) for n in names]
            data_vars[v + "_xky"] = (("species", "x", "ky"),
                                     np.stack(maps, axis=0))
            data_vars[v + "_x"] = (("species", "x"),
                                   np.stack([m.sum(axis=1) for m in maps],
                                            axis=0))
            data_vars[v + "_ky"] = (
                ("species", "ky"),
                np.stack([self._reduce_x(m, weights, xsl) for m in maps],
                         axis=0))
        ds = make_dataset(
            data_vars,
            {"x": np.asarray(self.coord["x_o_a"], dtype=float),
             "ky": raw["ky"]}, species=names, params=params)
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = self.geometry_kind
        x = np.asarray(self.coord["x_o_a"], dtype=float)[xsl]
        ds.attrs["x_avg_range"] = [float(x[0]), float(x[-1])]
        ds.attrs["n_times"] = int(np.size(raw["times"]))
        for label, ratio in self.consistency.items():
            ds.attrs["consistency_" + label.replace("/", "_")] = float(ratio)
        return ds

    def _plot_3d(self, t, which=None):
        """
        The ``(x, ky)`` maps by default; ``which`` adds or swaps the 1-D views.

        Mirrors the x-global layout, so the two global geometries are read the
        same way.
        """
        views = self._views(which)
        ds = self._dataset_3d(t)
        bases = [k[:-4] for k in ds.data_vars if k.endswith("_xky")]
        if not bases:
            raise ValueError("No flux spectra available to plot.")
        lo, hi = ds.attrs["x_avg_range"]
        window = rf"$x/a \in [{lo:.2f}, {hi:.2f}]$"
        builders = {"map": lambda: self._fig_3d_maps(ds, bases),
                    "ky": lambda: self._fig_3d_ky(ds, bases, window),
                    "profile": lambda: self._fig_3d_profiles(ds, bases)}
        figs = [builders[v]() for v in views]
        plt.show()
        return figs

    #: Views `plot` can draw for the global geometries.
    _VIEWS = ("map", "ky", "profile")
    #: Drawn when `which` is not given: the 2-D map, nothing averaged away.
    _DEFAULT_VIEW = ("map",)

    @classmethod
    def _views(cls, which):
        """Normalise the ``which`` argument to a validated tuple of views."""
        if which is None:
            return cls._DEFAULT_VIEW
        if which == "all":
            return cls._VIEWS
        views = (which,) if isinstance(which, str) else tuple(which)
        bad = [v for v in views if v not in cls._VIEWS]
        if bad:
            raise ValueError(
                f"unknown spectra view(s) {bad}; expected any of "
                f"{list(cls._VIEWS)} or 'all'")
        return views

    @staticmethod
    def _positive_ky(ds):
        """Indices of the non-negative half of a full FFT ky axis."""
        ky_full = np.asarray(ds["ky"])
        n_pos = (ky_full.size + 1) // 2
        return ky_full[:n_pos], n_pos

    def _fig_3d_maps(self, ds, bases):
        """One ``(x, ky)`` colour map per flux and species."""
        ky, n_pos = self._positive_ky(ds)
        names = list(ds["species"].values)
        fig, axes = plt.subplots(len(bases), len(names),
                                 figsize=(5.0 * len(names), 3.4 * len(bases)),
                                 squeeze=False)
        x = np.asarray(ds["x"])
        for row, base in enumerate(bases):
            for col, name in enumerate(names):
                ax = axes[row][col]
                arr = np.asarray(ds[base + "_xky"].sel(species=name))[:, :n_pos]
                vmax = float(np.max(np.abs(arr))) or 1.0
                mesh = ax.pcolormesh(x, ky, arr.T, shading="auto", cmap="bwr",
                                     vmin=-vmax, vmax=vmax)
                ax.set_xlabel(r"$x/a$")
                ax.set_ylabel(r"$k_y \rho_{\rm ref}$")
                ax.set_title(f"{self._TITLES_3D.get(base, base)}  {name}",
                             fontsize=9)
                fig.colorbar(mesh, ax=ax)
        fig.suptitle("GENE-3D flux spectra — (x, ky) maps")
        fig.tight_layout()
        return fig

    def _fig_3d_ky(self, ds, bases, window):
        """Three views of each radially averaged ky spectrum."""
        ky, n_pos = self._positive_ky(ds)
        fig, axes = plt.subplots(len(bases), 3,
                                 figsize=(13, 3.2 * len(bases)), squeeze=False)
        for row, base in enumerate(bases):
            label = self._TITLES_3D.get(base, base)
            for name in ds["species"].values:
                vals = np.asarray(ds[base + "_ky"].sel(species=name))[:n_pos]
                axes[row][0].plot(ky, np.abs(vals), label=str(name))
                axes[row][1].plot(ky, vals * ky, label=str(name))
                axes[row][2].plot(ky, vals, label=str(name))
            axes[row][0].loglog()
            axes[row][0].set_ylabel("|" + label + "|")
            axes[row][1].set_xscale("log")
            axes[row][1].set_ylabel(label + r"$\,k_y$")
            axes[row][2].set_ylabel(label)
            for ax in axes[row]:
                ax.set_xlabel(r"$k_y \rho_{\rm ref}$")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
        fig.suptitle("GENE-3D flux spectra, " + window)
        fig.tight_layout()
        return fig

    def _fig_3d_profiles(self, ds, bases):
        """The ky-summed flux profile against radius."""
        x = np.asarray(ds["x"])
        fig, axes = plt.subplots(1, len(bases),
                                 figsize=(4.6 * len(bases), 3.8), squeeze=False)
        for ax, base in zip(axes[0], bases):
            for name in ds["species"].values:
                ax.plot(x, np.asarray(ds[base + "_x"].sel(species=name)),
                        label=str(name))
            ax.set_xlabel(r"$x/a$")
            ax.set_ylabel(self._TITLES_3D.get(base, base))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(r"GENE-3D flux profiles, $\sum_{k_y}$")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------

    def plot(self, t=None, which=None, x_avg_lims=None, **kw):
        """
        Plot the flux spectra over the window *t*.

        Parameters
        ----------
        t : (float, float), optional
            Time window.
        which : str or sequence of str, optional
            Global geometries only — which views to draw:

            - ``'map'``     — the ``(x, ky)`` colour maps (**default**)
            - ``'ky'``      — the radially averaged ky spectra
            - ``'profile'`` — the ky-summed flux profile against x
            - ``'all'``     — all three

            A flux tube has no ``(x, ky)`` map, since x is spectral there;
            passing *which* on one raises rather than quietly ignoring it.
        x_avg_lims : (float, float), optional
            Radial range for the ``ky`` view.
        """
        if which is not None and not self.is_3d \
                and self.geometry_kind != "x_global":
            raise ValueError(
                f"'which' applies to the global geometries; this run is "
                f"{self.geometry_kind}, whose spectra are kx/ky/z and have no "
                "(x, ky) map — x is spectral.")
        if self.is_3d:
            return self._plot_3d(t, which=which)
        lo, hi = self._bounds(t)
        r = self.run
        if self.geometry_kind == "x_global":
            self.compute(t)
            return self._plot_global(self.coord, self.params, lo, hi,
                                     which=which,
                                     x_avg_lims=x_avg_lims, **kw)
        return self._plot_local(r.field, [r.mom(n) for n in r.species],
                                r.coords, r.geometry, r.params, lo, hi, **kw)

    def _plot_local(self, fld_reader, mom_readers, coords, geom, params_list,
             t_start, t_stop):
        if hasattr(params_list, 'tolist'):
            params_list = params_list.tolist()
        elif isinstance(params_list, dict):
            params_list = [params_list]

        self.compute_missing(fld_reader, mom_readers, coords[0], geom[0],
                             params_list, t_start, t_stop)
        flux_avg = self.load_time_average(t_start, t_stop)
        if not flux_avg:
            print("No spectra available to plot.")
            return

        with h5py.File(self.outfile, "r") as f:
            kx            = f["kx"][...]
            ky            = f["ky"][...]
            z             = f["z"][...]
            # Extract species names by stripping known flux suffixes
            _suffixes = tuple(
                f"_{flux}_{axis}"
                for flux in ("Q_es", "Q_em", "Gamma_es", "Gamma_em")
                for axis in ("kx", "ky", "z"))
            species_set = set()
            for name in f.keys():
                for sfx in _suffixes:
                    if name.endswith(sfx):
                        species_set.add(name[:-len(sfx)])
                        break
            species_names = sorted(species_set)

        nx2     = len(kx) // 2 + 1
        kx_half = kx[:nx2]
        labels  = ["Q_es", "Q_em", "Gamma_es", "Gamma_em"]

        def _get(key):
            """Return array if present and non-empty, else None."""
            arr = flux_avg.get(key)
            return arr if arr is not None and arr.size > 0 else None

        # ── kx spectra: all fluxes on one graph, lin + log ────────────
        for sp in species_names:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"{sp} — spectra vs kx")
            for ax, scale in zip(axes, ("linear", "log")):
                for label in labels:
                    arr = _get(f"{sp}_{label}_kx")
                    if arr is not None:
                        ax.plot(kx_half[:len(arr)], arr if scale == "linear" else np.abs(arr), label=label)
                ax.set_xlabel("kx")
                ax.set_ylabel("Flux")
                ax.set_xscale(scale)
                ax.set_yscale(scale)
                ax.set_title("lin-lin" if scale == "linear" else "log-log")
                ax.legend()
                ax.grid(True)
            plt.tight_layout()
            plt.show()

        # ── ky spectra: all fluxes on one graph, lin + log ────────────
        for sp in species_names:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"{sp} — spectra vs ky")
            for ax, scale in zip(axes, ("linear", "log")):
                for label in labels:
                    arr = _get(f"{sp}_{label}_ky")
                    if arr is not None:
                        ax.plot(ky[:len(arr)], arr if scale == "linear" else np.abs(arr), label=label)
                ax.set_xlabel("ky")
                ax.set_ylabel("Flux")
                ax.set_xscale(scale)
                ax.set_yscale(scale)
                ax.set_title("lin-lin" if scale == "linear" else "log-log")
                ax.legend()
                ax.grid(True)
            plt.tight_layout()
            plt.show()

        # ── z profiles ────────────────────────────────────────────────
        fig, axes = plt.subplots(len(species_names), 1,
                                 figsize=(8, 3*len(species_names)),
                                 sharex=True, squeeze=False)
        for i, sp in enumerate(species_names):
            ax = axes[i, 0]
            for label in labels:
                arr = _get(f"{sp}_{label}_z")
                if arr is not None:
                    ax.plot(z[:len(arr)], arr, label=label)
            ax.set_ylabel("Flux")
            ax.set_title(sp)
            ax.legend()
            ax.grid(True)
        axes[-1, 0].set_xlabel("z/a")
        plt.tight_layout()
        plt.show()

        # ── Total flux: print to stdout + bar chart ───────────────────
        print(f"\n{'─'*60}")
        print(f"{'Species':<14} {'Flux':<8} {'Sum(kx)':>14} {'Sum(ky)':>14}")
        print("─" * 60)
        all_totals = {}   # (sp, label) -> (total_kx, total_ky)
        for sp in species_names:
            for label in labels:
                tkx = np.sum(flux_avg[f"{sp}_{label}_kx"]) \
                      if _get(f"{sp}_{label}_kx") is not None else 0.0
                tky = np.sum(flux_avg[f"{sp}_{label}_ky"]) \
                      if _get(f"{sp}_{label}_ky") is not None else 0.0
                all_totals[(sp, label)] = (tkx, tky)
                print(f"{sp:<14} {label:<8} {tkx:>14.6g} {tky:>14.6g}")
        print("─" * 60)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].set_title("Total flux — sum over kx spectrum")
        axes[1].set_title("Total flux — sum over ky spectrum")
        x_pos = np.arange(len(labels))
        width = 0.8 / max(len(species_names), 1)
        for sp_i, sp in enumerate(species_names):
            offset    = (sp_i - (len(species_names) - 1) / 2) * width
            totals_kx = [all_totals[(sp, lb)][0] for lb in labels]
            totals_ky = [all_totals[(sp, lb)][1] for lb in labels]
            axes[0].bar(x_pos + offset, totals_kx, width=width, label=sp)
            axes[1].bar(x_pos + offset, totals_ky, width=width, label=sp)
        for ax in axes:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            ax.set_ylabel("Flux")
            ax.legend()
            ax.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # x-global: (x, ky) flux maps
    #
    # Absorbed from the former SpectraGlobal. The cache schema stays its own —
    # per-species groups holding (nx, nky, time) — but the flux names now match
    # the rest of the package (Gamma/Q/Pi), so the same physics has the same
    # name whatever the geometry.
    # ------------------------------------------------------------------

    #: Flux channels, electrostatic then electromagnetic, x-global only.
    _ES_GLOBAL = ("Q_es", "Gamma_es", "Pi_es")
    _EM_GLOBAL = ("Q_em", "Gamma_em", "Pi_em")

    _TITLES_GLOBAL = {
        "Q_es": r"$Q_{\rm es}$", "Gamma_es": r"$\Gamma_{\rm es}$",
        "Pi_es": r"$\Pi_{\rm es}$",
        "Q_em": r"$Q_{\rm em}$", "Gamma_em": r"$\Gamma_{\rm em}$",
        "Pi_em": r"$\Pi_{\rm em}$",
    }
    _COLORS_GLOBAL = {"Q_es": "b", "Gamma_es": "r", "Pi_es": "g",
                      "Q_em": "m", "Gamma_em": "k", "Pi_em": "c"}

    @staticmethod
    def _migrate_legacy_names(f) -> None:
        """
        Rename legacy flux datasets in an open, writable cache.

        Reading translates names on the fly, but appending needs the datasets
        themselves to carry the current names, or adding a time step to a cache
        written by an older version raises ``KeyError`` — after it has already
        extended ``time``. Both schemas are handled: the x-global one nests
        fluxes inside per-species groups, the flux-tube one is flat.
        """
        for name in list(f.keys()):                     # x-global groups
            grp = f[name]
            if not isinstance(grp, h5py.Group):
                continue
            for old, new in _LEGACY_FLUX_NAMES.items():
                if old in grp and new not in grp:
                    grp.move(old, new)
        for key in list(f.keys()):                      # flat flux-tube keys
            for old, new in _LEGACY_FLUX_INFIX.items():
                if old in key:
                    renamed = key.replace(old, new)
                    if renamed not in f:
                        f.move(key, renamed)
                    break

    @staticmethod
    def _current_name(key: str) -> str:
        """Translate one legacy HDF5 key to its current spelling."""
        if key in _LEGACY_FLUX_NAMES:
            return _LEGACY_FLUX_NAMES[key]
        for old, new in _LEGACY_FLUX_INFIX.items():
            if old in key:
                return key.replace(old, new)
        return key

    @staticmethod
    def _init_h5_global(f, species_names: list, nx: int, nky: int,
                        has_em: bool, keys, time_dtype=np.float64):
        """Create all datasets in a newly opened x-global cache file."""
        f.create_dataset("time", shape=(0,), maxshape=(None,),
                         dtype=time_dtype, chunks=True)
        for name in species_names:
            grp = f.create_group(name)
            for key in keys:
                grp.create_dataset(f"{key}_ky", shape=(nx, nky, 0),
                                   maxshape=(nx, nky, None),
                                   dtype=np.float64, chunks=True)

    @staticmethod
    def _append_global(f, species_names: list, species_data: dict,
                       time: float) -> None:
        """Append one time step to an already-open x-global cache file."""
        tds = f["time"]
        n = tds.shape[0]
        tds.resize((n + 1,))
        tds[n] = time
        for name in species_names:
            for key, val in species_data[name].items():
                ds = f[f"{name}/{key}"]
                ds.resize((ds.shape[0], ds.shape[1], n + 1))
                ds[:, :, n] = val

    def _compute_global(self, fld_reader, mom_readers, coords, geom, params,
                        t_start, t_stop, equilibrium_profiles=None) -> None:
        """
        Stream field and moment files and append ``(x, ky)`` flux maps to HDF5.

        x is already real space here, so no radial transform is involved: the
        flux is a per-ky cross-correlation, flux-surface averaged over z.
        """
        nx = params["box"]["nx0"]
        nky = params["box"]["nky0"]
        n_fields = params["info"]["n_fields"]
        species = params["species"]
        species_names = [sp["name"] for sp in species]
        ky = np.asarray(coords["ky"])
        has_em = n_fields > 1

        J = geom["Jacobian"]
        J_norm = J / J.sum(axis=1, keepdims=True)
        C_xy = geom["metric"]["C_xy"]

        units = params.get("units", {})
        Tref = units.get("Tref", 1.0)
        nref = units.get("nref", 1.0)
        nz = params["box"]["nz0"]

        prefactors = {}
        if equilibrium_profiles is not None:
            for sp in species:
                name = sp["name"]
                ep = equilibrium_profiles.get(name)
                if ep is None:
                    continue
                T0 = sp["temp"] * Tref
                n0 = sp["dens"] * nref
                T_map = (np.asarray(ep["T"]) / T0)[:, np.newaxis, np.newaxis] \
                    * np.ones((1, 1, nz))
                n_map = (np.asarray(ep["n"]) / n0)[:, np.newaxis, np.newaxis] \
                    * np.ones((1, 1, nz))
                prefactors[name] = {"n_map": n_map, "T_map": T_map}

        self._migrate_cache()
        idx_fld, idx_mom = self._sync_field_mom_indices(
            fld_reader, mom_readers, t_start, t_stop, params)
        if len(idx_fld) == 0 or len(idx_mom) == 0:
            return

        it_field = fld_reader.stream_selected(idx_fld)
        it_moms = [r.stream_selected(idx_mom) for r in mom_readers]
        keys = list(self._ES_GLOBAL) + (list(self._EM_GLOBAL) if has_em else [])

        with h5py.File(self.outfile, "a") as hf:
            initialised = "time" in hf
            if initialised and any(n not in hf for n in species_names):
                raise ValueError(
                    f"Cache '{self.outfile}' has a time axis but no data group "
                    f"for every species {species_names} — it was written by an "
                    "interrupted run or a different configuration. Delete it "
                    "and recompute.")

            for tm, fields in it_field:
                all_moments = []
                for it_m in it_moms:
                    _, moms = next(it_m)
                    all_moments.append(moms)

                phi = fields[0]
                ky3 = ky[np.newaxis, :, np.newaxis]
                v_E = -1j * ky3 * phi
                B_par = 1j * ky3 * fields[1] if has_em else None

                sp_data = {}
                for i_sp, sp in enumerate(species):
                    name = sp["name"]
                    n0 = sp["dens"]
                    T0 = sp["temp"]
                    mass = sp.get("mass", 1.0)
                    moments = all_moments[i_sp]

                    pf = prefactors.get(name, {})
                    n_map = pf.get("n_map", 1.0)
                    T_map = pf.get("T_map", 1.0)

                    dens = moments[0]
                    T_par = moments[1]
                    T_perp = moments[2]
                    u_par = moments[5]

                    tmp_q = (0.5 * T_par + T_perp) * n_map \
                        + 1.5 * dens * T_map
                    result = {
                        "Gamma_es_ky": n0 * _compute_flux_yspectra(
                            dens, v_E, C_xy, J_norm),
                        "Q_es_ky": n0 * T0 * _compute_flux_yspectra(
                            tmp_q, v_E, C_xy, J_norm),
                        "Pi_es_ky": n0 * mass * _compute_flux_yspectra(
                            v_E, u_par * n_map, C_xy, J_norm),
                    }

                    if B_par is not None:
                        q_par = moments[3]
                        q_perp = moments[4]
                        result.update({
                            "Gamma_em_ky": n0 * _compute_flux_yspectra(
                                u_par * n_map, B_par, C_xy, J_norm),
                            "Q_em_ky": n0 * T0 * _compute_flux_yspectra(
                                q_par + q_perp, B_par, C_xy, J_norm),
                            "Pi_em_ky": n0 * T0 * _compute_flux_yspectra(
                                B_par,
                                (T_par * n_map + dens * T_map) * n_map,
                                C_xy, J_norm),
                        })

                    sp_data[name] = result

                if not initialised:
                    self._init_h5_global(hf, species_names, nx, nky, has_em,
                                         keys,
                                         time_dtype=self._time_dtype(params))
                    initialised = True
                self._append_global(hf, species_names, sp_data, tm)

    def _load_global(self, t_start=None, t_stop=None) -> dict:
        """
        Load saved ``(x, ky)`` spectra, keyed ``'{species}_{flux}_ky'``.

        Arrays come back shaped ``(n_times, nx, nky)``. Legacy flux names in an
        older cache are translated here, so it stays readable.
        """
        if not os.path.exists(self.outfile):
            return {}
        with h5py.File(self.outfile, "r") as f:
            if "time" not in f:                     # partially written cache
                return {}
            time = f["time"][...]
            if time.size == 0:
                return {}
            time, read_idx, unsort = self._select_window(time, t_start, t_stop)
            result = {"time": time}
            for name in [k for k in f.keys() if k != "time"]:
                grp = f[name]
                for key in grp.keys():
                    data = grp[key][:, :, read_idx][:, :, unsort]
                    result[f"{name}_{self._current_name(key)}"] = \
                        np.transpose(data, (2, 0, 1))
        return result

    def _load_time_average_global(self, t_start=None, t_stop=None) -> dict:
        """Time-average the cached ``(x, ky)`` maps; values are ``(nx, nky)``."""
        data = self._load_global(t_start, t_stop)
        if not data or "time" not in data:
            return {}
        time = data["time"]
        if len(time) == 0:                          # window selects nothing
            return {}
        out = {}
        for key, arr in data.items():
            if key == "time":
                continue
            out[key] = (arr[0] if len(time) <= 1
                        else self._time_average(arr, time))
        return out

    def _radial_window(self, x):
        """
        Index slice of the radial range a ky spectrum is averaged over.

        Defaults to trimming ``buffer_frac`` from each end so the Krook buffer
        regions, where the fluxes are unphysical, stay out of the average — the
        same rule the GENE-3D path uses.
        """
        return g3.radial_slice(x, limits=self.x_avg_lims,
                               buffer_frac=self.buffer_frac)

    def _expand_reductions(self, raw: dict, xsl) -> dict:
        """
        Expand each ``(nx, nky)`` map into the map plus its two reductions.

        ``'ions_Q_es_ky'`` becomes ``'ions_Q_es_xky'`` (x, ky), ``'ions_Q_es_x'``
        — the flux profile summed over ky — and ``'ions_Q_es_ky'``, the ky
        spectrum averaged over the retained radial window. Same three products
        the GENE-3D path builds, so a script need not know which global geometry
        it has. (Unweighted in x here: the z average already carried the
        Jacobian, and this cache stores no geometry to weight with.)
        """
        out = {}
        for key, arr in raw.items():
            arr = np.asarray(arr)
            base = key[:-3] if key.endswith("_ky") else key
            if arr.ndim != 2:
                out[key] = arr
                continue
            out[f"{base}_xky"] = arr
            out[f"{base}_x"] = arr.sum(axis=1)
            out[f"{base}_ky"] = arr[xsl].mean(axis=0)
        return out

    def _dataset_global(self, coords, params, species, t_start=None,
                        t_stop=None):
        """The x-global spectra as ``(x, ky)`` maps plus their reductions."""
        import xarray as xr
        from genetools import _xr

        raw = self._load_time_average_global(t_start, t_stop)
        if not raw:
            return xr.Dataset()
        x = np.asarray(coords.get("x", []))
        if x.size == 0:
            x = np.asarray(coords.get("x_o_a", []))
        xsl = self._radial_window(x)
        data_vars, used = _xr.stacked_vars(
            self._expand_reductions(raw, xsl), species, _xr.dims_from_suffix)
        candidates = {"x": x, "ky": np.asarray(coords.get("ky", []))}
        ds = _xr.make_dataset(data_vars, candidates, species=used,
                              params=params)
        if x.size:
            ds.attrs["x_avg_range"] = [float(x[xsl][0]), float(x[xsl][-1])]
        return ds

    def _plot_global(self, coords, params, t_start=None, t_stop=None,
                     which=None, x_avg_lims=None):
        """
        The ``(x, ky)`` map by default; ``which`` selects the 1-D views too.

        ky weighting puts equal areas at equal flux contribution on a
        logarithmic ky axis, since ``int F dky = int ky F d(ln ky)``.
        """
        views = self._views(which)
        if x_avg_lims is not None:
            self.x_avg_lims = x_avg_lims
        ds = self._dataset_global(coords, params, self.run.species,
                                  t_start, t_stop)
        if not ds.data_vars:
            print("No global spectra available to plot.")
            return []

        x = np.asarray(ds["x"])
        bases = [k[:-4] for k in ds.data_vars if k.endswith("_xky")]
        order = [b for b in self._ES_GLOBAL + self._EM_GLOBAL if b in bases]
        lo, hi = ds.attrs.get("x_avg_range", (x[0], x[-1]))

        figs = []
        for name in ds["species"].values:
            if "map" in views:
                figs.append(self._fig_global_map(ds, name, order))
            lines = [v for v in ("ky", "profile") if v in views]
            if lines:
                figs.append(self._fig_global_lines(ds, name, order, lines,
                                                   lo, hi))
        plt.show()
        return figs

    @staticmethod
    def _drop_ky_zero(ky):
        """
        Return ``(ky_positive, mask)``, dropping the ky=0 column.

        It is a structural zero — the electrostatic fluxes are built from
        ``v_E = -i ky phi``, which vanishes there — so it carries no information
        and would force a linear region into an otherwise logarithmic axis.
        """
        mask = ky > 0
        return (ky[mask] if mask.any() else ky), mask

    def _fig_global_map(self, ds, name, order):
        """The ky-weighted ``(x, ky)`` map, one panel per flux."""
        x = np.asarray(ds["x"])
        ky = np.asarray(ds["ky"])
        ky_p, kpos = self._drop_ky_zero(ky)
        fig, axes = plt.subplots(1, len(order), figsize=(5 * len(order), 4),
                                 squeeze=False)
        fig.suptitle(f"{name} — ky-weighted flux maps")
        for ax, base in zip(axes[0], order):
            arr = np.asarray(ds[base + "_xky"].sel(species=name))
            weighted = (arr[:, kpos] * ky_p[np.newaxis, :] if kpos.any()
                        else arr * ky[np.newaxis, :])
            norm, cmap = _flux_norm_and_cmap(weighted)
            im = ax.pcolormesh(ky_p, x, weighted, shading="auto",
                               norm=norm, cmap=cmap)
            fig.colorbar(im, ax=ax)
            ax.set_xscale("log")
            ax.set_xlabel(r"$k_y \rho_{\rm ref}$")
            ax.set_ylabel(r"$x / a$")
            ax.set_title(_ky_weighted_label(
                self._TITLES_GLOBAL.get(base, base)))
        fig.tight_layout()
        return fig

    def _fig_global_lines(self, ds, name, order, lines, lo, hi):
        """The 1-D reductions: ky spectra and/or the radial flux profile."""
        x = np.asarray(ds["x"])
        ky = np.asarray(ds["ky"])
        # 'ky' contributes two panels (linear and log-log), 'profile' one.
        panels = (["ky_lin", "ky_log"] if "ky" in lines else []) \
            + (["profile"] if "profile" in lines else [])
        fig, axes = plt.subplots(1, len(panels), figsize=(5.3 * len(panels), 4),
                                 squeeze=False)
        by = dict(zip(panels, axes[0]))
        fig.suptitle(f"{name} — x-avg [{lo:.3f}, {hi:.3f}]")
        for base in order:
            colour = self._COLORS_GLOBAL.get(base, "b")
            label = self._TITLES_GLOBAL.get(base, base)
            if "ky" in lines:
                weighted = np.asarray(ds[base + "_ky"].sel(species=name)) * ky
                by["ky_lin"].plot(ky, weighted, color=colour,
                                  label=_ky_weighted_label(label))
                by["ky_log"].plot(ky, np.abs(weighted), color=colour,
                                  label=_ky_weighted_label(label))
            if "profile" in lines:
                # Unweighted: summed over ky this is the physical total flux
                # through each flux surface.
                by["profile"].plot(
                    x, np.asarray(ds[base + "_x"].sel(species=name)),
                    color=colour, label=label)
        if "ky" in lines:
            by["ky_lin"].set(xlabel=r"$k_y \rho_{\rm ref}$",
                             ylabel=r"$k_y\,$Flux [GB]", title="linear")
            by["ky_log"].set(xlabel=r"$k_y \rho_{\rm ref}$",
                             ylabel=r"$|k_y\,$Flux$|$ [GB]", title="log-log")
            by["ky_log"].set_xscale("log")
            by["ky_log"].set_yscale("log")
        if "profile" in lines:
            by["profile"].set(xlabel=r"$x / a$", ylabel="Flux [GB]",
                              title=r"$\sum_{k_y}$ — radial profile")
        for ax in axes[0]:
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
