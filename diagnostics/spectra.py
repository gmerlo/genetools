# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import matplotlib.pyplot as plt
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
                 x_avg_lims=None, buffer_frac=0.1):
        """
        Parameters
        ----------
        run : genetools.run.Run, optional
        outfile, folder : str, optional
            Override the HDF5 cache location; normally derived from the run.
        x_avg_lims : (float, float), optional
            GENE-3D only: radial averaging range in ``x/a``. Defaults to
            trimming *buffer_frac* from each end, keeping the Krook buffer
            regions out of the spectrum.
        buffer_frac : float
            GENE-3D only; fraction trimmed from each radial end.
        """
        self.x_avg_lims = x_avg_lims
        self.buffer_frac = buffer_frac
        self.consistency = {}
        self._global = None
        if run is not None:
            RunDiagnostic.__init__(self, run)
            if outfile:
                self.outfile = outfile
            if self.geometry_kind == "x_global":
                from genetools.diagnostics.spectra_global import SpectraGlobal
                self._global = SpectraGlobal(run)
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
            for label, flux in zip(["Q_es", "Q_em", "G_es", "G_em"],
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
                for label, flux in zip(["Q_es", "Q_em", "G_es", "G_em"],
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
            for label, flux in zip(["Q_es", "Q_em", "G_es", "G_em"],
                                   [Q_es,    Q_em,   G_es,   G_em]):
                for axis_name, arr in zip(["kx", "ky", "z"], flux):
                    dsname = f"{name}_{label}_{axis_name}"
                    if arr is not None:
                        f[dsname][row_idx, :] = np.asarray(arr, float)

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

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
                if len(time) <= 1:
                    flux_avg[key] = data[0] if len(time) == 1 else data
                else:
                    flux_avg[key] = _trapz(data, x=time, axis=0) / (time[-1] - time[0])
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
            target = self._global if self._global is not None else self
            return target._dataset_from_cache(self.coord, self.params,
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
        the ``(x, ky)`` maps through
        :class:`~genetools.diagnostics.spectra_global.SpectraGlobal`, whose cache
        schema is different enough to be worth keeping separate; this class owns
        the facade either way. GENE-3D rebuilds ky spectra in memory and
        cross-checks them against the fluxes the code wrote itself.
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
        if self._global is not None:
            self._global.compute_and_save(
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
                    acc[v].append(g3.xz_average(fluxes[v], J, xsl))
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
                  "code_flux": code_flux, "xslice": xsl}
        self._check(result)
        return result

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
                ratio = g3.check_flux_consistency(
                    np.sum(spectrum), result["code_flux"][name][v],
                    f"{name} {v}")
                self.consistency[f"{name}/{v}"] = ratio


    # ------------------------------------------------------------------
    # GENE-3D dataset and plot
    # ------------------------------------------------------------------

    def _dataset_3d(self, t):
        from genetools._xr import make_dataset, unit_attrs
        raw = self.compute(t)
        params = self.params
        names = list(self.run.species)
        present = [v for v in self._ES_FLUXES + self._EM_FLUXES
                   if all(v in raw["spectra"][n] for n in names)]
        ds = make_dataset(
            {v + "_ky": (("species", "ky"),
                         np.stack([np.real(raw["spectra"][n][v])
                                   for n in names], axis=0))
             for v in present},
            {"ky": raw["ky"]}, species=names, params=params)
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = self.geometry_kind
        x = np.asarray(self.coord["x_o_a"], dtype=float)[raw["xslice"]]
        ds.attrs["x_avg_range"] = [float(x[0]), float(x[-1])]
        ds.attrs["n_times"] = int(np.size(raw["times"]))
        for label, ratio in self.consistency.items():
            ds.attrs["consistency_" + label.replace("/", "_")] = float(ratio)
        return ds

    def _plot_3d(self, t):
        """Three panels per flux: log-log, ky-weighted, and lin-lin."""
        ds = self._dataset_3d(t)
        ky_full = np.asarray(ds["ky"])
        n_pos = (ky_full.size + 1) // 2
        ky = ky_full[:n_pos]
        keys = [k for k in ds.data_vars if k.endswith("_ky")]
        if not keys:
            raise ValueError("No flux spectra available to plot.")

        fig, axes = plt.subplots(len(keys), 3,
                                 figsize=(13, 3.2 * len(keys)), squeeze=False)
        for row, key in enumerate(keys):
            label = self._TITLES_3D.get(key[:-3], key)
            for name in ds["species"].values:
                vals = np.asarray(ds[key].sel(species=name))[:n_pos]
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
        lo, hi = ds.attrs["x_avg_range"]
        fig.suptitle("GENE-3D flux spectra, "
                     rf"$x/a \in [{lo:.2f}, {hi:.2f}]$")
        fig.tight_layout()
        plt.show()
        return fig

    # ------------------------------------------------------------------

    def plot(self, t=None, x_avg_lims=None, **kw):
        """Plot the flux spectra over the window *t*."""
        if self.is_3d:
            return self._plot_3d(t)
        lo, hi = self._bounds(t)
        r = self.run
        if self._global is not None:
            self.compute(t)
            return self._global.plot(self.coord, self.params, lo, hi,
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
            _suffixes = ("_Q_es_kx", "_Q_em_kx", "_G_es_kx", "_G_em_kx",
                         "_Q_es_ky", "_Q_em_ky", "_G_es_ky", "_G_em_ky",
                         "_Q_es_z",  "_Q_em_z",  "_G_es_z",  "_G_em_z")
            species_set = set()
            for name in f.keys():
                for sfx in _suffixes:
                    if name.endswith(sfx):
                        species_set.add(name[:-len(sfx)])
                        break
            species_names = sorted(species_set)

        nx2     = len(kx) // 2 + 1
        kx_half = kx[:nx2]
        labels  = ["Q_es", "Q_em", "G_es", "G_em"]

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


