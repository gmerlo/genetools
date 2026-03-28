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


class Spectra:
    def __init__(self, outfile="flux_spectra.h5"):
        self.outfile = outfile

    # ------------------------------------------------------------------
    # Synchronisation
    # ------------------------------------------------------------------

    def sync_indices(self, fld_reader, mom_readers, t_start, t_stop, params):
        istep_fld = int(params["in_out"]["istep_field"])
        istep_mom = int(params["in_out"]["istep_mom"])
        if istep_fld <= 0 or istep_mom <= 0:
            raise ValueError("istep_field and istep_mom must be positive integers")
        L          = int(np.lcm(istep_fld, istep_mom))
        stride_fld = L // istep_fld
        stride_mom = L // istep_mom
        times_fld  = fld_reader.read_all_times()
        idx_fld    = np.where((times_fld >= t_start) & (times_fld <= t_stop))[0][::stride_fld]
        times_mom  = mom_readers[0].read_all_times()
        idx_mom    = np.where((times_mom >= t_start) & (times_mom <= t_stop))[0][::stride_mom]
        if os.path.exists(self.outfile):
            with h5py.File(self.outfile, "r") as f:
                saved_times = f["time"][...] if "time" in f else np.array([], dtype=np.float32)
        else:
            saved_times = np.array([], dtype=np.float32)

        if saved_times.size == 0:
            return idx_fld.tolist(), idx_mom.tolist()

        saved_sorted = np.sort(saved_times.astype(np.float32))

        def _filter_unsaved(indices, all_times):
            if len(indices) == 0:
                return []
            candidate = np.float32(all_times[indices])
            tol = np.maximum(1e-6, np.abs(candidate) * 1e-6)
            pos = np.searchsorted(saved_sorted, candidate)
            found = np.zeros(len(candidate), dtype=bool)
            for offset in (0, -1):
                idx_check = np.clip(pos + offset, 0, len(saved_sorted) - 1)
                found |= np.abs(saved_sorted[idx_check] - candidate) <= tol
            return [i for i, f in zip(indices, found) if not f]

        idx_fld_filtered = _filter_unsaved(idx_fld, times_fld)
        idx_mom_filtered = _filter_unsaved(idx_mom, times_mom)
        return idx_fld_filtered, idx_mom_filtered

    # ------------------------------------------------------------------
    # Spectral computation
    # ------------------------------------------------------------------

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
    def _init_h5(f, coords, species_names, fluxes, n_alloc=1):
        """Create all datasets in a newly opened HDF5 file, pre-allocated."""
        f.create_dataset("kx",   data=coords["kx"])
        f.create_dataset("ky",   data=coords["ky"])
        f.create_dataset("z",    data=coords["z"])
        f.create_dataset("time", shape=(n_alloc,), dtype=np.float32,
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

        f["time"][row_idx] = np.float32(time_value)
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
                                      n_alloc=n_missing)
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
            time = f["time"][...]
            sorted_idx = np.argsort(time)
            time = time[sorted_idx]

            mask = np.ones(len(time), dtype=bool)
            if t_start is not None:
                mask &= time >= t_start
            if t_stop is not None:
                mask &= time <= t_stop

            # Compute final index array once for HDF5 fancy indexing
            final_idx = sorted_idx[mask]
            time = time[mask]

            flux_avg = {}
            for key in f.keys():
                if key in ("time", "kx", "ky", "z"):
                    continue
                data = f[key][final_idx]
                if len(time) <= 1:
                    flux_avg[key] = data[0] if len(time) == 1 else data
                else:
                    flux_avg[key] = _trapz(data, x=time, axis=0) / (time[-1] - time[0])
        return flux_avg

    def plot(self, fld_reader, mom_readers, coords, geom, params_list,
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
            species_names = sorted({name.split("_")[0]
                                    for name in f.keys() if "_" in name})

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
                        ax.plot(kx_half[:len(arr)], arr if scale is "linear" else np.abs(arr), label=label)
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
                        ax.plot(ky[:len(arr)], arr if scale is "linear" else np.abs(arr), label=label)
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


