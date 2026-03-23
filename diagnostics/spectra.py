import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from concurrent.futures import ThreadPoolExecutor

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = False
except ImportError:
    NUMBA_AVAILABLE = False

# numpy <2.0 compatibility
_trapz = getattr(np, 'trapz', None) or getattr(np, 'trapezoid')


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
        def is_saved(t):
            t32  = np.float32(t)
            tol  = max(1e-6, abs(t32) * 1e-6)
            return np.any(np.abs(saved_times - t32) <= tol)
        idx_fld_filtered = [i for i in idx_fld if not is_saved(times_fld[i])]
        idx_mom_filtered = [i for i in idx_mom if not is_saved(times_mom[i])]
        return idx_fld_filtered, idx_mom_filtered

    # ------------------------------------------------------------------
    # Spectral computation
    # ------------------------------------------------------------------

    def compute_spectra(self, fields, moments, ky3, J_norm, Bfield, params):
        """
        Compute ES and EM flux spectra for all species in parallel.

        Parameters
        ----------
        fields   : list of np.ndarray  (nx, nky, nz)
        moments  : list of list        one inner list per species, 9 moment arrays each
        ky3      : np.ndarray          (1, nky, 1) broadcast array, precomputed
        J_norm   : np.ndarray          (nz,) normalised Jacobian, precomputed
        Bfield   : np.ndarray          (nz,) equilibrium B field from geometry
        params   : dict

        Returns
        -------
        list of [Q_es, Q_em, G_es, G_em] per species
        """
        species  = params["species"]
        n_fields = params["info"]["n_fields"]
        
        def _one_species(i_sp):
            sp         = species[i_sp]
            mom        = moments[i_sp]
            n0, T0, q0 = sp['dens'], sp['temp'], sp['charge']
        
            G_es = self.averages(-1j*ky3*fields[0] * np.conj(mom[0])*n0, J_norm)
            Q_es = self.averages(-1j*ky3*fields[0] * np.conj(0.5*mom[1]+mom[2]+1.5*mom[0])*n0*T0, J_norm)

            if n_fields > 1:
                B_x  = 1j*ky3*fields[1]
                tmp1 = B_x * np.conj(mom[5])*n0
                tmp2 = B_x * np.conj(mom[3]+mom[4])*n0*T0
                if n_fields > 2:
                    dBpar_dy = -1j*ky3*fields[2] / Bfield[np.newaxis, np.newaxis, :]
                    tmp1 += dBpar_dy * np.conj(mom[6])*n0*T0/q0
                    tmp2 += dBpar_dy * np.conj(mom[7]+mom[8])*n0*T0**2/q0
                G_em = self.averages(tmp1, J_norm)
                Q_em = self.averages(tmp2, J_norm)
            else:
                G_em = Q_em = (None, None, None)

            return [Q_es, Q_em, G_es, G_em]

        n_species = len(species)
        if n_species == 1:
            return [_one_species(0)]
        with ThreadPoolExecutor(max_workers=n_species) as ex:
            return list(ex.map(_one_species, range(n_species)))

    @staticmethod
    def averages(flux, J_norm):
        """
        Compute kx spectrum, ky spectrum, and z profile of a flux array.

        Parameters
        ----------
        flux   : np.ndarray  (nx, nky, nz) complex
        J_norm : np.ndarray  (nz,) normalised Jacobian, precomputed

        Returns
        -------
        sp_kx : np.ndarray  (nx//2+1,)
        sp_ky : np.ndarray  (nky,)
        sum_z : np.ndarray  (nz,)
        """
        if flux is None:
            return (None, None, None)
        flux = flux.copy()
        flux[:, 1:, :] *= 2
        J      = J_norm[np.newaxis, np.newaxis, :]
        sum_z  = np.sum(flux.real * J, axis=(0, 1))
        avg_z  = np.sum(flux.real * J, axis=2)
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
    def _init_h5(f, coords, species_names, fluxes):
        """Create all datasets in a newly opened HDF5 file."""
        f.create_dataset("kx",   data=coords["kx"])
        f.create_dataset("ky",   data=coords["ky"])
        f.create_dataset("z",    data=coords["z"])
        f.create_dataset("time", data=np.array([], dtype=np.float32),
                         maxshape=(None,))
        for i_sp, name in enumerate(species_names):
            Q_es, Q_em, G_es, G_em = fluxes[i_sp]
            for label, flux in zip(["Q_es", "Q_em", "G_es", "G_em"],
                                   [Q_es,    Q_em,   G_es,   G_em]):
                for axis_name, arr in zip(["kx", "ky", "z"], flux):
                    dsname = f"{name}_{label}_{axis_name}"
                    if arr is None:
                        f.create_dataset(dsname, data=np.empty((0, 0)),
                                         maxshape=(None, None))
                    else:
                        f.create_dataset(dsname, data=np.empty((0, arr.size)),
                                         maxshape=(None, arr.size))

    @staticmethod
    def _append_to_open_file(f, fluxes, species_names, time_value):
        """Append one time step to an already-open HDF5 file handle."""
        tds = f["time"]
        tds.resize((tds.shape[0] + 1,))
        tds[-1] = np.float32(time_value)
        for i_sp, name in enumerate(species_names):
            Q_es, Q_em, G_es, G_em = fluxes[i_sp]
            for label, flux in zip(["Q_es", "Q_em", "G_es", "G_em"],
                                   [Q_es,    Q_em,   G_es,   G_em]):
                for axis_name, arr in zip(["kx", "ky", "z"], flux):
                    dsname = f"{name}_{label}_{axis_name}"
                    dset   = f[dsname]
                    if arr is None:
                        dset.resize((dset.shape[0] + 1, 0))
                    else:
                        arr = np.asarray(arr, float)
                        dset.resize((dset.shape[0] + 1, dset.shape[1]))
                        dset[-1, :] = arr

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

        # Change 3: precompute invariants once outside the time loop
        ky3    = coords["ky"][np.newaxis, :, np.newaxis]          # (1, nky, 1)
        J_norm = geom['Jacobian'] / np.sum(geom['Jacobian'])      # (nz,)
        Bfield = geom['Bfield']                                    # (nz,)

        idx_fld, idx_mom = self.sync_indices(fld_reader, mom_readers,
                                             t_start, t_stop, params)
        
        
        if len(idx_fld) == 0 or len(idx_mom) == 0:
            return

        it_field = fld_reader.stream_selected(idx_fld)
        it_moms  = [r.stream_selected(idx_mom) for r in mom_readers]

        # Change 6: open HDF5 once for all time steps
        with h5py.File(self.outfile, "a") as hf:
            initialised = "time" in hf

            for tm, fields in it_field:
                print('doing', tm)
                # Change 5: read all species moment files in parallel
                if len(it_moms) == 1:
                    moments_data = [next(it_moms[0])[1]]
                else:
                    with ThreadPoolExecutor(max_workers=len(it_moms)) as ex:
                        moments_data = [r[1] for r in ex.map(next, it_moms)]

                # Change 4: species computed in parallel (inside compute_spectra)
                fluxes = self.compute_spectra(fields, moments_data,
                                              ky3, J_norm, Bfield, params)

                if not initialised:
                    self._init_h5(hf, coords, species_names, fluxes)
                    initialised = True
                self._append_to_open_file(hf, fluxes, species_names, tm)

    def load_time_average(self, t_start=None, t_stop=None):
        if not os.path.exists(self.outfile):
            return {}
        with h5py.File(self.outfile, "r") as f:
            time       = f["time"][...]
            sorted_idx = np.argsort(time)
            time       = time[sorted_idx]
            mask       = np.ones(len(time), dtype=bool)
            if t_start is not None:
                mask &= time >= t_start
            if t_stop is not None:
                mask &= time <= t_stop
            time = time[mask]
            flux_avg = {}
            for key in f.keys():
                if key == "time" or key in ["kx", "ky", "z"]:
                    continue
                data = f[key][...][sorted_idx][mask]
                if len(time) <= 1:
                    print('at', time)
                    flux_avg[key] = data[0]
                else:
                    print('time avg:',time[0], time[-1])
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


