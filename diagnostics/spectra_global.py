"""
spectra_global.py — ky-resolved radial flux spectra for global GENE simulations.

Computes x-dependent, ky-resolved transport flux spectra Q(x,ky), G(x,ky),
P(x,ky) for electrostatic and electromagnetic contributions. This is the
global-geometry counterpart of the local ``Spectra`` diagnostic.

Only supports **global** geometry (x_local=False). For local runs, use
:class:`~genetools.diagnostics.spectra.Spectra` instead.

Physics
-------
The core operation is ``compute_flux_yspectra``: for each ky mode separately,
compute the flux-surface-averaged cross-correlation:

  - ky=0: ``F(x,0) = sum_z[ Re(conj(a) * b) * J_norm ] / C_xy``
  - ky>0: ``F(x,ky) = 2 * sum_z[ Re(conj(a) * b) * J_norm ] / C_xy``

where the factor 2 accounts for Hermitian symmetry (negative ky modes).

Electrostatic fluxes use ``v_E = -i*ky*phi``:

  - ``G_es(x,ky) = n0 * F(dens, v_E)``
  - ``Q_es(x,ky) = n0*T0 * F((0.5*T_par+T_perp)*n_map + 1.5*dens*T_map, v_E)``
  - ``P_es(x,ky) = n0*mass * F(v_E, u_par*n_map)``

Electromagnetic fluxes use ``B_par = i*ky*A_par``:

  - ``G_em(x,ky) = n0 * F(u_par*n_map, B_par)``
  - ``Q_em(x,ky) = n0*T0 * F(q_par+q_perp, B_par)``
  - ``P_em(x,ky) = n0*T0 * F(B_par, (T_par*n_map+dens*T_map)*n_map)``

Example
-------
>>> from genetools.diagnostics import SpectraGlobal
>>> sg = SpectraGlobal()
>>> sg.compute_and_save(fld_reader, mom_readers, coord, geom, params,
...                     t_start=10., t_stop=2000.,
...                     equilibrium_profiles=eq_profs)
>>> sg.plot(coord, params)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

# numpy <2.0 compatibility
_trapz = getattr(np, 'trapz', None) or getattr(np, 'trapezoid')


# ---------------------------------------------------------------------------
# Core per-ky flux computation
# ---------------------------------------------------------------------------

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
    nx, nky, nz = a.shape
    out = np.zeros((nx, nky))

    # ky=0: no factor 2
    cross_0 = np.real(np.conj(a[:, 0, :]) * b[:, 0, :])  # (nx, nz)
    out[:, 0] = np.sum(cross_0 * J_norm, axis=1)

    # ky>0: factor 2 for Hermitian symmetry
    for iky in range(1, nky):
        cross_k = np.real(np.conj(a[:, iky, :]) * b[:, iky, :])
        out[:, iky] = 2.0 * np.sum(cross_k * J_norm, axis=1)

    # Division by C_xy (applied after z-summation, matching MATLAB)
    C_xy_arr = np.asarray(C_xy)
    if C_xy_arr.ndim == 2:
        # (nx, nz) → average over z to get per-x scalar
        out /= np.mean(C_xy_arr, axis=1)[:, np.newaxis]
    elif C_xy_arr.ndim == 1:
        out /= C_xy_arr[:, np.newaxis]
    else:
        out /= float(C_xy_arr)

    return out


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SpectraGlobal:
    """
    Compute, cache, and plot ky-resolved radial flux spectra for global runs.

    Parameters
    ----------
    outfile : str, optional
        Path to the output HDF5 file (default ``'spectra_global.h5'``).
    """

    def __init__(self, outfile: str = "spectra_global.h5"):
        self.outfile = outfile

    # ------------------------------------------------------------------
    # HDF5 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_saved_times(outfile: str) -> np.ndarray:
        if not os.path.exists(outfile):
            return np.array([], dtype=np.float64)
        with h5py.File(outfile, "r") as f:
            if "time" not in f:
                return np.array([], dtype=np.float64)
            return f["time"][...]

    @staticmethod
    def _is_already_saved(time: float, saved_times: np.ndarray) -> bool:
        if saved_times.size == 0:
            return False
        tol = max(1e-6, abs(time) * 1e-6)
        return bool(np.any(np.abs(saved_times - time) <= tol))

    @staticmethod
    def _init_h5(f, species_names: list, nx: int, nky: int, has_em: bool):
        """Create all datasets in a newly opened HDF5 file handle."""
        f.create_dataset("time", shape=(0,), maxshape=(None,),
                         dtype=np.float64, chunks=True)
        es_keys = ["Qes_ky", "Ges_ky", "Pes_ky"]
        em_keys = ["Qem_ky", "Gem_ky", "Pem_ky"] if has_em else []
        for name in species_names:
            grp = f.create_group(name)
            for key in es_keys + em_keys:
                grp.create_dataset(key, shape=(nx, nky, 0),
                                   maxshape=(nx, nky, None),
                                   dtype=np.float64, chunks=True)

    @staticmethod
    def _append_to_open_file(f, species_names: list, species_data: dict,
                             time: float):
        """Append one time step to an already-open HDF5 file handle."""
        tds = f["time"]
        n = tds.shape[0]
        tds.resize((n + 1,))
        tds[n] = time
        for name in species_names:
            sd = species_data[name]
            for key, val in sd.items():
                ds = f[f"{name}/{key}"]
                ds.resize((ds.shape[0], ds.shape[1], n + 1))
                ds[:, :, n] = val

    # ------------------------------------------------------------------
    # Time synchronisation
    # ------------------------------------------------------------------

    def _sync_indices(self, fld_reader, mom_readers, t_start, t_stop,
                      params):
        istep_fld = int(params["in_out"]["istep_field"])
        istep_mom = int(params["in_out"]["istep_mom"])
        L         = int(np.lcm(istep_fld, istep_mom))
        stride_fld = L // istep_fld
        stride_mom = L // istep_mom

        times_fld = fld_reader.read_all_times()
        idx_fld = np.where(
            (times_fld >= t_start) & (times_fld <= t_stop))[0][::stride_fld]
        times_mom = mom_readers[0].read_all_times()
        idx_mom = np.where(
            (times_mom >= t_start) & (times_mom <= t_stop))[0][::stride_mom]

        saved_times = self._load_saved_times(self.outfile)
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

    # ------------------------------------------------------------------
    # Public interface — compute
    # ------------------------------------------------------------------

    def compute_and_save(
        self,
        fld_reader,
        mom_readers: list,
        coords: dict,
        geom: dict,
        params: dict,
        t_start: float,
        t_stop: float,
        equilibrium_profiles: dict = None,
    ) -> None:
        """
        Stream field + moment files, compute ky-resolved flux spectra,
        and append to HDF5.

        Parameters
        ----------
        fld_reader
            Field reader (BinaryReader or MultiSegmentReader).
        mom_readers : list
            One moment reader per species.
        coords : dict
            Coordinate dictionary.
        geom : dict
            Geometry dictionary.
        params : dict
            Parameter dictionary.
        t_start, t_stop : float
            Time window.
        equilibrium_profiles : dict, optional
            ``{species_name: {'T': array, 'n': array}}``.
            Required for profile-corrected flux computation.
        """
        x_local = params["general"].get("x_local", True)
        if x_local:
            print("SpectraGlobal is for global runs only. "
                  "Use Spectra for local runs.")
            return

        nx       = params["box"]["nx0"]
        nky      = params["box"]["nky0"]
        n_fields = params["info"]["n_fields"]
        species  = params["species"]
        species_names = [sp["name"] for sp in species]
        ky = np.asarray(coords["ky"])
        has_em = n_fields > 1

        # Jacobian normalization (global: (nx, nz))
        J = geom["Jacobian"]
        J_norm = J / J.sum(axis=1, keepdims=True)

        # C_xy
        C_xy = geom["metric"]["C_xy"]

        # Build prefactors from equilibrium profiles
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

        # Sync field + moment indices
        idx_fld, idx_mom = self._sync_indices(
            fld_reader, mom_readers, t_start, t_stop, params)

        if len(idx_fld) == 0 or len(idx_mom) == 0:
            return

        it_field = fld_reader.stream_selected(idx_fld)
        it_moms  = [r.stream_selected(idx_mom) for r in mom_readers]

        with h5py.File(self.outfile, "a") as hf:
            initialised = "time" in hf

            for tm, fields in it_field:
                # Read moments for all species
                all_moments = []
                for it_m in it_moms:
                    _, moms = next(it_m)
                    all_moments.append(moms)

                # Compute v_E = -i*ky*phi (no IFFT — global is real-space x)
                phi = fields[0]
                v_E = np.zeros_like(phi)
                ky3 = ky[np.newaxis, :, np.newaxis]
                v_E = -1j * ky3 * phi

                # Compute B_par = i*ky*A_par if EM
                B_par = None
                if has_em:
                    A_par = fields[1]
                    B_par = 1j * ky3 * A_par

                sp_data = {}
                for i_sp, sp in enumerate(species):
                    name = sp["name"]
                    n0   = sp["dens"]
                    T0   = sp["temp"]
                    mass = sp.get("mass", 1.0)
                    moments = all_moments[i_sp]

                    pf = prefactors.get(name, {})
                    n_map = pf.get("n_map", 1.0)
                    T_map = pf.get("T_map", 1.0)

                    dens   = moments[0]
                    T_par  = moments[1]
                    T_perp = moments[2]
                    u_par  = moments[5]

                    # ES fluxes
                    tmp_q = (0.5 * T_par + T_perp) * n_map \
                        + 1.5 * dens * T_map
                    Ges = n0 * _compute_flux_yspectra(
                        dens, v_E, C_xy, J_norm)
                    Qes = n0 * T0 * _compute_flux_yspectra(
                        tmp_q, v_E, C_xy, J_norm)
                    Pes = n0 * mass * _compute_flux_yspectra(
                        v_E, u_par * n_map, C_xy, J_norm)

                    result = {"Qes_ky": Qes, "Ges_ky": Ges, "Pes_ky": Pes}

                    # EM fluxes
                    if has_em and B_par is not None:
                        q_par  = moments[3]
                        q_perp = moments[4]

                        Gem = n0 * _compute_flux_yspectra(
                            u_par * n_map, B_par, C_xy, J_norm)
                        Qem = n0 * T0 * _compute_flux_yspectra(
                            q_par + q_perp, B_par, C_xy, J_norm)
                        Pem = n0 * T0 * _compute_flux_yspectra(
                            B_par,
                            (T_par * n_map + dens * T_map) * n_map,
                            C_xy, J_norm)
                        result.update({
                            "Qem_ky": Qem, "Gem_ky": Gem, "Pem_ky": Pem})

                    sp_data[name] = result

                if not initialised:
                    self._init_h5(hf, species_names, nx, nky, has_em)
                    initialised = True

                self._append_to_open_file(hf, species_names, sp_data, tm)

    # ------------------------------------------------------------------
    # Public interface — load
    # ------------------------------------------------------------------

    def load(self, t_start: float = None, t_stop: float = None) -> dict:
        """
        Load saved ky-resolved spectra from the HDF5 file.

        Returns
        -------
        dict
            Keys: ``'time'``, ``'{species}_Qes_ky'``, etc.
            Spectra arrays have shape ``(n_times, nx, nky)``.
        """
        if not os.path.exists(self.outfile):
            return {}

        with h5py.File(self.outfile, "r") as f:
            time = f["time"][...]
            if time.size == 0:
                return {}

            idx = np.argsort(time)
            time = time[idx]

            mask = np.ones(len(time), dtype=bool)
            if t_start is not None:
                mask &= time >= t_start
            if t_stop is not None:
                mask &= time <= t_stop
            final_idx = idx[mask]
            time = time[mask]

            result = {"time": time}
            species_names = [k for k in f.keys() if k != "time"]
            for name in species_names:
                grp = f[name]
                for key in grp.keys():
                    # Dataset shape: (nx, nky, n_times)
                    data = grp[key][:, :, final_idx]
                    # Transpose to (n_times, nx, nky)
                    result[f"{name}_{key}"] = np.transpose(data, (2, 0, 1))

        return result

    def load_time_average(self, t_start: float = None,
                          t_stop: float = None) -> dict:
        """
        Load and time-average ky-resolved spectra.

        Returns
        -------
        dict
            Keys: ``'{species}_Qes_ky'``, etc.
            Each value is shape ``(nx, nky)``.
        """
        data = self.load(t_start, t_stop)
        if not data or "time" not in data:
            return {}

        time = data["time"]
        result = {}
        for key, arr in data.items():
            if key == "time":
                continue
            if len(time) <= 1:
                result[key] = arr[0]
            else:
                result[key] = _trapz(arr, x=time, axis=0) / \
                    (time[-1] - time[0])
        return result

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, coords: dict, params: dict,
             t_start: float = None, t_stop: float = None,
             x_avg_lims: tuple = None) -> None:
        """
        Plot ky-resolved flux spectra from the saved HDF5 file.

        Per species produces:
        1. (ky, x) heatmap of time-averaged spectra
        2. x-averaged 1D ky spectrum (all fluxes overlaid)

        Parameters
        ----------
        coords : dict
        params : dict
        t_start, t_stop : float, optional
        x_avg_lims : tuple of (x_start, x_end), optional
            Radial window for x-averaging. Defaults to 10% around x0.
        """
        data = self.load(t_start, t_stop)
        if not data or "time" not in data or len(data["time"]) == 0:
            print("No spectra_global data available to plot.")
            return

        times = data["time"]
        x = np.asarray(coords["x"])
        ky = np.asarray(coords["ky"])
        species = params["species"]
        n_fields = params["info"]["n_fields"]
        has_em = n_fields > 1

        # Determine x-averaging region
        if x_avg_lims is not None:
            xs, xe = x_avg_lims
        else:
            x0 = params["box"].get("x0", (x[0] + x[-1]) / 2)
            span = (x[-1] - x[0]) * 0.1
            xs, xe = x0 - span / 2, x0 + span / 2
        i_s = np.argmin(np.abs(x - xs))
        i_e = np.argmin(np.abs(x - xe))
        x_inds = slice(i_s, i_e + 1)

        x_label = r"$x / a$"
        ky_label = r"$k_y \rho_{\rm ref}$"
        t_label = r"$t\;c_{\rm ref}/L_{\rm ref}$"

        es_keys = ["Qes_ky", "Ges_ky", "Pes_ky"]
        em_keys = ["Qem_ky", "Gem_ky", "Pem_ky"] if has_em else []
        all_keys = es_keys + em_keys

        flux_labels = {
            "Qes_ky": r"$Q_{\rm es}$",
            "Ges_ky": r"$\Gamma_{\rm es}$",
            "Pes_ky": r"$\Pi_{\rm es}$",
            "Qem_ky": r"$Q_{\rm em}$",
            "Gem_ky": r"$\Gamma_{\rm em}$",
            "Pem_ky": r"$\Pi_{\rm em}$",
        }
        flux_colors = {
            "Qes_ky": "b", "Ges_ky": "r", "Pes_ky": "g",
            "Qem_ky": "m", "Gem_ky": "k", "Pem_ky": "c",
        }

        for sp in species:
            name = sp["name"]
            present_keys = [k for k in all_keys
                           if f"{name}_{k}" in data]
            if not present_keys:
                continue

            # Time-average
            tavg = {}
            for key in present_keys:
                arr = data[f"{name}_{key}"]  # (n_times, nx, nky)
                if len(times) > 1:
                    tavg[key] = _trapz(arr, x=times, axis=0) / \
                        (times[-1] - times[0])
                else:
                    tavg[key] = arr[0]

            # ── Fig 1: (ky, x) heatmaps ──────────────────────────────
            ncols = len(present_keys)
            fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4),
                                     squeeze=False)
            title_str = (f"average [{times[0]:.1f} - {times[-1]:.1f}]"
                        if len(times) > 1 else f"t = {times[0]:.1f}")
            fig.suptitle(f"{name} — {title_str}")
            for ax, key in zip(axes[0], present_keys):
                im = ax.pcolormesh(ky, x, tavg[key], shading="auto")
                fig.colorbar(im, ax=ax)
                ax.set_xlabel(ky_label)
                ax.set_ylabel(x_label)
                ax.set_title(flux_labels.get(key, key))
            plt.tight_layout()
            plt.show()

            # ── Fig 2: x-averaged 1D ky spectrum (all fluxes) ────────
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(
                f"{name} — x-avg [{x[i_s]:.3f}, {x[i_e]:.3f}]"
                f" — {title_str}")
            for key in present_keys:
                spec_xavg = np.mean(tavg[key][x_inds, :], axis=0)
                axes[0].plot(ky, spec_xavg,
                            color=flux_colors.get(key, "b"),
                            label=flux_labels.get(key, key))
                axes[1].plot(ky, np.abs(spec_xavg),
                            color=flux_colors.get(key, "b"),
                            label=flux_labels.get(key, key))
                total = np.sum(spec_xavg)
                print(f"  {name} {key}: sum = {total:.6g}")

            axes[0].set_xlabel(ky_label)
            axes[0].set_ylabel("Flux [GB]")
            axes[0].legend()
            axes[0].grid(True)
            axes[0].set_title("linear")

            axes[1].set_xlabel(ky_label)
            axes[1].set_ylabel("|Flux| [GB]")
            axes[1].set_xscale("log")
            axes[1].set_yscale("log")
            axes[1].legend()
            axes[1].grid(True)
            axes[1].set_title("log-log")

            plt.tight_layout()
            plt.show()
