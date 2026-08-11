# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
from matplotlib.colors import LogNorm, SymLogNorm
import h5py

from genetools.compat import trapz as _trapz
from genetools.diagnostics._base import CachingDiagnostic


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
        # (nx, nz) → average over z to get per-x scalar
        out /= np.mean(C_xy_arr, axis=1)[:, np.newaxis]
    elif C_xy_arr.ndim == 1:
        out /= C_xy_arr[:, np.newaxis]
    else:
        out /= float(C_xy_arr)

    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SpectraGlobal(CachingDiagnostic):
    """
    Compute, cache, and plot ky-resolved radial flux spectra for global runs.

    Parameters
    ----------
    outfile : str, optional
        Path to the output HDF5 file (default ``'spectra_global.h5'``).
    """

    def __init__(self, outfile: str = "spectra_global.h5", folder: str = None):
        super().__init__(outfile, folder)

    # ------------------------------------------------------------------
    # HDF5 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_h5(f, species_names: list, nx: int, nky: int, has_em: bool,
                 time_dtype=np.float64):
        """Create all datasets in a newly opened HDF5 file handle."""
        f.create_dataset("time", shape=(0,), maxshape=(None,),
                         dtype=time_dtype, chunks=True)
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
        idx_fld, idx_mom = self._sync_field_mom_indices(
            fld_reader, mom_readers, t_start, t_stop, params)

        if len(idx_fld) == 0 or len(idx_mom) == 0:
            return

        it_field = fld_reader.stream_selected(idx_fld)
        it_moms  = [r.stream_selected(idx_mom) for r in mom_readers]

        with h5py.File(self.outfile, "a") as hf:
            initialised = "time" in hf
            if initialised and any(n not in hf for n in species_names):
                raise ValueError(
                    f"Cache '{self.outfile}' has a time axis but no data group "
                    f"for every species {species_names} — it was written by an "
                    "interrupted run or a different configuration. Delete it "
                    "and recompute.")

            for tm, fields in it_field:
                # Read moments for all species
                all_moments = []
                for it_m in it_moms:
                    _, moms = next(it_m)
                    all_moments.append(moms)

                # Compute v_E = -i*ky*phi (no IFFT — global is real-space x)
                phi = fields[0]
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
                    self._init_h5(hf, species_names, nx, nky, has_em,
                                  time_dtype=self._time_dtype(params))
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
            if "time" not in f:     # partially written cache
                return {}
            time = f["time"][...]
            if time.size == 0:
                return {}

            time, read_idx, unsort = self._select_window(time, t_start, t_stop)

            result = {"time": time}
            species_names = [k for k in f.keys() if k != "time"]
            for name in species_names:
                grp = f[name]
                for key in grp.keys():
                    # Dataset shape: (nx, nky, n_times)
                    data = grp[key][:, :, read_idx][:, :, unsort]
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
        if len(time) == 0:      # window selects no cached step
            return {}
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

    @staticmethod
    def _expand_reductions(raw: dict) -> dict:
        """Expand each ``(nx, nky)`` flux map into the map plus its reductions.

        ``'ions_Qes_ky'`` (2-D on disk) becomes ``'ions_Qes_xky'`` (x, ky),
        ``'ions_Qes_x'`` — the total flux profile summed over ky — and
        ``'ions_Qes_ky'`` — the radially averaged ky spectrum.
        """
        out = {}
        for key, arr in raw.items():
            arr = np.asarray(arr)
            base = key[:-3] if key.endswith("_ky") else key
            if arr.ndim != 2:
                out[key] = arr
                continue
            out[f"{base}_xky"] = arr
            out[f"{base}_x"] = arr.sum(axis=1)    # total flux at each radius
            out[f"{base}_ky"] = arr.mean(axis=0)  # radially averaged spectrum
        return out

    def dataset(self, coords, params, species, t_start=None, t_stop=None):
        """
        Return the time-averaged flux spectra as an ``xarray.Dataset``.

        Per species and flux channel (``Qes``, ``Ges``, ``Pes`` and their EM
        counterparts) the Dataset carries the 2-D map ``*_xky`` with dims
        ``(x, ky)`` plus its two 1-D reductions — ``*_x``, the total flux
        profile summed over ky, and ``*_ky``, the ky spectrum averaged over
        the full radial domain. Mirrors the layout of
        :class:`~genetools.diagnostics.amplitude.AmplitudeSpectra` for global
        runs. Slice the map directly for a windowed average, e.g.
        ``ds.Qes_xky.sel(x=slice(0.4, 0.6)).mean('x')``.
        """
        import xarray as xr
        from genetools import _xr

        raw = self.load_time_average(t_start, t_stop)
        if not raw:
            return xr.Dataset()

        data_vars, used = _xr.stacked_vars(
            self._expand_reductions(raw), species, _xr.dims_from_suffix)
        x = np.asarray(coords.get("x", []))
        if x.size == 0:
            x = np.asarray(coords.get("x_o_a", []))
        candidates = {"x": x, "ky": np.asarray(coords.get("ky", []))}
        return _xr.make_dataset(data_vars, candidates, species=used, params=params)

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

            # ── Fig 1: (ky, x) heatmaps of ky-weighted flux ──────────
            # ky weighting puts equal areas at equal flux contribution on a
            # logarithmic ky axis, since ∫F dky = ∫ ky F d(ln ky).
            ncols = len(present_keys)
            fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4),
                                     squeeze=False)
            title_str = (f"average [{times[0]:.1f} - {times[-1]:.1f}]"
                        if len(times) > 1 else f"t = {times[0]:.1f}")
            fig.suptitle(f"{name} — {title_str}")
            # The ky=0 column is a structural zero — the electrostatic fluxes
            # are built from v_E = -i ky phi, which vanishes there — so it
            # carries no information and would force a linear region into an
            # otherwise logarithmic scale. Drop it.
            kpos = ky > 0
            ky_p = ky[kpos] if kpos.any() else ky
            for ax, key in zip(axes[0], present_keys):
                weighted = tavg[key][:, kpos] * ky_p[np.newaxis, :] \
                    if kpos.any() else tavg[key] * ky[np.newaxis, :]
                norm, cmap = _flux_norm_and_cmap(weighted)
                im = ax.pcolormesh(ky_p, x, weighted, shading="auto",
                                   norm=norm, cmap=cmap)
                fig.colorbar(im, ax=ax)
                ax.set_xscale("log")
                ax.set_xlabel(ky_label)
                ax.set_ylabel(x_label)
                ax.set_title(_ky_weighted_label(flux_labels.get(key, key)))
            plt.tight_layout()
            plt.show()

            # ── Fig 2: 1D reductions — ky spectrum + radial profile ──
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            fig.suptitle(
                f"{name} — x-avg [{x[i_s]:.3f}, {x[i_e]:.3f}]"
                f" — {title_str}")
            for key in present_keys:
                spec_xavg = np.mean(tavg[key][x_inds, :], axis=0)
                weighted = spec_xavg * ky
                color = flux_colors.get(key, "b")
                label = flux_labels.get(key, key)
                axes[0].plot(ky, weighted, color=color,
                             label=_ky_weighted_label(label))
                axes[1].plot(ky, np.abs(weighted), color=color,
                             label=_ky_weighted_label(label))
                # Radial profile stays unweighted: summed over ky it is the
                # physical total flux through each flux surface.
                axes[2].plot(x, np.sum(tavg[key], axis=1),
                             color=color, label=label)
                total = np.sum(spec_xavg)
                print(f"  {name} {key}: sum = {total:.6g}")

            axes[0].set_xlabel(ky_label)
            axes[0].set_ylabel(r"$k_y\,$Flux [GB]")
            axes[0].legend()
            axes[0].grid(True)
            axes[0].set_title("linear")

            axes[1].set_xlabel(ky_label)
            axes[1].set_ylabel(r"$|k_y\,$Flux$|$ [GB]")
            axes[1].set_xscale("log")
            axes[1].set_yscale("log")
            axes[1].legend()
            axes[1].grid(True)
            axes[1].set_title("log-log")

            axes[2].set_xlabel(x_label)
            axes[2].set_ylabel("Flux [GB]")
            axes[2].legend()
            axes[2].grid(True)
            axes[2].set_title(r"$\sum_{k_y}$ — radial profile")

            plt.tight_layout()
            plt.show()
