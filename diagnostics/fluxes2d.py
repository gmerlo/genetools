"""
fluxes2d.py — Radial flux profile diagnostic for GENE simulations.

Computes x-resolved, flux-surface-averaged transport fluxes from field and
moment data: particle flux (G), heat flux (Q), and momentum flux (P) for
both electrostatic (ES) and electromagnetic (EM) contributions.

Supports both **local** (x_local=True, spectral in x) and **global**
(x_local=False, real-space radial grid) geometry, following the same
conventions as the MATLAB ``diag_fluxes_2D.m`` + ``plot_fluxes_2D.m``.

Physics
-------
Electrostatic fluxes use the ExB velocity ``v_E = -i*ky*phi / C_xy``:

  - Particle flux: ``G_es = n0 * <v_E * dens*>_ky``
  - Heat flux:     ``Q_es = n0*T0 * <v_E * (0.5*T_par + T_perp + 1.5*dens)*>_ky``
  - Momentum flux: ``P_es = n0*mass * <v_E * u_par*>_ky``

Electromagnetic fluxes use ``v_A = i*ky*A_par / C_xy``:

  - Particle flux: ``G_em = n0 * <v_A * u_par*>_ky``
  - Heat flux:     ``Q_em = n0*T0 * <v_A * (q_par + q_perp)*>_ky``
  - Momentum flux: ``P_em = n0*T0 * <v_A * (T_par + dens)*>_ky``

The ky summation uses Hermitian symmetry: ``f(ky=0) + 2*sum_{ky>0} f(ky)``.
Results are then flux-surface averaged over z using Jacobian weighting.

Example
-------
>>> from genetools.io import BinaryReader, Params, Geometry, Coordinates, set_runs
>>> from genetools.diagnostics import Fluxes2D
>>>
>>> folder = '/path/to/run/'
>>> runs   = set_runs(folder)
>>> params = Params(folder, runs)
>>> geom   = Geometry(folder, runs, params)
>>> coord  = Coordinates(folder, runs, params)
>>>
>>> fld_reader = BinaryReader('field', folder, runs[0], params.get(0))
>>> mom_readers = [
...     BinaryReader('mom', folder, runs[0], params.get(0), species=sp['name'])
...     for sp in params.get(0)['species']
... ]
>>>
>>> fl = Fluxes2D()
>>> fl.compute_and_save(fld_reader, mom_readers, coord[0], geom[0],
...                     params.get(0), t_start=10., t_stop=2000.)
>>> fl.plot(coord[0], params.get(0))
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

from genetools.compat import trapz as _trapz
from genetools.diagnostics._base import CachingDiagnostic


# ---------------------------------------------------------------------------
# Core flux computation
# ---------------------------------------------------------------------------

def _compute_flux(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute ky-summed cross-correlation ``Re{a * conj(b)}`` with Hermitian
    symmetry weighting (factor 2 for ky > 0).

    Parameters
    ----------
    a, b : np.ndarray
        Complex arrays of shape ``(nx, nky, nz)``.

    Returns
    -------
    np.ndarray
        Real array of shape ``(nx, nz)``.
    """
    out = np.real(a[:, 0, :] * np.conj(b[:, 0, :]))
    if a.shape[1] > 1:
        out += 2.0 * np.sum(
            np.real(np.conj(a[:, 1:, :]) * b[:, 1:, :]), axis=1)
    return out


def _compute_velocity(field: np.ndarray, ky: np.ndarray,
                      C_xy, x_local: bool, nx: int,
                      sign: float = -1.0) -> np.ndarray:
    """
    Compute ExB or A-parallel velocity from a field.

    Parameters
    ----------
    field : np.ndarray
        Complex field, shape ``(nx, nky, nz)``.
    ky : np.ndarray
        ky wavenumber array, shape ``(nky,)``.
    C_xy : float or np.ndarray
        Metric coefficient. Scalar for local, ``(nx, nz)`` for global.
    x_local : bool
        If True, IFFT field to real x first.
    nx : int
        Number of x points.
    sign : float
        -1 for v_E (from phi), +1 for v_A (from A_par).

    Returns
    -------
    np.ndarray
        Velocity field, shape ``(nx, nky, nz)``.
    """
    if x_local:
        vel = nx * np.fft.ifft(field, axis=0)
        scalar_cxy = float(np.ravel(C_xy)[0]) if np.ndim(C_xy) == 0 else float(C_xy)
        ky3 = ky[np.newaxis, :, np.newaxis]
        vel = sign * 1j * ky3 * vel / scalar_cxy
    else:
        vel = field.copy()
        if np.ndim(C_xy) > 0:
            C_xy_3d = np.asarray(C_xy)[:, np.newaxis, :]
        else:
            C_xy_3d = float(C_xy)
        ky3 = ky[np.newaxis, :, np.newaxis]
        vel = sign * 1j * ky3 * vel / C_xy_3d
    return vel


def _compute_es_fluxes(v_E: np.ndarray, moments: list,
                       n0: float, T0: float, mass: float,
                       x_local: bool, nx: int,
                       J_norm: np.ndarray,
                       n_map=None, T_map=None) -> dict:
    """
    Compute electrostatic fluxes for one species at one timestep.

    Parameters
    ----------
    v_E : np.ndarray
        ExB velocity, shape ``(nx, nky, nz)``.
    moments : list of np.ndarray
        [dens, T_par, T_perp, q_par, q_perp, u_par, ...], each ``(nx, nky, nz)``.
    n0, T0, mass : float
        Species reference density, temperature, mass.
    x_local : bool
    nx : int
    J_norm : np.ndarray
        Normalised Jacobian for FSA, shape ``(nx, nz)`` or ``(1, nz)``.
    n_map, T_map : np.ndarray or None
        Profile correction arrays for global runs, shape ``(nx, 1, nz)``.

    Returns
    -------
    dict with keys ``Ges_x``, ``Qes_x``, ``Pes_x``, each shape ``(nx,)``.
    """
    dens   = moments[0]
    T_par  = moments[1]
    T_perp = moments[2]
    u_par  = moments[5]

    if x_local:
        # Transform moments to real x space
        dens_x  = nx * np.fft.ifft(dens, axis=0)
        Tpar_x  = nx * np.fft.ifft(T_par, axis=0)
        Tperp_x = nx * np.fft.ifft(T_perp, axis=0)
        upar_x  = nx * np.fft.ifft(u_par, axis=0)

        G_xz = n0 * _compute_flux(v_E, dens_x)
        Q_xz = n0 * T0 * _compute_flux(
            v_E, 0.5 * Tpar_x + Tperp_x + 1.5 * dens_x)
        P_xz = n0 * mass * _compute_flux(v_E, upar_x)
    else:
        # Global: moments already in real x, apply profile corrections
        if n_map is None:
            n_map_3d = 1.0
            T_map_3d = 1.0
        else:
            n_map_3d = n_map
            T_map_3d = T_map

        G_xz = n0 * _compute_flux(v_E, dens)
        Q_xz = n0 * T0 * _compute_flux(
            v_E,
            (0.5 * T_par + T_perp) * n_map_3d + 1.5 * dens * T_map_3d)
        P_xz = n0 * mass * _compute_flux(v_E, u_par * n_map_3d)

    # Flux-surface average over z
    Ges_x = np.sum(G_xz * J_norm, axis=1)
    Qes_x = np.sum(Q_xz * J_norm, axis=1)
    Pes_x = np.sum(P_xz * J_norm, axis=1)

    return {"Ges_x": Ges_x, "Qes_x": Qes_x, "Pes_x": Pes_x}


def _compute_em_fluxes(v_A: np.ndarray, moments: list,
                       n0: float, T0: float,
                       x_local: bool, nx: int,
                       J_norm: np.ndarray,
                       n_map=None, T_map=None) -> dict:
    """
    Compute electromagnetic fluxes for one species at one timestep.

    Parameters
    ----------
    v_A : np.ndarray
        A-parallel velocity ``i*ky*A_par/C_xy``, shape ``(nx, nky, nz)``.
    moments : list of np.ndarray
    n0, T0 : float
    x_local : bool
    nx : int
    J_norm : np.ndarray
    n_map, T_map : np.ndarray or None

    Returns
    -------
    dict with keys ``Gem_x``, ``Qem_x``, ``Pem_x``, each shape ``(nx,)``.
    """
    dens   = moments[0]
    T_par  = moments[1]
    q_par  = moments[3]
    q_perp = moments[4]
    u_par  = moments[5]

    if x_local:
        upar_x  = nx * np.fft.ifft(u_par, axis=0)
        qpar_x  = nx * np.fft.ifft(q_par, axis=0)
        qperp_x = nx * np.fft.ifft(q_perp, axis=0)
        Tpar_x  = nx * np.fft.ifft(T_par, axis=0)
        dens_x  = nx * np.fft.ifft(dens, axis=0)

        G_xz = n0 * _compute_flux(v_A, upar_x)
        Q_xz = n0 * T0 * _compute_flux(v_A, qpar_x + qperp_x)
        P_xz = n0 * T0 * _compute_flux(v_A, Tpar_x + dens_x)
    else:
        if n_map is None:
            n_map_3d = 1.0
            T_map_3d = 1.0
        else:
            n_map_3d = n_map
            T_map_3d = T_map

        G_xz = n0 * _compute_flux(v_A, u_par * n_map_3d)
        Q_xz = n0 * T0 * _compute_flux(v_A, q_par + q_perp)
        P_xz = n0 * T0 * _compute_flux(
            v_A, T_par * n_map_3d + dens * T_map_3d)

    Gem_x = np.sum(G_xz * J_norm, axis=1)
    Qem_x = np.sum(Q_xz * J_norm, axis=1)
    Pem_x = np.sum(P_xz * J_norm, axis=1)

    return {"Gem_x": Gem_x, "Qem_x": Qem_x, "Pem_x": Pem_x}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Fluxes2D(CachingDiagnostic):
    """
    Compute, cache, and plot radial flux profile diagnostics.

    Results are streamed to an HDF5 file so that re-running only processes
    new time steps.

    Parameters
    ----------
    outfile : str, optional
        Path to the output HDF5 file (default ``'fluxes_2D.h5'``).
    """

    def __init__(self, outfile: str = "fluxes_2D.h5"):
        super().__init__(outfile)

    # ------------------------------------------------------------------
    # HDF5 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_h5(f, species_names: list, nx: int, x: np.ndarray,
                 has_em: bool):
        """Create all datasets in a newly opened HDF5 file handle."""
        f.create_dataset("time", shape=(0,), maxshape=(None,),
                         dtype=np.float64, chunks=True)
        f.create_dataset("x", data=x)
        for name in species_names:
            grp = f.create_group(name)
            for key in ("Qes_x", "Ges_x", "Pes_x"):
                grp.create_dataset(key, shape=(nx, 0), maxshape=(nx, None),
                                   dtype=np.float64, chunks=True)
            if has_em:
                for key in ("Qem_x", "Gem_x", "Pem_x"):
                    grp.create_dataset(key, shape=(nx, 0),
                                       maxshape=(nx, None),
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
                ds.resize((ds.shape[0], n + 1))
                ds[:, n] = val

    # ------------------------------------------------------------------
    # Build global prefactors
    # ------------------------------------------------------------------

    @staticmethod
    def build_prefactors(params: dict, geom: dict,
                         equilibrium_profiles: dict = None) -> dict:
        """
        Build profile-correction arrays for global runs.

        Parameters
        ----------
        params : dict
        geom : dict
        equilibrium_profiles : dict, optional
            ``{species_name: {'T': array, 'n': array}}``

        Returns
        -------
        dict
            ``{species_name: {'n_map': (nx,1,nz), 'T_map': (nx,1,nz)}}``
            or empty dict for local runs.
        """
        if params["general"].get("x_local", True):
            return {}
        if equilibrium_profiles is None:
            return {}

        nz = params["box"]["nz0"]
        nky = params["box"]["nky0"]
        units = params.get("units", {})
        Tref = units.get("Tref", 1.0)
        nref = units.get("nref", 1.0)

        prefactors = {}
        for sp in params["species"]:
            name = sp["name"]
            ep = equilibrium_profiles.get(name)
            if ep is None:
                continue
            T0 = sp["temp"] * Tref
            n0 = sp["dens"] * nref
            # Shape: (nx,) → (nx, 1, nz) for broadcasting with (nx, nky, nz)
            T_prof = np.asarray(ep["T"])
            n_prof = np.asarray(ep["n"])
            T_map = (T_prof / T0)[:, np.newaxis, np.newaxis] * np.ones((1, 1, nz))
            n_map = (n_prof / n0)[:, np.newaxis, np.newaxis] * np.ones((1, 1, nz))
            prefactors[name] = {"n_map": n_map, "T_map": T_map}

        return prefactors

    # ------------------------------------------------------------------
    # Convenience entry point
    # ------------------------------------------------------------------

    @classmethod
    def from_runs(cls, folder: str, runs: list, params, geom: list,
                  coords: list, t_start: float, t_stop: float,
                  equilibrium_profiles: dict = None,
                  outfile: str = "fluxes_2D.h5"):
        """
        Build multi-segment readers and compute flux profiles in one call.

        Parameters
        ----------
        folder : str
            Run directory.
        runs : list of str
            Run suffixes from :func:`~genetools.io.utils.set_runs`.
        params : Params
            Parameter object.
        geom : list of dict
            Geometry dicts from :func:`~genetools.io.geometry.Geometry`.
        coords : list of dict
            Coordinate dicts from :func:`~genetools.io.coordinates.Coordinates`.
        t_start, t_stop : float
            Time window.
        equilibrium_profiles : dict, optional
            Required for global runs.
        outfile : str, optional
            HDF5 output path (default ``'fluxes_2D.h5'``).

        Returns
        -------
        Fluxes2D
            Instance with results cached to *outfile*.
        """
        from genetools.io.data import BinaryReader, MultiSegmentReader

        p0 = params.get(0)

        # Multi-segment field reader
        fld_reader = MultiSegmentReader([
            BinaryReader('field', folder, ext, params.get(fn))
            for fn, ext in enumerate(runs)
        ])

        # Multi-segment moment readers — one per species
        species_names = [sp['name'] for sp in p0['species']]
        mom_readers = [
            MultiSegmentReader([
                BinaryReader('mom', folder, ext, params.get(fn), species=name)
                for fn, ext in enumerate(runs)
            ])
            for name in species_names
        ]

        obj = cls(outfile)
        obj.compute_and_save(fld_reader, mom_readers, coords[0], geom[0],
                             p0, t_start, t_stop, equilibrium_profiles)
        return obj

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
        Stream field + moment files, compute radial flux profiles, and
        append to HDF5.

        Parameters
        ----------
        fld_reader
            Field reader (BinaryReader or MultiSegmentReader).
        mom_readers : list
            One moment reader per species, in same order as params['species'].
        coords : dict
            Coordinate dictionary.
        geom : dict
            Geometry dictionary.
        params : dict
            Parameter dictionary.
        t_start, t_stop : float
            Time window to process.
        equilibrium_profiles : dict, optional
            Required for global runs. ``{species_name: {'T': array, 'n': array}}``.
        """
        # Accept both Params object and plain dict
        if hasattr(params, 'get') and callable(params.get) and not isinstance(params, dict):
            params = params.get(0)
        x_local  = params["general"].get("x_local", True)
        nx       = params["box"]["nx0"]
        n_fields = params["info"]["n_fields"]
        species  = params["species"]
        species_names = [sp["name"] for sp in species]
        ky = np.asarray(coords["ky"])
        x  = np.asarray(coords["x"])

        # Jacobian normalization
        J = geom["Jacobian"]
        if x_local:
            J_norm = (J / J.sum())[np.newaxis, :]   # (1, nz)
        else:
            J_norm = J / J.sum(axis=1, keepdims=True)  # (nx, nz)

        # C_xy
        C_xy = geom["metric"]["C_xy"]

        # Global prefactors
        prefactors = self.build_prefactors(params, geom, equilibrium_profiles)
        has_em = n_fields > 1

        # Sync field + moment indices
        idx_fld, idx_mom = self._sync_field_mom_indices(
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

                # Compute ExB velocity from phi (field index 0)
                phi = fields[0]
                v_E = _compute_velocity(phi, ky, C_xy, x_local, nx,
                                        sign=-1.0)

                # Compute A_par velocity if EM
                v_A = None
                if has_em:
                    A_par = fields[1]
                    v_A = _compute_velocity(A_par, ky, C_xy, x_local, nx,
                                            sign=1.0)

                sp_data = {}
                for i_sp, sp in enumerate(species):
                    name = sp["name"]
                    n0   = sp["dens"]
                    T0   = sp["temp"]
                    mass = sp.get("mass", 1.0)
                    moments = all_moments[i_sp]

                    pf = prefactors.get(name, {})
                    n_map = pf.get("n_map")
                    T_map = pf.get("T_map")

                    result = _compute_es_fluxes(
                        v_E, moments, n0, T0, mass,
                        x_local, nx, J_norm, n_map, T_map)

                    if has_em and v_A is not None:
                        em_result = _compute_em_fluxes(
                            v_A, moments, n0, T0,
                            x_local, nx, J_norm, n_map, T_map)
                        result.update(em_result)

                    sp_data[name] = result

                if not initialised:
                    self._init_h5(hf, species_names, nx, x, has_em)
                    initialised = True

                self._append_to_open_file(hf, species_names, sp_data, tm)

    # ------------------------------------------------------------------
    # Public interface — load
    # ------------------------------------------------------------------

    def load(self, t_start: float = None, t_stop: float = None) -> dict:
        """
        Load saved flux profiles from the HDF5 file.

        Returns
        -------
        dict
            Keys: ``'time'``, ``'x'``, ``'{species}_Qes_x'``, etc.
            Profile arrays have shape ``(n_times, nx)``.
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
            if "x" in f:
                result["x"] = f["x"][...]

            species_names = [k for k in f.keys() if k not in ("time", "x")]
            for name in species_names:
                grp = f[name]
                for key in grp.keys():
                    data = grp[key][:, final_idx]  # (nx, n_selected)
                    result[f"{name}_{key}"] = data.T  # (n_times, nx)

        return result

    def load_time_average(self, t_start: float = None,
                          t_stop: float = None) -> dict:
        """
        Load and time-average flux profiles.

        Returns
        -------
        dict
            Keys: ``'x'``, ``'{species}_Qes_x'``, etc.
            Each value is a 1-D array of shape ``(nx,)``.
        """
        data = self.load(t_start, t_stop)
        if not data or "time" not in data:
            return {}

        time = data["time"]
        result = {}
        if "x" in data:
            result["x"] = data["x"]

        for key, arr in data.items():
            if key in ("time", "x"):
                continue
            if len(time) <= 1:
                result[key] = arr[0] if arr.ndim > 0 else arr
            else:
                result[key] = _trapz(arr, x=time, axis=0) / (time[-1] - time[0])

        return result

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, coords: dict, params: dict,
             t_start: float = None, t_stop: float = None) -> None:
        """
        Plot flux profile diagnostics from the saved HDF5 file.

        Per species produces:
        1. (t, x) heatmaps of Q_es, G_es, P_es [+ EM]
        2. Time-averaged radial profiles

        Parameters
        ----------
        coords : dict
            Coordinate dictionary.
        params : dict
            Parameter dictionary.
        t_start, t_stop : float, optional
            Time window.
        """
        data = self.load(t_start, t_stop)
        if not data or "time" not in data or len(data["time"]) == 0:
            print("No flux data available to plot.")
            return

        times = data["time"]
        x = data.get("x", np.asarray(coords["x"]))
        x_local = params["general"].get("x_local", True)
        species = params["species"]
        n_fields = params["info"]["n_fields"]
        has_em = n_fields > 1

        x_label = (r"$x / \rho_{\rm ref}$" if x_local
                   else r"$x / a$")
        t_label = r"$t\;c_{\rm ref}/L_{\rm ref}$"

        es_keys = ["Qes_x", "Ges_x", "Pes_x"]
        em_keys = ["Qem_x", "Gem_x", "Pem_x"] if has_em else []
        all_keys = es_keys + em_keys

        flux_labels = {
            "Qes_x": r"$Q_{\rm es}\;[Q_{\rm GB}]$",
            "Ges_x": r"$\Gamma_{\rm es}\;[\Gamma_{\rm GB}]$",
            "Pes_x": r"$\Pi_{\rm es}\;[\Pi_{\rm GB}]$",
            "Qem_x": r"$Q_{\rm em}\;[Q_{\rm GB}]$",
            "Gem_x": r"$\Gamma_{\rm em}\;[\Gamma_{\rm GB}]$",
            "Pem_x": r"$\Pi_{\rm em}\;[\Pi_{\rm GB}]$",
        }

        for sp in species:
            name = sp["name"]
            present_keys = [k for k in all_keys
                           if f"{name}_{k}" in data]
            if not present_keys:
                continue

            ncols = len(present_keys)
            nt = len(times)

            # ── Fig 1: (t, x) heatmaps ───────────────────────────────
            if nt > 1:
                fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4),
                                         squeeze=False)
                fig.suptitle(f"{name} — flux evolution")
                for ax, key in zip(axes[0], present_keys):
                    arr = data[f"{name}_{key}"]
                    vmax = np.percentile(np.abs(arr), 98)
                    im = ax.pcolormesh(times, x, arr.T,
                                       cmap="bwr", vmin=-vmax, vmax=vmax,
                                       shading="auto")
                    fig.colorbar(im, ax=ax)
                    ax.set_xlabel(t_label)
                    ax.set_ylabel(x_label)
                    ax.set_title(flux_labels.get(key, key))
                plt.tight_layout()
                plt.show()

            # ── Fig 2: time-averaged profiles ─────────────────────────
            if nt > 1:
                dt = times[-1] - times[0]
                fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4),
                                         squeeze=False)
                title_str = f"average [{times[0]:.3f} - {times[-1]:.3f}]"
                fig.suptitle(f"{name} — {title_str}")
                for ax, key in zip(axes[0], present_keys):
                    arr = data[f"{name}_{key}"]
                    avg = _trapz(arr, x=times, axis=0) / dt
                    ax.plot(x, avg, "-b")
                    ax.axhline(0, color="k", lw=0.5, ls="--")
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(flux_labels.get(key, key))
                    ax.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4),
                                         squeeze=False)
                fig.suptitle(f"{name} — t = {times[0]:.3f}")
                for ax, key in zip(axes[0], present_keys):
                    arr = data[f"{name}_{key}"]
                    ax.plot(x, arr[0], "-b")
                    ax.axhline(0, color="k", lw=0.5, ls="--")
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(flux_labels.get(key, key))
                    ax.grid(True)
                plt.tight_layout()
                plt.show()
