# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
import warnings

import numpy as np
import matplotlib.pyplot as plt
import h5py

from genetools.compat import trapz as _trapz
from genetools.diagnostics._base import (CachingDiagnostic,
                                        RunDiagnostic)
from genetools.diagnostics import _gene3d as g3


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
        C_xy_arr = np.asarray(C_xy)
        if C_xy_arr.ndim == 2:
            C_xy_3d = C_xy_arr[:, np.newaxis, :]       # (nx, 1, nz)
        elif C_xy_arr.ndim == 1:
            C_xy_3d = C_xy_arr[:, np.newaxis, np.newaxis]  # (nx, 1, 1)
        else:
            C_xy_3d = float(C_xy_arr)
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

class Fluxes2D(RunDiagnostic):
    """
    Compute, cache, and plot radial flux profile diagnostics.

    Results are streamed to an HDF5 file so that re-running only processes
    new time steps.

    Parameters
    ----------
    outfile : str, optional
        Path to the output HDF5 file (default ``'fluxes_2D.h5'``).
    """

    name = "fluxes2d"
    cache_file = "fluxes_2D.h5"

    def __init__(self, run=None, outfile: str = None, folder: str = None):
        """
        Parameters
        ----------
        run : genetools.run.Run, optional
        outfile, folder : str, optional
            Override the HDF5 cache location; normally derived from the run.
        """
        if run is not None:
            RunDiagnostic.__init__(self, run)
            if outfile:
                self.outfile = outfile
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
    # HDF5 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_h5(f, species_names: list, nx: int, x: np.ndarray,
                 has_em: bool, time_dtype=np.float64):
        """Create all datasets in a newly opened HDF5 file handle."""
        f.create_dataset("time", shape=(0,), maxshape=(None,),
                         dtype=time_dtype, chunks=True)
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
        import warnings
        from genetools.io.data import BinaryReader, MultiSegmentReader

        p0 = params.get(0)

        # Check grid consistency across segments
        ref_box = (p0["box"]["nx0"], p0["box"]["nky0"], p0["box"]["nz0"])
        for fn in range(1, len(runs)):
            pi = params.get(fn)
            seg_box = (pi["box"]["nx0"], pi["box"]["nky0"], pi["box"]["nz0"])
            if seg_box != ref_box:
                warnings.warn(
                    f"Grid mismatch: segment 0 has {ref_box}, "
                    f"segment {fn} has {seg_box}. "
                    f"Using segment 0 params/geom/coords for all.",
                    stacklevel=2,
                )
                break

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

        obj = cls(outfile, folder=folder)
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
        # Accept Params object or dict, list or single element
        if hasattr(params, 'get') and callable(params.get) and not isinstance(params, dict):
            params = params.get(0)
        if isinstance(coords, list):
            coords = coords[0]
        if isinstance(geom, list):
            geom = geom[0]
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
                    self._init_h5(hf, species_names, nx, x, has_em,
                                  time_dtype=self._time_dtype(params))
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

            time, read_idx, unsort = self._select_window(time, t_start, t_stop)

            result = {"time": time}
            if "x" in f:
                result["x"] = f["x"][...]

            species_names = [k for k in f.keys() if k not in ("time", "x")]
            for name in species_names:
                grp = f[name]
                for key in grp.keys():
                    data = grp[key][:, read_idx][:, unsort]
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

    def dataset(self, t=None, params=None, species=None, t_start=None,
                t_stop=None):
        """
        Return the x-resolved fluxes as an :class:`xarray.Dataset`.

        Called with a time window when bound to a run; the older
        ``dataset(coords, params, species, ...)`` form still works.
        """
        if self.run is not None and not isinstance(t, dict):
            if self.is_3d:
                return self._dataset_3d(t)
            self.compute(t)
            a, b = self._window(t)
            return self._dataset_from_cache(self.coord, self.params,
                                            self.run.species, a, b)
        return self._dataset_from_cache(t, params, species, t_start, t_stop)

    def _dataset_from_cache(self, coords, params, species, t_start=None,
                            t_stop=None):
        """Return the time-averaged x-resolved fluxes as an ``xarray.Dataset``."""
        import xarray as xr
        from genetools import _xr

        raw = self.load_time_average(t_start, t_stop)
        if not raw:
            return xr.Dataset()
        x = np.asarray(raw.get("x", coords.get("x", [])))
        if x.size == 0:
            x = np.asarray(coords.get("x_o_a", []))
        data_vars, used = _xr.stacked_vars(
            raw, species, lambda var: ("x",), coord_keys=("x",))
        return _xr.make_dataset(data_vars, {"x": x}, species=used, params=params)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Run-native front end
    # ------------------------------------------------------------------

    #: GENE-3D moment -> (species prefactor, gyro-Bohm reference, SI unit).
    _FLUXES_3D = {
        "Gamma_es": ("dens", "Ggb", "1e19 m^-2 s^-1"),
        "Gamma_em": ("dens", "Ggb", "1e19 m^-2 s^-1"),
        "Q_es": ("dens_temp", "Qgb", "W m^-2"),
        "Q_em": ("dens_temp", "Qgb", "W m^-2"),
    }

    def compute(self, t=None):
        """
        Stream the data and cache the x-resolved fluxes.

        The spectral geometries reconstruct the fluxes from the potential and the
        moments and append to the HDF5 cache. GENE-3D computes its own fluxes and
        writes them to the moment file, so its path is a flux-surface average of
        data already on disk — no reconstruction, and nothing that depends on the
        ExB normalisation.
        """
        if self.is_3d:
            key = self._key(t)
            if key not in self._cache:
                self._cache[key] = self._compute_3d(t)
            return self._cache[key]
        a, b = self._bounds(t)
        r = self.run
        self.compute_and_save(r.field, [r.mom(n) for n in r.species],
                              self.coord, self.geom, self.params, a, b,
                              equilibrium_profiles=r.eq_profiles)
        return self

    def _prefactor_3d(self, kind: str, species: str) -> float:
        """
        Species factor from the namelist.

        GENE-3D's own ``diag_prof`` applies ``dens`` to the particle fluxes and
        ``dens*temp`` to the heat fluxes — to the electromagnetic parts exactly
        as to the electrostatic ones. (The reference GUI normalises only the
        electrostatic terms, leaving its EM fluxes inconsistent with both its own
        ES fluxes and the code's ``profile_<species>`` output.)
        """
        spec = next(s for s in self.params["species"] if s["name"] == species)
        dens = float(spec.get("dens", 1.0))
        return dens if kind == "dens" else dens * float(spec.get("temp", 1.0))

    def _compute_3d(self, t):
        run = self.run
        J = self.geom["Jacobian"]
        readers = [run.mom(n) for n in run.species]
        # Species files can hold different numbers of complete snapshots, so
        # take only the times all of them have (see _common_indices).
        times, index_of = self._common_indices(readers, t)
        out = {}
        for name, reader in zip(run.species, readers):
            idx = index_of[id(reader)]
            wanted = [v for v in self._FLUXES_3D if g3.has_var(reader, v)]
            slots = {v: reader.index_of(v) for v in wanted}
            stacks = {v: [] for v in wanted}
            for _, arrays in reader.stream_selected(idx):
                for v in wanted:
                    stacks[v].append(
                        g3.flux_surface_average(arrays[slots[v]], J))
            out[name] = {
                v: np.asarray(stacks[v]) * self._prefactor_3d(
                    self._FLUXES_3D[v][0], name)
                for v in wanted}
        return {"species": out, "times": times}

    def _surface_area_3d(self):
        """
        The area that turns a flux density into a total.

        ``norm_flux_projection`` decides, and it is the same flag that decides
        whether GENE-3D's ``flux_geomfac`` carries a ``1/sqrt(g^xx)``: with the
        projection on, the flux is per unit *physical* area and pairs with the
        ``sqrt(g^xx)``-weighted surface area; with it off, it is per unit ``x``
        and pairs with ``dVdx``. Mixing the two rescales every total.
        """
        projected = self.params.get("geometry", {}).get(
            "norm_flux_projection", False)
        return np.asarray(self.geom["area"]["Area" if projected else "dVdx"],
                          dtype=float)

    def _dataset_3d(self, t):
        from genetools._xr import make_dataset, unit_attrs
        raw = self.compute(t)
        params = self.params
        units = params.get("units", {}) or {}
        area = self._surface_area_3d()
        names = [n for n in self.run.species if n in raw["species"]]
        present = [v for v in self._FLUXES_3D
                   if all(v in raw["species"][n] for n in names)]

        data_vars = {}
        for v in present:
            data_vars[v] = (("species", "time", "x"),
                            np.stack([raw["species"][n][v] for n in names],
                                     axis=0))
        for base in ("Gamma", "Q"):
            parts = [v for v in present if v.startswith(base + "_")]
            if not parts:
                continue
            total = sum(np.asarray(data_vars[v][1]) for v in parts)
            data_vars[base + "_total"] = (("species", "time", "x"), total)
            data_vars[base + "_integrated"] = (
                ("species", "time", "x"),
                total * area[np.newaxis, np.newaxis, :])
        # Volume averages, comparable with `nrg`. Every flux gets one, including
        # the ES/EM parts separately, since nrg reports those separately too.
        J = self.geom["Jacobian"]
        for v in list(present) + [b + "_total" for b in ("Gamma", "Q")
                                  if b + "_total" in data_vars]:
            data_vars[v + "_volume"] = (
                ("species", "time"),
                g3.volume_average(np.asarray(data_vars[v][1]), J))

        ds = make_dataset(data_vars,
                          {"x": self.coord.get("x_o_a"), "time": raw["times"]},
                          species=names, params=params)
        ds = ds.assign(
            Area=("x", area),
            dVdx=("x", np.asarray(self.geom["area"]["dVdx"], dtype=float)),
            x_o_rho_ref=("x", np.asarray(self.coord["x"], dtype=float)))
        ds["Area"].attrs["units"] = "m^2 (per dx/Lref)"
        ds["dVdx"].attrs["units"] = "m^3 (per dx/Lref)"

        gb_label = {"Gamma": r"$\Gamma_{gB}$", "Q": r"$Q_{gB}$"}
        for v in present:
            _, ref_key, si_unit = self._FLUXES_3D[v]
            base = v.split("_")[0]
            ds[v].attrs["units"] = gb_label[base]
            ref = units.get(ref_key)
            if ref is not None:
                ds[v + "_SI"] = ds[v] * float(ref)
                ds[v + "_SI"].attrs["units"] = si_unit
        # The totals and area-integrated companions need units too, or the
        # gyro-Bohm figure comes out with blank y-axes.
        for base, dens_unit, int_unit in (
                ("Gamma", "1e19 m^-2 s^-1", "1e19 s^-1"), ("Q", "W m^-2", "W")):
            ref = units.get("Ggb" if base == "Gamma" else "Qgb")
            for key, si_unit in ((base + "_total", dens_unit),
                                 (base + "_integrated", int_unit)):
                if key not in ds:
                    continue
                ds[key].attrs["units"] = (
                    gb_label[base] if key.endswith("_total")
                    else gb_label[base] + r"$\,\mathrm{m^2}$")
                if ref is not None:
                    ds[key + "_SI"] = ds[key] * float(ref)
                    ds[key + "_SI"].attrs["units"] = si_unit
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = self.geometry_kind
        return ds

    def _si_available(self) -> bool:
        """
        Whether the parameter file provides real reference units.

        An all-1.0 ``&units`` block means nobody filled it in, and the SI
        conversion would just relabel gyro-Bohm numbers as watts.
        """
        units = self.params.get("units", {}) or {}
        return any(float(units.get(k, 1.0)) != 1.0
                   for k in ("Lref", "Bref", "Tref", "nref", "mref"))

    def volume_average(self, t=None):
        """
        GENE-3D only: the volume-averaged fluxes, for comparison with ``nrg``.

        The reduction is the Jacobian-weighted mean over all of x, y and z that
        ``nrg`` reports. It is built in two steps — the ``(y, z)`` average this
        diagnostic already does per surface, then an x-average weighted by
        ``sum_{y,z} J`` — which is exact, not an approximation. A plain mean over
        x is *not* this, and neither is a ``dVdx``-weighted one: ``dVdx`` carries
        a radially varying ``C_y`` factor.

        Parameters
        ----------
        t : (float, float), optional
            Time window; a negative bound means "unbounded". A degenerate window
            such as ``(t0, t0)`` selects the single output time at ``t0``, which
            streams one snapshot instead of the whole run.

        The species factor is always applied — ``dens`` to the particle fluxes
        and ``dens*temp`` to the heat fluxes, to the electromagnetic parts
        exactly as to the electrostatic ones. That is what ``nrg`` does
        (``diag_3d.F90`` lines 698-701, after ``sum_3d_real``), and the mom file
        this reads carries no species factor of its own (written at line 547,
        before those lines).

        Returns
        -------
        xarray.Dataset
            Dims ``(species, time)``, one variable per flux, plus the ES+EM
            totals.
        """
        self._require("xy_global")
        import xarray as xr

        raw = self.compute(t)
        J = self.geom["Jacobian"]
        names = [n for n in self.run.species if n in raw["species"]]
        present = [v for v in self._FLUXES_3D
                   if all(v in raw["species"][n] for n in names)]

        data_vars, per_species = {}, {}
        for v in present:
            # compute() has already applied the species factor.
            stack = [g3.volume_average(np.asarray(raw["species"][n][v]), J)
                     for n in names]
            data_vars[v] = (("species", "time"), np.stack(stack, axis=0))
            per_species[v] = data_vars[v][1]
        for base in ("Gamma", "Q"):
            parts = [v for v in present if v.startswith(base + "_")]
            if parts:
                data_vars[base + "_total"] = (
                    ("species", "time"),
                    sum(per_species[v] for v in parts))

        ds = xr.Dataset(data_vars,
                        coords={"species": names, "time": raw["times"]})
        ds.attrs["reduction"] = "sum_xyz f J / sum_xyz J  (nrg convention)"
        ds.attrs["species_factor"] = ("dens (particle), dens*temp (heat), "
                                      "ES and EM alike")
        return ds

    def _plot_3d(self, t, si=None, x_avg_lims=None, buffer_frac=0.1,
                 show_map=True, show_traces=None, components=True, **kw):
        """
        Time-averaged radial flux profiles, and the ``(x, t)`` map.

        Draws the gyro-Bohm figure and, when the parameter file carries real
        reference units, an SI figure alongside it (flux density in W/m^2 or
        1e19 m^-2 s^-1, and the area-integrated total in W or 1e19 s^-1).
        ``si=False`` gives gyro-Bohm only, ``si=True`` SI only.

        Each panel shows the total and its electrostatic and electromagnetic
        parts; pass ``components=False`` for totals alone.

        Returns the list of figures drawn.
        """
        ds = self._dataset_3d(t)
        x = np.asarray(ds["x"])
        sl = g3.radial_slice(x, limits=x_avg_lims, buffer_frac=buffer_frac)
        bases = [b for b in ("Q", "Gamma") if f"{b}_total" in ds]
        if not bases:
            raise ValueError("No flux variables available to plot.")

        si_ok = self._si_available()
        show_gb = si in (None, False)
        show_si = si is True or (si is None and si_ok)
        if si is True and not si_ok:
            warnings.warn(
                "fluxes2d: SI requested but the parameter file has no reference "
                "units; plotting gyro-Bohm instead.", RuntimeWarning)
            show_gb, show_si = True, False
        if show_traces is not None:      # older keyword
            show_map = show_traces

        figs = []
        if show_gb:
            figs.append(self._plot_3d_profiles(ds, x, sl, bases, si=False,
                                               components=components))
        if show_si:
            figs.append(self._plot_3d_profiles(ds, x, sl, bases, si=True,
                                               components=components))
        if show_map:
            figs.append(self._plot_3d_map(ds, x, bases, si=show_si and not show_gb))
        plt.show()
        return figs

    #: Component -> line style. Colour distinguishes species, style the
    #: electrostatic/electromagnetic split, so both read off one legend.
    _COMPONENT_STYLE = (("total", "-", 1.8), ("es", "--", 1.2), ("em", ":", 1.2))

    def _plot_3d_profiles(self, ds, x, sl, bases, si: bool, components=True):
        """
        One row per flux: the density profile and the area-integrated total.

        Each panel shows the total (solid) and, unless *components* is false, the
        electrostatic (dashed) and electromagnetic (dotted) parts. Colour is the
        species, line style the component. The quoted mean is over the radial
        window marked by the dashed verticals, and only for the total — that is
        the number usually reported.
        """
        fig, axes = plt.subplots(len(bases), 2,
                                 figsize=(12, 3.8 * len(bases)), squeeze=False)
        colours = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])

        for row, base in enumerate(bases):
            symbol = r"\Gamma" if base == "Gamma" else "Q"
            for col, kind in enumerate(("total", "integrated")):
                ax = axes[row][col]
                for si_name, colour in zip(ds["species"].values, colours):
                    for comp, style, lw in self._COMPONENT_STYLE:
                        if comp != "total" and not components:
                            continue
                        stem = (f"{base}_{kind}" if comp == "total"
                                else f"{base}_{comp}")
                        # The components have no area-integrated companion; scale
                        # them by the same area rather than omitting them.
                        key = f"{stem}_SI" if si else stem
                        if key not in ds:
                            if comp == "total":
                                raise KeyError(
                                    f"{key} is not in the dataset. An SI figure "
                                    "must not quietly fall back to gyro-Bohm "
                                    "values under an SI axis label; check that "
                                    "&units carries real reference quantities.")
                            continue
                        da = self._t_average(ds[key].sel(species=si_name))
                        trace = np.asarray(da)
                        if comp != "total" and kind == "integrated":
                            trace = trace * np.asarray(ds["Area"])
                        label = (f"{si_name} (mean {trace[sl].mean():.3g})"
                                 if comp == "total" else f"{si_name} {comp.upper()}")
                        ax.plot(x, trace, ls=style, lw=lw, color=colour,
                                label=label)
                ax.axhline(0.0, lw=0.6, color="grey")
                ax.set_xlabel(r"$x/a$")
                unit_key = (f"{base}_{kind}_SI" if si else f"{base}_{kind}")
                ax.set_ylabel(ds[unit_key].attrs.get("units", ""))
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7, ncol=1)
                # Mark the window the quoted means are taken over.
                ax.axvline(x[sl][0], ls="--", lw=0.8, color="k")
                ax.axvline(x[sl][-1], ls="--", lw=0.8, color="k")
            axes[row][0].set_title(rf"$\langle {symbol} \rangle_{{FS,t}}$")
            axes[row][1].set_title(
                rf"$\langle {symbol} \rangle_{{FS,t}} \times A$")
        fig.suptitle("Radial flux profiles " + ("[SI]" if si else "[gyro-Bohm]")
                     + ("  (solid total, dashed ES, dotted EM)" if components
                        else ""))
        fig.tight_layout()
        return fig

    def _plot_3d_map(self, ds, x, bases, si: bool):
        """``(x, t)`` heatmap per flux and species — shows avalanches and drift."""
        times = np.asarray(ds["time"])
        names = list(ds["species"].values)
        fig, axes = plt.subplots(len(bases), len(names),
                                 figsize=(5.2 * len(names), 3.6 * len(bases)),
                                 squeeze=False)
        for row, base in enumerate(bases):
            key = f"{base}_total"
            if si and f"{key}_SI" in ds:
                key = f"{key}_SI"
            arr_all = np.asarray(ds[key])
            # One symmetric colour scale per flux, so species are comparable.
            vmax = float(np.max(np.abs(arr_all))) or 1.0
            for col, name in enumerate(names):
                ax = axes[row][col]
                mesh = ax.pcolormesh(times, x,
                                     np.asarray(ds[key].sel(species=name)).T,
                                     shading="nearest", cmap="RdBu_r",
                                     vmin=-vmax, vmax=vmax)
                ax.set_title(f"{key} — {name}")
                ax.set_xlabel(r"$t\;[L_{\rm ref}/c_{\rm ref}]$")
                ax.set_ylabel(r"$x/a$")
                cb = fig.colorbar(mesh, ax=ax)
                cb.set_label(ds[key].attrs.get("units", ""))
        fig.suptitle("Flux evolution")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------

    def plot(self, t=None, show_heatmaps=False, **kw):
        """Plot the x-resolved fluxes over the window *t*."""
        if self.is_3d:
            return self._plot_3d(t, **kw)
        self.compute(t)
        a, b = self._bounds(t)
        return self._plot_spectral(self.coord, self.params, a, b,
                                   show_heatmaps=show_heatmaps, **kw)

    def _plot_spectral(self, coords: dict, params: dict,
             t_start: float = None, t_stop: float = None,
             show_heatmaps: bool = False) -> None:
        """
        Plot flux profile diagnostics from the saved HDF5 file.

        Per species produces a figure with two subplots:
        - Heat flux Q(x) — ES and EM overlaid
        - Particle flux Gamma(x) — ES and EM overlaid

        Optionally shows (t, x) heatmaps for each flux component.

        Parameters
        ----------
        coords : dict or list of dict
            Coordinate dictionary.
        params : dict or Params
            Parameter dictionary.
        t_start, t_stop : float, optional
            Time window.
        show_heatmaps : bool, optional
            Show (t, x) heatmaps in addition to time-averaged profiles
            (default False).
        """
        # Accept Params object or dict, list or single element
        if hasattr(params, 'get') and callable(params.get) and not isinstance(params, dict):
            params = params.get(0)
        if isinstance(coords, list):
            coords = coords[0]

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
        nt = len(times)

        x_label = (r"$x / \rho_{\rm ref}$" if x_local
                   else r"$x / a$")
        t_label = r"$t\;c_{\rm ref}/L_{\rm ref}$"

        # Flux groups: (label, es_key, em_key)
        flux_groups = [
            (r"$Q\;[Q_{\rm GB}]$", "Qes_x", "Qem_x"),
            (r"$\Gamma\;[\Gamma_{\rm GB}]$", "Ges_x", "Gem_x"),
        ]

        for sp in species:
            name = sp["name"]

            # ── Optional heatmaps ─────────────────────────────────────
            if show_heatmaps and nt > 1:
                hm_keys = ["Qes_x", "Ges_x"]
                if has_em:
                    hm_keys += ["Qem_x", "Gem_x"]
                present_hm = [k for k in hm_keys if f"{name}_{k}" in data]
                if present_hm:
                    hm_labels = {
                        "Qes_x": r"$Q_{\rm es}$", "Ges_x": r"$\Gamma_{\rm es}$",
                        "Qem_x": r"$Q_{\rm em}$", "Gem_x": r"$\Gamma_{\rm em}$",
                    }
                    ncols = len(present_hm)
                    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4),
                                             squeeze=False)
                    fig.suptitle(f"{name} — flux evolution")
                    for ax, key in zip(axes[0], present_hm):
                        arr = data[f"{name}_{key}"]
                        vmax = np.percentile(np.abs(arr), 98)
                        im = ax.pcolormesh(times, x, arr.T,
                                           cmap="bwr", vmin=-vmax, vmax=vmax,
                                           shading="auto")
                        fig.colorbar(im, ax=ax)
                        ax.set_xlabel(t_label)
                        ax.set_ylabel(x_label)
                        ax.set_title(hm_labels.get(key, key))
                    plt.tight_layout()
                    plt.show()

            # ── Time-averaged profiles: Q and Gamma ───────────────────
            # Compute time averages
            avgs = {}
            for key in ["Qes_x", "Qem_x", "Ges_x", "Gem_x"]:
                full_key = f"{name}_{key}"
                if full_key not in data:
                    continue
                arr = data[full_key]
                if nt > 1:
                    avgs[key] = _trapz(arr, x=times, axis=0) / (times[-1] - times[0])
                else:
                    avgs[key] = arr[0]

            if not avgs:
                continue

            if nt > 1:
                title_str = f"{name} — average [{times[0]:.3f}, {times[-1]:.3f}]"
            else:
                title_str = f"{name} — t = {times[0]:.3f}"

            fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
            fig.suptitle(title_str)

            for ax, (ylabel, es_key, em_key) in zip(axes[0], flux_groups):
                if es_key in avgs:
                    ax.plot(x, np.ravel(avgs[es_key]), "-b", label="ES")
                if em_key in avgs:
                    ax.plot(x, np.ravel(avgs[em_key]), "--m", label="EM")
                if es_key in avgs and em_key in avgs:
                    ax.plot(x, np.ravel(avgs[es_key] + avgs[em_key]),
                            "-k", lw=1.5, label="total")
                ax.axhline(0, color="k", lw=0.5, ls=":")
                ax.set_xlabel(x_label)
                ax.set_ylabel(ylabel)
                ax.legend()
                ax.grid(True)

            plt.tight_layout()
            plt.show()

    def plot_SI(self, coords, params, geom,
                t_start: float = None, t_stop: float = None) -> None:
        """
        Plot time-averaged flux profiles in SI units.

        Heat flux is converted to W and particle flux to 1/s by
        multiplying with the gyro-Bohm normalisation and the
        flux-surface area: ``Q_SI = Q_GB * Qgb * Area``.

        Parameters
        ----------
        coords : dict or list of dict
            Coordinate dictionary.
        params : dict or Params
            Parameter dictionary (must contain ``units`` block with
            ``Qgb``, ``Ggb``).
        geom : dict or list of dict
            Geometry dictionary (must contain ``area.Area``).
        t_start, t_stop : float, optional
            Time window.
        """
        if hasattr(params, 'get') and callable(params.get) and not isinstance(params, dict):
            params = params.get(0)
        if isinstance(coords, list):
            coords = coords[0]
        if isinstance(geom, list):
            geom = geom[0]

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
        nt = len(times)

        units = params["units"]
        Qgb = units["Qgb"]                       # [W / m^2]
        Ggb = units["Ggb"] * 1e19                 # nref is in 10^19 m^-3 → [1 / (m^2 s)]
        Area = np.squeeze(np.asarray(geom["area"]["Area"]))  # [m^2], 1D (nx,)

        x_label = (r"$x / \rho_{\rm ref}$" if x_local else r"$x / a$")

        # SI flux groups: (ylabel, es_key, em_key, norm_factor)
        flux_groups_SI = [
            (r"$Q\;[\mathrm{W}]$",            "Qes_x", "Qem_x", Qgb * Area),
            (r"$\Gamma\;[\mathrm{s}^{-1}]$",  "Ges_x", "Gem_x", Ggb * Area),
        ]

        for sp in species:
            name = sp["name"]

            avgs = {}
            for key in ["Qes_x", "Qem_x", "Ges_x", "Gem_x"]:
                full_key = f"{name}_{key}"
                if full_key not in data:
                    continue
                arr = data[full_key]
                if nt > 1:
                    avgs[key] = _trapz(arr, x=times, axis=0) / (times[-1] - times[0])
                else:
                    avgs[key] = arr[0]

            if not avgs:
                continue

            if nt > 1:
                title_str = f"{name} — SI units — average [{times[0]:.3f}, {times[-1]:.3f}]"
            else:
                title_str = f"{name} — SI units — t = {times[0]:.3f}"

            fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
            fig.suptitle(title_str)

            for ax, (ylabel, es_key, em_key, norm) in zip(axes[0], flux_groups_SI):
                if es_key in avgs:
                    ax.plot(x, np.ravel(avgs[es_key] * norm), "-b", label="ES")
                if em_key in avgs:
                    ax.plot(x, np.ravel(avgs[em_key] * norm), "--m", label="EM")
                if es_key in avgs and em_key in avgs:
                    ax.plot(x, np.ravel((avgs[es_key] + avgs[em_key]) * norm),
                            "-k", lw=1.5, label="total")
                ax.axhline(0, color="k", lw=0.5, ls=":")
                ax.set_xlabel(x_label)
                ax.set_ylabel(ylabel)
                ax.legend()
                ax.grid(True)

            plt.tight_layout()
            plt.show()
