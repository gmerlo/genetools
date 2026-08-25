# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
profiles.py — Radial profile diagnostic for GENE simulations.

Computes flux-surface-averaged radial profiles of temperature, density,
and parallel velocity from moment data, caches them to HDF5, and provides
time-averaged plotting with equilibrium background overlay.

Supports both **local** (x_local=True, spectral in x) and **global**
(x_local=False, real-space radial grid) geometry, following the same
conventions as the MATLAB ``diag_profiles.m`` + ``plot_profiles.m``.

Physics
-------
From moment data at each timestep:
  - Extract ky=0 (zonal) component
  - Temperature: ``T = (1/3)*T_par + (2/3)*T_perp``
  - Flux-surface average: ``<f> = sum(f * J_norm, axis=z)``
  - For local geometry: IFFT from kx to x before averaging

Logarithmic gradients:
  - Local: ``omt = -dT/dx / dx_n`` (profile derivative)
  - Global: ``omt = -d(ln T)/dx / dx_n`` (logarithmic derivative)

Example
-------
>>> from genetools.io import BinaryReader, Params, Geometry, Coordinates, set_runs
>>> from genetools.diagnostics import Profiles
>>>
>>> folder = '/path/to/run/'
>>> runs   = set_runs(folder)
>>> params = Params(folder, runs)
>>> geom   = Geometry(folder, runs, params)
>>> coord  = Coordinates(folder, runs, params)
>>>
>>> mom_readers = {
...     sp['name']: BinaryReader('mom', folder, runs[0], params.get(0), species=sp['name'])
...     for sp in params.get(0)['species']
... }
>>>
>>> prof = Profiles()
>>> prof.compute_and_save(mom_readers, coord[0], geom[0], params.get(0),
...                       t_start=10., t_stop=2000.)
>>> prof.plot(coord[0], params.get(0))
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

from genetools.compat import trapz as _trapz
from genetools.diagnostics._base import (CachingDiagnostic,
                                        RunDiagnostic)
from genetools.diagnostics import _gene3d as g3


# ---------------------------------------------------------------------------
# Finite-difference derivative (matches MATLAB computeDerivative(F, 2, 1))
# ---------------------------------------------------------------------------

def _finite_diff(f: np.ndarray) -> np.ndarray:
    """
    2nd-order central finite difference derivative.

    Uses 1st-order one-sided differences at the boundaries, matching
    the MATLAB ``computeDerivative(F, 2, 1)`` convention.

    Parameters
    ----------
    f : np.ndarray
        1-D array of function values.

    Returns
    -------
    np.ndarray
        Derivative array, same shape as *f*.
    """
    d = np.empty_like(f)
    if len(f) < 3:
        if len(f) == 2:
            d[0] = f[1] - f[0]
            d[1] = f[1] - f[0]
        elif len(f) == 1:
            d[0] = 0.0
        return d
    d[1:-1] = (f[2:] - f[:-2]) * 0.5
    d[0]    = f[1] - f[0]
    d[-1]   = f[-1] - f[-2]
    return d


# ---------------------------------------------------------------------------
# Core computation — flux-surface averaged profiles from one phi snapshot
# ---------------------------------------------------------------------------

def _compute_fsa_profiles(moments: list, x_local: bool, J_norm: np.ndarray,
                          nx: int) -> tuple:
    """
    Compute flux-surface averaged T, n, u from moment arrays.

    Parameters
    ----------
    moments : list of np.ndarray
        Moment arrays in GENE order: [dens, T_par, T_perp, q_par, q_perp, u_par, ...].
        Each has shape ``(nx, nky, nz)``.
    x_local : bool
        True for local (spectral x) geometry.
    J_norm : np.ndarray
        Normalised Jacobian. Shape ``(nz,)`` for local, ``(nx, nz)`` for global.
    nx : int
        Number of radial grid points.

    Returns
    -------
    T_fsa, n_fsa, u_fsa : np.ndarray
        Each shape ``(nx,)`` — flux-surface averaged profiles.
    """
    dens   = moments[0][:, 0, :]     # (nx, nz)
    T_par  = moments[1][:, 0, :]
    T_perp = moments[2][:, 0, :]
    u_par  = moments[5][:, 0, :]

    T_ky0 = (1.0/3.0) * T_par + (2.0/3.0) * T_perp

    if x_local:
        # IFFT from kx to x (no *nx normalization — matches MATLAB diag_profiles)
        T_ky0  = np.fft.ifft(T_ky0, axis=0).real
        dens   = np.fft.ifft(dens, axis=0).real
        u_par  = np.fft.ifft(u_par, axis=0).real

    # Flux-surface average: sum over z with Jacobian weight
    T_fsa = np.sum(T_ky0 * J_norm, axis=1)
    n_fsa = np.sum(dens  * J_norm, axis=1)
    u_fsa = np.sum(u_par * J_norm, axis=1)

    return T_fsa, n_fsa, u_fsa


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Profiles(RunDiagnostic):
    """
    Compute, cache, and plot radial profile diagnostics.

    Results are streamed to an HDF5 file so that re-running only processes
    new time steps.

    Parameters
    ----------
    outfile : str, optional
        Path to the output HDF5 file (default ``'profiles.h5'``).
    """

    name = "profiles"
    cache_file = "profiles.h5"

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
    def _init_h5(f, species_names: list, nx: int, time_dtype=np.float64):
        """Create all datasets in a newly opened HDF5 file handle."""
        f.create_dataset("time", shape=(0,), maxshape=(None,),
                         dtype=time_dtype, chunks=True)
        for name in species_names:
            grp = f.create_group(name)
            for key in ("T", "n", "u"):
                grp.create_dataset(key, shape=(nx, 0), maxshape=(nx, None),
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
            for key in ("T", "n", "u"):
                ds = f[f"{name}/{key}"]
                ds.resize((ds.shape[0], n + 1))
                ds[:, n] = species_data[name][key]

    # ------------------------------------------------------------------
    # Public interface — compute
    # ------------------------------------------------------------------

    def compute_and_save(
        self,
        mom_readers: dict,
        coords: dict,
        geom: dict,
        params: dict,
        t_start: float,
        t_stop: float,
    ) -> None:
        """
        Stream moment files, compute FSA profiles, and append to HDF5.

        Skips time steps already present in the output file.

        Parameters
        ----------
        mom_readers : dict
            ``{species_name: reader}`` — one moment reader per species.
        coords : dict
            Coordinate dictionary (from ``Coordinates()``).
        geom : dict
            Geometry dictionary (from ``Geometry()``).
        params : dict
            Parameter dictionary for this segment.
        t_start, t_stop : float
            Time window to process.
        """
        x_local = params["general"].get("x_local", True)
        nx = params["box"]["nx0"]
        species = params["species"]
        species_names = [sp["name"] for sp in species]

        # Precompute Jacobian normalization
        J = geom["Jacobian"]
        if x_local:
            # Local: J is (nz,), broadcast to (nx, nz)
            J_norm = (J / J.sum())[np.newaxis, :]   # (1, nz)
        else:
            # Global: J is (nx, nz)
            J_norm = J / J.sum(axis=1, keepdims=True)  # (nx, nz)

        saved_times = self._load_saved_times()

        # Use the first species reader to determine time indices
        first_reader = next(iter(mom_readers.values()))
        all_times = first_reader.read_all_times()
        mask = (all_times >= t_start) & (all_times <= t_stop)
        candidate_indices = np.where(mask)[0]

        # Filter out already-saved times
        indices = [i for i in candidate_indices
                   if not self._is_already_saved(float(all_times[i]), saved_times)]

        if len(indices) == 0:
            return

        # Set up streaming iterators for each species
        iterators = {name: mom_readers[name].stream_selected(indices)
                     for name in species_names}

        with h5py.File(self.outfile, "a") as hf:
            initialised = "time" in hf

            for _ in indices:
                # Read moments for all species at this timestep
                sp_data = {}
                tm = None
                for name in species_names:
                    t, moments = next(iterators[name])
                    if tm is None:
                        tm = t

                    T_fsa, n_fsa, u_fsa = _compute_fsa_profiles(
                        moments, x_local, J_norm, nx)
                    sp_data[name] = {"T": T_fsa, "n": n_fsa, "u": u_fsa}

                if not initialised:
                    self._init_h5(hf, species_names, nx,
                                  time_dtype=self._time_dtype(params))
                    initialised = True

                self._append_to_open_file(hf, species_names, sp_data, tm)

    # ------------------------------------------------------------------
    # Public interface — load
    # ------------------------------------------------------------------

    def load(self, t_start: float = None, t_stop: float = None) -> dict:
        """
        Load saved profiles from the HDF5 file.

        Returns
        -------
        dict
            Keys: ``'time'``, ``'{species}_T'``, ``'{species}_n'``,
            ``'{species}_u'``.  Time is sorted; profile arrays have shape
            ``(n_times, nx)``.
        """
        if not os.path.exists(self.outfile):
            return {}

        with h5py.File(self.outfile, "r") as f:
            time = f["time"][...]
            if time.size == 0:
                return {}

            time, read_idx, unsort = self._select_window(time, t_start, t_stop)

            result = {"time": time}

            # Discover species from group names
            species_names = [k for k in f.keys() if k != "time"]
            for name in species_names:
                grp = f[name]
                for key in ("T", "n", "u"):
                    if key in grp:
                        # Dataset shape is (nx, n_times); read selected columns
                        data = grp[key][:, read_idx][:, unsort]
                        result[f"{name}_{key}"] = data.T  # (n_times, nx)

        return result

    def dataset(self, t=None, params=None, species=None, t_start=None,
                t_stop=None, equilibrium_profiles=None):
        """
        Return the radial profiles as an :class:`xarray.Dataset`.

        Called with a time window when bound to a run; the older
        ``dataset(coords, params, species, ...)`` form still works.
        """
        if self.run is not None and not isinstance(t, dict):
            if self.is_3d:
                return self._dataset_3d(t)
            self.compute(t)
            a, b = self._window(t)
            return self._dataset_from_cache(
                self.coord, self.params, self.run.species, a, b,
                equilibrium_profiles=(equilibrium_profiles
                                      if equilibrium_profiles is not None
                                      else self.run.eq_profiles))
        return self._dataset_from_cache(t, params, species, t_start, t_stop,
                                        equilibrium_profiles)

    def _dataset_from_cache(self, coords, params, species, t_start=None,
                            t_stop=None,
                equilibrium_profiles=None):
        """
        Return the radial profiles as an ``xarray.Dataset`` (dims species, time, x).

        Variables: the cached fluctuating ``T``/``n``/``u`` and — when the
        equilibrium background can be built (always for local runs; for global
        runs only if *equilibrium_profiles* is supplied) — the background-inclusive
        totals ``T_total``/``n_total``/``u_total`` and normalised gradients
        ``omt``/``omn``/``omu``, exactly as shown by :meth:`plot`.
        """
        import xarray as xr
        from genetools import _xr

        raw = self.load(t_start, t_stop)
        if not raw:
            return xr.Dataset()
        time = np.asarray(raw.get("time", []))
        try:
            self._augment_with_totals(raw, coords, params, equilibrium_profiles)
        except Exception:
            pass  # global run without equilibrium profiles -> fluctuations only
        data_vars, used = _xr.stacked_vars(
            raw, species, lambda var: ("time", "x"), coord_keys=("time",))
        x = np.asarray(coords.get("x", []))
        if x.size == 0:
            x = np.asarray(coords.get("x_o_a", []))
        candidates = {"time": time, "x": x}
        return _xr.make_dataset(data_vars, candidates, species=used, params=params)

    def _augment_with_totals(self, raw, coords, params, equilibrium_profiles):
        """Add background-inclusive totals and gradients to *raw* in place."""
        x_local = params["general"].get("x_local", True)
        units = params.get("units", {})
        geom_p = params.get("geometry", {})
        rhostar = geom_p.get("rhostar", units.get("rho_starref", 1.0 / 500))
        minor_r = geom_p.get("minor_r", 1.0)
        x = np.asarray(coords["x"])
        if x_local:
            dx_n = (x[1] - x[0]) * rhostar * minor_r if len(x) > 1 else 1.0
        else:
            dx_n = (x[1] - x[0]) * minor_r if len(x) > 1 else 1.0

        backgrounds = self.build_background(params, coords, equilibrium_profiles)
        for sp in params["species"]:
            name = sp["name"]
            T_f, n_f, u_f = (raw.get(f"{name}_T"), raw.get(f"{name}_n"),
                             raw.get(f"{name}_u"))
            if T_f is None:
                continue
            bg = backgrounds[name]
            for key, fluct, back in (("T", T_f, bg["T_back"]),
                                     ("n", n_f, bg["n_back"]),
                                     ("u", u_f, bg["u_back"])):
                total = back[np.newaxis, :] + rhostar * fluct * minor_r
                grad = np.array([self._compute_gradient(total[i], dx_n, x_local)
                                 for i in range(total.shape[0])])
                raw[f"{name}_{key}_total"] = total
                raw[f"{name}_om{key.lower()}"] = grad   # omt / omn / omu

    # ------------------------------------------------------------------
    # Background profiles
    # ------------------------------------------------------------------

    @staticmethod
    def build_background(params: dict, coords: dict,
                         equilibrium_profiles: dict = None) -> dict:
        """
        Construct equilibrium background profiles for all species.

        Parameters
        ----------
        params : dict
            Parameter dictionary.
        coords : dict
            Coordinate dictionary.
        equilibrium_profiles : dict, optional
            ``{species_name: {'T': array, 'n': array, 'omt': array, 'omn': array}}``
            Required for global runs. For local runs, background is computed
            analytically from ``omt``/``omn`` in the params.

        Returns
        -------
        dict
            ``{species_name: {'T_back': array, 'n_back': array, 'u_back': array,
            'omt_back': array, 'omn_back': array}}``
        """
        x_local = params["general"].get("x_local", True)
        species = params["species"]
        units = params.get("units", {})
        geom_p = params.get("geometry", {})

        # Resolve rhostar
        rhostar = geom_p.get("rhostar",
                             units.get("rho_starref", 1.0 / 500))
        minor_r = geom_p.get("minor_r", 1.0)

        backgrounds = {}

        for sp in species:
            name = sp["name"]

            if x_local:
                x = np.asarray(coords["x"])
                omt = sp.get("omt", 0.0)
                omn = sp.get("omn", 0.0)

                T_back = 1.0 - x * omt * minor_r * rhostar
                n_back = 1.0 - x * omn * minor_r * rhostar
                u_back = np.zeros_like(x)
                omt_back = omt * np.ones_like(x)
                omn_back = omn * np.ones_like(x)

            else:
                if equilibrium_profiles is None or name not in equilibrium_profiles:
                    raise ValueError(
                        f"Equilibrium profiles required for global geometry "
                        f"(species '{name}'). Pass equilibrium_profiles dict.")

                ep = equilibrium_profiles[name]
                Tref = units.get("Tref", 1.0)
                nref = units.get("nref", 1.0)
                T0 = sp["temp"] * Tref
                n0 = sp["dens"] * nref

                T_back = np.asarray(ep["T"]) / T0
                n_back = np.asarray(ep["n"]) / n0
                u_back = np.zeros_like(T_back)
                omt_back = np.asarray(ep["omt"])
                omn_back = np.asarray(ep["omn"])

            backgrounds[name] = {
                "T_back": T_back,
                "n_back": n_back,
                "u_back": u_back,
                "omt_back": omt_back,
                "omn_back": omn_back,
            }

        return backgrounds

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_gradient(total_profile: np.ndarray, dx_n: float,
                          x_local: bool) -> np.ndarray:
        """
        Compute normalised gradient of a profile.

        Parameters
        ----------
        total_profile : np.ndarray
            1-D radial profile (equilibrium + fluctuation).
        dx_n : float
            Normalised grid spacing (``dx * rhostar * minor_r`` for local,
            ``dx * minor_r`` for global).
        x_local : bool
            If True, compute ``-dF/dx / dx_n``.
            If False, compute ``-d(ln F)/dx / dx_n``.

        Returns
        -------
        np.ndarray
            Gradient array, same shape as *total_profile*.
        """
        if x_local:
            return -_finite_diff(total_profile) / dx_n
        else:
            return -_finite_diff(np.log(total_profile)) / dx_n

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Run-native front end
    # ------------------------------------------------------------------

    def compute(self, t=None):
        """
        Stream the moment files and cache the radial profiles.

        The spectral geometries append to the HDF5 cache; GENE-3D keeps its
        result in memory — one radial profile per output time per species is
        small, and there is no partial-window streaming to amortise.
        """
        if self.is_3d:
            key = self._key(t)
            if key not in self._cache:
                self._cache[key] = self._compute_3d(t)
            return self._cache[key]
        a, b = self._bounds(t)
        r = self.run
        self.compute_and_save({n: r.mom(n) for n in r.species}, self.coord,
                              self.geom, self.params, a, b)
        return self

    def _background_3d(self, species):
        """
        Background ``(n_0, T_0)`` in units of ``n_ref`` / ``T_ref``.

        ``profiles_<species>`` stores ``Tref*temp*temp_prof`` (``profiles.F90``),
        so dividing by ``Tref`` alone leaves ``temp*temp_prof`` — the same
        normalisation GENE-3D's own ``profile_<species>`` uses, which writes
        ``spec%temp*(temp_prof + ...)`` (``diag_3d.F90``). Dividing by ``temp``
        as well would normalise to the *species* temperature instead, putting the
        dataset a factor ``temp`` away from the code's own output and making the
        ``T_ref`` label and the SI conversion wrong by that factor.
        """
        units = self.params["units"]
        prof = self.run.eq_profiles[species]
        T0 = np.asarray(prof["T"], dtype=float) / float(units["Tref"])
        n0 = np.asarray(prof["n"], dtype=float) / float(units["nref"])
        return n0, T0

    def _compute_3d(self, t):
        """
        Total T and n profiles, exactly as GENE-3D's ``diag_prof`` builds them.

        ``T = T_0 + rhostar*minor_r*<T_par/3 + 2 T_perp/3>_FS`` and likewise for
        the density. The ``rhostar*minor_r`` factor converts the normalised
        perturbation into the background's units; without it the perturbation
        comes out ``1/rhostar`` too large, two orders of magnitude for a typical
        run.
        """
        run = self.run
        params = self.params
        J = self.geom["Jacobian"]
        x_o_a = np.asarray(self.coord["x_o_a"], dtype=float)
        scale = (float(params["geometry"]["rhostar"])
                 * float(params["geometry"].get("minor_r") or 1.0))
        minor_r = float(params["geometry"].get("minor_r") or 1.0)

        readers = [run.mom(n) for n in run.species]
        times, index_of = self._common_indices(readers, t)
        out = {}
        for name, reader in zip(run.species, readers):
            n0, T0 = self._background_3d(name)
            spec = next(s for s in params["species"] if s["name"] == name)
            f_T = float(spec.get("temp", 1.0))
            f_n = float(spec.get("dens", 1.0))
            idx = index_of[id(reader)]
            i_n = reader.index_of("n")
            i_tpar = reader.index_of("T_par")
            i_tper = reader.index_of("T_per")

            stacks = {v: [] for v in ("T", "n", "omt", "omn")}
            for _, arrays in reader.stream_selected(idx):
                t_pert = g3.flux_surface_average(
                    arrays[i_tpar] / 3.0 + 2.0 * arrays[i_tper] / 3.0, J)
                n_pert = g3.flux_surface_average(arrays[i_n], J)
                # The perturbation is per species, so it carries the same
                # species factor the background already has baked in.
                T_tot = T0 + f_T * scale * t_pert
                n_tot = n0 + f_n * scale * n_pert
                stacks["T"].append(T_tot)
                stacks["n"].append(n_tot)
                stacks["omt"].append(_log_gradient(T_tot, x_o_a) / minor_r)
                stacks["omn"].append(_log_gradient(n_tot, x_o_a) / minor_r)
            out[name] = {v: np.asarray(stacks[v]) for v in stacks}
        return {"species": out, "times": times, "x_o_a": x_o_a}

    def _dataset_3d(self, t):
        from genetools._xr import make_dataset, unit_attrs
        raw = self.compute(t)
        params = self.params
        units = params.get("units", {}) or {}
        names = [n for n in self.run.species if n in raw["species"]]
        variables = ("T", "n", "omt", "omn")
        ds = make_dataset(
            {v: (("species", "time", "x"),
                 np.stack([raw["species"][n][v] for n in names], axis=0))
             for v in variables},
            {"x": raw["x_o_a"], "time": raw["times"]},
            species=names, params=params)
        labels = {"T": "T_ref", "n": "n_ref",
                  "omt": "L_ref/L_T", "omn": "L_ref/L_n"}
        si_ref = {"T": ("Tref", "keV"), "n": ("nref", "1e19 m^-3")}
        for v in variables:
            ds[v].attrs["units"] = labels[v]
            if v in si_ref:
                key, unit = si_ref[v]
                ref = units.get(key)
                if ref is not None:
                    ds[v + "_SI"] = ds[v] * float(ref)
                    ds[v + "_SI"].attrs["units"] = unit
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = self.geometry_kind
        return ds

    def _plot_3d(self, t, si=False):
        ds = self._dataset_3d(t)
        x = np.asarray(ds["x"])
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, v in zip(axes.ravel(), ("T", "n", "omt", "omn")):
            key = f"{v}_SI" if (si and f"{v}_SI" in ds) else v
            for name in ds["species"].values:
                ax.plot(x, np.asarray(self._t_average(ds[key].sel(species=name))),
                        label=str(name))
            ax.set_xlabel(r"$x/a$")
            ax.set_ylabel(ds[key].attrs.get("units", ""))
            ax.set_title(v)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        plt.show()
        return fig

    def compare_with_code(self, t=None, species=None):
        """
        GENE-3D only: compare this reconstruction with ``profile_<species>``.

        GENE-3D writes the same four quantities itself, so when that file is
        present the reconstruction is checkable rather than merely plausible.
        Returns ``{species: {var: max relative difference}}``.
        """
        self._require("xy_global")
        from genetools.diagnostics.profile_diag import ProfileDiag
        code = ProfileDiag(self.run).dataset()
        mine = self._dataset_3d(t)
        names = [species] if species else list(mine["species"].values)
        report = {}
        for name in names:
            per = {}
            for v in ("T", "n", "omt", "omn"):
                if v not in code:
                    continue
                a = np.asarray(self._t_average(mine[v].sel(species=name)))
                b = np.asarray(self._t_average(code[v].sel(species=name)))
                per[v] = float(np.max(np.abs(a - b))
                               / max(np.max(np.abs(b)), 1e-300))
            report[name] = per
        return report

    # ------------------------------------------------------------------

    def plot(self, t=None, si=False, eq_profs=None, **kw):
        """Plot the radial profiles over the window *t*."""
        if self.is_3d:
            return self._plot_3d(t, si=si)
        self.compute(t)
        a, b = self._bounds(t)
        if eq_profs is None:
            eq_profs = self.run.eq_profiles
        return self._plot_spectral(self.coord, self.params, a, b,
                                   equilibrium_profiles=eq_profs, **kw)

    def _plot_spectral(self, coords: dict, params: dict,
             t_start: float = None, t_stop: float = None,
             equilibrium_profiles: dict = None) -> None:
        """
        Plot profile diagnostics from the saved HDF5 file.

        Produces per-species:
        1. (t, x) heatmaps of T, n, u, omt, omn, omu
        2. Time-averaged radial profiles with background overlay

        Parameters
        ----------
        coords : dict
            Coordinate dictionary.
        params : dict
            Parameter dictionary.
        t_start, t_stop : float, optional
            Time window for averaging/plotting.
        equilibrium_profiles : dict, optional
            Required for global runs. See :meth:`build_background`.
        """
        data = self.load(t_start, t_stop)
        if not data or "time" not in data or len(data["time"]) == 0:
            print("No profile data available to plot.")
            return

        times = data["time"]
        x_local = params["general"].get("x_local", True)
        species = params["species"]
        units = params.get("units", {})
        geom_p = params.get("geometry", {})

        rhostar = geom_p.get("rhostar",
                             units.get("rho_starref", 1.0 / 500))
        minor_r = geom_p.get("minor_r", 1.0)

        backgrounds = self.build_background(params, coords,
                                            equilibrium_profiles)

        # Determine x-axis and normalization
        if x_local:
            x = np.asarray(coords["x"])
            dx_n = (x[1] - x[0]) * rhostar * minor_r if len(x) > 1 else 1.0
            x_label = r"$x / \rho_{\rm ref}$"
        else:
            x = np.asarray(coords["x"])
            dx_n = (x[1] - x[0]) * minor_r if len(x) > 1 else 1.0
            x_label = r"$x / a$"

        t_label = r"$t\;c_{\rm ref}/L_{\rm ref}$"

        for sp in species:
            name = sp["name"]
            bg = backgrounds[name]

            # Load fluctuating profiles
            T_fluct = data.get(f"{name}_T")
            n_fluct = data.get(f"{name}_n")
            u_fluct = data.get(f"{name}_u")
            if T_fluct is None:
                continue

            nt, nx_data = T_fluct.shape

            # Build total profiles: background + rhostar * fluctuation * minor_r
            total_T = bg["T_back"][np.newaxis, :] + rhostar * T_fluct * minor_r
            total_n = bg["n_back"][np.newaxis, :] + rhostar * n_fluct * minor_r
            total_u = bg["u_back"][np.newaxis, :] + rhostar * u_fluct * minor_r

            # Compute gradients for each timestep
            omt = np.zeros_like(total_T)
            omn = np.zeros_like(total_n)
            omu = np.zeros_like(total_u)
            for i_t in range(nt):
                omt[i_t, :] = self._compute_gradient(total_T[i_t, :], dx_n, x_local)
                omn[i_t, :] = self._compute_gradient(total_n[i_t, :], dx_n, x_local)
                omu[i_t, :] = self._compute_gradient(total_u[i_t, :], dx_n, x_local)

            # ── Fig 1: (t, x) heatmaps ───────────────────────────────────
            if nt > 1:
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                fig.suptitle(f"{name} — profile evolution")

                for ax, arr, label in zip(
                    axes.flat,
                    [total_T, total_n, total_u, omt, omn, omu],
                    [f"T {name}", f"n {name}", f"$u_\\parallel$ {name}",
                     f"$L_{{ref}}/L_T$ {name}", f"$L_{{ref}}/L_n$ {name}",
                     f"$L_{{ref}}/L_u$ {name}"],
                ):
                    im = ax.pcolormesh(times, x, arr.T, shading="auto")
                    fig.colorbar(im, ax=ax)
                    ax.set_xlabel(t_label)
                    ax.set_ylabel(x_label)
                    ax.set_title(label)

                plt.tight_layout()
                plt.show()

            # ── Fig 2: time-averaged profiles ─────────────────────────────
            if nt > 1:
                dt = times[-1] - times[0]
                T_avg = _trapz(total_T, x=times, axis=0) / dt
                n_avg = _trapz(total_n, x=times, axis=0) / dt
                u_avg = _trapz(total_u, x=times, axis=0) / dt
                omt_avg = _trapz(omt, x=times, axis=0) / dt
                omn_avg = _trapz(omn, x=times, axis=0) / dt
                omu_avg = _trapz(omu, x=times, axis=0) / dt
                title_str = f"average [{times[0]:.3f} - {times[-1]:.3f}]"
            else:
                T_avg = total_T[0]
                n_avg = total_n[0]
                u_avg = total_u[0]
                omt_avg = omt[0]
                omn_avg = omn[0]
                omu_avg = omu[0]
                title_str = f"t = {times[0]:.3f}"

            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle(f"{name} — {title_str}")

            # Top row: T, n, u
            for ax, avg, back, ylabel in zip(
                axes[0],
                [T_avg, n_avg, u_avg],
                [bg["T_back"], bg["n_back"], bg["u_back"]],
                [f"T {name}", f"n {name}", f"$u_\\parallel$ {name}"],
            ):
                ax.plot(x, avg, "-b", label="total")
                ax.plot(x, back, "--r", label="background")
                ax.set_xlabel(x_label)
                ax.set_ylabel(ylabel)
                ax.legend()
                ax.grid(True)

            # Bottom row: omt, omn, omu
            for ax, avg, back, ylabel in zip(
                axes[1],
                [omt_avg, omn_avg, omu_avg],
                [bg["omt_back"], bg["omn_back"],
                 np.zeros_like(bg["omt_back"])],
                [f"$L_{{ref}}/L_T$ {name}", f"$L_{{ref}}/L_n$ {name}",
                 f"$L_{{ref}}/L_u$ {name}"],
            ):
                ax.plot(x, avg, "-b", label="total")
                ax.plot(x, back, "--r", label="background")
                ax.set_xlabel(x_label)
                ax.set_ylabel(ylabel)
                ax.legend()
                ax.grid(True)

            plt.tight_layout()
            plt.show()


def _log_gradient(profile, x_o_a):
    """
    Return ``-d ln(profile) / d(x/a)``.

    Non-positive values make the logarithm meaningless; they are masked to NaN
    rather than allowed to produce a warning and an arbitrary number, since a
    profile that has gone negative means the run itself is in trouble.
    """
    arr = np.asarray(profile, dtype=float)
    safe = np.where(arr > 0, arr, np.nan)
    return -np.gradient(np.log(safe), x_o_a)
