# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
shearing.py — ExB shearing rate diagnostic for GENE simulations.

Computes the zonal (ky=0) component of the electrostatic potential and
derives from it:

    - phi_zonal    : flux-surface-averaged zonal potential (real space)
    - E_r          : radial electric field
    - v_ExB        : ExB velocity in the radial direction
    - omega_ExB    : ExB shearing rate  ω_ExB = -∂²φ_zonal/∂x² / C_xy
    - shearing_rms : rms shearing rate  √⟨ω_ExB²⟩_x  (local geometry only)

Supports both **local** (x_local=True, spectral in x) and **global**
(x_local=False, real-space radial grid) geometry, following the same
conventions as the original MATLAB implementation and Eq. 5.20 of the
Lapillonne thesis for the global case.

Results are streamed to an HDF5 file (`shearing_rate.h5`) so that repeated
calls skip already-computed time steps — identical caching strategy to
:class:`~genetools.spectra.Spectra`.

Example
-------
>>> from genetools.shearing import ShearingRate
>>> from genetools.data import BinaryReader
>>> from genetools.params import Params
>>> from genetools.geometry import Geometry
>>> from genetools.coordinates import Coordinates
>>> from genetools.utils import set_runs

>>> folder = "/path/to/run/"
>>> runs   = set_runs(folder)
>>> params = Params(folder, runs)
>>> geom   = Geometry(folder, runs, params)
>>> coord  = Coordinates(folder, runs, params)

>>> sr = ShearingRate(outfile="shearing_rate.h5")
>>> field_readers = [BinaryReader("field", folder, ext, params.get(fn))
...                  for fn, ext in enumerate(runs)]

>>> sr.compute_and_save(field_readers, coord, geom, params, t_start=10.5, t_stop=2850.)
>>> sr.plot()
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
# Finite-difference derivative helper (global geometry)
# ---------------------------------------------------------------------------

def _central_diff(f: np.ndarray) -> np.ndarray:
    """
    Second-order central finite difference derivative along axis 0.

    Uses one-sided differences at the boundaries.

    Parameters
    ----------
    f : np.ndarray
        1-D array of function values on a uniform grid.

    Returns
    -------
    np.ndarray
        Derivative array, same shape as *f*.
    """
    d = np.empty_like(f)
    n = len(f)
    if n < 2:
        d[:] = 0.0
        return d
    if n == 2:
        d[0] = f[1] - f[0]
        d[1] = f[1] - f[0]
        return d
    d[1:-1] = (f[2:] - f[:-2]) * 0.5
    d[0]    = (-3*f[0] + 4*f[1] - f[2])   * 0.5   # forward
    d[-1]   = ( 3*f[-1] - 4*f[-2] + f[-3]) * 0.5  # backward
    return d


# ---------------------------------------------------------------------------
# Core physics — compute all ExB quantities from a single phi snapshot
# ---------------------------------------------------------------------------

def compute_exb(phi: np.ndarray, params: dict, geom: dict, coord: dict) -> dict:
    """
    Compute ExB shearing quantities from a single phi snapshot.

    Parameters
    ----------
    phi : np.ndarray
        Complex field array of shape ``(nx, nky, nz)``.
    params : dict
        Parameter dictionary for this run segment.
    geom : dict
        Geometry dictionary for this run segment.
    coord : dict
        Coordinate dictionary for this run segment.

    Returns
    -------
    dict with keys:
        ``phi_zonal_fsavg`` : flux-surface-averaged zonal phi in kx space
                              (local geometry) or None (global geometry)
        ``phi_zonal_x``     : zonal phi in real space, shape ``(nx,)``
        ``E_r``             : radial electric field, shape ``(nx,)``
        ``v_ExB``           : ExB velocity, shape ``(nx,)``
        ``omega_ExB``       : ExB shearing rate, shape ``(nx,)``
        ``shearing_rms``    : rms shearing rate (scalar), local only
    """
    x_local = params["general"].get("x_local", True)
    nx  = params["box"]["nx0"]
    nz  = params["box"]["nz0"]
    J   = geom["Jacobian"]                          # shape (nz,)
    J_norm = J / J.sum()

    # ── ky=0 component ────────────────────────────────────────────────────
    phi_zonal_kx = phi[:, 0, :]                     # shape (nx, nz)

    if x_local:
        # ------------------------------------------------------------------
        # LOCAL geometry — x direction is spectral (kx space)
        # ------------------------------------------------------------------
        kx = np.asarray(coord["kx"])                # shape (nx,)

        # Flux-surface average: weighted sum over z using Jacobian
        # Result: phi_zonal_fsavg[ikx] = Σ_z J(z)*Re(phi[ikx,0,z]) / Σ_z J(z)
        phi_zonal_fsavg = np.einsum("iz,z->i", phi_zonal_kx.real, J_norm)
        # Note: phi[:,0,:] is already ky=0 so real part is physically meaningful

        # Real-space zonal potential via inverse FFT (GENE normalisation: multiply by nx)
        phi_zonal_x = nx * np.real(np.fft.ifft(phi_zonal_fsavg))

        # Radial electric field: E_r = -∂phi/∂x → multiply by -i*kx in Fourier
        E_r = nx * np.real(np.fft.ifft(-1j * kx * phi_zonal_fsavg))

        # ExB velocity: v_ExB = -E_r / C_xy
        C_xy = geom["metric"]["C_xy"]
        v_ExB = -E_r / C_xy

        # ExB shearing rate: ω_ExB = -∂²phi/∂x² / C_xy
        #                           = -IFFT(kx² * phi_zonal_fsavg) * nx / C_xy
        omega_ExB = -nx * np.real(np.fft.ifft(kx**2 * phi_zonal_fsavg)) / C_xy

        # RMS shearing rate (scalar diagnostic)
        shearing_rms = float(np.sqrt(np.mean(omega_ExB**2)))

        return dict(
            phi_zonal_fsavg = phi_zonal_fsavg,
            phi_zonal_x     = phi_zonal_x,
            E_r             = E_r,
            v_ExB           = v_ExB,
            omega_ExB       = omega_ExB,
            shearing_rms    = shearing_rms,
        )

    else:
        # ------------------------------------------------------------------
        # GLOBAL geometry — x direction is real-space
        # Following Eq. 5.20 of Lapillonne thesis
        # ------------------------------------------------------------------
        dx = coord["dx"]
        x  = np.asarray(coord["x"])                 # radial grid (rho_ref units)

        # Flux-surface average of zonal phi: weighted sum over z
        # phi_zonal_kx has shape (nx, nz); J has shape (nz,)
        phi_zonal_x = (phi_zonal_kx.real * J_norm).sum(axis=1)

        # Radial electric field: E_r = -∂phi_zonal/∂x
        E_r = -_central_diff(phi_zonal_x) / dx

        # ExB velocity (global: no C_xy factor, already in correct units)
        v_ExB = E_r.copy()

        # ExB shearing rate (global) needs the safety-factor profile:
        #   ω_ExB = (x/q) * ∂/∂x (q * E_r / x) / dx
        # The q-profile is absent if the geometry file carries no q array, so
        # degrade gracefully to NaN rather than crashing — the zonal potential
        # and E_r (used by the Zonal diagnostic) remain valid.
        q = (geom.get("profiles") or {}).get("q")
        if q is None:
            warnings.warn(
                "Global shearing rate: geometry has no q-profile; "
                "omega_ExB and shearing_rms set to NaN.")
            omega_ExB = np.full_like(E_r, np.nan)
            shearing_rms = float("nan")
        else:
            q = np.asarray(q)
            omega_ExB = (x / q) * _central_diff(q * E_r / x) / dx
            shearing_rms = float(np.sqrt(np.mean(omega_ExB**2)))

        return dict(
            phi_zonal_fsavg = None,         # not defined for global geometry
            phi_zonal_x     = phi_zonal_x,
            E_r             = E_r,
            v_ExB           = v_ExB,
            omega_ExB       = omega_ExB,
            shearing_rms    = shearing_rms,
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ShearingRate(RunDiagnostic):
    """
    Compute, cache, and plot ExB shearing rate diagnostics.

    Results are appended to an HDF5 file so that re-running only processes
    new time steps.

    Parameters
    ----------
    outfile : str, optional
        Path to the output HDF5 file (default ``'shearing_rate.h5'``).
    """

    name = "shearing"
    cache_file = "shearing_rate.h5"

    def __init__(self, run=None, outfile: str = None, folder: str = None):
        """
        Parameters
        ----------
        run : genetools.run.Run, optional
        outfile, folder : str, optional
            Override the HDF5 cache location; normally derived from the run.
        """
        if run is not None:
            super().__init__(run)
            if outfile:
                self.outfile = outfile
            return
        # Detached: keep the pre-Run constructor working for direct use.
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
    def _init_h5(f, result: dict, time: float, nx: int, x_local: bool,
                 time_dtype=np.float64) -> None:
        """Create all datasets in a newly opened HDF5 file handle."""
        f.create_dataset("time",        data=np.array([time], dtype=time_dtype),
                         maxshape=(None,), chunks=(1,))
        f.create_dataset("phi_zonal_x", data=result["phi_zonal_x"][np.newaxis, :],
                         maxshape=(None, nx), chunks=(1, nx))
        f.create_dataset("E_r",         data=result["E_r"][np.newaxis, :],
                         maxshape=(None, nx), chunks=(1, nx))
        f.create_dataset("v_ExB",       data=result["v_ExB"][np.newaxis, :],
                         maxshape=(None, nx), chunks=(1, nx))
        f.create_dataset("omega_ExB",   data=result["omega_ExB"][np.newaxis, :],
                         maxshape=(None, nx), chunks=(1, nx))
        f.create_dataset("shearing_rms", data=np.array([result["shearing_rms"]]),
                         maxshape=(None,), chunks=(1,))
        if x_local and result["phi_zonal_fsavg"] is not None:
            nkx = len(result["phi_zonal_fsavg"])
            f.create_dataset("abs_phi_zonal_kx",
                             data=np.abs(result["phi_zonal_fsavg"])[np.newaxis, :],
                             maxshape=(None, nkx), chunks=(1, nkx))

    @staticmethod
    def _append_to_open_file(f, result: dict, time: float, x_local: bool) -> None:
        """Append one time step to an already-open HDF5 file handle."""
        n = f["time"].shape[0]

        def _append_ds(name, value):
            ds = f[name]
            if ds.ndim == 1:
                ds.resize((n + 1,))
                ds[n] = value
            else:
                ds.resize((n + 1, ds.shape[1]))
                ds[n, :] = value

        _append_ds("time",         time)
        _append_ds("phi_zonal_x",  result["phi_zonal_x"])
        _append_ds("E_r",          result["E_r"])
        _append_ds("v_ExB",        result["v_ExB"])
        _append_ds("omega_ExB",    result["omega_ExB"])
        _append_ds("shearing_rms", result["shearing_rms"])
        if x_local and "abs_phi_zonal_kx" in f:
            _append_ds("abs_phi_zonal_kx",
                       np.abs(result["phi_zonal_fsavg"]))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_and_save(
        self,
        field_readers: list,
        coords: list,
        geoms: list,
        params,
        t_start: float,
        t_stop: float,
    ) -> None:
        """
        Stream field files, compute shearing rates, and append to HDF5.

        Skips time steps already present in the output file.

        Parameters
        ----------
        field_readers : list of BinaryReader / BPReader
            One reader per run segment (same order as *runs*).
        coords : list of dict
            Coordinate dictionaries, one per segment.
        geoms : list of dict
            Geometry dictionaries, one per segment.
        params : Params
            Parameter object for the full run.
        t_start, t_stop : float
            Time window to process.
        """
        saved_times = list(self._load_saved_times())

        with h5py.File(self.outfile, "a") as hf:
            initialised = "time" in hf

            # Process later segments first so that on restart overlaps the later
            # segment's values win (matching MultiSegmentReader semantics), and
            # track times written during this call so overlapping timesteps are
            # not written twice (stale-cache bug).
            for seg_idx in range(len(field_readers) - 1, -1, -1):
                reader = field_readers[seg_idx]
                p     = params.get(seg_idx)
                coord = coords[seg_idx]
                geom  = geoms[seg_idx]
                nx    = p["box"]["nx0"]
                x_local = p["general"].get("x_local", True)

                times = reader.read_all_times()
                mask  = (times >= t_start) & (times <= t_stop)
                indices = np.where(mask)[0].tolist()

                for t, arrays in reader.stream_selected(indices):
                    if self._is_already_saved(t, np.asarray(saved_times, dtype=float)):
                        continue

                    phi    = arrays[0]
                    result = compute_exb(phi, p, geom, coord)

                    if not initialised:
                        self._init_h5(hf, result, t, nx, x_local,
                                      time_dtype=self._time_dtype(p))
                        initialised = True
                    else:
                        self._append_to_open_file(hf, result, t, x_local)
                    saved_times.append(float(t))

    def load(self) -> dict:
        """
        Load all saved results from the HDF5 file.

        Returns
        -------
        dict
            Keys: ``'time'``, ``'phi_zonal_x'``, ``'E_r'``, ``'v_ExB'``,
            ``'omega_ExB'``, ``'shearing_rms'``, and optionally
            ``'abs_phi_zonal_kx'``.  Arrays are sorted by time.
        """
        with h5py.File(self.outfile, "r") as f:
            data = {k: f[k][...] for k in f.keys()}

        idx = np.argsort(data["time"])
        for k, v in data.items():
            data[k] = v[idx] if v.ndim >= 1 else v

        return data

    def dataset(self, t=None, params=None, species=None):
        """
        Return the cached zonal quantities as an :class:`xarray.Dataset`.

        Called with a time window when bound to a run; the older
        ``dataset(coords, params)`` form is still accepted for direct use.
        """
        if self.run is not None and not isinstance(t, dict):
            if self.is_3d:
                return self._dataset_3d(t)
            self.compute(t)
            ds = self._dataset_from_cache(self.coord, self.params)
            a, b = self._window(t)
            if (a is not None or b is not None) and "time" in ds.coords:
                ds = ds.sel(time=slice(a, b))
            return ds
        return self._dataset_from_cache(t, params, species)

    def _dataset_from_cache(self, coords, params, species=None):
        """Return the shearing-rate diagnostics as an ``xarray.Dataset``."""
        import xarray as xr
        from genetools import _xr

        raw = self.load()
        if not raw or np.asarray(raw.get("time", [])).size == 0:
            return xr.Dataset()
        time = np.asarray(raw["time"])

        def dim_of(var):
            if var == "shearing_rms":
                return ("time",)
            if var == "abs_phi_zonal_kx":
                return ("time", "kx")
            return ("time", "x")

        data_vars, _ = _xr.stacked_vars(raw, [], dim_of, coord_keys=("time",))
        x = np.asarray(coords.get("x", []))
        if x.size == 0:
            x = np.asarray(coords.get("x_o_a", []))
        candidates = {"time": time, "x": x,
                      "kx": np.asarray(coords.get("kx", []))}
        return _xr.make_dataset(data_vars, candidates, params=params)

    # ------------------------------------------------------------------
    # Run-native front end
    # ------------------------------------------------------------------

    def compute(self, t=None):
        """
        Stream the field file and cache the zonal quantities.

        GENE-3D is real space in y, so its zonal potential is a
        Jacobian-weighted average over y and z rather than the ``ky = 0``
        component of a transform, and it is held in memory rather than streamed
        to the HDF5 cache — the result is one radial profile per output time,
        which is small.
        """
        if self.is_3d:
            key = self._key(t)
            if key not in self._cache:
                self._cache[key] = self._compute_3d(t)
            return self._cache[key]
        a, b = self._bounds(t)
        r = self.run
        self.compute_and_save(r._field_segment_readers, r.coords, r.geometry,
                              r.params, a, b)
        return self

    def _compute_3d(self, t):
        """Zonal potential, ExB velocity and shearing rate for GENE-3D."""
        run = self.run
        J = self.geom["Jacobian"]
        C_xy = np.asarray(self.geom["metric"]["C_xy"], dtype=float)
        x = np.asarray(self.coord["x"], dtype=float)      # in rho_ref

        reader = run.field
        _, idx = self._indices(reader, t)
        i_phi = reader.index_of("phi")

        phi_fs, v_exb, w_exb, times = [], [], [], []
        for time, arrays in reader.stream_selected(idx):
            times.append(time)
            fs = g3.flux_surface_average(arrays[i_phi], J)
            # C_xy only: the 1/sqrt(g^xx) of GENE-3D's flux_geomfac belongs to a
            # flux per unit physical area. This is a flow, not a flux.
            v = -np.gradient(fs, x) / C_xy
            phi_fs.append(fs)
            v_exb.append(v)
            w_exb.append(np.gradient(v, x))

        return {
            "times": np.asarray(times), "x": x,
            "x_o_a": np.asarray(self.coord["x_o_a"], dtype=float),
            "phi_zonal": np.asarray(phi_fs),
            "v_exb": np.asarray(v_exb),
            "omega_exb": np.asarray(w_exb),
        }

    def _dataset_3d(self, t):
        from genetools._xr import make_dataset, unit_attrs
        raw = self.compute(t)
        params = self.params
        ds = make_dataset(
            {"phi_zonal": (("time", "x"), raw["phi_zonal"]),
             "v_exb": (("time", "x"), raw["v_exb"]),
             "omega_exb": (("time", "x"), raw["omega_exb"])},
            {"x": raw["x_o_a"], "time": raw["times"]}, params=params)
        ds = ds.assign(x_o_rho_ref=("x", raw["x"]))
        ds["phi_zonal"].attrs["units"] = "T_ref/e (normalised)"
        ds["v_exb"].attrs["units"] = "c_ref (normalised)"
        ds["omega_exb"].attrs["units"] = "c_ref/L_ref"
        ds["omega_exb_rms_x"] = np.sqrt((ds["omega_exb"] ** 2).mean("time"))
        ds["omega_exb_rms_t"] = np.sqrt((ds["omega_exb"] ** 2).mean("x"))
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = self.geometry_kind
        return ds

    def _plot_3d(self, t):
        """Three x-t maps plus the RMS shearing-rate summaries."""
        ds = self._dataset_3d(t)
        x = np.asarray(ds["x"])
        times = np.asarray(ds["time"])

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, name, title in zip(
                axes, ("phi_zonal", "v_exb", "omega_exb"),
                (r"$\langle\phi\rangle_{FS}$", r"$v_{E\times B}$",
                 r"$\omega_{E\times B}$")):
            mesh = ax.pcolormesh(times, x, np.asarray(ds[name]).T,
                                 shading="nearest")
            ax.set_title(title)
            ax.set_xlabel(r"$t\;[L_{\rm ref}/c_{\rm ref}]$")
            ax.set_ylabel(r"$x/a$")
            fig.colorbar(mesh, ax=ax)
        fig.tight_layout()

        fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
        axes2[0].plot(times, np.asarray(ds["omega_exb_rms_t"]))
        axes2[0].axhline(float(ds["omega_exb_rms_t"].mean()), ls="--", color="k")
        axes2[0].set_xlabel(r"$t\;[L_{\rm ref}/c_{\rm ref}]$")
        axes2[0].set_ylabel(r"$\langle|\omega_E|^2\rangle_x^{1/2}$")
        axes2[1].plot(x, np.asarray(ds["omega_exb_rms_x"]))
        axes2[1].axhline(float(ds["omega_exb_rms_x"].mean()), ls="--", color="k")
        axes2[1].set_xlabel(r"$x/a$")
        axes2[1].set_ylabel(r"$\langle|\omega_E|^2\rangle_t^{1/2}$")
        for ax in axes2:
            ax.grid(True, alpha=0.3)
        fig2.tight_layout()
        plt.show()
        return fig

    def plot(self, t=None, **kw):
        """Plot the zonal quantities over the window *t*."""
        if self.is_3d:
            return self._plot_3d(t)
        self.compute(t)
        a, b = self._bounds(t)
        return self._plot_spectral(self.coord, a, b)

    # ------------------------------------------------------------------

    def _plot_spectral(self, coord=None, t_start=None, t_stop=None) -> None:
        """
        Plot E_r and ω_ExB diagnostics from the saved HDF5 file.
 
        Produces six figures:
 
        1. **RMS shearing rate** time trace
        2. **E_r(x, t)** 2-D colour map
        3. **⟨E_r⟩_t** time-averaged radial electric field profile
        4. **ω_ExB(x, t)** 2-D colour map
        5. **⟨ω_ExB⟩_t** time-averaged shearing rate profile
        6. **Radial profiles** of ω_ExB at first, middle, last saved time
        7. **|φ_zonal(kx)|** spectrum (local geometry only)
 
        Parameters
        ----------
        coord : dict, optional
            Coordinate dictionary. If provided, uses physical x-axis;
            otherwise uses grid index.
        t_start, t_stop : float, optional
            Restrict the time average to this window. If omitted, all
            saved times are used.
        """
        data = self.load()
        times     = data["time"]
        E_r       = data["E_r"]           # shape (n_times, nx)
        omega     = data["omega_ExB"]     # shape (n_times, nx)
        shear_rms = data["shearing_rms"]  # shape (n_times,)
 
        x_axis  = (np.arange(omega.shape[1])
                   if coord is None else np.asarray(coord["x"]))
        x_label = "x index" if coord is None else r"$x / \rho_{\rm ref}$"
        t_label = r"$t\;c_{\rm ref}/L_{\rm ref}$"
 
        # ── Time window mask for averages ──────────────────────────────────
        mask = np.ones(len(times), dtype=bool)
        if t_start is not None:
            mask &= times >= t_start
        if t_stop is not None:
            mask &= times <= t_stop
        times_avg = times[mask]
        E_r_avg   = E_r[mask]
        omega_avg = omega[mask]
 
        # ── Fig 1: RMS shearing rate time trace ───────────────────────────
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(times, shear_rms, color="steelblue")
        if t_start is not None:
            ax.axvline(t_start, color="k", ls="--", lw=0.8, label="avg window")
        if t_stop is not None:
            ax.axvline(t_stop,  color="k", ls="--", lw=0.8)
        ax.set_xlabel(t_label)
        ax.set_ylabel(r"$\omega_{E\times B}^{\rm rms}$")
        ax.set_title("RMS ExB shearing rate")
        ax.grid(True)
        plt.tight_layout()
        plt.show()
 
        # ── Fig 2: E_r(x, t) colour map ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 4))
        vmax = np.percentile(np.abs(E_r), 98)
        im = ax.pcolormesh(times, x_axis, E_r.T,
                           cmap="bwr", vmin=-vmax, vmax=vmax,
                           shading="auto")
        fig.colorbar(im, ax=ax, label=r"$E_r$")
        ax.set_xlabel(t_label)
        ax.set_ylabel(x_label)
        ax.set_title(r"$E_r(x,\,t)$")
        plt.tight_layout()
        plt.show()
 
        # ── Fig 3: time-averaged E_r profile ──────────────────────────────
        E_r_mean = self._time_average(E_r_avg, times_avg)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_axis, E_r_mean, color="crimson")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$\langle E_r \rangle_t$")
        ax.set_title("Time-averaged radial electric field")
        ax.grid(True)
        plt.tight_layout()
        plt.show()
 
        # ── Fig 4: ω_ExB(x, t) colour map ────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 4))
        vmax = np.percentile(np.abs(omega), 98)
        im = ax.pcolormesh(times, x_axis, omega.T,
                           cmap="bwr", vmin=-vmax, vmax=vmax,
                           shading="auto")
        fig.colorbar(im, ax=ax,
                     label=r"$\omega_{E\times B}$")
        ax.set_xlabel(t_label)
        ax.set_ylabel(x_label)
        ax.set_title(r"$\omega_{E\times B}(x,\,t)$")
        plt.tight_layout()
        plt.show()
 
        # ── Fig 5: time-averaged ω_ExB profile ────────────────────────────
        omega_mean = self._time_average(omega_avg, times_avg)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_axis, omega_mean, color="steelblue")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$\langle \omega_{E\times B} \rangle_t$")
        ax.set_title("Time-averaged ExB shearing rate")
        ax.grid(True)
        plt.tight_layout()
        plt.show()
 
        # ── Fig 6: ω_ExB radial profiles at selected times ────────────────
        t_indices = [0, len(times) // 2, -1]
        fig, ax = plt.subplots(figsize=(7, 4))
        for ti in t_indices:
            ax.plot(x_axis, omega[ti, :], label=f"t={times[ti]:.1f}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$\omega_{E\times B}$")
        ax.set_title("ExB shearing rate — radial profiles")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()
 
        # ── Fig 7: |φ_zonal(kx)| spectrum (local geometry only) ───────────
        if "abs_phi_zonal_kx" in data:
            kx_spec = data["abs_phi_zonal_kx"]   # shape (n_times, nkx)
            fig, ax = plt.subplots(figsize=(9, 4))
            vmax = np.percentile(kx_spec, 98)
            im = ax.pcolormesh(times, np.arange(kx_spec.shape[1]), kx_spec.T,
                               cmap="inferno", vmin=0, vmax=vmax,
                               shading="auto")
            fig.colorbar(im, ax=ax,
                         label=r"$|\hat{\phi}_{\rm zonal}(k_x)|$")
            ax.set_xlabel(t_label)
            ax.set_ylabel(r"$k_x$ index")
            ax.set_title("Zonal potential kx spectrum")
            plt.tight_layout()
            plt.show()
