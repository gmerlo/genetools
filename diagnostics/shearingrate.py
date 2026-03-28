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
import numpy as np
import matplotlib.pyplot as plt
import h5py

from genetools.compat import trapz as _trapz


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
        q  = np.asarray(geom["profiles"]["q"])      # safety factor profile

        # Flux-surface average of zonal phi: weighted sum over z
        # phi_zonal_kx has shape (nx, nz); J has shape (nz,)
        phi_zonal_x = (phi_zonal_kx.real * J_norm).sum(axis=1)

        # Radial electric field: E_r = -∂phi_zonal/∂x
        E_r = -_central_diff(phi_zonal_x) / dx

        # ExB velocity (global: no C_xy factor, already in correct units)
        v_ExB = E_r.copy()

        # ExB shearing rate (global, accounts for q profile):
        # ω_ExB = (x/q) * ∂/∂x (q * E_r / x) / dx
        omega_ExB = (x / q) * _central_diff(q * E_r / x) / dx
    
        return dict(
            phi_zonal_fsavg = None,         # not defined for global geometry
            phi_zonal_x     = phi_zonal_x,
            E_r             = E_r,
            v_ExB           = v_ExB,
            omega_ExB       = omega_ExB,
            shearing_rms    = float(np.sqrt(np.mean(omega_ExB**2))),
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ShearingRate:
    """
    Compute, cache, and plot ExB shearing rate diagnostics.

    Results are appended to an HDF5 file so that re-running only processes
    new time steps.

    Parameters
    ----------
    outfile : str, optional
        Path to the output HDF5 file (default ``'shearing_rate.h5'``).
    """

    def __init__(self, outfile: str = "shearing_rate.h5"):
        self.outfile = outfile

    # ------------------------------------------------------------------
    # HDF5 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_saved_times(outfile: str) -> np.ndarray:
        """Load all saved times from the HDF5 file (empty array if none)."""
        if not os.path.exists(outfile):
            return np.array([], dtype=np.float32)
        with h5py.File(outfile, "r") as f:
            if "time" not in f:
                return np.array([], dtype=np.float32)
            return f["time"][...].astype(np.float32)

    @staticmethod
    def _is_already_saved(time: float, saved_times: np.ndarray) -> bool:
        """Check if *time* is in *saved_times* (pre-loaded array)."""
        if saved_times.size == 0:
            return False
        t32 = np.float32(time)
        tol = max(1e-6, abs(t32) * 1e-6)
        return bool(np.any(np.abs(saved_times - t32) <= tol))

    @staticmethod
    def _init_h5(f, result: dict, time: float, nx: int, x_local: bool) -> None:
        """Create all datasets in a newly opened HDF5 file handle."""
        f.create_dataset("time",        data=np.array([time]),
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
        saved_times = self._load_saved_times(self.outfile)

        with h5py.File(self.outfile, "a") as hf:
            initialised = "time" in hf

            for seg_idx, reader in enumerate(field_readers):
                p     = params.get(seg_idx)
                coord = coords[seg_idx]
                geom  = geoms[seg_idx]
                nx    = p["box"]["nx0"]
                x_local = p["general"].get("x_local", True)

                times = reader.read_all_times()
                mask  = (times >= t_start) & (times <= t_stop)
                indices = np.where(mask)[0].tolist()

                for t, arrays in reader.stream_selected(indices):
                    if self._is_already_saved(t, saved_times):
                        continue

                    phi    = arrays[0]
                    result = compute_exb(phi, p, geom, coord)

                    if not initialised:
                        self._init_h5(hf, result, t, nx, x_local)
                        initialised = True
                    else:
                        self._append_to_open_file(hf, result, t, x_local)

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

    @staticmethod
    def _time_average(arr: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Trapezoidal time average of *arr* (shape n_times x nx) over *times*.
 
        Returns a 1-D array of shape ``(nx,)``.
        """
        dt = times[-1] - times[0]
        if dt == 0 or len(times) == 1:
            return arr[0]
        return _trapz(arr, x=times, axis=0) / dt

    def plot(self, coord=None, t_start=None, t_stop=None) -> None:
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
