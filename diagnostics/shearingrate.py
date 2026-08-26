# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
shearing.py — ExB shearing rate diagnostic for GENE simulations.

Computes the zonal (ky=0) component of the electrostatic potential and
derives from it:

    - phi_zonal    : flux-surface-averaged zonal potential (real space)
    - e_r          : radial electric field
    - v_exb        : ExB velocity in the radial direction
    - omega_exb    : ExB shearing rate  ω_ExB = -∂²φ_zonal/∂x² / C_xy
    - shearing_rms : rms shearing rate  √⟨ω_ExB²⟩_x

Every geometry returns the same variable names, so ``run.shearing.data`` means
the same thing whatever the run is. The GENE-3D path additionally gives
``omega_exb_rms_x`` / ``omega_exb_rms_t``, and the spectral local path the zonal
kx spectrum ``phi_zonal_kx_abs``.

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

import warnings
import numpy as np
import matplotlib.pyplot as plt
import h5py

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
        ``phi_zonal``       : zonal phi in real space, shape ``(nx,)``
        ``e_r``             : radial electric field, shape ``(nx,)``
        ``v_exb``           : ExB velocity, shape ``(nx,)``
        ``omega_exb``       : ExB shearing rate, shape ``(nx,)``
        ``shearing_rms``    : rms shearing rate (scalar); NaN for global
                              geometry when the geometry file carries no
                              q-profile
    """
    x_local = params["general"].get("x_local", True)
    nx  = params["box"]["nx0"]
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
        phi_zonal = nx * np.real(np.fft.ifft(phi_zonal_fsavg))

        # Radial electric field: E_r = -∂phi/∂x → multiply by -i*kx in Fourier
        e_r = nx * np.real(np.fft.ifft(-1j * kx * phi_zonal_fsavg))

        # ExB velocity: v_ExB = -E_r / C_xy
        C_xy = geom["metric"]["C_xy"]
        v_exb = -e_r / C_xy

        # ExB shearing rate: ω_ExB = -∂²phi/∂x² / C_xy
        #                           = -IFFT(kx² * phi_zonal_fsavg) * nx / C_xy
        omega_exb = -nx * np.real(np.fft.ifft(kx**2 * phi_zonal_fsavg)) / C_xy

        # RMS shearing rate (scalar diagnostic)
        shearing_rms = float(np.sqrt(np.mean(omega_exb**2)))

        return dict(
            phi_zonal_fsavg = phi_zonal_fsavg,
            phi_zonal       = phi_zonal,
            e_r             = e_r,
            v_exb           = v_exb,
            omega_exb       = omega_exb,
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
        phi_zonal = (phi_zonal_kx.real * J_norm).sum(axis=1)

        # Radial electric field: E_r = -∂phi_zonal/∂x
        e_r = -_central_diff(phi_zonal) / dx

        # ExB velocity (global: no C_xy factor, already in correct units)
        v_exb = e_r.copy()

        # ExB shearing rate (global) needs the safety-factor profile:
        #   ω_ExB = (x/q) * ∂/∂x (q * E_r / x) / dx
        # The q-profile is absent if the geometry file carries no q array, so
        # degrade gracefully to NaN rather than crashing — the zonal potential
        # and E_r remain valid.
        q = (geom.get("profiles") or {}).get("q")
        if q is None:
            warnings.warn(
                "Global shearing rate: geometry has no q-profile; "
                "omega_exb and shearing_rms set to NaN.")
            omega_exb = np.full_like(e_r, np.nan)
            shearing_rms = float("nan")
        else:
            q = np.asarray(q)
            omega_exb = (x / q) * _central_diff(q * e_r / x) / dx
            shearing_rms = float(np.sqrt(np.mean(omega_exb**2)))

        return dict(
            phi_zonal_fsavg = None,         # not defined for global geometry
            phi_zonal       = phi_zonal,
            e_r             = e_r,
            v_exb           = v_exb,
            omega_exb       = omega_exb,
            shearing_rms    = shearing_rms,
        )


# ---------------------------------------------------------------------------
# Persisted variable names
# ---------------------------------------------------------------------------

#: Radial fields written once per output time, shape ``(nx,)`` each.
_RADIAL_VARS = ("phi_zonal", "e_r", "v_exb", "omega_exb")
#: Scalars written once per output time.
_SCALAR_VARS = ("shearing_rms",)
#: The zonal kx spectrum. Local geometry only — x is real space otherwise.
_KX_VAR = "phi_zonal_kx_abs"

#: Radial quantities that get an x-t map and a time-averaged profile, in order.
_PANEL_VARS = (
    ("phi_zonal", r"$\langle\phi\rangle_{\rm zonal}$"),
    ("e_r", r"$E_r$"),
    ("omega_exb", r"$\omega_{E\times B}$"),
)

#: Figure groups :meth:`ShearingRate.plot` can draw.
_PLOT_GROUPS = ("maps", "profiles", "summary", "zonal")
#: Drawn by ``which='all'``. ``'zonal'`` is a focused view, not part of the set.
_DEFAULT_GROUPS = ("maps", "profiles", "summary")

_T_LABEL = r"$t\;c_{\rm ref}/L_{\rm ref}$"

#: Names used before the two geometry paths were made to agree on them.
#: :meth:`ShearingRate.load` accepts either spelling, so a ``shearing_rate.h5``
#: written by an older version stays readable instead of reading as empty.
_LEGACY_NAMES = {
    "phi_zonal_x": "phi_zonal",
    "E_r": "e_r",
    "v_ExB": "v_exb",
    "omega_ExB": "omega_exb",
    "abs_phi_zonal_kx": _KX_VAR,
}


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
    def _migrate_legacy_names(f) -> None:
        """
        Rename any legacy datasets in an open, writable cache file.

        :meth:`load` translates the old spellings on read, but appending needs
        the datasets themselves to carry the current names — otherwise adding a
        time step to a cache written by an older version raises ``KeyError``.
        The move is a metadata operation; the cached data is preserved.
        """
        for old, new in _LEGACY_NAMES.items():
            if old in f and new not in f:
                f.move(old, new)

    @staticmethod
    def _init_h5(f, result: dict, time: float, nx: int, x_local: bool,
                 time_dtype=np.float64) -> None:
        """Create all datasets in a newly opened HDF5 file handle."""
        f.create_dataset("time", data=np.array([time], dtype=time_dtype),
                         maxshape=(None,), chunks=(1,))
        for name in _RADIAL_VARS:
            f.create_dataset(name, data=result[name][np.newaxis, :],
                             maxshape=(None, nx), chunks=(1, nx))
        for name in _SCALAR_VARS:
            f.create_dataset(name, data=np.array([result[name]]),
                             maxshape=(None,), chunks=(1,))
        if x_local and result["phi_zonal_fsavg"] is not None:
            nkx = len(result["phi_zonal_fsavg"])
            f.create_dataset(_KX_VAR,
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

        _append_ds("time", time)
        for name in _RADIAL_VARS + _SCALAR_VARS:
            _append_ds(name, result[name])
        if x_local and _KX_VAR in f:
            _append_ds(_KX_VAR, np.abs(result["phi_zonal_fsavg"]))

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
            self._migrate_legacy_names(hf)
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
            Keys: ``'time'``, ``'phi_zonal'``, ``'e_r'``, ``'v_exb'``,
            ``'omega_exb'``, ``'shearing_rms'``, and optionally
            ``'phi_zonal_kx_abs'``.  Arrays are sorted by time.

            A cache written before the geometry paths were made to agree on
            these names is translated on the way in, so an existing
            ``shearing_rate.h5`` keeps working.
        """
        with h5py.File(self.outfile, "r") as f:
            data = {_LEGACY_NAMES.get(k, k): f[k][...] for k in f.keys()}

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
            if var == _KX_VAR:
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

        phi_fs, e_r, v_exb, w_exb, times = [], [], [], [], []
        for time, arrays in reader.stream_selected(idx):
            times.append(time)
            fs = g3.flux_surface_average(arrays[i_phi], J)
            E = -np.gradient(fs, x)
            # C_xy only: the 1/sqrt(g^xx) of GENE-3D's flux_geomfac belongs to a
            # flux per unit physical area. This is a flow, not a flux.
            v = E / C_xy
            phi_fs.append(fs)
            e_r.append(E)
            v_exb.append(v)
            w_exb.append(np.gradient(v, x))

        return {
            "times": np.asarray(times), "x": x,
            "x_o_a": np.asarray(self.coord["x_o_a"], dtype=float),
            "phi_zonal": np.asarray(phi_fs),
            "e_r": np.asarray(e_r),
            "v_exb": np.asarray(v_exb),
            "omega_exb": np.asarray(w_exb),
        }

    def _dataset_3d(self, t):
        from genetools._xr import make_dataset, unit_attrs
        raw = self.compute(t)
        params = self.params
        ds = make_dataset(
            {name: (("time", "x"), raw[name]) for name in _RADIAL_VARS},
            {"x": raw["x_o_a"], "time": raw["times"]}, params=params)
        ds = ds.assign(x_o_rho_ref=("x", raw["x"]))
        ds["phi_zonal"].attrs["units"] = "T_ref/e (normalised)"
        ds["e_r"].attrs["units"] = "T_ref/(e rho_ref) (normalised)"
        ds["v_exb"].attrs["units"] = "c_ref (normalised)"
        ds["omega_exb"].attrs["units"] = "c_ref/L_ref"
        ds["omega_exb_rms_x"] = np.sqrt(self._t_average(ds["omega_exb"] ** 2))
        ds["omega_exb_rms_t"] = np.sqrt((ds["omega_exb"] ** 2).mean("x"))
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = self.geometry_kind
        return ds


    # ------------------------------------------------------------------
    # Plot
    #
    # One implementation for every geometry: the rename above means both paths
    # hand back the same variables on the same `(time, x)` dims, so there is no
    # need for a spectral and a 3-D plotter that drift apart.
    # ------------------------------------------------------------------

    def plot(self, t=None, which="all", **kw):
        """
        Plot the zonal quantities over the window *t*.

        Parameters
        ----------
        t : (float, float), optional
            Time window.
        which : str or sequence of str
            Which figures to draw. ``'all'`` (default) is
            ``('maps', 'profiles', 'summary')``:

            - ``'maps'``     — x-t colour maps of φ_zonal, E_r and ω_ExB
            - ``'profiles'`` — their time-averaged radial profiles
            - ``'summary'``  — RMS shearing rate, ω_ExB at three times, and the
              zonal kx spectrum where it exists
            - ``'zonal'``    — the φ_zonal x-t map beside its time-averaged
              profile, on its own

        Returns
        -------
        list of matplotlib.figure.Figure
        """
        ds = self.dataset(t)
        if not ds.data_vars:
            raise ValueError(
                f"{type(self).__name__}: nothing computed for this window.")
        figs = [fig for group in self._plot_groups(which)
                if (fig := getattr(self, f"_fig_{group}")(ds)) is not None]
        plt.show()
        return figs

    @staticmethod
    def _plot_groups(which):
        """Normalise the ``which`` argument to a validated tuple of groups."""
        if which == "all":
            return _DEFAULT_GROUPS
        groups = (which,) if isinstance(which, str) else tuple(which)
        bad = [g for g in groups if g not in _PLOT_GROUPS]
        if bad:
            raise ValueError(
                f"unknown plot group(s) {bad}; expected any of "
                f"{list(_PLOT_GROUPS)} or 'all'")
        return groups

    @property
    def _x_label(self):
        """
        Radial axis label.

        GENE-3D indexes its dataset by ``x/a``; the spectral paths carry the
        grid straight from ``coord['x']``, which is in ``rho_ref``.
        """
        return r"$x/a$" if self.is_3d else r"$x/\rho_{\rm ref}$"

    def _panels(self, ds):
        """The (variable, label) pairs from ``_PANEL_VARS`` present in *ds*."""
        return [(name, label) for name, label in _PANEL_VARS if name in ds]

    def _xt_map(self, fig, ax, ds, name, label):
        """One x-t colour map, on a symmetric scale robust to outliers."""
        arr = np.asarray(ds[name])
        finite = arr[np.isfinite(arr)]
        vmax = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
        mesh = ax.pcolormesh(np.asarray(ds["time"]), np.asarray(ds["x"]), arr.T,
                             cmap="bwr", vmin=-vmax, vmax=vmax or 1.0,
                             shading="auto")
        ax.set_xlabel(_T_LABEL)
        ax.set_ylabel(self._x_label)
        ax.set_title(label)
        fig.colorbar(mesh, ax=ax)

    def _fig_maps(self, ds):
        """x-t colour maps of every radial quantity."""
        panels = self._panels(ds)
        if not panels:
            return None
        fig, axes = plt.subplots(1, len(panels), figsize=(4.7 * len(panels), 4),
                                 squeeze=False)
        for ax, (name, label) in zip(axes[0], panels):
            self._xt_map(fig, ax, ds, name, label)
        fig.suptitle("ExB shearing — x-t maps")
        fig.tight_layout()
        return fig

    def _fig_profiles(self, ds):
        """Time-averaged radial profiles, trapezoidal over the uneven t axis."""
        panels = self._panels(ds)
        if not panels:
            return None
        x = np.asarray(ds["x"])
        fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4),
                                 squeeze=False)
        for ax, (name, label) in zip(axes[0], panels):
            ax.plot(x, np.asarray(self._t_average(ds[name])))
            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.set_xlabel(self._x_label)
            ax.set_ylabel(rf"$\langle$ {label} $\rangle_t$")
            ax.grid(True, alpha=0.3)
        fig.suptitle("ExB shearing — time-averaged profiles")
        fig.tight_layout()
        return fig

    def _fig_summary(self, ds):
        """
        RMS shearing rate and the supporting views.

        Which panels exist is geometry-dependent — the spectral paths save a
        per-time ``shearing_rms`` and, for a flux tube, the zonal kx spectrum;
        GENE-3D gives the RMS reduced over each axis separately. Only the
        panels backed by data are drawn.
        """
        times = np.asarray(ds["time"])
        x = np.asarray(ds["x"])
        draw = []

        if "shearing_rms" in ds:
            draw.append(("trace", "shearing_rms",
                         r"$\omega_{E\times B}^{\rm rms}$"))
        if "omega_exb_rms_t" in ds:
            draw.append(("trace", "omega_exb_rms_t",
                         r"$\langle|\omega_E|^2\rangle_x^{1/2}$"))
        if "omega_exb_rms_x" in ds:
            draw.append(("profile", "omega_exb_rms_x",
                         r"$\langle|\omega_E|^2\rangle_t^{1/2}$"))
        if "omega_exb" in ds and times.size:
            draw.append(("snapshots", "omega_exb",
                         r"$\omega_{E\times B}$"))
        if _KX_VAR in ds:
            draw.append(("kx", _KX_VAR,
                         r"$|\hat{\phi}_{\rm zonal}(k_x)|$"))
        if not draw:
            return None

        fig, axes = plt.subplots(1, len(draw), figsize=(4.7 * len(draw), 4),
                                 squeeze=False)
        for ax, (kind, name, label) in zip(axes[0], draw):
            if kind == "trace":
                ax.plot(times, np.asarray(ds[name]), color="steelblue")
                ax.axhline(float(np.nanmean(np.asarray(ds[name]))),
                           ls="--", color="k", lw=0.8)
                ax.set_xlabel(_T_LABEL)
                ax.grid(True, alpha=0.3)
            elif kind == "profile":
                ax.plot(x, np.asarray(ds[name]), color="steelblue")
                ax.set_xlabel(self._x_label)
                ax.grid(True, alpha=0.3)
            elif kind == "snapshots":
                arr = np.asarray(ds[name])
                for ti in dict.fromkeys((0, len(times) // 2, len(times) - 1)):
                    ax.plot(x, arr[ti], label=f"t={times[ti]:.3g}")
                ax.set_xlabel(self._x_label)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:                                   # kx spectrum
                spec = np.asarray(ds[name])
                kx = np.asarray(ds["kx"]) if "kx" in ds.coords else \
                    np.arange(spec.shape[1])
                mesh = ax.pcolormesh(times, kx, spec.T, cmap="inferno",
                                     vmin=0, vmax=np.percentile(spec, 98),
                                     shading="auto")
                ax.set_xlabel(_T_LABEL)
                ax.set_ylabel(r"$k_x \rho_{\rm ref}$")
                fig.colorbar(mesh, ax=ax)
            ax.set_title(label)
        fig.suptitle("ExB shearing — summary")
        fig.tight_layout()
        return fig

    def _fig_zonal(self, ds):
        """
        The zonal potential on its own: x-t map beside its t-averaged profile.

        This is the view the separate ``Zonal`` diagnostic used to give. The
        spectral paths cache ``phi_zonal`` but drew only E_r and ω_ExB, so
        without this the potential itself was never plotted for them.
        """
        if "phi_zonal" not in ds:
            return None
        x = np.asarray(ds["x"])
        phiz = np.asarray(ds["phi_zonal"])
        fig, (ax_xt, ax_prof) = plt.subplots(
            1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]})

        vmax = float(np.max(np.abs(phiz))) or 1.0
        pcm = ax_xt.pcolormesh(np.asarray(ds["time"]), x, phiz.T, cmap="bwr",
                               vmin=-vmax, vmax=vmax, shading="auto")
        ax_xt.set_xlabel(_T_LABEL)
        ax_xt.set_ylabel(self._x_label)
        ax_xt.set_title(r"$\langle\phi\rangle_{\rm zonal}(x,t)$")
        fig.colorbar(pcm, ax=ax_xt)

        ax_prof.plot(np.asarray(self._t_average(ds["phi_zonal"])), x)
        ax_prof.set_xlabel(r"$\langle\phi\rangle_{\rm zonal}$ (t-avg)")
        ax_prof.set_ylabel(self._x_label)
        ax_prof.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
