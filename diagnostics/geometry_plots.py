# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
geometry_plots.py — inspection of GENE geometry coefficients.

Every geometry kind writes the same set of metric and field terms; only their
rank differs, and every reader in :mod:`genetools.io.geometry` puts them under
the same keys::

    coefficient        flux_tube    x_global     xy_global
    Bfield, Jacobian   (nz,)        (nx, nz)     (nx, ny, nz)
    g^xx ... g^zz      (nz,)        (nx, nz)     (nx, ny, nz)
    K_x, K_y, sloc     (nz,)        (nx, nz)     (nx, ny, nz)
    shape gR/gZ/gPhi   (nz,)        (nx, nz)     — not written
    profiles q, dVdx   — none       q only       all of them

So this dispatches on the rank of the array rather than on the geometry, and
shows whatever the run actually carries. That matters in both directions: a flux
tube has a flux-surface shape and a local shear profile that a GENE-3D file does
not, and a GENE-3D file has radial profiles and an area profile that a flux tube
does not.

It reads the geometry file only, so it needs no field or moment output and works
on a run that has not started producing data yet.

What each view shows
--------------------
``overview``  every coefficient in one figure — a curve for a flux tube, an
              ``(x, z)`` map otherwise.
``detail``    one figure per coefficient: the cuts and planes through a chosen
              point. For a flux tube there is only one axis, so this is the same
              curve as the overview and is skipped.
``surface``   the poloidal cross-section, ``Z`` against ``R``, from the ``shape``
              arrays. The quickest check that a geometry is the shape intended.
``profiles``  the radial profiles written alongside the coefficients.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from genetools._xr import make_dataset, unit_attrs
from genetools.diagnostics._base import RunDiagnostic

#: Coefficient -> plot label. Anything absent from a given file is skipped.
_LABELS = {
    "gxx": r"$g^{xx}$", "gxy": r"$g^{xy}$", "gxz": r"$g^{xz}$",
    "gyy": r"$g^{yy}$", "gyz": r"$g^{yz}$", "gzz": r"$g^{zz}$",
    "Bfield": r"$B$", "dBdx": r"$\partial B/\partial x$",
    "dBdy": r"$\partial B/\partial y$", "dBdz": r"$\partial B/\partial z$",
    "Jacobian": r"$J$", "K_x": r"$K_x$", "K_y": r"$K_y$",
    "sloc": r"$s_{\rm loc}$",
}

#: Radial profiles a geometry file may carry alongside the coefficients.
_PROFILE_LABELS = {
    "q": r"$q$", "dVdx": r"$dV/dx$",
    "sqrtgxx_fs": r"$\langle\sqrt{g^{xx}}\rangle$",
    "gxx_fs": r"$\langle g^{xx}\rangle$", "dpdx_pm_arr": r"$dp/dx$",
}

#: Flux-surface shape arrays, drawn as a cross-section rather than as curves.
_SHAPE_KEYS = ("gR", "gZ", "gPhi")

#: Views :meth:`GeometryPlots.plot` can draw.
_VIEWS = ("overview", "detail", "surface", "profiles")
#: Drawn when ``which`` is not given. ``detail`` is one figure per coefficient,
#: so it is opt-in rather than part of a first look.
_DEFAULT_VIEWS = ("overview", "surface", "profiles")

_Z_LABEL = r"$z/\pi$"
_X_LABEL = r"$x/a$"
_Y_LABEL = r"$y/\rho_{\rm ref}$"


class GeometryPlots(RunDiagnostic):
    """
    Geometry coefficients of a GENE or GENE-3D run.

    Parameters
    ----------
    run : genetools.run.Run
    x_index, y_index, z_index : int, optional
        Index of the point the ``detail`` cuts pass through. Each defaults to
        the middle of its axis, and an axis the run does not have is ignored.
    """

    name = "geometry"

    def __init__(self, run, x_index=None, y_index=None, z_index=None):
        super().__init__(run)
        # The Jacobian's rank tells us which axes exist: (nz,), (nx, nz) or
        # (nx, ny, nz). Reading it from the array rather than the namelist keeps
        # this working whatever the file turns out to hold.
        shape = np.shape(run.geometry[0]["Jacobian"])
        self.ndim = len(shape)
        nx, ny, nz = self._axis_sizes(shape)
        self.ix = (nx // 2 if x_index is None else int(x_index)) if nx else None
        self.iy = (ny // 2 if y_index is None else int(y_index)) if ny else None
        self.iz = nz // 2 if z_index is None else int(z_index)

    @staticmethod
    def _axis_sizes(shape):
        """Return ``(nx, ny, nz)``, with ``None`` for an axis that is absent."""
        if len(shape) == 1:
            return None, None, shape[0]
        if len(shape) == 2:
            return shape[0], None, shape[1]
        return shape

    # ------------------------------------------------------------------
    # Collecting what the file holds
    # ------------------------------------------------------------------

    def _fields(self):
        """
        Collect the coefficients present in the file, keyed by label name.

        Anything whose rank does not match the Jacobian's is skipped — a scalar
        ``C_xy`` on a flux tube is not a coefficient to plot alongside ``g^xx``.
        """
        geom = self.run.geometry[0]
        out = {}
        for name in _LABELS:
            if name.startswith("g") and name != "gxx_fs":
                arr = geom["metric"].get(name)
            else:
                # `a or b` would take the truth value of an ndarray.
                arr = geom.get(name)
                if arr is None:
                    arr = (geom.get("curv") or {}).get(name)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == self.ndim:
                out[name] = arr
        return out

    def _shape_arrays(self):
        """The flux-surface shape arrays, or ``{}`` when the file has none."""
        shape = self.run.geometry[0].get("shape") or {}
        out = {}
        for key in _SHAPE_KEYS:
            arr = shape.get(key)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            if arr.ndim >= 1:
                out[key] = arr
        return out if {"gR", "gZ"} <= set(out) else {}

    def _dims(self):
        """Dimension names matching the rank of this run's coefficients."""
        return {1: ("z",), 2: ("x", "z"), 3: ("x", "y", "z")}[self.ndim]

    # ------------------------------------------------------------------

    def dataset(self):
        """Return the coefficients as an :class:`xarray.Dataset`."""
        run = self.run
        params = run.params.get(0)
        coord = run.coords[0]
        geom = run.geometry[0]
        dims = self._dims()

        data_vars = {name: (dims, arr)
                     for name, arr in self._fields().items()}
        for name, arr in self._shape_arrays().items():
            if arr.ndim == self.ndim:
                data_vars[name] = (dims, arr)

        profiles = geom.get("profiles", {}) or {}
        for name in _PROFILE_LABELS:
            arr = profiles.get(name)
            if arr is not None and np.ndim(arr) == 1:
                data_vars[name] = (("x",), np.asarray(arr, dtype=float))
        for name in ("C_y", "C_xy"):
            arr = geom["metric"].get(name)
            if arr is not None and np.ndim(arr) == 1:
                data_vars[name] = (("x",), np.asarray(arr, dtype=float))
        for name, key in (("Area", "Area"), ("dVdx_derived", "dVdx")):
            arr = (geom.get("area") or {}).get(key)
            if arr is not None and np.ndim(arr) == 1:
                data_vars[name] = (("x",), np.asarray(arr, dtype=float))

        candidates = {"z": coord["z"]}
        if self.ndim >= 2:
            candidates["x"] = coord.get("x_o_a", coord.get("x"))
        if self.ndim >= 3:
            candidates["y"] = coord["y"]
        ds = make_dataset(data_vars, candidates, params=params)
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = run.geometry_kind
        ds.attrs["magn_geometry"] = str(geom.get("kind", ""))
        ds.attrs["coefficient_rank"] = self.ndim
        ds.attrs["cut_indices"] = [v for v in (self.ix, self.iy, self.iz)
                                   if v is not None]
        for group in ("local", "area"):
            for key, value in (geom.get(group) or {}).items():
                if value is not None and np.isscalar(value):
                    ds.attrs[f"{group}_{key}"] = float(value)
        return ds

    def save(self, path=None):
        ds = self.dataset()
        out = path or str(self.run.path / "geometry.nc")
        ds.to_netcdf(out)
        return out

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    @staticmethod
    def _views(which):
        """Normalise the ``which`` argument to a validated tuple of views."""
        if which is None:
            return _DEFAULT_VIEWS
        if which == "all":
            return _VIEWS
        views = (which,) if isinstance(which, str) else tuple(which)
        bad = [v for v in views if v not in _VIEWS]
        if bad:
            raise ValueError(
                f"unknown geometry view(s) {bad}; expected any of "
                f"{list(_VIEWS)} or 'all'")
        return views

    def plot(self, t=None, names=None, which=None, **kw):
        """
        Plot the geometry coefficients.

        *t* is accepted and ignored — geometry is time-independent — so this
        plots through the same facade as every other diagnostic.

        Parameters
        ----------
        names : sequence of str, optional
            Restrict to these coefficients.
        which : str or sequence of str, optional
            Any of ``overview``, ``detail``, ``surface``, ``profiles``, or
            ``'all'``. Defaults to everything except ``detail``, which is one
            figure per coefficient.
        """
        views = self._views(which)
        ds = self.dataset()
        fields = self._fields()
        wanted = [n for n in (names or fields) if n in fields]
        if not wanted:
            raise ValueError(
                f"No geometry coefficients available to plot "
                f"(rank {self.ndim}, {self.geometry_kind}).")

        figs = []
        if "overview" in views:
            figs.append(self._fig_overview(ds, wanted))
        if "detail" in views and self.ndim > 1:
            figs.extend(self._fig_detail(ds, fields, n) for n in wanted)
        if "surface" in views:
            fig = self._fig_surface()
            if fig is not None:
                figs.append(fig)
        if "profiles" in views:
            fig = self._fig_profiles(ds)
            if fig is not None:
                figs.append(fig)
        plt.show()
        return figs

    # -- panels ---------------------------------------------------------

    def _panel(self, fig, ax, ds, name, arr):
        """
        Draw one coefficient at whatever rank it has.

        A flux tube gives a curve against z; anything radial gives an ``(x, z)``
        map with x horizontal, taking the mid-plane in y when there is a y axis.
        """
        label = _LABELS.get(name, name)
        if self.ndim == 1:
            ax.plot(np.asarray(ds["z"]) / np.pi, arr)
            ax.set_xlabel(_Z_LABEL)
            ax.grid(True, alpha=0.3)
        else:
            plane = arr if self.ndim == 2 else arr[:, self.iy, :]
            mesh = ax.pcolormesh(np.asarray(ds["x"]),
                                 np.asarray(ds["z"]) / np.pi,
                                 plane.T, shading="nearest")
            ax.set_xlabel(_X_LABEL)
            ax.set_ylabel(_Z_LABEL)
            fig.colorbar(mesh, ax=ax)
        ax.set_title(label, fontsize=9)

    def _fig_overview(self, ds, wanted):
        """Every coefficient in one figure, one panel each."""
        ncol = min(4, len(wanted))
        nrow = int(np.ceil(len(wanted) / ncol))
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(3.7 * ncol, 3.0 * nrow),
                                 squeeze=False)
        flat = axes.ravel()
        fields = self._fields()
        for ax, name in zip(flat, wanted):
            self._panel(fig, ax, ds, name, fields[name])
        for ax in flat[len(wanted):]:
            ax.set_visible(False)
        suffix = "" if self.ndim < 3 else f"  (y index {self.iy})"
        fig.suptitle(f"geometry coefficients — {self.geometry_kind}{suffix}")
        fig.tight_layout()
        return fig

    def _fig_detail(self, ds, fields, name):
        """Cuts and planes through the chosen point, for one coefficient."""
        arr = fields[name]
        label = _LABELS.get(name, name)
        z = np.asarray(ds["z"]) / np.pi
        x = np.asarray(ds["x"])

        if self.ndim == 2:
            fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
            axes[0].plot(z, arr[self.ix, :])
            axes[0].set_xlabel(_Z_LABEL)
            axes[0].set_title(f"{label} at x[{self.ix}]")
            axes[1].plot(x, arr[:, self.iz])
            axes[1].set_xlabel(_X_LABEL)
            axes[1].set_title(f"{label} at z[{self.iz}]")
            mesh = axes[2].pcolormesh(x, z, arr.T, shading="nearest")
            axes[2].set_xlabel(_X_LABEL)
            axes[2].set_ylabel(_Z_LABEL)
            fig.colorbar(mesh, ax=axes[2])
            for ax in axes[:2]:
                ax.grid(True, alpha=0.3)
            fig.suptitle(label)
            fig.tight_layout()
            return fig

        y = np.asarray(ds["y"])
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes[0][0].plot(z, arr[self.ix, self.iy, :])
        axes[0][0].set_xlabel(_Z_LABEL)
        axes[0][0].set_title(f"{label} at x[{self.ix}], y[{self.iy}]")
        axes[0][1].plot(x, arr[:, self.iy, self.iz])
        axes[0][1].set_xlabel(_X_LABEL)
        axes[0][1].set_title(f"{label} at y[{self.iy}], z[{self.iz}]")
        axes[0][2].plot(y, arr[self.ix, :, self.iz])
        axes[0][2].set_xlabel(_Y_LABEL)
        axes[0][2].set_title(f"{label} at x[{self.ix}], z[{self.iz}]")
        for ax, (h, v, plane, hl, vl) in zip(axes[1], (
                (x, z, arr[:, self.iy, :].T, _X_LABEL, _Z_LABEL),
                (y, z, arr[self.ix, :, :].T, _Y_LABEL, _Z_LABEL),
                (x, y, arr[:, :, self.iz].T, _X_LABEL, _Y_LABEL))):
            mesh = ax.pcolormesh(h, v, plane, shading="nearest")
            ax.set_xlabel(hl)
            ax.set_ylabel(vl)
            fig.colorbar(mesh, ax=ax)
        for ax in axes[0]:
            ax.grid(True, alpha=0.3)
        fig.suptitle(label)
        fig.tight_layout()
        return fig

    def _fig_surface(self):
        """
        The poloidal cross-section, ``Z`` against ``R``.

        One curve for a flux tube — the field line's poloidal projection — and
        one per radial position for a global run, giving the nested surfaces.
        Returns ``None`` when the file carries no shape arrays, which is the
        case for GENE-3D.
        """
        shape = self._shape_arrays()
        if not shape:
            return None
        R, Z = shape["gR"], shape["gZ"]
        fig, ax = plt.subplots(figsize=(5.2, 5.6))
        if R.ndim == 1:
            ax.plot(R, Z, ".-", lw=1.0, ms=3)
        else:
            for i in range(R.shape[0]):
                ax.plot(R[i], Z[i], lw=0.9)
        ax.set_xlabel(r"$R$")
        ax.set_ylabel(r"$Z$")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        fig.suptitle(f"poloidal cross-section — {self.geometry_kind}")
        fig.tight_layout()
        return fig

    def _fig_profiles(self, ds):
        """The radial profiles the file carried, if any."""
        radial = [n for n in _PROFILE_LABELS if n in ds]
        radial += [n for n in ("C_xy", "C_y", "Area", "dVdx_derived")
                   if n in ds]
        if not radial:
            return None
        x = np.asarray(ds["x"])
        ncol = min(4, len(radial))
        nrow = int(np.ceil(len(radial) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.2 * nrow),
                                 squeeze=False)
        flat = axes.ravel()
        for ax, name in zip(flat, radial):
            ax.plot(x, np.asarray(ds[name]))
            ax.set_xlabel(_X_LABEL)
            ax.set_title(_PROFILE_LABELS.get(name, name), fontsize=9)
            ax.grid(True, alpha=0.3)
        for ax in flat[len(radial):]:
            ax.set_visible(False)
        fig.suptitle("radial geometry profiles")
        fig.tight_layout()
        return fig
