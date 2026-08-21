# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
geometry_plots.py — inspection of GENE-3D geometry coefficients.

A GENE-3D geometry file holds fully three-dimensional metric and field terms, so
there is no single curve to look at. This presents each coefficient as three
cuts through a chosen point and the three 2-D planes through it, which is enough
to spot the usual problems: a metric that goes negative, a Jacobian that changes
sign, or a discontinuity at the ``z`` boundary from a bad ``q`` profile.

It reads the geometry file only, so it needs no field or moment output and works
on a run that has not started producing data yet.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from genetools._xr import make_dataset, unit_attrs
from genetools.diagnostics._base import RunDiagnostic

#: Coefficient -> plot label. Curvature is included when the file has it.
_LABELS = {
    "gxx": r"$g^{xx}$", "gxy": r"$g^{xy}$", "gxz": r"$g^{xz}$",
    "gyy": r"$g^{yy}$", "gyz": r"$g^{yz}$", "gzz": r"$g^{zz}$",
    "Bfield": r"$B$", "dBdx": r"$\partial B/\partial x$",
    "dBdy": r"$\partial B/\partial y$", "dBdz": r"$\partial B/\partial z$",
    "Jacobian": r"$J$", "K_x": r"$K_x$", "K_y": r"$K_y$",
}

#: Radial profiles GENE-3D writes alongside the 3-D coefficients.
_PROFILE_LABELS = {
    "q": r"$q$", "dVdx": r"$dV/dx$", "sqrtgxx_fs": r"$\langle\sqrt{g^{xx}}\rangle$",
    "gxx_fs": r"$\langle g^{xx}\rangle$", "dpdx_pm_arr": r"$dp/dx$",
}


class GeometryPlots(RunDiagnostic):
    """
    Geometry coefficients of a GENE-3D run.

    Parameters
    ----------
    run : genetools.run.Run
    x_index, y_index, z_index : int, optional
        Index of the point the cuts pass through. Each defaults to the middle
        of its axis.
    """

    name = "geometry"
    supported = ("xy_global",)

    def __init__(self, run, x_index=None, y_index=None, z_index=None):
        super().__init__(run)
        geom = run.geometry[0]
        nx, ny, nz = np.shape(geom["Jacobian"])
        self.ix = nx // 2 if x_index is None else int(x_index)
        self.iy = ny // 2 if y_index is None else int(y_index)
        self.iz = nz // 2 if z_index is None else int(z_index)

    # ------------------------------------------------------------------

    def _fields(self):
        """Collect the available 3-D coefficients, keyed by label name."""
        geom = self.run.geometry[0]
        out = {}
        for name in _LABELS:
            if name.startswith("g"):
                arr = geom["metric"].get(name)
            else:
                # `a or b` would try the truth value of an ndarray.
                arr = geom.get(name)
                if arr is None:
                    arr = (geom.get("curv") or {}).get(name)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 3:
                out[name] = arr
        return out

    def dataset(self):
        """Return the coefficients as an :class:`xarray.Dataset`."""
        run = self.run
        params = run.params.get(0)
        coord = run.coords[0]
        geom = run.geometry[0]

        data_vars = {name: (("x", "y", "z"), arr)
                     for name, arr in self._fields().items()}
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
            arr = geom["area"].get(key)
            if arr is not None and np.ndim(arr) == 1:
                data_vars[name] = (("x",), np.asarray(arr, dtype=float))

        ds = make_dataset(
            data_vars,
            {"x": coord["x_o_a"], "y": coord["y"], "z": coord["z"]},
            params=params)
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = run.geometry_kind
        ds.attrs["magn_geometry"] = str(geom.get("kind", ""))
        ds.attrs["cut_indices"] = [self.ix, self.iy, self.iz]
        for key, value in (geom.get("local") or {}).items():
            if value is not None and np.isscalar(value):
                ds.attrs[f"local_{key}"] = float(value)
        return ds


    def save(self, path=None):
        ds = self.dataset()
        out = path or str(self.run.path / "geometry3d.nc")
        ds.to_netcdf(out)
        return out

    # ------------------------------------------------------------------

    def plot(self, t=None, names=None, **kw):
        """
        One figure per coefficient: three cuts and three planes.

        *t* is accepted and ignored — geometry is time-independent — so this
        plots through the same facade as every other diagnostic.
        """
        ds = self.dataset()
        fields = self._fields()
        wanted = [n for n in (names or fields) if n in fields]
        if not wanted:
            raise ValueError("No 3-D geometry coefficients available to plot.")

        x = np.asarray(ds["x"])
        y = np.asarray(ds["y"])
        z = np.asarray(ds["z"]) / np.pi

        for name in wanted:
            arr = fields[name]
            label = _LABELS.get(name, name)
            fig, axes = plt.subplots(2, 3, figsize=(14, 7))

            axes[0][0].plot(z, arr[self.ix, self.iy, :])
            axes[0][0].set_xlabel(r"$z/\pi$")
            axes[0][0].set_title(f"{label} at x[{self.ix}], y[{self.iy}]")

            axes[0][1].plot(x, arr[:, self.iy, self.iz])
            axes[0][1].set_xlabel(r"$x/a$")
            axes[0][1].set_title(f"{label} at y[{self.iy}], z[{self.iz}]")

            axes[0][2].plot(y, arr[self.ix, :, self.iz])
            axes[0][2].set_xlabel(r"$y/\rho_{\rm ref}$")
            axes[0][2].set_title(f"{label} at x[{self.ix}], z[{self.iz}]")

            for ax, (h, v, plane, hl, vl) in zip(axes[1], (
                    (z, x, arr[:, self.iy, :], r"$z/\pi$", r"$x/a$"),
                    (z, y, arr[self.ix, :, :], r"$z/\pi$",
                     r"$y/\rho_{\rm ref}$"),
                    (y, x, arr[:, :, self.iz], r"$y/\rho_{\rm ref}$",
                     r"$x/a$"))):
                mesh = ax.pcolormesh(h, v, plane, shading="nearest")
                ax.set_xlabel(hl)
                ax.set_ylabel(vl)
                fig.colorbar(mesh, ax=ax)

            for ax in axes[0]:
                ax.grid(True, alpha=0.3)
            fig.suptitle(label)
            fig.tight_layout()

        # Radial profiles, if the file carried any.
        radial = [n for n in _PROFILE_LABELS if n in ds]
        if radial:
            fig, axes = plt.subplots(1, len(radial),
                                     figsize=(4 * len(radial), 3.4),
                                     squeeze=False)
            for ax, name in zip(axes[0], radial):
                ax.plot(x, np.asarray(ds[name]))
                ax.set_xlabel(r"$x/a$")
                ax.set_title(_PROFILE_LABELS[name])
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
        plt.show()
