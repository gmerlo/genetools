# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
slices.py — GENE-3D slices.

One reduction engine covering every projection of a 3-D snapshot: the three
2-D planes (``xy``, ``xz``, ``yz``) and the three 1-D profiles (``x``, ``y``,
``z``), each averaged over the coordinates it drops and optionally restricted to
an index window in them. Either horizontal direction can be viewed in Fourier
space instead of real space.

:class:`~genetools.diagnostics.contours.Contours` reuses this for its GENE-3D
path, keeping just the ``xy`` plane — the usual view of turbulence structure.

Averages here are plain means, not Jacobian-weighted: a slice is a picture of the
field on the grid, and weighting it by the volume element would show the metric
as much as the turbulence. The flux and profile diagnostics, which do need the
volume element, weight explicitly.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from genetools._xr import make_dataset, unit_attrs
from genetools.diagnostics._base import RunDiagnostic
from genetools.diagnostics import _gene3d as c

_PLANES = {"xy": 2, "xz": 1, "yz": 0}          # reduction -> axis removed
_LINES = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}

_AXIS_LABELS = {
    "x": r"$x/a$", "y": r"$y/\rho_{\rm ref}$", "z": r"$z/\pi$",
    "kx": r"$k_x \rho_{\rm ref}$", "ky": r"$k_y \rho_{\rm ref}$",
}


class Slices(RunDiagnostic):
    """
    Reduced views of GENE-3D field and moment snapshots.

    Parameters
    ----------
    run : genetools.run.Run
    quantities : sequence of str, optional
        Variable names to reduce, from the field or moment files. Defaults to
        ``('phi',)``.
    species : str, optional
        Species whose moment file supplies any moment quantities. Defaults to
        the first species.
    x_fourier, y_fourier : bool
        View that direction in Fourier space (``|FFT|``).
    square : bool
        Reduce ``|f|^2`` rather than ``f``.
    xlim, ylim, zlim : (float, float), optional
        Restrict the averaged-over ranges. ``xlim`` is in ``x/a``, ``ylim`` in
        ``y/rho_ref``, ``zlim`` in ``z``.
    t_avg : bool
        Average over time instead of keeping the time axis.
    """

    name = "slices"
    supported = ("xy_global",)

    def __init__(self, run, quantities=("phi",), species=None,
                 x_fourier=False, y_fourier=False, square=False,
                 xlim=None, ylim=None, zlim=None, t_avg=False):
        super().__init__(run)
        self.quantities = tuple(quantities)
        self.species = species or (run.species[0] if run.species else None)
        self.x_fourier = bool(x_fourier)
        self.y_fourier = bool(y_fourier)
        self.square = bool(square)
        self.xlim, self.ylim, self.zlim = xlim, ylim, zlim
        self.t_avg = bool(t_avg)

    # ------------------------------------------------------------------

    @property
    def x_axis(self) -> str:
        return "kx" if self.x_fourier else "x"

    @property
    def y_axis(self) -> str:
        return "ky" if self.y_fourier else "y"

    def _sources(self):
        """Map each requested quantity to the reader that holds it."""
        fld = self.run.field
        mom = self.run.mom(self.species) if self.species else None
        out = {}
        for name in self.quantities:
            if c.has_var(fld, name):
                out.setdefault(id(fld), (fld, []))[1].append(name)
            elif mom is not None and c.has_var(mom, name):
                out.setdefault(id(mom), (mom, []))[1].append(name)
            else:
                available = list(fld.var_names) + (
                    list(mom.var_names) if mom is not None else [])
                raise KeyError(
                    f"unknown quantity {name!r}; available: "
                    f"{', '.join(available)}")
        return list(out.values())

    def _windows(self, coord, shape):
        """Index slices for the three coordinates, from the requested limits."""
        nx, ny, nz = shape
        xsl = (c.radial_slice(coord["x_o_a"], limits=self.xlim)
               if self.xlim else slice(None))
        ysl = _index_window(coord["y"], self.ylim, ny)
        zsl = _index_window(coord["z"], self.zlim, nz)
        return xsl, ysl, zsl

    def _transform(self, var):
        """Apply the requested Fourier views, then ``|.|`` or ``|.|^2``."""
        out = var
        if self.x_fourier:
            out = np.fft.fftshift(np.abs(c.to_kx(out)), axes=0)
        if self.y_fourier:
            out = np.fft.fftshift(np.abs(c.to_ky(out)), axes=1)
        if self.square:
            out = np.abs(out) ** 2
        return out

    def compute(self, t=None):
        """Stream the requested variables and build every reduction."""
        key = (tuple(t) if isinstance(t, (tuple, list)) else t)
        if key in self._cache:
            return self._cache[key]

        coord = self.coord

        acc, times = {}, None
        for reader, names in self._sources():
            _, idx = self._indices(reader, t)
            slots = {n: reader.index_of(n) for n in names}
            got = []
            for time, arrays in reader.stream_selected(idx):
                got.append(time)
                for n in names:
                    var = self._transform(arrays[slots[n]])
                    xsl, ysl, zsl = self._windows(coord, var.shape)
                    store = acc.setdefault(n, {})
                    for plane, axis in _PLANES.items():
                        sub = _apply(var, xsl, ysl, zsl, keep=plane)
                        store.setdefault(plane, []).append(sub.mean(axis=axis))
                    for line, axes in _LINES.items():
                        sub = _apply(var, xsl, ysl, zsl, keep=line)
                        store.setdefault(line, []).append(sub.mean(axis=axes))
            if times is None:
                times = np.asarray(got)

        reduced = {}
        for name, store in acc.items():
            reduced[name] = {}
            for red, stack in store.items():
                arr = np.asarray(stack)
                reduced[name][red] = (self._time_average(arr, times)
                                      if self.t_avg else arr)

        result = {"reduced": reduced, "times": times, "coord": coord}
        self._cache[key] = result
        return result

    # ------------------------------------------------------------------

    def _axis_values(self, coord):
        """Coordinate values for each named axis, honouring the Fourier views."""
        return {
            "x": (np.fft.fftshift(np.asarray(coord["kx"])) if self.x_fourier
                  else np.asarray(coord["x_o_a"])),
            "y": (np.fft.fftshift(np.asarray(coord["ky"])) if self.y_fourier
                  else np.asarray(coord["y"])),
            "z": np.asarray(coord["z"]),
        }

    def dataset(self, t=None):
        """Return every reduction as an :class:`xarray.Dataset`."""
        raw = self.compute(t)
        params = self.params
        coord = raw["coord"]
        axis_vals = self._axis_values(coord)
        rename = {"x": self.x_axis, "y": self.y_axis, "z": "z"}

        data_vars, candidates = {}, {}
        for name, store in raw["reduced"].items():
            for red, arr in store.items():
                dims = tuple(rename[ch] for ch in red)
                if not self.t_avg:
                    dims = ("time",) + dims
                data_vars[f"{name}_{red}"] = (dims, np.asarray(arr))
        for ch, axis_name in rename.items():
            candidates[axis_name] = axis_vals[ch]
        if not self.t_avg:
            candidates["time"] = raw["times"]

        ds = make_dataset(data_vars, candidates, params=params)
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = self.geometry_kind
        ds.attrs["x_fourier"] = int(self.x_fourier)
        ds.attrs["y_fourier"] = int(self.y_fourier)
        ds.attrs["squared"] = int(self.square)
        if self.species:
            ds.attrs["species"] = self.species
        return ds

    # ------------------------------------------------------------------

    def plot(self, t=None, **kw):
        """Plot the three planes and the three line profiles per quantity."""
        ds = self.dataset(t)
        for name in self.quantities:
            planes = [p for p in _PLANES if f"{name}_{p}" in ds]
            lines = [l for l in _LINES if f"{name}_{l}" in ds]
            fig, axes = plt.subplots(2, max(len(planes), len(lines)),
                                     figsize=(13, 7), squeeze=False)
            for ax, plane in zip(axes[0], planes):
                self._plot_plane(fig, ax, ds, name, plane)
            for ax in axes[0][len(planes):]:
                ax.set_visible(False)
            for ax, line in zip(axes[1], lines):
                self._plot_line(ax, ds, name, line)
            for ax in axes[1][len(lines):]:
                ax.set_visible(False)
            fig.suptitle(self._title(name))
            fig.tight_layout()
        plt.show()

    def _title(self, name):
        base = f"{name}" + (f" ({self.species})" if self.species
                            and name not in self.run.field.var_names else "")
        if self.square:
            base = f"|{base}|²"
        return base

    def _plot_plane(self, fig, ax, ds, name, plane):
        da = ds[f"{name}_{plane}"]
        if "time" in da.dims:
            da = self._t_average(da)
        dims = da.dims
        h, v = np.asarray(ds[dims[0]]), np.asarray(ds[dims[1]])
        mesh = ax.pcolormesh(v, h, np.asarray(da), shading="nearest")
        ax.set_xlabel(_AXIS_LABELS.get(dims[1], dims[1]))
        ax.set_ylabel(_AXIS_LABELS.get(dims[0], dims[0]))
        fig.colorbar(mesh, ax=ax)

    def _plot_line(self, ax, ds, name, line):
        da = ds[f"{name}_{line}"]
        if "time" in da.dims:
            da = self._t_average(da)
        dim = da.dims[0]
        ax.plot(np.asarray(ds[dim]), np.asarray(da))
        ax.set_xlabel(_AXIS_LABELS.get(dim, dim))
        ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply(var, xsl, ysl, zsl, keep):
    """Apply the index windows, leaving the axes named in *keep* unrestricted."""
    sx = slice(None) if "x" in keep else xsl
    sy = slice(None) if "y" in keep else ysl
    sz = slice(None) if "z" in keep else zsl
    return var[sx, sy, sz]


def _index_window(values, limits, n):
    """Return an index slice covering *limits* in the coordinate *values*."""
    if limits is None:
        return slice(None)
    arr = np.asarray(values, dtype=float)
    if arr.size != n:
        return slice(None)
    lo, hi = float(limits[0]), float(limits[1])
    i0 = int(np.argmin(np.abs(arr - lo)))
    i1 = int(np.argmin(np.abs(arr - hi)))
    if i1 < i0:
        i0, i1 = i1, i0
    # Equal bounds mean one grid point, not two: widening would average in a
    # neighbour, which defeats asking for a single slice.
    return slice(i0, i1 + 1)
