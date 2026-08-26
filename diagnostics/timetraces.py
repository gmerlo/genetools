# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
timetraces.py — GENE-3D volume-averaged and ky-resolved time traces.

Two views of how a quantity evolves: its Jacobian-weighted volume average over a
chosen sub-box, and the ``|f(k_y)|^2`` spectrum at each output time. The second
is what shows a single binormal mode taking over, or the zonal (``k_y = 0``)
component separating from the rest.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from genetools._xr import make_dataset, unit_attrs
from genetools.diagnostics._base import RunDiagnostic
from genetools.diagnostics import _gene3d as g3


class TimeTraces(RunDiagnostic):
    """
    Time traces of GENE-3D fields and moments.

    Parameters
    ----------
    run : genetools.run.Run
    quantities : sequence of str
        Variables to trace, from the field or moment file.
    species : str, optional
        Species supplying moment quantities (default: the first).
    xlim, ylim, zlim : (float, float), optional
        Sub-box over which to average, in ``x/a``, ``y/rho_ref`` and ``z``.
    """

    name = "timetraces"
    supported = ("xy_global",)

    def __init__(self, run, quantities=("phi",), species=None,
                 xlim=None, ylim=None, zlim=None):
        super().__init__(run)
        self.quantities = tuple(quantities)
        self.species = species or (run.species[0] if run.species else None)
        self.xlim, self.ylim, self.zlim = xlim, ylim, zlim
        self._cache = {}

    # ------------------------------------------------------------------

    def compute(self, t=None):
        """Stream the requested variables and accumulate both trace kinds."""
        key = tuple(t) if isinstance(t, (tuple, list)) else t
        if key in self._cache:
            return self._cache[key]

        run = self.run
        J = run.geometry[0]["Jacobian"]
        J_yz = g3.jacobian_yz(J)
        coord = run.coords[0]
        nx, ny, nz = J.shape
        xsl = g3.index_window(coord["x_o_a"], self.xlim, nx)
        ysl = g3.index_window(coord["y"], self.ylim, ny)
        zsl = g3.index_window(coord["z"], self.zlim, nz)

        traces, ky_traces, times = {}, {}, None
        for reader, names in self._sources(self.quantities,
                                        self.species):
            _, idx = self._indices(reader, t)
            slots = {n: reader.index_of(n) for n in names}
            acc = {n: [] for n in names}
            acc_ky = {n: [] for n in names}
            got = []
            for time, arrays in reader.stream_selected(idx):
                got.append(time)
                for n in names:
                    var = arrays[slots[n]]
                    acc[n].append(np.average(var[xsl, ysl, zsl],
                                             weights=J[xsl, ysl, zsl]))
                    power = np.abs(g3.to_ky(var)) ** 2
                    acc_ky[n].append(np.average(power[xsl][:, :, zsl],
                                                weights=J_yz[xsl][:, :, zsl],
                                                axis=(0, 2)))
            for n in names:
                traces[n] = np.asarray(acc[n])
                ky_traces[n] = np.asarray(acc_ky[n])
            if times is None:
                times = np.asarray(got)

        result = {"times": times, "traces": traces, "ky_traces": ky_traces,
                  "ky": np.asarray(coord["ky"], dtype=float),
                  "x_range": _range_of(coord["x_o_a"], xsl)}
        self._cache[key] = result
        return result

    # ------------------------------------------------------------------

    def dataset(self, t=None):
        """Return the traces as an :class:`xarray.Dataset`."""
        raw = self.compute(t)
        params = self.run.params.get(0)
        data_vars = {}
        for name, arr in raw["traces"].items():
            data_vars[name] = (("time",), arr)
        for name, arr in raw["ky_traces"].items():
            data_vars[f"{name}_ky"] = (("time", "ky"), arr)
        ds = make_dataset(data_vars, {"time": raw["times"], "ky": raw["ky"]},
                          params=params)
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = self.run.geometry_kind
        ds.attrs["x_avg_range"] = list(raw["x_range"])
        if self.species:
            ds.attrs["species"] = self.species
        return ds



    # ------------------------------------------------------------------

    def plot(self, t=None, log=False, n_modes=6, **kw):
        """
        Left: the volume-averaged trace. Right: the loudest ``k_y`` modes.

        Only the *n_modes* strongest non-negative modes are drawn — a full ky
        grid produces an unreadable tangle, and the point is which few modes
        dominate.
        """
        ds = self.dataset(t)
        times = np.asarray(ds["time"])
        ky = np.asarray(ds["ky"])
        n_pos = (ky.size + 1) // 2

        bases = [n for n in ds.data_vars if not n.endswith("_ky")]
        fig, axes = plt.subplots(len(bases), 2,
                                 figsize=(11, 3.4 * len(bases)), squeeze=False)
        for row, name in enumerate(bases):
            ax = axes[row][0]
            ax.plot(times, np.asarray(ds[name]))
            ax.set_xlabel(r"$t\;[L_{\rm ref}/c_{\rm ref}]$")
            ax.set_ylabel(name)
            if log:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

            ax = axes[row][1]
            key = f"{name}_ky"
            if key in ds:
                arr = np.asarray(ds[key])[:, :n_pos]
                order = np.argsort(arr.mean(axis=0))[::-1][:n_modes]
                for j in sorted(order):
                    ax.plot(times, arr[:, j],
                            label=rf"$k_y$={ky[j]:.3g}")
                ax.set_yscale("log")
                ax.set_xlabel(r"$t\;[L_{\rm ref}/c_{\rm ref}]$")
                ax.set_ylabel(f"$|{name}(k_y)|^2$")
                ax.legend(fontsize=7)
                ax.grid(True, which="both", alpha=0.3)
            else:
                ax.set_visible(False)
        fig.tight_layout()
        plt.show()


def _range_of(values, sl):
    arr = np.asarray(values, dtype=float)[sl]
    return float(arr[0]), float(arr[-1])

