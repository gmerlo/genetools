# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
velocity.py — GENE-3D source moments and velocity-space diagnostics.

Two optional outputs that share nothing but their obscurity:

:class:`SrcMom`
    ``srcmom_<species><ext>.h5`` — the Krook heat and particle source moments as
    radial profiles over time (``ck_heat_M00`` and friends). These say how much
    of the transport is being supplied by the sources rather than by the
    turbulence, which is what decides whether a "steady state" really is one.
    GENE-3D writes six of these; the reference GUI's variable map expects nine,
    including ``f0_term_*`` moments that GENE-3D does not produce at all.

:class:`VspSlice`
    ``vsp<ext>.h5`` — velocity-space data on the ``(z, v_par, mu, species)``
    grid. Written by GENE-3D as one 4-D array per quantity, which lands on disk
    with its axes reversed like everything else.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from genetools._xr import make_dataset, unit_attrs
from genetools.diagnostics._base import RunDiagnostic


class SrcMom(RunDiagnostic):
    """
    Krook source moments as radial profiles over time.

    Parameters
    ----------
    run : genetools.run.Run
    """

    name = "srcmom"
    supported = ("xy_global",)

    def __init__(self, run):
        super().__init__(run)

    def _reader(self, species):
        ext = self.run.extensions[0]
        params = self.run.params.get(0)
        from genetools.io.data import H5Reader
        return H5Reader("srcmom", self.run._folder, ext + ".h5", params,
                        species=species)

    def compute(self, t=None):
        """Read the source moments for every species."""
        key = tuple(t) if isinstance(t, (tuple, list)) else t
        if key in self._cache:
            return self._cache[key]

        per, times, labels = {}, None, None
        missing = []
        for name in self.run.species:
            reader = self._reader(name)
            try:
                _, idx = self._indices(reader, t)
            except (OSError, KeyError):
                # No srcmom file for this species; report which are missing
                # once the whole loop is done rather than per species.
                missing.append(name)
                continue
            labels = list(reader.var_names)
            stacks = {v: [] for v in labels}
            got = []
            for time, arrays in reader.stream_selected(idx):
                got.append(time)
                for j, v in enumerate(labels):
                    stacks[v].append(np.asarray(arrays[j], dtype=float))
            per[name] = {v: np.asarray(stacks[v]) for v in labels}
            if times is None:
                times = np.asarray(got)

        if not per:
            raise FileNotFoundError(
                f"No srcmom files found in {self.run.path}. GENE-3D writes "
                "them when istep_srcmom > 0.")

        result = {"species": per, "times": times, "labels": labels,
                  "missing": missing}
        self._cache[key] = result
        return result

    def dataset(self, t=None):
        """Return the source moments as an :class:`xarray.Dataset`."""
        raw = self.compute(t)
        params = self.run.params.get(0)
        coord = self.run.coords[0]
        names = [n for n in self.run.species if n in raw["species"]]
        data_vars = {
            v: (("species", "time", "x"),
                np.stack([raw["species"][n][v] for n in names], axis=0))
            for v in raw["labels"]}
        ds = make_dataset(data_vars,
                          {"x": coord["x_o_a"], "time": raw["times"]},
                          species=names, params=params)
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = self.run.geometry_kind
        return ds



    def plot(self, t=None, **kw):
        """Time-averaged radial profile of each source moment."""
        ds = self.dataset(t)
        x = np.asarray(ds["x"])
        names = sorted(ds.data_vars)
        ncol = 3
        nrow = int(np.ceil(len(names) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.2 * nrow),
                                 squeeze=False)
        flat = axes.ravel()
        for ax, name in zip(flat, names):
            for sp in ds["species"].values:
                ax.plot(x, np.asarray(self._t_average(ds[name].sel(species=sp))),
                        label=str(sp))
            ax.set_xlabel(r"$x/a$")
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        for ax in flat[len(names):]:
            ax.set_visible(False)
        fig.tight_layout()
        plt.show()


class VspSlice(RunDiagnostic):
    """
    GENE-3D velocity-space output on the ``(z, v_par, mu)`` grid.

    Parameters
    ----------
    run : genetools.run.Run
    z_index : int, optional
        Parallel index for the ``(v_par, mu)`` slices (default: the middle,
        i.e. the outboard midplane for a standard grid).
    """

    name = "vsp"
    supported = ("xy_global",)

    def __init__(self, run, z_index=None):
        super().__init__(run)
        self.z_index = z_index
        self._cache = {}

    def _reader(self):
        from genetools.io.data import H5Reader
        ext = self.run.extensions[0]
        return H5Reader("vsp", self.run._folder, ext + ".h5",
                        self.run.params.get(0))

    def compute(self, t=None):
        """Read every velocity-space quantity over the requested window."""
        key = tuple(t) if isinstance(t, (tuple, list)) else t
        if key in self._cache:
            return self._cache[key]

        reader = self._reader()
        _, idx = self._indices(reader, t)

        labels = list(reader.var_names)
        stacks = {v: [] for v in labels}
        times = []
        for time, arrays in reader.stream_selected(idx):
            times.append(time)
            for j, v in enumerate(labels):
                stacks[v].append(np.asarray(arrays[j], dtype=float))

        result = {"labels": labels, "times": np.asarray(times),
                  "data": {v: np.asarray(stacks[v]) for v in labels}}
        self._cache[key] = result
        return result

    def dataset(self, t=None):
        """Return the velocity-space data as an :class:`xarray.Dataset`."""
        raw = self.compute(t)
        run = self.run
        params = run.params.get(0)
        coord = run.coords[0]
        # Fortran order is (nz, nv, nw, n_spec).
        data_vars = {v: (("time", "z", "vpar", "mu", "species"), arr)
                     for v, arr in raw["data"].items()}
        ds = make_dataset(
            data_vars,
            {"time": raw["times"], "z": coord["z"], "vpar": coord["vp"],
             "mu": coord["mu"]},
            species=list(run.species), params=params)
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = run.geometry_kind
        return ds



    def plot(self, t=None, **kw):
        """``(v_par, mu)`` maps at one ``z``, time-averaged, per species."""
        ds = self.dataset(t)
        nz = ds.sizes["z"]
        iz = self.z_index if self.z_index is not None else nz // 2
        vpar = np.asarray(ds["vpar"]) if "vpar" in ds.coords else None
        mu = np.asarray(ds["mu"]) if "mu" in ds.coords else None

        names = sorted(ds.data_vars)
        species = list(ds["species"].values)
        fig, axes = plt.subplots(len(names), len(species),
                                 figsize=(4.6 * len(species),
                                          3.4 * len(names)), squeeze=False)
        for r, name in enumerate(names):
            for col, sp in enumerate(species):
                ax = axes[r][col]
                arr = np.asarray(self._t_average(
                    ds[name].sel(species=sp).isel(z=iz)))
                if vpar is not None and mu is not None and mu.size:
                    mesh = ax.pcolormesh(mu, vpar, arr, shading="nearest")
                    ax.set_xlabel(r"$\mu$")
                    ax.set_ylabel(r"$v_\parallel$")
                else:
                    mesh = ax.pcolormesh(arr, shading="nearest")
                ax.set_title(f"{name} — {sp}  (z index {iz})")
                fig.colorbar(mesh, ax=ax)
        fig.tight_layout()
        plt.show()

