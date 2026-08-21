# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
amplitude.py — amplitude spectra of the fields and moments.

The time-averaged ``|f|^2`` spectrum of each field (φ, A∥, B∥) and the leading
moments, resolved differently in each geometry because the storage differs:

**flux tube** — kx and ky are already the storage axes. Spectra are the
Hermitian-weighted (ky=0 unweighted, ky>0 ×2) Jacobian-weighted z-averages, via
:meth:`~genetools.diagnostics.spectra.Spectra.averages`.

**x-global** — x is real space, ky spectral. The full 2-D map ``|f|^2(x, ky)`` is
built (flux-surface averaged over z with a per-surface-normalised Jacobian) and
the 1-D spectra are its reductions.

**GENE-3D** — x *and* y are real space, so both spectra come from an FFT and
there is no Hermitian factor of two anywhere: the stored real-space y direction
already carries both signs of ky. The kx spectrum uses the full 3-D Jacobian;
the ky spectrum uses its y-average, so the weight cannot itself mix binormal
modes.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from genetools.compat import trapz as _trapz
from genetools.diagnostics._base import RunDiagnostic
from genetools.diagnostics.spectra import Spectra
from genetools.diagnostics import _gene3d as g3
from genetools import _xr

_FIELD_NAMES = ["phi", "apar", "bpar"]
_MOM_NAMES = ["dens", "T_par", "T_perp", "q_par", "q_perp", "u_par"]

#: Moments given spectra by default for GENE-3D — the primitive fluctuating
#: quantities. Its flux moments (``Gamma_*``, ``Q_*``) are products of these, so
#: their amplitude spectra add little the flux spectra do not already show.
MOM_NAMES_3D = ("n", "u_par", "T_par", "T_per")


class AmplitudeSpectra(RunDiagnostic):
    """
    Time-averaged amplitude spectra of fields and moments.

    Parameters
    ----------
    run : genetools.run.Run
    moments : sequence of str or 'all', optional
        GENE-3D only: which moments to include (default :data:`MOM_NAMES_3D`).
        ``'all'`` takes every moment in the file.
    x_avg_lims : (float, float), optional
        GENE-3D only: radial window in ``x/a`` for the ky spectrum. Defaults to
        trimming *buffer_frac* from each end, keeping the Krook buffer regions
        out of the average.
    buffer_frac : float
        GENE-3D only; fraction trimmed from each radial end.
    """

    name = "amplitude"

    def __init__(self, run, moments=MOM_NAMES_3D, x_avg_lims=None,
                 buffer_frac=0.1):
        super().__init__(run)
        self.moments = moments
        self.x_avg_lims = x_avg_lims
        self.buffer_frac = buffer_frac

    # ------------------------------------------------------------------
    # Spectral-y geometries (flux tube and x-global)
    # ------------------------------------------------------------------

    def _ky_weight(self):
        # Index-based Hermitian weighting (first mode 1, rest 2), matching the
        # Spectra convention — so single-ky linear runs (ky=[kymin], finite)
        # are weighted identically across the package.
        ky = np.asarray(self.coord["ky"])
        w = np.full(ky.size, 2.0)
        if w.size:
            w[0] = 1.0
        return w

    def _accumulate(self, reader, idx, names, J_norm, ky_weight, out, is_local):
        """Stream *reader* over *idx*, time-average |·|² spectra for *names*.

        For local runs the radial axis is spectral (kx-folded). For global runs
        x is real space: the full |·|²(x, ky) map is computed (flux-surface
        averaged over z with the per-surface-normalised Jacobian ``J_norm`` of
        shape (nx, nz)), and the 1-D x/ky spectra are its reductions.
        """
        radial = "kx" if is_local else "x"
        axes = (radial, "ky") if is_local else (radial, "ky", "xky")
        per = {n: {ax: [] for ax in axes} for n in names}
        times = []
        for time, arrays in reader.stream_selected(list(idx)):
            times.append(time)
            for k, name in enumerate(names):
                amp = np.abs(arrays[k]) ** 2
                if is_local:
                    sp_r, sp_ky, _ = Spectra.averages(amp, J_norm, ky_weight)
                else:
                    xky = np.einsum("xkz,xz->xk", amp, J_norm) * ky_weight
                    per[name]["xky"].append(xky)
                    sp_r = xky.sum(axis=1)   # |·|² vs real x
                    sp_ky = xky.sum(axis=0)  # vs ky
                per[name][radial].append(sp_r)
                per[name]["ky"].append(sp_ky)
        times = np.asarray(times)
        for name in names:
            for ax in axes:
                stack = np.asarray(per[name][ax])
                if stack.shape[0] > 1:
                    avg = _trapz(stack, x=times, axis=0) / (times[-1] - times[0])
                else:
                    avg = stack[0]
                out[f"{name}_{ax}"] = avg

    def _compute_spectral(self, t):
        """kx/ky (flux tube) or x/ky (x-global) spectra."""
        run = self.run
        J = np.asarray(self.geom["Jacobian"])
        is_local = run.is_local
        if is_local:
            J_norm = J / np.sum(J)                      # J(z)
        else:
            J_norm = J / J.sum(axis=1, keepdims=True)   # J(x, z), per flux surface
        ky_weight = self._ky_weight()
        n_fields = int(self.params["info"]["n_fields"])

        out = {}
        _, idx = self._indices(run.field, t)
        field_names = [_FIELD_NAMES[i] for i in range(min(n_fields, 3))]
        self._accumulate(run.field, idx, field_names, J_norm, ky_weight, out,
                         is_local)

        for sp in run.species:
            rdr = run.mom(sp)
            _, idxm = self._indices(rdr, t)
            names = [f"{sp}_{m}" for m in _MOM_NAMES]
            self._accumulate(rdr, idxm, names, J_norm, ky_weight, out, is_local)
        return out

    # ------------------------------------------------------------------
    # GENE-3D
    # ------------------------------------------------------------------

    def _moment_names_3d(self, reader):
        if self.moments == "all":
            return list(reader.var_names)
        return [m for m in (self.moments or ()) if g3.has_var(reader, m)]

    def _accumulate_3d(self, reader, idx, names, prefix, J, J_yz, xsl, out):
        """
        Stream *reader*, time-averaging the kx and ky ``|f|^2`` spectra.

        Both directions are real space on disk, so both need an FFT. The kx
        spectrum is reduced over y and z with the full Jacobian; the ky spectrum
        over x and z with the y-averaged Jacobian, since a weight that varies in
        y would mix binormal modes.
        """
        slots = {n: reader.index_of(n) for n in names}
        acc_kx = {n: [] for n in names}
        acc_ky = {n: [] for n in names}
        times = []
        for time, arrays in reader.stream_selected(list(idx)):
            times.append(time)
            for n in names:
                var = arrays[slots[n]]
                pkx = np.abs(g3.to_kx(var)) ** 2
                acc_kx[n].append(np.average(pkx, weights=J, axis=(1, 2)))
                pky = np.abs(g3.to_ky(var)) ** 2
                acc_ky[n].append(np.average(pky[xsl], weights=J_yz[xsl],
                                            axis=(0, 2)))
        times = np.asarray(times)
        for n in names:
            out[f"{prefix}{n}_kx"] = self._time_average(
                np.asarray(acc_kx[n]), times)
            out[f"{prefix}{n}_ky"] = self._time_average(
                np.asarray(acc_ky[n]), times)

    def _compute_3d(self, t):
        """kx and ky spectra, both by FFT."""
        run = self.run
        J = np.asarray(self.geom["Jacobian"])
        J_yz = g3.jacobian_yz(J)
        xsl = g3.radial_slice(self.coord["x_o_a"], limits=self.x_avg_lims,
                              buffer_frac=self.buffer_frac)

        out = {}
        fld = run.field
        _, idx = self._indices(fld, t)
        self._accumulate_3d(fld, idx, list(fld.var_names), "", J, J_yz, xsl, out)

        for sp in run.species:
            rdr = run.mom(sp)
            names = self._moment_names_3d(rdr)
            if not names:
                continue
            _, idxm = self._indices(rdr, t)
            self._accumulate_3d(rdr, idxm, names, f"{sp}_", J, J_yz, xsl, out)

        out["_xslice"] = xsl
        return out

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(self, t=None):
        """Compute the time-averaged amplitude spectra; returns a dict."""
        key = self._key(t)
        if key not in self._cache:
            self._cache[key] = (self._compute_3d(t) if self.is_3d
                                else self._compute_spectral(t))
        return self._cache[key]

    def dataset(self, t=None):
        """Return the spectra as an :class:`xarray.Dataset`."""
        out = dict(self.compute(t))
        xsl = out.pop("_xslice", None)
        coord = self.coord
        data_vars, used = _xr.stacked_vars(out, self.run.species,
                                           _xr.dims_from_suffix)
        candidates = {
            "kx": np.asarray(coord.get("kx_2" if not self.is_3d else "kx",
                                       coord.get("kx", []))),
            "ky": np.asarray(coord.get("ky", [])),
            "x": np.asarray(coord.get("x", coord.get("x_o_a", []))),
        }
        ds = _xr.make_dataset(data_vars, candidates, species=used,
                              params=self.params)
        ds.attrs["geometry_kind"] = self.geometry_kind
        if xsl is not None:
            x = np.asarray(coord["x_o_a"], dtype=float)[xsl]
            ds.attrs["x_avg_range"] = [float(x[0]), float(x[-1])]
        return ds

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, t=None, **kw):
        """
        Plot the ky and radial amplitude spectra of every quantity (log-y).

        x-global runs additionally get the ``|f|^2(x, ky)`` field maps; GENE-3D
        gets one row per quantity with kx beside ky, since there is no 2-D map
        to show — both axes were transformed independently.
        """
        ds = self.dataset(t)
        if self.is_3d:
            return self._plot_3d(ds)
        return self._plot_spectral(ds)

    def _plot_spectral(self, ds):
        ky = ds["ky"].values if "ky" in ds.coords else None
        is_local = self.run.is_local
        rcoord = "kx" if is_local else "x"
        rax = ds[rcoord].values if rcoord in ds.coords else None

        maps = [] if is_local else [n for n in _FIELD_NAMES if f"{n}_xky" in ds]
        if maps:
            fig = plt.figure(figsize=(11, 8.5))
            gs = fig.add_gridspec(2, 1, hspace=0.35)
            gtop = gs[0].subgridspec(1, 2, wspace=0.3)
            axky, axr = (fig.add_subplot(g) for g in gtop)
            gbot = gs[1].subgridspec(1, len(maps), wspace=0.3)
            for name, g in zip(maps, gbot):
                _plot_map(fig.add_subplot(g), rax, ky, ds[f"{name}_xky"].values,
                          name)
        else:
            fig, (axky, axr) = plt.subplots(1, 2, figsize=(11, 4.5))

        for var in ds.data_vars:
            label, suffix = var.rsplit("_", 1)
            if suffix == "ky":
                _plot_spectrum(axky, ky, ds[var], label)
            elif suffix == rcoord:
                _plot_spectrum(axr, rax, ds[var], label)
        axky.set_xlabel(r"$k_y\rho$"); axky.set_ylabel("amplitude$^2$")
        axr.set_xlabel(r"$k_x\rho$" if is_local else r"$x/\rho_{\rm ref}$")
        axr.set_ylabel("amplitude$^2$")
        for ax in (axky, axr):
            ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3)
        axky.legend(fontsize=7, ncol=2)
        if not maps:
            fig.tight_layout()
        plt.show()
        return fig

    def _plot_3d(self, ds):
        bases = sorted({n.rsplit("_", 1)[0] for n in ds.data_vars})
        fig, axes = plt.subplots(len(bases), 2,
                                 figsize=(10, 2.9 * len(bases)), squeeze=False)
        for row, base in enumerate(bases):
            for col, axis in enumerate(("kx", "ky")):
                ax = axes[row][col]
                name = f"{base}_{axis}"
                if name not in ds:
                    ax.set_visible(False)
                    continue
                k_full = np.asarray(ds[axis])
                # Only the non-negative half is independent information.
                n_pos = (k_full.size + 1) // 2
                k = k_full[:n_pos]
                da = ds[name]
                if "species" in da.dims:
                    for sp in ds["species"].values:
                        ax.loglog(k[1:], np.asarray(da.sel(species=sp))[1:n_pos],
                                  label=str(sp))
                    ax.legend(fontsize=7)
                else:
                    ax.loglog(k[1:], np.asarray(da)[1:n_pos])
                ax.set_xlabel(rf"$k_{axis[1]} \rho_{{\rm ref}}$")
                ax.set_ylabel(f"$|{base}|^2$")
                ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        plt.show()
        return fig


def _plot_map(ax, x, ky, arr, name):
    """Pcolormesh of a |·|²(x, ky) map with a log colour scale."""
    from matplotlib.colors import LogNorm
    x = x if x is not None and len(x) == arr.shape[0] else np.arange(arr.shape[0])
    ky = ky if ky is not None and len(ky) == arr.shape[1] else np.arange(arr.shape[1])
    positive = arr[arr > 0]
    norm = LogNorm(positive.min(), positive.max()) if positive.size else None
    pc = ax.pcolormesh(x, ky, arr.T, norm=norm, cmap="viridis", shading="nearest")
    ax.figure.colorbar(pc, ax=ax, pad=0.02)
    ax.set_xlabel(r"$x/\rho_{\rm ref}$"); ax.set_ylabel(r"$k_y\rho$")
    tex = {"phi": r"\phi", "apar": r"A_\parallel", "bpar": r"B_\parallel"}
    ax.set_title(rf"$|{tex.get(name, name)}|^2(x, k_y)$")


def _plot_spectrum(ax, axis, da, label):
    """Plot a (possibly species-stacked) 1-D spectrum on *ax*."""
    if "species" in da.dims:
        for sp in da["species"].values:
            y = np.abs(da.sel(species=sp).values)
            x = axis if axis is not None and len(axis) == len(y) else np.arange(len(y))
            ax.plot(x, y, label=f"{label}:{sp}")
    else:
        y = np.abs(da.values)
        x = axis if axis is not None and len(axis) == len(y) else np.arange(len(y))
        ax.plot(x, y, label=label)
