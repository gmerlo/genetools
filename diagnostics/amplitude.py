# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
amplitude.py — kx/ky (local) and x/ky (global) amplitude spectra.

Computes the time-averaged |·|² spectrum of each field (φ, A∥, B∥) and the
leading moments (n, T∥, T⊥, q∥, q⊥, u∥) per species. For local runs the
spectra are resolved in kx and ky; for global runs (``x_local = F``) the full
2-D map |·|²(x, ky) is computed — the amplitude counterpart of
:class:`~genetools.diagnostics.spectra_global.SpectraGlobal` — plus its 1-D
reductions vs x and vs ky. Uses the same Hermitian ky-weighting (ky=0
unweighted, ky>0 weighted ×2) and Jacobian-weighted z-average as the flux
spectra (per flux surface for global geometry, where the Jacobian is (nx, nz)).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from genetools.compat import trapz as _trapz
from genetools.diagnostics.spectra import Spectra
from genetools import _xr

_FIELD_NAMES = ["phi", "apar", "bpar"]
_MOM_NAMES = ["dens", "T_par", "T_perp", "q_par", "q_perp", "u_par"]


class AmplitudeSpectra:
    """Time-averaged kx and ky amplitude spectra of fields and moments."""

    def __init__(self, run):
        self.run = run
        self._cache = None

    # ------------------------------------------------------------------

    def _ky_weight(self):
        # Index-based Hermitian weighting (first mode 1, rest 2), matching the
        # Spectra convention — so single-ky linear runs (ky=[kymin], finite)
        # are weighted identically across the package.
        ky = np.asarray(self.run.coords[0]["ky"])
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

    def compute(self, t=None):
        """Compute the time-averaged amplitude spectra; returns a dict."""
        run = self.run
        geom = run.geometry[0]
        J = np.asarray(geom["Jacobian"])
        is_local = run.is_local
        if is_local:
            J_norm = J / np.sum(J)                      # J(z)
        else:
            J_norm = J / J.sum(axis=1, keepdims=True)   # J(x, z), per flux surface
        ky_weight = self._ky_weight()
        n_fields = int(run.params.get(0)["info"]["n_fields"])

        out = {}
        _, idx = run._indices(run.field, t)
        if idx.size:
            field_names = [_FIELD_NAMES[i] for i in range(min(n_fields, 3))]
            self._accumulate(run.field, idx, field_names, J_norm, ky_weight, out,
                             is_local)

        for sp in run.species:
            rdr = run.mom(sp)
            _, idxm = run._indices(rdr, t)
            if idxm.size == 0:
                continue
            names = [f"{sp}_{m}" for m in _MOM_NAMES]
            self._accumulate(rdr, idxm, names, J_norm, ky_weight, out, is_local)

        self._cache = out
        return out

    # ------------------------------------------------------------------

    @property
    def data(self):
        """Return an ``xarray.Dataset`` of the amplitude spectra: kx/ky for
        local runs, x/ky maps (``*_xky``) plus 1-D reductions for global."""
        out = self.compute() if self._cache is None else self._cache
        coord = self.run.coords[0]
        data_vars, used = _xr.stacked_vars(out, self.run.species,
                                           _xr.dims_from_suffix)
        candidates = {
            "kx": np.asarray(coord.get("kx_2", coord.get("kx", []))),
            "ky": np.asarray(coord.get("ky", [])),
            "x": np.asarray(coord.get("x", coord.get("x_o_a", []))),
        }
        return _xr.make_dataset(data_vars, candidates, species=used,
                                params=self.run.params.get(0))

    def plot(self, t=None, **kw):
        """Plot ky and radial amplitude spectra (log-y) of all quantities;
        for global runs also the |·|²(x, ky) maps of the fields."""
        if t is not None or self._cache is None:
            self.compute(t)
        ds = self.data
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
