# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
amplitude.py — kx/ky amplitude spectra of fields and moments.

Computes the time-averaged |·|² spectrum of each field (φ, A∥, B∥) and the
leading moments (n, T∥, T⊥, q∥, q⊥, u∥) per species, resolved in kx and ky.
Complements :class:`~genetools.diagnostics.spectra.Spectra` (which gives *flux*
spectra) by reporting the amplitude content of every quantity. Uses the same
Hermitian ky-weighting (ky=0 unweighted, ky>0 weighted ×2) and Jacobian-weighted
z-average as the flux spectra.
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
        ky = np.asarray(self.run.coords[0]["ky"])
        w = np.full(ky.size, 2.0)
        w[ky == 0.0] = 1.0
        return w

    def _accumulate(self, reader, idx, names, J_norm, ky_weight, out, is_local):
        """Stream *reader* over *idx*, time-average |·|² spectra for *names*.

        For local runs the radial axis is spectral (kx-folded); for global runs
        x is real space, so a kx spectrum is meaningless — a Jacobian-weighted
        radial profile of |·|² is reported instead.
        """
        radial = "kx" if is_local else "x"
        per = {n: {radial: [], "ky": []} for n in names}
        times = []
        for time, arrays in reader.stream_selected(list(idx)):
            times.append(time)
            for k, name in enumerate(names):
                amp = np.abs(arrays[k]) ** 2
                if is_local:
                    sp_r, sp_ky, _ = Spectra.averages(amp, J_norm, ky_weight)
                else:
                    W = ky_weight[np.newaxis, :, np.newaxis]
                    J = J_norm[np.newaxis, np.newaxis, :]
                    weighted = amp * (W * J)
                    sp_r = np.sum(weighted, axis=(1, 2))   # |·|² vs real x
                    sp_ky = np.sum(weighted, axis=(0, 2))  # vs ky
                per[name][radial].append(sp_r)
                per[name]["ky"].append(sp_ky)
        times = np.asarray(times)
        for name in names:
            for ax in (radial, "ky"):
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
        J_norm = J / np.sum(J)
        ky_weight = self._ky_weight()
        n_fields = int(run.params.get(0)["info"]["n_fields"])

        is_local = run.is_local
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
        """Return an ``xarray.Dataset`` of kx/ky amplitude spectra."""
        out = self.compute() if self._cache is None else self._cache
        coord = self.run.coords[0]
        data_vars, used = _xr.stacked_vars(
            out, self.run.species, lambda var: (var.rsplit("_", 1)[1],))
        candidates = {
            "kx": np.asarray(coord.get("kx_2", coord.get("kx", []))),
            "ky": np.asarray(coord.get("ky", [])),
            "x": np.asarray(coord.get("x", coord.get("x_o_a", []))),
        }
        return _xr.make_dataset(data_vars, candidates, species=used,
                                params=self.run.params.get(0))

    def plot(self, t=None, **kw):
        """Plot ky and kx amplitude spectra (log-y) of all quantities."""
        if t is not None or self._cache is None:
            self.compute(t)
        ds = self.data
        ky = ds["ky"].values if "ky" in ds.coords else None
        is_local = self.run.is_local
        rcoord = "kx" if is_local else "x"
        rsuffix = "_kx" if is_local else "_x"
        rax = ds[rcoord].values if rcoord in ds.coords else None

        fig, (axky, axr) = plt.subplots(1, 2, figsize=(11, 4.5))
        for var in ds.data_vars:
            da = ds[var]
            label = var.rsplit("_", 1)[0]
            if var.endswith("_ky"):
                _plot_spectrum(axky, ky, da, label)
            elif var.endswith(rsuffix):
                _plot_spectrum(axr, rax, da, label)
        axky.set_xlabel(r"$k_y\rho$"); axky.set_ylabel("amplitude$^2$")
        axr.set_xlabel(r"$k_x\rho$" if is_local else r"$x/\rho_{\rm ref}$")
        axr.set_ylabel("amplitude$^2$")
        for ax in (axky, axr):
            ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3)
        axky.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        plt.show()
        return fig


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
