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

    def _accumulate(self, reader, idx, names, J_norm, ky_weight, out):
        """Stream *reader* over *idx*, time-average |·|² spectra for *names*."""
        per = {n: {"kx": [], "ky": []} for n in names}
        times = []
        for time, arrays in reader.stream_selected(list(idx)):
            times.append(time)
            for k, name in enumerate(names):
                amp = np.abs(arrays[k]) ** 2
                sp_kx, sp_ky, _ = Spectra.averages(amp, J_norm, ky_weight)
                per[name]["kx"].append(sp_kx)
                per[name]["ky"].append(sp_ky)
        times = np.asarray(times)
        for name in names:
            for ax in ("kx", "ky"):
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

        out = {}
        _, idx = run._indices(run.field, t)
        if idx.size:
            field_names = [_FIELD_NAMES[i] for i in range(min(n_fields, 3))]
            self._accumulate(run.field, idx, field_names, J_norm, ky_weight, out)

        for sp in run.species:
            rdr = run.mom(sp)
            _, idxm = run._indices(rdr, t)
            if idxm.size == 0:
                continue
            names = [f"{sp}_{m}" for m in _MOM_NAMES]
            self._accumulate(rdr, idxm, names, J_norm, ky_weight, out)

        self._cache = out
        return out

    # ------------------------------------------------------------------

    @property
    def data(self):
        """Return an ``xarray.Dataset`` of kx/ky amplitude spectra."""
        out = self.compute() if self._cache is None else self._cache
        coord = dict(self.run.coords[0])
        kx2 = np.asarray(coord.get("kx_2", []))
        if kx2.size:
            coord["kx"] = kx2          # kx spectra are over |kx| (nx//2+1)
        bases = set()
        for key in out:
            _, var = _xr._split_species(key, self.run.species)
            bases.add(var)
        dims = {v: ("kx",) if v.endswith("_kx") else ("ky",) for v in bases}
        return _xr.build_dataset(out, coords=coord, params=self.run.params.get(0),
                                 species=self.run.species, dims=dims)

    def plot(self, t=None, **kw):
        """Plot ky and kx amplitude spectra (log-y) of all quantities."""
        if t is not None or self._cache is None:
            self.compute(t)
        ds = self.data
        ky = ds["ky"].values if "ky" in ds.coords else None
        kx = ds["kx"].values if "kx" in ds.coords else None

        fig, (axky, axkx) = plt.subplots(1, 2, figsize=(11, 4.5))
        for var in ds.data_vars:
            da = ds[var]
            label = var.rsplit("_", 1)[0]
            if var.endswith("_ky"):
                _plot_spectrum(axky, ky, da, label)
            elif var.endswith("_kx"):
                _plot_spectrum(axkx, kx, da, label)
        axky.set_xlabel(r"$k_y\rho$"); axky.set_ylabel("amplitude$^2$")
        axkx.set_xlabel(r"$k_x\rho$"); axkx.set_ylabel("amplitude$^2$")
        for ax in (axky, axkx):
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
