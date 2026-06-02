"""
growthrate.py — linear growth rate and real frequency from the field file.

GENE derives the linear growth rate γ and real frequency ω from the time
evolution of the field: γ from how fast |φ| grows, ω from how fast its complex
phase rotates between outputs. This diagnostic does exactly that, fitting over a
trailing time window where a single linear mode dominates.

If an ``omega<ext>`` (or ``eigenvalues.dat``) file is present, its values are
attached as an optional cross-check; they are never required.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from genetools.io import omega as _omega


class GrowthRate:
    """Field-based linear growth rate γ(ky) and real frequency ω(ky)."""

    def __init__(self, run):
        self.run = run

    # ------------------------------------------------------------------

    def compute(self, t=None):
        """
        Return ``(ky, gamma, omega, window)`` from the field time evolution.

        With ``t=None`` the trailing half of the time series is used; otherwise
        the given ``(start, stop)`` window is used.
        """
        run = self.run
        times_all, idx = run._indices(run.field, t)
        if t is None:
            n = times_all.size
            idx = np.arange(n // 2, n)        # trailing half by default
        sel = list(idx)
        ky = np.asarray(run.coords[0]["ky"])

        if len(sel) < 2:
            nan = np.full(ky.size, np.nan)
            return ky, nan, nan, (None, None)

        phis, times = [], []
        for time, arrays in run.field.stream_selected(sel):
            phis.append(arrays[0])           # phi, shape (nx, nky, nz)
            times.append(time)
        phis = np.asarray(phis)              # (nt, nx, nky, nz)
        times = np.asarray(times)

        nky = phis.shape[2]
        mean_abs = np.mean(np.abs(phis), axis=0)   # (nx, nky, nz)
        gamma = np.full(nky, np.nan)
        omega = np.full(nky, np.nan)
        for j in range(nky):
            # Fixed reference location: peak time-averaged |phi| for this ky,
            # which avoids the phase cancellation of a coherent sum.
            ix, iz = np.unravel_index(np.argmax(mean_abs[:, j, :]),
                                      mean_abs[:, j, :].shape)
            amp = phis[:, ix, j, iz]
            good = np.abs(amp) > 0
            if good.sum() < 2:
                continue
            tt = times[good]
            gamma[j] = np.polyfit(tt, np.log(np.abs(amp[good])), 1)[0]
            omega[j] = -np.polyfit(tt, np.unwrap(np.angle(amp[good])), 1)[0]

        return ky, gamma, omega, (float(times[0]), float(times[-1]))

    # ------------------------------------------------------------------

    def _file_crosscheck(self):
        """Return omega-file values keyed by ext, or ``{}`` if none present."""
        out = {}
        for ext in self.run.extensions:
            data = _omega.read_omega(self.run._folder, ext)
            if data is not None:
                out[ext] = data
        return out

    @property
    def data(self):
        """Return an ``xarray.Dataset`` with ``gamma`` and ``omega`` over ``ky``."""
        import xarray as xr
        ky, gamma, omega, window = self.compute()
        ds = xr.Dataset(
            {"gamma": ("ky", gamma), "omega": ("ky", omega)},
            coords={"ky": ky},
        )
        ds.attrs["t_window"] = window
        cross = self._file_crosscheck()
        if cross:
            first = next(iter(cross.values()))
            ds.attrs["omega_file_ky"] = first["ky"]
            ds.attrs["omega_file_gamma"] = first["gamma"]
            ds.attrs["omega_file_omega"] = first["omega"]
        return ds

    def plot(self, t=None, **kw):
        """Plot γ(ky) and ω(ky) (or print a single linear mode)."""
        ky, gamma, omega, window = self.compute(t)
        cross = self._file_crosscheck()

        if ky.size == 1:
            print(f"Linear mode  ky={ky[0]:.4f}  (t in {window})")
            print(f"  field-based:  gamma={gamma[0]:.5g}  omega={omega[0]:.5g}")
            for ext, d in cross.items():
                print(f"  omega{ext} file: gamma={d['gamma'][0]:.5g}  "
                      f"omega={d['omega'][0]:.5g}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(ky, gamma, "o-", label="field-based")
        ax2.plot(ky, omega, "o-", label="field-based")
        for ext, d in cross.items():
            ax1.plot(d["ky"], d["gamma"], "x--", label=f"omega{ext}")
            ax2.plot(d["ky"], d["omega"], "x--", label=f"omega{ext}")
        ax1.set_xlabel(r"$k_y\rho$"); ax1.set_ylabel(r"$\gamma\,[c_{\rm ref}/L_{\rm ref}]$")
        ax2.set_xlabel(r"$k_y\rho$"); ax2.set_ylabel(r"$\omega\,[c_{\rm ref}/L_{\rm ref}]$")
        ax1.grid(True); ax2.grid(True)
        if cross:
            ax1.legend(fontsize=8); ax2.legend(fontsize=8)
        fig.tight_layout()
        plt.show()
        return fig
