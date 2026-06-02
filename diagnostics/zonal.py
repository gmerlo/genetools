"""
zonal.py — zonal-flow (ky=0) potential as an x-t contour.

Extracts the flux-surface-averaged ky=0 component of the electrostatic potential
over time, giving φ_zonal(x, t) (and the associated ExB shearing rate
∂²-structure). Reuses :func:`genetools.diagnostics.shearingrate.compute_exb` for
the per-time zonal extraction (IFFT over kx for local runs, real-space FSA for
global runs), so it is consistent with :class:`~genetools.diagnostics.shearingrate.ShearingRate`
— this diagnostic focuses on the *potential* x-t evolution rather than the RMS
shearing rate.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from genetools.compat import trapz as _trapz
from genetools.diagnostics.shearingrate import compute_exb


class Zonal:
    """Zonal (ky=0) potential x-t evolution."""

    def __init__(self, run):
        self.run = run
        self._cache = None

    # ------------------------------------------------------------------

    def compute(self, t=None):
        """Stream the field and build φ_zonal(x, t); returns a dict."""
        run = self.run
        coord = run.coords[0]
        geom = run.geometry[0]
        p = run.params.get(0)

        _, idx = run._indices(run.field, t)
        times, phiz, omeg = [], [], []
        for time, arrays in run.field.stream_selected(list(idx)):
            res = compute_exb(arrays[0], p, geom, coord)
            times.append(time)
            phiz.append(res["phi_zonal_x"])
            omeg.append(res["omega_ExB"])

        x = np.asarray(coord["x"])
        if x.size == 0:
            x = np.asarray(coord["x_o_a"])

        self._cache = {
            "time": np.asarray(times),
            "x": x,
            "phi_zonal": np.asarray(phiz),     # (nt, nx)
            "omega_ExB": np.asarray(omeg),     # (nt, nx)
        }
        return self._cache

    # ------------------------------------------------------------------

    @property
    def data(self):
        """Return an ``xarray.Dataset`` with φ_zonal(time, x) and ω_ExB(time, x)."""
        import xarray as xr
        c = self.compute() if self._cache is None else self._cache
        ds = xr.Dataset(
            {
                "phi_zonal": (("time", "x"), c["phi_zonal"]),
                "omega_ExB": (("time", "x"), c["omega_ExB"]),
            },
            coords={"time": c["time"], "x": c["x"]},
        )
        return ds

    def plot(self, t=None, **kw):
        """Plot the φ_zonal x-t contour and the time-averaged radial profile."""
        if t is not None or self._cache is None:
            self.compute(t)
        c = self._cache
        time, x, phiz = c["time"], c["x"], c["phi_zonal"]

        fig, (ax_xt, ax_prof) = plt.subplots(
            1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]})

        vmax = np.max(np.abs(phiz)) or 1.0
        pcm = ax_xt.pcolormesh(time, x, phiz.T, cmap="bwr",
                               vmin=-vmax, vmax=vmax, shading="auto")
        ax_xt.set_xlabel(r"$t\;c_{\rm ref}/L_{\rm ref}$")
        ax_xt.set_ylabel(r"$x/\rho_{\rm ref}$")
        ax_xt.set_title(r"$\langle\phi\rangle_{\rm zonal}(x,t)$")
        fig.colorbar(pcm, ax=ax_xt)

        if time.size > 1:
            prof = _trapz(phiz, x=time, axis=0) / (time[-1] - time[0])
        else:
            prof = phiz[0]
        ax_prof.plot(prof, x)
        ax_prof.set_xlabel(r"$\langle\phi\rangle_{\rm zonal}$ (t-avg)")
        ax_prof.set_ylabel(r"$x/\rho_{\rm ref}$")
        ax_prof.grid(True)

        fig.tight_layout()
        plt.show()
        return fig
