# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
from genetools.diagnostics._base import RunDiagnostic
from genetools.diagnostics.shearingrate import compute_exb


class Zonal(RunDiagnostic):
    """
    Zonal potential x-t evolution.

    For the spectral geometries the zonal component is the ``ky = 0`` mode, via
    :func:`~genetools.diagnostics.shearingrate.compute_exb`. GENE-3D is real
    space in y, so its zonal component is a Jacobian-weighted average over y and
    z; that path reuses
    :class:`~genetools.diagnostics.shearingrate.ShearingRate` rather than
    repeating the reduction.
    """

    name = "zonal"

    # ------------------------------------------------------------------

    def compute(self, t=None):
        """Stream the field and build φ_zonal(x, t); returns a dict."""
        key = self._key(t)
        if key in self._cache:
            return self._cache[key]
        result = self._compute_3d(t) if self.is_3d else self._compute_spectral(t)
        self._cache[key] = result
        return result

    def _compute_3d(self, t):
        """Delegate to ShearingRate, which owns the GENE-3D reduction."""
        from genetools.diagnostics.shearingrate import ShearingRate
        raw = ShearingRate(self.run).compute(t)
        return {
            "time": raw["times"],
            "x": raw["x"],
            "phi_zonal": raw["phi_zonal"],
            "omega_ExB": raw["omega_exb"],
        }

    def _compute_spectral(self, t):
        run = self.run
        coord = self.coord
        geom = self.geom
        p = self.params

        _, idx = self._indices(run.field, t)
        times, phiz, omeg = [], [], []
        for time, arrays in run.field.stream_selected(list(idx)):
            res = compute_exb(arrays[0], p, geom, coord)
            times.append(time)
            phiz.append(res["phi_zonal_x"])
            omeg.append(res["omega_ExB"])

        x = np.asarray(coord["x"])
        if x.size == 0:
            x = np.asarray(coord["x_o_a"])

        return {
            "time": np.asarray(times),
            "x": x,
            "phi_zonal": np.asarray(phiz),     # (nt, nx)
            "omega_ExB": np.asarray(omeg),     # (nt, nx)
        }

    # ------------------------------------------------------------------

    def dataset(self, t=None):
        """Return an ``xarray.Dataset`` with φ_zonal(time, x) and ω_ExB(time, x)."""
        import xarray as xr
        c = self.compute(t)
        ds = xr.Dataset(
            {
                "phi_zonal": (("time", "x"), c["phi_zonal"]),
                "omega_ExB": (("time", "x"), c["omega_ExB"]),
            },
            coords={"time": c["time"], "x": c["x"]},
        )
        ds.attrs["geometry_kind"] = self.geometry_kind
        return ds

    def plot(self, t=None, **kw):
        """Plot the φ_zonal x-t contour and the time-averaged radial profile."""
        c = self.compute(t)
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
