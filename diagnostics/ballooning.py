"""
ballooning.py — field-line (ballooning) mode structure for local GENE runs.

For a chosen ``ky`` mode, the kx modes are connected along the extended
ballooning angle χ using GENE's parallel boundary phase factor, giving the
mode structure φ(χ), A∥(χ), B∥(χ). Ported from GENE pydiag's
``ModeStructure`` (``plot_ball.py``) but Run/xarray-native.

By default the last available time is used for linear runs and the modulus
time-average over the window for nonlinear runs; pass ``t=(start, stop)`` to
restrict the window.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from genetools.compat import trapz as _trapz

_FIELD_NAMES = ["phi", "apar", "bpar"]


class Ballooning:
    """Ballooning mode structure for a single ``ky`` of the field data."""

    def __init__(self, run, ky=None, kyind=None, normalize=True, t=None):
        if not run.is_local:
            raise NotImplementedError(
                "Ballooning is only defined for local (flux-tube) runs.")
        self.run = run
        self.normalize = normalize

        coord = run.coords[0]
        ky_arr = np.asarray(coord["ky"])
        if kyind is None:
            if ky is not None:
                kyind = int(np.argmin(np.abs(ky_arr - ky)))
            else:
                nonlinear = run.params.get(0)["general"].get("nonlinear", False)
                kyind = 1 if (nonlinear and ky_arr.size > 1) else 0
        self.kyind = int(kyind)
        self.kyval = float(ky_arr[self.kyind])

        self.chi = None          # extended ballooning angle / pi
        self.amps = {}           # {field_name: complex ndarray over chi}
        self._compute(t)

    # ------------------------------------------------------------------

    def _connection(self):
        """Return (nexcj, nconn, phasefac) for the selected ky."""
        p = self.run.params.get(0)
        box, gp = p["box"], p["geometry"]
        geom = self.run.geometry[0]

        nx0 = int(box["nx0"])
        kymin = float(box["kymin"])
        lx = float(box.get("lx", 0.0) or 0.0)
        n_pol = int(gp.get("n_pol", 1) or 1)
        n0_global = float(box.get("n0_global", 0) or 0)
        x0 = float(box.get("x0", 1.0) or 1.0)
        adapt_lx = bool(box.get("adapt_lx", gp.get("adapt_lx", False)))
        magn = str(gp.get("magn_geometry", "")).strip().strip("'")

        shat = geom.get("local", {}).get("shat") or gp.get("shat") or 0.0
        q0 = geom.get("local", {}).get("q0") or gp.get("q0") or 0.0
        Cy = geom.get("metric", {}).get("C_y", 1.0)
        sign_Ip = gp.get("sign_Ip_CW", 1)
        sign_Bt = gp.get("sign_Bt_CW", 1)

        if self.kyval == 0.0 or kymin == 0.0:
            raise ValueError("Ballooning structure is undefined for ky = 0.")
        jglob = int(round(self.kyval / kymin))  # true global ky multiple

        Cyq0_x0 = 1.0 if magn in ("s_alpha", "circular") else (
            float(Cy) * float(q0) / x0 if x0 else 1.0)
        sign_shear = -1 if shat < 0 else 1
        nexc_sign = sign_shear * sign_Ip * sign_Bt
        if adapt_lx:
            nexc = 1
        else:
            nexc = int(np.round(lx * n_pol * abs(shat) * kymin * abs(Cyq0_x0)))
            nexc = nexc or 1
        nexc *= nexc_sign
        nexcj = nexc * jglob
        if nexcj == 0:
            nexcj = 1
        nconn = int(((nx0 - 1) // 2) // abs(nexcj)) * 2 + 1
        phasefac = ((-1) ** nexcj) * np.exp(2.0j * np.pi * n0_global * q0 * jglob)
        return nexcj, nconn, phasefac

    def _compute(self, t):
        run = self.run
        p = run.params.get(0)
        nz0 = int(p["box"]["nz0"])
        n_fields = int(p["info"]["n_fields"])
        z = np.asarray(run.coords[0]["z"])
        nonlinear = p["general"].get("nonlinear", False)

        nexcj, nconn, phasefac = self._connection()
        half = nconn // 2
        ncinds = range(-half, half + 1)
        self.chi = np.concatenate([z + nc * 2.0 * np.pi for nc in ncinds]) / np.pi

        fields = [n for i, n in enumerate(_FIELD_NAMES) if i < n_fields]

        _, idx = run._indices(run.field, t)
        if idx.size == 0:
            raise ValueError("No field time steps in the requested window.")
        if nonlinear:
            sel = idx
        else:
            sel = idx[-1:]            # linear: use the last available time

        # Accumulate the (possibly time-resolved) ballooning structure.
        series = {name: [] for name in fields}
        times = []
        for time, arrays in run.field.stream_selected(list(sel)):
            times.append(time)
            for fi, name in enumerate(fields):
                fld = arrays[fi]                          # (nx, nky, nz)
                balloon = np.concatenate([
                    np.conj(phasefac) ** nc * fld[nc * nexcj, self.kyind, :]
                    for nc in ncinds
                ])
                if self.normalize:
                    normval = fld[0, self.kyind, nz0 // 2]
                    if normval != 0:
                        balloon = balloon * np.exp(-1j * np.angle(normval)) \
                            / np.abs(normval)
                series[name].append(balloon)

        times = np.asarray(times)
        for name in fields:
            stack = np.asarray(series[name])
            if stack.shape[0] > 1:
                self.amps[name] = _trapz(stack, x=times, axis=0) \
                    / (times[-1] - times[0])
            else:
                self.amps[name] = stack[0]

    # ------------------------------------------------------------------

    @property
    def data(self):
        """Return an ``xarray.Dataset`` of the ballooning structure."""
        import xarray as xr
        ds = xr.Dataset(
            {name: ("chi", self.amps[name]) for name in self.amps},
            coords={"chi": self.chi},
        )
        ds.attrs["ky"] = self.kyval
        ds.attrs["kyind"] = self.kyind
        return ds

    def plot(self, t=None, **kw):
        """Plot |·|, Re, Im of each field versus the ballooning angle χ/π."""
        if t is not None:
            self._compute(t)
        fields = list(self.amps)
        fig, axes = plt.subplots(1, len(fields),
                                 figsize=(5 * len(fields), 4), squeeze=False)
        labels = {"phi": r"\phi", "apar": r"A_\parallel", "bpar": r"B_\parallel"}
        for ax, name in zip(axes[0], fields):
            arr = self.amps[name]
            lbl = labels.get(name, name)
            ax.plot(self.chi, np.abs(arr), "k", label=fr"$|{lbl}|$")
            ax.plot(self.chi, arr.real, "b", lw=1, label=fr"$\Re({lbl})$")
            ax.plot(self.chi, arr.imag, "r", lw=1, label=fr"$\Im({lbl})$")
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_xlabel(r"$\chi/\pi$")
            ax.set_title(fr"${lbl}\;(k_y\rho={self.kyval:.3f})$")
            ax.legend(fontsize=8)
        fig.tight_layout()
        plt.show()
        return fig
