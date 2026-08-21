# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
vexmax.py — origin of the nonlinear-CFL velocity ve_max(2) (x-global runs).

GENE's nonlinear timestep estimate is dominated by
``ve_max(2) = max |∂x chibar| / C_xy`` (the y-advection velocity), multiplied by
kjmax in ``adapt_dt``.  This diagnostic locates where that maximum originates
from saved field data, using phi as a proxy for the gyroaveraged chibar
(J0 ≈ 1 for electrons at the relevant low ky; check the A_par contribution
separately with ``var="apar"``):

  1. which ky   — per-ky column contributions and a cumulative ky<=K curve of
                  the reconstructed real-space maximum,
  2. which z    — per-z maxima,
  3. which x    — per-x profile with the Krook-buffer regions flagged,
  4. radial smoothing? — kx spectrum of ∂x phi at the dominant location: if the
                  energy sits near the grid scale, radial smoothing (hyp_x /
                  stencil) would reduce ve_max; if it is radially smooth, it
                  will not, and the amplitude is a physics/model question.

The reconstructed real-space maximum is directly comparable with the
``nl_cfl: ... ve_max(2)`` value printed by the (instrumented) code, which
validates the phi-for-chibar proxy.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ddx_4th", "y_reconstruct", "analyze_snapshot", "VexMax"]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def ddx_4th(f: np.ndarray, dx: float, axis: int = 0) -> np.ndarray:
    """4th-order centered d/dx (GENE interior stencil [1,-8,0,8,-1]/12dx);
    lower-order one-sided formulas at the two points near each edge."""
    fm = np.moveaxis(np.asarray(f), axis, 0)
    dfm = np.zeros_like(fm)
    n = fm.shape[0]
    if n < 5:
        if n >= 2:
            dfm[:] = np.gradient(fm, dx, axis=0)
        return np.moveaxis(dfm, 0, axis)
    coeff = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0 * dx)
    for s, w in zip(range(-2, 3), coeff):
        if w != 0.0:
            dfm[2:n - 2] += w * fm[2 + s:n - 2 + s]
    dfm[0] = (fm[1] - fm[0]) / dx
    dfm[1] = (fm[2] - fm[0]) / (2 * dx)
    dfm[n - 2] = (fm[n - 1] - fm[n - 3]) / (2 * dx)
    dfm[n - 1] = (fm[n - 1] - fm[n - 2]) / dx
    return np.moveaxis(dfm, 0, axis)


def y_reconstruct(vhat: np.ndarray, ny: int) -> np.ndarray:
    """Real-space field from GENE's half ky spectrum (ky index on the LAST axis):
    ``v(y) = Re(vhat_0) + 2 sum_{j>0} Re(vhat_j e^{i ky_j y})`` on ``ny`` points.

    numpy convention: ``irfft(a, n) * n = a_0 + 2 sum_k Re(a_k e^{2 pi i k m/n})``
    for k < n/2, which matches GENE's Fourier representation on a uniform ky grid.
    """
    vhat = np.asarray(vhat)
    nky = vhat.shape[-1]
    if ny < 2 * nky:
        raise ValueError(f"ny={ny} too small for nky={nky}")
    spec = np.zeros(vhat.shape[:-1] + (ny // 2 + 1,), dtype=complex)
    spec[..., :nky] = vhat
    return np.fft.irfft(spec, n=ny, axis=-1) * ny


def analyze_snapshot(field3d: np.ndarray, coord: dict, geom: dict, params: dict,
                     klist=None) -> dict:
    """Full ve_max(2)-origin decomposition of one (nx, nky, nz) field snapshot.

    Returns a dict of numpy arrays / scalars (see keys below); xarray assembly
    is done by :class:`VexMax`.
    """
    box = params["box"]
    nx, nky, nz = box["nx0"], box["nky0"], box["nz0"]
    if field3d.shape != (nx, nky, nz):
        raise ValueError(f"field shape {field3d.shape} != {(nx, nky, nz)}")
    dx = coord["dx"]
    ky = np.asarray(coord["ky"])

    # --- C_xy(x): array, scalar, or absent ---
    cxy = (geom.get("metric") or {}).get("C_xy", 1.0)
    cxy = np.asarray(cxy, dtype=float)
    if cxy.ndim == 0:
        cxy = np.full(nx, float(cxy))
    elif cxy.size != nx:
        cxy = np.full(nx, float(np.ravel(cxy)[0]))

    # v(x, ky, z) = d(field)/dx / C_xy  — the ky-decomposed y-advection velocity
    v = ddx_4th(field3d, dx, axis=0) / cxy[:, None, None]

    ny = 1 << int(np.ceil(np.log2(max(3 * nky, 8))))
    vt = np.moveaxis(v, 1, -1)                       # (nx, nz, nky)

    # full reconstruction: |v|(x, z) maximised over real-space y
    vmax_xz = np.max(np.abs(y_reconstruct(vt, ny)), axis=-1)
    vemax = float(vmax_xz.max())
    ix, iz = (int(i) for i in np.unravel_index(int(np.argmax(vmax_xz)),
                                               vmax_xz.shape))

    # 1) which ky
    colmax = np.max(np.abs(v), axis=(0, 2))          # per-ky max over (x, z)
    contrib_ky = np.where(np.arange(nky) == 0, 1.0, 2.0) * colmax
    if klist is None:
        klist = sorted({0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128,
                        192, 256, 384, 512, 1024, 2048, nky - 1})
    klist = sorted({min(int(k), nky - 1) for k in klist})
    vemax_cum = np.array([
        float(np.max(np.abs(y_reconstruct(vt[..., :K + 1], ny))))
        for K in klist])

    # 2) which z / 3) which x
    vmax_z = vmax_xz.max(axis=0)
    vmax_x = vmax_xz.max(axis=1)

    # buffers
    nlx = params.get("nonlocal_x", {}) or {}
    lbuf = float(nlx.get("l_buffer_size", 0.0) or 0.0)
    ubuf = float(nlx.get("u_buffer_size", 0.0) or 0.0)
    xfrac = ix / max(nx - 1, 1)
    in_buffer = bool(xfrac < lbuf or xfrac > 1.0 - ubuf)

    # 4) radial structure at the dominant ky column
    jstar = int(np.argmax(colmax))
    prof = v[:, jstar, iz]
    spec = np.abs(np.fft.rfft(prof * np.hanning(nx))) ** 2
    kxn = np.fft.rfftfreq(nx) * 2.0                  # 0..1 in units of kx_Nyquist
    tot = float(spec.sum()) or 1.0
    frac_hi = float(spec[kxn > 0.5].sum() / tot)
    frac_vhi = float(spec[kxn > 0.8].sum() / tot)

    return dict(
        vemax=vemax, ix=ix, iz=iz, xfrac=xfrac, in_buffer=in_buffer,
        lbuf=lbuf, ubuf=ubuf,
        ky=ky, contrib_ky=contrib_ky, jstar=jstar,
        klist=np.asarray(klist), vemax_cum=vemax_cum,
        vmax_x=vmax_x, vmax_z=vmax_z, vmax_xz=vmax_xz,
        kx_nyq_frac=kxn, spec_kx=spec / tot,
        frac_kx_gt_half_nyq=frac_hi, frac_kx_gt_08_nyq=frac_vhi,
        smoothing_helps=bool(frac_vhi > 0.2),
    )


# ---------------------------------------------------------------------------
# Run-native diagnostic
# ---------------------------------------------------------------------------

class VexMax:
    """Origin of the nonlinear-CFL velocity ve_max(2) (x-global runs).

    ``run.vexmax.data`` — xarray Dataset with the decomposition at the last
    time of the window (plus the ve_max(t) trace); ``run.vexmax.plot()`` — the
    four-panel summary (which ky / which x / which z / radial spectrum).
    """

    def __init__(self, run):
        self.run = run
        self._cache = None
        self._cache_key = None

    # ------------------------------------------------------------------

    def compute(self, t=None, var: str = "phi", klist=None, trace: bool = True):
        run = self.run
        p = run.params.get(0)
        if p["general"].get("x_local", True):
            raise NotImplementedError(
                "vexmax targets x-global runs (x_local = F); for flux-tube "
                "runs the mean flow is handled by the ExB remap and this "
                "decomposition is not meaningful.")
        coord = run.coords[0]
        geom = run.geometry[0]
        varidx = {"phi": 0, "apar": 1, "bpar": 2}[var]
        if varidx >= int(p["info"]["n_fields"]):
            raise ValueError(f"field '{var}' not present in this run")

        times, idx = run._indices(run.field, t)
        if idx.size == 0:
            raise ValueError("vexmax: no field time steps in the window.")

        trace_t, trace_v = [], []
        last = None
        for time, arrays in run.field.stream_selected(list(idx)):
            res = analyze_snapshot(arrays[varidx], coord, geom, p, klist=klist)
            last = (time, res)
            trace_t.append(time)
            trace_v.append(res["vemax"])
            if not trace and len(trace_t) < idx.size:
                # only the last snapshot requested: skip ahead cheaply
                continue

        tlast, res = last
        res["time"] = float(tlast)
        res["var"] = var
        res["trace_time"] = np.asarray(trace_t)
        res["trace_vemax"] = np.asarray(trace_v)
        res["x"] = np.asarray(coord["x"])
        self._cache, self._cache_key = res, (t, var)
        return res

    # ------------------------------------------------------------------

    @property
    def data(self):
        import xarray as xr
        c = self._cache if self._cache is not None else self.compute()
        nz = c["vmax_z"].size
        ds = xr.Dataset(
            {
                "contrib_ky": (("ky",), c["contrib_ky"]),
                "vemax_cum": (("K",), c["vemax_cum"]),
                "vmax_x": (("x",), c["vmax_x"]),
                "vmax_z": (("z_idx",), c["vmax_z"]),
                "spec_kx": (("kx_nyq_frac",), c["spec_kx"]),
                "vemax_trace": (("time",), c["trace_vemax"]),
            },
            coords={
                "ky": c["ky"],
                "K": c["klist"],
                "x": c["x"],
                "z_idx": np.arange(nz),
                "kx_nyq_frac": c["kx_nyq_frac"],
                "time": c["trace_time"],
            },
            attrs={
                "vemax": c["vemax"],
                "analysis_time": c["time"],
                "var": c["var"],
                "ix": c["ix"], "iz": c["iz"], "x_over_lx": c["xfrac"],
                "max_in_buffer": int(c["in_buffer"]),
                "l_buffer_size": c["lbuf"], "u_buffer_size": c["ubuf"],
                "dominant_ky_index": c["jstar"],
                "frac_kx_gt_half_nyq": c["frac_kx_gt_half_nyq"],
                "frac_kx_gt_08_nyq": c["frac_kx_gt_08_nyq"],
                "radial_smoothing_would_help": int(c["smoothing_helps"]),
            },
        )
        return ds

    # ------------------------------------------------------------------

    def report(self):
        """Print the plain-text summary of the last computation."""
        c = self._cache if self._cache is not None else self.compute()
        ky = c["ky"]
        print(f"vexmax ({c['var']}) at t = {c['time']:g}:")
        print(f"  proxy ve_max(2)      = {c['vemax']:.4e}   "
              f"(compare with the code's nl_cfl ve_max(2))")
        print(f"  located at x/lx = {c['xfrac']:.3f} (index {c['ix']}), "
              f"z index {c['iz']}"
              + ("   ** INSIDE the Krook buffer **" if c["in_buffer"] else ""))
        order = np.argsort(c["contrib_ky"])[::-1]
        print("  dominant ky columns:")
        for j in order[:8]:
            print(f"    j={j:5d}  ky={ky[j]:9.4f}  contribution={c['contrib_ky'][j]:.4e}")
        print("  cumulative ve_max with ky-index <= K:")
        for K, vk in zip(c["klist"], c["vemax_cum"]):
            print(f"    K={K:5d}  ky<={ky[min(K, len(ky)-1)]:9.4f}  "
                  f"{vk:.4e}  ({vk / c['vemax']:6.1%})")
        print(f"  radial spectrum at dominant column: "
              f"{c['frac_kx_gt_half_nyq']:.1%} above 0.5 kx_Nyq, "
              f"{c['frac_kx_gt_08_nyq']:.1%} above 0.8 kx_Nyq")
        print("  -> " + ("grid-scale radial structure: radial smoothing WOULD "
                         "reduce ve_max" if c["smoothing_helps"] else
                         "radially smooth structure: hyp_x/smoothing will NOT "
                         "reduce ve_max"))

    # ------------------------------------------------------------------

    def plot(self, t=None, var="phi", save=None, show=True):
        import matplotlib.pyplot as plt
        if self._cache is None or self._cache_key != (t, var):
            self.compute(t=t, var=var)
        c = self._cache
        self.report()

        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        a = ax[0, 0]
        a.semilogy(c["ky"], np.maximum(c["contrib_ky"],
                                       c["contrib_ky"].max() * 1e-8))
        a.set_xlabel(r"$k_y \rho$"); a.set_ylabel("column contribution")
        a.set_title("which $k_y$")
        a2 = a.twinx()
        kyK = c["ky"][np.minimum(c["klist"], len(c["ky"]) - 1)]
        a2.plot(kyK, c["vemax_cum"] / c["vemax"], "r.-")
        a2.set_ylabel(r"cumulative $v_{max}(k_y\le K)/v_{max}$", color="r")
        a2.set_ylim(0, 1.05)

        a = ax[0, 1]
        xn = np.linspace(0.0, 1.0, c["vmax_x"].size)
        a.plot(xn, c["vmax_x"])
        for b0, b1 in ((0.0, c["lbuf"]), (1.0 - c["ubuf"], 1.0)):
            if b1 > b0:
                a.axvspan(b0, b1, alpha=0.2, color="red")
        a.axvline(c["xfrac"], ls="--", color="k", lw=0.8)
        a.set_xlabel("$x/l_x$"); a.set_title("which $x$ (buffers red)")

        a = ax[1, 0]
        a.plot(np.arange(c["vmax_z"].size), c["vmax_z"], "o-")
        a.axvline(c["iz"], ls="--", color="k", lw=0.8)
        a.set_xlabel("z index"); a.set_title("which $z$")

        a = ax[1, 1]
        a.semilogy(c["kx_nyq_frac"], np.maximum(c["spec_kx"], 1e-12))
        a.axvline(0.5, ls=":", color="gray"); a.axvline(0.8, ls=":", color="gray")
        a.set_xlabel(r"$k_x / k_{x,\mathrm{Nyq}}$")
        a.set_title(f"radial spectrum at dominant $k_y$ "
                    f"(j={c['dominant_ky_index'] if 'dominant_ky_index' in c else c['jstar']})")

        fig.suptitle(f"ve_max(2) origin — {c['var']}, t={c['time']:g}, "
                     f"proxy ve_max={c['vemax']:.3e}")
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=140)
        if show:
            plt.show()
        return fig
