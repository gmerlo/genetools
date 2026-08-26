# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
gam.py — GENE-3D geodesic acoustic mode diagnostic.

A GAM test initialises a zonal perturbation and watches it ring down. What
matters is therefore the *history of a single mode*, normalised to its initial
value, rather than a spatial profile:

* ``phi_zonal`` — the ``(k_y = 0)`` potential, and its radial derivative, at the
  centre of the box.
* ``phi_kx1``  — the first radial harmonic of the zonal potential, which is what
  an initial-condition GAM test usually excites.

Each trace is divided by its value at the first output time, so a decaying
oscillation about zero is directly readable, and the frequency and damping rate
are fitted from it.

Averages over ``z`` are Jacobian-weighted; the flux-surface average of a zonal
quantity is a physical average over the surface, not over grid points. (The
reference GUI mixes the two, weighting some of these reductions and not others.)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from genetools._xr import make_dataset, unit_attrs
from genetools.diagnostics._base import RunDiagnostic
from genetools.diagnostics import _gene3d as g3


class Gam(RunDiagnostic):
    """
    Zonal-flow / GAM oscillation traces for a GENE-3D run.

    Parameters
    ----------
    run : genetools.run.Run
    x_index : int, optional
        Radial index for the point traces (default: the middle of the box).
    """

    name = "gam"
    supported = ("xy_global",)

    def __init__(self, run, x_index=None):
        super().__init__(run)
        self.x_index = x_index
        self._cache = {}

    # ------------------------------------------------------------------

    def compute(self, t=None):
        """Stream the field file and build the zonal traces."""
        key = tuple(t) if isinstance(t, (tuple, list)) else t
        if key in self._cache:
            return self._cache[key]

        run = self.run
        geom = run.geometry[0]
        J = geom["Jacobian"]
        C_xy = np.asarray(geom["metric"]["C_xy"], dtype=float)
        coord = run.coords[0]
        x = np.asarray(coord["x"], dtype=float)
        nx = x.size
        ix = self.x_index if self.x_index is not None else nx // 2

        reader = run.field
        _, idx = self._indices(reader, t)
        i_phi = reader.index_of("phi")

        # Weight for a z-average of an already y-averaged quantity.
        J_z = J.mean(axis=1)

        times, zonal, dzonal, kx1, amplitude = [], [], [], [], []
        for time, arrays in reader.stream_selected(idx):
            phi = arrays[i_phi]
            times.append(time)
            amplitude.append(float(np.max(np.abs(phi))))
            fs = g3.flux_surface_average(phi, J)
            zonal.append(fs)
            # v_ExB = -E_r/C_xy with E_r = -dphi/dx, i.e. +dphi/dx / C_xy.
            # See the convention note in `_gene3d`; the GAM frequency is
            # sign-blind but the trace is labelled `v_exb` and must agree with
            # what `ShearingRate` calls by that name.
            dzonal.append(np.gradient(fs, x) / C_xy)
            # First radial harmonic of the zonal (ky=0) component, z-averaged
            # with the y-averaged Jacobian (the radial transform leaves no
            # radial index to weight by).
            zonal_x = phi.mean(axis=1)                      # (nx, nz)
            kx_modes = np.fft.fft(zonal_x, axis=0) / nx
            kx1.append(np.average(kx_modes[1], weights=J_z.mean(axis=0)))

        times = np.asarray(times)
        result = {
            "times": times,
            "x": x,
            "x_o_a": np.asarray(coord["x_o_a"], dtype=float),
            "x_index": int(ix),
            "amplitude": np.asarray(amplitude),
            "phi_zonal": np.asarray(zonal),
            "v_exb": np.asarray(dzonal),
            "phi_kx1": np.real(np.asarray(kx1)),
        }

        # Everything below is normalised to its value at the first output time,
        # which is only meaningful if there is a zonal component to speak of.
        # Without one — a run with no zonal drive, or a pure ky != 0 initial
        # condition — the trace is round-off, and normalising noise by noise
        # produces a plausible-looking oscillation that a fit will happily turn
        # into a confident frequency. Compare against the amplitude of the field
        # itself and refuse rather than invent.
        scale = float(np.max(result["amplitude"])) if amplitude else 0.0
        result["has_zonal"] = _has_signal(result["phi_kx1"], scale)
        result["phi_zonal_mid"] = _normalise(result["phi_zonal"][:, ix], scale)
        result["v_exb_mid"] = _normalise(result["v_exb"][:, ix])
        result["phi_kx1_norm"] = _normalise(result["phi_kx1"], scale)
        result["fit"] = (_fit_damped_oscillation(times, result["phi_kx1_norm"])
                         if result["has_zonal"]
                         else {"omega": float("nan"), "gamma": float("nan")})
        self._cache[key] = result
        return result

    # ------------------------------------------------------------------

    def dataset(self, t=None):
        """Return the traces as an :class:`xarray.Dataset`."""
        raw = self.compute(t)
        params = self.run.params.get(0)
        data_vars = {
            "phi_zonal": (("time", "x"), raw["phi_zonal"]),
            "v_exb": (("time", "x"), raw["v_exb"]),
            "phi_zonal_mid": (("time",), raw["phi_zonal_mid"]),
            "v_exb_mid": (("time",), raw["v_exb_mid"]),
            "phi_kx1": (("time",), raw["phi_kx1_norm"]),
        }
        ds = make_dataset(data_vars, {"x": raw["x_o_a"], "time": raw["times"]},
                          params=params)
        for name in ("phi_zonal_mid", "v_exb_mid", "phi_kx1"):
            ds[name].attrs["units"] = "normalised to t = t_0"
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = self.run.geometry_kind
        ds.attrs["x_index"] = raw["x_index"]
        ds["amplitude"] = ("time", raw["amplitude"])
        ds["amplitude"].attrs["long_name"] = "max |phi| over the whole domain"
        ds.attrs["gam_frequency"] = raw["fit"]["omega"]
        ds.attrs["gam_damping"] = raw["fit"]["gamma"]
        ds.attrs["has_zonal_component"] = int(raw["has_zonal"])
        return ds



    # ------------------------------------------------------------------

    def plot(self, t=None, **kw):
        """Plot the normalised zonal traces and the fitted GAM parameters."""
        raw = self.compute(t)
        times = raw["times"]
        fit = raw["fit"]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(times, raw["phi_zonal_mid"])
        axes[0].set_ylabel(r"$\phi_{k_y=0}(x_0,t)\,/\,\phi_{k_y=0}(x_0,t_0)$")
        axes[1].plot(times, raw["v_exb_mid"])
        axes[1].set_ylabel(r"$v_{E\times B}(x_0,t)\,/\,v_{E\times B}(x_0,t_0)$")
        if not raw["has_zonal"]:
            for ax in axes:
                ax.text(0.5, 0.5, "no zonal component\nabove round-off",
                        ha="center", va="center", transform=ax.transAxes)
        axes[2].plot(times, raw["phi_kx1_norm"], label="data")
        if np.isfinite(fit["omega"]):
            envelope = np.exp(fit["gamma"] * (times - times[0]))
            axes[2].plot(times, envelope, "--", color="k",
                         label=rf"$e^{{\gamma t}}$, $\gamma$={fit['gamma']:.3g}")
            axes[2].set_title(rf"$\omega_{{GAM}}$ = {fit['omega']:.4g}")
        axes[2].set_ylabel(r"$\phi_{k_x=k_{x,\min}, k_y=0}$ (normalised)")
        axes[2].legend(fontsize=8)

        for ax in axes:
            ax.set_xlabel(r"$t\;[L_{\rm ref}/c_{\rm ref}]$")
            ax.axhline(0.0, lw=0.6, color="grey")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: A zonal trace below this fraction of the field amplitude is round-off.
ZONAL_FLOOR = 1e-6


def _has_signal(trace, scale) -> bool:
    """Whether *trace* is distinguishable from round-off at the field's scale."""
    arr = np.asarray(trace, dtype=float)
    if arr.size == 0 or scale <= 0:
        return False
    return bool(np.max(np.abs(arr)) > ZONAL_FLOOR * scale)


def _normalise(trace, scale=None):
    """
    Divide a trace by its first value.

    Returns NaN when the result would be meaningless: a zero first value, or —
    when *scale* is given — a trace that never rises above round-off at that
    scale. NaN propagates into the plots and the fitted parameters, which is the
    honest outcome; silently returning a normalised noise trace is not.
    """
    arr = np.asarray(trace, dtype=float)
    if arr.size == 0:
        return arr
    if scale is not None and not _has_signal(arr, scale):
        return np.full_like(arr, np.nan)
    first = arr[0]
    if first == 0:
        return np.full_like(arr, np.nan)
    return arr / first


def _fit_damped_oscillation(times, trace):
    """
    Fit ``exp(gamma t) cos(omega t)`` loosely: frequency from zero crossings,
    damping from the decay of successive extrema.

    A full nonlinear fit is more than the data usually supports — a GAM test is
    often only a few periods long — and these two estimators degrade gracefully
    to NaN rather than to a confidently wrong number.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(trace, dtype=float)
    out = {"omega": float("nan"), "gamma": float("nan")}
    if t.size < 4 or not np.all(np.isfinite(y)):
        return out

    # Frequency: mean spacing of sign changes is half a period. The crossing
    # is interpolated within the straddling interval rather than snapped to a
    # sample, which otherwise biases the period by up to one output interval —
    # a GAM is often only sampled a handful of times per period.
    sign_change = np.where(np.diff(np.sign(y)) != 0)[0]
    if sign_change.size >= 2:
        crossings = []
        for i in sign_change:
            dy = y[i + 1] - y[i]
            frac = 0.0 if dy == 0 else -y[i] / dy
            crossings.append(t[i] + frac * (t[i + 1] - t[i]))
        half_period = np.mean(np.diff(crossings))
        if half_period > 0:
            out["omega"] = float(np.pi / half_period)

    # Damping: slope of ln|extrema| against their times.
    interior = y[1:-1]
    is_extremum = ((np.abs(interior) > np.abs(y[:-2]))
                   & (np.abs(interior) > np.abs(y[2:])))
    peaks = np.where(is_extremum)[0] + 1
    if peaks.size >= 2:
        amp = np.abs(y[peaks])
        good = amp > 0
        if good.sum() >= 2:
            out["gamma"] = float(
                np.polyfit(t[peaks][good], np.log(amp[good]), 1)[0])
    return out

