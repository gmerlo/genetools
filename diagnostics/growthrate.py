# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
growthrate.py — linear growth rate and real frequency.

How the measurement is made depends on what the run resolves.

**Spectral y** (flux tube, x-global) — every ``ky`` is an independent linear
mode, so gamma and omega are fitted per mode from the complex amplitude of
``phi`` at a fixed reference point: gamma from how fast ``|phi|`` grows, omega
from how fast its phase rotates. If an ``omega<ext>`` file is present its values
are attached as a cross-check; they are never required.

**GENE-3D** — real space in x *and* y, so there is no per-``ky`` mode to fit. The
growth rate comes from the amplitude of the whole potential,
``gamma = d ln max|phi| / dt``, and the frequency from the oscillation of ``phi``
at a fixed point once that growth is divided out.

The GENE-3D path has to cope with *rescaling*: a linear run periodically
renormalises the distribution to keep it in range, which drops ``max|phi|``
discontinuously. Fitting straight through those drops biases the growth rate
towards zero, so the series is split at every decrease and fitted per segment.

GENE-3D creates an ``omega<ext>`` file but never writes to it — its convergence
monitor prints the growth rate and keeps the value in memory
(``convergence_monitoring.F90``). So there is no independent value on disk to
cross-check a GENE-3D run against, and none is attempted.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from genetools.diagnostics._base import RunDiagnostic
from genetools.io import omega as _omega

try:
    from scipy.signal import argrelextrema, welch
    _SCIPY = True
except ImportError:                                  # pragma: no cover
    _SCIPY = False


class GrowthRate(RunDiagnostic):
    """Linear growth rate and real frequency from the field file."""

    name = "growthrate"

    # ------------------------------------------------------------------
    # Spectral y: one mode per ky
    # ------------------------------------------------------------------

    def _compute_spectral(self, t):
        run = self.run
        times_all, idx = self._indices(run.field, t)
        if t is None:
            n = times_all.size
            idx = np.arange(n // 2, n)        # trailing half by default
        sel = list(idx)
        ky = np.asarray(self.coord["ky"])

        if len(sel) < 2:
            nan = np.full(ky.size, np.nan)
            return {"kind": "spectral", "ky": ky, "gamma": nan, "omega": nan,
                    "window": (None, None)}

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

        return {"kind": "spectral", "ky": ky, "gamma": gamma, "omega": omega,
                "window": (float(times[0]), float(times[-1]))}

    def _file_crosscheck(self):
        """Return omega-file values keyed by ext, or ``{}`` if none present."""
        if self.is_3d:
            # GENE-3D opens omega<ext> and never writes to it.
            return {}
        out = {}
        for ext in self.run.extensions:
            data = _omega.read_omega(self.run._folder, ext)
            if data is not None:
                out[ext] = data
        return out

    # ------------------------------------------------------------------
    # GENE-3D: one mode, rescaling-aware
    # ------------------------------------------------------------------

    def _compute_3d(self, t):
        """
        Stream the field file and return the amplitude series and its fits.

        The reference point for the frequency is the location of the maximum
        ``|phi|`` in the *last* snapshot of the window: by then the
        fastest-growing mode dominates, so that point tracks it rather than
        whatever happened to be largest at initialisation.
        """
        run = self.run
        reader = run.field
        _, idx = self._indices(reader, t)
        if idx.size < 2:
            raise ValueError(
                "Need at least two field snapshots to fit a growth rate; "
                f"found {idx.size} in the requested window.")
        i_phi = reader.index_of("phi")

        # Locate the reference point from the final snapshot first.
        (_, last_arrays), = reader.stream_selected([int(idx[-1])])
        ref = np.unravel_index(int(np.argmax(np.abs(last_arrays[i_phi]))),
                               last_arrays[i_phi].shape)

        times, amp, at_ref = [], [], []
        for time, arrays in reader.stream_selected(idx):
            phi = arrays[i_phi]
            times.append(time)
            amp.append(float(np.max(np.abs(phi))))
            at_ref.append(float(phi[ref]))

        times = np.asarray(times)
        amp = np.asarray(amp)
        at_ref = np.asarray(at_ref)

        segments = _rescaling_segments(amp)
        gamma_seg, seg_times = _fit_segments(times, amp, segments)
        gamma_inst, inst_times = _instantaneous_gamma(times, amp)
        omega_seg, omega_fft = _frequencies(times, at_ref, segments, gamma_seg)

        return {
            "kind": "3d",
            "times": times, "amplitude": amp, "phi_at_ref": at_ref,
            "reference_index": tuple(int(i) for i in ref),
            "segments": segments,
            "gamma_segments": gamma_seg, "segment_times": seg_times,
            "gamma_instant": gamma_inst, "instant_times": inst_times,
            "omega_segments": omega_seg, "omega_fft": omega_fft,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(self, t=None):
        """
        Return the raw fit results for the window *t*.

        The dict differs by geometry — ``ky``/``gamma``/``omega`` arrays for a
        spectral run, an amplitude history plus per-segment fits for GENE-3D —
        and carries a ``kind`` key saying which. Use :meth:`gamma`,
        :meth:`omega` or :meth:`dataset` for a geometry-independent answer.
        """
        key = self._key(t)
        if key not in self._cache:
            self._cache[key] = (self._compute_3d(t) if self.is_3d
                                else self._compute_spectral(t))
        return self._cache[key]

    def gamma(self, t=None) -> float:
        """
        Single best growth-rate estimate.

        Spectral runs report the fastest-growing mode. GENE-3D reports the last
        *complete* segment fit: the final segment is normally truncated by the
        end of the window, so the one before it is more trustworthy.
        """
        raw = self.compute(t)
        if raw["kind"] == "spectral":
            g = raw["gamma"]
            return float(np.nanmax(g)) if np.any(np.isfinite(g)) else float("nan")
        fits = raw["gamma_segments"]
        good = fits[np.isfinite(fits)]
        if good.size == 0:
            return float("nan")
        return float(good[-2] if good.size > 1 else good[-1])

    def omega(self, t=None) -> float:
        """
        Single best frequency estimate.

        Spectral runs report the frequency of the fastest-growing mode; GENE-3D
        prefers the spectral estimate over the peak-spacing one.
        """
        raw = self.compute(t)
        if raw["kind"] == "spectral":
            g, w = raw["gamma"], raw["omega"]
            if not np.any(np.isfinite(g)):
                return float("nan")
            return float(w[int(np.nanargmax(g))])
        for candidate in (raw["omega_fft"], raw["omega_segments"]):
            good = np.asarray(candidate, dtype=float)
            good = good[np.isfinite(good)]
            if good.size:
                return float(good[-1])
        return float("nan")

    def dataset(self, t=None):
        """Return the fit results as an :class:`xarray.Dataset`."""
        import xarray as xr
        raw = self.compute(t)
        if raw["kind"] == "spectral":
            ds = xr.Dataset(
                {"gamma": ("ky", raw["gamma"]), "omega": ("ky", raw["omega"])},
                coords={"ky": raw["ky"]},
            )
            ds.attrs["t_window"] = raw["window"]
            cross = self._file_crosscheck()
            if cross:
                first = next(iter(cross.values()))
                ds.attrs["omega_file_ky"] = first["ky"]
                ds.attrs["omega_file_gamma"] = first["gamma"]
                ds.attrs["omega_file_omega"] = first["omega"]
        else:
            ds = xr.Dataset(
                {"amplitude": ("time", raw["amplitude"]),
                 "phi_at_ref": ("time", raw["phi_at_ref"])},
                coords={"time": raw["times"]},
            )
            ds["amplitude"].attrs["long_name"] = \
                "max |phi| over the whole domain"
            if raw["instant_times"].size:
                ds = ds.merge(xr.Dataset(
                    {"gamma_instant": ("time_gamma", raw["gamma_instant"])},
                    coords={"time_gamma": raw["instant_times"]}))
            if raw["segment_times"].size:
                ds = ds.merge(xr.Dataset(
                    {"gamma_segment": ("time_segment", raw["gamma_segments"]),
                     "omega_segment": ("time_segment", raw["omega_segments"])},
                    coords={"time_segment": raw["segment_times"]}))
            ds.attrs["n_rescalings"] = int(len(raw["segments"]) - 1)
            ds.attrs["reference_index"] = list(raw["reference_index"])

        ds.attrs["geometry_kind"] = self.geometry_kind
        ds.attrs["gamma"] = self.gamma(t)
        ds.attrs["omega"] = self.omega(t)
        return ds

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, t=None, **kw):
        """Plot the fit: gamma/omega against ky, or the amplitude history."""
        raw = self.compute(t)
        if raw["kind"] == "3d":
            return self._plot_3d(raw, t)
        return self._plot_spectral(raw, t)

    def _plot_spectral(self, raw, t):
        ky, gamma, omega = raw["ky"], raw["gamma"], raw["omega"]
        window = raw["window"]
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
        ax1.set_xlabel(r"$k_y\rho$")
        ax1.set_ylabel(r"$\gamma\,[c_{\rm ref}/L_{\rm ref}]$")
        ax2.set_xlabel(r"$k_y\rho$")
        ax2.set_ylabel(r"$\omega\,[c_{\rm ref}/L_{\rm ref}]$")
        ax1.grid(True); ax2.grid(True)
        if cross:
            ax1.legend(fontsize=8); ax2.legend(fontsize=8)
        fig.tight_layout()
        plt.show()
        return fig

    def _plot_3d(self, raw, t):
        """Amplitude history, growth-rate fits and the reference-point signal."""
        gamma = self.gamma(t)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].semilogy(raw["times"], raw["amplitude"], ".-")
        for start, stop in raw["segments"]:
            axes[0].axvspan(raw["times"][start], raw["times"][stop],
                            alpha=0.08, color="C1")
        axes[0].set_xlabel(r"$t\;[L_{\rm ref}/c_{\rm ref}]$")
        axes[0].set_ylabel(r"$\max|\phi|$")
        axes[0].set_title(f"amplitude ({len(raw['segments']) - 1} rescalings)")

        if raw["instant_times"].size:
            axes[1].plot(raw["instant_times"], raw["gamma_instant"], ".",
                         label="instantaneous")
        if raw["segment_times"].size:
            axes[1].plot(raw["segment_times"], raw["gamma_segments"], "o-",
                         color="C3", label="per segment")
        axes[1].axhline(gamma, ls="--", color="k",
                        label=rf"$\gamma$ = {gamma:.4g}")
        axes[1].set_xlabel(r"$t\;[L_{\rm ref}/c_{\rm ref}]$")
        axes[1].set_ylabel(r"$\gamma\;[c_{\rm ref}/L_{\rm ref}]$")
        axes[1].legend(fontsize=8)

        axes[2].plot(raw["times"], raw["phi_at_ref"], ".-")
        axes[2].set_xlabel(r"$t\;[L_{\rm ref}/c_{\rm ref}]$")
        axes[2].set_ylabel(r"$\phi$ at reference point")
        axes[2].set_title(rf"$\omega$ = {self.omega(t):.4g}")

        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
        return fig


# ---------------------------------------------------------------------------
# Fitting helpers (GENE-3D)
# ---------------------------------------------------------------------------

def _rescaling_segments(amp):
    """
    Split the amplitude series at every decrease.

    A linear run renormalises to stay in range, which shows up as a sudden drop
    in ``max|phi|``. Each stretch between drops grows cleanly, so fitting them
    separately avoids the bias a fit straight through the drops would carry.
    Segments shorter than two samples cannot be fitted and are dropped.
    """
    amp = np.asarray(amp, dtype=float)
    drops = np.where(np.diff(amp) < 0)[0] + 1
    starts = np.concatenate(([0], drops))
    stops = np.concatenate((drops - 1, [amp.size - 1]))
    return [(int(a), int(b)) for a, b in zip(starts, stops) if b > a]


def _fit_segments(times, amp, segments):
    """Least-squares slope of ``ln(amp)`` against ``t`` within each segment."""
    gammas, centres = [], []
    for start, stop in segments:
        t_seg = times[start:stop + 1]
        a_seg = amp[start:stop + 1]
        if t_seg.size < 2 or np.any(a_seg <= 0):
            continue
        slope = np.polyfit(t_seg, np.log(a_seg), 1)[0]
        gammas.append(float(slope))
        centres.append(0.5 * (t_seg[0] + t_seg[-1]))
    return np.asarray(gammas), np.asarray(centres)


def _instantaneous_gamma(times, amp):
    """
    Pointwise ``d ln(amp)/dt``, keeping only the growing steps.

    The steps across a rescaling are meaningless — the amplitude dropped for
    numerical reasons, not physical ones — so they are excluded rather than
    plotted as large negative growth rates.
    """
    amp = np.asarray(amp, dtype=float)
    if amp.size < 2 or np.any(amp <= 0):
        return np.asarray([]), np.asarray([])
    d_log = np.diff(np.log(amp))
    dt = np.diff(times)
    keep = (d_log > 0) & (dt > 0)
    if not np.any(keep):
        return np.asarray([]), np.asarray([])
    centres = 0.5 * (times[:-1] + times[1:])
    return d_log[keep] / dt[keep], centres[keep]


def _frequencies(times, signal, segments, gammas):
    """
    Estimate the real frequency per segment, and once spectrally.

    Within a segment the growth is divided out and the spacing of successive
    maxima gives the period. ``omega = pi / period`` follows GENE's convention
    that ``|phi|`` peaks twice per oscillation of a complex mode.
    """
    per_segment, spectral = [], []
    for (start, stop), gamma in zip(segments, _pad(gammas, len(segments))):
        t_seg = times[start:stop + 1]
        s_seg = np.asarray(signal[start:stop + 1], dtype=float)
        if t_seg.size < 4 or not np.isfinite(gamma):
            per_segment.append(np.nan)
            continue
        detrended = s_seg / np.exp(gamma * (t_seg - t_seg[0]))
        if _SCIPY:
            peaks = argrelextrema(detrended, np.greater)[0]
        else:                                        # pragma: no cover
            peaks = np.where((detrended[1:-1] > detrended[:-2])
                             & (detrended[1:-1] > detrended[2:]))[0] + 1
        if peaks.size >= 2:
            period = (t_seg[peaks[-1]] - t_seg[peaks[0]]) / (peaks.size - 1)
            per_segment.append(float(np.pi / period) if period > 0 else np.nan)
        else:
            per_segment.append(np.nan)

        if _SCIPY and detrended.size >= 8:
            dt = float(np.mean(np.diff(t_seg)))
            if dt > 0:
                freq, power = welch(detrended, fs=np.pi / dt,
                                    nperseg=min(256, detrended.size))
                spectral.append(float(freq[int(np.argmax(power))]))

    return np.asarray(per_segment), np.asarray(spectral)


def _pad(values, n):
    """Yield *n* values from *values*, padding with NaN."""
    values = list(np.asarray(values, dtype=float))
    values += [np.nan] * (n - len(values))
    return values[:n]
