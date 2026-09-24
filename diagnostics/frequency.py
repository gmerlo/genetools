# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
frequency.py — frequency spectrum of the fluctuations in a nonlinear run.

The linear diagnostics (:class:`~genetools.diagnostics.growthrate.GrowthRate`,
:class:`~genetools.diagnostics.omega.Omega`) fit one coherent mode per ky. Once
a run saturates there is no single mode to fit: each ky carries a *band* of
frequencies, and the questions are where that band sits, how wide it is, and
which way it propagates. This builds

    P(k_y, omega) = < |phi(k_y, omega)|^2 >

and reduces it to a dominant frequency, a subdominant one, the spectral
centroid and a width, per ky.

Method — after ``gene/diagnostics/prog/frequency.pro``, FFT mode
----------------------------------------------------------------
The IDL diagnostic's ``fftmode`` branch is the reference. Same four steps:

1. Resample onto a uniform time grid. GENE's dt is adaptive, so output times
   are unevenly spaced and an FFT of them directly is meaningless.
2. Welch's method: a Hann window of a quarter of the record, advanced with 50%
   overlap, the periodograms averaged. IDL's ``twin = [0.25, 0.5]``. A single
   FFT over the whole record has the finest resolution, ``2 pi / T``, and is
   one realisation of a random process — far too noisy to read a peak off.
3. Average ``|.|^2`` over the other coordinates with the Jacobian weight, after
   the transform, never before: averaging complex amplitudes first cancels the
   very phase information the frequency lives in.
4. Extract the dominant and subdominant peak (local maxima above 5% of the
   largest, IDL's threshold), the centroid and the width.

Two deliberate departures from the IDL version:

* **Each segment has its mean removed.** A finite time-average is not a
  fluctuation, and left in it puts a spike at ``omega = 0`` that drags the
  centroid towards zero. IDL leaves it; ``detrend=False`` restores that.
* **The Nyquist frequency is reported and checked.** ``omega_max = pi / dt``
  with ``dt`` the *output* spacing, ``istep_field`` steps of an adaptive dt.
  In a nonlinear run the turbulence frequency is very often above it, and then
  the spectrum is aliased and every number here is fiction. IDL prints the
  range but never warns.

Sign convention
---------------
GENE's ansatz is ``f ~ exp(i(k_x x + k_y y - omega t))``, so a mode at ``+k_y``
has a time dependence ``exp(-i omega t)`` and positive omega means the wave
travels in ``+y``. numpy's forward FFT carries the kernel ``exp(-i omega t)``,
so its bin ``n`` holds this module's ``omega = -2 pi f_n``: the frequency axis
is the *negated* ``fftfreq``. Getting this backwards silently mirrors the
spectrum and reverses every reported propagation direction, so
``test_travelling_wave_lands_at_positive_omega`` pins it against a wave built
by hand from GENE's own ansatz.

Memory
------
A time-frequency analysis needs the whole time series in memory before it can
transform it — unlike every other diagnostic here, the time axis cannot be
streamed away. The cost is ``n_times * nx * nky * nz_kept`` complex64, so **z
is cut at the grid point nearest zero by default**, not averaged: keeping all z
multiplies the array by ``nz``. ``zlim`` widens it deliberately. The size is
reported by :meth:`compute` so a choice that will not fit is visible before it
is made.
"""

from __future__ import annotations

import warnings

import numpy as np
import matplotlib.pyplot as plt

from genetools._xr import make_dataset, unit_attrs
from genetools.diagnostics._base import RunDiagnostic
from genetools.diagnostics import _gene3d as g3

#: Local maxima below this fraction of the largest are not reported as peaks.
#: IDL's `frequency.pro` uses the same threshold.
_PEAK_FLOOR = 0.05

#: Warn when a reported frequency sits above this fraction of Nyquist.
_ALIAS_FRACTION = 0.8


def resample_uniform(times, data):
    """
    Linearly resample *data* onto a uniform time grid.

    Returns ``(t_uniform, resampled, spacing_ratio)``. *data* is indexed by time
    along axis 0; every trailing axis is carried along.

    GENE's timestep is adaptive and output happens every ``istep_*`` *steps*, so
    the output times are unevenly spaced — the same fact that makes an unweighted
    time average wrong everywhere else in this package. An FFT assumes uniform
    sampling, so the series is interpolated first, exactly as the IDL
    diagnostic does.

    *spacing_ratio* is ``max(dt) / min(dt)`` of the original series: 1.0 means
    the output was already uniform and the interpolation changed nothing, while
    a large value means it did real work and the high-frequency end of the
    spectrum is correspondingly smoothed.
    """
    times = np.asarray(times, dtype=float)
    n = times.size
    if n < 2:
        raise ValueError("a frequency spectrum needs at least two snapshots")
    gaps = np.diff(times)
    if np.any(gaps <= 0):
        raise ValueError("snapshot times are not strictly increasing")
    ratio = float(gaps.max() / gaps.min())

    t_new = np.linspace(times[0], times[-1], n)
    hi = np.clip(np.searchsorted(times, t_new, side="left"), 1, n - 1)
    lo = hi - 1
    span = times[hi] - times[lo]
    w = ((t_new - times[lo]) / span).reshape((-1,) + (1,) * (data.ndim - 1))
    out = data[lo] * (1.0 - w) + data[hi] * w
    return t_new, out.astype(data.dtype, copy=False), ratio


def welch(data, dt, nperseg, noverlap, detrend=True):
    """
    Two-sided power spectrum of a complex series, averaged over Hann windows.

    Returns ``(omega, power, n_windows)`` with *omega* ascending and *power*
    indexed by frequency along axis 0.

    The series is complex and its spectrum is **not** symmetric: ``+omega`` and
    ``-omega`` are the two propagation directions, and collapsing them would
    throw away which way the turbulence moves. Both halves are kept.

    The frequency axis is the negated ``fftfreq`` — see the module docstring on
    the sign convention.
    """
    n = data.shape[0]
    nperseg = int(min(max(nperseg, 2), n))
    noverlap = int(min(max(noverlap, 0), nperseg - 1))
    step = nperseg - noverlap

    window = np.hanning(nperseg)
    shape = (nperseg,) + (1,) * (data.ndim - 1)
    wb = window.reshape(shape)
    # Power spectral density normalisation: |sum w x e^{i w t}|^2 dt / sum w^2,
    # so the result is independent of the window length and of dt.
    norm = dt / float(np.sum(window ** 2))

    total, count = None, 0
    for start in range(0, n - nperseg + 1, step):
        seg = data[start:start + nperseg]
        if detrend:
            seg = seg - seg.mean(axis=0, keepdims=True)
        spec = np.fft.fft(seg * wb, axis=0)
        power = (spec.real ** 2 + spec.imag ** 2) * norm
        total = power if total is None else total + power
        count += 1
    if not count:                                  # pragma: no cover - guarded
        raise ValueError("no complete window fits in the record")

    omega = -2.0 * np.pi * np.fft.fftfreq(nperseg, d=dt)
    order = np.argsort(omega)
    return omega[order], (total / count)[order], count


def spectral_peaks(omega, power, n_peaks=2):
    """
    The *n_peaks* strongest local maxima of *power*, strongest first.

    Maxima below :data:`_PEAK_FLOOR` of the largest are dropped — IDL's rule,
    and what keeps the ripple of a noisy spectrum from being reported as a
    subdominant mode. Missing peaks come back as NaN, so the result always has
    *n_peaks* entries and stacks cleanly across ky.
    """
    power = np.asarray(power, dtype=float)
    out = np.full(n_peaks, np.nan)
    if power.size < 3 or not np.any(power > 0):
        return out
    interior = np.where((power[1:-1] > power[:-2])
                        & (power[1:-1] > power[2:]))[0] + 1
    if interior.size == 0:
        return out
    heights = power[interior]
    keep = interior[heights >= _PEAK_FLOOR * heights.max()]
    keep = keep[np.argsort(power[keep])[::-1]]
    take = min(n_peaks, keep.size)
    out[:take] = omega[keep[:take]]
    return out


def spectral_moments(omega, power):
    """
    ``(centroid, width)`` of a spectrum: its mean frequency and RMS spread.

    The centroid ``int omega P / int P`` is IDL's "first moment" and is the
    robust single number for a broad nonlinear band, where the tallest bin is
    noise-dominated and moves from one window to the next. The width is the
    square root of the second central moment; for a Gaussian band the FWHM is
    ``2 sqrt(2 ln 2)`` times it, and its inverse is a decorrelation time.
    """
    power = np.asarray(power, dtype=float)
    total = np.trapezoid(power, omega) if hasattr(np, "trapezoid") \
        else np.trapz(power, omega)
    if not np.isfinite(total) or total <= 0:
        return np.nan, np.nan
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    centroid = integrate(omega * power, omega) / total
    variance = integrate((omega - centroid) ** 2 * power, omega) / total
    return float(centroid), float(np.sqrt(max(variance, 0.0)))


class Frequency(RunDiagnostic):
    """
    Frequency spectrum of the fluctuations, per binormal mode.

    GENE-3D only for now: y is real space there, so the binormal transform is a
    forward FFT rather than a representation the file already carries.

    Parameters
    ----------
    run : genetools.run.Run
    quantities : sequence of str
        Variables to analyse, from the field or moment file. Default
        ``('phi',)`` — what the IDL diagnostic uses. The frequency of a flux
        (``Q_es``) is not the frequency of the potential, so which one is asked
        for matters.
    species : str, optional
        Species supplying moment quantities (default: the first).
    xlim : (float, float), optional
        Radial window in ``x/a``. Default: the whole domain. Restrict it when
        the profiles vary enough that different surfaces rotate at different
        rates — averaging those together broadens the band artificially.
    zlim : (float, float), optional
        Parallel window. Default: **a cut** at the grid point nearest ``z = 0``,
        the outboard midplane, which is also what keeps the time series small
        enough to hold (see the module docstring).
    window_frac : float
        Welch window length as a fraction of the record (default 0.25, IDL's).
        The frequency resolution is ``2 pi / (window_frac * T)``; smaller values
        average more windows and give a smoother, coarser spectrum.
    overlap : float
        Overlap between successive windows (default 0.5, IDL's).
    detrend : bool
        Remove each segment's mean before transforming (default ``True``).
    n_peaks : int
        How many peaks to report per ky (default 2: dominant and subdominant).
    """

    name = "frequency"
    supported = ("xy_global",)

    def __init__(self, run, quantities=("phi",), species=None,
                 xlim=None, zlim=None, window_frac=0.25, overlap=0.5,
                 detrend=True, n_peaks=2):
        super().__init__(run)
        self.quantities = tuple(quantities)
        self.species = species or (run.species[0] if run.species else None)
        self.xlim, self.zlim = xlim, zlim
        self.window_frac = float(window_frac)
        self.overlap = float(overlap)
        self.detrend = bool(detrend)
        self.n_peaks = int(n_peaks)
        self._cache = {}

    # ------------------------------------------------------------------

    def _z_cut(self, z):
        """Equal bounds at the grid point nearest ``z = 0``: a cut, not a mean."""
        z = np.asarray(z, dtype=float)
        v = float(z[int(np.argmin(np.abs(z)))])
        return (v, v)

    def compute(self, t=None):
        """
        Stream the window, transform to ky, and build ``P(x, ky, omega)``.

        The whole time series is held before the transform — see the module
        docstring — so the array size is reported as it is allocated.
        """
        key = self._key(t)
        if key in self._cache:
            return self._cache[key]

        coord, J = self.coord, self.geom["Jacobian"]
        nx, ny, nz = J.shape
        xsl = g3.index_window(coord["x_o_a"], self.xlim, nx)
        zsl = g3.index_window(coord["z"],
                              self.zlim if self.zlim is not None
                              else self._z_cut(coord["z"]), nz)
        # Per-surface z weight, y-independent by construction (jacobian_yz).
        weights = J.mean(axis=1)[xsl][:, zsl]            # (nx_sel, nz_sel)

        ky_full = np.asarray(coord["ky"], dtype=float)
        out, times, info = {}, None, {}
        for reader, names in self._sources(self.quantities, self.species):
            all_times, idx = self._indices(reader, t)
            slots = {n: reader.index_of(n) for n in names}
            acc = {n: [] for n in names}
            got = []
            for time, arrays in reader.stream_selected(idx, variables=names):
                got.append(time)
                for n in names:
                    # Transform in y, then keep only the window: the FFT needs
                    # the whole binormal axis, the time series does not need
                    # the whole box.
                    hat = g3.to_ky(arrays[slots[n]])
                    acc[n].append(hat[xsl][:, :, zsl])
            if times is None:
                times = np.asarray(got, dtype=float)
            for n in names:
                out[n] = np.asarray(acc[n])              # (nt, nx, nky, nz)

        if times is None or times.size < 2:
            raise ValueError(
                "a frequency spectrum needs at least two snapshots in the "
                "window; widen t.")

        nt = times.size
        sample = next(iter(out.values()))
        info["series_MiB"] = sample.nbytes / 1024 ** 2 * len(out)

        spectra, omega = {}, None
        for name, series in out.items():
            t_new, series, ratio = resample_uniform(times, series)
            dt = float(t_new[1] - t_new[0])
            nperseg = max(4, int(round(self.window_frac * nt)))
            noverlap = int(round(self.overlap * nperseg))
            omega, power, n_win = welch(series, dt, nperseg, noverlap,
                                        detrend=self.detrend)
            # Reduce after the transform, never before. z first, then x, each
            # weighted by the Jacobian, so the pair is the joint average.
            w_z = weights[np.newaxis, :, np.newaxis, :]
            p_xky = (power * w_z).sum(axis=3) / weights.sum(axis=1)[
                np.newaxis, :, np.newaxis]
            w_x = weights.sum(axis=1)[np.newaxis, :, np.newaxis]
            spectra[name] = ((p_xky * w_x).sum(axis=1) / w_x.sum(axis=1))
            info.update(dt=dt, n_windows=n_win, nperseg=nperseg,
                        spacing_ratio=ratio,
                        omega_nyquist=float(np.pi / dt),
                        omega_resolution=float(2 * np.pi / (nperseg * dt)))

        result = {"omega": omega, "ky": ky_full, "spectra": spectra,
                  "times": times, "info": info, "xslice": xsl, "zslice": zsl}
        self._alias_check(result)
        self._cache[key] = result
        return result

    def _alias_check(self, result):
        """
        Warn when the band sits near Nyquist, where the spectrum is aliased.

        ``omega_nyquist = pi / dt`` is set by ``istep_field``, not by the
        physics. A nonlinear run written too sparsely folds its real frequencies
        back into the resolved range, and nothing downstream can detect that
        from the spectrum alone — so it is said here, once, with the number
        needed to fix it.
        """
        nyq = result["info"].get("omega_nyquist")
        if not nyq:
            return
        for name, power in result["spectra"].items():
            peak = np.abs(result["omega"][np.argmax(power, axis=0)])
            if np.nanmax(peak) > _ALIAS_FRACTION * nyq:
                warnings.warn(
                    f"{name}: the strongest frequency reaches "
                    f"{np.nanmax(peak):.3g}, within {1 - _ALIAS_FRACTION:.0%} "
                    f"of the Nyquist frequency {nyq:.3g} set by the output "
                    "cadence. The spectrum is probably aliased; rerun with a "
                    "smaller istep_field before believing these numbers.",
                    RuntimeWarning, stacklevel=3)

    # ------------------------------------------------------------------

    def dataset(self, t=None):
        """
        The ``(ky, omega)`` power map and the per-ky frequency estimates.

        Variables per quantity ``q``: ``q_power`` ``(species-free, ky, omega)``,
        ``q_peak`` ``(ky, peak)``, ``q_centroid`` and ``q_width`` ``(ky,)``.
        """
        raw = self.compute(t)
        omega, ky = raw["omega"], raw["ky"]
        data_vars = {}
        for name, power in raw["spectra"].items():
            # power is (omega, ky); present it ky-first, as it is read.
            p = np.asarray(power).T
            data_vars[f"{name}_power"] = (("ky", "omega"), p)
            peaks = np.stack([spectral_peaks(omega, row, self.n_peaks)
                              for row in p])
            moments = np.array([spectral_moments(omega, row) for row in p])
            data_vars[f"{name}_peak"] = (("ky", "peak"), peaks)
            data_vars[f"{name}_centroid"] = (("ky",), moments[:, 0])
            data_vars[f"{name}_width"] = (("ky",), moments[:, 1])

        params = self.params
        ds = make_dataset(data_vars,
                          {"ky": ky, "omega": omega,
                           "peak": np.arange(self.n_peaks) + 1},
                          params=params)
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = self.geometry_kind
        for k, v in raw["info"].items():
            ds.attrs[k] = v
        ds.attrs["n_times"] = int(raw["times"].size)
        if self.species:
            ds.attrs["species"] = self.species
        return ds

    # ------------------------------------------------------------------

    def plot(self, t=None, ky_positive=True, **kw):
        """
        The ``(ky, omega)`` map beside the frequency estimates against ky.

        ``ky_positive`` drops the zonal and negative-ky half, which is what one
        usually wants: for a real field the ``-ky`` half is the mirror of ``+ky``
        with the frequency reflected, so plotting both shows every band twice.
        """
        ds = self.dataset(t)
        omega = np.asarray(ds["omega"])
        ky = np.asarray(ds["ky"])
        mask = ky > 0 if ky_positive else np.ones(ky.size, dtype=bool)
        if not mask.any():
            mask = np.ones(ky.size, dtype=bool)

        names = sorted({k[:-6] for k in ds.data_vars if k.endswith("_power")})
        nyq = ds.attrs.get("omega_nyquist")
        figs = []
        for name in names:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
            p = np.asarray(ds[f"{name}_power"])[mask]
            peak = np.nanmax(p, axis=1, keepdims=True)
            norm = np.divide(p, peak, out=np.zeros_like(p), where=peak > 0)
            mesh = axes[0].pcolormesh(ky[mask], omega, norm.T,
                                      shading="auto", cmap="magma")
            axes[0].set_xlabel(r"$k_y \rho_{\rm ref}$")
            axes[0].set_ylabel(r"$\omega\;[c_s/L_{\rm ref}]$")
            axes[0].set_title(rf"$|{name}(k_y,\omega)|^2$, "
                              "normalised per $k_y$", fontsize=9)
            fig.colorbar(mesh, ax=axes[0])

            for j in range(self.n_peaks):
                axes[1].plot(ky[mask],
                             np.asarray(ds[f"{name}_peak"])[mask, j],
                             "o-", ms=3, label=f"{j + 1}. peak")
            centroid = np.asarray(ds[f"{name}_centroid"])[mask]
            width = np.asarray(ds[f"{name}_width"])[mask]
            axes[1].plot(ky[mask], centroid, "k-", lw=1.6, label="centroid")
            axes[1].fill_between(ky[mask], centroid - width, centroid + width,
                                 color="0.7", alpha=0.4, label="width")
            if nyq:
                for sign in (1, -1):
                    axes[1].axhline(sign * nyq, color="r", ls=":", lw=1)
                axes[1].set_ylim(-1.1 * nyq, 1.1 * nyq)
            axes[1].axhline(0.0, color="0.6", lw=0.8)
            axes[1].set_xlabel(r"$k_y \rho_{\rm ref}$")
            axes[1].set_ylabel(r"$\omega\;[c_s/L_{\rm ref}]$")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=8)
            fig.suptitle(f"{name} — fluctuation frequency "
                         f"({ds.attrs.get('n_windows', '?')} windows, "
                         f"resolution {ds.attrs.get('omega_resolution', 0):.3g},"
                         f" Nyquist {nyq:.3g})" if nyq else name)
            fig.tight_layout()
            figs.append(fig)
        plt.show()
        return figs
