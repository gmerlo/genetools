# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
omega.py — frequency-focused view of the linear fit.

The same computation as :class:`~genetools.diagnostics.growthrate.GrowthRate`,
presented as the frequency estimate and the power spectrum of the de-trended
signal — the question ``run.omega`` is usually asked to answer. Kept separate so
that asking for a frequency does not mean reading a growth-rate plot.

GENE-3D only for now: a spectral run already reports omega per ky from
:class:`~genetools.diagnostics.growthrate.GrowthRate`, so there is nothing extra
to present.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from genetools.diagnostics._base import RunDiagnostic
from genetools.diagnostics.growthrate import GrowthRate, _SCIPY

try:
    from scipy.signal import welch
except ImportError:                                  # pragma: no cover
    welch = None


class Omega(RunDiagnostic):
    """
    Frequency-focused view of :class:`GrowthRate`.

    Same computation, presented as the frequency estimates and the power spectrum
    of the de-trended signal.
    """

    name = "omega"
    supported = ("xy_global",)

    def __init__(self, run):
        super().__init__(run)
        self._gr = GrowthRate(run)

    def compute(self, t=None):
        return self._gr.compute(t)

    def dataset(self, t=None):
        return self._gr.dataset(t)

    def plot(self, t=None, **kw):
        """Plot the de-trended reference signal and its power spectrum."""
        raw = self.compute(t)
        gamma = self._gr.gamma(t)
        times, signal = raw["times"], raw["phi_at_ref"]
        detrended = signal / np.exp(gamma * (times - times[0]))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(times, detrended, ".-")
        axes[0].set_xlabel(r"$t\;[L_{\rm ref}/c_{\rm ref}]$")
        axes[0].set_ylabel(r"$\phi\,e^{-\gamma t}$")
        axes[0].set_title(rf"growth removed ($\gamma$ = {gamma:.4g})")

        dt = float(np.mean(np.diff(times))) if times.size > 1 else 0.0
        if _SCIPY and dt > 0 and detrended.size >= 8:
            freq, power = welch(detrended, fs=np.pi / dt,
                                nperseg=min(256, detrended.size))
            axes[1].semilogy(freq, power)
            axes[1].axvline(self._gr.omega(t), ls="--", color="k",
                            label=rf"$\omega$ = {self._gr.omega(t):.4g}")
            axes[1].legend(fontsize=8)
            axes[1].set_xlabel(r"$\omega\;[c_{\rm ref}/L_{\rm ref}]$")
            axes[1].set_ylabel("power")
        else:
            axes[1].text(0.5, 0.5,
                         "power spectrum needs scipy\nand >= 8 samples",
                         ha="center", va="center", transform=axes[1].transAxes)
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
