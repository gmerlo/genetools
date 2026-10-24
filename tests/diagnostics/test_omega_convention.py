# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
The one definition of ``omega`` the package reports.

GENE fixes it operationally in ``convergence_monitoring.F90``::

    om_arr = log(phi_new/phi_old) / (dt*istep_omega)              ! :196
    write (OMEGAFILE, fmt) kymin, real(om_conv), aimag(om_conv)   ! :280

so ``gamma = Re[d ln phi/dt]`` and ``omega = +Im[d ln phi/dt]``, the coefficient
in ``phi ~ exp(gamma t + i omega t)``. The IDL diagnostic agrees in both modes:
``frequency.pro:247`` is the same expression, and its FFT mode leaves the axis
un-negated (``:450``) and pairs it with ``SHIFT(..., n/2-1)`` (``:505``), which
lands each bin on its own frequency.

Both genetools estimators used to negate this, so every frequency the package
reported was mirrored against GENE, ``omega<ext>`` and IDL alike -- reversing
every propagation direction while looking healthy. Nothing caught it: the
``omega<ext>`` cross-check in `GrowthRate` is only printed and plotted beside
the fit, never asserted, and there was no growth-rate test at all.
"""

import numpy as np
import pytest

from genetools.diagnostics.frequency import welch, spectral_peaks
from genetools.io.omega import read_omega


def gene_omega(phi, dt):
    """GENE's expression, transcribed: ``Im[log(phi_new/phi_old)/dt]``."""
    return float(np.mean(np.imag(np.log(phi[1:] / phi[:-1]) / dt)))


def gene_gamma(phi, dt):
    """Its real part, for completeness: a growing mode must give gamma > 0."""
    return float(np.mean(np.real(np.log(phi[1:] / phi[:-1]) / dt)))


def growthrate_fit(phi, times):
    """
    The expression `GrowthRate._compute_spectral` uses, verbatim.

    Kept in step with ``diagnostics/growthrate.py`` by hand -- there is no
    spectral field fixture to drive the class end to end, so this pins the
    formula rather than the code path.
    """
    gamma = np.polyfit(times, np.log(np.abs(phi)), 1)[0]
    omega = np.polyfit(times, np.unwrap(np.angle(phi)), 1)[0]
    return float(gamma), float(omega)


@pytest.mark.parametrize("theta", [0.37, -0.37, 1.8, -1.8])
@pytest.mark.parametrize("gam", [0.0, 0.11])
class TestEstimatorsMatchGene:
    """Every estimator on the same signal must give GENE's pair."""

    @staticmethod
    def _signal(gam, theta, nt=512, dt=0.05):
        t = np.arange(nt) * dt
        return np.exp(gam * t) * np.exp(1j * theta * t), t, dt

    def test_gene_expression_recovers_its_own_input(self, gam, theta):
        phi, _, dt = self._signal(gam, theta)
        assert gene_omega(phi, dt) == pytest.approx(theta, abs=1e-6)
        assert gene_gamma(phi, dt) == pytest.approx(gam, abs=1e-6)

    def test_the_growthrate_fit_agrees(self, gam, theta):
        phi, t, dt = self._signal(gam, theta)
        fit_gamma, fit_omega = growthrate_fit(phi, t)
        assert fit_omega == pytest.approx(gene_omega(phi, dt), abs=1e-6)
        assert fit_gamma == pytest.approx(gene_gamma(phi, dt), abs=1e-6)

    def test_the_welch_axis_agrees(self, gam, theta):
        """The spectral route, which has to land on the same sign."""
        phi, _, dt = self._signal(gam, 0.0 if gam else theta)
        if gam:
            pytest.skip("a growing signal has no single Welch peak")
        omega, power, _ = welch(phi[:, None], dt, nperseg=128, noverlap=64,
                                detrend=False)
        peak = float(spectral_peaks(omega, power[:, 0], 1)[0])
        assert peak == pytest.approx(gene_omega(phi, dt), abs=0.06)


class TestPlusYWaveIsNegativeOmega:
    """
    The consequence worth stating out loud.

    ``cos(k0 y - w t)`` travels in +y; its ``+k0`` binormal component goes as
    ``exp(-i w t)``, whose ``Im[d ln phi/dt]`` is ``-w``. So in GENE's
    convention a ``+y``-travelling wave sits at **negative** omega for
    ``+k_y``. Reading the old sign, one concluded the opposite direction.
    """

    def test_the_direction_is_what_gene_says(self):
        ny, nt, mode = 32, 512, 3
        ly, dt, w = 10.0, 0.05, 4.0
        y = np.arange(ny) * (ly / ny)
        t = np.arange(nt) * dt
        k0 = 2 * np.pi * mode / ly
        phi = np.cos(k0 * y[None, :] - w * t[:, None])
        hat = np.fft.fft(phi, axis=1)[:, mode] / ny       # the +k0 component
        assert gene_omega(hat, dt) == pytest.approx(-w, abs=0.05)
        omega, power, _ = welch(hat[:, None], dt, nperseg=128, noverlap=64)
        assert float(omega[int(np.argmax(power[:, 0]))]) < 0


class TestOmegaFileRoundTrip:
    """`read_omega` must not reinterpret the sign it is handed."""

    def test_values_are_passed_through(self, tmp_path):
        (tmp_path / "omega.dat").write_text(
            "0.3000   0.1234  -0.5678\n0.6000   0.2000   0.4000\n")
        got = read_omega(tmp_path, ".dat")
        np.testing.assert_allclose(got["ky"], [0.3, 0.6])
        np.testing.assert_allclose(got["gamma"], [0.1234, 0.2])
        np.testing.assert_allclose(got["omega"], [-0.5678, 0.4])
