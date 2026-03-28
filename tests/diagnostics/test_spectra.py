"""Tests for genetools.diagnostics.spectra.Spectra.averages."""

import numpy as np
import pytest

from genetools.diagnostics.spectra import Spectra


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _uniform_flux(nx, nky, nz, value=1.0):
    """Real-valued constant array (no imaginary part, so .real is the value)."""
    return np.full((nx, nky, nz), value, dtype=np.complex128)


def _uniform_J(nz, value=1.0):
    """Uniform Jacobian that sums to 1 after normalisation inside averages."""
    J = np.ones(nz) * value
    return J / J.sum()        # pre-normalised


def _ky_weight(nky):
    """Standard GENE ky_weight: 1 for ky=0, 2 for ky>0."""
    w = np.ones(nky)
    w[1:] = 2.0
    return w


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestAveragesShape:

    def test_even_nx_kx_shape(self):
        """kx spectrum should have shape (nx//2+1,) for even nx."""
        nx, nky, nz = 8, 3, 16
        flux = _uniform_flux(nx, nky, nz)
        J = _uniform_J(nz)
        sp_kx, sp_ky, sum_z = Spectra.averages(flux, J)
        assert sp_kx.shape == (nx // 2 + 1,)

    def test_odd_nx_kx_shape(self):
        """kx spectrum shape correct for odd nx."""
        nx, nky, nz = 7, 3, 16
        flux = _uniform_flux(nx, nky, nz)
        J = _uniform_J(nz)
        sp_kx, sp_ky, sum_z = Spectra.averages(flux, J)
        assert sp_kx.shape == (nx // 2 + 1,)

    def test_ky_spectrum_shape(self):
        nx, nky, nz = 8, 5, 16
        flux = _uniform_flux(nx, nky, nz)
        J = _uniform_J(nz)
        sp_kx, sp_ky, sum_z = Spectra.averages(flux, J)
        assert sp_ky.shape == (nky,)

    def test_z_profile_shape(self):
        nx, nky, nz = 8, 4, 16
        flux = _uniform_flux(nx, nky, nz)
        J = _uniform_J(nz)
        sp_kx, sp_ky, sum_z = Spectra.averages(flux, J)
        assert sum_z.shape == (nz,)

    def test_returns_three_items(self):
        flux = _uniform_flux(4, 2, 8)
        J = _uniform_J(8)
        result = Spectra.averages(flux, J)
        assert len(result) == 3

    def test_none_flux_returns_triple_none(self):
        result = Spectra.averages(None, _uniform_J(8))
        assert result == (None, None, None)


# ---------------------------------------------------------------------------
# Nyquist / folding tests (even nx)
# ---------------------------------------------------------------------------

class TestAveragesKxFolding:

    def test_nyquist_not_doubled_even_nx(self):
        """For even nx the Nyquist bin (index nx//2) must not be doubled."""
        nx = 8          # even
        nky, nz = 2, 4
        # Put all energy in the Nyquist kx-mode (index nx//2 = 4)
        flux = np.zeros((nx, nky, nz), dtype=complex)
        J = _uniform_J(nz)
        flux[nx // 2, 0, :] = 1.0    # real, ky=0

        # No ky_weight to isolate the x-folding
        sp_kx, _, _ = Spectra.averages(flux, J)

        # sp_kx[nx//2] = tmp[nx//2] (no folding partner added)
        assert sp_kx[nx // 2] == pytest.approx(1.0, abs=1e-10)

    def test_non_nyquist_modes_folded_even_nx(self):
        """For even nx modes 1..nx//2-1 should be doubled (kx + (-kx))."""
        nx = 8
        nky, nz = 2, 4
        J = _uniform_J(nz)
        # Put unit energy in kx index 1 and its mirror nx-1
        flux = np.zeros((nx, nky, nz), dtype=complex)
        flux[1, 0, :] = 1.0
        flux[nx - 1, 0, :] = 1.0

        sp_kx, _, _ = Spectra.averages(flux, J)
        # After folding sp_kx[1] = tmp[1] + tmp[nx-1] = 1 + 1 = 2
        assert sp_kx[1] == pytest.approx(2.0, abs=1e-10)

    def test_dc_mode_not_doubled(self):
        """DC mode (kx=0) must never be doubled."""
        nx = 8
        nky, nz = 2, 4
        J = _uniform_J(nz)
        flux = np.zeros((nx, nky, nz), dtype=complex)
        flux[0, 0, :] = 1.0
        sp_kx, _, _ = Spectra.averages(flux, J)
        assert sp_kx[0] == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# ky weighting tests
# ---------------------------------------------------------------------------

class TestAveragesKyWeighting:

    def test_ky0_weight_is_one(self):
        """ky=0 mode contributes without factor 2."""
        nx, nky, nz = 4, 3, 8
        J = _uniform_J(nz)
        w = _ky_weight(nky)

        # Flux with unit signal only at ky=0
        flux = np.zeros((nx, nky, nz), dtype=complex)
        flux[:, 0, :] = 1.0

        _, sp_ky, _ = Spectra.averages(flux, J, ky_weight=w)
        # sp_ky[0] = sum over x of (flux[:, 0, :] * w[0] * J) = nx * 1 * 1
        assert sp_ky[0] == pytest.approx(float(nx), rel=1e-9)
        for k in range(1, nky):
            assert sp_ky[k] == pytest.approx(0.0, abs=1e-12)

    def test_ky_gt0_weight_is_two(self):
        """ky>0 modes are weighted by 2."""
        nx, nky, nz = 4, 3, 8
        J = _uniform_J(nz)
        w = _ky_weight(nky)

        flux = np.zeros((nx, nky, nz), dtype=complex)
        flux[:, 1, :] = 1.0   # signal only at ky index 1

        _, sp_ky, _ = Spectra.averages(flux, J, ky_weight=w)
        # sp_ky[1] = sum over x of (1.0 * 2.0 * J) = nx * 2 * 1
        assert sp_ky[1] == pytest.approx(float(nx) * 2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# z profile tests
# ---------------------------------------------------------------------------

class TestAveragesZProfile:

    def test_sum_z_is_sum_over_x_and_ky(self):
        """sum_z[iz] should equal sum_{x, ky} flux.real * ky_weight * J_norm[iz]."""
        nx, nky, nz = 4, 3, 8
        rng = np.random.default_rng(5)
        flux = rng.standard_normal((nx, nky, nz)) + 0j
        J = np.abs(rng.standard_normal(nz)) + 0.1
        J_norm = J / J.sum()
        w = _ky_weight(nky)

        _, _, sum_z = Spectra.averages(flux, J_norm, ky_weight=w)

        W = w[np.newaxis, :, np.newaxis]
        expected_sum_z = np.sum(flux.real * W * J_norm[np.newaxis, np.newaxis, :],
                                axis=(0, 1))
        np.testing.assert_allclose(sum_z, expected_sum_z, rtol=1e-10)

    def test_sum_z_without_weight(self):
        """Without ky_weight (W=1) sum_z is plain sum over x and ky."""
        nx, nky, nz = 3, 2, 5
        flux = np.ones((nx, nky, nz), dtype=complex)
        J = _uniform_J(nz)

        _, _, sum_z = Spectra.averages(flux, J, ky_weight=None)
        # Each z slice: sum over (nx * nky) * 1/nz
        expected = np.full(nz, nx * nky / nz)
        np.testing.assert_allclose(sum_z, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Odd nx edge-case
# ---------------------------------------------------------------------------

class TestAveragesOddNx:

    def test_odd_nx_folding_symmetry(self):
        """For odd nx modes 1..nx//2 should all be folded."""
        nx = 7   # odd: nx//2 = 3, nx2 = 4
        nky, nz = 2, 4
        J = _uniform_J(nz)
        flux = np.zeros((nx, nky, nz), dtype=complex)
        # Place unit signal at kx=1 and its mirror kx=nx-1=6
        flux[1, 0, :] = 1.0
        flux[nx - 1, 0, :] = 1.0

        sp_kx, _, _ = Spectra.averages(flux, J)
        assert sp_kx[1] == pytest.approx(2.0, abs=1e-10)
