"""Tests for genetools.diagnostics.spectra_global._compute_flux_yspectra."""

import numpy as np
import pytest

from genetools.diagnostics.spectra_global import _compute_flux_yspectra


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform(nx, nky, nz, value=1.0):
    return np.full((nx, nky, nz), value, dtype=complex)


def _uniform_J(nx, nz):
    """Uniform normalised Jacobian summing to 1 per row."""
    J = np.ones((nx, nz)) / nz
    return J


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestComputeFluxYspectraShape:

    def test_output_shape(self):
        nx, nky, nz = 6, 4, 16
        a = _uniform(nx, nky, nz)
        b = _uniform(nx, nky, nz)
        J = _uniform_J(nx, nz)
        out = _compute_flux_yspectra(a, b, C_xy=1.0, J_norm=J)
        assert out.shape == (nx, nky)

    def test_output_is_real_array(self):
        nx, nky, nz = 4, 3, 8
        rng = np.random.default_rng(0)
        a = rng.standard_normal((nx, nky, nz)) + 1j * rng.standard_normal((nx, nky, nz))
        b = rng.standard_normal((nx, nky, nz)) + 1j * rng.standard_normal((nx, nky, nz))
        J = _uniform_J(nx, nz)
        out = _compute_flux_yspectra(a, b, C_xy=1.0, J_norm=J)
        assert np.isrealobj(out)

    def test_single_ky_shape(self):
        nx, nky, nz = 5, 1, 8
        a = _uniform(nx, nky, nz)
        b = _uniform(nx, nky, nz)
        J = _uniform_J(nx, nz)
        out = _compute_flux_yspectra(a, b, C_xy=1.0, J_norm=J)
        assert out.shape == (nx, 1)


# ---------------------------------------------------------------------------
# ky=0 vs ky>0 weighting
# ---------------------------------------------------------------------------

class TestKyWeighting:

    def test_ky0_no_factor_2(self):
        """ky=0 mode contributes with factor 1."""
        nx, nky, nz = 4, 3, 8
        # Signal at ky=0 only
        a = np.zeros((nx, nky, nz), dtype=complex)
        b = np.zeros((nx, nky, nz), dtype=complex)
        a[:, 0, :] = 1.0
        b[:, 0, :] = 1.0
        J = _uniform_J(nx, nz)
        out = _compute_flux_yspectra(a, b, C_xy=1.0, J_norm=J)
        # out[:, 0] = sum_z(Re(conj(1)*1) * J) = sum_z(J) = 1 per x
        np.testing.assert_allclose(out[:, 0], 1.0, rtol=1e-12)
        # ky>0 contributions should be zero
        np.testing.assert_allclose(out[:, 1:], 0.0, atol=1e-14)

    def test_ky_gt0_factor_2(self):
        """ky>0 modes have factor 2 for Hermitian symmetry."""
        nx, nky, nz = 4, 3, 8
        a = np.zeros((nx, nky, nz), dtype=complex)
        b = np.zeros((nx, nky, nz), dtype=complex)
        a[:, 1, :] = 1.0
        b[:, 1, :] = 1.0
        J = _uniform_J(nx, nz)
        out = _compute_flux_yspectra(a, b, C_xy=1.0, J_norm=J)
        # out[:, 1] = 2 * sum_z(J) = 2 per x
        np.testing.assert_allclose(out[:, 1], 2.0, rtol=1e-12)
        # ky=0 and ky=2 should be zero
        np.testing.assert_allclose(out[:, 0], 0.0, atol=1e-14)
        np.testing.assert_allclose(out[:, 2], 0.0, atol=1e-14)

    def test_ky0_vs_ky1_ratio(self):
        """With same signal, ky=1 bin should be twice ky=0 bin."""
        nx, nky, nz = 4, 3, 8
        # Both ky=0 and ky=1 have the same signal
        a = np.ones((nx, nky, nz), dtype=complex)
        b = np.ones((nx, nky, nz), dtype=complex)
        J = _uniform_J(nx, nz)
        out = _compute_flux_yspectra(a, b, C_xy=1.0, J_norm=J)
        np.testing.assert_allclose(out[:, 1], 2.0 * out[:, 0], rtol=1e-10)


# ---------------------------------------------------------------------------
# C_xy division
# ---------------------------------------------------------------------------

class TestCxyDivision:

    def test_scalar_C_xy(self):
        """Scalar C_xy: result = raw / C_xy."""
        nx, nky, nz = 4, 2, 8
        a = _uniform(nx, nky, nz, 1.0)
        b = _uniform(nx, nky, nz, 1.0)
        J = _uniform_J(nx, nz)
        out_1 = _compute_flux_yspectra(a, b, C_xy=1.0, J_norm=J)
        out_2 = _compute_flux_yspectra(a, b, C_xy=2.0, J_norm=J)
        np.testing.assert_allclose(out_1, 2.0 * out_2, rtol=1e-12)

    def test_2d_C_xy_uniform_matches_scalar(self):
        """Uniform 2D C_xy array should give same result as scalar."""
        nx, nky, nz = 4, 2, 8
        C_scalar = 3.0
        C_arr    = np.full((nx, nz), C_scalar)
        a = _uniform(nx, nky, nz, 1.0)
        b = _uniform(nx, nky, nz, 1.0)
        J = _uniform_J(nx, nz)
        out_s = _compute_flux_yspectra(a, b, C_xy=C_scalar, J_norm=J)
        out_a = _compute_flux_yspectra(a, b, C_xy=C_arr,    J_norm=J)
        np.testing.assert_allclose(out_s, out_a, rtol=1e-10)

    def test_2d_C_xy_per_x_division(self):
        """Non-uniform C_xy: each x-row should be divided by mean_z(C_xy[x,:])."""
        nx, nky, nz = 4, 2, 8
        a = _uniform(nx, nky, nz, 1.0)
        b = _uniform(nx, nky, nz, 1.0)
        J = _uniform_J(nx, nz)
        # C_xy varies linearly with x
        C_arr = np.outer(np.arange(1, nx + 1, dtype=float), np.ones(nz))
        out = _compute_flux_yspectra(a, b, C_xy=C_arr, J_norm=J)
        # For each x: mean_z(C_xy[x,:]) = x+1
        for ix in range(nx):
            c_mean = np.mean(C_arr[ix, :])   # = ix+1
            # ky=0 contribution without C_xy: sum_z(J) = 1
            np.testing.assert_allclose(out[ix, 0], 1.0 / c_mean, rtol=1e-10)


# ---------------------------------------------------------------------------
# Numerical values
# ---------------------------------------------------------------------------

class TestNumericalValues:

    def test_zero_inputs_give_zero(self):
        nx, nky, nz = 3, 2, 5
        a = np.zeros((nx, nky, nz), dtype=complex)
        b = np.zeros((nx, nky, nz), dtype=complex)
        J = _uniform_J(nx, nz)
        out = _compute_flux_yspectra(a, b, C_xy=1.0, J_norm=J)
        np.testing.assert_array_equal(out, np.zeros((nx, nky)))

    def test_cross_correlation_is_real_part(self):
        """_compute_flux_yspectra uses Re(conj(a)*b)."""
        nx, nky, nz = 2, 2, 4
        # Use complex values and verify manually
        a = np.zeros((nx, nky, nz), dtype=complex)
        b = np.zeros((nx, nky, nz), dtype=complex)
        a[:, 0, :] = 1 + 1j
        b[:, 0, :] = 2 + 3j
        J = np.ones((nx, nz)) / nz
        out = _compute_flux_yspectra(a, b, C_xy=1.0, J_norm=J)
        # Re(conj(1+1j)*(2+3j)) = Re((1-1j)*(2+3j)) = Re(2+3j-2j-3j^2) = Re(5+j) = 5
        # FSA: sum over z * J = 5 * nz * (1/nz) = 5
        np.testing.assert_allclose(out[:, 0], 5.0, rtol=1e-10)

    def test_ky_sum_all_modes(self):
        """Uniform a=b=1 → ky=0 contributes 1, each ky>0 contributes 2."""
        nx, nky, nz = 3, 4, 8
        a = _uniform(nx, nky, nz, 1.0)
        b = _uniform(nx, nky, nz, 1.0)
        J = _uniform_J(nx, nz)
        out = _compute_flux_yspectra(a, b, C_xy=1.0, J_norm=J)
        # ky=0: 1, ky=1,2,3: 2 each
        np.testing.assert_allclose(out[:, 0], 1.0, rtol=1e-12)
        np.testing.assert_allclose(out[:, 1:], 2.0, rtol=1e-12)

    def test_jacobian_weighting(self):
        """Concentrate J weight at iz=0, verify result uses that slice."""
        nx, nky, nz = 2, 2, 8
        a = np.zeros((nx, nky, nz), dtype=complex)
        b = np.zeros((nx, nky, nz), dtype=complex)
        # Set z-varying signal: only iz=0 is nonzero in a
        a[:, 0, 0] = 2.0
        b[:, 0, 0] = 3.0
        # Put all J weight at iz=0
        J = np.zeros((nx, nz))
        J[:, 0] = 1.0
        out = _compute_flux_yspectra(a, b, C_xy=1.0, J_norm=J)
        # Re(conj(2)*3) = 6, times J[iz=0]=1
        np.testing.assert_allclose(out[:, 0], 6.0, rtol=1e-10)
