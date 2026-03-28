"""Tests for genetools.diagnostics.fluxes2d."""

import numpy as np
import pytest

from genetools.diagnostics.fluxes2d import _compute_flux, _compute_velocity


# ---------------------------------------------------------------------------
# _compute_flux
# ---------------------------------------------------------------------------

class TestComputeFlux:
    """_compute_flux(a, b) — ky-summed Re(a * conj(b)) with Hermitian weighting."""

    def test_output_shape(self):
        nx, nky, nz = 6, 4, 16
        a = np.ones((nx, nky, nz), dtype=complex)
        b = np.ones((nx, nky, nz), dtype=complex)
        out = _compute_flux(a, b)
        assert out.shape == (nx, nz)

    def test_output_is_real(self):
        nx, nky, nz = 4, 3, 8
        rng = np.random.default_rng(0)
        a = rng.standard_normal((nx, nky, nz)) + 1j * rng.standard_normal((nx, nky, nz))
        b = rng.standard_normal((nx, nky, nz)) + 1j * rng.standard_normal((nx, nky, nz))
        out = _compute_flux(a, b)
        assert np.isrealobj(out)

    def test_single_ky_no_factor_2(self):
        """nky=1: only ky=0 present — result = Re(a[:,0,:]*conj(b[:,0,:]))."""
        nx, nky, nz = 4, 1, 8
        a = np.full((nx, nky, nz), 2.0 + 0j)
        b = np.full((nx, nky, nz), 3.0 + 0j)
        out = _compute_flux(a, b)
        # Re(2 * conj(3)) = 6, no factor of 2
        np.testing.assert_allclose(out, 6.0, rtol=1e-12)

    def test_ky0_only_contribution_nky_gt1(self):
        """With signal only at ky=0 there is no factor-2 doubling."""
        nx, nky, nz = 4, 3, 8
        a = np.zeros((nx, nky, nz), dtype=complex)
        b = np.zeros((nx, nky, nz), dtype=complex)
        a[:, 0, :] = 2.0
        b[:, 0, :] = 3.0
        out = _compute_flux(a, b)
        np.testing.assert_allclose(out, 6.0, rtol=1e-12)

    def test_ky_gt0_gets_factor_2(self):
        """Signal at ky=1 (and ky=0 zero) → result = 2*Re(a*conj(b))."""
        nx, nky, nz = 4, 3, 8
        a = np.zeros((nx, nky, nz), dtype=complex)
        b = np.zeros((nx, nky, nz), dtype=complex)
        a[:, 1, :] = 1.0
        b[:, 1, :] = 1.0
        out = _compute_flux(a, b)
        # 2 * Re(1 * conj(1)) = 2
        np.testing.assert_allclose(out, 2.0, rtol=1e-12)

    def test_multiple_ky_modes_summed(self):
        """Result sums ky=0 (×1) + ky>0 (×2) contributions."""
        nx, nky, nz = 4, 3, 8
        a = np.ones((nx, nky, nz), dtype=complex)
        b = np.ones((nx, nky, nz), dtype=complex)
        out = _compute_flux(a, b)
        # ky=0: 1, ky=1: 2, ky=2: 2 → total = 1 + 2 + 2 = 5
        np.testing.assert_allclose(out, 5.0, rtol=1e-12)

    def test_hermitian_symmetry_re_a_conj_b_equals_re_conj_a_b(self):
        """Re(a * conj(b)) == Re(conj(a) * b) — symmetry check."""
        nx, nky, nz = 5, 3, 7
        rng = np.random.default_rng(11)
        a = rng.standard_normal((nx, nky, nz)) + 1j * rng.standard_normal((nx, nky, nz))
        b = rng.standard_normal((nx, nky, nz)) + 1j * rng.standard_normal((nx, nky, nz))

        # The function computes: Re(conj(a[:,1:,:])*b[:,1:,:]) for ky>0
        # Here we verify: Re(a*conj(b)) == Re(conj(a)*b)
        out1 = _compute_flux(a, b)
        # swap and reverse role: Re(b * conj(a))
        out2 = _compute_flux(b, a)
        # Re(a*conj(b)) = Re(b*conj(a)) because Re(z) = Re(conj(z))
        np.testing.assert_allclose(out1, out2, rtol=1e-12)

    def test_all_zeros_gives_zero(self):
        nx, nky, nz = 3, 2, 5
        a = np.zeros((nx, nky, nz), dtype=complex)
        b = np.zeros((nx, nky, nz), dtype=complex)
        out = _compute_flux(a, b)
        np.testing.assert_array_equal(out, np.zeros((nx, nz)))

    def test_complex_cross_correlation(self):
        """Verify the exact cross-correlation formula for a small case.

        Array layout: shape (nx, nky, nz).  Here nx=2, nky=1, nz=2.
        a[:, 0, 0] = 1+2j,  a[:, 0, 1] = 3+4j  (z-points 0 and 1)
        b[:, 0, 0] = 5+6j,  b[:, 0, 1] = 7+8j

        Re((1+2j)*conj(5+6j)) = Re((1+2j)*(5-6j))
                               = Re(5 - 6j + 10j - 12j²) = 5 + 12 = 17
        Re((3+4j)*conj(7+8j)) = Re((3+4j)*(7-8j))
                               = Re(21 - 24j + 28j - 32j²) = 21 + 32 = 53
        """
        nx, nky, nz = 2, 1, 2
        a = np.array([1 + 2j, 3 + 4j]).reshape(1, 1, 2).repeat(nx, 0)
        b = np.array([5 + 6j, 7 + 8j]).reshape(1, 1, 2).repeat(nx, 0)
        out = _compute_flux(a, b)
        expected = np.array([[17.0, 53.0]] * nx)
        np.testing.assert_allclose(out, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# _compute_velocity
# ---------------------------------------------------------------------------

class TestComputeVelocity:
    """_compute_velocity — ExB or A-par velocity from a field."""

    def _make_field(self, nx=8, nky=3, nz=8, seed=0):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((nx, nky, nz))
                + 1j * rng.standard_normal((nx, nky, nz)))

    def _make_ky(self, nky=3):
        return np.arange(nky, dtype=float) * 0.3

    def test_output_shape_local(self):
        nx, nky, nz = 8, 3, 8
        field = self._make_field(nx, nky, nz)
        ky    = self._make_ky(nky)
        vel   = _compute_velocity(field, ky, C_xy=1.0, x_local=True,
                                   nx=nx, sign=-1.0)
        assert vel.shape == (nx, nky, nz)

    def test_output_shape_global(self):
        nx, nky, nz = 8, 3, 8
        field = self._make_field(nx, nky, nz)
        ky    = self._make_ky(nky)
        C_xy  = np.ones((nx, nz))
        vel   = _compute_velocity(field, ky, C_xy=C_xy, x_local=False,
                                   nx=nx, sign=-1.0)
        assert vel.shape == (nx, nky, nz)

    def test_local_applies_ifft(self):
        """For local geometry the field is IFFT'd to x-space first."""
        nx, nky, nz = 8, 1, 4
        # Use a pure x=0 mode (constant in x-space)
        field = np.zeros((nx, nky, nz), dtype=complex)
        field[0, 0, :] = 1.0   # kx=0 → constant in real space
        ky = np.array([0.5])   # single ky mode

        vel = _compute_velocity(field, ky, C_xy=1.0, x_local=True,
                                nx=nx, sign=-1.0)
        # IFFT of (1,0,...,0) = (1,1,...,1)/nx *nx = 1 in real space
        # Then multiply by sign * i * ky = -1 * i * 0.5 = -0.5i
        expected_real_x = np.ones(nz, dtype=complex)   # IFFT * nx
        expected = -1.0 * 1j * 0.5 * expected_real_x / 1.0  # sign*i*ky*v/C_xy
        np.testing.assert_allclose(vel[0, 0, :], expected, rtol=1e-10)

    def test_local_ky0_gives_zero_velocity(self):
        """ky=0 → velocity is zero regardless of field amplitude."""
        nx, nky, nz = 8, 3, 8
        field = self._make_field(nx, nky, nz)
        ky    = np.array([0.0, 0.3, 0.6])
        vel   = _compute_velocity(field, ky, C_xy=1.0, x_local=True,
                                   nx=nx, sign=-1.0)
        np.testing.assert_allclose(vel[:, 0, :], 0.0, atol=1e-14)

    def test_global_ky0_gives_zero_velocity(self):
        """ky=0 → velocity is zero for global geometry too."""
        nx, nky, nz = 8, 3, 8
        field = self._make_field(nx, nky, nz)
        ky    = np.array([0.0, 0.3, 0.6])
        C_xy  = np.ones((nx, nz))
        vel   = _compute_velocity(field, ky, C_xy=C_xy, x_local=False,
                                   nx=nx, sign=-1.0)
        np.testing.assert_allclose(vel[:, 0, :], 0.0, atol=1e-14)

    def test_sign_inverts_velocity(self):
        """sign=-1 and sign=+1 give opposite velocities."""
        nx, nky, nz = 8, 3, 8
        field = self._make_field(nx, nky, nz)
        ky    = self._make_ky(nky)
        vel_neg = _compute_velocity(field, ky, C_xy=1.0, x_local=True,
                                     nx=nx, sign=-1.0)
        vel_pos = _compute_velocity(field, ky, C_xy=1.0, x_local=True,
                                     nx=nx, sign=+1.0)
        np.testing.assert_allclose(vel_neg, -vel_pos, rtol=1e-12)

    def test_scalar_vs_2d_C_xy_global(self):
        """Uniform 2D C_xy array should give same result as scalar C_xy."""
        nx, nky, nz = 6, 3, 8
        field = self._make_field(nx, nky, nz)
        ky    = self._make_ky(nky)
        C_scalar = 2.0
        C_arr    = np.full((nx, nz), C_scalar)

        vel_scalar = _compute_velocity(field, ky, C_xy=C_scalar,
                                        x_local=False, nx=nx)
        vel_arr    = _compute_velocity(field, ky, C_xy=C_arr,
                                        x_local=False, nx=nx)
        np.testing.assert_allclose(vel_scalar, vel_arr, rtol=1e-12)

    def test_local_c_xy_scales_velocity(self):
        """Doubling C_xy halves the velocity."""
        nx, nky, nz = 8, 3, 8
        field = self._make_field(nx, nky, nz)
        ky    = self._make_ky(nky)
        vel1  = _compute_velocity(field, ky, C_xy=1.0, x_local=True, nx=nx)
        vel2  = _compute_velocity(field, ky, C_xy=2.0, x_local=True, nx=nx)
        np.testing.assert_allclose(vel1, 2.0 * vel2, rtol=1e-12)
