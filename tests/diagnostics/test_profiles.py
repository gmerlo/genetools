"""Tests for genetools.diagnostics.profiles."""

import numpy as np
import pytest

from genetools.diagnostics.profiles import (
    _finite_diff,
    _compute_fsa_profiles,
    Profiles,
)


# ---------------------------------------------------------------------------
# _finite_diff
# ---------------------------------------------------------------------------

class TestFiniteDiff:
    """_finite_diff(f) — 2nd-order central FD with 1st-order boundaries."""

    def test_length_1_returns_zero(self):
        f = np.array([42.0])
        d = _finite_diff(f)
        assert d.shape == (1,)
        assert d[0] == pytest.approx(0.0)

    def test_length_2_both_equal_forward_diff(self):
        f = np.array([1.0, 5.0])
        d = _finite_diff(f)
        assert d.shape == (2,)
        expected = 5.0 - 1.0
        np.testing.assert_allclose(d, [expected, expected], rtol=1e-12)

    def test_length_2_first_order_boundary(self):
        """1st-order boundary: d[0] = f[1]-f[0], d[-1] = f[-1]-f[-2]."""
        f = np.array([3.0, 7.0])
        d = _finite_diff(f)
        assert d[0]  == pytest.approx(f[1] - f[0])
        assert d[-1] == pytest.approx(f[-1] - f[-2])

    def test_constant_function_zero_derivative(self):
        f = np.full(8, 2.5)
        d = _finite_diff(f)
        np.testing.assert_allclose(d, 0.0, atol=1e-14)

    def test_linear_function_interior_exact(self):
        """Central differences exact for linear f at interior points."""
        n = 11
        f = np.linspace(0.0, 10.0, n)   # step h=1.0
        d = _finite_diff(f)
        h = f[1] - f[0]
        # Interior: (f[i+1]-f[i-1])/2 / h should equal slope/h * h = slope
        np.testing.assert_allclose(d[1:-1] / h, 1.0, rtol=1e-10)

    def test_boundary_is_first_order(self):
        """Boundary uses simple forward/backward (1st-order) differences."""
        f = np.array([1.0, 4.0, 9.0, 16.0, 25.0])   # f = (i+1)^2
        d = _finite_diff(f)
        assert d[0]  == pytest.approx(f[1] - f[0])
        assert d[-1] == pytest.approx(f[-1] - f[-2])

    def test_interior_central_difference(self):
        """Interior: (f[i+1] - f[i-1]) / 2."""
        f = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        d = _finite_diff(f)
        for i in range(1, len(f) - 1):
            expected = (f[i + 1] - f[i - 1]) * 0.5
            assert d[i] == pytest.approx(expected, rel=1e-12)

    def test_same_shape_as_input(self):
        f = np.arange(15, dtype=float)
        d = _finite_diff(f)
        assert d.shape == f.shape

    def test_dtype_preserved(self):
        f = np.linspace(0, 1, 5, dtype=np.float32)
        d = _finite_diff(f)
        assert d.dtype == f.dtype

    def test_length_5(self):
        """Explicit check for all five entries of a length-5 array."""
        f = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
        d = _finite_diff(f)
        assert d[0]  == pytest.approx(f[1] - f[0])         # 1st-order forward
        assert d[1]  == pytest.approx((f[2] - f[0]) * 0.5) # central
        assert d[2]  == pytest.approx((f[3] - f[1]) * 0.5) # central
        assert d[3]  == pytest.approx((f[4] - f[2]) * 0.5) # central
        assert d[-1] == pytest.approx(f[-1] - f[-2])       # 1st-order backward


# ---------------------------------------------------------------------------
# _compute_fsa_profiles
# ---------------------------------------------------------------------------

def _make_uniform_moments(nx, nky, nz, values=None):
    """
    Create 6 moment arrays (dens, T_par, T_perp, q_par, q_perp, u_par),
    each with shape (nx, nky, nz). values is a list of 6 floats.
    """
    if values is None:
        values = [1.0, 1.0, 1.0, 0.0, 0.0, 0.5]
    moments = []
    for v in values:
        arr = np.full((nx, nky, nz), v, dtype=complex)
        moments.append(arr)
    return moments


class TestComputeFsaProfiles:
    """_compute_fsa_profiles(moments, x_local, J_norm, nx)"""

    def test_returns_three_arrays(self):
        nx, nky, nz = 8, 3, 16
        moments = _make_uniform_moments(nx, nky, nz)
        J_norm = np.ones((1, nz)) / nz
        T_fsa, n_fsa, u_fsa = _compute_fsa_profiles(moments, x_local=False,
                                                     J_norm=J_norm, nx=nx)
        assert T_fsa.shape == (nx,)
        assert n_fsa.shape == (nx,)
        assert u_fsa.shape == (nx,)

    def test_uniform_global_temperature(self):
        """Uniform T_par=1, T_perp=1 → T = 1/3+2/3=1, FSA should be 1."""
        nx, nky, nz = 4, 2, 8
        T_par_val  = 1.0
        T_perp_val = 1.0
        expected_T = (1.0/3.0) * T_par_val + (2.0/3.0) * T_perp_val  # = 1.0
        moments = _make_uniform_moments(nx, nky, nz,
                                        values=[1.0, T_par_val, T_perp_val,
                                                0.0, 0.0, 0.0])
        J_norm = np.ones((1, nz)) / nz    # uniform, sums to 1
        T_fsa, _, _ = _compute_fsa_profiles(moments, x_local=False,
                                             J_norm=J_norm, nx=nx)
        np.testing.assert_allclose(T_fsa, expected_T, rtol=1e-10)

    def test_uniform_global_density(self):
        nx, nky, nz = 4, 2, 8
        dens_val = 2.5
        moments = _make_uniform_moments(nx, nky, nz,
                                        values=[dens_val, 0.0, 0.0,
                                                0.0, 0.0, 0.0])
        J_norm = np.ones((1, nz)) / nz
        _, n_fsa, _ = _compute_fsa_profiles(moments, x_local=False,
                                            J_norm=J_norm, nx=nx)
        np.testing.assert_allclose(n_fsa, dens_val, rtol=1e-10)

    def test_temperature_combines_par_and_perp(self):
        """T = (1/3)*T_par + (2/3)*T_perp — coefficient test."""
        nx, nky, nz = 4, 2, 8
        T_par_val  = 3.0
        T_perp_val = 6.0
        expected_T = (1.0/3.0) * T_par_val + (2.0/3.0) * T_perp_val   # = 5.0
        moments = _make_uniform_moments(nx, nky, nz,
                                        values=[0.0, T_par_val, T_perp_val,
                                                0.0, 0.0, 0.0])
        J_norm = np.ones((1, nz)) / nz
        T_fsa, _, _ = _compute_fsa_profiles(moments, x_local=False,
                                            J_norm=J_norm, nx=nx)
        np.testing.assert_allclose(T_fsa, expected_T, rtol=1e-10)

    def test_ky0_extracted_not_ky1(self):
        """FSA should use ky=0 slice, not ky=1."""
        nx, nky, nz = 4, 3, 8
        moments = []
        for _ in range(6):
            arr = np.zeros((nx, nky, nz), dtype=complex)
            arr[:, 0, :] = 1.0    # ky=0 = 1
            arr[:, 1, :] = 99.0   # ky=1 should be ignored
            moments.append(arr)
        J_norm = np.ones((1, nz)) / nz
        T_fsa, n_fsa, _ = _compute_fsa_profiles(moments, x_local=False,
                                                 J_norm=J_norm, nx=nx)
        # If FSA uses ky=0 correctly, T = 1/3+2/3 = 1 and n = 1 (not 99)
        np.testing.assert_allclose(n_fsa, 1.0, rtol=1e-10)

    def test_local_geometry_returns_correct_shape(self):
        """x_local=True: IFFT is applied; output should still be (nx,)."""
        nx, nky, nz = 8, 3, 16
        moments = _make_uniform_moments(nx, nky, nz)
        # Local: J_norm has shape (1, nz)
        J_norm = np.ones((1, nz)) / nz
        T_fsa, n_fsa, u_fsa = _compute_fsa_profiles(moments, x_local=True,
                                                     J_norm=J_norm, nx=nx)
        assert T_fsa.shape == (nx,)
        assert n_fsa.shape == (nx,)
        assert u_fsa.shape == (nx,)

    def test_jacobian_weighting(self):
        """FSA is weighted by J_norm — concentrate weight at iz=0 to test."""
        nx, nky, nz = 4, 2, 8
        # Moments: ky=0 real part varies with z
        moments = []
        for _ in range(6):
            arr = np.zeros((nx, nky, nz), dtype=complex)
            arr[:, 0, :] = np.arange(nz, dtype=float)   # z-varying
            moments.append(arr)
        # J_norm puts all weight on iz=0
        J_norm = np.zeros((1, nz))
        J_norm[0, 0] = 1.0
        T_fsa, _, _ = _compute_fsa_profiles(moments, x_local=False,
                                            J_norm=J_norm, nx=nx)
        # T = (1/3+2/3)*0 = 0 (value at z=0 is 0)
        np.testing.assert_allclose(T_fsa, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Profiles.build_background — local geometry
# ---------------------------------------------------------------------------

class TestProfilesBuildBackgroundLocal:
    """Profiles.build_background for x_local=True."""

    def _make_params_local(self, omt=2.0, omn=1.5):
        return {
            "general": {"x_local": True},
            "species": [{"name": "ions", "omt": omt, "omn": omn}],
            "geometry": {"rhostar": 1.0 / 500, "minor_r": 1.0},
            "units": {},
        }

    def _make_coords(self, nx=20):
        return {"x": np.linspace(-5.0, 5.0, nx)}

    def test_returns_background_for_species(self):
        params = self._make_params_local()
        coords = self._make_coords()
        bg = Profiles.build_background(params, coords)
        assert "ions" in bg

    def test_background_keys(self):
        params = self._make_params_local()
        coords = self._make_coords()
        bg = Profiles.build_background(params, coords)
        assert set(bg["ions"].keys()) == {
            "T_back", "n_back", "u_back", "omt_back", "omn_back"
        }

    def test_background_shape(self):
        nx = 20
        params = self._make_params_local()
        coords = self._make_coords(nx=nx)
        bg = Profiles.build_background(params, coords)
        for key in ("T_back", "n_back", "u_back", "omt_back", "omn_back"):
            assert bg["ions"][key].shape == (nx,), \
                f"{key} has wrong shape"

    def test_T_back_at_x0_is_one(self):
        """At x=0, T_back = 1 - 0*omt*... = 1."""
        nx = 21  # odd so x=0 is at index nx//2
        params = self._make_params_local(omt=3.0)
        coords = {"x": np.linspace(-5.0, 5.0, nx)}
        bg = Profiles.build_background(params, coords)
        mid = nx // 2
        assert bg["ions"]["T_back"][mid] == pytest.approx(1.0, rel=1e-9)

    def test_u_back_is_zero(self):
        params = self._make_params_local()
        coords = self._make_coords()
        bg = Profiles.build_background(params, coords)
        np.testing.assert_array_equal(bg["ions"]["u_back"],
                                      np.zeros_like(bg["ions"]["u_back"]))

    def test_omt_back_is_constant(self):
        omt = 2.7
        params = self._make_params_local(omt=omt)
        coords = self._make_coords()
        bg = Profiles.build_background(params, coords)
        np.testing.assert_allclose(bg["ions"]["omt_back"], omt, rtol=1e-12)

    def test_global_requires_equilibrium_profiles(self):
        """Global geometry without equilibrium_profiles raises ValueError."""
        params = {
            "general": {"x_local": False},
            "species": [{"name": "ions", "temp": 1.0, "dens": 1.0}],
            "geometry": {},
            "units": {},
        }
        coords = {"x": np.linspace(0, 1, 10)}
        with pytest.raises(ValueError, match="Equilibrium profiles"):
            Profiles.build_background(params, coords)
