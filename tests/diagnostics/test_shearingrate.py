"""Tests for genetools.diagnostics.shearingrate."""

import numpy as np
import pytest

from genetools.diagnostics.shearingrate import _central_diff, compute_exb


# ---------------------------------------------------------------------------
# _central_diff
# ---------------------------------------------------------------------------

class TestCentralDiff:
    """_central_diff(f) — finite difference helper."""

    # --- edge-case regressions (B6) ---

    def test_length_1_returns_zero(self):
        """Regression B6: len=1 must return [0] without crashing."""
        f = np.array([5.0])
        d = _central_diff(f)
        assert d.shape == (1,)
        assert d[0] == pytest.approx(0.0)

    def test_length_2_forward_diff(self):
        """len=2: both entries equal the forward difference."""
        f = np.array([3.0, 7.0])
        d = _central_diff(f)
        assert d.shape == (2,)
        expected = 7.0 - 3.0   # = 4.0
        np.testing.assert_allclose(d, [expected, expected], rtol=1e-12)

    # --- interior accuracy ---

    def test_constant_function_zero_derivative(self):
        f = np.ones(10) * 3.5
        d = _central_diff(f)
        np.testing.assert_allclose(d, 0.0, atol=1e-14)

    def test_linear_function_exact(self):
        """2nd-order central diff is exact for linear functions."""
        x = np.linspace(0, 1, 11)
        f = 2.5 * x + 1.0
        d = _central_diff(f)
        h = x[1] - x[0]
        # All interior points exact; boundaries use one-sided formula
        np.testing.assert_allclose(d[1:-1] / h, 2.5, rtol=1e-10)

    def test_quadratic_interior_exact(self):
        """Central differences are exact for quadratics at interior points."""
        x = np.linspace(0.0, 1.0, 51)
        # f = a*x^2 + b*x + c  => f' = 2a*x + b
        a, b, c = 3.0, -1.5, 2.0
        f = a * x**2 + b * x + c
        d = _central_diff(f)
        h = x[1] - x[0]
        expected_interior = (2.0 * a * x + b)[1:-1]
        np.testing.assert_allclose(d[1:-1] / h, expected_interior, rtol=1e-10)

    # --- output properties ---

    def test_same_shape_as_input(self):
        f = np.linspace(0, 5, 20)
        d = _central_diff(f)
        assert d.shape == f.shape

    def test_output_dtype_preserved(self):
        f = np.linspace(0, 1, 5, dtype=np.float32)
        d = _central_diff(f)
        # Implementation uses np.empty_like, so dtype is preserved
        assert d.dtype == f.dtype

    def test_boundary_values_use_one_sided_formula(self):
        """Boundaries use the explicit one-sided 2nd-order formula."""
        x = np.linspace(0.0, 1.0, 10)
        f = x**2      # f' = 2x
        d = _central_diff(f)
        h = x[1] - x[0]
        # Forward at x=0:  (-3f[0] + 4f[1] - f[2])/2 / h  ≈ 2*x[0]
        expected_0 = (-3 * f[0] + 4 * f[1] - f[2]) * 0.5
        assert d[0] == pytest.approx(expected_0, rel=1e-10)
        # Backward at x[-1]:  (3f[-1] - 4f[-2] + f[-3])/2 / h  ≈ 2*x[-1]
        expected_n = (3 * f[-1] - 4 * f[-2] + f[-3]) * 0.5
        assert d[-1] == pytest.approx(expected_n, rel=1e-10)


# ---------------------------------------------------------------------------
# compute_exb — local geometry
# ---------------------------------------------------------------------------

def _make_local_params(nx=8, nky=3, nz=16):
    return {
        "general": {"x_local": True},
        "box": {"nx0": nx, "nky0": nky, "nz0": nz},
    }


def _make_local_geom(nx=8, nz=16, C_xy=1.0):
    J = np.ones(nz)
    metric = {"C_xy": C_xy}
    return {"Jacobian": J, "metric": metric}


def _make_local_coord(nx=8, nky=3):
    kx = np.fft.fftfreq(nx) * 2 * np.pi
    return {"kx": kx}


def _make_phi(nx=8, nky=3, nz=16, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((nx, nky, nz))
            + 1j * rng.standard_normal((nx, nky, nz)))


class TestComputeExbLocal:
    """compute_exb with local geometry."""

    def test_returns_dict_with_all_expected_keys(self):
        nx, nky, nz = 8, 3, 16
        phi = _make_phi(nx, nky, nz)
        params = _make_local_params(nx, nky, nz)
        geom   = _make_local_geom(nx, nz)
        coord  = _make_local_coord(nx, nky)

        result = compute_exb(phi, params, geom, coord)
        expected_keys = {
            "phi_zonal_fsavg", "phi_zonal_x",
            "E_r", "v_ExB", "omega_ExB", "shearing_rms",
        }
        assert expected_keys == set(result.keys())

    def test_phi_zonal_fsavg_shape(self):
        nx, nky, nz = 8, 3, 16
        phi = _make_phi(nx, nky, nz)
        result = compute_exb(phi, _make_local_params(nx, nky, nz),
                             _make_local_geom(nx, nz),
                             _make_local_coord(nx, nky))
        assert result["phi_zonal_fsavg"].shape == (nx,)

    @pytest.mark.parametrize("key", ["phi_zonal_x", "E_r", "v_ExB", "omega_ExB"])
    def test_real_arrays_shape_nx(self, key):
        nx, nky, nz = 12, 4, 8
        phi = _make_phi(nx, nky, nz)
        result = compute_exb(phi, _make_local_params(nx, nky, nz),
                             _make_local_geom(nx, nz),
                             _make_local_coord(nx, nky))
        assert result[key].shape == (nx,)

    def test_shearing_rms_is_scalar(self):
        nx, nky, nz = 8, 3, 16
        phi = _make_phi(nx, nky, nz)
        result = compute_exb(phi, _make_local_params(nx, nky, nz),
                             _make_local_geom(nx, nz),
                             _make_local_coord(nx, nky))
        assert np.ndim(result["shearing_rms"]) == 0
        assert np.isfinite(result["shearing_rms"])

    def test_shearing_rms_nonnegative(self):
        nx, nky, nz = 8, 3, 16
        phi = _make_phi(nx, nky, nz)
        result = compute_exb(phi, _make_local_params(nx, nky, nz),
                             _make_local_geom(nx, nz),
                             _make_local_coord(nx, nky))
        assert result["shearing_rms"] >= 0.0

    def test_zero_phi_gives_zero_fields(self):
        """All-zero phi should yield all-zero ExB quantities."""
        nx, nky, nz = 8, 3, 16
        phi = np.zeros((nx, nky, nz), dtype=complex)
        result = compute_exb(phi, _make_local_params(nx, nky, nz),
                             _make_local_geom(nx, nz),
                             _make_local_coord(nx, nky))
        np.testing.assert_allclose(result["phi_zonal_x"],  0.0, atol=1e-14)
        np.testing.assert_allclose(result["E_r"],          0.0, atol=1e-14)
        np.testing.assert_allclose(result["omega_ExB"],    0.0, atol=1e-14)
        assert result["shearing_rms"] == pytest.approx(0.0, abs=1e-14)

    def test_outputs_are_real(self):
        nx, nky, nz = 8, 3, 16
        phi = _make_phi(nx, nky, nz)
        result = compute_exb(phi, _make_local_params(nx, nky, nz),
                             _make_local_geom(nx, nz),
                             _make_local_coord(nx, nky))
        for key in ("phi_zonal_x", "E_r", "v_ExB", "omega_ExB"):
            assert np.isrealobj(result[key]), f"{key} should be real"


# ---------------------------------------------------------------------------
# compute_exb — global geometry
# ---------------------------------------------------------------------------

def _make_global_params(nx=10, nky=3, nz=8):
    return {
        "general": {"x_local": False},
        "box": {"nx0": nx, "nky0": nky, "nz0": nz},
    }


def _make_global_geom(nx=10, nz=8):
    J = np.ones((nx, nz))
    metric = {"C_xy": np.ones((nx, nz))}
    q = np.linspace(1.4, 2.0, nx)
    return {"Jacobian": J, "metric": metric, "profiles": {"q": q}}


def _make_global_coord(nx=10, dx=0.1):
    # Start from dx/2 so that x[0] > 0, avoiding division-by-zero in omega_ExB
    x = (np.arange(nx) + 0.5) * dx
    return {"dx": dx, "x": x}


class TestComputeExbGlobal:

    def test_returns_dict_with_expected_keys(self):
        nx, nky, nz = 10, 3, 8
        phi = _make_phi(nx, nky, nz)
        result = compute_exb(phi,
                             _make_global_params(nx, nky, nz),
                             _make_global_geom(nx, nz),
                             _make_global_coord(nx))
        expected_keys = {
            "phi_zonal_fsavg", "phi_zonal_x",
            "E_r", "v_ExB", "omega_ExB", "shearing_rms",
        }
        assert expected_keys == set(result.keys())

    def test_phi_zonal_fsavg_is_none_for_global(self):
        nx, nky, nz = 10, 3, 8
        phi = _make_phi(nx, nky, nz)
        result = compute_exb(phi,
                             _make_global_params(nx, nky, nz),
                             _make_global_geom(nx, nz),
                             _make_global_coord(nx))
        assert result["phi_zonal_fsavg"] is None

    @pytest.mark.parametrize("key", ["phi_zonal_x", "E_r", "v_ExB", "omega_ExB"])
    def test_real_arrays_shape_nx(self, key):
        nx, nky, nz = 10, 3, 8
        phi = _make_phi(nx, nky, nz)
        result = compute_exb(phi,
                             _make_global_params(nx, nky, nz),
                             _make_global_geom(nx, nz),
                             _make_global_coord(nx))
        assert result[key].shape == (nx,)

    def test_zero_phi_gives_zero_fields(self):
        nx, nky, nz = 10, 3, 8
        phi = np.zeros((nx, nky, nz), dtype=complex)
        result = compute_exb(phi,
                             _make_global_params(nx, nky, nz),
                             _make_global_geom(nx, nz),
                             _make_global_coord(nx))
        np.testing.assert_allclose(result["phi_zonal_x"], 0.0, atol=1e-14)
        np.testing.assert_allclose(result["E_r"],         0.0, atol=1e-14)
