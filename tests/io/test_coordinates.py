"""Tests for genetools.io.coordinates."""

import numpy as np
import pytest

from genetools.io.coordinates import load_coord_single_run, Coordinates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_params(
    nx0=4,
    nky0=3,
    nz0=8,
    lx=10.0,
    kymin=0.5,
    lv=3.0,
    nv0=16,
    nw0=4,
    lw=2.0,
    mu_grid_type="gau_lag",
    n_pol=1,
    edge_opt=0,
    rhostar=0.01,
    x_local=True,
    collision_op="none",
    arakawa_zv=True,
):
    return {
        "general": {
            "x_local": x_local,
            "collision_op": collision_op,
            "arakawa_zv": arakawa_zv,
        },
        "box": {
            "nx0": nx0,
            "nky0": nky0,
            "nz0": nz0,
            "lx": lx,
            "kymin": kymin,
            "lv": lv,
            "nv0": nv0,
            "nw0": nw0,
            "lw": lw,
            "mu_grid_type": mu_grid_type,
        },
        "geometry": {
            "n_pol": n_pol,
            "edge_opt": edge_opt,
            "rhostar": rhostar,
        },
    }


def run(params):
    """Convenience wrapper: folder and file_number are unused."""
    return load_coord_single_run(None, None, params)


# ---------------------------------------------------------------------------
# kx — local geometry
# ---------------------------------------------------------------------------

class TestKxLocal:
    def test_even_nx_shape(self):
        coords = run(make_params(nx0=4, lx=10.0))
        assert coords["kx"].shape == (4,)

    def test_even_nx_modes(self):
        """nx=4, lx=10: kxmin=2pi/10, modes should be [0, 1, 2, -1]*kxmin."""
        coords = run(make_params(nx0=4, lx=10.0))
        kxmin = 2 * np.pi / 10.0
        expected = np.array([0, 1, 2, -1]) * kxmin
        np.testing.assert_allclose(coords["kx"], expected)

    def test_even_nx_is_fft_ordered(self):
        """Positive modes first, then negative (FFT convention)."""
        coords = run(make_params(nx0=6, lx=6.0))
        kx = coords["kx"]
        # First half+1 should be non-negative, rest negative
        half = 6 // 2
        assert np.all(kx[:half + 1] >= 0)
        assert np.all(kx[half + 1:] < 0)

    def test_odd_nx_shape(self):
        coords = run(make_params(nx0=5, lx=10.0))
        assert coords["kx"].shape == (5,)

    def test_odd_nx_modes(self):
        """nx=5, lx=10: modes [0,1,2,-2,-1]*kxmin."""
        coords = run(make_params(nx0=5, lx=10.0))
        kxmin = 2 * np.pi / 10.0
        expected = np.array([0, 1, 2, -2, -1]) * kxmin
        np.testing.assert_allclose(coords["kx"], expected)

    def test_nx1_kx_shape(self):
        """Regression B5: nx=1 must produce 1-D kx of shape (1,), not (1,1)."""
        coords = run(make_params(nx0=1))
        assert coords["kx"].ndim == 1
        assert coords["kx"].shape == (1,)

    def test_nx1_x_value(self):
        """Regression B5: nx=1 must give x=[0.0]."""
        coords = run(make_params(nx0=1))
        np.testing.assert_array_equal(coords["x"], np.array([0.0]))

    def test_x_grid_length(self):
        coords = run(make_params(nx0=4, lx=10.0))
        assert len(coords["x"]) == 4

    def test_x_grid_span(self):
        coords = run(make_params(nx0=4, lx=10.0))
        # linspace(-5, 5, 5)[:-1] → [-5, -2.5, 0, 2.5]
        expected = np.linspace(-5.0, 5.0, 5)[:-1]
        np.testing.assert_allclose(coords["x"], expected)


# ---------------------------------------------------------------------------
# ky
# ---------------------------------------------------------------------------

class TestKy:
    def test_ky_values(self):
        """nky0=4, kymin=0.5 → [0, 0.5, 1.0, 1.5]."""
        coords = run(make_params(nky0=4, kymin=0.5))
        expected = np.array([0.0, 0.5, 1.0, 1.5])
        np.testing.assert_allclose(coords["ky"], expected)

    def test_ky_shape(self):
        coords = run(make_params(nky0=4, kymin=0.5))
        assert coords["ky"].shape == (4,)

    def test_ky_starts_at_zero(self):
        coords = run(make_params(nky0=6, kymin=0.3))
        assert coords["ky"][0] == pytest.approx(0.0)

    def test_ky_spacing(self):
        kymin = 0.25
        coords = run(make_params(nky0=5, kymin=kymin))
        diffs = np.diff(coords["ky"])
        np.testing.assert_allclose(diffs, kymin)

    def test_ky_nky1_equals_kymin(self):
        """When nky0==1 the single mode should be kymin itself."""
        coords = run(make_params(nky0=1, kymin=0.7))
        np.testing.assert_allclose(coords["ky"], np.array([0.7]))


# ---------------------------------------------------------------------------
# z grid
# ---------------------------------------------------------------------------

class TestZGrid:
    def test_z_uniform_shape(self):
        coords = run(make_params(nz0=8, n_pol=1, edge_opt=0))
        assert coords["z"].shape == (8,)

    def test_z_uniform_span(self):
        """edge_opt=0 → linspace(-pi, pi, 9)[:-1]."""
        coords = run(make_params(nz0=8, n_pol=1, edge_opt=0))
        expected = np.linspace(-np.pi, np.pi, 9)[:-1]
        np.testing.assert_allclose(coords["z"], expected)

    def test_z_n_pol_scales_range(self):
        """n_pol=2 doubles the z extent."""
        coords = run(make_params(nz0=8, n_pol=2, edge_opt=0))
        expected = np.linspace(-2 * np.pi, 2 * np.pi, 9)[:-1]
        np.testing.assert_allclose(coords["z"], expected)

    def test_z_nonuniform_with_edge_opt(self):
        """edge_opt != 0 triggers sinh transformation → non-uniform spacing."""
        coords_uniform = run(make_params(nz0=16, n_pol=1, edge_opt=0))
        coords_sinh = run(make_params(nz0=16, n_pol=1, edge_opt=2))

        diffs_uniform = np.diff(coords_uniform["z"])
        diffs_sinh = np.diff(coords_sinh["z"])

        # Uniform spacings are all equal; sinh spacings are not
        assert not np.allclose(diffs_sinh, diffs_sinh[0]), \
            "edge_opt=2 should produce non-uniform z spacing"
        np.testing.assert_allclose(diffs_uniform, diffs_uniform[0])

    def test_z_edge_opt_shape_preserved(self):
        coords = run(make_params(nz0=12, n_pol=1, edge_opt=1))
        assert coords["z"].shape == (12,)

    def test_dz_zero_for_nz1(self):
        coords = run(make_params(nz0=1, n_pol=1, edge_opt=0))
        assert coords["dz"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# mu grid
# ---------------------------------------------------------------------------

class TestMuGrid:
    def test_gau_lag_mu_is_ndarray(self):
        """Regression B4: mu_type=gau_lag must return np.ndarray, not a scalar."""
        coords = run(make_params(mu_grid_type="gau_lag"))
        assert isinstance(coords["mu"], np.ndarray)

    def test_gau_lag_mu_weight_is_ndarray(self):
        """Regression B4: mu_weight must also be np.ndarray."""
        coords = run(make_params(mu_grid_type="gau_lag"))
        assert isinstance(coords["mu_weight"], np.ndarray)

    def test_eq_vperp_mu_shape(self):
        coords = run(make_params(nw0=4, lw=2.0, mu_grid_type="eq_vperp"))
        assert coords["mu"].shape == (4,)
        assert coords["mu_weight"].shape == (4,)

    def test_eq_vperp_mu_values(self):
        """nw0=4, lw=2.0: deltamu=lw/nw**2=0.125, mu=((i-0.5)**2)*deltamu."""
        nw, lw = 4, 2.0
        coords = run(make_params(nw0=nw, lw=lw, mu_grid_type="eq_vperp"))
        deltamu = lw / nw**2
        idx = np.arange(1, nw + 1)
        expected_mu = ((idx - 0.5) ** 2) * deltamu
        np.testing.assert_allclose(coords["mu"], expected_mu)

    def test_eq_vperp_mu_weight_values(self):
        """mu_weight=(2*i-1)*deltamu."""
        nw, lw = 4, 2.0
        coords = run(make_params(nw0=nw, lw=lw, mu_grid_type="eq_vperp"))
        deltamu = lw / nw**2
        idx = np.arange(1, nw + 1)
        expected_weight = (2 * idx - 1) * deltamu
        np.testing.assert_allclose(coords["mu_weight"], expected_weight)

    def test_eq_vperp_mu_positive(self):
        coords = run(make_params(nw0=6, lw=3.0, mu_grid_type="eq_vperp"))
        assert np.all(coords["mu"] > 0)

    def test_unknown_mu_type_returns_empty_arrays(self):
        coords = run(make_params(mu_grid_type="unknown_type"))
        assert isinstance(coords["mu"], np.ndarray)
        assert isinstance(coords["mu_weight"], np.ndarray)


# ---------------------------------------------------------------------------
# vp / vp_weight
# ---------------------------------------------------------------------------

class TestVpWeights:
    def test_vp_shape(self):
        coords = run(make_params(nv0=16, lv=3.0))
        assert coords["vp"].shape == (16,)

    def test_vp_symmetric(self):
        coords = run(make_params(nv0=16, lv=3.0))
        np.testing.assert_allclose(coords["vp"], -coords["vp"][::-1])

    def test_vp_weight_uniform_when_arakawa_zv(self):
        """arakawa_zv=True suppresses endpoint corrections → uniform weights."""
        coords = run(make_params(nv0=16, lv=3.0, collision_op="none", arakawa_zv=True))
        dv = coords["vp"][1] - coords["vp"][0]
        np.testing.assert_allclose(coords["vp_weight"], dv)

    def test_vp_weight_endpoint_correction_applied(self):
        """
        Endpoint correction fires when: collision_op not in {nonlin,sugama,exact},
        arakawa_zv=False, nv0 > 8.
        Weights at endpoints must differ from the uniform value.
        """
        coords_corr = run(make_params(
            nv0=16, lv=3.0,
            collision_op="none",
            arakawa_zv=False,
        ))
        coords_flat = run(make_params(
            nv0=16, lv=3.0,
            collision_op="none",
            arakawa_zv=True,
        ))
        # Corrected endpoint weights differ from uniform
        assert not np.allclose(coords_corr["vp_weight"][:4],
                               coords_flat["vp_weight"][:4])

    def test_vp_weight_no_correction_when_nv_small(self):
        """nv0 <= 8 → no endpoint correction even if collision_op and arakawa_zv allow it."""
        coords = run(make_params(
            nv0=8, lv=3.0,
            collision_op="none",
            arakawa_zv=False,
        ))
        dv = coords["vp"][1] - coords["vp"][0]
        np.testing.assert_allclose(coords["vp_weight"], dv)

    def test_vp_weight_no_correction_for_nonlin(self):
        """collision_op='nonlin' → no endpoint correction."""
        coords = run(make_params(
            nv0=16, lv=3.0,
            collision_op="nonlin",
            arakawa_zv=False,
        ))
        dv = coords["vp"][1] - coords["vp"][0]
        np.testing.assert_allclose(coords["vp_weight"], dv)

    def test_vp_weight_endpoint_values(self):
        """Explicit check of the corrected weights [17,59,43,49]*dv/48."""
        coords = run(make_params(
            nv0=16, lv=3.0,
            collision_op="none",
            arakawa_zv=False,
        ))
        dv = coords["vp"][1] - coords["vp"][0]
        expected_head = np.array([17, 59, 43, 49]) * dv / 48.0
        np.testing.assert_allclose(coords["vp_weight"][:4], expected_head)
        np.testing.assert_allclose(coords["vp_weight"][-4:], expected_head[::-1])


# ---------------------------------------------------------------------------
# Coordinates (multi-run interface)
# ---------------------------------------------------------------------------

class TestCoordinatesWrapper:
    def test_returns_list(self):
        params = {0: make_params(nx0=4), 1: make_params(nx0=6)}
        result = Coordinates(None, [0, 1], params)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_each_element_is_dict(self):
        params = {0: make_params(nx0=4)}
        result = Coordinates(None, [0], params)
        assert isinstance(result[0], dict)

    def test_each_run_uses_correct_params(self):
        """Two runs with different nx should produce different kx arrays."""
        params = {0: make_params(nx0=4, lx=10.0), 1: make_params(nx0=6, lx=10.0)}
        result = Coordinates(None, [0, 1], params)
        assert result[0]["kx"].shape == (4,)
        assert result[1]["kx"].shape == (6,)

    def test_folder_and_file_number_unused(self):
        """folder and file_number args are dead; arbitrary values must not affect output."""
        params = {0: make_params(nx0=4)}
        r1 = Coordinates("/fake/path", [0], params)
        r2 = Coordinates(None, [0], params)
        np.testing.assert_allclose(r1[0]["kx"], r2[0]["kx"])
