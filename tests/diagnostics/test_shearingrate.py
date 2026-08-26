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
            "phi_zonal_fsavg", "phi_zonal",
            "e_r", "v_exb", "omega_exb", "shearing_rms",
        }
        assert expected_keys == set(result.keys())

    def test_phi_zonal_fsavg_shape(self):
        nx, nky, nz = 8, 3, 16
        phi = _make_phi(nx, nky, nz)
        result = compute_exb(phi, _make_local_params(nx, nky, nz),
                             _make_local_geom(nx, nz),
                             _make_local_coord(nx, nky))
        assert result["phi_zonal_fsavg"].shape == (nx,)

    @pytest.mark.parametrize("key", ["phi_zonal", "e_r", "v_exb", "omega_exb"])
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
        np.testing.assert_allclose(result["phi_zonal"],  0.0, atol=1e-14)
        np.testing.assert_allclose(result["e_r"],          0.0, atol=1e-14)
        np.testing.assert_allclose(result["omega_exb"],    0.0, atol=1e-14)
        assert result["shearing_rms"] == pytest.approx(0.0, abs=1e-14)

    def test_outputs_are_real(self):
        nx, nky, nz = 8, 3, 16
        phi = _make_phi(nx, nky, nz)
        result = compute_exb(phi, _make_local_params(nx, nky, nz),
                             _make_local_geom(nx, nz),
                             _make_local_coord(nx, nky))
        for key in ("phi_zonal", "e_r", "v_exb", "omega_exb"):
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
            "phi_zonal_fsavg", "phi_zonal",
            "e_r", "v_exb", "omega_exb", "shearing_rms",
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

    @pytest.mark.parametrize("key", ["phi_zonal", "e_r", "v_exb", "omega_exb"])
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
        np.testing.assert_allclose(result["phi_zonal"], 0.0, atol=1e-14)
        np.testing.assert_allclose(result["e_r"],         0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Legacy cache compatibility
# ---------------------------------------------------------------------------

class TestLegacyCacheNames:
    """
    A `shearing_rate.h5` written before the two geometry paths were made to
    agree on variable names must keep working: read through the alias map, and
    migrated in place before anything is appended to it.
    """

    LEGACY = {"phi_zonal_x": "phi_zonal", "E_r": "e_r", "v_ExB": "v_exb",
              "omega_ExB": "omega_exb", "abs_phi_zonal_kx": "phi_zonal_kx_abs"}

    def _write_legacy(self, path, nx=6, nkx=4, nt=3):
        import h5py
        with h5py.File(path, "w") as f:
            f.create_dataset("time", data=np.arange(nt, dtype=float),
                             maxshape=(None,))
            for old in ("phi_zonal_x", "E_r", "v_ExB", "omega_ExB"):
                f.create_dataset(old, data=np.ones((nt, nx)),
                                 maxshape=(None, nx))
            f.create_dataset("shearing_rms", data=np.ones(nt),
                             maxshape=(None,))
            f.create_dataset("abs_phi_zonal_kx", data=np.ones((nt, nkx)),
                             maxshape=(None, nkx))
        return nx, nkx, nt

    def test_load_translates_the_old_names(self, tmp_path):
        from genetools.diagnostics.shearingrate import ShearingRate
        path = tmp_path / "shearing_rate.h5"
        self._write_legacy(path)
        data = ShearingRate(outfile=str(path)).load()
        for old, new in self.LEGACY.items():
            assert new in data, new
            assert old not in data, old

    def _result(self, nx, nkx):
        result = {n: np.zeros(nx) for n in
                  ("phi_zonal", "e_r", "v_exb", "omega_exb")}
        result["shearing_rms"] = 0.0
        result["phi_zonal_fsavg"] = np.zeros(nkx)
        return result

    def test_append_without_migration_raises(self, tmp_path):
        """
        Why the migration exists. `load` translates names on read, so without
        migrating the file itself an append goes looking for datasets that are
        not there — and it half-succeeds first, growing `time` before it fails,
        which is why `compute_and_save` migrates before writing anything.
        """
        import h5py
        from genetools.diagnostics.shearingrate import ShearingRate
        path = tmp_path / "shearing_rate.h5"
        nx, nkx, _ = self._write_legacy(path)
        with h5py.File(path, "a") as f:
            with pytest.raises(KeyError):
                ShearingRate._append_to_open_file(f, self._result(nx, nkx),
                                                  99.0, True)

    def test_append_after_migration_does_not_raise(self, tmp_path):
        import h5py
        from genetools.diagnostics.shearingrate import ShearingRate
        path = tmp_path / "shearing_rate.h5"
        nx, nkx, nt = self._write_legacy(path)
        with h5py.File(path, "a") as f:
            ShearingRate._migrate_legacy_names(f)
            assert "phi_zonal_x" not in f and "phi_zonal" in f
            ShearingRate._append_to_open_file(f, self._result(nx, nkx),
                                              99.0, True)
            assert f["time"].shape[0] == nt + 1
            assert f["phi_zonal"].shape == (nt + 1, nx)

    def test_migration_preserves_the_cached_values(self, tmp_path):
        import h5py
        from genetools.diagnostics.shearingrate import ShearingRate
        path = tmp_path / "shearing_rate.h5"
        self._write_legacy(path)
        with h5py.File(path, "r") as f:
            before = f["phi_zonal_x"][...]
        with h5py.File(path, "a") as f:
            ShearingRate._migrate_legacy_names(f)
        with h5py.File(path, "r") as f:
            np.testing.assert_array_equal(f["phi_zonal"][...], before)

    def test_migration_is_idempotent(self, tmp_path):
        import h5py
        from genetools.diagnostics.shearingrate import ShearingRate
        path = tmp_path / "shearing_rate.h5"
        self._write_legacy(path)
        with h5py.File(path, "a") as f:
            ShearingRate._migrate_legacy_names(f)
            ShearingRate._migrate_legacy_names(f)
            assert sorted(f.keys()) == sorted(
                ["time", "shearing_rms"] + list(self.LEGACY.values()))


# ---------------------------------------------------------------------------
# Conventions shared by every geometry
# ---------------------------------------------------------------------------

class TestExbConventions:
    """
    `v_ExB = -E_r/C_xy` with `E_r = -dphi/dx`, so `omega_ExB = d(v_ExB)/dx`
    matches GENE's own `ExBrate = -dEraddx_prof(0)` (`profiles.F90`). The global
    branch used to return `v_exb = e_r` — the opposite sign, and with no C_xy at
    all, so it disagreed with the flux tube on both counts.
    """

    @staticmethod
    def _ramp_phi(nx, nky, nz):
        """A y- and z-independent radial ramp: dphi/dx > 0 everywhere."""
        phi = np.zeros((nx, nky, nz), dtype=complex)
        phi[:, 0, :] = np.linspace(0.0, 1.0, nx)[:, None]
        return phi

    def test_global_v_exb_is_minus_e_r_over_C_xy(self):
        nx, nky, nz = 10, 3, 8
        cxy = 1.7
        geom = _make_global_geom(nx, nz)
        geom["metric"]["C_xy"] = np.full((nx, nz), cxy)
        result = compute_exb(self._ramp_phi(nx, nky, nz),
                             _make_global_params(nx, nky, nz),
                             geom, _make_global_coord(nx))
        np.testing.assert_allclose(result["v_exb"],
                                   -result["e_r"] / cxy, rtol=1e-12)

    def test_global_ramp_gives_negative_e_r_and_positive_v_exb(self):
        nx, nky, nz = 10, 3, 8
        geom = _make_global_geom(nx, nz)
        geom["metric"]["C_xy"] = np.full((nx, nz), 2.0)
        result = compute_exb(self._ramp_phi(nx, nky, nz),
                             _make_global_params(nx, nky, nz),
                             geom, _make_global_coord(nx))
        assert np.all(result["e_r"][1:-1] < 0)
        assert np.all(result["v_exb"][1:-1] > 0)

    def test_local_v_exb_is_minus_e_r_over_C_xy(self):
        """The flux tube was already right; pin it so it stays that way."""
        nx, nky, nz = 8, 3, 8
        rng = np.random.default_rng(3)
        phi = rng.normal(size=(nx, nky, nz)) + 1j * rng.normal(size=(nx, nky, nz))
        cxy = 1.9
        result = compute_exb(phi, _make_local_params(nx, nky, nz),
                             _make_local_geom(nx, nz, C_xy=cxy),
                             _make_local_coord(nx))
        np.testing.assert_allclose(result["v_exb"],
                                   -result["e_r"] / cxy, rtol=1e-12)

    def test_global_flux_surface_average_normalises_per_surface(self):
        """
        A 2-D Jacobian must be normalised over z alone. Dividing by its total
        sum makes every flux-surface average a factor ~1/nx too small, and
        x-dependently so — invisible with the uniform Jacobian the other
        fixtures use, so this one varies it in both directions.
        """
        nx, nky, nz = 6, 3, 4
        rng = np.random.default_rng(11)
        J = 1.0 + rng.uniform(size=(nx, nz))
        geom = _make_global_geom(nx, nz)
        geom["Jacobian"] = J
        phi = np.zeros((nx, nky, nz), dtype=complex)
        phi[:, 0, :] = rng.normal(size=(nx, nz))

        result = compute_exb(phi, _make_global_params(nx, nky, nz),
                             geom, _make_global_coord(nx))
        expected = ((phi[:, 0, :].real * J).sum(axis=1) / J.sum(axis=1))
        np.testing.assert_allclose(result["phi_zonal"], expected, rtol=1e-12)

    def test_uniform_phi_averages_to_itself(self):
        """
        The sharpest check that the weights normalise: a potential constant on
        each flux surface must come back unchanged, whatever the Jacobian.
        """
        nx, nky, nz = 6, 3, 4
        rng = np.random.default_rng(5)
        geom = _make_global_geom(nx, nz)
        geom["Jacobian"] = 1.0 + rng.uniform(size=(nx, nz))
        profile = np.linspace(2.0, 5.0, nx)
        phi = np.zeros((nx, nky, nz), dtype=complex)
        phi[:, 0, :] = profile[:, None]

        result = compute_exb(phi, _make_global_params(nx, nky, nz),
                             geom, _make_global_coord(nx))
        np.testing.assert_allclose(result["phi_zonal"], profile, rtol=1e-12)


# ---------------------------------------------------------------------------
# The -1 time-bound sentinel (shared by every diagnostic)
# ---------------------------------------------------------------------------

class TestNegativeTimeBound:
    """
    A negative bound means "unbounded on that side": `t=(500, -1)` is "from 500
    to the end". GENE times start at zero and increase, so a negative bound
    could never select anything and is free to carry that meaning.
    """

    @pytest.mark.parametrize("t, expected", [
        (None, (None, None)),
        ((500, 2000), (500.0, 2000.0)),
        ((500, -1), (500.0, None)),
        ((-1, 2000), (None, 2000.0)),
        ((-1, -1), (None, None)),
        (500, (500.0, None)),
        (-1, (None, None)),
    ])
    def test_window(self, t, expected):
        from genetools.diagnostics._base import RunDiagnostic
        assert RunDiagnostic._window(t) == expected

    def test_bounds_open_out_to_the_streaming_limits(self):
        from genetools.diagnostics._base import RunDiagnostic
        lo, hi = RunDiagnostic._bounds((500, -1))
        assert lo == 500.0
        assert hi > 1e29

    def test_zero_is_a_real_bound_not_a_sentinel(self):
        """t=0 is the first output time of a run, not 'unbounded'."""
        from genetools.diagnostics._base import RunDiagnostic
        assert RunDiagnostic._window((0, 10)) == (0.0, 10.0)
