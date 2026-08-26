"""Tests for the x-global spectra path, now part of `Spectra`."""

import numpy as np
import pytest

from genetools.diagnostics.spectra import _compute_flux_yspectra


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


# ---------------------------------------------------------------------------
# Dataset layout: (x, ky) map plus 1-D reductions
# ---------------------------------------------------------------------------

def _bare(x_avg_lims=None, buffer_frac=0.0):
    """
    A Spectra instance with no run attached, for the pure x-global helpers.

    These paths only need the radial-window options, so bypassing __init__ keeps
    the test off the HDF5 and Run machinery.
    """
    from genetools.diagnostics.spectra import Spectra
    sg = object.__new__(Spectra)
    sg.x_avg_lims = x_avg_lims
    sg.buffer_frac = buffer_frac
    return sg


class TestExpandReductions:

    def test_map_and_reduction_keys(self):
        raw = {"ions_Q_es_ky": np.ones((5, 3)),
               "ions_Gamma_es_ky": np.ones((5, 3))}
        out = _bare()._expand_reductions(raw, slice(None))
        assert set(out) == {
            "ions_Q_es_xky", "ions_Q_es_x", "ions_Q_es_ky",
            "ions_Gamma_es_xky", "ions_Gamma_es_x", "ions_Gamma_es_ky",
        }

    def test_reduction_shapes_and_values(self):
        nx, nky = 4, 3
        arr = np.arange(nx * nky, dtype=float).reshape(nx, nky)
        out = _bare()._expand_reductions({"ions_Q_es_ky": arr}, slice(None))
        np.testing.assert_allclose(out["ions_Q_es_xky"], arr)
        # radial profile is the ky-sum; ky spectrum is the radial mean
        np.testing.assert_allclose(out["ions_Q_es_x"], arr.sum(axis=1))
        np.testing.assert_allclose(out["ions_Q_es_ky"], arr.mean(axis=0))
        assert out["ions_Q_es_x"].shape == (nx,)
        assert out["ions_Q_es_ky"].shape == (nky,)

    def test_ky_spectrum_honours_the_radial_window(self):
        """
        The ky spectrum averages over the retained window only, so the Krook
        buffer regions stay out of it — the same rule the GENE-3D path uses.
        """
        # Quadratic in x, not linear: a linear ramp has the same mean over a
        # centred sub-window as over the whole axis, which makes the test vacuous.
        arr = (np.arange(5.0) ** 2)[:, None] * np.ones(4)
        full = _bare()._expand_reductions({"i_Q_es_ky": arr}, slice(None))
        inner = _bare()._expand_reductions({"i_Q_es_ky": arr}, slice(1, 4))
        np.testing.assert_allclose(full["i_Q_es_ky"], arr.mean(axis=0))
        np.testing.assert_allclose(inner["i_Q_es_ky"], arr[1:4].mean(axis=0))
        assert not np.allclose(full["i_Q_es_ky"], inner["i_Q_es_ky"])

    def test_non_2d_entries_pass_through(self):
        out = _bare()._expand_reductions({"time": np.arange(4.0)}, slice(None))
        np.testing.assert_allclose(out["time"], np.arange(4.0))


class TestDatasetLayout:

    def _dataset(self):
        nx, nky = 6, 4
        rng = np.random.default_rng(0)
        maps = {sp: rng.standard_normal((nx, nky)) for sp in ("ions", "elec")}
        sg = _bare()
        sg._load_time_average_global = lambda a=None, b=None: {
            f"{sp}_Q_es_ky": m for sp, m in maps.items()}
        coords = {"x": np.linspace(0.3, 0.7, nx), "ky": 0.05 * np.arange(nky)}
        ds = sg._dataset_global(coords, {"units": {}}, ["ions", "elec"])
        return ds, maps

    def test_dims_and_coords(self):
        ds, _ = self._dataset()
        assert ds["Q_es_xky"].dims == ("species", "x", "ky")
        assert ds["Q_es_x"].dims == ("species", "x")
        assert ds["Q_es_ky"].dims == ("species", "ky")
        assert list(ds["species"].values) == ["ions", "elec"]

    def test_reductions_consistent_with_map(self):
        ds, _ = self._dataset()
        np.testing.assert_allclose(
            ds["Q_es_x"].values, ds["Q_es_xky"].values.sum(axis=2), rtol=1e-12)
        np.testing.assert_allclose(
            ds["Q_es_ky"].values, ds["Q_es_xky"].values.mean(axis=1),
            rtol=1e-12)

    def test_species_order_preserved(self):
        ds, maps = self._dataset()
        np.testing.assert_allclose(
            ds["Q_es_xky"].sel(species="elec").values, maps["elec"])

    def test_empty_cache_returns_empty_dataset(self):
        sg = _bare()
        sg._load_time_average_global = lambda a=None, b=None: {}
        ds = sg._dataset_global({"x": [], "ky": []}, {}, [])
        assert len(ds.data_vars) == 0


# ---------------------------------------------------------------------------
# Legacy cache names
# ---------------------------------------------------------------------------

class TestLegacyFluxNames:
    """
    Caches written before the geometries agreed on flux names must keep
    working: translated on read, and renamed in place before anything is
    appended — an unmigrated append raises KeyError after extending `time`.
    """

    def test_current_name_translates_both_schemas(self):
        from genetools.diagnostics.spectra import Spectra
        assert Spectra._current_name("Qes_ky") == "Q_es_ky"
        assert Spectra._current_name("Ges_ky") == "Gamma_es_ky"
        assert Spectra._current_name("Pem_ky") == "Pi_em_ky"
        assert Spectra._current_name("ions_G_es_kx") == "ions_Gamma_es_kx"
        assert Spectra._current_name("ions_Q_es_kx") == "ions_Q_es_kx"

    def test_migrates_x_global_groups(self, tmp_path):
        import h5py
        from genetools.diagnostics.spectra import Spectra
        path = tmp_path / "spectra_global.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("time", data=np.arange(3.0), maxshape=(None,))
            grp = f.create_group("ions")
            for old in ("Qes_ky", "Ges_ky", "Pes_ky"):
                grp.create_dataset(old, data=np.ones((4, 3, 3)),
                                   maxshape=(4, 3, None))
        with h5py.File(path, "a") as f:
            Spectra._migrate_legacy_names(f)
            assert set(f["ions"].keys()) == {"Q_es_ky", "Gamma_es_ky",
                                             "Pi_es_ky"}

    def test_migrates_flux_tube_keys(self, tmp_path):
        import h5py
        from genetools.diagnostics.spectra import Spectra
        path = tmp_path / "flux_spectra.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("time", data=np.arange(2.0), maxshape=(None,))
            for old in ("ions_G_es_kx", "ions_G_em_ky", "ions_Q_es_z"):
                f.create_dataset(old, data=np.ones((2, 3)),
                                 maxshape=(None, 3))
        with h5py.File(path, "a") as f:
            Spectra._migrate_legacy_names(f)
            assert "ions_Gamma_es_kx" in f and "ions_G_es_kx" not in f
            assert "ions_Gamma_em_ky" in f
            assert "ions_Q_es_z" in f          # already current, untouched

    def test_migration_preserves_values_and_is_idempotent(self, tmp_path):
        import h5py
        from genetools.diagnostics.spectra import Spectra
        path = tmp_path / "spectra_global.h5"
        payload = np.arange(24.0).reshape(4, 3, 2)
        with h5py.File(path, "w") as f:
            f.create_dataset("time", data=np.arange(2.0), maxshape=(None,))
            f.create_group("ions").create_dataset(
                "Qes_ky", data=payload, maxshape=(4, 3, None))
        with h5py.File(path, "a") as f:
            Spectra._migrate_legacy_names(f)
            Spectra._migrate_legacy_names(f)
            np.testing.assert_array_equal(f["ions/Q_es_ky"][...], payload)
            assert list(f["ions"].keys()) == ["Q_es_ky"]

    def test_migrate_cache_runs_even_when_nothing_is_missing(self, tmp_path):
        """
        The writers return early when every requested step is already cached, so
        the migration has to happen before that check — otherwise a legacy cache
        stays unmigrated until the first append, which is precisely when the
        rename becomes mandatory.
        """
        import h5py
        from genetools.diagnostics.spectra import Spectra
        path = tmp_path / "flux_spectra.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("time", data=np.arange(2.0), maxshape=(None,))
            f.create_dataset("ions_G_es_kx", data=np.ones((2, 3)),
                             maxshape=(None, 3))
        sg = _bare()
        sg.outfile = str(path)
        sg._migrate_cache()
        with h5py.File(path) as f:
            assert "ions_Gamma_es_kx" in f and "ions_G_es_kx" not in f

    def test_migrate_cache_tolerates_a_missing_file(self, tmp_path):
        sg = _bare()
        sg.outfile = str(tmp_path / "absent.h5")
        sg._migrate_cache()          # must not raise
