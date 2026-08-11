"""Tests for genetools.diagnostics.amplitude.AmplitudeSpectra._accumulate."""

import numpy as np

from genetools.diagnostics.amplitude import AmplitudeSpectra


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeReader:
    """Minimal reader: yields pre-built ``(time, [arrays])`` snapshots."""

    def __init__(self, snapshots):
        self._snapshots = snapshots

    def stream_selected(self, indices):
        for i in indices:
            yield self._snapshots[i]


def _amp():
    """AmplitudeSpectra instance without a Run (only _accumulate is used)."""
    return object.__new__(AmplitudeSpectra)


def _uniform_J(nx, nz):
    """Per-surface-normalised global Jacobian (rows sum to 1)."""
    return np.ones((nx, nz)) / nz


def _ky_weight(nky):
    w = np.full(nky, 2.0)
    w[0] = 1.0
    return w


# ---------------------------------------------------------------------------
# Global (x, ky) map
# ---------------------------------------------------------------------------

class TestGlobalXKyMap:

    def test_shapes_and_keys(self):
        nx, nky, nz = 6, 4, 8
        arr = np.ones((nx, nky, nz), dtype=complex)
        out = {}
        _amp()._accumulate(FakeReader([(0.0, [arr])]), [0], ["phi"],
                           _uniform_J(nx, nz), _ky_weight(nky), out,
                           is_local=False)
        assert set(out) == {"phi_x", "phi_ky", "phi_xky"}
        assert out["phi_xky"].shape == (nx, nky)
        assert out["phi_x"].shape == (nx,)
        assert out["phi_ky"].shape == (nky,)

    def test_values_with_ky_weighting(self):
        """|c_j|² per ky mode, ky=0 weight 1, ky>0 weight 2."""
        nx, nky, nz = 4, 3, 8
        arr = np.zeros((nx, nky, nz), dtype=complex)
        coeffs = [1.0, 2.0, 3.0]
        for j, c in enumerate(coeffs):
            arr[:, j, :] = c
        out = {}
        _amp()._accumulate(FakeReader([(0.0, [arr])]), [0], ["phi"],
                           _uniform_J(nx, nz), _ky_weight(nky), out,
                           is_local=False)
        expected = np.array([1.0, 2 * 4.0, 2 * 9.0])  # w_j * |c_j|²
        for x in range(nx):
            np.testing.assert_allclose(out["phi_xky"][x], expected, rtol=1e-12)

    def test_reductions_consistent_with_map(self):
        nx, nky, nz = 5, 4, 6
        rng = np.random.default_rng(1)
        arr = rng.standard_normal((nx, nky, nz)) \
            + 1j * rng.standard_normal((nx, nky, nz))
        out = {}
        _amp()._accumulate(FakeReader([(0.0, [arr])]), [0], ["phi"],
                           _uniform_J(nx, nz), _ky_weight(nky), out,
                           is_local=False)
        np.testing.assert_allclose(out["phi_x"], out["phi_xky"].sum(axis=1),
                                   rtol=1e-12)
        np.testing.assert_allclose(out["phi_ky"], out["phi_xky"].sum(axis=0),
                                   rtol=1e-12)

    def test_per_surface_jacobian_normalisation(self):
        """A Jacobian rescaled per flux surface must not change the result."""
        nx, nky, nz = 4, 3, 8
        rng = np.random.default_rng(2)
        arr = rng.standard_normal((nx, nky, nz)) \
            + 1j * rng.standard_normal((nx, nky, nz))
        J = np.outer(np.arange(1, nx + 1), rng.uniform(0.5, 1.5, nz))
        J_norm = J / J.sum(axis=1, keepdims=True)
        base = np.tile(J[0] / J[0].sum(), (nx, 1))  # same z-shape, no x factor
        out_a, out_b = {}, {}
        for out, Jn in ((out_a, J_norm), (out_b, base)):
            _amp()._accumulate(FakeReader([(0.0, [arr])]), [0], ["phi"],
                               Jn, _ky_weight(nky), out, is_local=False)
        np.testing.assert_allclose(out_a["phi_xky"], out_b["phi_xky"],
                                   rtol=1e-12)

    def test_time_average_is_trapezoidal(self):
        nx, nky, nz = 3, 2, 4
        a = np.full((nx, nky, nz), 1.0, dtype=complex)
        b = np.full((nx, nky, nz), 3.0, dtype=complex)
        snaps = [(0.0, [a]), (1.0, [b])]
        out = {}
        _amp()._accumulate(FakeReader(snaps), [0, 1], ["phi"],
                           _uniform_J(nx, nz), _ky_weight(nky), out,
                           is_local=False)
        # trapz of |1|²=1 and |3|²=9 over t in [0, 1] -> 5, ky weights (1, 2)
        np.testing.assert_allclose(out["phi_xky"][:, 0], 5.0, rtol=1e-12)
        np.testing.assert_allclose(out["phi_xky"][:, 1], 10.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# Local path stays kx/ky only
# ---------------------------------------------------------------------------

class TestLocalPath:

    def test_no_xky_key_for_local(self):
        nkx, nky, nz = 6, 4, 8
        arr = np.ones((nkx, nky, nz), dtype=complex)
        J = np.ones(nz) / nz
        out = {}
        _amp()._accumulate(FakeReader([(0.0, [arr])]), [0], ["phi"],
                           J, _ky_weight(nky), out, is_local=True)
        assert set(out) == {"phi_kx", "phi_ky"}


# ---------------------------------------------------------------------------
# Dataset dimension mapping (shared with SpectraGlobal)
# ---------------------------------------------------------------------------

class TestDimsFromSuffix:

    def test_suffix_mapping(self):
        from genetools import _xr
        assert _xr.dims_from_suffix("phi_xky") == ("x", "ky")
        assert _xr.dims_from_suffix("phi_ky") == ("ky",)
        assert _xr.dims_from_suffix("phi_kx") == ("kx",)
        assert _xr.dims_from_suffix("dens_x") == ("x",)
