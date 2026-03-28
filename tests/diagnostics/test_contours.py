"""Tests for genetools.diagnostics.contours."""

import numpy as np
import pytest

from genetools.diagnostics.contours import Contours


# ---------------------------------------------------------------------------
# _resolve_ifft
# ---------------------------------------------------------------------------

class TestResolveIfft:
    """Contours._resolve_ifft(ifft_option, x_local)"""

    @pytest.mark.parametrize("opt", [None, "x", "y", "xy", "unknown"])
    def test_x_local_true_passthrough(self, opt):
        """When x_local=True every option is returned unchanged."""
        result = Contours._resolve_ifft(opt, x_local=True)
        assert result == opt

    @pytest.mark.parametrize("opt,expected", [
        ("xy", "y"),
        ("x",  None),
        ("y",  "y"),
        (None, None),
    ])
    def test_x_local_false_restricted(self, opt, expected):
        """When x_local=False the mapping removes x-IFFT options."""
        result = Contours._resolve_ifft(opt, x_local=False)
        assert result == expected

    def test_x_local_false_unknown_passthrough(self):
        """Unknown option not in the map is returned as-is (dict.get fallback)."""
        # The implementation uses _map.get(ifft_option, ifft_option),
        # so an unrecognised key passes through unchanged.
        result = Contours._resolve_ifft("z", x_local=False)
        assert result == "z"


# ---------------------------------------------------------------------------
# _compute_slices
# ---------------------------------------------------------------------------

class TestComputeSlices:
    """Contours._compute_slices(field_3d, effective_ifft, iz, iy, del_zonal, zero_range, nky)"""

    def _make_field(self, nx=8, nky=4, nz=16, seed=0):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((nx, nky, nz))
                + 1j * rng.standard_normal((nx, nky, nz))).astype(np.complex64)

    def test_ifft_none_returns_float32(self):
        c = Contours()
        field = self._make_field()
        nx, nky, nz = field.shape
        f_xy, f_xz = c._compute_slices(field, None, iz=nz // 2, iy=0,
                                        del_zonal=False, zero_range=None,
                                        nky=nky)
        assert f_xy.dtype == np.float32
        assert f_xz.dtype == np.float32

    def test_ifft_none_xy_is_real_part(self):
        c = Contours()
        field = self._make_field()
        nx, nky, nz = field.shape
        iz = nz // 2
        f_xy, _ = c._compute_slices(field, None, iz=iz, iy=0,
                                     del_zonal=False, zero_range=None,
                                     nky=nky)
        expected = field[:, :, iz].real.astype(np.float32)
        np.testing.assert_array_equal(f_xy, expected)

    def test_ifft_none_shapes(self):
        nx, nky, nz = 10, 5, 12
        c = Contours()
        field = self._make_field(nx=nx, nky=nky, nz=nz)
        f_xy, f_xz = c._compute_slices(field, None, iz=0, iy=1,
                                        del_zonal=False, zero_range=None,
                                        nky=nky)
        # Without y-IFFT shape is (nx, nky); x-IFFT not done either
        assert f_xy.shape == (nx, nky)
        assert f_xz.shape == (nx, nz)

    def test_ifft_y_shape(self):
        """With effective_ifft='y' the y-dimension expands to 2*(nky-1)."""
        nx, nky, nz = 8, 5, 16
        c = Contours()
        field = self._make_field(nx=nx, nky=nky, nz=nz)
        ny_full = 2 * (nky - 1)
        f_xy, f_xz = c._compute_slices(field, "y", iz=0, iy=0,
                                        del_zonal=False, zero_range=None,
                                        nky=nky)
        assert f_xy.shape == (nx, ny_full)
        assert f_xz.shape == (nx, nz)   # XZ does not do y-IFFT

    def test_ifft_x_shape(self):
        nx, nky, nz = 8, 4, 16
        c = Contours()
        field = self._make_field(nx=nx, nky=nky, nz=nz)
        f_xy, f_xz = c._compute_slices(field, "x", iz=0, iy=0,
                                        del_zonal=False, zero_range=None,
                                        nky=nky)
        assert f_xy.shape == (nx, nky)
        assert f_xz.shape == (nx, nz)

    def test_del_zonal_zeros_ky0(self):
        """del_zonal=True must set ky=0 to zero before transforms."""
        nx, nky, nz = 8, 4, 16
        c = Contours()
        rng = np.random.default_rng(1)
        field = (rng.standard_normal((nx, nky, nz))
                 + 1j * rng.standard_normal((nx, nky, nz))).astype(np.complex64)
        iz = nz // 2
        f_xy, _ = c._compute_slices(field, None, iz=iz, iy=0,
                                     del_zonal=True, zero_range=None,
                                     nky=nky)
        # With ifft=None the xy slice is real(field[:, :, iz]) after zeroing ky=0
        # So f_xy[:, 0] should be all zeros
        np.testing.assert_array_equal(f_xy[:, 0], np.zeros(nx, dtype=np.float32))

    def test_del_zonal_preserves_other_ky(self):
        nx, nky, nz = 6, 3, 8
        c = Contours()
        rng = np.random.default_rng(7)
        field = (rng.standard_normal((nx, nky, nz))
                 + 1j * rng.standard_normal((nx, nky, nz))).astype(np.complex64)
        iz = 0
        f_xy_del, _ = c._compute_slices(field, None, iz=iz, iy=0,
                                         del_zonal=True, zero_range=None,
                                         nky=nky)
        f_xy_raw, _ = c._compute_slices(field, None, iz=iz, iy=0,
                                         del_zonal=False, zero_range=None,
                                         nky=nky)
        # ky > 0 modes unchanged
        np.testing.assert_array_equal(f_xy_del[:, 1:], f_xy_raw[:, 1:])

    def test_nky_1_does_not_crash(self):
        """Regression: nky=1 must not crash (B11)."""
        nx, nky, nz = 4, 1, 8
        c = Contours()
        field = np.ones((nx, nky, nz), dtype=np.complex64)
        # Should not raise
        f_xy, f_xz = c._compute_slices(field, "y", iz=0, iy=0,
                                        del_zonal=False, zero_range=None,
                                        nky=nky)
        assert f_xy.dtype == np.float32
        assert f_xz.dtype == np.float32

    def test_nky_1_ifft_none_shape(self):
        nx, nky, nz = 4, 1, 8
        c = Contours()
        field = np.ones((nx, nky, nz), dtype=np.complex64)
        f_xy, f_xz = c._compute_slices(field, None, iz=0, iy=0,
                                        del_zonal=False, zero_range=None,
                                        nky=nky)
        assert f_xy.shape == (nx, nky)
        assert f_xz.shape == (nx, nz)

    def test_zero_range_zeros_lower_ky(self):
        nx, nky, nz = 6, 4, 8
        c = Contours()
        rng = np.random.default_rng(42)
        field = (rng.standard_normal((nx, nky, nz))
                 + 1j * rng.standard_normal((nx, nky, nz))).astype(np.complex64)
        iz = 0
        zero_range = 2  # zero out ky=0,1
        f_xy, _ = c._compute_slices(field, None, iz=iz, iy=0,
                                     del_zonal=False, zero_range=zero_range,
                                     nky=nky)
        np.testing.assert_array_equal(f_xy[:, :zero_range],
                                      np.zeros((nx, zero_range), dtype=np.float32))


# ---------------------------------------------------------------------------
# select_indices
# ---------------------------------------------------------------------------

class TestSelectIndices:
    """Contours.select_indices — pure index-selection logic."""

    def _make_mock_reader(self, times):
        """Minimal mock with read_all_times()."""
        class _R:
            def read_all_times(self_):
                return np.array(times)
        return _R()

    def test_selects_within_window(self):
        c = Contours()
        times = np.linspace(0, 10, 21)   # 0, 0.5, 1.0, ..., 10.0
        reader = self._make_mock_reader(times)
        idx = c.select_indices(reader, t_start=3.0, t_stop=5.0, max_loads=100)
        assert all(3.0 <= times[i] <= 5.0 for i in idx)

    def test_empty_window_returns_empty(self):
        c = Contours()
        reader = self._make_mock_reader(np.linspace(0, 5, 11))
        idx = c.select_indices(reader, t_start=10.0, t_stop=20.0, max_loads=10)
        assert idx == []

    def test_max_loads_respected(self):
        c = Contours()
        reader = self._make_mock_reader(np.linspace(0, 100, 201))
        idx = c.select_indices(reader, t_start=0.0, t_stop=100.0, max_loads=5)
        assert len(idx) <= 5

    def test_returns_list(self):
        c = Contours()
        reader = self._make_mock_reader([0.0, 1.0, 2.0])
        idx = c.select_indices(reader, t_start=0.0, t_stop=2.0, max_loads=10)
        assert isinstance(idx, list)

    def test_single_point_in_window(self):
        c = Contours()
        reader = self._make_mock_reader([0.0, 1.0, 2.0, 3.0])
        idx = c.select_indices(reader, t_start=1.5, t_stop=2.5, max_loads=10)
        assert len(idx) == 1
        assert idx[0] == 2   # index of t=2.0


# ---------------------------------------------------------------------------
# _get_axes
# ---------------------------------------------------------------------------

class TestGetAxes:
    """Contours._get_axes — returns correct axis arrays and labels."""

    def _make_coord(self, nx=8, nky=4, nz=16):
        kx = np.linspace(-1.0, 1.0, nx)
        x  = np.linspace(-5.0, 5.0, nx)
        ky = np.arange(nky, dtype=float) * 0.5
        z  = np.linspace(-np.pi, np.pi, nz)
        return {"kx": kx, "x": x, "ky": ky, "z": z}

    def test_no_ifft_uses_kx_ky(self):
        coord = self._make_coord()
        x_ax, y_ax, z_ax, x_lbl, y_lbl, _ = \
            Contours._get_axes(coord, effective_ifft=None, x_local=True)
        np.testing.assert_array_equal(x_ax, coord["kx"])
        np.testing.assert_array_equal(y_ax, coord["ky"])

    def test_ifft_xy_uses_x_and_physical_y(self):
        nky = 5
        coord = self._make_coord(nky=nky)
        x_ax, y_ax, z_ax, _, _, _ = \
            Contours._get_axes(coord, effective_ifft="xy", x_local=True, nky=nky)
        np.testing.assert_array_equal(x_ax, coord["x"])
        # y_ax should have length ny_full = 2*(nky-1)
        assert len(y_ax) == 2 * (nky - 1)

    def test_global_always_uses_x(self):
        """For x_local=False the x-axis must always be the real-space x."""
        coord = self._make_coord()
        x_ax, _, _, _, _, _ = \
            Contours._get_axes(coord, effective_ifft=None, x_local=False)
        np.testing.assert_array_equal(x_ax, coord["x"])

    def test_z_axis_always_from_coord(self):
        coord = self._make_coord()
        _, _, z_ax, _, _, _ = \
            Contours._get_axes(coord, effective_ifft=None, x_local=True)
        np.testing.assert_array_equal(z_ax, coord["z"])
