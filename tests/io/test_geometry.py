"""Tests for genetools.io.geometry."""

import textwrap
from unittest.mock import MagicMock

import numpy as np
import pytest

from genetools.io.geometry import Geometry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NZ = 16  # number of z-points used in synthetic data


def _make_params_dict(nz: int = NZ) -> dict:
    """Return a minimal params dict for a local tracer_efit run."""
    return {
        "general": {"x_local": True},
        "box": {"nx0": 4, "nky0": 2, "nz0": nz},
        "geometry": {"magn_geometry": "tracer_efit", "n_pol": 1, "edge_opt": 0},
        "units": {"Lref": 1.0},
    }


def _make_mock_params(nz: int = NZ) -> MagicMock:
    """Return a mock Params object whose .get() returns a params dict."""
    params_dict = _make_params_dict(nz)
    mock = MagicMock()
    mock.get.return_value = params_dict
    return mock


def _write_local_geom_file(folder, ext: str = "_0001", nz: int = NZ) -> None:
    """Write a synthetic tracer_efit geometry file to *folder*."""
    namelist = textwrap.dedent("""\
        &tracer_efit
         q0=1.4
         shat=0.8
         trpeps=0.18
         cxy=0.5
         cy=1.0
        /
    """)

    rng = np.random.default_rng(0)
    # All columns positive and non-zero so metric operations are well-defined.
    # Columns: gxx gxy gxz gyy gyz gzz Bfield dBdx dBdy dBdz
    #          Jacobian R Phi Z dxdR dxdZ
    data = rng.uniform(0.5, 1.5, size=(nz, 16))

    rows = "\n".join(
        "  ".join(f"{v:.8e}" for v in row) for row in data
    )

    fpath = folder / f"tracer_efit{ext}"
    fpath.write_text(namelist + rows + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLocalGeometryParsing:
    """_read_local: namelist header + 16-column numeric section."""

    def test_returns_list_of_dicts(self, tmp_path):
        _write_local_geom_file(tmp_path)
        params = _make_mock_params()
        result = Geometry(str(tmp_path), ["_0001"], params)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_kind_is_tracer_efit(self, tmp_path):
        _write_local_geom_file(tmp_path)
        params = _make_mock_params()
        geom = Geometry(str(tmp_path), ["_0001"], params)[0]
        assert geom["kind"] == "tracer_efit"

    def test_namelist_values_parsed(self, tmp_path):
        _write_local_geom_file(tmp_path)
        params = _make_mock_params()
        geom = Geometry(str(tmp_path), ["_0001"], params)[0]
        assert geom["local"]["q0"] == pytest.approx(1.4)
        assert geom["local"]["shat"] == pytest.approx(0.8)
        assert geom["local"]["trpeps"] == pytest.approx(0.18)

    def test_metric_cxy_cy_from_namelist(self, tmp_path):
        _write_local_geom_file(tmp_path)
        params = _make_mock_params()
        geom = Geometry(str(tmp_path), ["_0001"], params)[0]
        assert geom["metric"]["C_xy"] == pytest.approx(0.5)
        assert geom["metric"]["C_y"] == pytest.approx(1.0)

    def test_numeric_arrays_shape(self, tmp_path):
        nz = NZ
        _write_local_geom_file(tmp_path, nz=nz)
        params = _make_mock_params(nz)
        geom = Geometry(str(tmp_path), ["_0001"], params)[0]
        assert geom["Bfield"].shape == (nz,)
        assert geom["Jacobian"].shape == (nz,)
        for key in ("dBdx", "dBdy", "dBdz"):
            assert geom[key].shape == (nz,)

    def test_metric_arrays_shape(self, tmp_path):
        nz = NZ
        _write_local_geom_file(tmp_path, nz=nz)
        params = _make_mock_params(nz)
        geom = Geometry(str(tmp_path), ["_0001"], params)[0]
        for key in ("gxx", "gxy", "gxz", "gyy", "gyz", "gzz", "dxdR", "dxdZ"):
            assert geom["metric"][key].shape == (nz,), f"metric[{key!r}] wrong shape"

    def test_shape_arrays_present(self, tmp_path):
        _write_local_geom_file(tmp_path)
        params = _make_mock_params()
        geom = Geometry(str(tmp_path), ["_0001"], params)[0]
        for key in ("gR", "gZ", "gPhi"):
            assert key in geom["shape"]

    def test_multiple_extensions(self, tmp_path):
        _write_local_geom_file(tmp_path, ext="_0001")
        _write_local_geom_file(tmp_path, ext="_0002")
        params = _make_mock_params()
        result = Geometry(str(tmp_path), ["_0001", "_0002"], params)
        assert len(result) == 2

    def test_single_string_extension(self, tmp_path):
        _write_local_geom_file(tmp_path)
        params = _make_mock_params()
        result = Geometry(str(tmp_path), "_0001", params)
        assert len(result) == 1


class TestCurvatureComputation:
    """Geometry must populate geom['curv'] with K_x, K_y, sloc."""

    def test_curv_keys_present(self, tmp_path):
        _write_local_geom_file(tmp_path)
        params = _make_mock_params()
        geom = Geometry(str(tmp_path), ["_0001"], params)[0]
        assert "curv" in geom
        for key in ("K_x", "K_y", "sloc"):
            assert key in geom["curv"], f"curv[{key!r}] missing"

    def test_Kx_Ky_shape(self, tmp_path):
        nz = NZ
        _write_local_geom_file(tmp_path, nz=nz)
        params = _make_mock_params(nz)
        geom = Geometry(str(tmp_path), ["_0001"], params)[0]
        assert geom["curv"]["K_x"].shape == (nz,)
        assert geom["curv"]["K_y"].shape == (nz,)

    def test_sloc_not_nan_for_local(self, tmp_path):
        nz = NZ
        _write_local_geom_file(tmp_path, nz=nz)
        params = _make_mock_params(nz)
        geom = Geometry(str(tmp_path), ["_0001"], params)[0]
        sloc = geom["curv"]["sloc"]
        # sloc should be a finite array, not a scalar NaN
        assert not np.all(np.isnan(sloc)), "sloc is entirely NaN for local geometry"


class TestLocalShear:
    """sloc = d(gxy/gxx)/dz, computed with np.gradient (no scipy)."""

    @staticmethod
    def _curv(gxy, gxx, nz, edge_opt=0):
        from genetools.io.geometry import _compute_curvature
        zeros = np.zeros_like(gxx)
        geom = {
            "metric": {"gxx": gxx, "gxy": gxy, "gxz": zeros,
                       "gyy": np.ones_like(gxx), "gyz": zeros},
            "dBdx": zeros, "dBdy": zeros, "dBdz": zeros,
        }
        params = {"box": {"nz0": nz},
                  "geometry": {"n_pol": 1, "edge_opt": edge_opt}}
        return _compute_curvature(geom, params)["sloc"]

    def test_constant_shear_recovered_local(self):
        """gxy/gxx = shat*z  ->  sloc == shat everywhere."""
        from genetools.io._zgrid import build_zgrid
        nz, shat = 32, 0.8
        z = build_zgrid(nz, 1, 0)
        gxx = np.ones(nz)
        sloc = self._curv(shat * z, gxx, nz)
        assert sloc.shape == (nz,)
        # exact for a linear profile, including the one-sided end points
        np.testing.assert_allclose(sloc, shat, rtol=1e-12)

    def test_constant_shear_recovered_global(self):
        """Global geometry: the ratio is (nx, nz) and z is the last axis."""
        from genetools.io._zgrid import build_zgrid
        nx, nz = 5, 32
        z = build_zgrid(nz, 1, 0)
        shat = np.linspace(0.5, 1.5, nx)[:, np.newaxis]   # x-dependent shear
        gxx = np.ones((nx, nz))
        sloc = self._curv(shat * z[np.newaxis, :], gxx, nz)
        assert sloc.shape == (nx, nz)
        np.testing.assert_allclose(sloc, np.broadcast_to(shat, (nx, nz)),
                                   rtol=1e-12)

    def test_nonuniform_grid_uses_z_spacing(self):
        """edge_opt != 0 clusters the z grid; the derivative must still be exact
        for a linear profile, which requires spacing-aware differencing."""
        from genetools.io._zgrid import build_zgrid
        nz, shat, edge = 32, 0.8, 4.0
        z = build_zgrid(nz, 1, edge)
        assert not np.allclose(np.diff(z), np.diff(z)[0]), "grid should be non-uniform"
        sloc = self._curv(shat * z, np.ones(nz), nz, edge_opt=edge)
        np.testing.assert_allclose(sloc, shat, rtol=1e-10)

    def test_scalar_nan_when_single_z_point(self):
        sloc = self._curv(np.array([1.0]), np.array([1.0]), nz=1)
        assert np.isscalar(sloc) or np.ndim(sloc) == 0
        assert np.isnan(sloc)

    def test_mismatched_z_length_falls_back_to_nan(self):
        """A z grid that disagrees with the data length yields all-NaN, not a crash."""
        nz = 32
        # params say nz0=16 while the arrays carry 32 points
        sloc = self._curv(np.ones(nz), np.ones(nz), nz=16)
        assert np.all(np.isnan(sloc))


class TestAreaComputation:
    """Geometry must populate geom['area'] with Area and dVdx."""

    def test_area_keys_present(self, tmp_path):
        _write_local_geom_file(tmp_path)
        params = _make_mock_params()
        geom = Geometry(str(tmp_path), ["_0001"], params)[0]
        assert "area" in geom
        assert "Area" in geom["area"]
        assert "dVdx" in geom["area"]

    def test_area_positive(self, tmp_path):
        _write_local_geom_file(tmp_path)
        params = _make_mock_params()
        geom = Geometry(str(tmp_path), ["_0001"], params)[0]
        assert geom["area"]["Area"] > 0

    def test_dVdx_positive(self, tmp_path):
        _write_local_geom_file(tmp_path)
        params = _make_mock_params()
        geom = Geometry(str(tmp_path), ["_0001"], params)[0]
        assert geom["area"]["dVdx"] > 0

    def test_area_is_scalar_for_local(self, tmp_path):
        _write_local_geom_file(tmp_path)
        params = _make_mock_params()
        geom = Geometry(str(tmp_path), ["_0001"], params)[0]
        # For local geometry both should be 0-d (scalar, not an array)
        assert np.ndim(geom["area"]["Area"]) == 0
        assert np.ndim(geom["area"]["dVdx"]) == 0


class TestMissingFileError:
    """Geometry raises FileNotFoundError when the file does not exist."""

    def test_missing_file_raises(self, tmp_path):
        params = _make_mock_params()
        with pytest.raises(FileNotFoundError):
            Geometry(str(tmp_path), ["_0001"], params)

    def test_error_message_contains_path(self, tmp_path):
        params = _make_mock_params()
        with pytest.raises(FileNotFoundError, match="tracer_efit"):
            Geometry(str(tmp_path), ["_0001"], params)
