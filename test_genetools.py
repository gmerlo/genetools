"""
tests/test_genetools.py — Unit tests for the genetools package.

Run with:
    pytest tests/ -v
or with coverage:
    pytest tests/ -v --cov=genetools --cov-report=term-missing
"""

import io
import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_params(
    n_spec=1,
    n_fields=2,
    nrgcols=10,
    nx0=4,
    nky0=3,
    nz0=8,
    precision="double",
    y_local=True,
):
    """Build a minimal params dict that satisfies all readers."""
    return {
        "box": {"n_spec": n_spec, "nx0": nx0, "nky0": nky0, "nz0": nz0},
        "info": {"n_fields": n_fields, "n_moms": 9, "nrgcols": nrgcols, "precision": precision},
        "general": {"y_local": y_local, "x_local": True},
        "species": [{"name": f"sp{i}", "dens": 1.0, "temp": 1.0, "charge": 1.0}
                    for i in range(n_spec)],
    }


# ===========================================================================
# 1. utils.set_runs
# ===========================================================================

class TestSetRuns:
    def test_empty_folder_returns_empty_list(self, tmp_path):
        from genetools.utils import set_runs
        assert set_runs(tmp_path) == []

    def test_missing_folder_raises(self):
        from genetools.utils import set_runs
        with pytest.raises(FileNotFoundError):
            set_runs("/path/that/does/not/exist_xyz")

    def test_numeric_suffixes_sorted(self, tmp_path):
        from genetools.utils import set_runs
        for name in ["nrg_0003", "nrg_0001", "nrg_0002"]:
            (tmp_path / name).touch()
        result = set_runs(tmp_path)
        assert result == ["_0001", "_0002", "_0003"]

    def test_dat_appended_last(self, tmp_path):
        from genetools.utils import set_runs
        (tmp_path / "nrg_0001").touch()
        (tmp_path / "nrg.dat").touch()
        result = set_runs(tmp_path)
        assert result[-1] == ".dat"
        assert result[0] == "_0001"

    def test_exclusion_works(self, tmp_path):
        from genetools.utils import set_runs
        for name in ["nrg_0001", "nrg_0002"]:
            (tmp_path / name).touch()
        result = set_runs(tmp_path, exclude=["_0001"])
        assert "_0001" not in result
        assert "_0002" in result

    def test_h5_stride(self, tmp_path):
        """HDF5 mode should skip every other nrg file."""
        from genetools.utils import set_runs
        (tmp_path / "all_params_0001.h5").touch()
        for name in ["nrg_0001", "nrg_0002", "nrg_0003", "nrg_0004"]:
            (tmp_path / name).touch()
        result = set_runs(tmp_path)
        # With stride=2 and 4 files → 2 suffixes
        assert len(result) == 2

    def test_pathlib_input(self, tmp_path):
        from genetools.utils import set_runs
        (tmp_path / "nrg_0001").touch()
        # Should accept a Path object, not just str
        result = set_runs(tmp_path)
        assert "_0001" in result


# ===========================================================================
# 2. params.Params
# ===========================================================================

def _write_param_file(path: Path, content: str) -> None:
    path.write_text(content)


MINIMAL_PARAMS = """
&general
  x_local = .true.
  y_local = .true.
/

&box
  nx0 = 16
  nky0 = 4
  nz0 = 32
  n_spec = 2
/

&info
  precision = 'double'
  nrgcols = 10
/

&units
  Lref = 1.0
  Bref = 1.0
  nref = 1.0
  Tref = 1.0
  mref = 1.0
/

&species
  name = 'ions'
  charge = 1.0
  mass = 1.0
  temp = 1.0
/

&species
  name = 'electrons'
  charge = -1.0
  mass = 0.000272
  temp = 1.0
/
"""


class TestParams:
    def test_load_single_file(self, tmp_path):
        from genetools.params import Params
        f = tmp_path / "parameters"
        _write_param_file(f, MINIMAL_PARAMS)
        p = Params(f)
        assert len(p.params_list) == 1

    def test_get_returns_dict(self, tmp_path):
        from genetools.params import Params
        f = tmp_path / "parameters"
        _write_param_file(f, MINIMAL_PARAMS)
        d = Params(f).get(0)
        assert isinstance(d, dict)

    def test_species_loaded_as_list(self, tmp_path):
        from genetools.params import Params
        f = tmp_path / "parameters"
        _write_param_file(f, MINIMAL_PARAMS)
        d = Params(f).get()
        assert isinstance(d["species"], list)
        assert len(d["species"]) == 2

    def test_box_values(self, tmp_path):
        from genetools.params import Params
        f = tmp_path / "parameters"
        _write_param_file(f, MINIMAL_PARAMS)
        d = Params(f).get()
        assert d["box"]["nx0"] == 16
        assert d["box"]["nky0"] == 4

    def test_derived_units_computed(self, tmp_path):
        from genetools.params import Params
        f = tmp_path / "parameters"
        _write_param_file(f, MINIMAL_PARAMS)
        d = Params(f).get()
        assert "cref" in d["units"]
        assert "rhoref" in d["units"]
        assert d["units"]["cref"] > 0

    def test_defaults_applied(self, tmp_path):
        from genetools.params import Params
        f = tmp_path / "parameters"
        _write_param_file(f, MINIMAL_PARAMS)
        d = Params(f).get()
        # 'collision_op' is a default that isn't in MINIMAL_PARAMS
        assert d["general"]["collision_op"] == "none"

    def test_fc_lines_commented_out(self, tmp_path):
        """FC continuation lines should not cause a parse error."""
        from genetools.params import Params
        content = MINIMAL_PARAMS + "\nFC some continuation\n"
        f = tmp_path / "parameters"
        _write_param_file(f, content)
        # Should not raise
        Params(f).get()

    def test_folder_glob(self, tmp_path):
        from genetools.params import Params
        _write_param_file(tmp_path / "parameters_0001", MINIMAL_PARAMS)
        _write_param_file(tmp_path / "parameters_0002", MINIMAL_PARAMS)
        p = Params(tmp_path, extensions=["_0001", "_0002"])
        assert len(p.params_list) == 2

    def test_file_not_found(self, tmp_path):
        from genetools.params import Params
        with pytest.raises(FileNotFoundError):
            Params(tmp_path / "nonexistent_file")

    def test_tolist(self, tmp_path):
        from genetools.params import Params
        f = tmp_path / "parameters"
        _write_param_file(f, MINIMAL_PARAMS)
        lst = Params(f).tolist()
        assert isinstance(lst, list) and len(lst) == 1


# ===========================================================================
# 3. data.BinaryReader
# ===========================================================================

def _write_fortran_record(buf: io.BytesIO, payload: bytes) -> None:
    """Write a Fortran unformatted record (marker + payload + marker)."""
    n = len(payload)
    marker = struct.pack("<i", n)
    buf.write(marker + payload + marker)


def _make_binary_file(
    tmp_path: Path,
    n_iters: int = 3,
    ni: int = 2,
    nj: int = 2,
    nk: int = 2,
    n_arrays: int = 1,
    dtype=np.complex128,
    real_dtype=np.float64,
) -> tuple:
    """Create a synthetic Fortran binary file and return (path, times, arrays)."""
    buf = io.BytesIO()
    times = []
    arrays = []
    npts = ni * nj * nk
    for it in range(n_iters):
        t = float(it) * 0.5
        times.append(t)
        _write_fortran_record(buf, struct.pack("<d", t))
        it_arrays = []
        for _ in range(n_arrays):
            arr = np.random.randn(npts) + 1j * np.random.randn(npts)
            arr = arr.astype(dtype)
            _write_fortran_record(buf, arr.tobytes())
            it_arrays.append(arr.reshape((ni, nj, nk), order="F"))
        arrays.append(it_arrays)
    fpath = tmp_path / "field_0001"
    fpath.write_bytes(buf.getvalue())
    return fpath, np.array(times), arrays


class TestBinaryReader:
    def _make_reader(self, tmp_path, **kwargs):
        from genetools.data import BinaryReader
        ni = kwargs.pop("ni", 2)
        nj = kwargs.pop("nj", 2)
        nk = kwargs.pop("nk", 2)
        n_arrays = kwargs.pop("n_arrays", 1)
        n_iters = kwargs.pop("n_iters", 3)
        fpath, expected_times, expected_arrays = _make_binary_file(
            tmp_path, n_iters=n_iters, ni=ni, nj=nj, nk=nk, n_arrays=n_arrays
        )
        params = _make_params(nx0=ni, nky0=nj, nz0=nk, n_fields=n_arrays)
        reader = BinaryReader("field", str(tmp_path) + "/", "_0001", params)
        return reader, expected_times, expected_arrays

    def test_read_all_times_shape(self, tmp_path):
        reader, expected_times, _ = self._make_reader(tmp_path, n_iters=4)
        times = reader.read_all_times()
        assert times.shape == expected_times.shape

    def test_read_all_times_values(self, tmp_path):
        reader, expected_times, _ = self._make_reader(tmp_path, n_iters=3)
        times = reader.read_all_times()
        np.testing.assert_allclose(times, expected_times)

    def test_stream_selected_yields_correct_count(self, tmp_path):
        reader, _, _ = self._make_reader(tmp_path, n_iters=5)
        indices = [0, 2, 4]
        results = list(reader.stream_selected(indices))
        assert len(results) == len(indices)

    def test_stream_selected_time_values(self, tmp_path):
        reader, expected_times, _ = self._make_reader(tmp_path, n_iters=4)
        results = list(reader.stream_selected([1, 3]))
        assert pytest.approx(results[0][0]) == expected_times[1]
        assert pytest.approx(results[1][0]) == expected_times[3]

    def test_stream_selected_array_shape(self, tmp_path):
        ni, nj, nk = 3, 2, 4
        reader, _, _ = self._make_reader(tmp_path, ni=ni, nj=nj, nk=nk, n_iters=2)
        results = list(reader.stream_selected([0]))
        arr = results[0][1][0]
        assert arr.shape == (ni, nj, nk)

    def test_record_index_is_cached(self, tmp_path):
        """Second call to read_all_times should reuse the cached index."""
        reader, _, _ = self._make_reader(tmp_path, n_iters=2)
        with open(reader.filename, "rb") as f:
            import mmap
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            idx1 = reader._get_record_index(mm)
            idx2 = reader._get_record_index(mm)
            mm.close()
        assert idx1 is idx2  # same object → cached

    def test_single_precision(self, tmp_path):
        """Verify that single-precision files are parsed correctly."""
        from genetools.data import BinaryReader
        ni, nj, nk = 2, 2, 2
        buf = io.BytesIO()
        t = np.float32(1.5)
        _write_fortran_record(buf, struct.pack("<f", float(t)))
        arr = np.ones(ni * nj * nk, dtype=np.complex64)
        _write_fortran_record(buf, arr.tobytes())
        fpath = tmp_path / "field_sp"
        fpath.write_bytes(buf.getvalue())
        params = _make_params(nx0=ni, nky0=nj, nz0=nk, n_fields=1, precision="single")
        reader = BinaryReader("field", str(tmp_path) + "/", "_sp", params)
        times = reader.read_all_times()
        assert times.dtype == np.float32
        assert pytest.approx(times[0]) == 1.5


# ===========================================================================
# 4. nrg.NrgReader
# ===========================================================================

def _write_nrg_file(path: Path, n_times: int, n_spec: int, n_cols: int) -> tuple:
    """Write a synthetic nrg file and return (times, data) arrays."""
    rng = np.random.default_rng(42)
    times = np.arange(n_times, dtype=float) * 0.1
    data = rng.random((n_times, n_spec, n_cols))
    lines = []
    for it in range(n_times):
        lines.append(f"{times[it]:.6e}")
        for sp in range(n_spec):
            lines.append("  ".join(f"{v:.6e}" for v in data[it, sp, :]))
    path.write_text("\n".join(lines) + "\n")
    # Reshape to expected output: (n_spec, n_cols, n_times)
    expected = np.transpose(data, (1, 2, 0))
    return times, expected


class TestNrgReader:
    def _make_reader(self, tmp_path, n_times=5, n_spec=2, n_cols=10):
        from genetools.nrg import NrgReader
        fpath = tmp_path / "nrg_0001"
        times, expected_data = _write_nrg_file(fpath, n_times, n_spec, n_cols)
        params = _make_params(n_spec=n_spec, nrgcols=n_cols)
        params["species"] = [{"name": f"sp{i}"} for i in range(n_spec)]
        reader = NrgReader(str(tmp_path), params)
        return reader, times, expected_data

    def test_detect_files(self, tmp_path):
        reader, _, _ = self._make_reader(tmp_path)
        assert len(reader.nrg_files) == 1

    def test_no_files_raises(self, tmp_path):
        from genetools.nrg import NrgReader
        params = _make_params()
        params["species"] = [{"name": "sp0"}]
        with pytest.raises(FileNotFoundError):
            NrgReader(str(tmp_path), params)

    def test_read_all_times_shape(self, tmp_path):
        reader, expected_times, _ = self._make_reader(tmp_path, n_times=5)
        times, _ = reader.read_all()
        assert times.shape == (5,)

    def test_read_all_times_values(self, tmp_path):
        reader, expected_times, _ = self._make_reader(tmp_path, n_times=4)
        times, _ = reader.read_all()
        np.testing.assert_allclose(times, expected_times, rtol=1e-5)

    def test_data_shape(self, tmp_path):
        n_spec, n_cols, n_times = 2, 10, 6
        reader, _, expected_data = self._make_reader(tmp_path, n_times=n_times,
                                                     n_spec=n_spec, n_cols=n_cols)
        _, data = reader.read_all()
        assert data.shape == (n_spec, n_cols, n_times)

    def test_data_values(self, tmp_path):
        reader, _, expected_data = self._make_reader(tmp_path, n_times=4)
        _, data = reader.read_all()
        np.testing.assert_allclose(data, expected_data, rtol=1e-5)

    def test_multiple_files_concatenated(self, tmp_path):
        from genetools.nrg import NrgReader
        n_cols = 10
        n_spec = 1
        for i, fname in enumerate(["nrg_0001", "nrg_0002"]):
            _write_nrg_file(tmp_path / fname, n_times=3, n_spec=n_spec, n_cols=n_cols)
        params = _make_params(n_spec=n_spec, nrgcols=n_cols)
        params["species"] = [{"name": "sp0"}]
        reader = NrgReader(str(tmp_path), params)
        times, data = reader.read_all()
        assert data.shape[2] == 6  # 3 + 3 times

    def test_plot_does_not_crash(self, tmp_path):
        """Smoke test: plot() should run without raising."""
        reader, _, _ = self._make_reader(tmp_path, n_times=3, n_spec=1)
        with patch("matplotlib.pyplot.show"):
            reader.plot()

    def test_plot_without_read_triggers_read(self, tmp_path):
        """plot() auto-calls read_all() if data is not yet loaded."""
        reader, _, _ = self._make_reader(tmp_path)
        assert reader.times is None
        with patch("matplotlib.pyplot.show"):
            reader.plot()
        assert reader.times is not None

    def test_require_data_raises(self, tmp_path):
        reader, _, _ = self._make_reader(tmp_path)
        with pytest.raises(ValueError, match="read_all"):
            reader.plot_fluxes()


# ===========================================================================
# 5. BPReader — import-error path
# ===========================================================================

class TestBPReaderImportError:
    def test_raises_import_error_when_adios2_missing(self):
        """When adios2 is not installed, BPReader.__init__ should raise ImportError."""
        import sys
        # Temporarily hide adios2 if it happens to be installed
        adios2_backup = sys.modules.pop("adios2", None)
        # Force re-import of data module without adios2
        import importlib
        import genetools.data as data_mod
        importlib.reload(data_mod)
        try:
            with pytest.raises(ImportError):
                data_mod.BPReader("field", "/tmp/", "_0001", _make_params())
        finally:
            if adios2_backup is not None:
                sys.modules["adios2"] = adios2_backup
            importlib.reload(data_mod)
