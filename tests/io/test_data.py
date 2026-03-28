"""Tests for genetools.io.data."""

import io
import struct
import mmap

import numpy as np
import pytest

from genetools.io.data import BinaryReader
from tests.conftest import make_params, make_binary_file, write_fortran_record


class TestBinaryReader:
    def _make_reader(self, tmp_path, **kwargs):
        ni = kwargs.pop("ni", 2)
        nj = kwargs.pop("nj", 2)
        nk = kwargs.pop("nk", 2)
        n_arrays = kwargs.pop("n_arrays", 1)
        n_iters = kwargs.pop("n_iters", 3)
        fpath, expected_times, expected_arrays = make_binary_file(
            tmp_path, n_iters=n_iters, ni=ni, nj=nj, nk=nk, n_arrays=n_arrays
        )
        params = make_params(nx0=ni, nky0=nj, nz0=nk, n_fields=n_arrays)
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
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            idx1 = reader._get_record_index(mm)
            idx2 = reader._get_record_index(mm)
            mm.close()
        assert idx1 is idx2  # same object -> cached

    def test_single_precision(self, tmp_path):
        """Verify that single-precision files are parsed correctly."""
        ni, nj, nk = 2, 2, 2
        buf = io.BytesIO()
        t = np.float32(1.5)
        write_fortran_record(buf, struct.pack("<f", float(t)))
        arr = np.ones(ni * nj * nk, dtype=np.complex64)
        write_fortran_record(buf, arr.tobytes())
        fpath = tmp_path / "field_sp"
        fpath.write_bytes(buf.getvalue())
        params = make_params(nx0=ni, nky0=nj, nz0=nk, n_fields=1, precision="single")
        reader = BinaryReader("field", str(tmp_path) + "/", "_sp", params)
        times = reader.read_all_times()
        assert times.dtype == np.float32
        assert pytest.approx(times[0]) == 1.5


class TestBPReaderImportError:
    def test_raises_import_error_when_adios2_missing(self):
        """When adios2 is not installed, BPReader.__init__ should raise ImportError."""
        import sys
        import importlib
        # Temporarily hide adios2 if it happens to be installed
        adios2_backup = sys.modules.pop("adios2", None)
        import genetools.io.data as data_mod
        importlib.reload(data_mod)
        try:
            with pytest.raises(ImportError):
                data_mod.BPReader("field", "/tmp/", "_0001", make_params())
        finally:
            if adios2_backup is not None:
                sys.modules["adios2"] = adios2_backup
            importlib.reload(data_mod)
