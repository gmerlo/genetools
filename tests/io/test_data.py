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


    def test_single_iteration_file(self, tmp_path):
        """B1 regression: single-iteration files must be read correctly."""
        ni, nj, nk, n_arrays = 2, 2, 2, 2
        reader, expected_times, expected_arrays = self._make_reader(
            tmp_path, ni=ni, nj=nj, nk=nk, n_arrays=n_arrays, n_iters=1
        )
        times = reader.read_all_times()
        assert len(times) == 1
        assert pytest.approx(times[0]) == expected_times[0]
        results = list(reader.stream_selected([0]))
        assert len(results) == 1
        assert len(results[0][1]) == n_arrays
        for k in range(n_arrays):
            np.testing.assert_allclose(results[0][1][k], expected_arrays[0][k])


class TestMultiSegmentReader:
    def test_repr_empty_timeline(self, tmp_path):
        """B7 regression: repr must not crash when timeline is empty."""
        from genetools.io.data import MultiSegmentReader

        # Create a reader with real data, then force empty timeline
        _, _, _ = make_binary_file(tmp_path, n_iters=1, ni=2, nj=2, nk=2)
        params = make_params(nx0=2, nky0=2, nz0=2, n_fields=1)
        reader = BinaryReader("field", str(tmp_path) + "/", "_0001", params)
        msr = MultiSegmentReader([reader])
        # Force empty timeline to test the guard
        msr._global_times = np.array([], dtype=np.float64)
        msr._global_map = []
        r = repr(msr)
        assert "0 unique steps" in r

    def test_repr_with_data(self, tmp_path):
        """repr should show time range when data exists."""
        from genetools.io.data import MultiSegmentReader

        _, _, _ = make_binary_file(tmp_path, n_iters=3, ni=2, nj=2, nk=2)
        params = make_params(nx0=2, nky0=2, nz0=2, n_fields=1)
        reader = BinaryReader("field", str(tmp_path) + "/", "_0001", params)
        msr = MultiSegmentReader([reader])
        r = repr(msr)
        assert "3 unique steps" in r
        assert "t=[" in r


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


class TestSubrecords:
    """Fortran splits payloads larger than 2**31-1 bytes into subrecords.

    The leading marker is negative to mean "continues in the next subrecord".
    Production meshes reach this routinely — 512x4096x128 in double precision
    is 4 GiB per field record. Reading such a file with a single-marker parser
    walks the offset backwards past the start of the file and dies with
    ``IndexError``, or silently drops timesteps for smaller splits.
    """

    NI, NJ, NK, N_ARRAYS = 4, 2, 3, 2
    TIMES = (1.0, 2.0, 3.0)

    def _truth(self):
        rng = np.random.default_rng(0)
        shape = (self.NI, self.NJ, self.NK)
        return [[rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
                 for _ in range(self.N_ARRAYS)] for _ in self.TIMES]

    @staticmethod
    def _plain(payload):
        marker = struct.pack("<i", len(payload))
        return marker + payload + marker

    @staticmethod
    def _split(payload, chunk):
        """Write *payload* as sign-flagged subrecords of at most *chunk* bytes."""
        out = b""
        parts = [payload[i:i + chunk] for i in range(0, len(payload), chunk)]
        for j, part in enumerate(parts or [b""]):
            last = j == len(parts) - 1
            lead = len(part) if last else -len(part)
            trail = len(part) if j == 0 else -len(part)
            out += struct.pack("<i", lead) + part + struct.pack("<i", trail)
        return out

    def _write(self, tmp_path, ext, encode, truth):
        path = tmp_path / f"field{ext}"
        with open(path, "wb") as f:
            for t, arrays in zip(self.TIMES, truth):
                f.write(self._plain(struct.pack("<d", t)))
                for arr in arrays:
                    f.write(encode(arr.astype(np.complex128).tobytes(order="F")))
        return path

    def _reader(self, tmp_path, ext):
        params = make_params(nx0=self.NI, nky0=self.NJ, nz0=self.NK,
                             n_fields=self.N_ARRAYS)
        return BinaryReader("field", str(tmp_path) + "/", ext, params)

    @pytest.mark.parametrize("chunk", [None, 64, 17, 8, 1])
    def test_roundtrip_matches_ground_truth(self, tmp_path, chunk):
        """chunk=17 and 1 split values mid-complex, so spans must be joined
        as bytes before being viewed as complex."""
        truth = self._truth()
        encode = (self._plain if chunk is None
                  else (lambda p: self._split(p, chunk)))
        ext = f"_c{chunk}"
        self._write(tmp_path, ext, encode, truth)
        reader = self._reader(tmp_path, ext)

        np.testing.assert_allclose(reader.read_all_times(), self.TIMES)
        got = list(reader.stream_selected(range(len(self.TIMES))))
        assert len(got) == len(self.TIMES)
        for (t, arrays), t_exp, arrays_exp in zip(got, self.TIMES, truth):
            assert t == pytest.approx(t_exp)
            for arr, exp in zip(arrays, arrays_exp):
                assert arr.shape == (self.NI, self.NJ, self.NK)
                np.testing.assert_allclose(arr, exp)

    def test_huge_negative_marker_does_not_crash(self, tmp_path):
        """The reported failure: a >2 GiB subrecord marker sent the scan
        offset negative, slicing to an empty buffer and raising IndexError."""
        path = tmp_path / "field_0001"
        with open(path, "wb") as f:
            f.write(self._plain(struct.pack("<d", 1.0)))
            f.write(struct.pack("<i", -2147483644))   # continuation marker
            f.write(b"\x00" * 64)                     # payload never present
        times = self._reader(tmp_path, "_0001").read_all_times()
        assert times.size == 0        # truncated record dropped, no exception

    def test_truncated_final_record_is_dropped(self, tmp_path):
        truth = self._truth()
        path = self._write(tmp_path, "_0001", self._plain, truth)
        with open(path, "r+b") as f:
            f.truncate(path.stat().st_size - 40)
        times = self._reader(tmp_path, "_0001").read_all_times()
        np.testing.assert_allclose(times, self.TIMES[:-1])

    def test_trailing_bytes_ignored(self, tmp_path):
        """A killed write can leave a partial marker at the end of the file."""
        truth = self._truth()
        path = self._write(tmp_path, "_0001", self._plain, truth)
        with open(path, "ab") as f:
            f.write(b"\x01\x02\x03")
        np.testing.assert_allclose(
            self._reader(tmp_path, "_0001").read_all_times(), self.TIMES)
