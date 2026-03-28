"""Tests for genetools.diagnostics.nrg."""

from unittest.mock import patch

import numpy as np
import pytest

from genetools.diagnostics.nrg import NrgReader
from tests.conftest import make_params, write_nrg_file


class TestNrgReader:
    def _make_reader(self, tmp_path, n_times=5, n_spec=2, n_cols=10):
        fpath = tmp_path / "nrg_0001"
        times, expected_data = write_nrg_file(fpath, n_times, n_spec, n_cols)
        params = make_params(n_spec=n_spec, nrgcols=n_cols)
        params["species"] = [{"name": f"sp{i}"} for i in range(n_spec)]
        reader = NrgReader(str(tmp_path), params)
        return reader, times, expected_data

    def test_detect_files(self, tmp_path):
        reader, _, _ = self._make_reader(tmp_path)
        assert len(reader.nrg_files) == 1

    def test_no_files_raises(self, tmp_path):
        params = make_params()
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
        n_cols = 10
        n_spec = 1
        for fname in ["nrg_0001", "nrg_0002"]:
            write_nrg_file(tmp_path / fname, n_times=3, n_spec=n_spec, n_cols=n_cols)
        params = make_params(n_spec=n_spec, nrgcols=n_cols)
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
