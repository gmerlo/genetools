"""Tests for genetools.io.utils."""

import pytest

from genetools.io.utils import set_runs


class TestSetRuns:
    def test_empty_folder_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            set_runs(tmp_path)

    def test_missing_folder_raises(self):
        with pytest.raises(FileNotFoundError):
            set_runs("/path/that/does/not/exist_xyz")

    def test_numeric_suffixes_sorted(self, tmp_path):
        for name in ["nrg_0003", "nrg_0001", "nrg_0002"]:
            (tmp_path / name).touch()
        result = set_runs(tmp_path)
        assert result == ["_0001", "_0002", "_0003"]

    def test_dat_appended_last(self, tmp_path):
        (tmp_path / "nrg_0001").touch()
        (tmp_path / "nrg.dat").touch()
        result = set_runs(tmp_path)
        assert result[-1] == ".dat"
        assert result[0] == "_0001"

    def test_exclusion_works(self, tmp_path):
        for name in ["nrg_0001", "nrg_0002"]:
            (tmp_path / name).touch()
        result = set_runs(tmp_path, exclude=["_0001"])
        assert "_0001" not in result
        assert "_0002" in result

    def test_h5_stride(self, tmp_path):
        """HDF5 mode should skip every other nrg file."""
        (tmp_path / "all_params_0001.h5").touch()
        for name in ["nrg_0001", "nrg_0002", "nrg_0003", "nrg_0004"]:
            (tmp_path / name).touch()
        result = set_runs(tmp_path)
        # With stride=2 and 4 files -> 2 suffixes
        assert len(result) == 2

    def test_pathlib_input(self, tmp_path):
        (tmp_path / "nrg_0001").touch()
        # Should accept a Path object, not just str
        result = set_runs(tmp_path)
        assert "_0001" in result
