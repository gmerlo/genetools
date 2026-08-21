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

    def test_h5_twins_collapse_to_one_suffix(self, tmp_path):
        """A run with HDF5 output writes nrg<ext> and nrg<ext>.h5 per segment."""
        (tmp_path / "all_params_0001.h5").touch()
        for name in ["nrg_0001", "nrg_0001.h5", "nrg_0002", "nrg_0002.h5"]:
            (tmp_path / name).touch()
        assert set_runs(tmp_path) == ["_0001", "_0002"]

    def test_h5_only_segments_are_found(self, tmp_path):
        """A segment present only as .h5 still counts once."""
        for name in ["nrg.dat", "nrg.dat.h5"]:
            (tmp_path / name).touch()
        assert set_runs(tmp_path) == [".dat"]

    def test_every_distinct_segment_is_reported(self, tmp_path):
        """Distinct segments are never dropped, whatever else the folder holds."""
        (tmp_path / "all_params_0001.h5").touch()
        for name in ["nrg_0001", "nrg_0002", "nrg_0003", "nrg_0004"]:
            (tmp_path / name).touch()
        assert set_runs(tmp_path) == ["_0001", "_0002", "_0003", "_0004"]

    def test_non_padded_suffix_is_preserved(self, tmp_path):
        """GENE-3D scans produce _1, _2, ...; the literal suffix must survive."""
        for name in ["nrg_1", "nrg_2", "nrg_10"]:
            (tmp_path / name).touch()
        # Sorted numerically, not lexicographically, and not reformatted.
        assert set_runs(tmp_path) == ["_1", "_2", "_10"]

    def test_pathlib_input(self, tmp_path):
        (tmp_path / "nrg_0001").touch()
        # Should accept a Path object, not just str
        result = set_runs(tmp_path)
        assert "_0001" in result
