"""Tests for genetools.io.params."""

import pytest

from genetools.io.params import Params
from tests.conftest import write_param_file, MINIMAL_PARAMS


class TestParams:
    def test_load_single_file(self, tmp_path):
        f = tmp_path / "parameters"
        write_param_file(f, MINIMAL_PARAMS)
        p = Params(f)
        assert len(p.params_list) == 1

    def test_get_returns_dict(self, tmp_path):
        f = tmp_path / "parameters"
        write_param_file(f, MINIMAL_PARAMS)
        d = Params(f).get(0)
        assert isinstance(d, dict)

    def test_species_loaded_as_list(self, tmp_path):
        f = tmp_path / "parameters"
        write_param_file(f, MINIMAL_PARAMS)
        d = Params(f).get()
        assert isinstance(d["species"], list)
        assert len(d["species"]) == 2

    def test_box_values(self, tmp_path):
        f = tmp_path / "parameters"
        write_param_file(f, MINIMAL_PARAMS)
        d = Params(f).get()
        assert d["box"]["nx0"] == 16
        assert d["box"]["nky0"] == 4

    def test_derived_units_computed(self, tmp_path):
        f = tmp_path / "parameters"
        write_param_file(f, MINIMAL_PARAMS)
        d = Params(f).get()
        assert "cref" in d["units"]
        assert "rhoref" in d["units"]
        assert d["units"]["cref"] > 0

    def test_defaults_applied(self, tmp_path):
        f = tmp_path / "parameters"
        write_param_file(f, MINIMAL_PARAMS)
        d = Params(f).get()
        # 'collision_op' is a default that isn't in MINIMAL_PARAMS
        assert d["general"]["collision_op"] == "none"

    def test_fc_lines_commented_out(self, tmp_path):
        """FC continuation lines should not cause a parse error."""
        content = MINIMAL_PARAMS + "\nFC some continuation\n"
        f = tmp_path / "parameters"
        write_param_file(f, content)
        # Should not raise
        Params(f).get()

    def test_folder_glob(self, tmp_path):
        write_param_file(tmp_path / "parameters_0001", MINIMAL_PARAMS)
        write_param_file(tmp_path / "parameters_0002", MINIMAL_PARAMS)
        p = Params(tmp_path, extensions=["_0001", "_0002"])
        assert len(p.params_list) == 2

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Params(tmp_path / "nonexistent_file")

    def test_tolist(self, tmp_path):
        f = tmp_path / "parameters"
        write_param_file(f, MINIMAL_PARAMS)
        lst = Params(f).tolist()
        assert isinstance(lst, list) and len(lst) == 1
