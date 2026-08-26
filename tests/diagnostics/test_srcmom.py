"""
Tests for `SrcMom`, which covers both global geometries.

It is not a GENE-3D diagnostic: GENE writes source moments for a global run too,
and refuses for a flux tube — `diag_df.F90` forces `istep_srcmom = 0` when
`xy_local`, printing "istep_srcmom > 0 not possible in local simulations". So the
diagnostic refuses there because the file cannot exist, not because the code path
is missing.
"""

import numpy as np
import pytest

from genetools.diagnostics import SrcMom
from genetools.run import Run
from tests.gene_fixture import (SRCMOM_LABELS, make_xglobal_run,
                                   write_srcmom)


@pytest.fixture
def xglobal_srcmom(tmp_path):
    nx0, n_times = 12, 4
    folder = make_xglobal_run(tmp_path / "run", nx0=nx0)
    times, truth = write_srcmom(folder, nx0=nx0, n_times=n_times)
    return Run(folder), times, truth


class TestXGlobalSrcMom:

    def test_run_is_x_global(self, xglobal_srcmom):
        run, _, _ = xglobal_srcmom
        assert run.geometry_kind == "x_global"
        assert not run.is_3d

    def test_all_nine_gene_source_moments_are_read(self, xglobal_srcmom):
        """GENE writes nine; GENE-3D writes six, omitting `f0_term`."""
        run, _, _ = xglobal_srcmom
        ds = SrcMom(run).dataset()
        assert len(ds.data_vars) == 9
        for label in SRCMOM_LABELS:
            assert label in ds, label
        assert any(v.startswith("f0_term") for v in ds.data_vars)

    def test_dims_and_species(self, xglobal_srcmom):
        run, times, _ = xglobal_srcmom
        ds = SrcMom(run).dataset()
        assert ds["ck_heat_M00"].dims == ("species", "time", "x")
        assert list(ds["species"].values) == list(run.species)
        np.testing.assert_allclose(np.asarray(ds["time"]), times)

    def test_values_match_the_file(self, xglobal_srcmom):
        run, _, truth = xglobal_srcmom
        ds = SrcMom(run).dataset()
        for name, per in truth.items():
            for label, arr in per.items():
                got = np.asarray(ds[label].sel(species=name))
                np.testing.assert_allclose(got, arr, rtol=1e-6)

    def test_time_window_selects_snapshots(self, xglobal_srcmom):
        run, times, _ = xglobal_srcmom
        ds = SrcMom(run).dataset(t=(times[1], -1))
        assert ds.sizes["time"] == len(times) - 1

    def test_plot_runs(self, xglobal_srcmom, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        run, _, _ = xglobal_srcmom
        SrcMom(run).plot()
        plt.close("all")

    def test_missing_files_name_the_restriction(self, tmp_path):
        """No srcmom written at all: the message says when GENE writes them."""
        folder = make_xglobal_run(tmp_path / "run")
        with pytest.raises(FileNotFoundError, match="global, nonlinear"):
            SrcMom(Run(folder)).compute()

    def test_unformatted_only_is_reported_not_misparsed(self, tmp_path):
        """
        The Fortran-unformatted layout is its own — a time record then one
        record of (nx0, 3) reals per source term — and is not what BinaryReader
        decodes, so it must be reported rather than silently mangled.
        """
        folder = make_xglobal_run(tmp_path / "run", species=("ions",))
        (folder / "srcmom_ions.dat").write_bytes(b"\x00" * 64)
        with pytest.raises(FileNotFoundError, match="write_h5"):
            SrcMom(Run(folder)).compute()


class TestFluxTubeRefuses:

    def test_refuses_for_a_flux_tube(self, tmp_path):
        from tests.conftest import MINIMAL_PARAMS
        (tmp_path / "parameters").write_text(MINIMAL_PARAMS)
        (tmp_path / "nrg").touch()
        run = Run(tmp_path, ext=[""])
        assert run.geometry_kind == "flux_tube"
        with pytest.raises(NotImplementedError):
            run.srcmom

    def test_supported_names_both_global_geometries(self):
        assert set(SrcMom.supported) == {"x_global", "xy_global"}
