"""
Tests for `VspSlice`, which covers every geometry.

`diag_vsp.F90` puts no geometry restriction on the velocity-space output, and
both codes write the same five quantities as the same `(nz0, nv0, nw0, n_spec)`
array — only GENE's `Q_es`/`Q_em` are `Q_ese`/`Q_eme` in GENE-3D, and those are
discovered from the file. So the diagnostic used to refuse on data it could
read.
"""

import numpy as np
import pytest

from genetools.diagnostics import VspSlice
from genetools.io.data import VSP_VARS, VSP_VARS_3D
from genetools.run import Run
from tests.gene_fixture import (VSP_LABELS, make_fluxtube_run,
                                make_xglobal_run, write_vsp)


@pytest.fixture
def headless(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


NZ, NV, NW, NSPEC, NT = 16, 8, 4, 2, 3


def _build(folder, ext=""):
    times, truth = write_vsp(folder, ext=ext, nz0=NZ, nv0=NV, nw0=NW,
                             n_spec=NSPEC, n_times=NT)
    return Run(folder), times, truth


@pytest.fixture(params=["flux_tube", "x_global"])
def gene_run(request, tmp_path):
    """The same vsp file under both regular-GENE geometries."""
    if request.param == "flux_tube":
        folder = make_fluxtube_run(tmp_path / "run", nz0=NZ, nv0=NV, nw0=NW)
        run, times, truth = _build(folder, ext=".dat")
    else:
        folder = make_xglobal_run(tmp_path / "run", nz0=NZ)
        run, times, truth = _build(folder, ext=".dat")
    assert run.geometry_kind == request.param
    return run, times, truth


class TestEveryGeometry:

    def test_all_five_quantities_are_read(self, gene_run):
        run, _, _ = gene_run
        ds = VspSlice(run).dataset()
        assert len(ds.data_vars) == 5
        for label in VSP_LABELS:
            assert label in ds, label

    def test_dims_are_the_velocity_space_grid(self, gene_run):
        run, times, _ = gene_run
        ds = VspSlice(run).dataset()
        assert ds["G_es"].dims == ("time", "z", "vpar", "mu", "species")
        assert ds.sizes["z"] == NZ
        assert ds.sizes["vpar"] == NV
        assert ds.sizes["mu"] == NW
        assert ds.sizes["species"] == NSPEC
        np.testing.assert_allclose(np.asarray(ds["time"]), times)

    def test_values_survive_the_futils_axis_flip(self, gene_run):
        """
        The fixture writes the reversed array the code would, so this checks
        `H5Reader` undoes the flip rather than merely producing right-shaped
        numbers.
        """
        run, _, truth = gene_run
        ds = VspSlice(run).dataset()
        for label, arr in truth.items():
            np.testing.assert_allclose(np.asarray(ds[label]), arr, rtol=1e-6)

    def test_plot_runs(self, gene_run, headless):
        run, _, _ = gene_run
        VspSlice(run).plot()

    def test_time_window(self, gene_run):
        run, times, _ = gene_run
        ds = VspSlice(run).dataset(t=(times[1], -1))
        assert ds.sizes["time"] == NT - 1


class TestMissingFiles:

    def test_no_file_names_the_switch(self, tmp_path):
        folder = make_fluxtube_run(tmp_path / "run")
        with pytest.raises(FileNotFoundError, match="istep_vsp"):
            VspSlice(Run(folder)).compute()

    def test_unformatted_only_is_reported(self, tmp_path):
        """
        GENE's write_std form is one record per snapshot holding the whole 5-D
        array — not what BinaryReader decodes, so it must be reported.
        """
        folder = make_fluxtube_run(tmp_path / "run")
        (folder / "vsp.dat").write_bytes(b"\x00" * 64)
        with pytest.raises(FileNotFoundError, match="write_h5"):
            VspSlice(Run(folder)).compute()


class TestNames:

    def test_supported_is_unrestricted(self):
        assert VspSlice.supported is None

    def test_the_two_codes_differ_only_in_the_em_heat_flux_names(self):
        assert VSP_VARS == ["G_es", "G_em", "Q_es", "Q_em", "<f_>"]
        assert VSP_VARS_3D == ["G_es", "G_em", "Q_ese", "Q_eme", "<f_>"]
        assert set(VSP_VARS) ^ set(VSP_VARS_3D) == {
            "Q_es", "Q_em", "Q_ese", "Q_eme"}
