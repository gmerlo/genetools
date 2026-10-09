# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Tests for ``NrgReader.summary`` / ``print_summary``.

The nrg file is written by hand here rather than taken from a fixture: the
point of most of these tests is that a *known* set of numbers on a *deliberately
uneven* time axis comes back with the right average, and the fixtures randomise
both.
"""

import numpy as np
import pytest

from genetools.diagnostics.nrg import NrgReader


# Uneven on purpose: GENE's dt is adaptive and nrg is written every istep_nrg
# *steps*, so a plain mean and a trapezoidal one differ here.
TIMES = [0.0, 1.0, 10.0]
N_COLS = 10


def _write_nrg(folder, values, times=TIMES, ext="_0001"):
    """
    values : (n_spec, n_cols, n_times)
    """
    values = np.asarray(values, dtype=float)
    lines = []
    for it, t in enumerate(times):
        lines.append(f"{t:13.6f}")
        for isp in range(values.shape[0]):
            lines.append("".join(f"{v:16.8e}" for v in values[isp, :, it]))
    (folder / f"nrg{ext}").write_text("\n".join(lines) + "\n")


def _params(n_spec=2, units=True, nrgcols=N_COLS):
    u = ({"Lref": 1.65, "Bref": 2.5, "Tref": 1.0, "nref": 5.0, "mref": 2.0,
          "Ggb": 3.0, "Qgb": 7.0, "Pgb": 11.0}
         if units else
         {"Lref": 1.0, "Bref": 1.0, "Tref": 1.0, "nref": 1.0, "mref": 1.0,
          "Ggb": 1.0, "Qgb": 1.0, "Pgb": 1.0})
    return {
        "box": {"n_spec": n_spec},
        "info": {"nrgcols": nrgcols},
        "species": [{"name": n} for n in ("ions", "electrons")][:n_spec],
        "units": u,
    }


def _reader(tmp_path, values, **kw):
    _write_nrg(tmp_path, values)
    return NrgReader(str(tmp_path), _params(**kw), extensions=["_0001"])


@pytest.fixture
def constant(tmp_path):
    """Every column constant in time, distinct per (species, column)."""
    vals = np.zeros((2, N_COLS, len(TIMES)))
    for isp in range(2):
        for ic in range(N_COLS):
            vals[isp, ic, :] = 1.0 + isp * 100 + ic
    return _reader(tmp_path, vals), vals


class TestAverage:

    def test_constant_trace_averages_to_itself(self, constant):
        reader, vals = constant
        ds = reader.summary()
        assert float(ds.Q_es.sel(species="ions")) == pytest.approx(vals[0, 6, 0])
        assert float(ds.Q_em.sel(species="electrons")) == pytest.approx(vals[1, 7, 0])

    def test_average_is_trapezoidal_not_a_plain_mean(self, tmp_path):
        # A ramp on an uneven axis: the two differ, and only one is right.
        vals = np.zeros((1, N_COLS, 3))
        vals[0, 6, :] = [0.0, 1.0, 10.0]      # Q_es == t
        reader = _reader(tmp_path, vals, n_spec=1)
        got = float(reader.summary().Q_es.sel(species="ions"))

        t = np.asarray(TIMES)
        expected = np.trapezoid(vals[0, 6], x=t) / (t[-1] - t[0])
        assert got == pytest.approx(expected)
        # The plain mean would be 11/3; make sure we did not get that.
        assert got != pytest.approx(vals[0, 6].mean())

    def test_std_is_over_the_window(self, constant):
        reader, _ = constant
        ds = reader.summary()
        assert float(ds.Q_es_std.sel(species="ions")) == pytest.approx(0.0)

    def test_single_sample_window(self, constant):
        reader, vals = constant
        ds = reader.summary(t=(TIMES[0], TIMES[0]))
        assert ds.attrs["n_samples"] == 1
        assert float(ds.Q_es.sel(species="ions")) == pytest.approx(vals[0, 6, 0])


class TestWindow:

    def test_negative_bound_means_unbounded(self, constant):
        reader, _ = constant
        ds = reader.summary(t=(1.0, -1))
        assert ds.attrs["t_start"] == pytest.approx(1.0)
        assert ds.attrs["t_stop"] == pytest.approx(TIMES[-1])

    def test_default_is_the_whole_run(self, constant):
        reader, _ = constant
        ds = reader.summary()
        assert ds.attrs["t_start"] == pytest.approx(TIMES[0])
        assert ds.attrs["t_stop"] == pytest.approx(TIMES[-1])
        assert ds.attrs["n_samples"] == len(TIMES)

    def test_empty_window_raises(self, constant):
        reader, _ = constant
        with pytest.raises(ValueError, match="no nrg output"):
            reader.summary(t=(1e9, 2e9))

    def test_scalar_t_is_a_start_bound(self, constant):
        reader, _ = constant
        ds = reader.summary(t=1.0)
        assert ds.attrs["t_start"] == pytest.approx(1.0)
        assert ds.attrs["n_samples"] == 2


class TestSI:

    def test_si_columns_use_the_right_reference(self, constant):
        reader, vals = constant
        ds = reader.summary()
        u = _params()["units"]
        for name, ref in (("Gamma_es", "Ggb"), ("Gamma_em", "Ggb"),
                          ("Q_es", "Qgb"), ("Q_em", "Qgb"),
                          ("Pi_es", "Pgb"), ("Pi_em", "Pgb")):
            assert float(ds[name + "_SI"].sel(species="ions")) == pytest.approx(
                float(ds[name].sel(species="ions")) * u[ref]), name

    def test_fluctuation_amplitudes_have_no_si_column(self, constant):
        reader, _ = constant
        ds = reader.summary()
        for name in ("n_sq", "u_par_sq", "T_par_sq", "T_perp_sq"):
            assert name in ds
            assert name + "_SI" not in ds, (
                f"{name} has no physical conversion and must not be relabelled")

    def test_no_reference_units_means_no_si_at_all(self, tmp_path):
        vals = np.ones((2, N_COLS, len(TIMES)))
        reader = _reader(tmp_path, vals, units=False)
        ds = reader.summary()
        assert ds.attrs["si"] is False
        assert not [v for v in ds.data_vars if v.endswith("_SI")]

    def test_si_units_are_labelled(self, constant):
        reader, _ = constant
        ds = reader.summary()
        assert ds["Q_es_SI"].attrs["units"] == "W m^-2"
        assert ds["Gamma_es_SI"].attrs["units"] == "1e19 m^-2 s^-1"
        assert ds["Pi_es_SI"].attrs["units"] == "N m^-2"


class TestTotals:

    def test_totals_are_es_plus_em(self, constant):
        reader, _ = constant
        ds = reader.summary()
        for base in ("Gamma", "Q", "Pi"):
            for sp in ("ions", "electrons"):
                assert float(ds[base + "_total"].sel(species=sp)) == pytest.approx(
                    float(ds[base + "_es"].sel(species=sp))
                    + float(ds[base + "_em"].sel(species=sp)))

    def test_totals_have_si_companions(self, constant):
        reader, _ = constant
        ds = reader.summary()
        u = _params()["units"]
        assert float(ds.Q_total_SI.sel(species="ions")) == pytest.approx(
            float(ds.Q_total.sel(species="ions")) * u["Qgb"])

    def test_missing_columns_give_no_total(self, tmp_path):
        # GENE-3D writes eight columns, so there is no Pi at all.
        vals = np.ones((2, 8, len(TIMES)))
        reader = _reader(tmp_path, vals, nrgcols=8)
        ds = reader.summary()
        assert "Q_total" in ds
        assert "Pi_total" not in ds
        assert "Pi_es" not in ds


class TestPrintSummary:

    def test_writes_a_table_with_both_unit_systems(self, constant, capsys):
        reader, _ = constant
        text = reader.print_summary()
        out = capsys.readouterr().out
        assert out == text
        assert "GENE units" in text and "SI" in text
        assert "ions" in text and "electrons" in text
        assert "Q_total" in text
        assert "trapezoidal" in text

    def test_writes_to_a_file(self, constant, tmp_path):
        reader, _ = constant
        path = tmp_path / "summary.txt"
        text = reader.print_summary(file=str(path))
        assert path.read_text() == text

    def test_says_so_when_there_are_no_reference_units(self, tmp_path):
        vals = np.ones((2, N_COLS, len(TIMES)))
        reader = _reader(tmp_path, vals, units=False)
        text = reader.print_summary()
        assert "gyro-Bohm only" in text

    def test_std_column_can_be_dropped(self, constant):
        reader, _ = constant
        assert "std" in reader.print_summary(std=True)
        assert "std" not in reader.print_summary(std=False)


class TestIntegrated:
    """``flux x dVdx`` -- the total through the surface. Flux tube only."""

    DVDX = 2.5

    def test_integrated_is_flux_times_dVdx(self, constant):
        reader, _ = constant
        ds = reader.summary(dVdx=self.DVDX)
        for name in ("Gamma_es", "Gamma_em", "Q_es", "Q_em", "Pi_es", "Pi_em"):
            assert float(ds[name + "_integrated"].sel(species="ions")) == (
                pytest.approx(float(ds[name].sel(species="ions")) * self.DVDX))

    def test_integrated_si_is_the_si_flux_times_dVdx(self, constant):
        reader, _ = constant
        ds = reader.summary(dVdx=self.DVDX)
        assert float(ds.Q_total_integrated_SI.sel(species="ions")) == (
            pytest.approx(float(ds.Q_total_SI.sel(species="ions")) * self.DVDX))

    def test_integrated_units_drop_the_per_area(self, constant):
        reader, _ = constant
        ds = reader.summary(dVdx=self.DVDX)
        assert ds["Q_es_integrated_SI"].attrs["units"] == "W"
        assert ds["Gamma_es_integrated_SI"].attrs["units"] == "1e19 s^-1"
        assert ds["Pi_es_integrated_SI"].attrs["units"] == "N"

    def test_totals_are_integrated_too(self, constant):
        reader, _ = constant
        ds = reader.summary(dVdx=self.DVDX)
        for base in ("Gamma", "Q", "Pi"):
            assert base + "_total_integrated" in ds

    def test_amplitudes_are_not_integrated(self, constant):
        reader, _ = constant
        ds = reader.summary(dVdx=self.DVDX)
        for name in ("n_sq", "u_par_sq", "T_par_sq", "T_perp_sq"):
            assert name + "_integrated" not in ds, (
                f"{name} is not a flux; multiplying it by an area means nothing")

    def test_absent_by_default(self, constant):
        reader, _ = constant
        ds = reader.summary()
        assert not [v for v in ds.data_vars if v.endswith("_integrated")]
        assert "dVdx" not in ds.attrs

    def test_dVdx_recorded_in_attrs(self, constant):
        reader, _ = constant
        assert reader.summary(dVdx=self.DVDX).attrs["dVdx"] == pytest.approx(
            self.DVDX)

    def test_a_radial_profile_is_rejected(self, constant):
        reader, _ = constant
        with pytest.raises(ValueError, match="must be a scalar"):
            reader.summary(dVdx=np.array([1.0, 2.0, 3.0]))

    def test_no_si_units_means_no_integrated_si(self, tmp_path):
        vals = np.ones((2, N_COLS, len(TIMES)))
        reader = _reader(tmp_path, vals, units=False)
        ds = reader.summary(dVdx=self.DVDX)
        assert "Q_es_integrated" in ds          # gyro-Bohm x m^2 is still real
        assert "Q_es_integrated_SI" not in ds

    def test_table_gains_a_column(self, constant):
        reader, _ = constant
        text = reader.print_summary(dVdx=self.DVDX)
        assert "x dVdx" in text
        assert f"{self.DVDX:.6g} m^2" in text
        assert "->  W" in text
        assert "x dVdx" not in reader.print_summary()


class TestThroughRun:
    """
    End-to-end through the ``Run`` facade.

    These exist because ``_BoundNrg._dVdx`` reached for ``run.geom``, which is a
    ``RunDiagnostic`` shortcut that ``_BoundNrg`` does not inherit -- ``Run``
    itself only has the per-segment ``geometry`` list. Nothing exercised the
    flux-tube path through ``Run``, so it raised ``AttributeError`` on first use.
    """

    def test_flux_tube_integrates_by_default(self, tmp_path):
        from genetools.run import Run
        from tests.gene_fixture import make_fluxtube_run

        make_fluxtube_run(tmp_path)
        run = Run(str(tmp_path))
        assert run.geometry_kind == "flux_tube"

        ds = run.nrg.summary()
        assert "dVdx" in ds.attrs
        assert float(ds.attrs["dVdx"]) == pytest.approx(
            float(run.geometry[0]["area"]["dVdx"]))
        assert float(ds.Q_total_integrated_SI.sel(species="ions")) == (
            pytest.approx(float(ds.Q_total_SI.sel(species="ions"))
                          * ds.attrs["dVdx"]))

    def test_flux_tube_table_renders(self, tmp_path):
        from genetools.run import Run
        from tests.gene_fixture import make_fluxtube_run

        make_fluxtube_run(tmp_path)
        text = Run(str(tmp_path)).nrg.print_summary(t=(0, -1))
        assert "x dVdx" in text and "->  W" in text

    def test_global_run_does_not_integrate_by_default(self, tmp_path):
        from genetools.run import Run
        from tests.gene_fixture import make_xglobal_run

        make_xglobal_run(tmp_path)
        run = Run(str(tmp_path))
        assert run.geometry_kind == "x_global"
        ds = run.nrg.summary()
        assert not [v for v in ds.data_vars if v.endswith("_integrated")]
        assert "dVdx" not in ds.attrs

    def test_global_run_refuses_integrate(self, tmp_path):
        from genetools.run import Run
        from tests.gene_fixture import make_xglobal_run

        make_xglobal_run(tmp_path)
        with pytest.raises(ValueError, match="only a flux tube"):
            Run(str(tmp_path)).nrg.summary(integrate=True)


class TestNrgcolsDefault:
    """
    GENE writes ``nrgcols`` into ``&info`` only when ``momentum_flux`` is on
    (``parameters_IO.F90:2175-2180``), so an ordinary run can omit it entirely
    while still writing the module default of ten columns (``diag.F90:114``).
    Before the default, ``NrgReader.__init__`` raised ``KeyError: 'nrgcols'``.
    """

    def test_absent_nrgcols_defaults_to_ten_for_gene(self, tmp_path):
        from genetools.io.params import Params
        from tests.gene_fixture import make_fluxtube_run

        make_fluxtube_run(tmp_path)
        path = tmp_path / "parameters.dat"
        text = path.read_text().replace("&info\n nrgcols = 10\n/\n", "")
        assert "nrgcols" not in text
        path.write_text(text)

        params = Params(str(tmp_path), [".dat"]).get(0)
        assert params["info"]["nrgcols"] == 10

    def test_gene3d_defaults_to_eight(self, tmp_path):
        from genetools.io.params import Params
        from tests.gene3d_fixture import make_gene3d_run

        make_gene3d_run(tmp_path)
        path = next(tmp_path.glob("parameters*"))
        text = "\n".join(l for l in path.read_text().splitlines()
                         if "nrgcols" not in l)
        path.write_text(text + "\n")

        params = Params(str(tmp_path), [path.name.replace("parameters", "")]).get(0)
        assert params["info"]["is_3d"] is True
        assert params["info"]["nrgcols"] == 8
