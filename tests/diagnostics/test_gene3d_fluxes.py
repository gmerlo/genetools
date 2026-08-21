"""
Tests for the GENE-3D flux diagnostics.

The central test here is :meth:`TestFluxIdentity.test_ky_sum_reproduces_the_code_flux`.
GENE-3D computes its own turbulent fluxes and writes them to the moment file, so
a flux rebuilt from ``phi`` and the moments can be compared against the code's
answer instead of against an assumption. That pins, in one check, the ExB
velocity normalisation, the sign of the magnetic-flutter velocity, both
heat-flux integrands, the FFT normalisation and the Jacobian weighting.
"""

import numpy as np
import pytest

from genetools.diagnostics import Fluxes2D, Spectra
from genetools.diagnostics import _gene3d as c
from genetools.diagnostics._gene3d import flux_geomfac
from genetools.run import Run
from tests.gene3d_fixture import make_gene3d_run


@pytest.fixture
def noisy_run(tmp_path):
    """Independent noise everywhere — fine for plumbing, shapes and units."""
    return make_gene3d_run(tmp_path / "run", n_times=4)


@pytest.fixture
def physical_run(tmp_path):
    """Fluxes derived from the fields exactly as diag_3d.F90 derives them."""
    return make_gene3d_run(tmp_path / "run", n_times=3, physical=True)


# ---------------------------------------------------------------------------
# The identity that pins every normalisation
# ---------------------------------------------------------------------------

class TestFluxIdentity:

    @pytest.mark.parametrize("norm_flux_projection", [True, False])
    def test_ky_sum_reproduces_the_code_flux(self, tmp_path,
                                             norm_flux_projection):
        """
        Summing a reconstructed spectrum over ky must give back the flux
        GENE-3D wrote.

        Both settings of ``norm_flux_projection`` are exercised because the flag
        changes ``flux_geomfac`` by a factor ``1/sqrt(g^xx)`` — a factor neither
        branch of the reference GUI applies.
        """
        g = make_gene3d_run(tmp_path / "run", n_times=3, physical=True,
                            norm_flux_projection=norm_flux_projection)
        diag = Spectra(Run(g.folder), buffer_frac=0.0)
        diag.compute()

        assert diag.consistency, "no fluxes were checked"
        for label, ratio in diag.consistency.items():
            # Data is stored as float32, so round-off is the only budget.
            assert ratio == pytest.approx(1.0, abs=1e-5), label

    def test_all_four_fluxes_are_checked(self, physical_run):
        """Including Q_em, which the reference GUI reports as identically zero."""
        diag = Spectra(Run(physical_run.folder), buffer_frac=0.0)
        diag.compute()
        for spec in physical_run.species:
            for flux in ("Gamma_es", "Gamma_em", "Q_es", "Q_em"):
                assert f"{spec}/{flux}" in diag.consistency

    def test_em_heat_flux_is_not_zero(self, physical_run):
        """
        Q_em needs q_par + q_perp, which GENE-3D writes but the reference GUI
        omits from its variable map — leaving its Q_em hard-zeroed.
        """
        ds = Spectra(Run(physical_run.folder), buffer_frac=0.0).dataset()
        assert np.any(np.abs(np.asarray(ds["Q_em_ky"])) > 0)

    def test_a_wrong_normalisation_is_caught(self, physical_run, monkeypatch):
        """
        The consistency check has to actually fail when the reconstruction is
        wrong, or it is decoration. Divide the ExB velocity by Bref — the
        reference GUI's x-global convention — and the ratio must move.
        """
        run = Run(physical_run.folder)
        bref = float(run.params.get(0)["units"]["Bref"])
        assert bref != 1.0, "fixture needs Bref != 1 for this to bite"

        good = c.exb_velocity_ky

        def wrong(phi, ky, geomfac):
            return good(phi, ky, geomfac) / bref

        monkeypatch.setattr(c, "exb_velocity_ky", wrong)
        diag = Spectra(run, buffer_frac=0.0)
        with pytest.warns(RuntimeWarning, match="ky-summed flux"):
            diag.compute()
        es = [v for k, v in diag.consistency.items() if "_es" in k]
        assert all(r == pytest.approx(1.0 / bref, rel=1e-4) for r in es)


# ---------------------------------------------------------------------------
# flux_geomfac
# ---------------------------------------------------------------------------

class TestFluxGeomfac:

    def test_without_projection_it_is_one_over_c_xy(self, tmp_path):
        g = make_gene3d_run(tmp_path / "run", norm_flux_projection=False)
        run = Run(g.folder)
        fac = flux_geomfac(run.geometry[0], run.params.get(0))
        expected = 1.0 / g.geometry["C_xy"][:, None, None]
        assert np.allclose(fac, np.broadcast_to(expected, fac.shape))

    def test_projection_adds_the_sqrt_gxx_factor(self, tmp_path):
        g = make_gene3d_run(tmp_path / "run", norm_flux_projection=True)
        run = Run(g.folder)
        fac = flux_geomfac(run.geometry[0], run.params.get(0))
        expected = (1.0 / g.geometry["C_xy"][:, None, None]
                    / np.sqrt(g.geometry["g^xx"]))
        assert np.allclose(fac, expected)


# ---------------------------------------------------------------------------
# Radial flux profiles
# ---------------------------------------------------------------------------

class TestFluxProfiles3D:

    def test_dataset_shape_and_variables(self, noisy_run):
        ds = Fluxes2D(Run(noisy_run.folder)).dataset()
        assert dict(ds.sizes) == {"species": 2, "time": 4, "x": noisy_run.nx0}
        for name in ("Gamma_es", "Gamma_em", "Q_es", "Q_em",
                     "Q_total", "Q_integrated"):
            assert name in ds

    def test_values_are_the_jacobian_weighted_surface_average(self, noisy_run):
        run = Run(noisy_run.folder)
        ds = Fluxes2D(run).dataset()
        J = run.geometry[0]["Jacobian"]
        spec = next(s for s in run.params.get(0)["species"]
                    if s["name"] == "ions")
        raw = noisy_run.moments["ions"]["Q_es"][2].astype(np.float32)
        expected = (np.average(raw, weights=J, axis=(1, 2))
                    * spec["dens"] * spec["temp"])
        got = np.asarray(ds["Q_es"].sel(species="ions").isel(time=2))
        assert np.allclose(got, expected, rtol=1e-5)

    def test_em_and_es_share_the_species_normalisation(self, noisy_run):
        """
        GENE-3D's own diag_prof applies dens to both particle fluxes and
        dens*temp to both heat fluxes. The reference GUI normalises only the
        electrostatic terms, so its EM fluxes are inconsistent with the code's
        profile_<species> output.
        """
        run = Run(noisy_run.folder)
        ds = Fluxes2D(run).dataset()
        J = run.geometry[0]["Jacobian"]
        spec = next(s for s in run.params.get(0)["species"]
                    if s["name"] == "ions")
        raw = noisy_run.moments["ions"]["Q_em"][0].astype(np.float32)
        expected = (np.average(raw, weights=J, axis=(1, 2))
                    * spec["dens"] * spec["temp"])
        got = np.asarray(ds["Q_em"].sel(species="ions").isel(time=0))
        assert np.allclose(got, expected, rtol=1e-5)

    def test_si_companions_carry_units(self, noisy_run):
        ds = Fluxes2D(Run(noisy_run.folder)).dataset()
        assert ds["Q_es_SI"].attrs["units"] == "W m^-2"
        assert ds["Q_integrated_SI"].attrs["units"] == "W"
        assert ds["Gamma_es_SI"].attrs["units"] == "1e19 m^-2 s^-1"

    @pytest.mark.parametrize("norm_flux_projection, key",
                             [(True, "Area"), (False, "dVdx")])
    def test_integration_area_follows_the_projection_flag(
            self, tmp_path, norm_flux_projection, key):
        """
        A projected flux is per unit physical area and pairs with the
        sqrt(g^xx)-weighted surface area; an unprojected one is per unit x and
        pairs with dVdx. Mixing them rescales every total.
        """
        g = make_gene3d_run(tmp_path / "run", n_times=2,
                            norm_flux_projection=norm_flux_projection)
        run = Run(g.folder)
        ds = Fluxes2D(run).dataset()
        area = np.asarray(run.geometry[0]["area"][key])
        ratio = (np.asarray(ds["Q_integrated"].isel(species=0, time=0))
                 / np.asarray(ds["Q_total"].isel(species=0, time=0)))
        assert np.allclose(ratio, area)

    def test_time_window_selects_samples(self, noisy_run):
        ds = Fluxes2D(Run(noisy_run.folder)).dataset(t=(5.0, 25.0))
        assert np.allclose(np.asarray(ds["time"]), [10.0, 20.0])

    def test_empty_time_window_is_reported(self, noisy_run):
        """The shared helper names the range that is actually available."""
        with pytest.raises(ValueError, match="no output in the requested time"):
            Fluxes2D(Run(noisy_run.folder)).compute(t=(1e6, 2e6))

    def test_time_average_is_trapezoidal_not_a_plain_mean(self, tmp_path):
        """
        GENE's timestep is adaptive and output happens every istep_mom *steps*,
        so output times are generally unevenly spaced. A plain mean weights every
        sample equally regardless of the interval it stands for; on a realistic
        uneven axis the two differ by tens of percent, and the time-averaged
        flux is the number people quote.
        """
        import h5py
        g = make_gene3d_run(tmp_path / "run", nx0=8, n_times=6, physical=True)
        uneven = np.array([0.0, 1.0, 2.0, 10.0, 30.0, 70.0])
        for name in ["field.dat.h5"] + [f"mom_{s}.dat.h5" for s in g.species]:
            with h5py.File(g.folder / name, "a") as f:
                key = list(f.keys())[0]
                f[f"{key}/time"][...] = uneven

        diag = Fluxes2D(Run(g.folder))
        da = diag.dataset()["Q_total"].sel(species="ions")
        got = np.asarray(diag._t_average(da))
        expected = (np.trapezoid(np.asarray(da), x=uneven, axis=0)
                    / (uneven[-1] - uneven[0]))
        assert np.allclose(got, expected)
        # And it must actually differ from the plain mean, or the test is vacuous.
        assert not np.allclose(got, np.asarray(da).mean(axis=0))

    def test_single_sample_time_average(self, tmp_path):
        """One snapshot has no interval to integrate; return it unchanged."""
        g = make_gene3d_run(tmp_path / "run", nx0=6, n_times=1, physical=True)
        diag = Fluxes2D(Run(g.folder))
        da = diag.dataset()["Q_total"].sel(species="ions")
        assert np.allclose(np.asarray(diag._t_average(da)),
                           np.asarray(da.isel(time=0)))

    def test_totals_have_si_companions(self, noisy_run):
        """
        Without Q_total_SI an SI figure would fall back to the gyro-Bohm array
        under an SI axis label.
        """
        ds = Fluxes2D(Run(noisy_run.folder)).dataset()
        assert ds["Q_total_SI"].attrs["units"] == "W m^-2"
        assert ds["Q_integrated_SI"].attrs["units"] == "W"
        assert ds["Gamma_total_SI"].attrs["units"] == "1e19 m^-2 s^-1"
        assert ds["Gamma_integrated_SI"].attrs["units"] == "1e19 s^-1"

    def test_si_is_gyrobohm_times_the_reference(self, noisy_run):
        run = Run(noisy_run.folder)
        ds = Fluxes2D(run).dataset()
        gb = np.asarray(ds["Q_total"].sel(species="ions").isel(time=0))
        si = np.asarray(ds["Q_total_SI"].sel(species="ions").isel(time=0))
        area = np.asarray(ds["Area"])
        Qgb = float(run.params.get(0)["units"]["Qgb"])
        assert np.allclose(si, gb * Qgb)
        assert np.allclose(
            np.asarray(ds["Q_integrated_SI"].sel(species="ions").isel(time=0)),
            gb * area * Qgb)

    def test_gyrobohm_axes_are_labelled(self, noisy_run):
        """The gyro-Bohm figure needs units on the totals, not just the parts."""
        ds = Fluxes2D(Run(noisy_run.folder)).dataset()
        for key in ("Q_total", "Q_integrated", "Gamma_total"):
            assert ds[key].attrs.get("units")

    def test_plot_draws_gyrobohm_si_and_the_map(self, noisy_run, monkeypatch):
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        figs = Fluxes2D(Run(noisy_run.folder)).plot()
        titles = [f._suptitle.get_text() for f in figs]
        assert titles == ["Radial flux profiles [gyro-Bohm]",
                          "Radial flux profiles [SI]", "Flux evolution"]
        plt.close("all")

    @pytest.mark.parametrize("si, expected", [
        (False, ["Radial flux profiles [gyro-Bohm]"]),
        (True, ["Radial flux profiles [SI]"]),
    ])
    def test_si_override(self, noisy_run, monkeypatch, si, expected):
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        figs = Fluxes2D(Run(noisy_run.folder)).plot(si=si, show_map=False)
        assert [f._suptitle.get_text() for f in figs] == expected
        plt.close("all")

    def test_map_covers_every_flux_and_species(self, noisy_run, monkeypatch):
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        figs = Fluxes2D(Run(noisy_run.folder)).plot(si=False, show_map=True)
        # 2 fluxes x 2 species, each with a colourbar axis.
        assert len(figs[-1].axes) == 8
        plt.close("all")


# ---------------------------------------------------------------------------
# ky spectra
# ---------------------------------------------------------------------------

class TestSpectra3DPath:

    def test_dataset_shape_and_variables(self, physical_run):
        ds = Spectra(Run(physical_run.folder), buffer_frac=0.0).dataset()
        assert dict(ds.sizes) == {"species": 2, "ky": physical_run.ny0}
        for name in ("Gamma_es_ky", "Gamma_em_ky", "Q_es_ky", "Q_em_ky"):
            assert name in ds

    def test_radial_window_is_recorded(self, physical_run):
        run = Run(physical_run.folder)
        x = np.asarray(run.coords[0]["x_o_a"])
        ds = Spectra(run, x_avg_lims=(x[1], x[-2])).dataset()
        lo, hi = ds.attrs["x_avg_range"]
        assert lo == pytest.approx(x[1])
        assert hi == pytest.approx(x[-2])

    def test_default_window_trims_the_buffers(self, tmp_path):
        """Needs enough radial points that a 10% trim is a whole cell."""
        g = make_gene3d_run(tmp_path / "run", nx0=20, n_times=2, physical=True)
        run = Run(g.folder)
        x = np.asarray(run.coords[0]["x_o_a"])
        ds = Spectra(run).dataset()
        lo, hi = ds.attrs["x_avg_range"]
        assert lo == pytest.approx(x[2])
        assert hi == pytest.approx(x[-3])

    def test_coarse_grid_keeps_every_point(self, physical_run):
        """A 10% trim of six points rounds to zero; nothing may be dropped."""
        run = Run(physical_run.folder)
        x = np.asarray(run.coords[0]["x_o_a"])
        ds = Spectra(run).dataset()
        assert ds.attrs["x_avg_range"] == [pytest.approx(x[0]),
                                           pytest.approx(x[-1])]

    def test_mismatched_field_and_moment_times_are_paired_by_value(
            self, tmp_path):
        """
        istep_field and istep_mom need not agree. Pairing snapshots by position
        instead of by time would silently combine a field from one time with a
        moment from another.
        """
        import h5py
        g = make_gene3d_run(tmp_path / "run", n_times=4, physical=True)
        # Drop the second moment snapshot for both species, leaving the field
        # file with one extra sample.
        for spec in g.species:
            with h5py.File(g.folder / f"mom_{spec}.dat.h5", "a") as f:
                grp = f[f"mom_{spec}"]
                for label in list(grp.keys()):
                    if label != "time":
                        del grp[f"{label}/0000000001"]
        diag = Spectra(Run(g.folder), buffer_frac=0.0)
        raw = diag.compute()
        assert np.allclose(raw["times"], [0.0, 20.0, 30.0])
        for ratio in diag.consistency.values():
            assert ratio == pytest.approx(1.0, abs=1e-5)

    def test_plot_runs_headless(self, physical_run, monkeypatch):
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        Spectra(Run(physical_run.folder), buffer_frac=0.0).plot()
        plt.close("all")
