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

    def test_matches_the_code_when_temp_and_dens_are_not_one(self, tmp_path):
        """
        diag_3d.F90 applies dens to Gamma and dens*temp to Q before writing
        profile_<species>. Fluxes2D must apply the same factors to the moment
        values, or the two disagree by them. A fixture with every temp = dens = 1
        cannot see this.
        """
        from genetools.diagnostics import ProfileDiag
        g = make_gene3d_run(tmp_path / "run", nx0=8, n_times=3, physical=True,
                            write_profile_diag=True,
                            temps=[2.0, 0.5], denss=[3.0, 1.5])
        run = Run(g.folder)
        mine = Fluxes2D(run).dataset()
        code = ProfileDiag(run).dataset()
        for sp in run.species:
            for mine_v, code_v in (("Gamma_total", "Gamma"), ("Q_total", "Q")):
                a = np.asarray(mine[mine_v].sel(species=sp).isel(time=0))
                b = np.asarray(code[code_v].sel(species=sp).isel(time=0))
                assert np.allclose(a, b, rtol=1e-5), f"{sp} {code_v}"

    @staticmethod
    def _truncate_one_species(folder, species, index):
        """Drop one snapshot from a single species' moment file."""
        import h5py
        with h5py.File(folder / f"mom_{species}.dat.h5", "a") as f:
            grp = f[f"mom_{species}"]
            for label in list(grp.keys()):
                if label != "time":
                    del grp[f"{label}/{index:010d}"]

    def test_species_with_unequal_snapshot_counts(self, tmp_path):
        """
        GENE-3D writes mom_<species> one species at a time, so a run that is
        still going -- or was killed mid-write -- leaves one file a snapshot
        short. Streaming each species over its own indices then produces arrays
        of different lengths, surfacing much later as
        'all input arrays must have the same shape'.
        """
        g = make_gene3d_run(tmp_path / "run", nx0=8, n_times=5, physical=True)
        self._truncate_one_species(g.folder, g.species[-1], 4)
        run = Run(g.folder)
        assert (run.mom(g.species[0]).read_all_times().size
                != run.mom(g.species[-1]).read_all_times().size)

        ds = Fluxes2D(run).dataset()
        assert ds.sizes["time"] == 4                     # the common times
        assert np.allclose(np.asarray(ds["time"]), [0.0, 10.0, 20.0, 30.0])

    def test_profiles_also_handles_unequal_counts(self, tmp_path):
        from genetools.diagnostics import Profiles
        g = make_gene3d_run(tmp_path / "run", nx0=8, n_times=5, physical=True)
        self._truncate_one_species(g.folder, g.species[-1], 4)
        ds = Profiles(Run(g.folder)).dataset()
        assert ds.sizes["time"] == 4

    def test_no_common_time_is_reported_clearly(self, tmp_path):
        """Disjoint species files must say so, not stack mismatched arrays."""
        import h5py
        g = make_gene3d_run(tmp_path / "run", nx0=8, n_times=4, physical=True)
        with h5py.File(g.folder / f"mom_{g.species[-1]}.dat.h5", "a") as f:
            key = f"mom_{g.species[-1]}"
            f[f"{key}/time"][...] = np.asarray(f[f"{key}/time"][...]) + 1e5
        with pytest.raises(ValueError, match="no output time is common"):
            Fluxes2D(Run(g.folder)).dataset()

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
        # startswith, so appending a legend hint to a title is not a failure.
        titles = [f._suptitle.get_text() for f in figs]
        assert len(titles) == 3
        assert titles[0].startswith("Radial flux profiles [gyro-Bohm]")
        assert titles[1].startswith("Radial flux profiles [SI]")
        assert titles[2].startswith("Flux evolution")
        plt.close("all")

    @pytest.mark.parametrize("si, expected", [
        (False, "Radial flux profiles [gyro-Bohm]"),
        (True, "Radial flux profiles [SI]"),
    ])
    def test_si_override(self, noisy_run, monkeypatch, si, expected):
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        figs = Fluxes2D(Run(noisy_run.folder)).plot(si=si, show_map=False)
        assert len(figs) == 1
        assert figs[0]._suptitle.get_text().startswith(expected)
        plt.close("all")

    def test_profiles_show_total_and_both_components(self, noisy_run, monkeypatch):
        """
        Colour is the species, line style the component, so both read off one
        legend: solid total, dashed ES, dotted EM.
        """
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        figs = Fluxes2D(Run(noisy_run.folder)).plot(si=False, show_map=False)
        ax = next(a for a in figs[0].axes
                  if a.get_title().startswith(r"$\langle Q"))
        styles = {l.get_label(): (l.get_linestyle(), l.get_color())
                  for l in ax.get_lines() if not l.get_label().startswith("_")}
        for sp in noisy_run.species:
            total = next(k for k in styles if k.startswith(sp) and "mean" in k)
            assert styles[total][0] == "-"
            assert styles[f"{sp} ES"][0] == "--"
            assert styles[f"{sp} EM"][0] == ":"
            # One colour per species across all three components.
            assert (styles[total][1] == styles[f"{sp} ES"][1]
                    == styles[f"{sp} EM"][1])
        plt.close("all")

    def test_components_can_be_switched_off(self, noisy_run, monkeypatch):
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        figs = Fluxes2D(Run(noisy_run.folder)).plot(
            si=False, show_map=False, components=False)
        ax = next(a for a in figs[0].axes
                  if a.get_title().startswith(r"$\langle Q"))
        labels = [l.get_label() for l in ax.get_lines()
                  if not l.get_label().startswith("_")]
        assert all("ES" not in n and "EM" not in n for n in labels)
        assert len(labels) == len(noisy_run.species)
        plt.close("all")

    def test_components_sum_to_the_total(self, noisy_run):
        ds = Fluxes2D(Run(noisy_run.folder)).dataset()
        for base in ("Q", "Gamma"):
            es = np.asarray(ds[f"{base}_es"].isel(species=0, time=0))
            em = np.asarray(ds[f"{base}_em"].isel(species=0, time=0))
            tot = np.asarray(ds[f"{base}_total"].isel(species=0, time=0))
            assert np.allclose(es + em, tot)

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
        assert dict(ds.sizes) == {"species": 2, "ky": physical_run.ny0,
                                  "x": physical_run.nx0}
        for base in ("Gamma_es", "Gamma_em", "Q_es", "Q_em"):
            assert ds[base + "_xky"].dims == ("species", "x", "ky")
            assert ds[base + "_x"].dims == ("species", "x")
            assert ds[base + "_ky"].dims == ("species", "ky")

    def test_reductions_come_from_the_map(self, physical_run):
        """
        The map is the primary product; both 1-D views must be reductions of it,
        not separately accumulated quantities that could drift from it.
        """
        run = Run(physical_run.folder)
        ds = Spectra(run, buffer_frac=0.0).dataset()
        J = run.geometry[0]["Jacobian"]
        w = c.radial_weights(J)[:, 0]                 # y-independent
        for base in ("Gamma_es", "Q_es"):
            for sp in run.species:
                m = np.asarray(ds[base + "_xky"].sel(species=sp))
                assert np.allclose(np.asarray(ds[base + "_x"].sel(species=sp)),
                                   m.sum(axis=1))
                expect = (m * w[:, None]).sum(axis=0) / w.sum()
                assert np.allclose(
                    np.asarray(ds[base + "_ky"].sel(species=sp)), expect)

    def test_ky_spectrum_matches_the_joint_xz_average(self, physical_run):
        """
        Keeping x and reducing afterwards must give exactly what the old
        one-step x-z average gave, or the consistency check below is comparing
        against a differently weighted reference.
        """
        run = Run(physical_run.folder)
        diag = Spectra(run, buffer_frac=0.0)
        raw = diag.compute()
        J = run.geometry[0]["Jacobian"]
        for sp, per in raw["spectra"].items():
            for v, m in per.items():
                reduced = diag._reduce_x(m, raw["x_weights"], raw["xslice"])
                # Rebuilding the joint average from the map has to agree with
                # the helper that does it in one step.
                direct = (np.asarray(m) * c.radial_weights(J)).sum(axis=0) \
                    / c.radial_weights(J).sum(axis=0)
                assert np.allclose(reduced, direct), f"{sp} {v}"

    def test_map_is_not_radially_constant(self, physical_run):
        """Otherwise keeping the x axis would be pointless and untested."""
        ds = Spectra(Run(physical_run.folder), buffer_frac=0.0).dataset()
        m = np.asarray(ds["Q_es_xky"].isel(species=0))
        assert m.std(axis=0).max() > 0

    def test_radial_window_is_recorded(self, physical_run):
        run = Run(physical_run.folder)
        x = np.asarray(run.coords[0]["x_o_a"])
        ds = Spectra(run, x_avg_lims=(x[1], x[-2])).dataset()
        lo, hi = ds.attrs["x_avg_range"]
        assert lo == pytest.approx(x[1])
        assert hi == pytest.approx(x[-2])

    def test_default_keeps_every_radial_point(self, tmp_path):
        """
        Nothing is trimmed or averaged away unless asked: the (x, ky) map is the
        primary output, so the default window is the whole domain.
        """
        g = make_gene3d_run(tmp_path / "run", nx0=20, n_times=2, physical=True)
        run = Run(g.folder)
        x = np.asarray(run.coords[0]["x_o_a"])
        ds = Spectra(run).dataset()
        assert ds.attrs["x_avg_range"] == [pytest.approx(x[0]),
                                           pytest.approx(x[-1])]
        assert ds.sizes["x"] == 20

    def test_buffer_frac_still_trims_when_asked(self, tmp_path):
        """Needs enough radial points that a 10% trim is a whole cell."""
        g = make_gene3d_run(tmp_path / "run", nx0=20, n_times=2, physical=True)
        run = Run(g.folder)
        x = np.asarray(run.coords[0]["x_o_a"])
        ds = Spectra(run, buffer_frac=0.1).dataset()
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

    def test_default_plot_is_the_two_dimensional_map(self):
        """
        `plot()` leads with the (x, ky) maps rather than an x-averaged spectrum.
        """
        from genetools.diagnostics.spectra import Spectra as S
        assert S._views(None) == ("map",)
        assert S._views("all") == ("map", "ky", "profile")
        assert S._views("profile") == ("profile",)
        assert S._views(("map", "ky")) == ("map", "ky")

    def test_unknown_view_is_rejected(self):
        from genetools.diagnostics.spectra import Spectra as S
        with pytest.raises(ValueError, match="unknown spectra view"):
            S._views("nonsense")

    def test_plot_draws_only_the_requested_views(self, physical_run,
                                                 monkeypatch):
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        run = Run(physical_run.folder)
        diag = Spectra(run)
        assert len(diag.plot()) == 1                        # the map
        assert len(diag.plot(which="all")) == 3
        assert len(diag.plot(which=("ky", "profile"))) == 2

    def test_default_plot_panels_are_colour_maps(self, physical_run,
                                                 monkeypatch):
        """A 1-D panel would satisfy a figure count; check it is really 2-D."""
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        run = Run(physical_run.folder)
        fig = Spectra(run).plot()[0]
        panels = [a for a in fig.axes if a.get_title()]
        assert panels
        for ax in panels:
            assert ax.collections, f"{ax.get_title()} is not a colour map"

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


# ---------------------------------------------------------------------------
# Volume average vs nrg
# ---------------------------------------------------------------------------

class TestVolumeAverage:
    """
    The x-reduction that turns a flux-surface profile into the single number
    `nrg` reports. Only the maths is tested here: the fixture has no nrg derived
    from the fields, so a comparison against it would be circular.
    """

    def test_composition_is_exact(self):
        """
        Reducing (y, z) then x with `sum_{y,z} J` weights equals the full 3-D
        Jacobian-weighted mean, so the two-step route is exact.
        """
        rng = np.random.default_rng(0)
        J = 0.5 + rng.uniform(size=(9, 6, 5))
        f = rng.normal(size=(9, 6, 5))
        composed = c.volume_average(c.flux_surface_average(f, J), J)
        assert composed == pytest.approx(np.average(f, weights=J), rel=1e-12)

    def test_plain_x_mean_differs(self):
        """`.mean("x")` drops the weight; on a varying Jacobian that matters."""
        rng = np.random.default_rng(1)
        J = 0.5 + rng.uniform(size=(40, 6, 5)) ** 3
        f = rng.normal(size=(40, 6, 5))
        profile = c.flux_surface_average(f, J)
        assert profile.mean() != pytest.approx(
            c.volume_average(profile, J), rel=1e-9)

    def test_dataset_carries_volume_averages(self, physical_run):
        ds = Fluxes2D(Run(physical_run.folder)).dataset()
        for base in ("Gamma_es", "Gamma_em", "Q_es", "Q_em",
                     "Gamma_total", "Q_total"):
            key = base + "_volume"
            assert key in ds, key
            assert ds[key].dims == ("species", "time")

    def test_volume_average_matches_a_direct_3d_mean(self, physical_run):
        """
        Against a 3-D Jacobian mean of the array on disk, including whatever
        prefactor `_prefactor_3d` applies — so this pins the reduction, not the
        normalisation.
        """
        run = Run(physical_run.folder)
        ds = Fluxes2D(run).dataset()
        J = run.geometry[0]["Jacobian"]
        diag = Fluxes2D(run)
        for name in run.species:
            reader = run.mom(name)
            arrays = next(iter(reader.stream_selected([0])))[1]
            raw = np.average(arrays[reader.index_of("Q_es")], weights=J)
            direct = raw * diag._prefactor_3d("Q_es", name)
            mine = float(ds["Q_es_volume"].sel(species=name).isel(time=0))
            assert mine == pytest.approx(direct, rel=1e-10)

    def test_prefactor_false_is_the_raw_mom_file_reduction(self, physical_run):
        """
        With the species factor off, the result must be exactly a 3-D
        Jacobian-weighted mean of the array on disk — nothing else applied.
        """
        run = Run(physical_run.folder)
        J = run.geometry[0]["Jacobian"]
        ds = Fluxes2D(run).volume_average(prefactor=False)
        for name in run.species:
            reader = run.mom(name)
            arrays = next(iter(reader.stream_selected([0])))[1]
            direct = np.average(arrays[reader.index_of("Q_es")], weights=J)
            mine = float(ds["Q_es"].sel(species=name).isel(time=0))
            assert mine == pytest.approx(direct, rel=1e-10)

    def test_prefactor_is_the_only_difference(self, tmp_path):
        """Species factors non-unity, so the ratio is a real check."""
        g = make_gene3d_run(tmp_path / "run", nx0=8, nz0=8, n_times=2,
                            physical=True, temps=[2.0, 0.5], denss=[3.0, 1.5])
        run = Run(g.folder)
        diag = Fluxes2D(run)
        on = diag.volume_average(prefactor=True)
        off = diag.volume_average(prefactor=False)
        for name in run.species:
            spec = next(s for s in run.params.get(0)["species"]
                        if s["name"] == name)
            got = (float(on["Q_es"].sel(species=name).isel(time=0))
                   / float(off["Q_es"].sel(species=name).isel(time=0)))
            assert got == pytest.approx(spec["dens"] * spec["temp"], rel=1e-10)

    def test_totals_are_es_plus_em(self, physical_run):
        ds = Fluxes2D(Run(physical_run.folder)).volume_average()
        np.testing.assert_allclose(
            ds["Q_total"].values, ds["Q_es"].values + ds["Q_em"].values,
            rtol=1e-12)

    def test_refuses_for_a_flux_tube(self, tmp_path):
        from tests.gene_fixture import make_fluxtube_run
        run = Run(make_fluxtube_run(tmp_path / "ft"))
        with pytest.raises(NotImplementedError):
            Fluxes2D(run).volume_average()


class TestSpeciesPrefactor:
    """
    The prefactor must match `nrg` for *any* dens/temp, not just when they
    happen to be 1.

    `diag_3d.F90` applies it after `sum_3d_real` and symmetrically across ES and
    EM — `var(5:6) *= dens`, `var(7:8) *= dens*temp` (lines 698-701) — while the
    mom file is written earlier (line 547) with no species factor at all. So the
    diagnostic has to supply exactly that factor, to both parts of each flux.

    The reference GUI (`diag_fluxesgene3d.py:88-97`) scales only the ES terms,
    which is what this pins against.
    """

    #: Deliberately none of these is 1, and dens != temp, so a swapped or
    #: dropped factor cannot pass.
    TEMPS = [2.0, 0.5]
    DENSS = [3.0, 7.0]

    @pytest.fixture
    def scaled_run(self, tmp_path):
        g = make_gene3d_run(tmp_path / "run", nx0=8, nz0=8, n_times=2,
                            physical=True, temps=self.TEMPS,
                            denss=self.DENSS)
        return g, Run(g.folder)

    def test_prefactor_values_come_from_the_namelist(self, scaled_run):
        _, run = scaled_run
        diag = Fluxes2D(run)
        for name in run.species:
            spec = next(s for s in run.params.get(0)["species"]
                        if s["name"] == name)
            dens, temp = spec["dens"], spec["temp"]
            assert diag._prefactor_3d("dens", name) == pytest.approx(dens)
            assert diag._prefactor_3d("dens_temp", name) == pytest.approx(
                dens * temp)

    @pytest.mark.parametrize("es, em, kind", [
        ("Gamma_es", "Gamma_em", "dens"),
        ("Q_es", "Q_em", "dens_temp"),
    ])
    def test_em_is_scaled_exactly_like_es(self, scaled_run, es, em, kind):
        """
        The one place genetools and the reference GUI disagree. If the EM factor
        were dropped, this ratio would come out as dens (or dens*temp).
        """
        _, run = scaled_run
        diag = Fluxes2D(run)
        assert (diag._FLUXES_3D[es][0] == diag._FLUXES_3D[em][0] == kind)
        ds_on = diag.volume_average(prefactor=True)
        ds_off = diag.volume_average(prefactor=False)
        for name in run.species:
            expect = diag._prefactor_3d(kind, name)
            for flux in (es, em):
                got = (float(ds_on[flux].sel(species=name).isel(time=0))
                       / float(ds_off[flux].sel(species=name).isel(time=0)))
                assert got == pytest.approx(expect, rel=1e-10), f"{name} {flux}"

    def test_a_dropped_em_factor_would_be_detected(self, scaled_run):
        """
        Guards the guard: with these dens/temp the ES and EM factors are far
        from 1, so omitting one is a large error rather than a rounding one.
        """
        _, run = scaled_run
        diag = Fluxes2D(run)
        for name in run.species:
            assert abs(diag._prefactor_3d("dens", name) - 1.0) > 0.5
            assert abs(diag._prefactor_3d("dens_temp", name) - 1.0) > 0.5
