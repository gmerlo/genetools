"""
Tests for the remaining GENE-3D diagnostics and the Run facade dispatch.

The flux reconstruction is covered separately in ``test_gene3d_fluxes.py``; what
is checked here is that each diagnostic reduces the data the way it claims to,
that the facade routes to the GENE-3D implementation, and that the diagnostics
with no GENE-3D meaning refuse rather than silently doing something else.
"""

import numpy as np
import pytest

import genetools.diagnostics as g3
from genetools.diagnostics import _gene3d as c
from genetools.diagnostics._base import CachingDiagnostic, RunDiagnostic
from genetools.run import Run
from tests.gene3d_fixture import make_gene3d_run


@pytest.fixture
def run3d(tmp_path):
    g = make_gene3d_run(tmp_path / "run", nx0=16, n_times=6, physical=True,
                        write_profile_diag=True, write_srcmom=True,
                        write_vsp=True)
    return g, Run(g.folder)


@pytest.fixture
def headless(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield plt
    plt.close("all")


# ---------------------------------------------------------------------------
# Shared reductions
# ---------------------------------------------------------------------------

class TestCommon:

    def test_flux_surface_average_is_jacobian_weighted(self):
        rng = np.random.default_rng(0)
        var = rng.standard_normal((4, 5, 6))
        J = np.abs(rng.standard_normal((4, 5, 6))) + 0.5
        got = c.flux_surface_average(var, J)
        expected = (var * J).sum(axis=(1, 2)) / J.sum(axis=(1, 2))
        assert np.allclose(got, expected)

    def test_ky_transform_is_normalised_by_the_grid_size(self):
        """A uniform field must give its own value in the ky=0 mode."""
        var = np.full((3, 8, 2), 2.5)
        spectrum = c.to_ky(var)
        assert np.allclose(spectrum[:, 0, :], 2.5)
        assert np.allclose(spectrum[:, 1:, :], 0.0)

    def test_jacobian_yz_removes_the_y_dependence(self):
        rng = np.random.default_rng(1)
        J = np.abs(rng.standard_normal((3, 4, 5))) + 1.0
        out = c.jacobian_yz(J)
        assert out.shape == J.shape
        assert np.allclose(out, out[:, :1, :])           # constant in y
        assert np.allclose(out[:, 0, :], J.mean(axis=1))

    def test_time_average_is_trapezoidal(self):
        times = np.array([0.0, 1.0, 3.0])
        stack = np.array([[0.0], [2.0], [4.0]])
        # trapz = 1*(0+2)/2 + 2*(2+4)/2 = 1 + 6 = 7, over a span of 3.
        assert CachingDiagnostic._time_average(stack, times)[0] == pytest.approx(7.0 / 3.0)

    def test_single_sample_returns_itself(self):
        assert CachingDiagnostic._time_average(np.array([[5.0]]), [2.0])[0] == 5.0

    def test_t_average_is_trapezoidal_on_a_dataarray(self):
        """
        Every diagnostic that time-averages a DataArray must go through this,
        not `.mean("time")`. GENE's dt is adaptive, so output times are unevenly
        spaced and a plain mean is biased by tens of percent.
        """
        import xarray as xr
        t = np.array([0.0, 1.0, 2.0, 10.0, 30.0, 70.0])
        y = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 9.0])
        da = xr.DataArray(y, dims="time", coords={"time": t})
        expected = np.trapezoid(y, x=t) / (t[-1] - t[0])
        assert float(RunDiagnostic._t_average(da)) == pytest.approx(expected)
        # And it must differ from the plain mean, or the test is vacuous.
        assert abs(float(RunDiagnostic._t_average(da)) - y.mean()) > 0.5

    def test_t_average_degenerate_windows(self):
        import xarray as xr
        one = xr.DataArray([7.0], dims="time", coords={"time": [3.0]})
        assert float(RunDiagnostic._t_average(one)) == 7.0
        flat = xr.DataArray([2.0, 6.0], dims="time", coords={"time": [5.0, 5.0]})
        assert float(RunDiagnostic._t_average(flat)) == 2.0

    def test_no_diagnostic_uses_a_plain_time_mean(self):
        """
        Guard against the bias creeping back in: `.mean("time")` on a time axis
        that GENE wrote unevenly is wrong wherever a physical average is meant.
        """
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[2] / "diagnostics"
        offenders = []
        for path in sorted(root.glob("*.py")):
            for n, line in enumerate(path.read_text().splitlines(), 1):
                if 'mean("time")' in line and "Not ``" not in line:
                    offenders.append(f"{path.name}:{n}")
        assert not offenders, f"plain time means found: {offenders}"

    def test_every_diagnostic_calls_the_base_constructor(self):
        """
        ``RunDiagnostic.__init__`` is what enforces ``supported``. A subclass
        that assigns ``self.run`` itself instead of calling ``super().__init__``
        silently loses the geometry guard, so a GENE-3D-only diagnostic runs on
        a flux tube and dies somewhere deep instead of refusing.
        """
        import ast, pathlib
        root = pathlib.Path(__file__).resolve().parents[2] / "diagnostics"
        offenders = []
        for path in sorted(root.glob("*.py")):
            tree = ast.parse(path.read_text())
            for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                bases = {b.id for b in cls.bases if isinstance(b, ast.Name)}
                if "RunDiagnostic" not in bases:
                    continue
                init = next((f for f in cls.body
                             if isinstance(f, ast.FunctionDef)
                             and f.name == "__init__"), None)
                if init is None:          # inherited, so the guard runs
                    continue
                calls_super = any(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "__init__"
                    for n in ast.walk(init))
                if not calls_super:
                    offenders.append(f"{path.name}:{cls.name}")
        assert not offenders, (
            f"__init__ bypasses the base constructor: {offenders}")

    def test_zero_length_window_returns_the_first_sample(self):
        """
        A zero-duration window has no interval to integrate over. The shared
        helper returns the first sample rather than averaging; this only arises
        for duplicate timestamps, which the readers already deduplicate.
        """
        stack = np.array([[1.0], [3.0]])
        assert CachingDiagnostic._time_average(stack, [7.0, 7.0])[0] == 1.0

    @pytest.mark.parametrize("limits, expected", [
        ((0.3, 0.5), (2, 5)),
        ((0.5, 0.3), (2, 5)),          # reversed bounds are accepted
        ((0.4, 0.4), (3, 4)),          # equal bounds -> the single nearest point
    ])
    def test_radial_slice_from_limits(self, limits, expected):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        sl = c.radial_slice(x, limits=limits)
        assert (sl.start, sl.stop) == expected

    def test_radial_slice_buffer_never_empties_the_grid(self):
        x = np.linspace(0.1, 0.9, 4)
        sl = c.radial_slice(x, buffer_frac=0.9)
        assert x[sl].size > 0


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class TestProfiles3D:

    def test_total_profile_is_background_plus_scaled_perturbation(self, run3d):
        """
        GENE-3D adds the perturbation scaled by rhostar*minor_r. Omitting that
        factor overstates the perturbation by 1/rhostar — two orders of
        magnitude for a typical run.
        """
        g, run = run3d
        ds = g3.Profiles(run).dataset()
        params = run.params.get(0)
        scale = (float(params["geometry"]["rhostar"])
                 * float(params["geometry"]["minor_r"]))
        J = run.geometry[0]["Jacobian"]
        spec = next(s for s in params["species"] if s["name"] == "ions")

        T0 = (np.asarray(run.eq_profiles["ions"]["T"])
              / (spec["temp"] * params["units"]["Tref"]))
        tpar = g.moments["ions"]["T_par"][0].astype(np.float32)
        tper = g.moments["ions"]["T_per"][0].astype(np.float32)
        pert = c.flux_surface_average(tpar / 3.0 + 2.0 * tper / 3.0, J)
        expected = T0 + scale * pert

        got = np.asarray(ds["T"].sel(species="ions").isel(time=0))
        assert np.allclose(got, expected, rtol=1e-5)

    def test_gradients_are_logarithmic_and_scaled_by_minor_radius(self, run3d):
        g, run = run3d
        ds = g3.Profiles(run).dataset()
        minor_r = float(run.params.get(0)["geometry"]["minor_r"])
        x = np.asarray(ds["x"])
        T = np.asarray(ds["T"].sel(species="ions").isel(time=0))
        expected = -np.gradient(np.log(T), x) / minor_r
        got = np.asarray(ds["omt"].sel(species="ions").isel(time=0))
        assert np.allclose(got, expected)

    def test_si_companion_is_present(self, run3d):
        _, run = run3d
        ds = g3.Profiles(run).dataset()
        assert ds["T_SI"].attrs["units"] == "keV"

    def test_matches_the_code_when_temp_and_dens_are_not_one(self, tmp_path):
        """
        profiles_<species> stores Tref*temp*temp_prof and GENE-3D's own
        profile_<species> writes spec%temp*(temp_prof + ...), so the dataset must
        be in T_ref units. Dividing by `temp` as well normalises to the species
        temperature instead — a factor `temp` away from the code's own output,
        with the T_ref label and the SI conversion wrong by the same factor.
        A fixture with every temp = 1 cannot see this.
        """
        g = make_gene3d_run(tmp_path / "run", nx0=8, n_times=3, physical=True,
                            write_profile_diag=True,
                            temps=[2.0, 0.5], denss=[3.0, 1.5])
        run = Run(g.folder)
        mine = g3.Profiles(run).dataset()
        code = g3.ProfileDiag(run).dataset()
        for sp in run.species:
            for v in ("T", "n", "omt", "omn"):
                a = np.asarray(mine[v].sel(species=sp).isel(time=0))
                b = np.asarray(code[v].sel(species=sp).isel(time=0))
                assert np.allclose(a, b, rtol=1e-6), f"{sp} {v}"

    def test_gradient_units_name_the_reference_length(self, run3d):
        """The quantity is -L_ref dln(T)/dx, not R/L_T or a/L_T."""
        _, run = run3d
        ds = g3.Profiles(run).dataset()
        assert ds["omt"].attrs["units"] == "L_ref/L_T"
        assert ds["omn"].attrs["units"] == "L_ref/L_n"
        assert ds["T"].attrs["units"] == "T_ref"
        assert ds["T_SI"].attrs["units"] == "keV"

    def test_maps_adds_the_evolution_figure(self, run3d, headless):
        """
        `maps` was implemented for the regular-GENE path and silently dropped
        here: `plot` never forwarded it to `_plot_3d`, so it did nothing.
        """
        _, run = run3d
        n_species = len(run.species)
        plain = g3.Profiles(run).plot()
        assert len(plain) == n_species
        with_maps = g3.Profiles(run).plot(maps=True)
        assert len(with_maps) == 2 * n_species
        titles = [f._suptitle.get_text() for f in with_maps]
        assert sum("evolution" in t for t in titles) == n_species

    def test_map_panels_are_two_dimensional(self, run3d, headless):
        """A line plot under an 'evolution' title would pass a count check."""
        _, run = run3d
        figs = g3.Profiles(run).plot(maps=True)
        evo = next(f for f in figs
                   if "evolution" in f._suptitle.get_text())
        panels = [a for a in evo.axes if a.get_title()]
        assert panels, "no titled panels on the evolution figure"
        for ax in panels:
            assert ax.collections, f"{ax.get_title()} is not a colour map"

    def test_maps_honours_si(self, run3d, headless):
        _, run = run3d
        figs = g3.Profiles(run).plot(si=True, maps=True)
        assert all("[SI]" in f._suptitle.get_text() for f in figs)

    def test_compare_with_code_reports_per_variable(self, run3d):
        _, run = run3d
        report = g3.Profiles(run).compare_with_code()
        assert set(report) == set(run.species)
        assert "T" in report["ions"]


# ---------------------------------------------------------------------------
# Shearing / zonal
# ---------------------------------------------------------------------------

class TestShearing3D:

    @staticmethod
    def _write_zonal_ramp(folder, nx0):
        """
        Replace phi with a y- and z-independent radial ramp.

        The stock fixture field is built from cos(m*y) modes only, so its
        flux-surface average is round-off (~1e-9) and every sign assertion on it
        passes vacuously — `allclose(v, -v)` is True at that magnitude. A ramp
        rising in x makes dphi/dx unambiguously positive, which pins the sign.
        """
        import h5py
        with h5py.File(folder / "field.dat.h5", "a") as f:
            times = np.asarray(f["field/time"][...])
            shape = f["field/phi/0000000000"].shape        # (nz, ny, nx)
            ramp = np.linspace(0.0, 1.0, nx0)
            for it in range(len(times)):
                f[f"field/phi/{it:010d}"][...] = np.broadcast_to(ramp, shape)

    def test_velocity_and_shear_are_successive_radial_derivatives(self, tmp_path):
        # On the stock fixture this passes whatever the signs are: its field has
        # no ky=0 part, so every array here is round-off.
        g = make_gene3d_run(tmp_path / "run", nx0=12, nz0=8, n_times=2)
        self._write_zonal_ramp(g.folder, 12)
        run = Run(g.folder)
        ds = g3.ShearingRate(run).dataset()
        x = np.asarray(ds["x_o_rho_ref"])
        C_xy = np.asarray(run.geometry[0]["metric"]["C_xy"])
        phi_fs = np.asarray(ds["phi_zonal"].isel(time=0))
        e_r = -np.gradient(phi_fs, x)
        v = -e_r / C_xy
        assert np.allclose(np.asarray(ds["e_r"].isel(time=0)), e_r)
        assert np.allclose(np.asarray(ds["v_exb"].isel(time=0)), v)
        assert np.allclose(np.asarray(ds["omega_exb"].isel(time=0)),
                           np.gradient(v, x))

    def test_v_exb_sign_follows_gene(self, tmp_path):
        """
        `v_ExB = -E_r/C_xy` with `E_r = -dphi/dx`, so a potential rising in x
        gives a *positive* v_ExB. GENE fixes this: `profiles.F90` defines
        `ExBrate = -dEraddx_prof(0)`, i.e. omega_ExB = -dE_r/dx = d(v_ExB)/dx.
        The GENE-3D path used to return the opposite sign.
        """
        g = make_gene3d_run(tmp_path / "run", nx0=12, nz0=8, n_times=2)
        self._write_zonal_ramp(g.folder, 12)
        ds = g3.ShearingRate(Run(g.folder)).dataset()
        e_r = np.asarray(ds["e_r"].isel(time=0))
        v = np.asarray(ds["v_exb"].isel(time=0))
        # Interior points only: np.gradient's one-sided ends are noisier.
        assert np.all(e_r[1:-1] < 0), "E_r = -dphi/dx must be negative on a ramp"
        assert np.all(v[1:-1] > 0), "v_ExB = -E_r/C_xy must be positive"
        # And the assertion is not vacuous at this amplitude.
        assert not np.allclose(v, -v)

    def test_gam_v_exb_agrees_with_shearing(self, tmp_path):
        """Both call it `v_exb`; they must mean the same thing."""
        g = make_gene3d_run(tmp_path / "run", nx0=12, nz0=8, n_times=2)
        self._write_zonal_ramp(g.folder, 12)
        run = Run(g.folder)
        sr = np.asarray(g3.ShearingRate(run).dataset()["v_exb"].isel(time=0))
        gam = np.asarray(g3.Gam(run).dataset()["v_exb"].isel(time=0))
        assert np.allclose(sr, gam), "Gam and ShearingRate disagree on v_exb"

    def test_zonal_potential_is_the_flux_surface_average(self, run3d):
        g, run = run3d
        ds = g3.ShearingRate(run).dataset()
        J = run.geometry[0]["Jacobian"]
        expected = c.flux_surface_average(g.fields["phi"][0].astype(np.float32), J)
        assert np.allclose(np.asarray(ds["phi_zonal"].isel(time=0)), expected,
                           rtol=1e-5)

    def test_rms_summaries_have_the_right_axes(self, run3d):
        _, run = run3d
        ds = g3.ShearingRate(run).dataset()
        assert ds["omega_exb_rms_x"].dims == ("x",)
        assert ds["omega_exb_rms_t"].dims == ("time",)

    def test_every_geometry_returns_the_same_variable_names(self, run3d):
        """
        The two paths used to disagree (`phi_zonal_x`/`omega_ExB` on spectral
        runs, `phi_zonal`/`omega_exb` here), so `run.shearing.data` meant
        different things depending on the run.
        """
        _, run = run3d
        ds = g3.ShearingRate(run).dataset()
        for name in ("phi_zonal", "e_r", "v_exb", "omega_exb"):
            assert name in ds, name

    def test_e_r_is_v_exb_times_C_xy(self, tmp_path):
        g = make_gene3d_run(tmp_path / "run", nx0=12, nz0=8, n_times=2)
        self._write_zonal_ramp(g.folder, 12)
        run = Run(g.folder)
        ds = g3.ShearingRate(run).dataset()
        C_xy = np.asarray(run.geometry[0]["metric"]["C_xy"])
        assert np.allclose(-np.asarray(ds["e_r"].isel(time=0)) / C_xy,
                           np.asarray(ds["v_exb"].isel(time=0)))

    def test_zonal_view_is_one_figure(self, run3d, headless):
        """The view the separate `Zonal` diagnostic used to give."""
        _, run = run3d
        figs = g3.ShearingRate(run).plot(which="zonal")
        assert len(figs) == 1
        titles = [a.get_title() for a in figs[0].axes if a.get_title()]
        assert any("zonal" in t for t in titles)

    def test_plot_groups_are_validated(self, run3d):
        _, run = run3d
        with pytest.raises(ValueError, match="unknown plot group"):
            g3.ShearingRate(run).plot(which="nonsense")

    def test_default_plot_draws_the_three_groups(self, run3d, headless):
        _, run = run3d
        assert len(g3.ShearingRate(run).plot()) == 3


# ---------------------------------------------------------------------------
# Amplitude spectra
# ---------------------------------------------------------------------------

class TestAmplitude3D:

    def test_both_directions_are_transformed(self, run3d):
        _, run = run3d
        ds = g3.AmplitudeSpectra(run).dataset()
        assert "phi_kx" in ds and "phi_ky" in ds
        assert ds["phi_kx"].dims == ("kx",)
        assert ds["phi_ky"].dims == ("ky",)

    def test_moment_spectra_carry_a_species_dimension(self, run3d):
        _, run = run3d
        ds = g3.AmplitudeSpectra(run).dataset()
        assert ds["n_kx"].dims == ("species", "kx")

    def test_moment_selection_is_respected(self, run3d):
        _, run = run3d
        ds = g3.AmplitudeSpectra(run, moments=("n",)).dataset()
        assert "n_kx" in ds
        assert "T_par_kx" not in ds

    def test_all_moments_can_be_requested(self, run3d):
        _, run = run3d
        ds = g3.AmplitudeSpectra(run, moments="all").dataset()
        assert "Q_es_kx" in ds


# ---------------------------------------------------------------------------
# Contours
# ---------------------------------------------------------------------------

class TestContourReductions3D:
    """
    The reduction engine, which used to be a separate ``Slices`` diagnostic.

    ``Contours`` draws the ``xy``/``xz`` pair by default; every other projection
    is reachable through ``reductions``, so nothing the old class offered is out
    of reach.
    """

    def test_every_reduction_is_available(self, run3d):
        _, run = run3d
        ds = g3.Contours(run).dataset(reductions="all")
        for red in ("xy", "xz", "yz", "x", "y", "z"):
            assert f"phi_{red}" in ds

    def test_reductions_are_validated(self, run3d):
        _, run = run3d
        with pytest.raises(ValueError, match="unknown reduction"):
            g3.Contours(run).dataset(reductions=("xq",))

    def test_unknown_option_is_named(self, run3d):
        """A typo must not be swallowed as an unused keyword."""
        _, run = run3d
        with pytest.raises(TypeError, match="zlmi"):
            g3.Contours(run).dataset(zlmi=(0.0, 0.0))

    def test_line_reduction_is_one_dimensional(self, run3d):
        _, run = run3d
        ds = g3.Contours(run).dataset(reductions=("x",))
        assert ds["phi_x"].dims == ("time", "x")

    def test_plane_values_are_plain_means(self, run3d):
        """Not Jacobian-weighted: a cut is a picture of the field on the grid."""
        g, run = run3d
        z = np.asarray(run.coords[0]["z"])
        ds = g3.Contours(run).dataset(zlim=(z[0], z[-1]))
        expected = g.fields["phi"][0].astype(np.float32).mean(axis=2)
        assert np.allclose(np.asarray(ds["phi_xy"].isel(time=0)), expected,
                           rtol=1e-5)

    def test_fourier_view_renames_the_axis(self, run3d):
        _, run = run3d
        ds = g3.Contours(run).dataset(y_fourier=True)
        assert "ky" in ds["phi_xy"].dims
        assert "y" not in ds["phi_xy"].dims

    def test_time_average_drops_the_time_axis(self, run3d):
        _, run = run3d
        ds = g3.Contours(run).dataset(t_avg=True)
        assert "time" not in ds["phi_xy"].dims

    def test_moment_quantities_come_from_the_species_file(self, run3d):
        _, run = run3d
        ds = g3.Contours(run).dataset(quantities=("Q_es",),
                                      species="electrons")
        assert ds.attrs["species"] == "electrons"
        assert "Q_es_xy" in ds

    def test_unknown_quantity_lists_what_is_available(self, run3d):
        _, run = run3d
        with pytest.raises(KeyError, match="phi"):
            g3.Contours(run).compute(quantities=("nonsense",))

    def test_one_streaming_pass_serves_any_reduction(self, run3d):
        """
        ``compute`` builds all six regardless, so asking for more reductions
        after the first call must not re-read the file.
        """
        _, run = run3d
        diag = g3.Contours(run)
        diag.dataset()
        n_cached = len(diag._cache)
        diag.dataset(reductions="all")
        assert len(diag._cache) == n_cached

    def test_contours_gives_the_xy_and_xz_cuts(self, run3d):
        _, run = run3d
        ds = g3.Contours(run).dataset()
        assert set(ds.data_vars) == {"phi_xy", "phi_xz"}
        assert ds["phi_xy"].dims == ("time", "x", "y")
        assert ds["phi_xz"].dims == ("time", "x", "z")

    def test_contours_default_is_a_slice_at_z0_not_a_z_average(self, tmp_path):
        """
        Averaging over the whole of z smears the outboard and inboard sides of a
        global run together, so the default has to be a single cut.
        """
        g = make_gene3d_run(tmp_path / "run", nx0=8, nz0=8, n_times=2,
                            physical=True)
        run = Run(g.folder)
        z = np.asarray(run.coords[0]["z"])
        iz0 = int(np.argmin(np.abs(z)))
        raw = g.fields["phi"][0].astype(np.float32)

        ds = g3.Contours(run).dataset()
        got = np.asarray(ds["phi_xy"].isel(time=0))
        assert np.allclose(got, raw[:, :, iz0])
        assert not np.allclose(got, raw.mean(axis=2))

    def test_contours_xz_default_is_a_slice_at_y0(self, tmp_path):
        g = make_gene3d_run(tmp_path / "run", nx0=8, nz0=8, n_times=2,
                            physical=True)
        run = Run(g.folder)
        iy0 = int(np.argmin(np.abs(np.asarray(run.coords[0]["y"]))))
        raw = g.fields["phi"][0].astype(np.float32)
        ds = g3.Contours(run).dataset()
        assert np.allclose(np.asarray(ds["phi_xz"].isel(time=0)),
                           raw[:, iy0, :])

    def test_explicit_zlim_range_still_averages(self, tmp_path):
        g = make_gene3d_run(tmp_path / "run", nx0=8, nz0=8, n_times=2,
                            physical=True)
        run = Run(g.folder)
        z = np.asarray(run.coords[0]["z"])
        raw = g.fields["phi"][0].astype(np.float32)
        ds = g3.Contours(run).dataset(zlim=(z[0], z[-1]))
        assert np.allclose(np.asarray(ds["phi_xy"].isel(time=0)),
                           raw.mean(axis=2))

    def test_options_do_not_leak_between_calls(self, tmp_path):
        """
        An explicit zlim on one call must not silently persist into the next,
        which would show a different cut than the caller asked for.
        """
        g = make_gene3d_run(tmp_path / "run", nx0=8, nz0=8, n_times=2,
                            physical=True)
        run = Run(g.folder)
        z = np.asarray(run.coords[0]["z"])
        iz0 = int(np.argmin(np.abs(z)))
        raw = g.fields["phi"][0].astype(np.float32)
        diag = g3.Contours(run)

        diag.dataset(zlim=(z[2], z[2]))                 # override
        after = np.asarray(diag.dataset()["phi_xy"].isel(time=0))
        assert np.allclose(after, raw[:, :, iz0])

    def test_contours_puts_x_on_the_horizontal_axis(self, run3d, headless):
        """Both cuts share a radial axis, so they can be read together."""
        _, run = run3d
        figs = g3.Contours(run).plot()
        labelled = [(a.get_title(), a.get_xlabel(), a.get_ylabel())
                    for a in figs[0].axes if a.get_title()]
        xy = next(t for t in labelled if t[0].startswith("xy"))
        xz = next(t for t in labelled if t[0].startswith("xz"))
        assert xy[1] == r"$x/a$"
        assert xz[1] == r"$x/a$"
        assert xz[2] == r"$z/\pi$"

    def test_contour_titles_name_the_held_coordinate(self, run3d, headless):
        _, run = run3d
        figs = g3.Contours(run).plot()
        titles = [a.get_title() for a in figs[0].axes if a.get_title()]
        assert any(t.startswith("xy") and "z=0" in t for t in titles)
        assert any(t.startswith("xz") and "y=0" in t for t in titles)

    def test_contours_accepts_options_at_plot_time(self, run3d):
        """run.contours is a property, so selection happens later."""
        _, run = run3d
        diag = g3.Contours(run)
        ds = diag.dataset(quantities=("phi", "n"))
        assert set(ds.data_vars) == {"phi_xy", "phi_xz", "n_xy", "n_xz"}


# ---------------------------------------------------------------------------
# Growth rate and frequency
# ---------------------------------------------------------------------------

class TestGrowthRate3D:

    def test_pure_exponential_growth_is_recovered(self, tmp_path):
        """A field scaled by exp(gamma t) must give back gamma."""
        import h5py
        gamma = 0.037
        g = make_gene3d_run(tmp_path / "run", n_times=8, physical=True)
        with h5py.File(g.folder / "field.dat.h5", "a") as f:
            times = np.asarray(f["field/time"][...])
            base = np.asarray(f["field/phi/0000000000"][...])
            for it, tv in enumerate(times):
                f[f"field/phi/{it:010d}"][...] = base * np.exp(gamma * tv)
        diag = g3.GrowthRate(Run(g.folder))
        assert diag.gamma() == pytest.approx(gamma, rel=1e-3)

    def test_rescaling_is_split_out(self, tmp_path):
        """
        A drop in amplitude is a renormalisation, not decay. Fitting across it
        biases the growth rate towards zero.
        """
        import h5py
        gamma = 0.05
        g = make_gene3d_run(tmp_path / "run", n_times=10, physical=True)
        with h5py.File(g.folder / "field.dat.h5", "a") as f:
            times = np.asarray(f["field/time"][...])
            base = np.asarray(f["field/phi/0000000000"][...])
            for it, tv in enumerate(times):
                scale = np.exp(gamma * tv)
                if it >= 5:                       # renormalise by 1/1000
                    scale *= 1e-3
                f[f"field/phi/{it:010d}"][...] = base * scale
        diag = g3.GrowthRate(Run(g.folder))
        raw = diag.compute()
        assert len(raw["segments"]) == 2
        assert np.allclose(raw["gamma_segments"], gamma, rtol=1e-3)

    def test_too_few_snapshots_is_reported(self, tmp_path):
        g = make_gene3d_run(tmp_path / "run", n_times=1)
        with pytest.raises(ValueError, match="at least two"):
            g3.GrowthRate(Run(g.folder)).compute()

    def test_dataset_records_the_summary(self, run3d):
        _, run = run3d
        ds = g3.GrowthRate(run).dataset()
        assert "gamma" in ds.attrs
        assert "n_rescalings" in ds.attrs

    def test_omega_view_shares_the_computation(self, run3d):
        _, run = run3d
        assert "amplitude" in g3.Omega(run).dataset()


# ---------------------------------------------------------------------------
# Time traces, GAM, chi
# ---------------------------------------------------------------------------

class TestTracesAndDerived:

    def test_time_trace_is_the_volume_average(self, run3d):
        g, run = run3d
        ds = g3.TimeTraces(run, quantities=("phi",)).dataset()
        J = run.geometry[0]["Jacobian"]
        expected = np.average(g.fields["phi"][0].astype(np.float32), weights=J)
        assert float(ds["phi"].isel(time=0)) == pytest.approx(expected, rel=1e-5)

    def test_time_trace_has_a_ky_companion(self, run3d):
        _, run = run3d
        ds = g3.TimeTraces(run, quantities=("phi",)).dataset()
        assert ds["phi_ky"].dims == ("time", "ky")

    @staticmethod
    def _write_zonal_oscillation(folder, omega, nx0):
        """
        Replace phi with a zonal standing wave ringing at *omega*.

        The base fixture field is built from cos(m*y) modes only, so it has no
        ky = 0 component at all — a GAM diagnostic has nothing to measure on it.
        This writes cos(k_x x) * cos(omega t), which is y-independent and so is
        pure zonal.
        """
        import h5py
        with h5py.File(folder / "field.dat.h5", "a") as f:
            times = np.asarray(f["field/time"][...])
            shape = f["field/phi/0000000000"].shape        # (nz, ny, nx)
            kx_profile = np.cos(2 * np.pi * np.arange(nx0) / nx0)
            zonal = np.broadcast_to(kx_profile, shape).copy()
            for it, tv in enumerate(times):
                f[f"field/phi/{it:010d}"][...] = zonal * np.cos(omega * tv)

    def test_gam_traces_start_at_one(self, tmp_path):
        g = make_gene3d_run(tmp_path / "run", nx0=8, n_times=8, physical=True)
        self._write_zonal_oscillation(g.folder, 0.05, 8)
        ds = g3.Gam(Run(g.folder)).dataset()
        assert ds.attrs["has_zonal_component"] == 1
        for name in ("phi_zonal_mid", "phi_kx1"):
            assert float(ds[name].isel(time=0)) == pytest.approx(1.0)

    def test_gam_recovers_a_known_frequency(self, tmp_path):
        """
        A cos(omega t) zonal mode must give back omega.

        The fixture samples every 10 time units, so omega has to sit well below
        the Nyquist frequency pi/dt = 0.314 for the oscillation to be resolved
        at all.
        """
        omega = 0.05
        g = make_gene3d_run(tmp_path / "run", nx0=8, n_times=40, physical=True)
        self._write_zonal_oscillation(g.folder, omega, 8)
        ds = g3.Gam(Run(g.folder)).dataset()
        assert ds.attrs["gam_frequency"] == pytest.approx(omega, rel=0.05)

    def test_gam_refuses_without_a_zonal_component(self, run3d):
        """
        The base fixture field has no ky = 0 part, so the zonal trace is
        round-off. Normalising noise by noise yields a plausible oscillation
        that a fit turns into a confident frequency — it must report nothing.
        """
        _, run = run3d
        ds = g3.Gam(run).dataset()
        assert ds.attrs["has_zonal_component"] == 0
        assert np.isnan(ds.attrs["gam_frequency"])
        assert np.all(np.isnan(np.asarray(ds["phi_kx1"])))

    def test_gam_frequency_is_nan_without_oscillation(self, tmp_path):
        """A monotonic trace has no crossings; report nothing, not a number."""
        import h5py
        g = make_gene3d_run(tmp_path / "run", nx0=8, n_times=6, physical=True)
        with h5py.File(g.folder / "field.dat.h5", "a") as f:
            shape = f["field/phi/0000000000"].shape
            zonal = np.ones(shape)
            for it in range(6):
                f[f"field/phi/{it:010d}"][...] = zonal * (it + 1)
        ds = g3.Gam(Run(g.folder)).dataset()
        assert np.isnan(ds.attrs["gam_frequency"])

    def test_chi_is_flux_over_gradient_drive(self, run3d):
        _, run = run3d
        diag = g3.ChiGradient(run)
        ds = diag.dataset()
        fluxes = g3.Fluxes2D(run).dataset()
        profs = g3.Profiles(run).dataset()
        geom_fac = np.asarray(ds["geom_factor"])
        Q = np.asarray(fluxes["Q_total"].sel(species="ions").isel(time=0))
        n = np.asarray(profs["n"].sel(species="ions").isel(time=0))
        T = np.asarray(profs["T"].sel(species="ions").isel(time=0))
        omt = np.asarray(profs["omt"].sel(species="ions").isel(time=0))
        expected = Q / (geom_fac * n * T * omt)
        got = np.asarray(ds["chi"].sel(species="ions").isel(time=0))
        assert np.allclose(got, expected, equal_nan=True)

    def test_chi_has_an_si_companion(self, run3d):
        _, run = run3d
        ds = g3.ChiGradient(run).dataset()
        assert ds["chi_SI"].attrs["units"] == "m^2 s^-1"


# ---------------------------------------------------------------------------
# Geometry, source moments, velocity space
# ---------------------------------------------------------------------------

class TestAuxiliary:

    def test_geometry_dataset_has_the_three_dimensional_terms(self, run3d):
        _, run = run3d
        ds = g3.GeometryPlots(run).dataset()
        assert ds["gxx"].dims == ("x", "y", "z")
        assert ds["K_x"].dims == ("x", "y", "z")
        assert ds["q"].dims == ("x",)

    def test_srcmom_has_six_variables(self, run3d):
        _, run = run3d
        ds = g3.SrcMom(run).dataset()
        assert len(ds.data_vars) == 6
        assert ds["ck_heat_M00"].dims == ("species", "time", "x")

    def test_srcmom_absence_is_reported(self, tmp_path):
        g = make_gene3d_run(tmp_path / "run", write_srcmom=False)
        with pytest.raises(FileNotFoundError, match="istep_srcmom"):
            g3.SrcMom(Run(g.folder)).compute()

    def test_vsp_keeps_the_velocity_axes(self, run3d):
        _, run = run3d
        ds = g3.VspSlice(run).dataset()
        assert ds["G_es"].dims == ("time", "z", "vpar", "mu", "species")


# ---------------------------------------------------------------------------
# Planes and VTK export
# ---------------------------------------------------------------------------

class TestPlanesAndVis:

    def test_remapped_plane_has_the_angular_grid(self, run3d):
        _, run = run3d
        ds = run.planes(quantities=("phi",), n_theta=24, n_phi=12).dataset()
        assert ds["phi"].dims == ("x", "varphi", "theta")
        assert ds.sizes["theta"] == 24
        assert ds.sizes["varphi"] == 12

    def test_toroidal_angle_is_not_called_phi(self, run3d):
        """`phi` is the potential; a coordinate of the same name collides."""
        _, run = run3d
        ds = run.planes(quantities=("phi",), n_theta=16, n_phi=8).dataset()
        assert "varphi" in ds.coords
        assert "phi" in ds.data_vars

    def test_remapping_stays_inside_the_data_range(self, run3d):
        """
        The y interpolation is periodic, so a wrapped point comes back into the
        box. Extrapolating instead would put values outside the original range.
        """
        g, run = run3d
        ds = run.planes(quantities=("phi",), n_theta=32, n_phi=16,
                        t_avg=True).dataset()
        source = np.stack([f.astype(np.float32) for f in g.fields["phi"]])
        lo, hi = source.min(), source.max()
        plane = np.asarray(ds["phi"])
        assert plane.min() >= lo - 1e-4 * abs(lo)
        assert plane.max() <= hi + 1e-4 * abs(hi)

    def test_mode_spectrum_scales_n_by_n0_global(self, run3d):
        _, run = run3d
        diag = run.planes(quantities=("phi",), n_theta=16, n_phi=8)
        n, m, power = diag.mode_spectrum()
        n0 = float(run.params.get(0)["box"]["n0_global"])
        assert n[1] == pytest.approx(n0)
        assert power.shape == (8, 16)

    def test_missing_q_profile_is_reported(self, tmp_path):
        import h5py
        g = make_gene3d_run(tmp_path / "run")
        with h5py.File(g.folder / "circular.dat.h5", "a") as f:
            del f["profile"]["q_prof"]
        run = Run(g.folder)
        with pytest.raises(ValueError, match="safety-factor"):
            run.planes(quantities=("phi",)).compute()

    def test_vtk_export_writes_one_file_per_snapshot(self, run3d, tmp_path):
        _, run = run3d
        written = run.vis3d(quantities=("phi",)).write_vtk(
            out_dir=tmp_path / "vtk")
        assert len(written) == 6
        text = open(written[0]).read()
        assert "DATASET STRUCTURED_GRID" in text
        assert "DIMENSIONS 16 8 4" in text
        assert "SCALARS phi float" in text

    def test_vtk_needs_the_cartesian_grid(self, tmp_path):
        import h5py
        g = make_gene3d_run(tmp_path / "run", n_times=2)
        with h5py.File(g.folder / "circular.dat.h5", "a") as f:
            del f["cart_coords"]
        run = Run(g.folder)
        with pytest.raises(ValueError, match="cart_coords"):
            run.vis3d(quantities=("phi",)).write_vtk(out_dir=tmp_path)

    def test_gvec_path_reports_its_dependency(self, run3d):
        _, run = run3d
        diag = run.vis3d(quantities=("phi",))
        with pytest.raises((ImportError, ValueError, NotImplementedError)):
            diag.write_vtk_via_gvec(gvec_file="nonexistent.dat")


# ---------------------------------------------------------------------------
# Run facade
# ---------------------------------------------------------------------------

class TestFacade:

    @pytest.mark.parametrize("name, cls", [
        ("spectra", g3.Spectra), ("fluxes2d", g3.Fluxes2D),
        ("profiles", g3.Profiles), ("shearing", g3.ShearingRate),
        ("contours", g3.Contours), ("growthrate", g3.GrowthRate),
        ("amplitude", g3.AmplitudeSpectra),
        ("profile_diag", g3.ProfileDiag), ("gam", g3.Gam),
        ("chi", g3.ChiGradient), ("omega", g3.Omega),
        ("geometry_plots", g3.GeometryPlots), ("srcmom", g3.SrcMom),
        ("vsp", g3.VspSlice),
    ])
    def test_properties_resolve_to_the_merged_class(self, run3d, name, cls):
        _, run = run3d
        assert isinstance(getattr(run, name), cls)

    @pytest.mark.parametrize("name, cls", [
        ("timetraces", g3.TimeTraces),
        ("planes", g3.Planes), ("vis3d", g3.Vis),
    ])
    def test_callables_resolve_to_the_merged_class(self, run3d, name, cls):
        _, run = run3d
        assert isinstance(getattr(run, name)(quantities=("phi",)), cls)

    def test_ballooning_refuses_for_gene3d(self, run3d):
        """There is no single ky mode in a real-space-y run."""
        _, run = run3d
        with pytest.raises(NotImplementedError, match="real-space in y"):
            run.ballooning(ky=0.1)

    @pytest.mark.parametrize("name", ["gam", "chi", "omega", "geometry_plots",
                                      "srcmom", "vsp"])
    def test_gene3d_only_properties_refuse_elsewhere(self, tmp_path, name):
        from tests.conftest import MINIMAL_PARAMS
        (tmp_path / "parameters").write_text(MINIMAL_PARAMS)
        (tmp_path / "nrg").touch()
        run = Run(tmp_path, ext=[""])
        assert run.geometry_kind == "flux_tube"
        # Each diagnostic declares `supported`; the base refuses on construction.
        with pytest.raises(NotImplementedError, match="supports xy_global"):
            getattr(run, name)

    def test_spectral_runs_keep_their_diagnostics(self, tmp_path):
        from tests.conftest import MINIMAL_PARAMS
        (tmp_path / "parameters").write_text(MINIMAL_PARAMS)
        (tmp_path / "nrg").touch()
        run = Run(tmp_path, ext=[""])
        # One class per diagnostic now, so the check is that the spectral
        # geometry is reported and the GENE-3D-only paths refuse.
        assert isinstance(run.spectra, g3.Spectra)
        assert run.spectra.geometry_kind == "flux_tube"
        # A flux tube has no (x, ky) map — x is spectral — so `which` must be
        # refused, not quietly ignored the way `maps` once was.
        with pytest.raises(ValueError, match="no.*x, ky.*map|global geometries"):
            run.spectra.plot(which="map")
        assert not run.fluxes2d.is_3d


# ---------------------------------------------------------------------------
# Plot smoke tests
# ---------------------------------------------------------------------------

class TestPlots:

    @pytest.mark.parametrize("name", [
        "spectra", "fluxes2d", "profiles", "shearing", "contours",
        "growthrate", "amplitude", "profile_diag", "gam", "chi",
        "omega", "geometry_plots", "srcmom", "vsp",
    ])
    def test_property_plots_run(self, run3d, headless, name):
        _, run = run3d
        getattr(run, name).plot()

    @pytest.mark.parametrize("name, kw", [
        ("timetraces", {"quantities": ("phi",)}),
        ("planes", {"quantities": ("phi",), "n_theta": 16, "n_phi": 8}),
    ])
    def test_callable_plots_run(self, run3d, headless, name, kw):
        _, run = run3d
        getattr(run, name)(**kw).plot()

    @pytest.mark.parametrize("reductions", ["all", ("yz",), ("x", "z")])
    def test_contour_reductions_plot(self, run3d, headless, reductions):
        """Planes and 1-D lines both draw, whichever mix is asked for."""
        _, run = run3d
        run.contours.plot(reductions=reductions)
