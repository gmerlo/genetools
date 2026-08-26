"""Tests for Run.eq_profiles and its wiring into the global diagnostics.

Global heat and momentum fluxes carry background n(x)/T(x) prefactors. If the
facade does not supply the equilibrium profiles, those fluxes are computed as
if the background were flat, which is silently wrong for a global run. These
tests pin the loading, the failure modes, and the fact that the profiles
actually reach the diagnostics.
"""

import textwrap

import numpy as np
import pytest

from genetools.run import Run


NX, NKY, NZ = 7, 3, 8
SPECIES = ("ions", "electrons")


def _write_parameters(folder, ext="_0001", x_local=False, nx=NX):
    """Write a parameters file for a minimal (global by default) run."""
    flag = ".true." if x_local else ".false."
    species_blocks = "".join(textwrap.dedent(f"""
        &species
          name = '{name}'
          dens = 1.0
          temp = 1.0
          mass = 1.0
          charge = 1.0
        /
        """) for name in SPECIES)
    (folder / f"parameters{ext}").write_text(textwrap.dedent(f"""
        &general
          x_local = {flag}
          y_local = .true.
        /

        &box
          nx0 = {nx}
          nky0 = {NKY}
          nz0 = {NZ}
          n_spec = {len(SPECIES)}
        /

        &info
          precision = 'double'
          n_fields = 1
          n_moms = 6
          nrgcols = 10
        /

        &units
          Lref = 1.0
          Bref = 1.0
          nref = 1.0
          Tref = 1.0
          mref = 1.0
        /
        """) + species_blocks)
    # set_runs discovers segments by scanning for nrg* files
    (folder / f"nrg{ext}").write_bytes(b"")


def _write_profiles(folder, species_name, ext="_0001", nx=NX, scale=1.0):
    """Write a synthetic profiles_<species><ext> file with nx radial points."""
    x = np.linspace(0.3, 0.7, nx)
    T = scale * (2.0 - x)          # falling temperature profile
    n = scale * (1.5 - 0.5 * x)    # falling density profile
    rows = np.column_stack([x, x, T, n, np.full(nx, 2.5), np.full(nx, 2.1)])
    lines = ["# x/rho_ref x/a T n omt omn\n", "#\n"]
    lines += ["  ".join(f"{v:.8e}" for v in row) + "\n" for row in rows]
    (folder / f"profiles_{species_name}{ext}").write_text("".join(lines))


@pytest.fixture
def global_run(tmp_path):
    """A minimal global run directory with equilibrium profiles present."""
    _write_parameters(tmp_path)
    for name in SPECIES:
        _write_profiles(tmp_path, name)
    return tmp_path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestEqProfilesLoading:

    def test_loads_all_species(self, global_run):
        run = Run(global_run)
        assert not run.is_local
        eq = run.eq_profiles
        assert set(eq) == set(SPECIES)
        for name in SPECIES:
            assert eq[name]["T"].shape == (NX,)
            assert eq[name]["n"].shape == (NX,)

    def test_values_match_file(self, global_run):
        run = Run(global_run)
        x = np.linspace(0.3, 0.7, NX)
        np.testing.assert_allclose(run.eq_profiles["ions"]["T"], 2.0 - x)
        np.testing.assert_allclose(run.eq_profiles["ions"]["n"], 1.5 - 0.5 * x)

    def test_is_cached(self, global_run):
        run = Run(global_run)
        assert run.eq_profiles is run.eq_profiles

    def test_assignable_override(self, global_run):
        """Users may supply profiles from another source."""
        run = Run(global_run)
        custom = {name: {"T": np.ones(NX), "n": np.ones(NX)} for name in SPECIES}
        run.eq_profiles = custom
        assert run.eq_profiles is custom

    def test_local_run_returns_none(self, tmp_path):
        _write_parameters(tmp_path, x_local=True)
        run = Run(tmp_path)
        assert run.is_local
        assert run.eq_profiles is None

    def test_local_run_ignores_missing_files(self, tmp_path):
        """A local run must not need profile files at all."""
        _write_parameters(tmp_path, x_local=True)
        assert Run(tmp_path).eq_profiles is None


# ---------------------------------------------------------------------------
# Failure modes — loud, not silently wrong
# ---------------------------------------------------------------------------

class TestEqProfilesErrors:

    def test_missing_file_raises_naming_the_species(self, tmp_path):
        _write_parameters(tmp_path)
        _write_profiles(tmp_path, "ions")   # electrons deliberately absent
        run = Run(tmp_path)
        with pytest.raises(FileNotFoundError, match="profiles_electrons_0001"):
            run.eq_profiles

    def test_all_missing_files_reported_together(self, tmp_path):
        _write_parameters(tmp_path)
        run = Run(tmp_path)
        with pytest.raises(FileNotFoundError) as exc:
            run.eq_profiles
        for name in SPECIES:
            assert f"profiles_{name}_0001" in str(exc.value)

    def test_wrong_radial_grid_raises(self, tmp_path):
        """A profile on a different radial grid would broadcast-fail deep in
        the flux computation; catch it up front with a clear message."""
        _write_parameters(tmp_path, nx=NX)
        for name in SPECIES:
            _write_profiles(tmp_path, name, nx=NX + 3)
        run = Run(tmp_path)
        with pytest.raises(ValueError, match=f"nx0={NX}"):
            run.eq_profiles


# ---------------------------------------------------------------------------
# Wiring — the profiles must actually reach the diagnostics
# ---------------------------------------------------------------------------

class TestDiagnosticWiring:

    def test_spectra_passes_profiles(self, global_run, monkeypatch):
        run = Run(global_run)
        seen = {}

        def spy(self, fld, moms, coords, geom, params, t0, t1,
                equilibrium_profiles=None):
            seen["eq"] = equilibrium_profiles

        monkeypatch.setattr(
            "genetools.diagnostics.spectra.Spectra._compute_global",
            spy)
        monkeypatch.setattr(Run, "field", property(lambda self: None))
        monkeypatch.setattr(Run, "_mom_list", lambda self: [])
        monkeypatch.setattr(Run, "coords", property(lambda self: [{}]))
        monkeypatch.setattr(Run, "geometry", property(lambda self: [{}]))

        run.spectra.compute()
        assert seen["eq"] is not None, "spectra got no equilibrium profiles"
        assert set(seen["eq"]) == set(SPECIES)

    def test_fluxes2d_passes_profiles(self, global_run, monkeypatch):
        run = Run(global_run)
        seen = {}

        def spy(self, fld, moms, coords, geom, params, t0, t1,
                equilibrium_profiles=None):
            seen["eq"] = equilibrium_profiles

        monkeypatch.setattr(
            "genetools.diagnostics.fluxes2d.Fluxes2D.compute_and_save", spy)
        monkeypatch.setattr(Run, "field", property(lambda self: None))
        monkeypatch.setattr(Run, "_mom_list", lambda self: [])
        monkeypatch.setattr(Run, "coords", property(lambda self: [{}]))
        monkeypatch.setattr(Run, "geometry", property(lambda self: [{}]))

        run.fluxes2d.compute()
        assert seen["eq"] is not None, "fluxes2d got no equilibrium profiles"
        assert set(seen["eq"]) == set(SPECIES)


# ---------------------------------------------------------------------------
# The correction is not cosmetic: it changes the computed fluxes
# ---------------------------------------------------------------------------

class TestProfileCorrectionMatters:

    def test_prefactors_change_heat_flux(self, global_run):
        """Q built with real profiles must differ from the flat-profile result."""
        from genetools.diagnostics.fluxes2d import Fluxes2D

        run = Run(global_run)
        params = run.params.get(0)
        flat = Fluxes2D.build_prefactors(params, {}, None)
        real = Fluxes2D.build_prefactors(params, {}, run.eq_profiles)

        assert flat == {}, "no profiles -> no correction (the silent-wrong path)"
        assert set(real) == set(SPECIES)
        n_map = real["ions"]["n_map"]
        T_map = real["ions"]["T_map"]
        assert n_map.shape == (NX, 1, NZ)
        # a genuinely radially varying correction, not all-ones
        assert not np.allclose(n_map, 1.0)
        assert not np.allclose(T_map, 1.0)
        x = np.linspace(0.3, 0.7, NX)
        np.testing.assert_allclose(T_map[:, 0, 0], 2.0 - x)
        np.testing.assert_allclose(n_map[:, 0, 0], 1.5 - 0.5 * x)
