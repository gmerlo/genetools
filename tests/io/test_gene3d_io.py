"""
Tests for the GENE-3D I/O layer: params classification, coordinates, geometry
and equilibrium profiles.
"""

import h5py
import numpy as np
import pytest

from genetools.io.coordinates import Coordinates
from genetools.io.geometry import Geometry
from genetools.io.params import Params
from genetools.io.profiles_loader import load_equilibrium_profiles
from tests.gene3d_fixture import gene3d_grids, make_gene3d_run


@pytest.fixture
def g3d(tmp_path):
    return make_gene3d_run(tmp_path / "run")


# ---------------------------------------------------------------------------
# Params: reconciling &info against &general / &in_out / &box
# ---------------------------------------------------------------------------

class TestParamsClassification:

    def test_gene3d_is_not_mistaken_for_a_flux_tube(self, g3d):
        """
        GENE-3D hard-codes ``x_local = F``/``y_local = F`` into ``&info``, not
        ``&general``. Reading only ``&general`` leaves the defaults in place and
        every diagnostic then branches as though this were a flux tube — the
        worst possible failure mode, because nothing errors.
        """
        p = Params(g3d.folder, [".dat"])
        assert p.geometry_kind() == "xy_global"
        assert p.is_3d() is True
        general = p.get(0)["general"]
        assert general["x_local"] is False
        assert general["y_local"] is False

    def test_write_h5_is_mirrored_into_in_out(self, g3d):
        """GENE puts write_h5 in &in_out; GENE-3D puts it in &info."""
        d = Params(g3d.folder, [".dat"]).get(0)
        assert d["in_out"]["write_h5"] is True
        assert d["info"]["write_h5"] is True

    def test_ly_is_mirrored_into_box(self, g3d):
        """
        GENE-3D writes the binormal box length in &info, and writes no nky0 or
        kymin anywhere, so the coordinate builder needs ly under &box.
        """
        box = Params(g3d.folder, [".dat"]).get(0)["box"]
        assert box["ly"] == pytest.approx(80.0)
        assert "nky0" not in box
        assert "kymin" not in box

    @pytest.mark.parametrize(
        "x_local, y_local, kind",
        [(True, True, "flux_tube"), (False, True, "x_global"),
         (True, False, "y_global"), (False, False, "xy_global")])
    def test_all_four_geometries_are_classified(self, tmp_path, x_local,
                                                y_local, kind):
        text = (f"&general\n x_local = {'T' if x_local else 'F'}\n"
                f" y_local = {'T' if y_local else 'F'}\n/\n"
                "&box\n nx0 = 4\n nky0 = 2\n nz0 = 4\n/\n"
                "&info\n precision = 'DOUBLE'\n/\n")
        (tmp_path / "parameters").write_text(text)
        p = Params(tmp_path, [""])
        assert p.geometry_kind() == kind
        assert p.is_3d() == (kind == "xy_global")

    def test_general_wins_over_info_when_both_are_present(self, tmp_path):
        """A real GENE file carries x_local in &general; that is authoritative."""
        (tmp_path / "parameters").write_text(
            "&general\n x_local = T\n y_local = T\n/\n"
            "&box\n nx0 = 4\n nky0 = 2\n nz0 = 4\n/\n"
            "&info\n x_local = F\n y_local = F\n/\n")
        assert Params(tmp_path, [""]).geometry_kind() == "flux_tube"


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

class TestCoordinates:

    def test_grids_come_from_coord_h5(self, g3d):
        c = Coordinates(g3d.folder, [".dat"], Params(g3d.folder, [".dat"]))[0]
        assert np.allclose(c["x"], g3d.grids["xval"])
        assert np.allclose(c["x_o_a"], g3d.grids["xval_a"])
        assert np.allclose(c["y"], g3d.grids["yval"])
        assert np.allclose(c["z"], g3d.grids["zval"])

    def test_y_grid_starts_at_zero(self, g3d):
        """GENE-3D's yval runs 0..ly-dy; it is not centred like the local case."""
        c = Coordinates(g3d.folder, [".dat"], Params(g3d.folder, [".dat"]))[0]
        assert c["y"][0] == pytest.approx(0.0)
        assert c["y"][-1] < 80.0

    @pytest.mark.parametrize("nz0", [4, 5])
    def test_namelist_fallback_reproduces_the_file(self, tmp_path, nz0):
        """
        Without coord.h5 the grids must be rebuilt exactly — including the
        half-cell z shift GENE-3D applies for odd nz0, and the rad_bc_type
        dependence of dx.
        """
        run = make_gene3d_run(tmp_path / "run", nz0=nz0)
        params = Params(run.folder, [".dat"])
        from_file = Coordinates(run.folder, [".dat"], params)[0]
        (run.folder / "coord.dat.h5").unlink()
        rebuilt = Coordinates(run.folder, [".dat"], params)[0]
        for key in ("x", "x_o_a", "y", "z", "vp"):
            assert np.allclose(rebuilt[key], from_file[key]), key

    def test_odd_nz0_is_shifted_by_half_a_cell(self, tmp_path):
        run = make_gene3d_run(tmp_path / "run", nz0=5)
        c = Coordinates(run.folder, [".dat"], Params(run.folder, [".dat"]))[0]
        dz = 2 * np.pi / 5
        assert np.allclose(c["z"], -np.pi + np.arange(5) * dz + dz / 2)

    def test_ky_is_the_fft_grid_of_y(self, g3d):
        """There is no kymin in the parameters file; ky comes from ly and ny0."""
        c = Coordinates(g3d.folder, [".dat"], Params(g3d.folder, [".dat"]))[0]
        assert np.allclose(c["ky"], 2 * np.pi * np.fft.fftfreq(g3d.ny0,
                                                               d=80.0 / g3d.ny0))
        assert c["ky"][1] == pytest.approx(2 * np.pi / 80.0)

    def test_ky_pos_excludes_the_negative_nyquist(self, g3d):
        """For even ny0, fftfreq signs the Nyquist bin negative."""
        c = Coordinates(g3d.folder, [".dat"], Params(g3d.folder, [".dat"]))[0]
        assert np.all(c["ky_pos"] >= 0)
        assert c["ky_pos"].size == g3d.ny0 // 2


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class TestGeometry:

    def test_metric_and_field_terms_are_three_dimensional(self, g3d):
        g = Geometry(g3d.folder, [".dat"], Params(g3d.folder, [".dat"]))[0]
        shape = (g3d.nx0, g3d.ny0, g3d.nz0)
        assert g["Jacobian"].shape == shape
        assert g["metric"]["gxx"].shape == shape
        assert g["dBdz"].shape == shape

    def test_values_survive_the_axis_flip(self, g3d):
        g = Geometry(g3d.folder, [".dat"], Params(g3d.folder, [".dat"]))[0]
        assert np.allclose(g["Jacobian"], g3d.geometry["Jacobian"])
        assert np.allclose(g["metric"]["gxx"], g3d.geometry["g^xx"])
        assert np.allclose(g["metric"]["gyz"], g3d.geometry["g^yz"])

    def test_c_y_and_c_xy_are_radial_arrays(self, g3d):
        """Scalars in a flux tube, but profiles over x in GENE-3D."""
        g = Geometry(g3d.folder, [".dat"], Params(g3d.folder, [".dat"]))[0]
        assert g["metric"]["C_y"].shape == (g3d.nx0,)
        assert np.allclose(g["metric"]["C_y"], g3d.geometry["C_y"])

    def test_curvature_is_taken_from_the_file(self, g3d):
        """GENE-3D writes K_x and K_y, so there is nothing to recompute."""
        g = Geometry(g3d.folder, [".dat"], Params(g3d.folder, [".dat"]))[0]
        assert np.allclose(g["curv"]["K_x"], g3d.geometry["K_x"])
        assert np.allclose(g["curv"]["K_y"], g3d.geometry["K_y"])

    def test_q_profile_is_radial(self, g3d):
        g = Geometry(g3d.folder, [".dat"], Params(g3d.folder, [".dat"]))[0]
        assert np.allclose(g["profiles"]["q"], g3d.geometry["q_prof"])

    def test_area_from_file_agrees_with_recomputing_it(self, g3d):
        """
        GENE-3D writes dVdx and sqrtgxx_fs; the reader prefers them. They must
        match what the metric gives, or the two routes disagree silently.
        """
        params = Params(g3d.folder, [".dat"])
        g = Geometry(g3d.folder, [".dat"], params)[0]
        Lref = params.get(0)["units"]["Lref"]
        A0 = (2 * np.pi) ** 2 * np.abs(g3d.geometry["C_y"]) * Lref ** 2
        J, gxx = g3d.geometry["Jacobian"], g3d.geometry["g^xx"]
        assert np.allclose(g["area"]["dVdx"], A0 * J.mean(axis=(1, 2)))
        assert np.allclose(g["area"]["Area"],
                           A0 * (np.sqrt(gxx) * J).mean(axis=(1, 2)))

    def test_area_falls_back_to_the_metric(self, g3d):
        params = Params(g3d.folder, [".dat"])
        expected = Geometry(g3d.folder, [".dat"], params)[0]["area"]
        with h5py.File(g3d.folder / "circular.dat.h5", "a") as f:
            del f["profile"]["dVdx"]
            del f["profile"]["sqrtgxx_fs"]
        got = Geometry(g3d.folder, [".dat"], params)[0]["area"]
        assert np.allclose(got["dVdx"], expected["dVdx"])
        assert np.allclose(got["Area"], expected["Area"])

    def test_missing_geometry_names_both_candidates(self, g3d):
        (g3d.folder / "circular.dat.h5").unlink()
        with pytest.raises(FileNotFoundError, match=r"circular\.dat.*\.h5"):
            Geometry(g3d.folder, [".dat"], Params(g3d.folder, [".dat"]))


# ---------------------------------------------------------------------------
# Equilibrium profiles
# ---------------------------------------------------------------------------

class TestProfiles:

    def test_ascii_is_preferred_and_read_correctly(self, g3d):
        prof = load_equilibrium_profiles(str(g3d.folder) + "/", ".dat", "ions")
        truth = g3d.profiles["ions"]
        assert np.allclose(prof["x_o_a"], truth["x_o_a"])
        assert np.allclose(prof["x_o_rho_ref"], truth["x_o_rho_ref"])
        assert np.allclose(prof["T"], truth["T"])
        assert np.allclose(prof["n"], truth["n"])

    def test_radial_columns_are_not_swapped(self, g3d):
        """
        Both codes write x/a first and x/rho_ref second. Swapping them rescales
        the radial axis by rhostar, which looks plausible and is wrong.
        """
        prof = load_equilibrium_profiles(str(g3d.folder) + "/", ".dat", "ions")
        assert prof["x_o_a"].max() < 1.0
        assert prof["x_o_rho_ref"].max() > 1.0

    def test_h5_is_used_when_the_text_file_is_absent(self, g3d):
        folder = str(g3d.folder) + "/"
        expected = load_equilibrium_profiles(folder, ".dat", "ions")
        (g3d.folder / "profiles_ions.dat").unlink()
        got = load_equilibrium_profiles(folder, ".dat", "ions")
        for key in ("x_o_a", "x_o_rho_ref", "T", "n", "omt", "omn"):
            assert np.allclose(got[key], expected[key]), key

    def test_last_block_wins_for_evolving_profiles(self, tmp_path):
        """
        GENE-3D appends a new block whenever the background profiles are
        updated; the current state is the last one, not the first.
        """
        nx = 4
        header = "#   x/a             x/rho_ref           T\n"
        first = "".join(f"{i:e} {10*i:e} 1.0 1.0 0.0 0.0\n" for i in range(nx))
        second = "".join(f"{i:e} {10*i:e} 2.0 3.0 0.0 0.0\n" for i in range(nx))
        (tmp_path / "profiles_ions.dat").write_text(
            header + "#   0.0\n" + first + "\n\n" + "#  10.0\n" + second)
        prof = load_equilibrium_profiles(str(tmp_path) + "/", ".dat", "ions")
        assert np.allclose(prof["T"], 2.0)
        assert np.allclose(prof["n"], 3.0)

    def test_missing_profile_names_both_candidates(self, g3d):
        (g3d.folder / "profiles_ions.dat").unlink()
        (g3d.folder / "profiles_ions.dat.h5").unlink()
        with pytest.raises(FileNotFoundError, match=r"profiles_ions\.dat.*\.h5"):
            load_equilibrium_profiles(str(g3d.folder) + "/", ".dat", "ions")


# ---------------------------------------------------------------------------
# Grid helper used by the fixture — worth pinning independently
# ---------------------------------------------------------------------------

class TestGridFormulas:

    def test_periodic_and_dirichlet_radial_spacing_differ(self):
        per = gene3d_grids(6, 4, 4, 60.0, 80.0, 0.5, 0.01, rad_bc_type=0)[0]
        dir_ = gene3d_grids(6, 4, 4, 60.0, 80.0, 0.5, 0.01, rad_bc_type=1)[0]
        assert np.diff(per)[0] == pytest.approx(60.0 / 6)
        assert np.diff(dir_)[0] == pytest.approx(60.0 / 5)

    def test_n_pol_widens_the_parallel_domain(self):
        z1 = gene3d_grids(4, 4, 8, 60.0, 80.0, 0.5, 0.01, n_pol=1)[3]
        z2 = gene3d_grids(4, 4, 8, 60.0, 80.0, 0.5, 0.01, n_pol=2)[3]
        assert z1[0] == pytest.approx(-np.pi)
        assert z2[0] == pytest.approx(-2 * np.pi)
