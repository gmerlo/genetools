# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
`Fluxes2D` against GENE-3D's own two implementations, on the same data.

Three independent transcriptions of the radial flux profile are run over one
synthetic run and compared number for number:

* **Fortran** — ``diag_3d.F90``'s ``diag_prof`` path: ``flux_surface_average``
  from ``spatial_averages.F90:19``, then the species factors applied at
  ``diag_3d.F90:600-603``, then ``Gammaref``/``Qref`` from ``:915-916``.
* **GUI** — ``diag-python/GENE_gui_python/diagnostics/diag_fluxesgene3d.py:88-97``,
  transcribed literally, *including* the species factor it applies to the
  electrostatic terms and omits from the electromagnetic ones.
* **genetools** — ``run.fluxes2d.dataset()``.

The point is not that three things agree in spirit. The Fortran is the
definition, so genetools must equal it to round-off; where the GUI differs, the
test records by how much and why, so the discrepancy is a documented fact rather
than a surprise in someone's notebook.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gene3d_fixture import make_gene3d_run          # noqa: E402

from genetools import Run                            # noqa: E402

ELEMENTARY_CHARGE = 1.60217662e-19
PROTON_MASS = 1.6726219e-27


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A run with fluxes derived from the fields, as diag_3d.F90 derives them."""
    folder = tmp_path_factory.mktemp("flux_equiv") / "run"
    make_gene3d_run(folder, nx0=12, ny0=8, nz0=6, n_times=4, physical=True,
                    temps=(1.0, 2.5), denss=(1.0, 0.8))
    return Run(str(folder))


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def fortran_flux_surface_average(dat, jacobian):
    """
    ``spatial_averages.F90:19`` verbatim.

    ``avg = sum_{y,z}(dat*J) / (avg_jaco_yz * nz0 * ny0)`` with
    ``avg_jaco_yz(x) = sum_{y,z} J / (ny0*nz0)`` (``geometry.F90:1007-1012``),
    so the divisor is ``sum_{y,z} J`` — written out the long way here so the
    identity is visible rather than assumed.
    """
    nx, ny, nz = dat.shape
    avg = np.zeros(nx)
    for k in range(nz):
        for j in range(ny):
            avg += dat[:, j, k] * jacobian[:, j, k]
    avg_jaco_yz = jacobian.sum(axis=(1, 2)) / (ny * nz)
    return avg / (avg_jaco_yz * nz * ny)


def fortran_profile(run, species, jacobian, fluxes):
    """
    ``diag_3d.F90`` ``diag_prof``: FSA, sum ES+EM, then the species factors.

    ``var_prof(:,5) = mom_avg(:,5) + mom_avg(:,6)`` (:575),
    ``var_prof(:,6) = mom_avg(:,7) + mom_avg(:,8)`` (:578),
    ``out_arr(:,5) *= dens`` (:600), ``out_arr(:,6) *= dens*temp`` (:603).
    """
    spec = next(s for s in run.params.get(0)["species"]
                if s["name"] == species)
    gamma = (fortran_flux_surface_average(fluxes["Gamma_es"], jacobian)
             + fortran_flux_surface_average(fluxes["Gamma_em"], jacobian))
    heat = (fortran_flux_surface_average(fluxes["Q_es"], jacobian)
            + fortran_flux_surface_average(fluxes["Q_em"], jacobian))
    return (gamma * float(spec["dens"]),
            heat * float(spec["dens"]) * float(spec["temp"]))


def gui_profile(run, species, jacobian, fluxes):
    """
    ``diag_fluxesgene3d.py:88-97`` verbatim, bug included.

    The GUI multiplies ``Q_es`` by ``temp*dens`` and ``Gamma_es`` by ``dens``,
    and applies **nothing** to ``Q_em``/``Gamma_em``.
    """
    spec = next(s for s in run.params.get(0)["species"]
                if s["name"] == species)
    dens, temp = float(spec["dens"]), float(spec["temp"])
    g_es = np.average(fluxes["Gamma_es"], weights=jacobian, axis=(1, 2)) * dens
    q_es = np.average(fluxes["Q_es"], weights=jacobian, axis=(1, 2)) * temp * dens
    g_em = np.average(fluxes["Gamma_em"], weights=jacobian, axis=(1, 2))
    q_em = np.average(fluxes["Q_em"], weights=jacobian, axis=(1, 2))
    return g_es + g_em, q_es + q_em


def fortran_references(run):
    """``Qref`` and ``Gammaref`` from ``diag_3d.F90:915-916``, verbatim."""
    p = run.params.get(0)
    u = p["units"]
    tref, nref, mref = float(u["Tref"]), float(u["nref"]), float(u["mref"])
    bref, lref = float(u["Bref"]), float(u["Lref"])
    e = ELEMENTARY_CHARGE
    cref = np.sqrt(tref * 1e3 * e / (mref * PROTON_MASS))
    rhoref = mref * PROTON_MASS * cref / (e * bref)
    rhostar = rhoref / lref
    qref = (1e19 * nref) * (tref * e * 1000) * cref * rhostar ** 2
    gammaref = (1e19 * nref) * cref * rhostar ** 2
    return gammaref, qref


# ---------------------------------------------------------------------------

def _one_snapshot(run, species, index=0):
    """The four flux arrays of one snapshot, straight from the moment file."""
    reader = run.mom(species)
    wanted = ["Gamma_es", "Gamma_em", "Q_es", "Q_em"]
    slots = {v: reader.index_of(v) for v in wanted}
    _, arrays = next(iter(reader.stream_selected([index], variables=wanted)))
    return {v: np.asarray(arrays[slots[v]], dtype=float) for v in wanted}


class TestAgainstFortran:
    """genetools must equal diag_3d.F90 to round-off. This is the definition."""

    def test_flux_surface_average_is_the_fortran_one(self, run):
        from genetools.diagnostics import _gene3d as g3
        J = np.asarray(run.geometry[0]["Jacobian"], dtype=float)
        dat = _one_snapshot(run, run.species[0])["Q_es"]
        assert np.allclose(g3.flux_surface_average(dat, J),
                           fortran_flux_surface_average(dat, J), rtol=1e-12)

    @pytest.mark.parametrize("index", [0, 2])
    def test_profile_matches_the_fortran_formula(self, run, index):
        J = np.asarray(run.geometry[0]["Jacobian"], dtype=float)
        raw = run.fluxes2d.compute(None)
        times = np.asarray(raw["times"])
        for name in run.species:
            fluxes = _one_snapshot(run, name, index)
            g_ref, q_ref = fortran_profile(run, name, J, fluxes)
            per = raw["species"][name]
            g_ours = (np.asarray(per["Gamma_es"])[index]
                      + np.asarray(per["Gamma_em"])[index])
            q_ours = (np.asarray(per["Q_es"])[index]
                      + np.asarray(per["Q_em"])[index])
            assert np.allclose(g_ours, g_ref, rtol=1e-10), f"{name} Gamma"
            assert np.allclose(q_ours, q_ref, rtol=1e-10), f"{name} Q"

    def test_heat_flux_reference_equals_Qref(self, run):
        """Our Qgb is diag_3d.F90's Qref, in W/m^2."""
        _, qref = fortran_references(run)
        assert float(run.params.get(0)["units"]["Qgb"]) == pytest.approx(
            qref, rel=1e-10)

    def test_particle_reference_differs_from_Gammaref_by_1e19(self, run):
        """
        Ggb is in 1e19 m^-2 s^-1 where the Fortran's Gammaref is in m^-2 s^-1.

        A convention difference, not an error — but it is exactly 1e19, and the
        dataset labels Gamma_SI as ``1e19 m^-2 s^-1`` so the two agree once the
        unit is read. Pinned so the relationship cannot drift silently.
        """
        gammaref, _ = fortran_references(run)
        ggb = float(run.params.get(0)["units"]["Ggb"])
        assert ggb * 1e19 == pytest.approx(gammaref, rel=1e-10)


class TestAgainstTheGUI:
    """
    Where the reference GUI differs, and by exactly how much.

    Its electrostatic terms agree with the Fortran; its electromagnetic ones
    carry no species factor, so for a species with ``dens*temp != 1`` its heat
    flux is short by that factor on the EM part alone.
    """

    def test_electrostatic_terms_agree(self, run):
        J = np.asarray(run.geometry[0]["Jacobian"], dtype=float)
        raw = run.fluxes2d.compute(None)
        for name in run.species:
            fluxes = _one_snapshot(run, name, 0)
            spec = next(s for s in run.params.get(0)["species"]
                        if s["name"] == name)
            gui_q_es = (np.average(fluxes["Q_es"], weights=J, axis=(1, 2))
                        * float(spec["temp"]) * float(spec["dens"]))
            ours = np.asarray(raw["species"][name]["Q_es"])[0]
            assert np.allclose(ours, gui_q_es, rtol=1e-10), name

    def test_gui_electromagnetic_flux_is_short_by_the_species_factor(self, run):
        J = np.asarray(run.geometry[0]["Jacobian"], dtype=float)
        raw = run.fluxes2d.compute(None)
        for name in run.species:
            spec = next(s for s in run.params.get(0)["species"]
                        if s["name"] == name)
            factor = float(spec["dens"]) * float(spec["temp"])
            if factor == 1.0:
                continue                       # nothing to see for this species
            fluxes = _one_snapshot(run, name, 0)
            gui_q_em = np.average(fluxes["Q_em"], weights=J, axis=(1, 2))
            ours = np.asarray(raw["species"][name]["Q_em"])[0]
            assert np.allclose(ours / gui_q_em, factor, rtol=1e-10), name

    def test_gui_heat_reference_is_a_thousand_times_smaller(self, run):
        """
        The GUI's ``Qturb`` is kW/m^2 -- its own print says so.

        ``gyrobohm_SI`` builds ``Gamma_gB*Tref/1e3`` with Tref in joules, so a
        number taken from the GUI is 1000x smaller than the same flux in W/m^2.
        Comparing one against the other is the easiest way to be out by a
        thousand.
        """
        p = run.params.get(0)
        u = p["units"]
        e = ELEMENTARY_CHARGE
        nref = float(u["nref"]) * 1e19
        tref = float(u["Tref"]) * 1e3 * e
        mref = float(u["mref"]) * PROTON_MASS
        cref = np.sqrt(tref / mref)
        rhoref = mref * cref / (e * float(u["Bref"]))
        gamma_gb = nref * cref * rhoref ** 2 / float(u["Lref"]) ** 2
        gui_qturb = gamma_gb * tref / 1e3
        assert float(u["Qgb"]) / gui_qturb == pytest.approx(1000.0, rel=1e-10)
