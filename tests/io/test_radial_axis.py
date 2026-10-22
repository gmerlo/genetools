# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
The radial axis of a GENE-3D run, and the one case where the file lies.

``coordinates.F90:95-99`` writes ``xval_a = x0`` -- the scalar, broadcast over
every radial point -- when ``radial_dependence`` is ``.false.``. Taken at face
value that is not a coordinate: ``argmin(|const - bound|)`` is 0 for any bound,
so every radial window collapses to the innermost surface and every plot
against ``x/a`` stacks on one point. Nothing caught it because the fixture
wrote ``xval*rhostar`` whatever the flag said.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gene3d_fixture import make_gene3d_run          # noqa: E402

from genetools import Run                            # noqa: E402
from genetools.diagnostics import _gene3d as g3      # noqa: E402

X0 = 0.5


def _run(tmp_path, radial_dependence):
    folder = tmp_path / f"rd_{radial_dependence}"
    make_gene3d_run(folder, nx0=8, ny0=8, nz0=4, n_times=2, n_fields=1,
                    species=("ions",), x0=X0,
                    radial_dependence=radial_dependence)
    return folder


def _loaded(tmp_path, radial_dependence):
    """A Run with its coordinates realised: `Run.coords` is lazy, so the
    warning fires on first access, not on construction."""
    run = Run(str(_run(tmp_path, radial_dependence)))
    run.coords                                   # noqa: B018 - force the load
    return run


class TestFixtureMatchesTheCode:
    """The fixture has to write the degenerate axis, or nothing is tested."""

    def test_radially_local_writes_a_constant_xval_a(self, tmp_path):
        import h5py
        folder = _run(tmp_path, False)
        with h5py.File(folder / "coord.dat.h5", "r") as f:
            xa = f["/coord/xval_a"][...]
        assert np.ptp(xa) == 0.0
        assert xa[0] == pytest.approx(X0)

    def test_radially_dependent_writes_a_varying_one(self, tmp_path):
        import h5py
        folder = _run(tmp_path, True)
        with h5py.File(folder / "coord.dat.h5", "r") as f:
            xa = f["/coord/xval_a"][...]
        assert np.ptp(xa) > 0


class TestDegenerateAxisIsReconstructed:

    def test_the_axis_varies_either_way(self, tmp_path):
        with pytest.warns(RuntimeWarning, match="constant xval_a"):
            local = _loaded(tmp_path, False)
        glob = _loaded(tmp_path, True)
        x_local = np.asarray(local.coords[0]["x_o_a"])
        x_glob = np.asarray(glob.coords[0]["x_o_a"])
        assert np.ptp(x_local) > 0
        np.testing.assert_allclose(x_local, x_glob, rtol=1e-12)

    def test_it_is_xval_times_rhostar(self, tmp_path):
        with pytest.warns(RuntimeWarning):
            run = _loaded(tmp_path, False)
        coord = run.coords[0]
        rhostar = float(run.params.get(0)["geometry"]["rhostar"])
        np.testing.assert_allclose(np.asarray(coord["x_o_a"]),
                                   np.asarray(coord["x"]) * rhostar,
                                   rtol=1e-12)

    def test_windows_select_the_same_surfaces_either_way(self, tmp_path):
        with pytest.warns(RuntimeWarning):
            local = _loaded(tmp_path, False)
        glob = _loaded(tmp_path, True)
        for lim in [(0.2, 0.3), (0.4, 0.6), (0.6, 0.8)]:
            a = g3.radial_slice(local.coords[0]["x_o_a"], limits=lim)
            b = g3.radial_slice(glob.coords[0]["x_o_a"], limits=lim)
            assert a == b, lim
            assert len(range(*a.indices(8))) > 1, lim

    def test_a_constant_axis_would_collapse_every_window(self):
        """
        Guards the guard: states plainly what the unfixed behaviour was, so the
        test above is known to be testing something.
        """
        const = np.full(8, X0)
        for lim in [(0.2, 0.3), (0.4, 0.6), (0.6, 0.8)]:
            assert g3.radial_slice(const, limits=lim) == slice(0, 1)

    def test_no_warning_when_the_axis_is_genuine(self, tmp_path, recwarn):
        _loaded(tmp_path, True)
        assert not [w for w in recwarn
                    if "constant xval_a" in str(w.message)]
