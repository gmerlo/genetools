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
        for lim in [(0.25, 0.35), (0.4, 0.6), (0.6, 0.7)]:
            a = g3.radial_slice(local.coords[0]["x_o_a"], limits=lim)
            b = g3.radial_slice(glob.coords[0]["x_o_a"], limits=lim)
            assert a == b, lim
            assert len(range(*a.indices(8))) > 1, lim

    def test_a_constant_axis_cannot_serve_a_window(self):
        """
        Guards the guard: on a constant axis every window either misses it
        entirely (now an error) or collapses to the single repeated point.
        Either way it cannot answer "which surfaces are in [lo, hi]", which is
        why the axis is rebuilt rather than used.
        """
        const = np.full(8, X0)
        for lim in [(0.2, 0.3), (0.6, 0.8)]:
            with pytest.raises(ValueError, match="outside this axis"):
                g3.radial_slice(const, limits=lim)
        assert g3.radial_slice(const, limits=(X0, X0)) == slice(0, 1)

    def test_no_warning_when_the_axis_is_genuine(self, tmp_path, recwarn):
        _loaded(tmp_path, True)
        assert not [w for w in recwarn
                    if "constant xval_a" in str(w.message)]


class TestWindowGuards:
    """
    A window the grid cannot serve must say so.

    Nearest-point selection answers *something* for any bound: ask for a range
    the run does not cover and `argmin` pins both ends to the same edge index,
    so the caller gets one surface from the far side of the box with no
    indication the window was not honoured. Downstream that is
    indistinguishable from having got what was asked for.
    """

    GRID = np.round(np.linspace(0.20, 0.80, 13), 4)

    @pytest.mark.parametrize("lim", [(0.90, 1.00), (0.00, 0.10), (1.0, 2.0)])
    def test_a_window_off_the_grid_raises(self, lim):
        with pytest.raises(ValueError, match="outside this axis"):
            g3.radial_slice(self.GRID, limits=lim)

    def test_the_error_names_the_range_that_does_exist(self):
        with pytest.raises(ValueError, match=r"\[0.2, 0.8\]"):
            g3.radial_slice(self.GRID, limits=(0.9, 1.0))

    def test_a_partly_overlapping_window_is_clipped_and_reported(self):
        with pytest.warns(RuntimeWarning, match="clipped to the overlap"):
            sl = g3.radial_slice(self.GRID, limits=(0.75, 0.95))
        assert self.GRID[sl][0] == pytest.approx(0.75)
        assert self.GRID[sl][-1] == pytest.approx(0.80)

    def test_a_window_inside_the_grid_is_silent(self, recwarn):
        sl = g3.radial_slice(self.GRID, limits=(0.35, 0.50))
        assert self.GRID[sl].size == 4
        assert not recwarn.list

    def test_half_a_cell_of_slack_is_allowed(self, recwarn):
        """
        Nearest-point selection legitimately reaches half a cell past each end,
        so asking for that much is the window landing on the edge point, not
        overrunning the grid.
        """
        d = self.GRID[1] - self.GRID[0]
        sl = g3.radial_slice(self.GRID, limits=(0.20 - 0.4 * d, 0.80 + 0.4 * d))
        assert self.GRID[sl].size == self.GRID.size
        assert not recwarn.list

    def test_a_length_mismatch_raises(self):
        """
        The radial coordinate and the radial data axis are the same grid, so a
        disagreement means a slice from one would be applied to the other --
        silently empty or silently short, since numpy slices never raise.
        """
        with pytest.raises(ValueError, match="data axis has 8"):
            g3.radial_slice(self.GRID, limits=(0.3, 0.5), nx=8)

    def test_a_matching_length_is_accepted(self):
        sl = g3.radial_slice(self.GRID, limits=(0.3, 0.5), nx=self.GRID.size)
        assert self.GRID[sl].size == 5

    def test_the_old_behaviour_was_a_silent_wrong_answer(self):
        """Guards the guard: states what was being returned before."""
        arr = self.GRID
        lo = hi = 0.95
        i0 = int(np.argmin(np.abs(arr - lo)))
        i1 = int(np.argmin(np.abs(arr - hi)))
        assert slice(i0, i1 + 1) == slice(12, 13)
        assert arr[12] == pytest.approx(0.80), "0.80 returned for a 0.95 request"
