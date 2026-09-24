# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Unit tests for the geometry-independent parts of `Contours`.

The end-to-end GENE-3D behaviour lives in ``test_gene3d_diagnostics.py``; this
file covers the pieces that make one interface serve every geometry — the
transform onto the requested axes, the axis construction, and the option
vocabulary that replaced the old spectral-only arguments.

No fixture in the repo writes a spectral run *with field output*, so the
spectral paths are exercised by fixing the geometry kind on a detached instance
and feeding it arrays of the shape that geometry stores. That is enough: every
geometry difference in this module is carried by ``_x_is_spectral`` and
``_y_is_spectral``.
"""

import numpy as np
import pytest

from genetools.diagnostics.contours import (
    Contours, _RETIRED, _ALL, _inverse_y, _inverse_x,
)


class _Fixed(Contours):
    """A detached `Contours` pinned to one geometry kind."""

    def __init__(self, kind):
        super().__init__(None)
        self._kind = kind

    @property
    def geometry_kind(self):
        return self._kind

    @property
    def is_3d(self):
        return self._kind == "xy_global"


def _stored(kind, nx=8, nj=4, nz=6, seed=0):
    """One snapshot in the layout *kind* stores it in."""
    rng = np.random.default_rng(seed)
    if kind == "xy_global":                       # GENE-3D: real x and y
        return rng.standard_normal((nx, 2 * nj, nz)).astype(np.float32)
    return (rng.standard_normal((nx, nj, nz))
            + 1j * rng.standard_normal((nx, nj, nz))).astype(np.complex64)


# ---------------------------------------------------------------------------
# The option vocabulary
# ---------------------------------------------------------------------------

class TestRetiredOptions:
    """The old spectral-only arguments name their replacement when used."""

    @pytest.mark.parametrize("old", sorted(_RETIRED))
    def test_retired_option_names_its_replacement(self, old):
        diag = _Fixed("flux_tube")
        with pytest.raises(TypeError) as exc:
            diag._select({old: 0})
        assert old in str(exc.value)
        assert "unified" in str(exc.value)

    def test_unknown_option_lists_what_is_accepted(self):
        with pytest.raises(TypeError, match="unknown contour option"):
            _Fixed("flux_tube")._select({"nonsense": 1})


class TestReductions:
    def test_all_expands(self):
        assert Contours._reductions("all") == _ALL

    def test_single_string_is_wrapped(self):
        assert Contours._reductions("xz") == ("xz",)

    def test_unknown_reduction_raises(self):
        with pytest.raises(ValueError, match="unknown reduction"):
            Contours._reductions(("xy", "qq"))

    def test_none_is_empty(self):
        assert Contours._reductions(None) == ()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class TestInverseY:
    """
    GENE keeps ``nky0 = ny0 / 2`` modes and drops the Nyquist bin.

    ``discretization.F90`` sets ``ly0da = 2 * nky0`` (``3 * nky0`` under the 3/2
    dealiasing rule), so the real-space axis has ``2 * nky`` points — not the
    ``2 * (nky - 1)`` the old spectral path used, which was two short.
    """

    def test_length_is_twice_nky(self):
        arr = _stored("flux_tube", nx=3, nj=5, nz=2)
        assert _inverse_y(arr).shape[1] == 10

    def test_matches_nyquist_zeroed_reconstruction(self):
        rng = np.random.default_rng(3)
        ny, nky = 16, 8
        real = rng.standard_normal((2, ny, 3))
        full = np.fft.rfft(real, axis=1)               # ny//2 + 1 = 9 modes
        stored = full[:, :nky, :]                      # what GENE writes
        zeroed = full.copy()
        zeroed[:, nky:, :] = 0.0
        expected = np.fft.irfft(zeroed, n=ny, axis=1) * ny
        assert np.allclose(_inverse_y(stored), expected)

    def test_round_trip_with_gene_normalisation(self):
        rng = np.random.default_rng(4)
        ny = 12
        real = rng.standard_normal((2, ny, 3))
        stored = np.fft.rfft(real, axis=1)[:, :ny // 2, :] / ny
        # The dropped Nyquist bin is the only thing lost.
        back = _inverse_y(stored)
        assert back.shape[1] == ny
        assert np.allclose(back.mean(axis=1), real.mean(axis=1))

    def test_inverse_x_preserves_length(self):
        arr = _stored("flux_tube", nx=7, nj=3, nz=2)
        assert _inverse_x(arr).shape[0] == 7


class TestToAxes:
    """Every geometry lands on the axes the selection names."""

    def _sel(self, diag, **kw):
        return diag._select(kw)

    @pytest.mark.parametrize("kind", ["flux_tube", "x_global", "xy_global"])
    def test_default_is_real_space_everywhere(self, kind):
        diag = _Fixed(kind)
        arr = _stored(kind)
        sel = self._sel(diag)
        out = diag._to_axes(sel, arr, shift={})
        assert not np.iscomplexobj(out)
        assert out.shape == diag._transformed_shape(sel, arr.shape)
        assert out.shape[1] == 8          # 2 * nj, or the stored real y

    def test_flux_tube_inverts_x_before_y(self):
        """
        The y inverse is only valid once x is real.

        GENE stores ky >= 0 because the real-space field is real, which gives
        ``f(-kx, -ky) = conj(f(kx, ky))``; at fixed kx the ky axis is not
        Hermitian, so inverting y first would be meaningless.
        """
        diag = _Fixed("flux_tube")
        arr = _stored("flux_tube")
        out = diag._to_axes(self._sel(diag), arr, shift={})
        assert np.allclose(out, _inverse_y(_inverse_x(arr)).real)
        assert not np.allclose(out, _inverse_x(_inverse_y(arr)).real)

    def test_flux_tube_kx_with_real_y_goes_out_and_back(self):
        """kx and real y is reachable only through real space."""
        diag = _Fixed("flux_tube")
        arr = _stored("flux_tube")
        sel = self._sel(diag, x_fourier=True)
        out = diag._to_axes(sel, arr, shift={})
        from genetools.diagnostics import _gene3d as g3
        assert np.allclose(out, np.abs(g3.to_kx(_inverse_y(_inverse_x(arr)))))

    def test_x_global_inverts_only_y(self):
        diag = _Fixed("x_global")
        arr = _stored("x_global")
        out = diag._to_axes(self._sel(diag), arr, shift={})
        assert np.allclose(out, _inverse_y(arr).real)

    def test_spectral_view_of_a_flux_tube_is_the_stored_magnitude(self):
        """Nothing is transformed when the view matches what is stored."""
        diag = _Fixed("flux_tube")
        arr = _stored("flux_tube")
        sel = self._sel(diag, x_fourier=True, y_fourier=True)
        out = diag._to_axes(sel, arr, shift={})
        assert np.allclose(out, np.abs(arr))

    def test_magnitude_is_taken_once_not_per_axis(self):
        """
        ``x_fourier`` and ``y_fourier`` together mean ``|FFT_xy(f)|``.

        The old GENE-3D path applied ``|.|`` after each axis, computing
        ``FFT_y(|FFT_x(f)|)`` — a different quantity.
        """
        diag = _Fixed("xy_global")
        arr = _stored("xy_global")
        sel = self._sel(diag, x_fourier=True, y_fourier=True)
        out = diag._to_axes(sel, arr, shift={})
        from genetools.diagnostics import _gene3d as g3
        assert np.allclose(out, np.abs(g3.to_ky(g3.to_kx(arr))))
        assert not np.allclose(out, np.abs(g3.to_ky(np.abs(g3.to_kx(arr)))))

    def test_square_is_the_squared_magnitude(self):
        diag = _Fixed("x_global")
        arr = _stored("x_global")
        plain = diag._to_axes(self._sel(diag), arr, shift={})
        squared = diag._to_axes(self._sel(diag, square=True), arr, shift={})
        assert np.allclose(squared, np.abs(plain) ** 2)

    def test_shift_moves_the_data_with_its_axis(self):
        diag = _Fixed("xy_global")
        arr = _stored("xy_global")
        sel = self._sel(diag, x_fourier=True)
        unshifted = diag._to_axes(sel, arr, shift={"x": False})
        shifted = diag._to_axes(sel, arr, shift={"x": True})
        assert np.allclose(shifted, np.fft.fftshift(unshifted, axes=0))


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

class TestFftOrdered:
    def test_one_sided_ky_is_not_shifted(self):
        assert not Contours._fft_ordered(np.array([0.0, 0.1, 0.2, 0.3]))

    def test_fftfreq_order_is_shifted(self):
        assert Contours._fft_ordered(np.fft.fftfreq(8, d=0.5))

    def test_too_short_is_not_shifted(self):
        assert not Contours._fft_ordered(np.array([0.0]))


class TestBoxGrid:
    def test_zero_based_with_length_from_the_smallest_mode(self):
        diag = _Fixed("flux_tube")
        ky = np.array([0.0, 0.5, 1.0, 1.5])
        grid = diag._box_grid(ky, 8)
        assert grid[0] == 0.0
        assert grid.size == 8
        # L = 2 pi / 0.5, sampled on 8 points
        assert np.isclose(grid[1] - grid[0], (2 * np.pi / 0.5) / 8)

    def test_no_positive_mode_falls_back_to_indices(self):
        diag = _Fixed("flux_tube")
        assert np.allclose(diag._box_grid(np.array([0.0]), 3), [0, 1, 2])


class TestAxes:
    """The displayed axis, its values and the label that says what it means."""

    def _coord(self, nx=8, nky=4, nz=6):
        return {"x_o_a": np.linspace(0.4, 0.8, nx),
                "kx": np.fft.fftfreq(nx, d=0.5) * 2 * np.pi,
                "y": np.linspace(0.0, 10.0, 2 * nky, endpoint=False),
                "ky": 0.5 * np.arange(nky),
                "z": np.linspace(-np.pi, np.pi, nz)}

    def test_flux_tube_real_x_is_a_box_coordinate(self):
        diag = _Fixed("flux_tube")
        values, labels, shift = diag._axes(diag._select({}), self._coord(),
                                           (8, 8, 6))
        assert labels["x"] == "x_box"
        assert values["x"][0] == 0.0
        assert not shift["x"]

    def test_global_real_x_is_radial(self):
        diag = _Fixed("x_global")
        values, labels, _ = diag._axes(diag._select({}), self._coord(),
                                       (8, 8, 6))
        assert labels["x"] == "x"
        assert np.isclose(values["x"][0], 0.4)

    def test_fourier_x_is_shifted_and_labelled_kx(self):
        diag = _Fixed("x_global")
        sel = diag._select({"x_fourier": True})
        values, labels, shift = diag._axes(sel, self._coord(), (8, 8, 6))
        assert labels["x"] == "kx" and shift["x"]
        assert values["x"][0] < 0                      # shifted: most negative

    def test_one_sided_ky_axis_is_not_shifted(self):
        diag = _Fixed("flux_tube")
        sel = diag._select({"y_fourier": True})
        values, labels, shift = diag._axes(sel, self._coord(), (8, 4, 6))
        assert labels["y"] == "ky" and not shift["y"]
        assert values["y"][0] == 0.0

    def test_reconstructed_y_has_the_transformed_length(self):
        diag = _Fixed("flux_tube")
        sel = diag._select({})
        values, labels, _ = diag._axes(sel, self._coord(), (8, 8, 6))
        assert labels["y"] == "y"
        assert values["y"].size == 8


class TestLimits:
    """y and z default to a cut; x defaults to its whole range."""

    def _values(self):
        return {"x": np.linspace(0.4, 0.8, 5),
                "y": np.linspace(-2.0, 2.0, 5),
                "z": np.linspace(-np.pi, np.pi, 5)}

    def test_y_and_z_default_to_a_cut_at_zero(self):
        diag = _Fixed("x_global")
        limits = diag._limits(diag._select({}), self._values())
        assert limits["y"] == (0.0, 0.0)
        assert limits["z"][0] == limits["z"][1] == 0.0

    def test_x_defaults_to_the_whole_range(self):
        diag = _Fixed("x_global")
        assert diag._limits(diag._select({}), self._values())["x"] is None

    def test_explicit_limits_win(self):
        diag = _Fixed("x_global")
        sel = diag._select({"zlim": (-1.0, 1.0)})
        assert diag._limits(sel, self._values())["z"] == (-1.0, 1.0)

    def test_only_a_cut_is_reported_as_fixed(self):
        fixed = Contours._fixed_values({"x": None, "y": (0.0, 0.0),
                                        "z": (-1.0, 1.0)})
        assert fixed == {"y": 0.0}
