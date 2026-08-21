# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the pure compute helpers of the vexmax diagnostic."""

import numpy as np
import pytest

from genetools.diagnostics.vexmax import ddx_4th, y_reconstruct, analyze_snapshot


class TestDdx4th:
    def test_linear_exact(self):
        x = np.linspace(0.0, 1.0, 32)
        dx = x[1] - x[0]
        f = 3.0 * x + 1.0
        d = ddx_4th(f, dx)
        assert np.allclose(d, 3.0)

    def test_quartic_interior_exact(self):
        # 4th-order centered stencil differentiates x^4 exactly in the interior
        x = np.linspace(0.0, 1.0, 64)
        dx = x[1] - x[0]
        d = ddx_4th(x**4, dx)
        assert np.allclose(d[2:-2], 4.0 * x[2:-2] ** 3, atol=1e-10)

    def test_axis_argument(self):
        x = np.linspace(0.0, 1.0, 32)
        dx = x[1] - x[0]
        f = np.tile(x**2, (5, 1))            # x on axis 1
        d = ddx_4th(f, dx, axis=1)
        assert np.allclose(d[:, 2:-2], 2.0 * x[None, 2:-2], atol=1e-10)

    def test_complex_input(self):
        x = np.linspace(0.0, 1.0, 32)
        dx = x[1] - x[0]
        f = (1.0 + 2.0j) * x
        assert np.allclose(ddx_4th(f, dx), 1.0 + 2.0j)


class TestYReconstruct:
    def test_ky0_only(self):
        vhat = np.zeros(8, dtype=complex)
        vhat[0] = 2.5
        v = y_reconstruct(vhat, 64)
        assert np.allclose(v, 2.5)

    def test_single_finite_ky_amplitude(self):
        # one ky>0 mode of amplitude A gives a real field 2A cos(...): max = 2A
        vhat = np.zeros(8, dtype=complex)
        vhat[3] = 0.7 * np.exp(0.3j)
        v = y_reconstruct(vhat, 256)
        assert np.max(np.abs(v)) == pytest.approx(1.4, rel=1e-3)

    def test_ny_too_small_raises(self):
        with pytest.raises(ValueError):
            y_reconstruct(np.zeros(8, dtype=complex), 8)


def _make_inputs(nx=48, nky=6, nz=4, lx=10.0):
    params = {
        "box": {"nx0": nx, "nky0": nky, "nz0": nz},
        "general": {"x_local": False},
        "nonlocal_x": {"l_buffer_size": 0.1, "u_buffer_size": 0.1},
        "info": {"n_fields": 1},
    }
    dx = lx / (nx - 1)
    x = np.linspace(0.0, lx, nx)
    coord = {"x": x, "dx": dx, "ky": 0.1 * np.arange(nky)}
    geom = {"metric": {"C_xy": 1.0}}
    return params, coord, geom, x


class TestAnalyzeSnapshot:
    def test_single_column_located(self):
        params, coord, geom, x = _make_inputs()
        nx, nky, nz = 48, 6, 4
        phi = np.zeros((nx, nky, nz), dtype=complex)
        # smooth radial structure in ky column j=2, z plane 1
        phi[:, 2, 1] = np.sin(2 * np.pi * x / x[-1])
        res = analyze_snapshot(phi, coord, geom, params)
        assert res["jstar"] == 2
        assert res["iz"] == 1
        # d/dx sin(2 pi x/L) peaks at the edges -> located in buffer here
        assert res["in_buffer"]
        # max of |dphi/dx| = 2 pi/L; real-space reconstruction doubles ky>0
        expect = 2 * (2 * np.pi / x[-1])
        assert res["vemax"] == pytest.approx(expect, rel=5e-2)
        # cumulative curve: zero up to K=1, full from K=2 on
        k = res["klist"]
        cum = res["vemax_cum"]
        assert cum[np.searchsorted(k, 1)] == pytest.approx(0.0, abs=1e-12)
        assert cum[np.searchsorted(k, 2)] == pytest.approx(res["vemax"], rel=1e-6)

    def test_smooth_structure_no_smoothing_flag(self):
        params, coord, geom, x = _make_inputs()
        phi = np.zeros((48, 6, 4), dtype=complex)
        phi[:, 1, 0] = np.sin(2 * np.pi * x / x[-1])
        res = analyze_snapshot(phi, coord, geom, params)
        assert not res["smoothing_helps"]

    def test_grid_scale_structure_flags_smoothing(self):
        params, coord, geom, x = _make_inputs()
        nx = 48
        phi = np.zeros((nx, 6, 4), dtype=complex)
        phi[:, 1, 0] = np.cos(np.pi * np.arange(nx))   # +1/-1 zigzag
        res = analyze_snapshot(phi, coord, geom, params)
        assert res["smoothing_helps"]

    def test_cxy_array_scaling(self):
        params, coord, geom, x = _make_inputs()
        phi = np.zeros((48, 6, 4), dtype=complex)
        phi[:, 1, 0] = np.sin(2 * np.pi * x / x[-1])
        r1 = analyze_snapshot(phi, coord, geom, params)
        geom2 = {"metric": {"C_xy": 2.0 * np.ones(48)}}
        r2 = analyze_snapshot(phi, coord, geom2, params)
        assert r2["vemax"] == pytest.approx(0.5 * r1["vemax"], rel=1e-12)

    def test_ky0_counts_once(self):
        params, coord, geom, x = _make_inputs()
        phi = np.zeros((48, 6, 4), dtype=complex)
        phi[:, 0, 0] = np.sin(2 * np.pi * x / x[-1])   # zonal only
        res = analyze_snapshot(phi, coord, geom, params)
        expect = 2 * np.pi / x[-1]                     # no factor 2 for ky=0
        assert res["vemax"] == pytest.approx(expect, rel=5e-2)
