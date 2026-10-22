# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Tests for the nonlinear frequency diagnostic.

The load-bearing one is the sign convention: a spectrum that is mirrored in
omega looks perfectly healthy and reports every propagation direction backwards.
It is pinned here by building a wave from GENE's own ansatz,
``f ~ exp(i(k_y y - omega t))``, and checking the peak comes back at ``+omega``.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gene3d_fixture import make_gene3d_run          # noqa: E402

from genetools import Run                            # noqa: E402
from genetools.diagnostics import _gene3d as g3      # noqa: E402
from genetools.diagnostics.frequency import (        # noqa: E402
    Frequency, welch, resample_uniform, spectral_peaks, spectral_moments,
)


# ---------------------------------------------------------------------------
# The sign convention
# ---------------------------------------------------------------------------

class TestSignConvention:
    """
    GENE's ansatz is ``exp(i(k_y y - omega t))``.

    So the ``+k_y`` component of a wave travelling in ``+y`` varies as
    ``exp(-i omega t)`` and must appear at ``+omega``. numpy's forward FFT
    carries the same ``exp(-i omega t)`` kernel, which is why the frequency
    axis has to be the *negated* fftfreq.
    """

    def _wave(self, omega0, ny=32, nt=512, mode=3):
        ly, dt = 10.0, 0.05
        y = np.arange(ny) * (ly / ny)
        t = np.arange(nt) * dt
        k0 = 2 * np.pi * mode / ly
        phi = np.cos(k0 * y[None, :] - omega0 * t[:, None])
        # to_ky is an FFT along the binormal axis with numpy's kernel.
        hat = np.fft.fft(phi, axis=1) / ny
        return hat[:, mode][:, None], dt          # the +k0 component

    def _peak(self, series, dt):
        omega, power, _ = welch(series, dt, nperseg=series.shape[0] // 4,
                               noverlap=series.shape[0] // 8)
        return float(omega[int(np.argmax(power[:, 0]))])

    def test_travelling_wave_lands_at_positive_omega(self):
        series, dt = self._wave(omega0=4.0)
        assert self._peak(series, dt) == pytest.approx(4.0, abs=0.25)

    def test_reversing_the_wave_reverses_the_frequency(self):
        series, dt = self._wave(omega0=-4.0)
        assert self._peak(series, dt) == pytest.approx(-4.0, abs=0.25)

    def test_the_two_directions_are_not_symmetric(self):
        """A complex series has a one-sided spectrum; both halves are kept."""
        series, dt = self._wave(omega0=4.0)
        omega, power, _ = welch(series, dt, nperseg=128, noverlap=64)
        pos = power[omega > 0].max()
        neg = power[omega < 0].max()
        assert pos > 20 * neg


# ---------------------------------------------------------------------------
# Estimator pieces
# ---------------------------------------------------------------------------

class TestWelch:
    def test_averages_over_several_windows(self):
        data = np.random.default_rng(0).standard_normal((256, 2)) + 0j
        _, _, n_win = welch(data, 0.1, nperseg=64, noverlap=32)
        assert n_win == 7                       # (256 - 64) // 32 + 1

    def test_frequency_axis_is_ascending_and_spans_nyquist(self):
        data = np.zeros((128, 1), dtype=complex)
        dt = 0.25
        omega, _, _ = welch(data, dt, nperseg=64, noverlap=32)
        assert np.all(np.diff(omega) > 0)
        assert np.isclose(np.abs(omega).max(), np.pi / dt, rtol=0.05)

    def test_detrend_removes_the_zero_frequency_spike(self):
        n = 256
        data = (np.ones((n, 1)) * 5.0).astype(complex)      # pure DC
        omega, plain, _ = welch(data, 0.1, 64, 32, detrend=False)
        _, flat, _ = welch(data, 0.1, 64, 32, detrend=True)
        i0 = int(np.argmin(np.abs(omega)))
        assert plain[i0, 0] > 0
        assert flat[i0, 0] == pytest.approx(0.0, abs=1e-20)

    def test_window_longer_than_record_is_clamped(self):
        data = np.zeros((16, 1), dtype=complex)
        _, power, n_win = welch(data, 0.1, nperseg=999, noverlap=0)
        assert n_win == 1 and power.shape[0] == 16


class TestResample:
    def test_uniform_input_is_unchanged(self):
        t = np.linspace(0.0, 4.0, 9)
        data = np.arange(9.0)[:, None]
        t_new, out, ratio = resample_uniform(t, data)
        assert np.allclose(t_new, t)
        assert np.allclose(out, data)
        assert ratio == pytest.approx(1.0)

    def test_uneven_input_is_interpolated_linearly(self):
        t = np.array([0.0, 1.0, 5.0])
        data = np.array([0.0, 1.0, 5.0])[:, None]           # f(t) = t
        t_new, out, ratio = resample_uniform(t, data)
        assert np.allclose(out[:, 0], t_new)                # exact for a ramp
        assert ratio == pytest.approx(4.0)

    def test_non_monotonic_times_raise(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            resample_uniform(np.array([0.0, 2.0, 1.0]), np.zeros((3, 1)))

    def test_single_snapshot_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            resample_uniform(np.array([1.0]), np.zeros((1, 1)))


class TestPeaksAndMoments:
    def _spectrum(self, omega, centres, heights):
        return sum(h * np.exp(-((omega - c) / 0.3) ** 2)
                   for c, h in zip(centres, heights))

    def test_peaks_come_back_strongest_first(self):
        omega = np.linspace(-5, 5, 401)
        power = self._spectrum(omega, [1.0, -2.0], [1.0, 0.6])
        peaks = spectral_peaks(omega, power, 2)
        assert peaks[0] == pytest.approx(1.0, abs=0.05)
        assert peaks[1] == pytest.approx(-2.0, abs=0.05)

    def test_peaks_below_the_floor_are_dropped(self):
        omega = np.linspace(-5, 5, 401)
        power = self._spectrum(omega, [1.0, -2.0], [1.0, 0.01])
        assert np.isnan(spectral_peaks(omega, power, 2)[1])

    def test_missing_peaks_are_nan_not_short(self):
        omega = np.linspace(-5, 5, 401)
        assert spectral_peaks(omega, np.zeros_like(omega), 3).shape == (3,)

    def test_centroid_and_width_of_a_gaussian_band(self):
        omega = np.linspace(-10, 10, 2001)
        sigma = 0.8
        power = np.exp(-((omega - 2.0) / (np.sqrt(2) * sigma)) ** 2)
        centroid, width = spectral_moments(omega, power)
        assert centroid == pytest.approx(2.0, abs=0.01)
        assert width == pytest.approx(sigma, rel=0.02)

    def test_empty_spectrum_is_nan(self):
        omega = np.linspace(-1, 1, 11)
        assert np.isnan(spectral_moments(omega, np.zeros_like(omega))[0])


# ---------------------------------------------------------------------------
# End to end on a GENE-3D run
# ---------------------------------------------------------------------------

def _write_travelling_phi(folder, omega0, mode=2):
    """Overwrite the fixture's phi with a wave of known frequency."""
    path = Path(folder) / "field.dat.h5"
    with h5py.File(path, "r+") as f:
        grp = f["field"]
        times = np.asarray(grp["time"][...], dtype=float)
        keys = sorted(grp["phi"].keys())
        ny = grp["phi"][keys[0]].shape[1]
        ly = 80.0                                   # fixture default
        y = np.arange(ny) * (ly / ny)
        k0 = 2 * np.pi * mode / ly
        for it, key in enumerate(keys):
            dset = grp["phi"][key]
            nz, _, nx = dset.shape
            wave = np.cos(k0 * y - omega0 * times[it])
            dset[...] = np.broadcast_to(wave[None, :, None],
                                        (nz, ny, nx)).astype(dset.dtype)
    return times


@pytest.fixture
def wave_run(tmp_path):
    """A GENE-3D run whose phi is a single wave travelling in +y."""
    folder = tmp_path / "wave"
    make_gene3d_run(folder, nx0=4, ny0=16, nz0=4, n_times=64, n_fields=1,
                    species=("ions",))
    _write_travelling_phi(folder, omega0=0.05)
    return Run(str(folder))


class TestEndToEnd:
    def test_reports_the_frequency_of_a_known_wave(self, wave_run):
        """
        The whole chain — to_ky, the time transform, the reductions.

        The fixture writes times ``0, 10, 20, ...``, so the Nyquist frequency
        is ``pi/10``; the wave at 0.05 sits well inside it.
        """
        ds = wave_run.frequency(window_frac=0.5).dataset()
        ky = np.asarray(ds["ky"])
        imode = int(np.argmin(np.abs(ky - ky[ky > 0].min() * 2)))
        peak = float(np.asarray(ds["phi_peak"])[imode, 0])
        assert peak == pytest.approx(0.05, abs=0.02)

    def test_dataset_carries_the_diagnostics_of_its_own_resolution(self,
                                                                  wave_run):
        ds = wave_run.frequency().dataset()
        assert ds.attrs["omega_nyquist"] == pytest.approx(np.pi / 10.0)
        assert ds.attrs["n_windows"] >= 1
        assert ds.attrs["spacing_ratio"] == pytest.approx(1.0)
        assert ds.attrs["omega_resolution"] > 0

    def test_products_have_the_expected_dims(self, wave_run):
        ds = wave_run.frequency().dataset()
        assert ds["phi_power"].dims == ("ky", "omega")
        assert ds["phi_peak"].dims == ("ky", "peak")
        assert ds["phi_centroid"].dims == ("ky",)
        assert ds["phi_width"].dims == ("ky",)

    def test_z_is_cut_not_averaged_by_default(self, wave_run):
        """One z plane is kept, which is what bounds the held time series."""
        raw = wave_run.frequency().compute()
        zsl = raw["zslice"]
        z = np.asarray(wave_run.coords[0]["z"])
        assert len(range(*zsl.indices(z.size))) == 1

    def test_zlim_widens_the_kept_range(self, wave_run):
        z = np.asarray(wave_run.coords[0]["z"])
        raw = wave_run.frequency(zlim=(z[0], z[-1])).compute()
        assert len(range(*raw["zslice"].indices(z.size))) == z.size

    def test_aliasing_is_warned_about(self, tmp_path):
        """A wave near Nyquist must not be reported silently."""
        folder = tmp_path / "fast"
        make_gene3d_run(folder, nx0=4, ny0=16, nz0=4, n_times=64, n_fields=1,
                        species=("ions",))
        _write_travelling_phi(folder, omega0=0.30)     # Nyquist is pi/10
        with pytest.warns(RuntimeWarning, match="aliased"):
            Run(str(folder)).frequency().compute()

    def test_refuses_a_non_3d_run(self, tmp_path):
        from gene_fixture import make_xglobal_run
        folder = make_xglobal_run(tmp_path / "xg")
        with pytest.raises(Exception):
            Run(str(folder)).frequency()

    def test_a_single_snapshot_is_refused(self, tmp_path):
        folder = tmp_path / "one"
        make_gene3d_run(folder, nx0=4, ny0=8, nz0=4, n_times=1, n_fields=1,
                        species=("ions",))
        with pytest.raises(ValueError, match="at least two"):
            Run(str(folder)).frequency().compute()


class TestRadialWindow:
    """
    Which surfaces the (ky, omega) map is an average over.

    The spectrum is a Jacobian-weighted average over x, so the window is part
    of the answer: two spectra averaged over different ranges are not
    comparable, however alike they look. Hence both that the window can be
    chosen and that the choice is recorded.
    """

    @staticmethod
    def _kept(run, raw):
        x = np.asarray(run.coords[0]["x_o_a"])
        return x[raw["xslice"]]

    def test_whole_domain_by_default(self, wave_run):
        raw = wave_run.frequency().compute()
        x = np.asarray(wave_run.coords[0]["x_o_a"])
        assert self._kept(wave_run, raw).size == x.size

    def test_xlim_restricts_the_average(self, wave_run):
        x = np.asarray(wave_run.coords[0]["x_o_a"])
        lo, hi = float(x[1]), float(x[-2])
        raw = wave_run.frequency(xlim=(lo, hi)).compute()
        kept = self._kept(wave_run, raw)
        assert kept[0] == pytest.approx(lo)
        assert kept[-1] == pytest.approx(hi)
        assert kept.size == x.size - 2

    @pytest.mark.filterwarnings("ignore:.*aliased.*:RuntimeWarning")
    def test_buffer_frac_trims_each_end(self, tmp_path):
        folder = tmp_path / "wide"
        make_gene3d_run(folder, nx0=20, ny0=8, nz0=4, n_times=8, n_fields=1,
                        species=("ions",))
        run = Run(str(folder))
        raw = run.frequency(buffer_frac=0.1).compute()
        assert len(range(*raw["xslice"].indices(20))) == 16

    @pytest.mark.filterwarnings("ignore:.*aliased.*:RuntimeWarning")
    def test_buffer_frac_none_takes_the_declared_widths(self, tmp_path):
        """
        ``None`` asks the run, so an asymmetric ``&nonlocal_x`` is honoured —
        the case a symmetric fraction cannot express.
        """
        folder = tmp_path / "declared"
        make_gene3d_run(folder, nx0=20, ny0=8, nz0=4, n_times=8, n_fields=1,
                        species=("ions",))
        run = Run(str(folder))
        run.params.get(0)["nonlocal_x"] = {"l_buffer_size": 0.10,
                                           "u_buffer_size": 0.25}
        raw = run.frequency(buffer_frac=None).compute()
        assert raw["xslice"] == g3.buffer_slice(run.params.get(0), 20)
        assert len(range(*raw["xslice"].indices(20))) == 20 - 2 - 5

    def test_xlim_beats_buffer_frac(self, wave_run):
        x = np.asarray(wave_run.coords[0]["x_o_a"])
        raw = wave_run.frequency(xlim=(float(x[0]), float(x[-1])),
                                 buffer_frac=0.25).compute()
        assert self._kept(wave_run, raw).size == x.size

    def test_the_window_used_is_recorded(self, wave_run):
        x = np.asarray(wave_run.coords[0]["x_o_a"])
        lo, hi = float(x[1]), float(x[-2])
        ds = wave_run.frequency(xlim=(lo, hi)).dataset()
        assert ds.attrs["x_avg_range"] == pytest.approx([lo, hi])
        assert ds.attrs["nx_averaged"] == x.size - 2
        assert len(ds.attrs["z_avg_range"]) == 2

    def test_the_reduction_normalises_rather_than_sums(self, wave_run):
        """
        The fixture's phi is uniform in x, so narrowing the window must leave
        the spectrum untouched. It would scale with the number of surfaces if
        the x reduction summed instead of averaging.
        """
        x = np.asarray(wave_run.coords[0]["x_o_a"])
        whole = wave_run.frequency().compute()["spectra"]["phi"]
        one = wave_run.frequency(
            xlim=(float(x[0]), float(x[0]))).compute()["spectra"]["phi"]
        assert one.shape == whole.shape
        np.testing.assert_allclose(one, whole, rtol=1e-6)


class TestSharedTimeAxis:
    """
    Every quantity is transformed on the times all its source files share.

    `istep_field` and `istep_mom` need not agree, so phi and a moment
    generally arrive on different axes. Taking the times from whichever reader
    came first pairs each moment sample with the wrong instant — and where the
    moment file is the shorter one, raises out of `resample_uniform` instead.
    """

    @staticmethod
    def _truncate_moments(folder, keep):
        """Leave the moment file short, as a killed or still-running job does."""
        with h5py.File(Path(folder) / "mom_ions.dat.h5", "r+") as f:
            grp = f["mom_ions"]
            for key in sorted(grp["n"].keys())[keep:]:
                for var in grp:
                    if var != "time" and key in grp[var]:
                        del grp[var][key]
            times = grp["time"][...]
            del grp["time"]
            grp.create_dataset("time", data=times[:keep], maxshape=(None,))

    @pytest.fixture
    def ragged_run(self, tmp_path):
        folder = tmp_path / "ragged"
        make_gene3d_run(folder, nx0=4, ny0=8, nz0=4, n_times=16, n_fields=1,
                        species=("ions",))
        self._truncate_moments(folder, keep=10)
        return Run(str(folder))

    @pytest.mark.filterwarnings("ignore:.*aliased.*:RuntimeWarning")
    def test_a_short_moment_file_does_not_raise(self, ragged_run):
        assert len(ragged_run.field.read_all_times()) == 16
        assert len(ragged_run.mom("ions").read_all_times()) == 10
        ds = ragged_run.frequency(quantities=("phi", "n")).dataset()
        assert ds.attrs["n_times"] == 10
        assert {"phi_power", "n_power"} <= set(ds.data_vars)

    @pytest.mark.filterwarnings("ignore:.*aliased.*:RuntimeWarning")
    def test_both_quantities_share_one_omega_axis(self, ragged_run):
        """A single omega coordinate is only meaningful if dt was the same."""
        ds = ragged_run.frequency(quantities=("phi", "n")).dataset()
        assert ds["phi_power"].dims == ds["n_power"].dims == ("ky", "omega")
        assert ds["phi_power"].shape == ds["n_power"].shape


class TestRadialAverageIsWeighted:
    """
    The x reduction is a Jacobian-weighted average, and composes exactly.

    Built on a phi whose inner and outer halves oscillate at different known
    frequencies, so a wrong weight shows up as a shifted band rather than as a
    shape difference nothing would notice.
    """

    @pytest.fixture
    def split_run(self, tmp_path):
        folder = tmp_path / "split"
        make_gene3d_run(folder, nx0=8, ny0=16, nz0=4, n_times=128, n_fields=1,
                        species=("ions",))
        with h5py.File(folder / "field.dat.h5", "r+") as f:
            grp = f["field"]
            times = np.asarray(grp["time"][...], dtype=float)
            keys = sorted(grp["phi"].keys())
            nz, ny, nx = grp["phi"][keys[0]].shape
            ly = 80.0
            y = np.arange(ny) * (ly / ny)
            k0 = 2 * np.pi * 2 / ly
            for it, key in enumerate(keys):
                arr = np.empty((nz, ny, nx))
                for ix in range(nx):
                    omega = 0.05 if ix < nx // 2 else 0.20
                    arr[:, :, ix] = np.cos(k0 * y - omega * times[it])[None, :]
                grp["phi"][key][...] = arr.astype(grp["phi"][key].dtype)
        return Run(str(folder)), k0

    @staticmethod
    def _at_k0(run, k0, **kw):
        ds = run.frequency(window_frac=0.5, **kw).dataset()
        ky = np.asarray(ds["ky"])
        i = int(np.argmin(np.abs(ky - k0)))
        return np.asarray(ds["phi_power"])[i], np.asarray(ds["phi_peak"])[i, 0]

    def test_each_half_reports_its_own_frequency(self, split_run):
        run, k0 = split_run
        x = np.asarray(run.coords[0]["x_o_a"])
        half = x.size // 2
        _, inner = self._at_k0(run, k0, xlim=(float(x[0]), float(x[half - 1])))
        _, outer = self._at_k0(run, k0, xlim=(float(x[half]), float(x[-1])))
        assert inner == pytest.approx(0.05, abs=0.01)
        assert outer == pytest.approx(0.20, abs=0.01)

    @pytest.mark.filterwarnings("ignore:.*aliased.*:RuntimeWarning")
    def test_the_whole_domain_is_the_weighted_combination(self, split_run):
        """
        Exact composition: the full-domain spectrum is the sum of the two
        halves weighted by their own `sum_z J`, not their plain mean.
        """
        run, k0 = split_run
        x = np.asarray(run.coords[0]["x_o_a"])
        z = np.asarray(run.coords[0]["z"])
        half = x.size // 2
        iz = int(np.argmin(np.abs(z)))
        w = run.geometry[0]["Jacobian"].mean(axis=1)[:, iz]
        w_in, w_out = w[:half].sum(), w[half:].sum()

        whole, _ = self._at_k0(run, k0)
        inner, _ = self._at_k0(run, k0, xlim=(float(x[0]), float(x[half - 1])))
        outer, _ = self._at_k0(run, k0, xlim=(float(x[half]), float(x[-1])))

        expect = (w_in * inner + w_out * outer) / (w_in + w_out)
        np.testing.assert_allclose(whole, expect, rtol=1e-6)
        # Guards the guard: the Jacobian varies enough that the plain mean is
        # a visibly different answer, so this is not a tautology.
        assert not np.allclose(whole, 0.5 * (inner + outer),
                               rtol=1e-3, atol=whole.max() * 1e-4)
