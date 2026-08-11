"""Tests for CachingDiagnostic._sync_field_mom_indices.

Every flux diagnostic correlates a field snapshot with moment snapshots from
the *same* time. Field and moment files are written at different cadences and
either may be truncated, so the pairing must be by time value. Pairing by
position silently correlates data from different times (wrong fluxes, no
warning) and raises StopIteration when the moment file is the shorter one.
"""

import numpy as np
import pytest

from genetools.diagnostics._base import CachingDiagnostic


class _Diag(CachingDiagnostic):
    """CachingDiagnostic with a cache that never exists on disk."""

    def __init__(self, saved=None):
        self.outfile = "/nonexistent/cache.h5"
        self._saved = np.array([] if saved is None else saved, dtype=np.float64)

    def _load_saved_times(self):
        return self._saved


class _Reader:
    def __init__(self, times):
        self._times = np.asarray(times, dtype=np.float64)

    def read_all_times(self):
        return self._times


PARAMS = {"in_out": {"istep_field": 1, "istep_mom": 2}}


def _sync(field_times, mom_times, saved=None, window=(-1e30, 1e30)):
    """Return the paired (field_times, mom_times) actually selected."""
    tf = np.asarray(field_times, dtype=np.float64)
    tm = np.asarray(mom_times, dtype=np.float64)
    diag = _Diag(saved)
    i_f, i_m = diag._sync_field_mom_indices(
        _Reader(tf), [_Reader(tm)], window[0], window[1], PARAMS)
    return tf[i_f], tm[i_m]


class TestTimeAlignment:
    """The paired times must be equal — this is the correctness property."""

    @pytest.mark.parametrize("field_times, mom_times, expected", [
        # field every step, moments every 2nd — the common GENE setup
        ([1., 2., 3., 4., 5., 6.], [2., 4., 6.], [2., 4., 6.]),
        # identical cadence
        ([1., 2., 3.], [1., 2., 3.], [1., 2., 3.]),
        # moment file truncated by a killed run
        ([1., 2., 3.], [1., 2.], [1., 2.]),
        # moment file missing an early step
        ([1., 2., 3.], [2., 3.], [2., 3.]),
        # field coarser than moments
        ([2., 4.], [1., 2., 3., 4.], [2., 4.]),
    ])
    def test_paired_times_are_equal(self, field_times, mom_times, expected):
        got_f, got_m = _sync(field_times, mom_times)
        np.testing.assert_allclose(got_f, got_m)
        np.testing.assert_allclose(got_f, expected)

    def test_no_common_times_returns_empty(self):
        got_f, got_m = _sync([1., 2.], [7., 8.])
        assert got_f.size == 0 and got_m.size == 0

    def test_equal_lengths_always(self):
        """Unequal lengths are what crashed the streaming loop."""
        got_f, got_m = _sync([1., 2., 3., 4.], [2., 3.])
        assert len(got_f) == len(got_m)

    def test_unsorted_moment_times_still_pair(self):
        """Restart segments can append out of order."""
        got_f, got_m = _sync([1., 2., 3.], [3., 1., 2.])
        np.testing.assert_allclose(got_f, got_m)
        np.testing.assert_allclose(got_f, [1., 2., 3.])


class TestWindowing:

    def test_window_restricts_both(self):
        got_f, got_m = _sync([1., 2., 3., 4.], [1., 2., 3., 4.],
                             window=(2., 3.))
        np.testing.assert_allclose(got_f, [2., 3.])
        np.testing.assert_allclose(got_m, [2., 3.])

    def test_window_outside_data_is_empty(self):
        got_f, got_m = _sync([1., 2.], [1., 2.], window=(100., 200.))
        assert got_f.size == 0 and got_m.size == 0


class TestAlreadySavedFiltering:

    def test_saved_times_are_skipped(self):
        got_f, got_m = _sync([1., 2., 3.], [1., 2., 3.], saved=[1., 2.])
        np.testing.assert_allclose(got_f, [3.])
        np.testing.assert_allclose(got_m, [3.])

    def test_filtering_keeps_pairs_aligned(self):
        """Filtering must drop a field/moment pair together, never one side."""
        got_f, got_m = _sync([1., 2., 3., 4.], [1., 2., 3., 4.], saved=[2.])
        np.testing.assert_allclose(got_f, got_m)
        np.testing.assert_allclose(got_f, [1., 3., 4.])

    def test_all_saved_returns_empty(self):
        got_f, got_m = _sync([1., 2.], [1., 2.], saved=[1., 2.])
        assert got_f.size == 0 and got_m.size == 0

    def test_tolerant_match_on_float_noise(self):
        """Cached times are float32-rounded for single-precision runs."""
        got_f, _ = _sync([1., 2.], [1., 2.], saved=[1.0000000001])
        np.testing.assert_allclose(got_f, [2.])
