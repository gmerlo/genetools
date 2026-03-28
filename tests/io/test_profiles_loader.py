"""Tests for genetools.io.profiles_loader."""

import numpy as np
import pytest

from genetools.io.profiles_loader import load_equilibrium_profiles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COLUMNS = ("x_o_rho_ref", "x_o_a", "T", "n", "omt", "omn")

# 7 data rows, one per line, values chosen to be easy to verify
_DATA_ROWS = [
    (0.1, 0.10, 1.0, 0.90, 2.5, 2.1),
    (0.2, 0.20, 1.1, 0.91, 2.6, 2.2),
    (0.3, 0.30, 1.2, 0.92, 2.7, 2.3),
    (0.4, 0.40, 1.3, 0.93, 2.8, 2.4),
    (0.5, 0.50, 1.4, 0.94, 2.9, 2.5),
    (0.6, 0.60, 1.5, 0.95, 3.0, 2.6),
    (0.7, 0.70, 1.6, 0.96, 3.1, 2.7),
]


def _write_profile_file(folder, species_name, ext, rows=None):
    """Write a synthetic profiles file and return its path."""
    if rows is None:
        rows = _DATA_ROWS
    path = folder / f"profiles_{species_name}{ext}"
    lines = [
        "# header line one\n",
        "# header line two\n",
    ]
    for row in rows:
        lines.append("  ".join(f"{v:.6e}" for v in row) + "\n")
    path.write_text("".join(lines))
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadEquilibriumProfiles:

    def test_happy_path_returns_all_keys(self, tmp_path):
        """All six expected keys are present in the returned dict."""
        _write_profile_file(tmp_path, "ions", "_0001")
        result = load_equilibrium_profiles(str(tmp_path), "_0001", "ions")
        assert set(result.keys()) == set(COLUMNS)

    def test_happy_path_values_correct(self, tmp_path):
        """Loaded values match the data written to the file."""
        _write_profile_file(tmp_path, "ions", "_0001")
        result = load_equilibrium_profiles(str(tmp_path), "_0001", "ions")

        expected = {col: np.array([row[i] for row in _DATA_ROWS])
                    for i, col in enumerate(COLUMNS)}

        for col in COLUMNS:
            np.testing.assert_allclose(result[col], expected[col], rtol=1e-6,
                                       err_msg=f"Mismatch in column '{col}'")

    def test_all_arrays_same_shape(self, tmp_path):
        """Every returned array has the same length (one entry per data row)."""
        _write_profile_file(tmp_path, "electrons", "_0002")
        result = load_equilibrium_profiles(str(tmp_path), "_0002", "electrons")

        lengths = {col: len(result[col]) for col in COLUMNS}
        assert len(set(lengths.values())) == 1, (
            f"Arrays have different lengths: {lengths}"
        )
        assert list(lengths.values())[0] == len(_DATA_ROWS)

    def test_values_are_numpy_arrays(self, tmp_path):
        """Each value in the dict is a 1-D numpy ndarray."""
        _write_profile_file(tmp_path, "ions", "_0001")
        result = load_equilibrium_profiles(str(tmp_path), "_0001", "ions")

        for col in COLUMNS:
            assert isinstance(result[col], np.ndarray), (
                f"'{col}' is not an ndarray"
            )
            assert result[col].ndim == 1, (
                f"'{col}' has ndim={result[col].ndim}, expected 1"
            )

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """FileNotFoundError is raised when the profile file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_equilibrium_profiles(str(tmp_path), "_0001", "ions")

    def test_species_name_used_in_filename(self, tmp_path):
        """The species name is correctly embedded in the looked-up filename."""
        # Write a file for 'electrons' only; loading 'ions' must fail.
        _write_profile_file(tmp_path, "electrons", "_0001")
        with pytest.raises(FileNotFoundError):
            load_equilibrium_profiles(str(tmp_path), "_0001", "ions")

    def test_ext_used_in_filename(self, tmp_path):
        """The extension suffix is correctly embedded in the looked-up filename."""
        # Write a file with ext '_0001' only; loading '_0002' must fail.
        _write_profile_file(tmp_path, "ions", "_0001")
        with pytest.raises(FileNotFoundError):
            load_equilibrium_profiles(str(tmp_path), "_0002", "ions")
