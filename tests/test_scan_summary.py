# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Tests for ``scripts/scan_summary.py``.

Mostly the header parser. GENE writes scan.log in more than one shape and the
one it actually writes for a multi-parameter scan is the awkward one: the
eigenvalue marker has no column separator of its own, so the header has fewer
fields than the data rows, and every parameter name carries a trailing scan
index. Parsing that wrongly is how the script failed on its first real
directory.
"""

import importlib.util
import textwrap
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "scan_summary",
    Path(__file__).resolve().parents[1] / "scripts" / "scan_summary.py")
scan_summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scan_summary)


class TestCleanHeader:
    """One header field in, the parameter's name out."""

    @pytest.mark.parametrize("raw, expected", [
        ("#Run", "Run"),
        ("x0        1", "x0"),
        ("kymin     1  /Eigenvalue1", "kymin"),   # marker hung off the last one
        ("kymin", "kymin"),
        ("/Eigenvalue", ""),                      # marker alone: no name
        ("omn1     2", "omn1"),                   # index stripped, digit kept
        ("growthrate", "growthrate"),
    ])
    def test_names(self, raw, expected):
        assert scan_summary._clean_header(raw) == expected

    def test_a_trailing_digit_without_a_separator_is_part_of_the_name(self):
        """``omn1`` is a parameter; ``omn 1`` is a parameter plus its index."""
        assert scan_summary._clean_header("omn1") == "omn1"
        assert scan_summary._clean_header("omn 1") == "omn"


class TestParseScanLog:
    def _write(self, tmp_path, text):
        path = tmp_path / "scan.log"
        path.write_text(textwrap.dedent(text))
        return str(path)

    def test_gene_multi_parameter_format(self, tmp_path):
        """
        The format GENE actually writes, verbatim from a production scan.

        Three header fields, four data columns: the eigenvalue block is
        separated by ``|`` in the rows but only by ``/`` in the header.
        """
        path = self._write(tmp_path, """\
            #Run  | x0        1  | kymin     1  /Eigenvalue1
            0001  | 9.750000e-01 | 5.000000e-02 |  0.0670 -1.0260
            0002  | 9.500000e-01 | 5.000000e-02 |  0.0570 -0.0880
            """)
        entries, names = scan_summary.parse_scan_log(path)
        assert names == ["x0", "kymin"]
        assert scan_summary.identify_ky(names) == "kymin"
        assert entries[0]["run"] == "0001"
        assert entries[0]["params"] == {"x0": 0.975, "kymin": 0.05}
        assert entries[0]["gamma"] == pytest.approx(0.067)
        assert entries[0]["omega"] == pytest.approx(-1.026)

    def test_pipe_format_with_its_own_eigenvalue_field(self, tmp_path):
        path = self._write(tmp_path, """\
            #Run       |kymin      |omt1       |/Eigenvalue
            0001     |  0.100000 |  6.000000 |   1.2000E-01  -3.4000E-01
            """)
        entries, names = scan_summary.parse_scan_log(path)
        assert names == ["kymin", "omt1"]
        assert entries[0]["gamma"] == pytest.approx(0.12)
        assert entries[0]["omega"] == pytest.approx(-0.34)

    def test_whitespace_format_with_named_columns(self, tmp_path):
        path = self._write(tmp_path, """\
            #Run kymin growthrate frequency
            0001 0.150000 3.000000E-02 -7.500000E-02
            """)
        entries, names = scan_summary.parse_scan_log(path)
        assert names == ["kymin"]
        assert entries[0]["gamma"] == pytest.approx(0.03)
        assert entries[0]["omega"] == pytest.approx(-0.075)

    def test_fortran_d_exponent_is_read(self, tmp_path):
        path = self._write(tmp_path, """\
            #Run  | kymin     1  /Eigenvalue1
            0001  | 5.000000D-02 |  0.0670 -1.0260
            """)
        entries, _ = scan_summary.parse_scan_log(path)
        assert entries[0]["params"]["kymin"] == pytest.approx(0.05)

    def test_a_file_with_no_rows_says_so(self, tmp_path):
        path = self._write(tmp_path, "nonsense\nwith no runs\n")
        with pytest.raises(ValueError):
            entries, names = scan_summary.parse_scan_log(path)
            scan_summary.identify_ky(names)

    def test_unknown_ky_column_lists_what_there_is(self, tmp_path):
        path = self._write(tmp_path, """\
            #Run  | x0        1  /Eigenvalue1
            0001  | 9.750000e-01 |  0.0670 -1.0260
            """)
        _, names = scan_summary.parse_scan_log(path)
        with pytest.raises(ValueError, match="x0"):
            scan_summary.identify_ky(names)
        assert scan_summary.identify_ky(names, "x0") == "x0"


class TestNrgRatios:
    def _nrg(self, tmp_path, ion, ele, n_times=3):
        path = tmp_path / "nrg_0001"
        with open(path, "w") as fh:
            for it in range(n_times):
                fh.write(f"   {it * 2.0:.6f}\n")
                for row in (ion, ele):
                    fh.write("  " + "  ".join(f"{v:.6E}" for v in row) + "\n")
        return str(path)

    def test_ratios_use_the_right_columns_and_species(self, tmp_path):
        """Q_es is column 7 and Q_em column 8; electrons are the negative charge."""
        ion = [1, 1, 1, 1, 0, 0, 80.0, -6.0, 0, 0]
        ele = [1, 1, 1, 1, 0, 0, 40.0, -8.0, 0, 0]
        path = self._nrg(tmp_path, ion, ele)
        species = [("ions", 1.0), ("electrons", -1.0)]
        out = scan_summary.flux_ratios(path, species, navg=1)
        assert out["q_es_e_over_i"] == pytest.approx(0.5)
        assert out["q_em_over_es_e"] == pytest.approx(-0.2)

    def test_species_order_is_read_not_assumed(self, tmp_path):
        """Electrons written first must not be taken for the ions."""
        ele = [1, 1, 1, 1, 0, 0, 40.0, -8.0, 0, 0]
        ion = [1, 1, 1, 1, 0, 0, 80.0, -6.0, 0, 0]
        path = self._nrg(tmp_path, ele, ion)          # reversed on disk
        species = [("electrons", -1.0), ("ions", 1.0)]
        out = scan_summary.flux_ratios(path, species, navg=1)
        assert out["q_es_e_over_i"] == pytest.approx(0.5)

    def test_without_a_parameters_file_the_ratios_are_nan(self, tmp_path):
        path = self._nrg(tmp_path, [1] * 10, [2] * 10)
        out = scan_summary.flux_ratios(path, [], navg=1)
        assert np.isnan(out["q_es_e_over_i"])

    def test_species_order_reads_charge_from_the_parameters_file(self, tmp_path):
        (tmp_path / "parameters_0001").write_text(
            "&species\n name = 'ions'\n charge =    1.0000000    \n/\n"
            "&species\n name = 'electrons'\n charge =   -1.0000000\n/\n")
        order = scan_summary.species_order(str(tmp_path), "0001")
        assert [n for n, _ in order] == ["ions", "electrons"]
        assert scan_summary.electron_ion_indices(order) == (1, 0)
