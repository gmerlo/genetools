"""Tests for genetools.io.data.H5Reader (GENE `write_h5` and GENE-3D)."""

import h5py
import numpy as np
import pytest

from genetools.io.data import H5Reader
from genetools.io.params import Params
from tests.gene3d_fixture import MOM_LABELS, make_gene3d_run


@pytest.fixture
def g3d(tmp_path):
    """A synthetic GENE-3D run plus its params dict and reader folder string."""
    run = make_gene3d_run(tmp_path / "run", n_times=4,
                          write_vsp=True, write_srcmom=True)
    params = Params(run.folder, [".dat"]).get(0)
    return run, params, str(run.folder) + "/"


def _reader(folder, params, file_type="field", species=None):
    return H5Reader(file_type, folder, ".dat.h5", params, species=species)


# ---------------------------------------------------------------------------
# Times
# ---------------------------------------------------------------------------

class TestTimes:

    def test_reads_the_time_dataset(self, g3d):
        run, params, folder = g3d
        assert np.allclose(_reader(folder, params).read_all_times(), run.times)

    def test_truncated_file_drops_the_dangling_time(self, g3d):
        """
        GENE appends the timestamp *before* writing the arrays.

        A run killed between those two writes leaves one more entry in ``time``
        than there are snapshots; that must not surface as a missing-dataset
        crash on the last step.
        """
        run, params, folder = g3d
        path = run.folder / "field.dat.h5"
        with h5py.File(path, "a") as f:
            last = f"{len(run.times) - 1:010d}"
            for label in run.field_labels:
                del f[f"field/{label}/{last}"]

        reader = _reader(folder, params)
        times = reader.read_all_times()
        assert times.size == run.times.size - 1
        assert np.allclose(times, run.times[:-1])
        # And the surviving steps still stream.
        assert len(list(reader.stream_selected(range(times.size)))) == times.size

    def test_partially_written_step_is_dropped(self, g3d):
        """A step present for phi but not A_par is not a complete snapshot."""
        run, params, folder = g3d
        with h5py.File(run.folder / "field.dat.h5", "a") as f:
            del f["field/A_par/0000000002"]
        times = _reader(folder, params).read_all_times()
        assert np.allclose(times, np.delete(run.times, 2))


# ---------------------------------------------------------------------------
# Variable discovery
# ---------------------------------------------------------------------------

class TestVariableDiscovery:

    def test_field_vars_in_gene_order(self, g3d):
        run, params, folder = g3d
        assert _reader(folder, params).var_names == ["phi", "A_par"]

    def test_moments_ignore_the_wrong_n_moms(self, g3d):
        """
        GENE-3D's ``parameters`` reports ``n_moms = 6`` while ``diag_3d``
        writes ten datasets — two different module variables of the same name.
        Trusting the namelist would silently drop Gamma_em..q_perp.
        """
        run, params, folder = g3d
        assert params["info"]["n_moms"] == 6
        reader = _reader(folder, params, "mom", species="ions")
        assert reader.var_names == MOM_LABELS
        assert reader.n_arrays == 10

    def test_unknown_variables_are_kept_after_the_known_ones(self, g3d):
        run, params, folder = g3d
        with h5py.File(run.folder / "field.dat.h5", "a") as f:
            grp = f.create_group("field/zzz_custom")
            for it in range(run.times.size):
                grp.create_dataset(f"{it:010d}", data=np.zeros((4, 8, 6),
                                                               dtype=np.float32))
        assert _reader(folder, params).var_names == ["phi", "A_par", "zzz_custom"]

    def test_index_of_finds_a_named_variable(self, g3d):
        run, params, folder = g3d
        reader = _reader(folder, params, "mom", species="ions")
        assert reader.index_of("Q_es") == MOM_LABELS.index("Q_es")

    def test_index_of_reports_what_is_available(self, g3d):
        run, params, folder = g3d
        reader = _reader(folder, params, "mom", species="ions")
        with pytest.raises(KeyError, match="Gamma_es"):
            reader.index_of("not_a_moment")

    def test_srcmom_has_six_datasets_not_nine(self, g3d):
        """GENE-3D writes ck_heat/ck_part only — there are no f0_term moments."""
        run, params, folder = g3d
        reader = _reader(folder, params, "srcmom", species="ions")
        assert reader.var_names == ["ck_heat_M00", "ck_heat_M10", "ck_heat_M22",
                                    "ck_part_M00", "ck_part_M10", "ck_part_M22"]


# ---------------------------------------------------------------------------
# Array decoding
# ---------------------------------------------------------------------------

class TestArrays:

    def test_axis_order_is_undone(self, g3d):
        """futils stores (nx,ny,nz) as (nz,ny,nx); the reader must flip it back."""
        run, params, folder = g3d
        (_, arrays), = _reader(folder, params).stream_selected([0])
        assert arrays[0].shape == (run.nx0, run.ny0, run.nz0)

    def test_values_match_what_was_written(self, g3d):
        run, params, folder = g3d
        reader = _reader(folder, params)
        for pos, (t, arrays) in enumerate(reader.stream_selected([0, 2])):
            it = [0, 2][pos]
            assert t == pytest.approx(run.times[it])
            for k, label in enumerate(run.field_labels):
                expected = run.fields[label][it].astype(np.float32)
                assert np.allclose(arrays[k], expected)

    def test_dtype_comes_from_the_file_not_from_precision(self, g3d):
        """
        ``creatf(..., 's')`` stores 32-bit reals even though this run's
        ``PRECISION`` says DOUBLE, and GENE-3D data is real, not complex.
        """
        run, params, folder = g3d
        assert str(params["info"]["precision"]).upper() == "DOUBLE"
        (_, arrays), = _reader(folder, params).stream_selected([0])
        assert arrays[0].dtype == np.float32
        assert not np.iscomplexobj(arrays[0])

    def test_moment_values_are_addressable_by_name(self, g3d):
        run, params, folder = g3d
        reader = _reader(folder, params, "mom", species="electrons")
        idx = reader.index_of("Gamma_em")
        (_, arrays), = reader.stream_selected([1])
        expected = run.moments["electrons"]["Gamma_em"][1].astype(np.float32)
        assert np.allclose(arrays[idx], expected)

    def test_velocity_space_arrays_keep_four_dimensions(self, g3d):
        run, params, folder = g3d
        (_, arrays), = _reader(folder, params, "vsp").stream_selected([0])
        # Fortran order is (nz0, nv0, nw0, n_spec).
        assert arrays[0].shape == (run.nz0, 8, 4, len(run.species))

    def test_source_moments_are_radial_profiles(self, g3d):
        run, params, folder = g3d
        (_, arrays), = _reader(folder, params, "srcmom",
                               species="ions").stream_selected([0])
        assert arrays[0].shape == (run.nx0,)

    def test_empty_index_list_yields_nothing(self, g3d):
        run, params, folder = g3d
        assert list(_reader(folder, params).stream_selected([])) == []

    def test_requested_order_is_honoured(self, g3d):
        run, params, folder = g3d
        got = [t for t, _ in _reader(folder, params).stream_selected([2, 0, 1])]
        assert got == pytest.approx([run.times[2], run.times[0], run.times[1]])


# ---------------------------------------------------------------------------
# Complex data (GENE's own write_h5 output)
# ---------------------------------------------------------------------------

class TestComplexCompound:

    @staticmethod
    def _write_complex_field(path, nx, nky, nz, n_times, compound):
        rng = np.random.default_rng(11)
        truth = []
        with h5py.File(path, "w") as f:
            grp = f.create_group("field")
            grp.create_dataset("time", data=np.arange(n_times, dtype=float))
            sub = grp.create_group("phi")
            for it in range(n_times):
                arr = (rng.standard_normal((nx, nky, nz))
                       + 1j * rng.standard_normal((nx, nky, nz)))
                truth.append(arr)
                disk = arr.T                      # futils Fortran ordering
                if compound:
                    dtype = np.dtype([("real", "<f8"), ("imaginary", "<f8")])
                    rec = np.empty(disk.shape, dtype=dtype)
                    rec["real"] = disk.real
                    rec["imaginary"] = disk.imag
                    sub.create_dataset(f"{it:010d}", data=rec)
                else:
                    sub.create_dataset(f"{it:010d}", data=disk)
        return truth

    @pytest.mark.parametrize("compound", [True, False])
    def test_complex_fields_round_trip(self, tmp_path, compound):
        """
        GENE writes complex data either natively or as a ``{real, imaginary}``
        compound type depending on how HDF5 was built. Both must decode.
        """
        nx, nky, nz, n_times = 5, 3, 4, 2
        truth = self._write_complex_field(
            tmp_path / "field.dat.h5", nx, nky, nz, n_times, compound)
        params = {
            "box": {"nx0": nx, "nky0": nky, "nz0": nz},
            "info": {"n_fields": 1, "precision": "DOUBLE",
                     "y_local": True, "is_3d": False},
        }
        reader = H5Reader("field", str(tmp_path) + "/", ".dat.h5", params)
        assert reader.var_names == ["phi"]
        for it, (_, arrays) in enumerate(reader.stream_selected(range(n_times))):
            assert arrays[0].shape == (nx, nky, nz)
            assert np.iscomplexobj(arrays[0])
            assert np.allclose(arrays[0], truth[it])
