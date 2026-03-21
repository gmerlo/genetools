"""
data.py — Binary and ADIOS2 BP file readers for GENE simulation output.

GENE writes field and moment data in two formats:

* **Fortran unformatted binary** (default): Each iteration is a sequence of
  Fortran records — one scalar (time) followed by one record per complex array.
* **ADIOS2 BP** (optional): Structured, self-describing HDF5-like format.

Both readers expose the same public interface so calling code can treat them
interchangeably:

    reader.read_all_times()          → np.ndarray of shape (n_iters,)
    reader.stream_selected(indices)  → generator of (time, [arrays])

Example
-------
>>> from genetools.data import BinaryReader
>>> reader = BinaryReader('field', '/path/to/run/', '_0001', params)
>>> times = reader.read_all_times()
>>> for t, arrays in reader.stream_selected([0, 10, 20]):
...     phi = arrays[0]          # first field array, shape (nx, nky, nz)
"""

import mmap
import numpy as np

try:
    import adios2
    _ADIOS2_AVAILABLE = True
except ImportError:
    _ADIOS2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_dtypes(precision: str) -> tuple:
    """Return (real_dtype, complex_dtype) for *precision* string ('single'/'double')."""
    if precision.lower() == "single":
        return np.float32, np.complex64
    return np.float64, np.complex128


def _build_dims(params: dict, file_type: str) -> tuple:
    """Extract (ni, nj, nk, n_arrays) from a params dictionary."""
    box = params["box"]
    info = params["info"]
    ni = box.get("nx0", 1)
    nj = box.get("nky0", 1)
    nk = box.get("nz0", 1)
    key = "n_fields" if file_type == "field" else "n_moms"
    n_arrays = info.get(key, 1)
    return ni, nj, nk, n_arrays


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _BaseReader:
    """
    Abstract base for GENE data readers.

    Sub-classes must implement :meth:`read_all_times` and
    :meth:`stream_selected`.

    Parameters
    ----------
    file_type : str
        ``'field'`` or ``'mom'``.
    folder : str
        Directory that contains the data file.
    ext : str
        File-name suffix, e.g. ``'_0001'`` or ``'_0001.bp'``.
    params : dict
        Simulation parameter dictionary (as returned by :class:`~genetools.params.Params`).
    species : str, optional
        Species name inserted into the filename for moment files.
    """

    def __init__(self, file_type: str, folder: str, ext: str, params: dict, species: str = None):
        species_str = f"_{species}" if species else ""
        self.filename = f"{folder}{file_type}{species_str}{ext}"
        self.ni, self.nj, self.nk, self.n_arrays = _build_dims(params, file_type)
        precision = params["info"].get("precision", "double")
        self.real_dtype, self.cpx_dtype = _resolve_dtypes(precision)
        self.npts = self.ni * self.nj * self.nk

    # ------------------------------------------------------------------
    # Public interface (must be implemented by sub-classes)
    # ------------------------------------------------------------------

    def read_all_times(self) -> np.ndarray:
        """Return a 1-D array of all simulation times in the file."""
        raise NotImplementedError

    def stream_selected(self, iteration_indices):
        """
        Yield ``(time, arrays)`` for the given *iteration_indices*.

        Parameters
        ----------
        iteration_indices : sequence of int
            Zero-based iteration indices to read.

        Yields
        ------
        time : float
            Simulation time for the iteration.
        arrays : list of np.ndarray
            One complex array per field/moment, each shaped ``(ni, nj, nk)``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Fortran unformatted binary reader
# ---------------------------------------------------------------------------

class BinaryReader(_BaseReader):
    """
    Streaming reader for Fortran unformatted binary GENE output files.

    The record index (byte offsets for every record) is built once on the
    first access and cached, so repeated calls to :meth:`read_all_times` or
    :meth:`stream_selected` do not re-scan the file.

    Inherits all parameters from :class:`_BaseReader`.
    """

    def __init__(self, file_type: str, folder: str, ext: str, params: dict, species: str = None):
        super().__init__(file_type, folder, ext, params, species)
        self._record_index = None  # lazily built and cached

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_record_index(self, mm) -> list:
        """
        Return the cached record index, building it from *mm* if necessary.

        Each entry is a ``(payload_start, payload_size)`` tuple.
        """
        if self._record_index is None:
            self._record_index = self._build_record_index(mm)
        return self._record_index

    @staticmethod
    def _build_record_index(mm) -> list:
        """
        Scan a Fortran unformatted file and return
        ``[(payload_start, payload_bytes), ...]`` for every record.

        Fortran wraps each record with a 4-byte integer marker before *and*
        after the payload that stores the byte length of the payload.
        """
        index = []
        pos = 0
        size = mm.size()
        while pos < size:
            nbytes = int(np.frombuffer(mm[pos: pos + 4], dtype=np.int32)[0])
            start = pos + 4
            index.append((start, nbytes))
            pos = start + nbytes + 4  # skip trailing marker
        return index

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read_all_times(self) -> np.ndarray:
        """
        Fast first pass — read only the scalar time value from every iteration.

        Returns
        -------
        np.ndarray
            Shape ``(n_iters,)``, dtype matches the file's precision setting.
        """
        with open(self.filename, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            idx = self._get_record_index(mm)
            records_per_iter = 1 + self.n_arrays
            n_iters = len(idx) // records_per_iter
            times = np.empty(n_iters, dtype=self.real_dtype)
            for it in range(n_iters):
                start, nbytes = idx[it * records_per_iter]
                times[it] = np.frombuffer(mm[start: start + nbytes], dtype=self.real_dtype)[0]
            mm.close()
        return times

    def stream_selected(self, iteration_indices):
        """
        Stream only the requested iterations from the binary file.

        Yields ``(time, [array_0, array_1, ...])`` where each array has
        shape ``(ni, nj, nk)`` in Fortran (column-major) order.
        """
        with open(self.filename, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            idx = self._get_record_index(mm)
            records_per_iter = 1 + self.n_arrays
            for it in iteration_indices:
                base = it * records_per_iter
                # scalar record
                start, nbytes = idx[base]
                time = np.frombuffer(mm[start: start + nbytes], dtype=self.real_dtype)[0]
                # complex array records
                data = []
                for k in range(self.n_arrays):
                    s, nb = idx[base + 1 + k]
                    arr = np.frombuffer(mm[s: s + nb], dtype=self.cpx_dtype, count=self.npts)
                    arr = arr.reshape((self.ni, self.nj, self.nk), order="F")
                    data.append(arr)
                yield float(time), data
            mm.close()


# ---------------------------------------------------------------------------
# ADIOS2 BP reader (only defined when adios2 is importable)
# ---------------------------------------------------------------------------

if _ADIOS2_AVAILABLE:

    class BPReader(_BaseReader):
        """
        Lazy streaming reader for ADIOS2 BP files.

        Exposes the same interface as :class:`BinaryReader` so that calling
        code does not need to know which format is in use.

        Inherits all parameters from :class:`_BaseReader`.
        """

        def read_all_times(self) -> np.ndarray:
            """
            Read every ``'time'`` variable from the BP file.

            Returns
            -------
            np.ndarray
                Shape ``(n_iters,)``.
            """
            times = []
            with adios2.open(self.filename, "r") as fh:
                for step in fh:
                    times.append(step.read("time", np_type=self.real_dtype))
            return np.array(times, dtype=self.real_dtype)

        def stream_selected(self, iteration_indices):
            """
            Stream only the requested iterations from the BP file.

            Uses a set for O(1) membership tests. Iterations not in
            *iteration_indices* are skipped without reading their payload.

            Yields ``(time, [array_0, array_1, ...])`` — same contract as
            :class:`BinaryReader`.
            """
            iter_set = set(iteration_indices)
            with adios2.open(self.filename, "r") as fh:
                for step_idx, step in enumerate(fh):
                    if step_idx not in iter_set:
                        continue
                    time = step.read("time", np_type=self.real_dtype)
                    data = []
                    for k in range(self.n_arrays):
                        arr = step.read(f"array_{k}", np_type=self.cpx_dtype)
                        arr = arr.reshape((self.ni, self.nj, self.nk), order="F")
                        data.append(arr)
                    yield float(time), data

else:

    class BPReader(_BaseReader):  # type: ignore[no-redef]
        """Placeholder raised when the ``adios2`` Python package is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "BPReader requires the 'adios2' Python package. "
                "Install it via: pip install adios2"
            )
