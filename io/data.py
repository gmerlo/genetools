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

    def segment_of(self, global_idx: int) -> int:
        """Return segment index for *global_idx* (always 0 for single readers)."""
        return 0

    def stream_selected_with_seg(self, iteration_indices):
        """
        Like :meth:`stream_selected` but also yields segment index (always 0).

        Yields
        ------
        time : float
        arrays : list of np.ndarray
        seg_idx : int  (always 0 for single-segment readers)
        """
        for t, arrays in self.stream_selected(iteration_indices):
            yield t, arrays, 0


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

        Optimisation: after scanning the first full iteration (1 + n_arrays
        records), if all records within that group have fixed sizes, the
        remaining offsets are computed arithmetically instead of scanning
        byte-by-byte.
        """
        size = mm.size()
        if size == 0:
            return []

        # Scan first record to bootstrap
        index = []
        pos = 0
        first_nbytes = int(np.frombuffer(mm[pos: pos + 4], dtype=np.int32)[0])
        index.append((pos + 4, first_nbytes))
        pos = pos + 4 + first_nbytes + 4

        # Scan more records to detect periodicity
        while pos < size:
            nbytes = int(np.frombuffer(mm[pos: pos + 4], dtype=np.int32)[0])
            start = pos + 4
            index.append((start, nbytes))
            pos = start + nbytes + 4
            # Check if we've found a repeating pattern: the current record
            # has the same size as the first → likely start of next iteration
            if nbytes == first_nbytes and len(index) > 1:
                break

        # Determine stride (bytes per full iteration group)
        group_len = len(index) - 1  # last record starts next group
        if group_len < 1:
            # Only one record type, finish scanning
            while pos < size:
                nbytes = int(np.frombuffer(mm[pos: pos + 4], dtype=np.int32)[0])
                index.append((pos + 4, nbytes))
                pos += 4 + nbytes + 4
            return index

        # Compute stride = sum of (4 + payload + 4) for each record in group
        stride = sum(4 + index[i][1] + 4 for i in range(group_len))
        record_sizes = [index[i][1] for i in range(group_len)]

        # Remove the extra record we read (it's the first of the next group)
        index.pop()
        pos_group_start = index[0][0] - 4  # byte offset of first group

        # Fill remaining groups arithmetically
        n_full_groups = (size - pos_group_start) // stride
        for g in range(1, n_full_groups):
            base = pos_group_start + g * stride
            rec_pos = base
            for rec_size in record_sizes:
                index.append((rec_pos + 4, rec_size))
                rec_pos += 4 + rec_size + 4

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
            nbytes_time = idx[0][1]

            # Gather all time-record byte offsets and read in one shot
            offsets = np.array([idx[it * records_per_iter][0] for it in range(n_iters)],
                              dtype=np.int64)
            buf = np.empty(n_iters, dtype=self.real_dtype)
            byte_width = np.dtype(self.real_dtype).itemsize
            for i, off in enumerate(offsets):
                buf[i] = np.frombuffer(mm[off: off + byte_width],
                                       dtype=self.real_dtype)[0]
            mm.close()
        return buf

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

        def __init__(self, file_type: str, folder: str, ext: str,
                     params: dict, species: str = None):
            super().__init__(file_type, folder, ext, params, species)
            self._file_type = file_type   # 'field' or 'mom'

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

        # Variable names written by GENE for each file type, in array order
        _FIELD_VARS = ["phi", "A_par", "B_par"]
        _MOM_VARS   = ["dens", "T_par", "T_perp", "q_par", "q_perp",
                       "u_par", "densI1", "T_parI1", "T_ppI1"]

        def _var_names(self) -> list:
            """Return the ordered list of variable names for this file type."""
            if self._file_type == "field":
                return self._FIELD_VARS[:self.n_arrays]
            return self._MOM_VARS[:self.n_arrays]

        def stream_selected(self, iteration_indices):
            """
            Stream only the requested iterations from the BP file.

            Yields ``(time, [array_0, array_1, ...])`` in the same order as
            *iteration_indices*, matching the contract of :class:`BinaryReader`.

            ADIOS2 only supports forward sequential access, so the file is
            walked once in ascending step order. Results that arrive before
            their turn in the requested order are buffered and yielded as soon
            as the caller's order is satisfied.
            """
            iteration_indices = list(iteration_indices)
            if not iteration_indices:
                return

            iter_set  = set(iteration_indices)
            var_names = self._var_names()
            buffer    = {}          # step_idx -> (time, data)
            next_out  = 0           # next position in iteration_indices to yield

            with adios2.open(self.filename, "r") as fh:
                for step_idx, step in enumerate(fh):
                    if step_idx not in iter_set:
                        continue
                    time = float(step.read("time", np_type=self.real_dtype))
                    data = []
                    for name in var_names:
                        arr = step.read(name, np_type=self.cpx_dtype)
                        arr = arr.reshape((self.ni, self.nj, self.nk), order="F")
                        data.append(arr)
                    buffer[step_idx] = (time, data)

                    # yield as many consecutive results as are ready
                    while (next_out < len(iteration_indices) and
                           iteration_indices[next_out] in buffer):
                        yield buffer.pop(iteration_indices[next_out])
                        next_out += 1

else:

    class BPReader(_BaseReader):  # type: ignore[no-redef]
        """Placeholder raised when the ``adios2`` Python package is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "BPReader requires the 'adios2' Python package. "
                "Install it via: pip install adios2"
            )


# ---------------------------------------------------------------------------
# Multi-segment reader
# ---------------------------------------------------------------------------

class MultiSegmentReader:
    """
    Transparent wrapper that stitches multiple run-segment readers into one.

    Exposes exactly the same interface as :class:`BinaryReader` so that all
    diagnostics (Contours, ShearingRate, Spectra) work without modification.

    **Overlap handling (restarts)**
    When two segments share time values, the *later segment wins* — its
    version of a time step replaces the earlier one.  This matches GENE's
    restart behaviour where the new segment rewinds slightly.

    **Variable grids / precisions**
    Each segment reader retains its own grid and dtype from the corresponding
    params file.  Arrays yielded by :meth:`stream_selected` have the native
    shape of the segment they come from.

    The ``ni``, ``nj``, ``nk`` attributes always reflect the **first segment**
    and are used by Contours only as fallback default slice indices.

    Parameters
    ----------
    readers : list of _BaseReader
        Segment readers in file order (e.g. ``_0001``, ``_0002``, ...).
    tol : float, optional
        Relative tolerance for deduplicating overlapping times (default 1e-6).

    Examples
    --------
    >>> msr = MultiSegmentReader([
    ...     BinaryReader('field', folder, ext, params.get(fn))
    ...     for fn, ext in enumerate(runs)
    ... ])
    >>> times = msr.read_all_times()
    >>> for t, arrays in msr.stream_selected([0, 10, 50]):
    ...     phi = arrays[0]
    """

    def __init__(self, readers: list, tol: float = 1e-6):
        if not readers:
            raise ValueError("MultiSegmentReader requires at least one reader.")
        self.readers  = readers
        self.tol      = tol
        self.ni       = readers[0].ni
        self.nj       = readers[0].nj
        self.nk       = readers[0].nk
        self.n_arrays = readers[0].n_arrays
        self._global_times = None
        self._global_map   = None

    def _build_timeline(self) -> None:
        """Scan all segments, deduplicate overlapping times (later wins), sort."""
        # Collect all entries into NumPy arrays for fast sort/dedup
        all_times = []
        all_segs = []
        all_iters = []
        for seg_idx, reader in enumerate(self.readers):
            t_arr = reader.read_all_times()
            n = len(t_arr)
            all_times.append(t_arr.astype(np.float64))
            all_segs.append(np.full(n, seg_idx, dtype=np.int32))
            all_iters.append(np.arange(n, dtype=np.int32))

        times = np.concatenate(all_times)
        segs = np.concatenate(all_segs)
        iters = np.concatenate(all_iters)

        # Sort by time first, then by segment (later segment = higher index)
        order = np.lexsort((segs, times))
        times = times[order]
        segs = segs[order]
        iters = iters[order]

        # Remove duplicates — keep entry from later segment
        n = len(times)
        if n == 0:
            self._global_times = np.array([], dtype=np.float64)
            self._global_map = []
            return

        keep = np.ones(n, dtype=bool)
        tol_abs = self.tol * np.maximum(1.0, np.abs(times))

        # Compare adjacent entries
        if n > 1:
            dt = np.abs(np.diff(times))
            same = dt <= tol_abs[1:]
            for i in range(n - 1, 0, -1):
                if same[i - 1]:
                    if segs[i] >= segs[i - 1]:
                        keep[i - 1] = False
                    else:
                        keep[i] = False

        self._global_times = times[keep]
        kept_segs = segs[keep]
        kept_iters = iters[keep]
        self._global_map = list(zip(kept_segs.tolist(), kept_iters.tolist()))

    def _ensure_timeline(self) -> None:
        if self._global_times is None:
            self._build_timeline()

    def read_all_times(self) -> np.ndarray:
        """
        Return the merged, deduplicated, sorted time array across all segments.

        Returns
        -------
        np.ndarray
            Shape ``(n_total_iters,)``.
        """
        self._ensure_timeline()
        return self._global_times.copy()

    def segment_of(self, global_idx: int) -> int:
        """Return the segment index that owns *global_idx*."""
        self._ensure_timeline()
        return self._global_map[global_idx][0]

    def stream_selected(self, global_indices):
        """
        Yield ``(time, arrays)`` for the requested global indices.

        Routes each index to the correct segment reader automatically.
        Arrays have the native shape and dtype of their source segment.

        Memory-efficient: buffers at most one segment's worth of results
        at a time rather than accumulating all results before yielding.

        Parameters
        ----------
        global_indices : sequence of int
            Indices into the merged timeline from :meth:`read_all_times`.

        Yields
        ------
        time : float
        arrays : list of np.ndarray
        """
        self._ensure_timeline()
        yield from self._stream_with_seg(global_indices, include_seg=False)

    def stream_selected_with_seg(self, global_indices):
        """
        Like :meth:`stream_selected` but also yields the segment index.

        Yields
        ------
        time : float
        arrays : list of np.ndarray
        seg_idx : int
            Index into the original readers list that produced this step.
        """
        self._ensure_timeline()
        yield from self._stream_with_seg(global_indices, include_seg=True)

    def _stream_with_seg(self, global_indices, include_seg: bool):
        """
        Core streaming implementation.

        Reads each segment file exactly once, in segment order.
        Yields results in the original *global_indices* order using a
        minimal reorder buffer — at most one extra copy per segment
        boundary rather than holding all snapshots simultaneously.
        """
        from collections import defaultdict

        global_indices = list(global_indices)
        if not global_indices:
            return

        # Map each global index to its (seg_idx, local_iter)
        # and build a position map for output ordering
        request_pos = {g_idx: pos for pos, g_idx in enumerate(global_indices)}

        # Group by segment, preserving local iteration order within each
        seg_requests = defaultdict(list)
        for g_idx in global_indices:
            seg_idx, local_iter = self._global_map[g_idx]
            seg_requests[seg_idx].append((g_idx, local_iter))

        # Read segments in ascending order (each file opened once)
        # Buffer results only when needed for reordering
        buffer = {}          # g_idx -> (t, arrays[, seg_idx])
        next_to_yield = 0    # position in global_indices to yield next

        for seg_idx in sorted(seg_requests.keys()):
            pairs       = sorted(seg_requests[seg_idx], key=lambda x: x[1])
            local_iters = [li for _, li in pairs]
            g_idxs      = [gi for gi, _ in pairs]
            reader      = self.readers[seg_idx]

            for (t, arrays), g_idx in zip(
                    reader.stream_selected(local_iters), g_idxs):
                entry = (t, arrays, seg_idx) if include_seg else (t, arrays)
                buffer[g_idx] = entry

                # Yield as many consecutive results as are ready
                while (next_to_yield < len(global_indices) and
                       global_indices[next_to_yield] in buffer):
                    yield buffer.pop(global_indices[next_to_yield])
                    next_to_yield += 1

        # Flush any remaining buffered results (should be empty for sorted input)
        while next_to_yield < len(global_indices):
            g_idx = global_indices[next_to_yield]
            if g_idx in buffer:
                yield buffer.pop(g_idx)
            next_to_yield += 1

    def __repr__(self) -> str:
        self._ensure_timeline()
        return (
            f"MultiSegmentReader("
            f"{len(self.readers)} segments, "
            f"{len(self._global_times)} unique steps, "
            f"t=[{self._global_times[0]:.3f}, {self._global_times[-1]:.3f}])"
        )
