import numpy as np
import mmap
try:                       
    import adios2
    ADIOS2_AVAILABLE = True
except ImportError:
    ADIOS2_AVAILABLE = False

    
class BinaryReader:
    """
    Generic streaming reader for Fortran unformatted files.
    Supports arbitrary number of complex arrays per iteration.
    """

    def __init__(self, file_type, folder, ext, params, species=None):
        """
        folder      : folder containing the files
        file_type   : 'field' or 'mom'
        ext         : string suffix, e.g. '_0000'
        params      : dictionary with simulation parameters
        species     : optional, for moments files, name of the species
        """
        # --- Construct filename ---
        species_str = f"_{species}" if species else ""
        self.filename = f"{folder}{file_type}{species_str}{ext}"

        # --- Extract dimensions ---
        box = params['box']
        info = params['info']
        if params['general']['y_local']:
            self.ni = box.get('nx0', 1)
            self.nj = box.get('nky0', 1)
            self.nk = box.get('nz0', 1)
            
        self.n_arrays = info.get('n_fields' if file_type == 'field' else 'n_moms', 1)

        # --- Set precision ---
        precision = info.get('precision', 'double').lower()
        if precision == 'single':
            self.real_dtype = np.float32
            self.cpx_dtype = np.complex64
        else:
            self.real_dtype = np.float64
            self.cpx_dtype = np.complex128

        self.npts = self.ni * self.nj * self.nk


    # -------------------------------------------------------------

    def _build_record_index(self, mm):
        """
        Scan Fortran unformatted file and return list of
        (payload_start, payload_size) for all records.
        """
        index = []
        pos = 0
        size = mm.size()

        while pos < size:
            nbytes = int(np.frombuffer(mm[pos:pos+4], dtype=np.int32)[0])
            start = pos + 4
            end = start + nbytes
            index.append((start, nbytes))
            pos = end + 4

        return index

    # -------------------------------------------------------------

    def read_all_times(self):
        """
        FAST PASS 1: read only the scalar value from every iteration.
        Returns array of shape (n_iters,)
        """
        with open(self.filename, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            idx = self._build_record_index(mm)

            records_per_iter = 1 + self.n_arrays
            n_iters = len(idx) // records_per_iter

            times = np.zeros(n_iters, dtype=self.real_dtype)

            for it in range(n_iters):
                start, nbytes = idx[it * records_per_iter]
                times[it] = np.frombuffer(mm[start:start+nbytes], dtype=self.real_dtype)[0]

            mm.close()

        return times

    # -------------------------------------------------------------

    def stream_selected(self, iteration_indices):
        """
        Stream ONLY selected iteration numbers.
        Yields (scalar, arrays)
        """
        with open(self.filename, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            idx = self._build_record_index(mm)
            records_per_iter = 1 + self.n_arrays

            for it in iteration_indices:
                base = it * records_per_iter

                # Scalar record
                start, nbytes = idx[base]
                time = np.frombuffer(mm[start:start+nbytes], dtype=self.real_dtype)[0]

                # Complex arrays
                data = []
                for k in range(self.n_arrays):
                    startA, nbytesA = idx[base + 1 + k]
                    arr = np.frombuffer(mm[startA:startA+nbytesA],
                                        dtype=self.cpx_dtype,
                                        count=self.npts)
                    arr = arr.reshape((self.ni, self.nj, self.nk), order='F')
                    data.append(arr)

                yield time, data

            mm.close()

if ADIOS2_AVAILABLE:
    class BPReader:
        """
        Lazy streaming reader for ADIOS2 BP files.
        Supports arbitrary number of complex arrays per iteration, like BinaryReader.
        """

        def __init__(self, file_type, folder, ext, params, species=None):
            """
            folder      : folder containing the file
            file_type   : 'field' or 'mom'
            ext         : string suffix, e.g. '_0000.bp'
            params      : dictionary with simulation parameters
            species     : optional, for moments files, name of the species
            """
            species_str = f"_{species}" if species else ""
            self.filename = f"{folder}{file_type}{species_str}{ext}"

            # --- Extract dimensions ---
            box = params['box']
            info = params['info']
            self.ni = box.get('nx0', 1)
            self.nj = box.get('nky0', 1)
            self.nk = box.get('nz0', 1)

            self.n_arrays = info.get('n_fields' if file_type=='field' else 'n_moms', 1)

            precision = info.get('precision', 'double').lower()
            if precision == 'single':
                self.real_dtype = np.float32
                self.cpx_dtype = np.complex64
            else:
                self.real_dtype = np.float64
                self.cpx_dtype = np.complex128

            self.npts = self.ni*self.nj*self.nk

        # -------------------------------------------------------------
        def read_all_times(self):
            """
            Read all 'time' variables from the BP file.
            Returns numpy array of times.
            """
            times = []
            with adios2.open(self.filename, "r") as fh:
                for step in fh:
                    t = step.read("time", np_type=self.real_dtype)
                    times.append(t)
            return np.array(times, dtype=self.real_dtype)

        # -------------------------------------------------------------
        def stream_selected(self, iteration_indices):
            """
            Stream ONLY the selected iteration numbers.
            Yields (scalar time, list of complex arrays)
            """
            iter_set = set(iteration_indices)
            with adios2.open(self.filename, "r") as fh:
                for step_idx, step in enumerate(fh):
                    if step_idx not in iter_set:
                        continue

                    # --- Read time ---
                    time = step.read("time", np_type=self.real_dtype)

                    # --- Read arrays ---
                    data = []
                    for k in range(self.n_arrays):
                        arr = step.read(f"array_{k}", np_type=self.cpx_dtype)
                        arr = arr.reshape((self.ni,self.nj,self.nk), order='F')
                        data.append(arr)

                    yield time, data
else:
    raise ImportError("Need ADIOS@ python binding to use adios2 files")