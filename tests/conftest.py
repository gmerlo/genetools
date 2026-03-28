"""
Shared test fixtures and helpers for genetools tests.
"""

import io
import struct
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def make_params(
    n_spec=1,
    n_fields=2,
    nrgcols=10,
    nx0=4,
    nky0=3,
    nz0=8,
    precision="double",
    y_local=True,
):
    """Build a minimal params dict that satisfies all readers."""
    return {
        "box": {"n_spec": n_spec, "nx0": nx0, "nky0": nky0, "nz0": nz0},
        "info": {"n_fields": n_fields, "n_moms": 9, "nrgcols": nrgcols, "precision": precision},
        "general": {"y_local": y_local, "x_local": True},
        "species": [{"name": f"sp{i}", "dens": 1.0, "temp": 1.0, "charge": 1.0}
                    for i in range(n_spec)],
    }


MINIMAL_PARAMS = """
&general
  x_local = .true.
  y_local = .true.
/

&box
  nx0 = 16
  nky0 = 4
  nz0 = 32
  n_spec = 2
/

&info
  precision = 'double'
  nrgcols = 10
/

&units
  Lref = 1.0
  Bref = 1.0
  nref = 1.0
  Tref = 1.0
  mref = 1.0
/

&species
  name = 'ions'
  charge = 1.0
  mass = 1.0
  temp = 1.0
/

&species
  name = 'electrons'
  charge = -1.0
  mass = 0.000272
  temp = 1.0
/
"""


def write_param_file(path: Path, content: str) -> None:
    path.write_text(content)


def write_fortran_record(buf: io.BytesIO, payload: bytes) -> None:
    """Write a Fortran unformatted record (marker + payload + marker)."""
    n = len(payload)
    marker = struct.pack("<i", n)
    buf.write(marker + payload + marker)


def make_binary_file(
    tmp_path: Path,
    n_iters: int = 3,
    ni: int = 2,
    nj: int = 2,
    nk: int = 2,
    n_arrays: int = 1,
    dtype=np.complex128,
    real_dtype=np.float64,
    filename: str = "field_0001",
) -> tuple:
    """Create a synthetic Fortran binary file and return (path, times, arrays)."""
    buf = io.BytesIO()
    times = []
    arrays = []
    npts = ni * nj * nk
    for it in range(n_iters):
        t = float(it) * 0.5
        times.append(t)
        write_fortran_record(buf, struct.pack("<d", t))
        it_arrays = []
        for _ in range(n_arrays):
            arr = np.random.randn(npts) + 1j * np.random.randn(npts)
            arr = arr.astype(dtype)
            write_fortran_record(buf, arr.tobytes())
            it_arrays.append(arr.reshape((ni, nj, nk), order="F"))
        arrays.append(it_arrays)
    fpath = tmp_path / filename
    fpath.write_bytes(buf.getvalue())
    return fpath, np.array(times), arrays


def write_nrg_file(path: Path, n_times: int, n_spec: int, n_cols: int) -> tuple:
    """Write a synthetic nrg file and return (times, data) arrays."""
    rng = np.random.default_rng(42)
    times = np.arange(n_times, dtype=float) * 0.1
    data = rng.random((n_times, n_spec, n_cols))
    lines = []
    for it in range(n_times):
        lines.append(f"{times[it]:.6e}")
        for sp in range(n_spec):
            lines.append("  ".join(f"{v:.6e}" for v in data[it, sp, :]))
    path.write_text("\n".join(lines) + "\n")
    # Reshape to expected output: (n_spec, n_cols, n_times)
    expected = np.transpose(data, (1, 2, 0))
    return times, expected
