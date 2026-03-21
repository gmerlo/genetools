# genetools

Post-processing toolkit for [GENE](http://genecode.org) gyrokinetic plasma simulations.

`genetools` provides Python readers and plotters for the binary and ADIOS2 output files produced by the GENE code, making it straightforward to load, inspect and visualise simulation results from a Jupyter notebook or script.

---

## Features

| Module | What it does |
|---|---|
| `params` | Parse GENE Fortran-90 namelist `parameters` files with physics defaults applied |
| `data` | Stream field and moment data from Fortran binary (`.dat`, `_NNNN`) or ADIOS2 BP files |
| `nrg` | Read and plot energy/flux diagnostic files (`nrg*`) |
| `utils` | Scan run directories and discover available output segments |
| `geometry` | Parse local and global geometry files |
| `spectra` | Compute and store time-averaged flux spectra as HDF5 |
| `coordinates` | Build coordinate arrays (kx, ky, z) |
| `contours` | 2-D contour visualisations |

---

## Installation

```bash
git clone https://github.com/gmerlo/genetools.git
cd genetools
pip install -e .
```

**Dependencies** (installed automatically):

- `numpy >= 1.21`
- `matplotlib >= 3.4`
- `f90nml >= 1.4`
- `h5py >= 3.0`

**Optional** — ADIOS2 BP file support:

```bash
pip install adios2
```

---

## Quick start

### Load parameters

```python
from genetools.params import Params

# Load from a run directory (reads 'parameters' by default)
p = Params('/path/to/run/')
params = p.get(0)          # dict for the first (only) file

# Load multiple restart segments
p = Params('/path/to/run/', extensions=['_0001', '_0002'])
p0, p1 = p.tolist()

# Pretty-print everything
p.show()
```

### Read and plot nrg diagnostics

```python
from genetools.nrg import NrgReader

reader = NrgReader('/path/to/run/', params)
times, data = reader.read_all()
# data.shape → (n_species, n_cols, n_times)

reader.plot()              # shows flux and fluctuation plots
reader.plot_fluxes()       # heat and particle flux only
reader.plot_fluctuations() # n, T_∥, T_⊥, u_∥ only
```

### Stream field/moment data

```python
from genetools.data import BinaryReader
from genetools.utils import set_runs

# Discover available run segments
suffixes = set_runs('/path/to/run/')  # e.g. ['_0001', '_0002', '.dat']

# Create a reader for the first segment
reader = BinaryReader('field', '/path/to/run/', suffixes[0], params)

# Fast scan of all times (no array data loaded)
times = reader.read_all_times()

# Stream only selected iterations (memory-efficient)
for t, arrays in reader.stream_selected([0, 50, 100]):
    phi = arrays[0]   # shape: (nx, nky, nz)
    print(f"t={t:.3f}, max(|φ|)={abs(phi).max():.4e}")
```

### ADIOS2 BP files

```python
from genetools.data import BPReader

reader = BPReader('field', '/path/to/run/', '_0001.bp', params)
times = reader.read_all_times()
for t, arrays in reader.stream_selected([0, 10]):
    phi = arrays[0]
```

---

## Running the tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
# With coverage report:
pytest tests/ -v --cov=genetools --cov-report=term-missing
```

---

## Parameter file format

GENE writes simulation parameters in Fortran-90 namelist format:

```fortran
&general
  x_local = .true.
  y_local = .true.
/

&box
  nx0 = 64
  nky0 = 16
  nz0 = 32
  n_spec = 2
/

&species
  name = 'ions'
  charge = 1.0
  mass   = 1.0
  temp   = 1.0
/
```

`Params` reads these files, applies defaults for any missing keys (see `params.Params._DEFAULTS`), and computes derived unit quantities (`cref`, `rhoref`, `rho_starref`, etc.) from the `[units]` block.

---

## Contributing

Pull requests and issue reports are welcome. Please run the test suite before submitting.

---

## License

This project is research software. See `LICENSE` for details.
