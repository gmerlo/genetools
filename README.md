# genetools

Post-processing toolkit for [GENE](http://genecode.org) gyrokinetic plasma simulations.

`genetools` provides Python readers and plotters for the binary and ADIOS2 output files produced by the GENE code, making it straightforward to load, inspect and visualise simulation results from a Jupyter notebook or the command line.

The recommended entry point is the **`Run` facade**: point it at a run directory and every diagnostic is one line away, returning labelled [`xarray`](https://docs.xarray.dev) datasets and plots. Continuation/restart segments are discovered and stitched automatically.

```python
from genetools import Run

run = Run("/path/to/run")        # discovers segments, params, geometry, coords
run.nrg.plot()                   # energy/flux time traces
run.spectra.plot(t=(500, 2000))  # flux spectra over a time window
ds = run.spectra.data            # -> xarray.Dataset (labelled dims/coords/units)
run.ballooning(ky=0.3).plot()    # ballooning mode structure
```

Or from the command line:

```bash
genetools /path/to/run --nrg
genetools /path/to/run --spectra --t 500 2000 --save spectra.png
genetools /path/to/run --growthrate
```

---

## Diagnostics

| Accessor / flag | What it computes |
|---|---|
| `run.nrg` · `--nrg` | Energy/flux time traces (`nrg*`) |
| `run.spectra` · `--spectra` | Time-averaged flux spectra (auto local kx/ky or global) |
| `run.profiles` · `--profiles` | Flux-surface-averaged radial profiles |
| `run.fluxes2d` · `--fluxes2d` | x-resolved transport fluxes (Γ, Q, Π; ES/EM) |
| `run.shearing` · `--shearing` | ExB shearing rate / zonal Eᵣ |
| `run.contours` · `--contours` | 2-D field/moment slices |
| `run.ballooning(ky=…)` · `--ballooning` | Field-line (ballooning) mode structure |
| `run.growthrate` · `--growthrate` | Linear γ/ω from the field time evolution |
| `run.amplitude` · `--amplitude` | kx/ky amplitude spectra of fields & moments |
| `run.zonal` · `--zonal` | Zonal (ky=0) potential x-t contour |

Each diagnostic exposes a uniform surface: `.data` (an `xarray.Dataset`), `.plot(t=(start, stop))`, and (where caching applies) `.save()`.

---

## Installation

```bash
git clone https://github.com/gmerlo/genetools.git
cd genetools
pip install -e .
```

This installs a `genetools` command-line entry point.

**Dependencies** (installed automatically):

- `numpy >= 1.21`
- `matplotlib >= 3.4`
- `f90nml >= 1.4`
- `h5py >= 3.0`
- `xarray >= 2022.3`

**Optional** — ADIOS2 BP file support:

```bash
pip install adios2
```

---

## Quick start — the `Run` facade

```python
from genetools import Run

run = Run("/path/to/run")            # ext=None -> all segments; or ext=["_0002", ".dat"]
run.species                          # ['ions', 'electrons']
run.is_local                         # True for flux-tube (spectral-x) runs

# Every diagnostic: .data (xarray) + .plot(t=...) + .save()
ds = run.profiles.data               # xarray.Dataset, dims (species, time, x)
run.profiles.plot(t=(1000, 2000))

ds = run.spectra.data                # auto local kx/ky or global, by geometry
ds.Q_es.sel(species="ions").plot()   # label-based selection, auto-axes

ky, gamma, omega, window = run.growthrate.compute()   # linear stability
run.ballooning(ky=0.3).plot()        # mode structure along the field line
```

**Continuation runs** are handled automatically: all segments are discovered,
their timelines merged and de-duplicated (later segment wins), and the grid is
validated for consistency (a warning is raised if it changes across segments —
scope to a subset with `Run(path, ext=[...])`).

## Command line

```bash
genetools [PATH] --DIAG [options]

genetools /run --nrg
genetools /run --spectra --t 500 2000 --save spectra.png   # save instead of show
genetools /run --profiles --no-show
genetools /run --ballooning --ky 0.3
genetools /run --contours --field 0 --ifft xy
```

Common options: `--t START STOP`, `--species …`, `--ext …`, `--save FILE`
(a directory auto-names PNGs), `--no-show`.

---

## Low-level API

The original modules remain available for fine-grained control (the `Run` facade
is a thin layer over them).

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

Released under the [Mozilla Public License 2.0](LICENSE). © 2026 Gabriele Merlo.
