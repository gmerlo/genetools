# genetools

Post-processing toolkit for [GENE](http://genecode.org) gyrokinetic plasma simulations.

`genetools` provides Python readers and plotters for the binary, HDF5 and ADIOS2 output files produced by the GENE code — including [GENE-3D](https://gitlab.mpcdf.mpg.de/GENE/gene-3d), which is real-space in both perpendicular directions and writes HDF5 only. Loading, inspecting and visualising simulation results is a one-liner from a Jupyter notebook or the command line.

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

# GENE-3D
genetools /path/to/run3d --fluxes2d --si
genetools /path/to/run3d --slices --quantities phi n --fourier y
genetools /path/to/run3d --chi --t 200 800
```

---

## Diagnostics

| Accessor / flag | What it computes |
|---|---|
| `run.nrg` · `--nrg` | Energy/flux time traces (`nrg*`) |
| `run.spectra` · `--spectra` | Time-averaged flux spectra (auto local kx/ky or global) |
| `run.profiles` · `--profiles` | Flux-surface-averaged radial profiles |
| `run.fluxes2d` · `--fluxes2d` | x-resolved transport fluxes (Γ, Q, Π; ES/EM) — time-averaged profiles in gyro-Bohm *and* SI, plus the (x, t) map |
| `run.shearing` · `--shearing` | ExB shearing rate / zonal Eᵣ |
| `run.contours` · `--contours` | 2-D field/moment slices — `xy` at z=0 and `xz` at y=0 |
| `run.ballooning(ky=…)` · `--ballooning` | Field-line (ballooning) mode structure |
| `run.growthrate` · `--growthrate` | Linear γ/ω from the field time evolution |
| `run.amplitude` · `--amplitude` | kx/ky amplitude spectra of fields & moments |
| `run.zonal` · `--zonal` | Zonal (ky=0) potential x-t contour |
| `run.profile_diag` · `--profile-diag` | GENE `profile_<species>` radial profiles + turbulent/neoclassical fluxes + bootstrap current (global nonlinear) |

Each diagnostic exposes a uniform surface: `.data` (an `xarray.Dataset`), `.plot(t=(start, stop))`, and (where caching applies) `.save()`.

### GENE-3D

`Run` reports which geometry a run uses via `run.geometry_kind` — one of `flux_tube`, `x_global`, `y_global` or `xy_global` — and `run.is_3d` is true for the last.

There is **one class per diagnostic**, each handling every geometry it supports internally, so every accessor above keeps working for a GENE-3D run and `run.spectra.plot()` means the same thing whatever the run is:

```python
run.spectra          # -> Spectra, for a flux tube, an x-global run or GENE-3D
run.spectra.geometry_kind
```

These have no spectral counterpart and are GENE-3D only:

| Accessor / flag | What it computes |
|---|---|
| `run.slices(quantities=…)` · `--slices` | Every 1-D and 2-D reduction of a snapshot, optionally in Fourier space |
| `run.timetraces(quantities=…)` · `--timetraces` | Volume-averaged and ky-resolved time traces |
| `run.gam` · `--gam` | Zonal-flow / GAM oscillation, with fitted frequency and damping |
| `run.chi` · `--chi` | Heat diffusivity χ against the self-consistent driving gradient |
| `run.omega` · `--omega` | Real frequency and the power spectrum of the de-trended signal |
| `run.geometry_plots` · `--geometry` | 3-D geometry coefficients along cuts and planes |
| `run.srcmom` · `--srcmom` | Krook heat/particle source moments (needs `istep_srcmom`) |
| `run.vsp` · `--vsp` | Velocity-space output on the (z, v∥, μ) grid |
| `run.planes(quantities=…)` · `--planes` | Remapped onto geometric (θ, φ) angles, with (n, m) mode analysis |
| `run.vis3d(quantities=…)` · `--vis3d` | VTK export for ParaView/VisIt |

These GENE-3D-only diagnostics declare which geometries they support and refuse on construction otherwise, rather than quietly reducing the data some other way:

```python
>>> Run("/path/to/flux_tube/run").gam
NotImplementedError: Gam supports xy_global; this run is 'flux_tube'.
```

`run.ballooning(...)` likewise raises for a GENE-3D run: there is no single `ky` mode to follow when `y` is real space.

Because GENE-3D computes its own turbulent fluxes and writes them to the moment
file, `run.spectra` cross-checks every reconstructed spectrum against the code's
own `Gamma_es`/`Q_es`/`Gamma_em`/`Q_em` and warns if the ky-sum disagrees:

```python
run = Run("/path/to/gene3d/run")
run.spectra.compute()
run.spectra.consistency   # {'ions/Q_es': 1.0000, ...} — 1.0 means they agree
```

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
ds.Q_es_ky.sel(species="ions").plot()  # label-based selection, auto-axes
# global runs: ds.Qes_xky is the (x, ky) map; ds.Qes_x / ds.Qes_ky its
# 1-D reductions (ky-sum / radial mean). Same layout for run.amplitude.

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
is a thin layer over them). Modules live under `genetools.io` and
`genetools.diagnostics`; all major classes are also re-exported flat
(`from genetools import Params, NrgReader, BinaryReader, ...`).

### Load parameters

```python
from genetools.io import Params

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
from genetools.diagnostics import NrgReader

reader = NrgReader('/path/to/run/', params)
times, data = reader.read_all()
# data.shape → (n_species, n_cols, n_times)

reader.plot()              # shows flux and fluctuation plots
reader.plot_fluxes()       # heat and particle flux only
reader.plot_fluctuations() # n, T_∥, T_⊥, u_∥ only
```

### Stream field/moment data

```python
from genetools.io import BinaryReader, set_runs

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

### HDF5 files (`write_h5`, and all GENE-3D field/moment data)

```python
from genetools.io import H5Reader, Params

params = Params('/path/to/run/', ['.dat']).get(0)

# Note the '.h5' on the extension: GENE writes field.dat.h5, mom_ions.dat.h5, ...
reader = H5Reader('field', '/path/to/run/', '.dat.h5', params)

# Variables are discovered from the file, not from n_moms — GENE-3D's
# parameters file reports 6 moments while diag_3d writes 10.
reader.var_names                     # ['phi', 'A_par', ...]
i = reader.index_of('Q_es')          # read a moment by name

for t, arrays in reader.stream_selected([0, 5, 10]):
    phi = arrays[0]                  # (nx, ny, nz) real for GENE-3D,
                                     # (nx, nky, nz) complex for GENE
```

Arrays come back in GENE's `(ni, nj, nk)` order and with the dtype the file
actually holds: futils stores 32-bit reals for field/moment data regardless of
the run's `PRECISION`, and GENE's complex data may be a `{real, imaginary}`
compound type. All of that is handled by the reader.

### ADIOS2 BP files

```python
from genetools.io import BPReader

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
