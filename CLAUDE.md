# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Post-processing toolkit for [GENE](http://genecode.org) gyrokinetic plasma simulations. Reads binary/ADIOS2 output files and provides diagnostics and visualization.

## Commands

```bash
# Install (editable)
pip install -e .
pip install -e ".[dev]"       # with pytest

# Run tests (test file is at root, not in tests/)
pytest test_genetools.py -v
pytest test_genetools.py -v -k "TestParams"              # single test class
pytest test_genetools.py -v -k "test_defaults_applied"   # single test

# Coverage
pytest test_genetools.py -v --cov=genetools --cov-report=term-missing
```

Note: `pyproject.toml` sets `testpaths = ["tests"]` but the actual test file is `test_genetools.py` at the repo root. Pass the file path explicitly to pytest.

## Architecture

Two subpackages under the root `genetools` package:

### `io/` — Data loading
- **`params.py`** — `Params` class: parses Fortran-90 namelist parameter files, applies physics defaults (`_DEFAULTS`), computes derived units. Supports multi-segment runs.
- **`data.py`** — `_BaseReader` ABC with `BinaryReader` (Fortran unformatted) and `BPReader` (ADIOS2, optional). `MultiSegmentReader` stitches segments transparently. Key interface: `read_all_times()` and `stream_selected(indices)`.
- **`geometry.py`** — `Geometry()` function: loads local (spectral) or global (real-space) geometry. Returns dict with metric, curvature, Jacobian, etc.
- **`coordinates.py`** — `Coordinates()` function: builds kx, ky, z, vp, mu arrays from params.
- **`utils.py`** — `set_runs()`: discovers output segment suffixes by scanning for `nrg*` files.
- **`profiles_loader.py`** — `load_equilibrium_profiles()`: loads external equilibrium profiles from HDF5.

### `diagnostics/` — Physics computations and plotting
All diagnostics follow a common pattern: stream data from readers, compute physics quantities, cache results to HDF5, provide `plot()` methods.

- **`nrg.py`** — `NrgReader`: energy/flux time traces (no HDF5 caching, reads nrg binary directly)
- **`spectra.py`** — `Spectra`: time-averaged kx/ky/z-resolved flux spectra (local geometry)
- **`spectra_global.py`** — `SpectraGlobal`: ky-resolved flux spectra Q(x,ky) for global runs
- **`contours.py`** — `Contours`: 2D field visualizations with memory-efficient IFFT (slice-before-transform)
- **`shearingrate.py`** — `ShearingRate`: ExB shearing rate from zonal potential
- **`profiles.py`** — `Profiles`: flux-surface-averaged radial profiles
- **`fluxes2d.py`** — `Fluxes2D`: x-resolved transport fluxes (particle, heat, momentum)

### Data flow
1. `set_runs(folder)` → segment suffixes
2. `Params(folder, extensions)` → parameter dicts
3. `Geometry(...)` + `Coordinates(...)` → geometry/coordinate arrays
4. `BinaryReader`/`BPReader`/`MultiSegmentReader` → stream field/moment data
5. Diagnostics consume readers → compute → cache to HDF5 → plot

### `__init__.py` flat re-exports
The root `__init__.py` re-exports all major classes for backward compatibility (`from genetools import Params, BinaryReader, ...`).

## Key conventions

- **Local vs global geometry**: `x_local=True` uses spectral (kx) space; `x_local=False` uses real-space radial grid. Most diagnostics branch on this.
- **Hermitian symmetry**: only positive ky modes are stored; ky=0 is unweighted, ky>0 weighted by factor 2 in summations.
- **numpy compat**: code handles both `np.trapz` (numpy <2.0) and `np.trapezoid` (numpy >=2.0).
- **Optional deps**: `adios2` for BP files, `numba` for JIT acceleration in spectra. Both use try/except import guards.
- **Temperature**: T = (1/3)T_par + (2/3)T_perp throughout diagnostics.
