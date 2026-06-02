# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Post-processing toolkit for [GENE](http://genecode.org) gyrokinetic plasma simulations. Reads binary/ADIOS2 output files and provides diagnostics and visualization.

## Commands

```bash
# Install (editable)
pip install -e .
pip install -e ".[dev]"       # with pytest

# Run tests (suite lives under tests/)
pytest tests/ -v
pytest tests/io/test_params.py -v -k "test_defaults_applied"   # single test
pytest tests/diagnostics -v                                    # diagnostics only

# Coverage
pytest tests/ -v --cov=genetools --cov-report=term-missing
```

Note: tests are under `tests/` (matching `testpaths = ["tests"]` in `pyproject.toml`). A `genetools` console-script CLI is installed by the editable install.

## Architecture

### High-level layer (the recommended entry point)
- **`run.py`** — `Run` facade: `Run(path, ext=None)` auto-wires `set_runs` → `Params` → `Geometry`/`Coordinates` and lazily builds multi-segment readers. Exposes every diagnostic as a lazy attribute (`run.nrg`, `run.spectra`, `run.profiles`, `run.fluxes2d`, `run.shearing`, `run.contours`, `run.growthrate`, `run.amplitude`, `run.zonal`) or callable (`run.ballooning(ky=...)`). `run.spectra` auto-dispatches local (`Spectra`) vs global (`SpectraGlobal`). Validates grid consistency across continuation segments (warns on mismatch; assumes same grid). Each bound diagnostic offers `.data` (xarray), `.plot(t=(start,stop))`, `.save()`.
- **`_xr.py`** — adapter that wraps the existing diagnostics' numpy-dict outputs into labelled `xarray.Dataset` objects (species-dim stacking, length-matched coord attachment, unit attrs from the `units` block). Phase-1 adapter; existing diagnostics still compute in numpy.
- **`cli.py`** — flag-style CLI (`genetools /run --spectra --t 500 2000 [--save fig.png] [--no-show]`), one flag per diagnostic. Entry point: `genetools = genetools.cli:main`.

The repo root **is** the `genetools` package (so `run.py` → `genetools.run`, etc.). `xarray` is a hard dependency; the data layer returns `xarray.Dataset`s.

Two subpackages under the root `genetools` package:

### `io/` — Data loading
- **`params.py`** — `Params` class: parses Fortran-90 namelist parameter files, applies physics defaults (`_DEFAULTS`), computes derived units. Supports multi-segment runs.
- **`data.py`** — `_BaseReader` ABC with `BinaryReader` (Fortran unformatted) and `BPReader` (ADIOS2, optional). `MultiSegmentReader` stitches segments transparently. Key interface: `read_all_times()` and `stream_selected(indices)`.
- **`geometry.py`** — `Geometry()` function: loads local (spectral) or global (real-space) geometry. Returns dict with metric, curvature, Jacobian, etc.
- **`coordinates.py`** — `Coordinates()` function: builds kx, ky, z, vp, mu arrays from params.
- **`utils.py`** — `set_runs()`: discovers output segment suffixes by scanning for `nrg*` files.
- **`profiles_loader.py`** — `load_equilibrium_profiles()`: loads external equilibrium profiles from HDF5.
- **`omega.py`** — `read_omega()` / `read_eigenvalues()`: parse GENE linear `omega<ext>` / `eigenvalues.dat` files (used as an optional cross-check by the growth-rate diagnostic).

### `diagnostics/` — Physics computations and plotting
All diagnostics follow a common pattern: stream data from readers, compute physics quantities, cache results to HDF5, provide `plot()` methods.

- **`nrg.py`** — `NrgReader`: energy/flux time traces (no HDF5 caching, reads nrg binary directly)
- **`spectra.py`** — `Spectra`: time-averaged kx/ky/z-resolved flux spectra (local geometry)
- **`spectra_global.py`** — `SpectraGlobal`: ky-resolved flux spectra Q(x,ky) for global runs
- **`contours.py`** — `Contours`: 2D field visualizations with memory-efficient IFFT (slice-before-transform)
- **`shearingrate.py`** — `ShearingRate`: ExB shearing rate from zonal potential
- **`profiles.py`** — `Profiles`: flux-surface-averaged radial profiles
- **`fluxes2d.py`** — `Fluxes2D`: x-resolved transport fluxes (particle, heat, momentum)
- **`ballooning.py`** — `Ballooning`: field-line (ballooning) mode structure φ/A∥/B∥(χ) for a chosen ky (local runs only); Run/xarray-native
- **`growthrate.py`** — `GrowthRate`: linear γ/ω from the field time evolution (γ from |φ| growth, ω from phase rotation), optional `omega<ext>` cross-check; Run/xarray-native
- **`amplitude.py`** — `AmplitudeSpectra`: time-averaged kx/ky |·|² spectra of fields and moments; reuses `Spectra.averages`; Run/xarray-native
- **`zonal.py`** — `Zonal`: zonal (ky=0) potential x-t contour; reuses `shearingrate.compute_exb`; Run/xarray-native

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
