# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Post-processing toolkit for [GENE](http://genecode.org) gyrokinetic plasma simulations, including GENE-3D. Reads Fortran-binary, HDF5 and ADIOS2 output and provides diagnostics and visualization.

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

```bash
pytest tests/ -k gene3d -v     # GENE-3D only
```

GENE-3D has no reference output in the repo; `tests/gene3d_fixture.py` builds
synthetic run directories reproducing what `gene3d-dev/src` writes (futils axis
order and precision, snapshot groups, the deliberately wrong `n_moms`).
`make_gene3d_run(..., physical=True)` derives the fluxes from the fields exactly
as `diag_3d.F90` does, which is what makes the flux reconstruction checkable
rather than merely plausible.

## Architecture

### High-level layer (the recommended entry point)
- **`run.py`** — `Run` facade: `Run(path, ext=None)` auto-wires `set_runs` → `Params` → `Geometry`/`Coordinates` and lazily builds multi-segment readers. Every diagnostic is a lazy attribute (`run.nrg`, `run.spectra`, `run.profiles`, `run.fluxes2d`, `run.shearing`, `run.contours`, `run.growthrate`, `run.amplitude`, `run.zonal`, `run.profile_diag`, `run.gam`, `run.chi`, `run.omega`, `run.geometry_plots`, `run.srcmom`, `run.vsp`) or a callable where it needs arguments (`run.ballooning(ky=...)`, `run.slices(...)`, `run.timetraces(...)`, `run.planes(...)`, `run.vis3d(...)`). Each accessor just constructs the one class for that diagnostic — the class handles the geometry, so `run.spectra.plot()` means the same thing whatever the run is. `run.geometry_kind` / `run.is_3d` report which geometry. Validates grid consistency across continuation segments (warns on mismatch; assumes same grid). Every diagnostic offers `.data` (xarray), `.plot(t=(start,stop))`, `.save()`.

  The one exception is `run.nrg`, which still goes through a thin `_BoundNrg` wrapper: `NrgReader` is a folder-based ASCII reader with no geometry branching, so it was left as-is.
- **`_xr.py`** — small explicit helpers for building labelled `xarray.Dataset`s (`stacked_vars` for species-dim stacking, `attach_coords` for length-matched coord attachment, `make_dataset`, `unit_attrs`, `split_species`). Each diagnostic owns its xarray construction: `NrgReader.dataset(params)` and `Spectra`/`SpectraGlobal`/`Profiles`/`Fluxes2D`/`ShearingRate.dataset(coords, params, species, …)` build their Datasets with explicit dims. The compute stays numpy; xarray is the return layer.
- **`cli.py`** — flag-style CLI (`genetools /run --spectra --t 500 2000 [--save fig.png] [--no-show]`), one flag per diagnostic. Entry point: `genetools = genetools.cli:main`.

The repo root **is** the `genetools` package (so `run.py` → `genetools.run`, etc.). `xarray` is a hard dependency; the data layer returns `xarray.Dataset`s.

Two subpackages under the root `genetools` package:

### `io/` — Data loading
- **`params.py`** — `Params` class: parses Fortran-90 namelist parameter files, applies physics defaults (`_DEFAULTS`), computes derived units. Supports multi-segment runs. `_normalise()` reconciles keys the two codes write in different namelists (GENE puts `x_local`/`y_local` in `&general` and `write_h5` in `&in_out`; GENE-3D hard-codes all three into `&info`) and records `info['geometry_kind']` / `info['is_3d']`. `Params.geometry_kind()` / `Params.is_3d()` expose it.
- **`data.py`** — `_BaseReader` ABC with `BinaryReader` (Fortran unformatted), `H5Reader` (GENE `write_h5` and all GENE-3D field/moment data) and `BPReader` (ADIOS2, optional). `MultiSegmentReader` stitches segments transparently. Key interface: `read_all_times()`, `stream_selected(indices)`, plus `var_names` / `index_of(name)` for reading a variable by name. Canonical variable orders live in `FIELD_VARS` / `MOM_VARS` / `MOM_VARS_3D` / `canonical_vars()`.
- **`geometry.py`** — `Geometry()` function: loads local (spectral), global (real-space) or GENE-3D 3-D geometry, from ASCII or HDF5. Returns dict with metric, curvature, Jacobian, `area`, and for GENE-3D also `profiles` (q, dVdx, sqrtgxx_fs) and `cart_coords`.
- **`coordinates.py`** — `Coordinates()` function: builds kx, ky, z, vp, mu arrays from params. GENE-3D takes `load_coord_xy_global()`, which prefers the grids in `coord<ext>.h5` and falls back to reconstructing them.
- **`utils.py`** — `set_runs()`: discovers output segment suffixes by scanning for `nrg*` files.
- **`profiles_loader.py`** — `load_equilibrium_profiles()`: loads per-species equilibrium profiles from `profiles_<species><ext>` (ASCII, columns `x/a, x/rho_ref, T, n, omt, omn` — in that order) or its `.h5` twin. GENE-3D appends a block per profile update; the last block wins.
- **`omega.py`** — `read_omega()` / `read_eigenvalues()`: parse GENE linear `omega<ext>` / `eigenvalues.dat` files (used as an optional cross-check by the growth-rate diagnostic).

### `diagnostics/` — Physics computations and plotting

**One class per diagnostic, handling every geometry internally.** There is no
per-geometry subpackage: a diagnostic branches on `run.geometry_kind`
(`flux_tube` / `x_global` / `y_global` / `xy_global`) inside its own module, so
there is one place to look for "how is this computed" and one entry point on
`Run`. Diagnostics with no meaning outside GENE-3D set `supported =
("xy_global",)` and the base class refuses on construction.

Shared machinery lives in two private modules — put anything reusable there
rather than re-implementing it in a diagnostic:

- **`_base.py`** — `CachingDiagnostic`: HDF5 persistence (`_load_saved_times`,
  `_select_window`), `_time_average`, and `_sync_field_mom_indices`, which pairs
  field with moment snapshots **by time value**. Use that helper; `istep_field`
  and `istep_mom` need not agree and positional pairing silently correlates a
  field snapshot with moments from another time.
  `RunDiagnostic`: the Run-native surface every diagnostic inherits — `run`,
  `params`/`coord`/`geom`/`geometry_kind`/`is_3d` shortcuts, `_window`/`_bounds`/
  `_key`/`_indices`, `_require`, and `.data`/`.save()`. Set `cache_file` on a
  subclass to make it HDF5-backed.
- **`_gene3d.py`** — GENE-3D-specific physics only: `flux_geomfac`,
  `exb_velocity_ky`, `flutter_velocity_ky`, `check_flux_consistency`,
  `to_ky`/`to_kx`, `flux_surface_average`, `xz_average`, `jacobian_yz`,
  `radial_slice`.

Diagnostics available for every geometry:

- **`nrg.py`** — `NrgReader`: energy/flux time traces (reads `nrg` directly)
- **`spectra.py`** — `Spectra`: flux spectra. Flux tube = kx/ky/z spectra with Hermitian ky weighting; x-global delegates to `spectra_global.SpectraGlobal` (its HDF5 cache schema differs enough to keep separate, but `Spectra` owns the facade); GENE-3D rebuilds ky spectra from φ and the moments and cross-checks them against the fluxes the code wrote itself (`.consistency`)
- **`spectra_global.py`** — `SpectraGlobal`: the x-global `(x, ky)` implementation behind `Spectra`
- **`contours.py`** — `Contours`: 2-D slices; spectral paths stream/draw/discard (usable on field files larger than memory), GENE-3D reuses `slices.Slices` for its xy plane
- **`shearingrate.py`** — `ShearingRate`: zonal potential, v_ExB, ω_ExB. GENE-3D uses `C_xy` only — the `1/sqrt(g^xx)` of `flux_geomfac` belongs to a flux, not a flow
- **`profiles.py`** — `Profiles`: flux-surface-averaged radial profiles; GENE-3D builds the total profile as `T_0 + rhostar*minor_r*<T>_FS` (omitting that factor overstates the perturbation by `1/rhostar`) and offers `compare_with_code()` against `profile_<species>`
- **`fluxes2d.py`** — `Fluxes2D`: x-resolved fluxes. Spectral paths reconstruct from φ; GENE-3D reads its own `Gamma_*`/`Q_*` from the moment file, and the integration area follows `norm_flux_projection`. `plot()` draws three figures — gyro-Bohm profiles, SI profiles, and the `(x, t)` map. **Time-average with `_t_average` (trapezoidal), never `.mean("time")`**: GENE's dt is adaptive and output is every `istep_mom` *steps*, so output times are unevenly spaced and a plain mean is biased by tens of percent
- **`amplitude.py`** — `AmplitudeSpectra`: `|f|²` spectra. Flux tube reuses `Spectra.averages`; x-global builds `(x, ky)` maps; GENE-3D transforms **both** directions
- **`growthrate.py`** — `GrowthRate`: per-ky γ/ω from the complex amplitude (spectral, optional `omega<ext>` cross-check) or rescaling-aware γ from `max|φ|` (GENE-3D)
- **`zonal.py`** — `Zonal`: zonal potential x-t map; reuses `compute_exb` or `ShearingRate`
- **`profile_diag.py`** — `ProfileDiag`: `profile_<species>`. **13 columns for GENE, 8 for GENE-3D** — the column set is chosen from the geometry; plus `flux_profiles()`
- **`ballooning.py`** — `Ballooning`: field-line mode structure (needs a single ky, so it refuses for GENE-3D)

GENE-3D-only for now (each declares `supported = ("xy_global",)`):

- **`slices.py`** — `Slices`: every 1-D/2-D reduction, optionally in Fourier space
- **`timetraces.py`** — `TimeTraces`: volume-averaged and ky-resolved traces
- **`gam.py`** — `Gam`: zonal-flow oscillation; refuses when there is no zonal component above round-off rather than fitting a frequency to noise
- **`chi.py`** — `ChiGradient`: χ vs. the self-consistent driving gradient; reuses `Fluxes2D` and `Profiles` so all three agree on normalisation
- **`omega.py`** — `Omega`: frequency view over `GrowthRate`
- **`geometry_plots.py`** — `GeometryPlots`: 3-D geometry coefficients along cuts/planes
- **`velocity.py`** — `SrcMom` (Krook source moments), `VspSlice` (velocity space)
- **`planes.py`** — `Planes`: remap onto geometric (θ, φ) angles + (n, m) analysis
- **`vis.py`** — `Vis`: VTK export using the `cart_coords` GENE-3D writes

### Data flow
1. `set_runs(folder)` → segment suffixes
2. `Params(folder, extensions)` → parameter dicts
3. `Geometry(...)` + `Coordinates(...)` → geometry/coordinate arrays
4. `BinaryReader`/`BPReader`/`MultiSegmentReader` → stream field/moment data
5. Diagnostics consume readers → compute → cache to HDF5 → plot

### `__init__.py` flat re-exports
The root `__init__.py` re-exports all major classes for backward compatibility (`from genetools import Params, BinaryReader, ...`).

## Key conventions

- **Geometry kinds**: prefer `params.geometry_kind()` / `run.geometry_kind` over the two-valued `x_local`. `flux_tube` is spectral in x and y; `x_global` real-space in x; `xy_global` (GENE-3D) real-space in both. `x_local` alone cannot distinguish `x_global` from GENE-3D.
- **Hermitian symmetry**: in the spectral-y cases only non-negative ky modes are stored, so ky=0 is unweighted and ky>0 is weighted by 2 in summations. **This does not apply to GENE-3D**: y is real space, so its FFT already contains both signs of ky and no factor of 2 belongs anywhere.
- **GENE-3D axis order and dtype**: futils stores a Fortran `(nx, ny, nz)` array as an HDF5 `(nz, ny, nx)` dataset, and `creatf(..., 's')` stores 32-bit reals whatever `PRECISION` says. `H5Reader` undoes both; never derive the dtype from the namelist.
- **GENE-3D flux conventions** (from `diag_3d.F90` / `geometry.F90`, not guesswork): `Gamma_es = -n·∂φ/∂y·flux_geomfac` and `Gamma_em = +u_par·∂A∥/∂y·flux_geomfac`, where `flux_geomfac = 1/C_xy`, times `1/sqrt(g^xx)` when `norm_flux_projection`. There is **no** `1/Bref`, and the flutter velocity has the **opposite sign** to the ExB one. The same flag decides which area converts a flux density to a total (`area['Area']` when projecting, `area['dVdx']` otherwise).
- **`n_moms` is unreliable for GENE-3D**: the parameters file reports `par_in::n_moms = 6` while `diag_3d::n_moms = 10` datasets are written. Discover variables from the file (`reader.var_names`).
- **`nrg` column order** (`diag.F90:nrg_label1`): `n², u_par², T_par, T_per, Γ_es, Γ_em, Q_es, Q_em, Π_es, Π_em, ...`. GENE-3D writes the first eight (`nrgcols = 8`).
- **numpy compat**: code handles both `np.trapz` (numpy <2.0) and `np.trapezoid` (numpy >=2.0).
- **Optional deps**: `adios2` for BP files, `numba` for JIT acceleration in spectra. Both use try/except import guards.
- **Temperature**: T = (1/3)T_par + (2/3)T_perp throughout diagnostics.
