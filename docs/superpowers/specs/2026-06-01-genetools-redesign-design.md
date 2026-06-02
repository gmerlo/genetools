# genetools redesign — Run facade, xarray data layer, CLI, and new diagnostics

**Date:** 2026-06-01
**Status:** Approved design (pre-implementation)
**Author:** brainstormed with Gabriele

## 1. Purpose & goals

`genetools` is a post-processing toolkit for GENE gyrokinetic simulations. It is
well-structured internally (clean `io/` vs `diagnostics/` split, unified reader
streaming interface, incremental HDF5 caching, local + global geometry support),
but the **user-facing workflow is verbose**: running a single diagnostic requires
manually calling `set_runs`, `Params`, `Geometry`, `Coordinates`, building readers
per species, and threading all of that into the diagnostic. There is no CLI and
only one scratch notebook.

This effort makes `genetools` **simpler and better to use in Jupyter notebooks and
on the command line**, and **ports the highest-value diagnostics from GENE's
`pydiag`** (`~/codes/gene/python-diag/pydiag`) that `genetools` lacks.

### Decisions locked during brainstorming

1. **Scope:** both a new high-level API/CLI layer **and** porting missing diagnostics.
2. **API shape:** a **`Run` facade** — one object that auto-wires everything, with
   diagnostics as lazy attributes/methods (tab-completable in notebooks).
3. **Data layer:** diagnostics return **`xarray.Dataset`** objects (labeled dims,
   coords, units). `xarray` is a **hard (required) dependency**.
4. **CLI style:** **flag-style** (pydiag-like): `genetools /run --spectra --t 500 2000`.
5. **Refactor latitude:** free to refactor diagnostic internals (will update tests).
6. **Diagnostics to port first:** ballooning / mode structure, growth rate &
   frequency, amplitude spectra, zonal-flow x-t.
7. **`run.spectra` auto-dispatches** local (`Spectra`) vs global (`SpectraGlobal`)
   based on `x_local`.
8. **Growth rate is extracted from the field file** (γ from |φ| growth, ω from the
   field's complex-phase rotation between outputs); an `omega<ext>` file is read as
   an optional cross-check when present. No dependence on the nrg file.

### Non-goals (explicitly deferred)

- pydiag diagnostics not in the priority list: FSA moments, source moments,
  anisotropy, flux PDFs, torus cuts, neoclassical (Chang-Hinton). These remain
  candidates for a later effort.
- Replacing the internal incremental HDF5 caching (it is restart-safe and good; it
  stays as the internal cache — xarray is only the return/wrapper layer).
- A GUI.

## 2. Implementation approach: phased (Approach B)

The work lands the same clean end state as a big-bang refactor, but in
independently-shippable, reviewable slices:

- **Phase 1 — Foundation.** Build `Run`, the `_xr.py` xarray wrapper layer (used as
  a thin adapter over existing diagnostics' numpy-dict outputs), and the flag-style
  CLI over the **existing** diagnostics. End-to-end notebook + CLI experience works.
- **Phase 2 — New diagnostics.** Implement ballooning, growth rate, amplitude
  spectra, and zonal x-t — `Run`-native and xarray-native from the start.
- **Phase 3 — Deepen.** Move the xarray wrapping *inside* each existing diagnostic,
  retire the Phase-1 adapter, and update those diagnostics' tests. User-visible
  behavior is unchanged by Phase 3; it is purely internal consolidation.

Each phase is its own review checkpoint.

## 3. Package layout

New files; nothing existing is moved in Phase 1.

> **Physical layout note.** The repository root *is* the `genetools` package
> (the repo dir contains `__init__.py`, `compat.py`, `io/`, `diagnostics/`).
> So the module paths below map to physical files at the repo root: `genetools/run.py`
> → `./run.py`, `genetools/_xr.py` → `./_xr.py`, `genetools/cli.py` → `./cli.py`,
> `genetools/io/omega.py` → `./io/omega.py`, etc. Paths are written in module form
> for clarity.

```
genetools/
  __init__.py            # + re-export Run (keep existing flat re-exports)
  run.py                 # NEW  — Run facade (the one object users touch)
  _xr.py                 # NEW  — numpy -> xarray.Dataset wrappers + coord/unit builders
  cli.py                 # NEW  — flag-style argparse CLI (entry point: `genetools`)
  io/
    omega.py             # NEW  — parse omega<ext> / eigenvalues.dat (optional cross-check)
    ... (existing unchanged)
  diagnostics/
    _base.py             # existing CachingDiagnostic (reused by new streaming diagnostics)
    ballooning.py        # NEW
    growthrate.py        # NEW
    amplitude.py         # NEW  — amplitude spectra of moments & fields
    zonal.py             # NEW  — zonal-flow x-t
    ... (existing unchanged in Phase 1; xarray-native in Phase 3)
```

`pyproject.toml` changes:
- Add `xarray` to `[project] dependencies` (hard dependency).
- Add `[project.scripts]`: `genetools = "genetools.cli:main"`.
- (netCDF export is intentionally out of scope; the internal HDF5 cache remains
  the on-disk format.)

## 4. The `Run` object (`run.py`)

A single object that performs the wiring users currently do by hand.

```python
from genetools import Run
run = Run("/path/to/run", ext=None)   # ext=None -> auto-discover all segments
```

On construction it runs `set_runs`, `Params`, `Geometry`, `Coordinates`, and stores
the segment list. Readers are built **lazily** and cached. Multi-segment runs are
stitched with `MultiSegmentReader` automatically.

### Metadata (properties)
- `run.params` — the `Params` object (`run.params.get(0)` etc. still available).
- `run.species` — list of species names.
- `run.is_local` — `x_local` boolean.
- `run.extensions` — discovered segment suffixes.
- `run.geometry` — geometry (per-segment list, as today).
- `run.coords` — coordinates (per-segment list, as today).
- `run.times` — global merged/deduplicated time axis.

### Readers (lazy, cached, multi-segment aware)
- `run.field` — field reader (phi/apar/bpar).
- `run.mom(species=None)` — moment reader; `species=None` -> first species.

### Diagnostic accessors
Each returns a diagnostic object **bound to the run**:

| Accessor | Backed by |
|---|---|
| `run.nrg` | `NrgReader` |
| `run.spectra` | auto-dispatch: `Spectra` (local) / `SpectraGlobal` (global) |
| `run.profiles` | `Profiles` |
| `run.fluxes2d` | `Fluxes2D` |
| `run.shearing` | `ShearingRate` |
| `run.contours` | `Contours` |
| `run.ballooning(ky=...)` | NEW (callable; `ky` required) |
| `run.growthrate` | NEW |
| `run.amplitude` | NEW |
| `run.zonal` | NEW |

### Uniform diagnostic surface
Every diagnostic object exposes:
- `.data` -> `xarray.Dataset` (computes-or-loads; uses the existing HDF5 cache
  underneath where applicable).
- `.plot(t=..., **opts)` -> matplotlib figure(s).
- `.save()` -> write/refresh the HDF5 cache (where the diagnostic caches).

Parametrized diagnostics are callable and return a configured instance:
`run.ballooning(ky=0.3).plot()`.

### Time-window convention (everywhere)
- `t=(start, stop)` — time window.
- `t=None` — full available range.
- `t=<scalar>` — nearest single time (for contour/ballooning snapshots).

### Continuation / restart runs

GENE runs are commonly split into restarted **segments** (`_0001`, `_0002`, …,
`.dat`). This is a first-class concern, handled at two layers.

**Reader layer (existing, reused as-is).** `Run(path, ext=None)` discovers *all*
segments via `set_runs` and builds a single `MultiSegmentReader` across them
automatically — continuation is transparent, segments are never named by the user.
The reader (`io/data.py`):
- merges every segment's time array, sorts, and **deduplicates overlaps with
  "later segment wins"** — matching GENE's restart-rewind, where a restart
  re-emits a few steps that must not be double-counted;
- exposes the merged/deduplicated timeline as `run.times`; `t=(start, stop)`
  windows select across segment boundaries seamlessly;
- routes each requested step to the owning segment and yields arrays in that
  segment's native shape/dtype.

**Diagnostic layer (policy for this redesign).** Confirmed decision:
**continuation runs keep the same numerical grid (extend in time); validate and
warn, assume consistent.** Concretely:
- On `Run` construction, validate grid/geometry consistency across the spanned
  segments (perpendicular/parallel dims and key geometry). If they differ, emit a
  **clear warning** naming the divergent segments and advising the user to scope
  to a consistent subset. (This replaces the current *silent* segment-0
  assumption in `Spectra`/`Profiles`/`Fluxes2D`, which use `coords[0]`/`geom[0]`.)
- Allow scoping to a subset of segments — `Run(path, ext=["_0002", ".dat"])` — for
  the rare case where a user wants only the post-change portion.
- With a consistent grid (the assumed/validated case), the single grid is used for
  weighting and for the xarray output coordinates; results are correct and labels
  unambiguous.
- Full per-segment-grid correctness (per-step geometry via
  `stream_selected_with_seg`, interpolating across grid changes) is explicitly a
  non-goal for this effort.
- New diagnostics follow the same model. Linear diagnostics (growth-rate,
  ballooning) naturally use the trailing window of the final segment, so
  continuation barely affects them.

**Caching across continuations.** Re-running a diagnostic after a run is extended
recomputes only the new times (`_is_already_saved` skips cached steps) and appends
to the HDF5 cache — cheap incremental updates. *Known limitation:* if a restart
*overwrote* already-cached overlapping timesteps, those cached values are not
recomputed (the cache is keyed by time value, which is unchanged). Documented; not
addressed in this effort.

## 5. xarray data layer (`_xr.py`)

Internal computation stays **numpy**. The existing incremental HDF5 cache is
untouched. xarray is the **return/wrapper layer only**.

`_xr.py` provides builder helpers that turn a numpy result + a `Run`'s
coords/params into a labeled `xarray.Dataset`:
- Attach real coordinate values: `ky`, `kx`, `x` / `x_o_a`, `z`, `species`, `time`.
- Put units and normalizations into `.attrs` (Q_gb, Γ_gb, ρ_ref, c_ref, etc.) so
  plots and exports are self-describing.

Example:
```python
ds = run.spectra.data
# Dimensions: (species, ky); coords ky=[...], species=['ions','elec']
ds.Q_es.sel(species="ions").plot()
```

**Phase-1 mechanism:** existing diagnostics still return numpy dicts; a thin
adapter in `_xr.py` wraps them at the facade boundary. **Phase 3** moves the
wrapping inside each diagnostic and retires the adapter. The user-facing result is
identical; this is internal sequencing only.

## 6. New diagnostics (Phase 2)

All four are `Run`-native + xarray-native, ported from `pydiag` but trimmed to the
genetools conventions (Hermitian ky-weighting: ky=0 unweighted, ky>0 weighted ×2;
T = (1/3)T∥ + (2/3)T⊥; numpy trapz/trapezoid compat).

### 6.1 `ballooning.py` — mode structure (local / flux-tube)
For a chosen `ky` mode, connect the `kx` modes along the extended ballooning angle
χ using GENE's parallel boundary phase factor (the kx connection set and the
`2π·shat`-type phase between connected modes). Produces φ(χ), A∥(χ), B∥(χ).
- Default time selection: last available time (linear runs); for nonlinear runs,
  use the |·| time-average over the window.
- Normalize to the maximum amplitude (configurable).
- Plot real / imaginary / absolute parts vs χ.
- Output: `Dataset(dims=(field, chi))`, `field` ∈ {phi, apar, bpar} as available.
- Local geometry only (ballooning is a flux-tube concept); raise a clear error for
  global runs.

### 6.2 `growthrate.py` — linear growth rate & frequency (from field file)
Primary method (field-based, matching how GENE derives it):
- γ = d ln|φ| / dt — from the growth of the field amplitude between outputs.
- ω = d arg(φ) / dt — from the rotation of the field's complex phase between
  consecutive outputs.
- Computed via a fit/average over a configurable trailing time window (default: the
  later portion of the run, where a single linear mode dominates), reducing to a
  per-mode (per-ky) value.
Optional cross-check:
- If an `omega<ext>` file (or `eigenvalues.dat`) is present, parse it via
  `io/omega.py` and include those values for comparison; never required.
- Output: `Dataset` with a `ky` coord and vars `gamma`, `omega` (degrades to a
  single ky for a single linear run). Plot γ(ky), ω(ky).

### 6.3 `amplitude.py` — amplitude spectra
Like `Spectra` but computes |A|² of each **moment and field** (φ, A∥, B∥, n, T∥,
T⊥, u∥, and other available moments), resolved in `kx` and `ky`, time-averaged with
the same Hermitian ky-weighting. Reuses the streaming + caching machinery
(`CachingDiagnostic`).
- Output: `Dataset` with `kx`/`ky` coords, one variable per quantity (and per
  species for moments). Plot kx and ky spectra.

### 6.4 `zonal.py` — zonal-flow x-t
Extract the ky=0 component of φ over time -> φ_zonal(x, t):
- Local: IFFT over kx to real-space x.
- Global: real-space, flux-surface-averaged over z.
- Optionally also compute ∂²φ_zonal/∂x² (radial structure of the shear).
- Complements `ShearingRate` (which provides the RMS shearing rate); this provides
  the x-t evolution of the zonal potential.
- Output: `Dataset(dims=(time, x))`. Plot an x-t contour. Streams + caches.

## 7. CLI (`cli.py`) — flag-style

```
genetools [PATH] --DIAG [diag-opts] [common-opts]
```
- **Path:** positional `PATH` (defaults to current working directory) or
  `--runpath PATH`.
- **Diagnostic flags (mutually exclusive group):**
  `--nrg --spectra --profiles --fluxes2d --shearing --contours --ballooning
   --growthrate --amplitude --zonal`.
- **Common options:**
  - `--t START STOP` — time window (two floats).
  - `--species NAME [NAME ...]` — restrict species.
  - `--ext EXT [EXT ...]` — explicit segment suffixes (else auto-discover).
  - `--save FILE` — file path saves there; a directory auto-names a PNG.
  - `--no-show` — do not open a window (headless / tests).
- **Diagnostic-specific:** `--ky VAL` (ballooning).

Behavior: build a `Run`, call the matching accessor's `.plot(...)`, then
`plt.show()` unless `--save`/`--no-show`. Examples:
```bash
genetools /run --nrg
genetools /run --spectra --t 500 2000 --save spectra.png
genetools /run --ballooning --ky 0.3
genetools /run --growthrate
```

## 8. Testing

Aligned to the existing `tests/` layout. Extend `tests/conftest.py` with synthetic
field/mom/nrg/omega fixtures.

- `tests/test_run.py` — segment discovery, lazy reader construction, `species` /
  `is_local`, accessors return bound diagnostics, multi-segment wiring, time axis,
  **continuation runs**: spanning multiple segments, overlap dedup (later wins),
  `ext=[...]` subsetting, and the grid-consistency validation warning on mismatch.
- `tests/test_xr.py` — wrappers produce correct dims/coords/units/attrs.
- `tests/diagnostics/test_ballooning.py` — χ connection length, dims, local-only
  guard, normalization.
- `tests/diagnostics/test_growthrate.py` — γ/ω recovered from a synthetic
  exponentially-growing, phase-rotating field; optional omega-file cross-check.
- `tests/diagnostics/test_amplitude.py` — spectrum shapes, Hermitian weighting,
  caching round-trip.
- `tests/diagnostics/test_zonal.py` — x-t shape, local IFFT vs global FSA paths.
- `tests/test_cli.py` — argparse parsing + dispatch with `--no-show`.

Existing tests stay green in Phases 1–2 (existing diagnostic internals unchanged).
In Phase 3, each refactored diagnostic's tests are updated to assert on xarray
outputs.

Run: `pytest tests/ -v` (and `--cov=genetools` for coverage).

## 9. Documentation

- Update `README.md` quick-start to lead with the `Run` facade and the CLI.
- Update `CLAUDE.md`: correct the stale note about the test file location (tests are
  in `tests/`, not `test_genetools.py` at root), and document the new `Run`/CLI
  entry points and xarray return convention.
- Replace/augment `Untitled1.ipynb` with a clean example notebook demonstrating the
  `Run` facade and the new diagnostics.

## 10. Risks & mitigations

- **xarray coordinate/units correctness** — validated by `tests/test_xr.py` against
  known coord arrays from `Coordinates`/`Params`.
- **Ballooning connection logic** is subtle (phase factor, kx connection set) —
  port carefully from `pydiag/diagplots/plot_ball.py`, test χ length and continuity
  on synthetic data.
- **Growth-rate windowing** — a poor window over a not-yet-converged run yields
  noisy γ/ω; default to the trailing portion and expose the window via `t=`.
- **Continuation runs with a changed grid** — out of scope by decision; mitigated by
  the `Run` grid-consistency validation warning + `ext=[...]` subsetting, so the
  user is told rather than silently given wrong results.
- **Stale cache on overwritten overlaps** — if a restart rewrites already-cached
  timesteps, the time-keyed cache keeps the old values. Documented limitation; the
  workaround is to delete the diagnostic's HDF5 cache and recompute.
- **Phase-1 adapter is temporary** — clearly marked; removed in Phase 3 to avoid
  lingering dual code paths.

## 11. Definition of done (per phase)

- **Phase 1:** `Run(...)` constructs from a real run folder; every existing
  diagnostic is reachable as `run.<diag>` and returns xarray; `genetools` CLI runs
  each existing diagnostic; `tests/test_run.py`, `tests/test_xr.py`,
  `tests/test_cli.py` pass; existing tests still pass.
- **Phase 2:** the four new diagnostics work via `run.*` and the CLI, with passing
  diagnostic tests.
- **Phase 3:** README/CLAUDE.md updated to lead with the `Run` facade + CLI + new
  diagnostics; example notebook added (`examples/quickstart.ipynb`).

  *Implementation note (2026-06):* the internal "deepen" refactor is **complete**.
  Each existing diagnostic now owns its xarray construction via a `dataset(coords,
  params, species, …)` method (explicit dimensions, real coords, unit attrs);
  `NrgReader` gains `dataset(params)`. The generic inference adapter
  (`_xr.build_dataset`/`_infer_dims`/`_coord_arrays`/`nrg_dataset`) is retired,
  replaced by small explicit helpers in `_xr.py` (`stacked_vars`, `attach_coords`,
  `make_dataset`, `unit_attrs`, `split_species`). The `Run` bound wrappers now call
  the diagnostics' `dataset()` methods directly. Two latent Phase-1 bugs were
  fixed in passing: `Profiles.plot` takes `equilibrium_profiles=` and
  `Fluxes2D.plot` takes `show_heatmaps=` (the wrappers had passed wrong kwargs).
  The existing numpy `load`/`load_time_average` methods are unchanged, so all 223
  existing tests still pass.
