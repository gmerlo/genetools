"""
_xr.py — wrap genetools numpy diagnostic outputs into ``xarray.Dataset`` objects.

This is the **Phase-1 adapter** of the redesign (see
``docs/superpowers/specs/2026-06-01-genetools-redesign-design.md``): the existing
diagnostics return plain dicts of numpy arrays, and these helpers attach
coordinate values, species labels, and physical-unit metadata so the
:class:`~genetools.run.Run` facade can hand back self-describing
``xarray.Dataset`` objects.

The wrapping is intentionally generic. Each diagnostic wrapper passes a small
``dims`` hint mapping a *base* variable name (after stripping the species prefix)
to a tuple of dimension names; anything not hinted is inferred by matching axis
lengths to the known coordinate arrays. Coordinate *values* are attached only
when their length matches the data axis, so a mismatch degrades gracefully to an
integer index rather than raising.
"""

from __future__ import annotations

import numpy as np

try:
    import xarray as xr
except ImportError as exc:  # pragma: no cover - exercised only without xarray
    raise ImportError(
        "genetools requires the 'xarray' package for its data layer. "
        "Install it with `pip install xarray`."
    ) from exc


# Reference/normalisation quantities worth carrying into Dataset.attrs.
_UNIT_KEYS = (
    "cref", "rhoref", "rho_starref", "Lref", "Bref", "nref", "Tref", "mref",
    "Oref", "pref", "Ggb", "Qgb", "Pgb",
)

# Names treated as coordinate sources rather than data variables.
_COORD_KEYS = ("x", "kx", "ky", "z", "time")

# Priority order when inferring an ambiguous axis from its length.
_DIM_PRIORITY = ("time", "species", "x", "ky", "kx", "z")


def unit_attrs(params) -> dict:
    """Return a flat dict of scalar unit/normalisation quantities for ``attrs``."""
    if not isinstance(params, dict):
        return {}
    units = params.get("units", {}) or {}
    out = {}
    for k in _UNIT_KEYS:
        v = units.get(k)
        if v is not None and np.isscalar(v):
            out[k] = float(v)
    return out


def _coord_arrays(coords, params) -> dict:
    """Build ``{dimname: 1d ndarray}`` of available physical axes."""
    out = {}
    if coords:
        for src, name in (("ky", "ky"), ("kx", "kx"), ("z", "z")):
            arr = np.asarray(coords.get(src, []))
            if arr.size:
                out[name] = arr
        x = np.asarray(coords.get("x", []))
        x_o_a = np.asarray(coords.get("x_o_a", []))
        if x.size:
            out["x"] = x
        elif x_o_a.size:
            out["x"] = x_o_a
    return out


def _split_species(key: str, species) -> tuple:
    """Split ``'ions_Q_es_kx'`` into ``('ions', 'Q_es_kx')`` when prefixed."""
    for name in sorted(species or [], key=len, reverse=True):
        prefix = f"{name}_"
        if key.startswith(prefix):
            return name, key[len(prefix):]
    return None, key


def _infer_dims(arr: np.ndarray, caxes: dict) -> tuple:
    """Infer dimension names for *arr* by matching axis lengths to coords."""
    dims = []
    for axis, length in enumerate(arr.shape):
        match = None
        for name in _DIM_PRIORITY:
            cand = caxes.get(name)
            if cand is not None and len(cand) == length:
                match = name
                break
        dims.append(match if match is not None else f"dim{axis}")
    return tuple(dims)


def build_dataset(data: dict, *, coords=None, params=None, species=None,
                  dims=None, time=None) -> "xr.Dataset":
    """
    Assemble an ``xarray.Dataset`` from a diagnostic's ``{key: ndarray}`` output.

    Parameters
    ----------
    data : dict
        Mapping of variable name to numpy array. Keys named ``x``/``kx``/``ky``/
        ``z``/``time`` are treated as coordinate sources, not variables. Keys may
        be species-prefixed (``'ions_Q_es'``); the prefix is stripped and the
        remaining base name becomes the variable, stacked along a ``species`` dim.
    coords : dict, optional
        A coordinate dict from :func:`genetools.io.Coordinates` (a single
        segment), used as the source of axis values.
    params : dict, optional
        A single-segment parameter dict; its ``units`` block populates ``attrs``.
    species : list of str, optional
        Species names, controlling species-dim ordering.
    dims : dict, optional
        Hints mapping a *base* variable name to a tuple of dimension names.
    time : array-like, optional
        Time axis values (attached as the ``time`` coordinate when used).
    """
    species = list(species or [])
    dims = dims or {}

    caxes = _coord_arrays(coords, params)
    for ck in ("x", "kx", "ky", "z"):
        if ck in data:
            arr = np.asarray(data[ck])
            if arr.ndim == 1 and arr.size:
                caxes[ck] = arr
    if time is not None:
        caxes["time"] = np.asarray(time)

    # Group variables by base name, keyed by species (None for unprefixed).
    groups: dict = {}
    var_dims: dict = {}
    for key, val in data.items():
        if key in _COORD_KEYS:
            continue
        arr = np.asarray(val)
        sp, var = _split_species(key, species)
        hinted = dims.get(var)
        if hinted is not None and len(hinted) == arr.ndim:
            base_dims = tuple(hinted)
        else:
            base_dims = _infer_dims(arr, caxes)
        groups.setdefault(var, {})[sp] = arr
        var_dims[var] = base_dims

    data_vars: dict = {}
    used_species: list = []
    used_dims: set = set()
    for var, spmap in groups.items():
        base_dims = var_dims[var]
        if set(spmap) == {None}:
            arr = spmap[None]
            data_vars[var] = (base_dims, arr)
            used_dims.update(base_dims)
        else:
            names = [n for n in species if n in spmap]
            names += [n for n in spmap if n is not None and n not in names]
            stack = np.stack([spmap[n] for n in names], axis=0)
            full_dims = ("species",) + base_dims
            data_vars[var] = (full_dims, stack)
            used_dims.update(full_dims)
            for n in names:
                if n not in used_species:
                    used_species.append(n)

    # Attach coordinate values where lengths line up.
    xr_coords: dict = {}
    for dim in used_dims:
        if dim == "species":
            xr_coords["species"] = used_species
            continue
        axis_len = _dim_length(data_vars, dim)
        cand = caxes.get(dim)
        if cand is not None and axis_len is not None and len(cand) == axis_len:
            xr_coords[dim] = cand

    ds = xr.Dataset(data_vars, coords=xr_coords)
    ds.attrs.update(unit_attrs(params))
    return ds


def _dim_length(data_vars: dict, dim: str):
    """Return the length of *dim* as it appears in the assembled variables."""
    for dims_tuple, arr in data_vars.values():
        if dim in dims_tuple:
            return arr.shape[dims_tuple.index(dim)]
    return None


# Named GENE nrg columns (others are kept in the raw ``nrg`` variable).
_NRG_NAMED = {
    0: "n_sq", 1: "T_par_sq", 2: "T_perp_sq", 3: "u_par_sq",
    6: "Q_es", 7: "Q_em", 8: "Gamma_es", 9: "Gamma_em",
}


def nrg_dataset(times, data, species=None, params=None) -> "xr.Dataset":
    """
    Wrap ``NrgReader.read_all()`` output into an ``xarray.Dataset``.

    Parameters
    ----------
    times : array-like
        Shape ``(n_times,)``.
    data : array-like
        Shape ``(n_species, n_cols, n_times)``.
    species : list of str, optional
        Species names; defaults to ``sp0, sp1, ...``.
    params : dict, optional
        Single-segment parameter dict; its ``units`` block populates ``attrs``.
    """
    data = np.asarray(data)
    times = np.asarray(times)
    n_spec, n_cols, _ = data.shape
    sp = list(species or [])[:n_spec]
    if len(sp) < n_spec:
        sp += [f"sp{i}" for i in range(len(sp), n_spec)]

    coords = {"species": sp, "time": times, "column": np.arange(n_cols)}
    data_vars = {"nrg": (("species", "column", "time"), data)}
    for col, name in _NRG_NAMED.items():
        if col < n_cols:
            data_vars[name] = (("species", "time"), data[:, col, :])

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs.update(unit_attrs(params))
    return ds
