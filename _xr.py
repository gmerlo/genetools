# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
_xr.py — shared helpers for building labelled ``xarray.Dataset`` objects.

After the internal refactor each diagnostic owns its own xarray construction (a
``dataset(...)`` method) using the small, *explicit* helpers below — there is no
generic dimension-inference adapter anymore. The helpers handle the two pieces of
boilerplate every diagnostic shares: stacking per-species variables along a
``species`` dimension, and attaching coordinate values when their length matches
the data.
"""

from __future__ import annotations

import numpy as np

try:
    import xarray as xr  # noqa: F401  (re-exported for convenience)
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


def split_species(key: str, species) -> tuple:
    """Split ``'ions_Q_es_kx'`` into ``('ions', 'Q_es_kx')`` when prefixed."""
    for name in sorted(species or [], key=len, reverse=True):
        prefix = f"{name}_"
        if key.startswith(prefix):
            return name, key[len(prefix):]
    return None, key


def stacked_vars(raw: dict, species, dim_of, *, coord_keys=(), skip_empty=True):
    """
    Group a ``{key: ndarray}`` dict into xarray-ready variables.

    Keys may be species-prefixed (``'ions_Q_es_kx'``); the prefix is stripped and
    matching base names are stacked along a leading ``species`` dimension.

    Parameters
    ----------
    raw : dict
        Mapping of variable name to numpy array.
    species : list of str
        Species names, controlling species-dim ordering.
    dim_of : callable
        ``dim_of(base_name) -> tuple[str]`` giving the (non-species) dimension
        names of a variable, or ``None`` to drop it.
    coord_keys : tuple of str
        Keys in *raw* to ignore (they are coordinate sources, not variables).
    skip_empty : bool
        Drop variables whose array has a zero-length axis (e.g. EM terms in an
        electrostatic run) to avoid conflicting dimension sizes.

    Returns
    -------
    (data_vars, used_species) : (dict, list)
        ``data_vars`` maps name -> ``(dims_tuple, ndarray)``.
    """
    species = list(species or [])
    groups: dict = {}
    for key, val in raw.items():
        if key in coord_keys:
            continue
        sp, var = split_species(key, species)
        groups.setdefault(var, {})[sp] = np.asarray(val)

    data_vars: dict = {}
    used_species: list = []
    for var, per in groups.items():
        dims = dim_of(var)
        if dims is None:
            continue
        if skip_empty and any(a.size == 0 for a in per.values()):
            continue
        if set(per) == {None}:
            data_vars[var] = (tuple(dims), per[None])
        else:
            names = [n for n in species if n in per]
            names += [n for n in per if n is not None and n not in names]
            stack = np.stack([per[n] for n in names], axis=0)
            data_vars[var] = (("species",) + tuple(dims), stack)
            for n in names:
                if n not in used_species:
                    used_species.append(n)
    return data_vars, used_species


def dim_length(data_vars: dict, dim: str):
    """Return the length of *dim* across the assembled variables, or None."""
    for dims_tuple, arr in data_vars.values():
        if dim in dims_tuple:
            return np.asarray(arr).shape[dims_tuple.index(dim)]
    return None


def attach_coords(data_vars: dict, candidates: dict) -> dict:
    """Return the subset of *candidates* whose length matches its dim's length."""
    coords = {}
    for dim, arr in candidates.items():
        if arr is None:
            continue
        arr = np.asarray(arr)
        length = dim_length(data_vars, dim)
        if length is not None and arr.size == length:
            coords[dim] = arr
    return coords


def make_dataset(data_vars: dict, candidates: dict = None, *,
                 species=None, params=None) -> "xr.Dataset":
    """Assemble a Dataset from *data_vars*, attaching matching coords + units."""
    coords = attach_coords(data_vars, candidates or {})
    if species:
        coords["species"] = list(species)
    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs.update(unit_attrs(params))
    return ds
