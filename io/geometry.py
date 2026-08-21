# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
geometry.py — GENE geometry file reader.
 
Reads the geometry file for one or more run segments and returns a list of
geometry dictionaries, one per segment.
 
The geometry file has two sections:
 
1. A Fortran-90 namelist block (``&...  /``) containing scalar parameters
   such as ``q0``, ``shat``, ``cxy``, ``cy``, etc.
2. A numeric data section whose layout depends on the geometry type:
 
   * **Local** (``x_local=True``): 16 columns of floats, one row per z-point.
     Columns are: gxx, gxy, gxz, gyy, gyz, gzz, Bfield, dBdx, dBdy, dBdz,
     Jacobian, R, Phi, Z, dxdR, dxdZ.
   * **Global** (``x_local=False``): named arrays written sequentially, each
     preceded by its name as a string token, followed by ``nx0 * nz0`` floats
     reshaped to ``(nx0, nz0)``.
 
Only the standard binary output format is supported (HDF5 is not).
 
Public interface
----------------
``Geometry(folder, extensions, params)``
    Load geometry for all run segments.  Returns a list of dicts.
 
Each dict contains:
 
    ``kind``       geometry type string (e.g. ``'tracer_efit'``)
    ``Bfield``     magnetic field magnitude along z
    ``Jacobian``   coordinate Jacobian
    ``dBdx/y/z``   field gradient components
    ``metric``     dict: C_xy, C_y, gxx, gxy, gxz, gyy, gyz, gzz, dxdR, dxdZ
    ``shape``      dict: gR, gZ, gPhi
    ``local``      dict: q0, shat, trpeps, gridpoints
    ``curv``       dict: K_x, K_y, sloc  (curvature, computed post-load)
    ``area``       dict: Area, dVdx       (computed post-load)
    ``profiles``   dict: q  (global geometry only); for GENE-3D also
                   dVdx, sqrtgxx_fs, gxx_fs, dpdx_pm_arr, xval_a
    ``cart_coords`` dict: x, y, z — Cartesian position of every grid point
                   (GENE-3D HDF5 geometry only)
 
Example
-------
>>> from genetools.geometry import Geometry
>>> geoms = Geometry('/path/to/run/', ['_0001', '_0002'], params)
>>> J = geoms[0]['Jacobian']   # shape (nz,) for local, (nx, nz) for global
"""
 
import os
import re

import h5py
import numpy as np

from genetools.io._zgrid import build_zgrid
 
 
# ---------------------------------------------------------------------------
# Filename helper
# ---------------------------------------------------------------------------
 
def _geometry_filename(folder: str, geom_type: str, ext: str) -> str:
    """Return the full path to a geometry file."""
    return os.path.join(folder, geom_type + ext)
 
 
# ---------------------------------------------------------------------------
# Namelist parser
# ---------------------------------------------------------------------------
 
def _parse_namelist(text: str) -> dict:
    """
    Parse a Fortran-90 namelist block (everything before the closing ``/``)
    into a flat dict of {key: value} with numeric values cast to float.
 
    Parameters
    ----------
    text : str
        Raw text of the namelist section (up to and including ``/``).
 
    Returns
    -------
    dict
        All key=value pairs found, values as float where possible else str.
    """
    result = {}
    for key, val in re.findall(r'(\w+)\s*=\s*([^\s,/]+)', text):
        try:
            result[key.lower()] = float(val)
        except ValueError:
            result[key.lower()] = val.strip().strip("'\"")
    return result
 
 
# ---------------------------------------------------------------------------
# Local geometry reader
# ---------------------------------------------------------------------------
 
def _read_local(fid, tmp_geom: dict) -> dict:
    """
    Read the numeric section of a **local** geometry file.
 
    Expects 16 whitespace-separated columns of floats, one row per z-point.
 
    Columns
    -------
    0  gxx      1  gxy      2  gxz
    3  gyy      4  gyz      5  gzz
    6  Bfield   7  dBdx     8  dBdy     9  dBdz
    10 Jacobian 11 R        12 Phi      13 Z
    14 dxdR     15 dxdZ
 
    Parameters
    ----------
    fid : file object
        Positioned just after the namelist ``/`` line.
    tmp_geom : dict
        Parsed namelist values.
 
    Returns
    -------
    dict
        Partial geometry dict (without curvature/area).
    """
    # Read remaining numeric data efficiently via np.loadtxt
    try:
        coeffs = np.loadtxt(fid)
    except Exception:
        raise ValueError("No numeric data found in local geometry file.")

    if coeffs.ndim == 1:
        coeffs = coeffs.reshape(1, -1)
    if coeffs.shape[0] == 0 or coeffs.shape[1] < 16:
        raise ValueError("No numeric data found in local geometry file.")
 
    gxx, gxy, gxz = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]
    gyy, gyz, gzz = coeffs[:, 3], coeffs[:, 4], coeffs[:, 5]
    Bfield        = coeffs[:, 6]
    dBdx, dBdy, dBdz = coeffs[:, 7], coeffs[:, 8], coeffs[:, 9]
    Jacobian      = coeffs[:, 10]
    R, Phi, Z     = coeffs[:, 11], coeffs[:, 12], coeffs[:, 13]
    dxdR, dxdZ   = coeffs[:, 14], coeffs[:, 15]
 
    # C_xy: from namelist if present, else derived from metric
    if "cxy" in tmp_geom:
        C_xy = tmp_geom["cxy"]
    else:
        C_xy = float(np.sqrt(Bfield[0]**2 / (gxx[0]*gyy[0] - gxy[0]**2)))
 
    C_y = tmp_geom.get("cy", 1.0)
 
    metric = dict(
        C_xy=C_xy, C_y=C_y,
        gxx=gxx, gxy=gxy, gxz=gxz,
        gyy=gyy, gyz=gyz, gzz=gzz,
        dxdR=dxdR, dxdZ=dxdZ,
    )
    shape = dict(gR=R, gZ=Z, gPhi=Phi)
    local = dict(
        q0         = tmp_geom.get("q0"),
        shat       = tmp_geom.get("shat"),
        trpeps     = tmp_geom.get("trpeps"),
        gridpoints = tmp_geom.get("gridpoints"),
    )
 
    return dict(
        Bfield=Bfield, Jacobian=Jacobian,
        dBdx=dBdx, dBdy=dBdy, dBdz=dBdz,
        metric=metric, shape=shape, local=local,
        dxdR=dxdR, dxdZ=dxdZ,
    )
 
 
# ---------------------------------------------------------------------------
# Global geometry reader
# ---------------------------------------------------------------------------
 
def _read_global(fid, nx: int, tmp_geom: dict = None) -> dict:
    """
    Read the numeric section of a **global** geometry file.
 
    The format is: a string token naming the array, followed by
    ``nx * nz`` floats reshaped to ``(nx, nz)``.  Repeated until EOF.
 
    Parameters
    ----------
    fid : file object
        Positioned just after the namelist ``/`` line.
    nx : int
        Number of radial grid points (``box.nx0``).
 
    Returns
    -------
    dict
        Partial geometry dict (without curvature/area).
    """
    # Read all remaining tokens
    content = fid.read()
    tokens = content.split()
 
    arrays = {}
    i = 0
    while i < len(tokens):
        # Try to parse as float; if it fails, it's an array name
        try:
            float(tokens[i])
            i += 1   # stray number, skip
        except ValueError:
            name = tokens[i].lower()
            i += 1
            # Collect all following floats
            floats = []
            while i < len(tokens):
                try:
                    floats.append(float(tokens[i]))
                    i += 1
                except ValueError:
                    break   # next token is a name
            if floats:
                n_vals = len(floats)
                nz = n_vals // nx
                arrays[name] = np.array(floats).reshape(nx, nz, order='F')
 
    # Canonical name aliases (GENE writes different names in different versions)
    def _get(d, *keys):
        for k in keys:
            if k in d:
                return d[k]
        return None
 
    Bfield   = _get(arrays, 'bfield')
    Jacobian = _get(arrays, 'jacobian')
    dBdx     = _get(arrays, 'dbdx')
    dBdy     = _get(arrays, 'dbdy')
    dBdz     = _get(arrays, 'dbdz')
    R        = _get(arrays, 'geo_r')
    Z        = _get(arrays, 'geo_z')
    Phi      = _get(arrays, 'geo_phi')
    dxdR     = _get(arrays, 'geo_c1', 'c_1', 'dxdr')
    dxdZ     = _get(arrays, 'geo_c2', 'c_2', 'dxdz')
 
    # Build metric from whatever metric arrays are present
    metric = {}
    for tag in ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']:
        val = _get(arrays, tag)
        if val is not None:
            metric[tag] = val
    metric['dxdR'] = dxdR
    metric['dxdZ'] = dxdZ

    # C_xy: try file arrays, then namelist, then derive from metric
    if tmp_geom is None:
        tmp_geom = {}
    C_xy = _get(arrays, 'c_xy', 'cxy')
    if C_xy is None and 'cxy' in tmp_geom:
        C_xy = tmp_geom['cxy']
    if C_xy is None and 'gxx' in metric and 'gyy' in metric and 'gxy' in metric:
        C_xy = np.sqrt(Bfield**2 / (metric['gxx'] * metric['gyy'] - metric['gxy']**2))
    if C_xy is not None:
        metric['C_xy'] = np.squeeze(C_xy) if hasattr(C_xy, 'squeeze') else C_xy

    # C_y: try file arrays, then namelist
    C_y = _get(arrays, 'c_y', 'cy')
    if C_y is None and 'cy' in tmp_geom:
        C_y = tmp_geom['cy']
    if C_y is not None:
        metric['C_y'] = np.squeeze(C_y) if hasattr(C_y, 'squeeze') else C_y
 
    # q profile (if present)
    q_prof = _get(arrays, 'q', 'q_prof')
 
    shape = dict(gR=R, gZ=Z, gPhi=Phi)
    local = dict(q0=None, shat=None, trpeps=None, gridpoints=None)
 
    geom = dict(
        Bfield=Bfield, Jacobian=Jacobian,
        dBdx=dBdx, dBdy=dBdy, dBdz=dBdz,
        metric=metric, shape=shape, local=local,
        dxdR=dxdR, dxdZ=dxdZ,
    )
    if q_prof is not None:
        geom['profiles'] = {'q': q_prof[:,0]}
 
    return geom
 
 
# ---------------------------------------------------------------------------
# HDF5 geometry reader (GENE `write_h5` and GENE-3D)
# ---------------------------------------------------------------------------

#: ``/metric`` dataset name -> the key this module uses.
_H5_METRIC = {"g^xx": "gxx", "g^xy": "gxy", "g^xz": "gxz",
              "g^yy": "gyy", "g^yz": "gyz", "g^zz": "gzz"}

#: ``/Bfield_terms`` datasets that map onto top-level geometry keys.
_H5_BFIELD = {"Bfield": "Bfield", "dBdx": "dBdx", "dBdy": "dBdy",
              "dBdz": "dBdz", "Jacobian": "Jacobian"}


def _h5_get(group, name):
    """
    Return one dataset from *group* in GENE's axis order, or ``None``.

    ``h5py.Group.get`` returns ``None`` for a missing name rather than raising,
    so callers must check the result — guarding these lookups with
    ``except AttributeError`` (as the GUI does) never fires and leaves ``None``
    in place of the intended fallback.

    The transpose reverses every axis, undoing the Fortran-to-C flip that
    ``futils.putarr`` introduced: ``(nz,)`` stays put, ``(nz, nx)`` becomes
    ``(nx, nz)``, and GENE-3D's ``(nz, ny, nx)`` becomes ``(nx, ny, nz)``.
    """
    if group is None:
        return None
    dset = group.get(name)
    if dset is None:
        return None
    return np.asarray(dset[...]).T


def _h5_scalar(group, name):
    """Return a length-1 ``/parameters`` dataset as a float, or ``None``."""
    arr = _h5_get(group, name)
    if arr is None:
        return None
    flat = np.ravel(arr)
    return float(flat[0]) if flat.size else None


def _read_geom_h5(fpath: str, params: dict) -> dict:
    """
    Read a geometry file written through *futils*.

    Covers both flavours, which share a group layout but not their contents:

    * **GENE** (``write_h5``) writes ``/shape/{R,Z,dxdR,dxdZ}`` and scalar
      ``C_y``/``C_xy``; metric and field terms are ``(nz,)`` for a flux tube
      and ``(nz, nx)`` for an x-global run.
    * **GENE-3D** writes 3-D ``(nz, ny, nx)`` metric and field terms, *array*
      ``C_y``/``C_xy`` over x, no ``/shape`` group at all, and a ``/profile``
      group carrying ``q_prof`` plus the flux-surface quantities ``dVdx``,
      ``sqrtgxx_fs`` and ``gxx_fs`` that it has already computed.

    Returns the same dictionary structure as the ASCII readers.
    """
    metric, profiles, extra = {}, {}, {}
    with h5py.File(fpath, "r") as f:
        g_metric = f.get("metric")
        g_bfield = f.get("Bfield_terms")
        g_shape = f.get("shape")
        g_prof = f.get("profile")
        g_pars = f.get("parameters")
        g_cart = f.get("cart_coords")

        for h5name, key in _H5_METRIC.items():
            val = _h5_get(g_metric, h5name)
            if val is not None:
                metric[key] = val
        for name in ("C_y", "C_xy"):
            val = _h5_get(g_metric, name)
            if val is not None:
                metric[name] = np.squeeze(val)[()] if val.size == 1 else val

        for h5name, key in _H5_BFIELD.items():
            extra[key] = _h5_get(g_bfield, h5name)

        # GENE-3D writes the curvature; GENE's flux-tube files may not.
        K_x = _h5_get(g_bfield, "K_x")
        K_y = _h5_get(g_bfield, "K_y")

        R = _h5_get(g_shape, "R")
        Z = _h5_get(g_shape, "Z")
        Phi = _h5_get(g_shape, "phi")
        dxdR = _h5_get(g_shape, "dxdR")
        dxdZ = _h5_get(g_shape, "dxdZ")

        for name in ("q_prof", "dpdx_pm_arr", "gxx_fs", "sqrtgxx_fs",
                     "dVdx", "xval_a"):
            val = _h5_get(g_prof, name)
            if val is not None:
                profiles["q" if name == "q_prof" else name] = val

        # GENE-3D writes the Cartesian position of every grid point, which is
        # what lets a snapshot be exported to a real-space 3-D viewer without
        # reconstructing the flux-surface mapping.
        cart = {}
        for name in ("x", "y", "z"):
            val = _h5_get(g_cart, name)
            if val is not None:
                cart[name] = val

        local = dict(
            q0=_h5_scalar(g_pars, "q0"),
            shat=_h5_scalar(g_pars, "shat"),
            trpeps=_h5_scalar(g_pars, "trpeps"),
            gridpoints=_h5_scalar(g_pars, "gridpoints"),
        )
        for name in ("beta", "minor_r", "major_R", "Bref", "Lref"):
            value = _h5_scalar(g_pars, name)
            if value is not None:
                local[name] = value

    metric["dxdR"] = dxdR
    metric["dxdZ"] = dxdZ

    geom = dict(
        Bfield=extra.get("Bfield"), Jacobian=extra.get("Jacobian"),
        dBdx=extra.get("dBdx"), dBdy=extra.get("dBdy"),
        dBdz=extra.get("dBdz"),
        metric=metric, shape=dict(gR=R, gZ=Z, gPhi=Phi), local=local,
        dxdR=dxdR, dxdZ=dxdZ,
    )
    if profiles:
        geom["profiles"] = profiles
    if len(cart) == 3:
        geom["cart_coords"] = cart
    if K_x is not None and K_y is not None:
        # Stashed for _compute_curvature to prefer over recomputing.
        geom["_curv_from_file"] = dict(K_x=K_x, K_y=K_y, sloc=None)
    return geom


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------
 
def _compute_curvature(geom: dict, params: dict) -> dict:
    """
    Compute magnetic curvature components K_x, K_y and local shear ``sloc``.
 
    Follows the MATLAB implementation exactly.
 
    Parameters
    ----------
    geom : dict
        Geometry dict (metric must be populated).
    params : dict
        Parameter dict for this segment.
 
    Returns
    -------
    dict
        ``{'K_x': ..., 'K_y': ..., 'sloc': ...}``
    """
    m = geom['metric']
    gxx, gxy, gxz = m['gxx'], m['gxy'], m['gxz']
    gyy, gyz       = m['gyy'], m['gyz']
 
    gamma1 = gxx * gyy - gxy**2
    gamma2 = gxx * gyz - gxy * gxz
    gamma3 = gxy * gyz - gyy * gxz
 
    K_x = -geom['dBdy'] - (gamma2 / gamma1) * geom['dBdz']
    K_y =  geom['dBdx'] - (gamma3 / gamma1) * geom['dBdz']
 
    # z grid
    nz   = params['box']['nz0']
    npol = params['geometry'].get('n_pol', 1)
    edge = params['geometry'].get('edge_opt', 0)
    z = build_zgrid(nz, npol, edge)
 
    # Local shear: d(gxy/gxx)/dz by central differences (one-sided at the
    # ends). Passing z rather than a spacing keeps this correct on the
    # non-uniform grid produced by edge_opt != 0. axis=-1 is z for both local
    # geometry, where the ratio is (nz,), and global, where it is (nx, nz).
    if nz > 1:
        try:
            sloc = np.gradient(gxy / gxx, z, axis=-1)
        except ValueError:
            sloc = np.full_like(gxy, np.nan)
    else:
        sloc = np.nan
 
    return dict(K_x=K_x, K_y=K_y, sloc=sloc)
 
 
# ---------------------------------------------------------------------------
# Area / volume element
# ---------------------------------------------------------------------------
 
def _get_area(geom: dict, params: dict) -> dict:
    """
    Compute flux-surface area and volume element dV/dx.
 
    For local geometry these are scalars; for global geometry they are
    1-D arrays over the radial grid.
 
    Parameters
    ----------
    geom : dict
        Geometry dict (metric and Jacobian must be populated).
    params : dict
        Parameter dict for this segment.
 
    Returns
    -------
    dict
        ``{'Area': ..., 'dVdx': ...}``
    """
    x_local = params['general'].get('x_local', True)
    is_3d   = bool(params.get('info', {}).get('is_3d', False))
    nz      = params['box']['nz0']
    Lref    = params.get('units', {}).get('Lref', 1.0)
    C_y     = geom['metric'].get('C_y', 1.0)
 
    A0 = (2*np.pi)**2 * abs(C_y) * Lref**2
 
    J   = geom['Jacobian']
    gxx = geom['metric'].get('gxx', np.ones_like(J))
 
    if is_3d:
        # J and gxx are (nx, ny, nz); average over the flux surface. GENE-3D
        # already writes dVdx and sqrtgxx_fs, so use those when present — but
        # divide out n_pol, which GENE-3D folds in (its dVdx spans the whole
        # n_pol-turn simulation domain) while the surface area of interest is
        # one turn. Identical for the usual n_pol = 1.
        n_pol = float(params.get('geometry', {}).get('n_pol', 1) or 1)
        stored = geom.get('profiles', {}) or {}
        if 'dVdx' in stored:
            dVdx = np.abs(np.asarray(stored['dVdx'], dtype=float)) / n_pol
            dVdx = dVdx * Lref**2
            sqrtgxx = stored.get('sqrtgxx_fs')
            Area = (dVdx * np.asarray(sqrtgxx, dtype=float)
                    if sqrtgxx is not None
                    else A0 * np.mean(J * np.sqrt(gxx), axis=(1, 2)))
        else:
            Area = A0 * np.mean(J * np.sqrt(gxx), axis=(1, 2))
            dVdx = A0 * np.mean(J, axis=(1, 2))
    elif x_local:
        Area  = A0 * np.sum(J * np.sqrt(gxx)) / nz
        dVdx  = A0 * np.sum(J) / nz
    else:
        # Global: sum along z axis (axis=1), result shape (nx,)
        Area  = A0 * np.sum(J * np.sqrt(gxx), axis=1) / nz
        dVdx  = A0 * np.sum(J, axis=1) / nz
 
    return dict(Area=Area, dVdx=dVdx)
 
 
# ---------------------------------------------------------------------------
# Single-segment loader
# ---------------------------------------------------------------------------
 
def _read_single_geom(folder: str, ext: str, params: dict) -> dict:
    """
    Load geometry for one run segment.
 
    Parameters
    ----------
    folder : str
        Run directory.
    ext : str
        File-name suffix (e.g. ``'_0001'``).
    params : dict
        Parameter dict for this segment.
 
    Returns
    -------
    dict
        Full geometry dictionary including curvature and area.
    """
    geom_type = params['geometry']['magn_geometry']
    x_local   = params['general'].get('x_local', True)
    nx        = params['box'].get('nx0', 1)
 
    fpath = _geometry_filename(folder, geom_type, ext)
    h5path = fpath + '.h5'
    if not os.path.isfile(fpath):
        # GENE-3D writes its geometry only as HDF5; GENE with write_h5 writes
        # both, and the ASCII form above is preferred there.
        if os.path.isfile(h5path):
            geom = _read_geom_h5(h5path, params)
            geom['kind'] = geom_type
            geom['curv'] = (geom.pop('_curv_from_file', None)
                            or _compute_curvature(geom, params))
            geom.pop('_curv_from_file', None)
            geom['area'] = _get_area(geom, params)
            return geom
        raise FileNotFoundError(
            f"Geometry file not found: {fpath} (nor {h5path})")
 
    with open(fpath, 'r') as fid:
        # ── 1. Read namelist (everything up to the first '/') ──────────
        namelist_lines = []
        for line in fid:
            if line.strip() == '/':
                break
            namelist_lines.append(line)
        namelist_text = ''.join(namelist_lines)
        tmp_geom = _parse_namelist(namelist_text)
 
        # ── 2. Read numeric data ───────────────────────────────────────
        if x_local:
            geom = _read_local(fid, tmp_geom)
        else:
            geom = _read_global(fid, nx, tmp_geom)
 
    geom['kind'] = geom_type
 
    # Local: fill in q0/shat/trpeps from namelist if missing
    if x_local:
        for key in ('q0', 'shat', 'trpeps', 'gridpoints'):
            if geom['local'].get(key) is None:
                geom['local'][key] = tmp_geom.get(key)
 
    # ── 3. Derived quantities ──────────────────────────────────────────
    geom['curv'] = _compute_curvature(geom, params)
    geom['area'] = _get_area(geom, params)
 
    return geom
 
 
# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
 
def Geometry(folder: str, extensions, params) -> list:
    """
    Load geometry for one or more GENE run segments.
 
    Parameters
    ----------
    folder : str
        Run directory containing the geometry files.
    extensions : str or list of str
        File-name suffix(es), e.g. ``'_0001'`` or ``['_0001', '_0002']``.
    params : Params
        Parameter object (as returned by :class:`~genetools.params.Params`).
 
    Returns
    -------
    list of dict
        One geometry dictionary per segment, in the same order as
        *extensions*.  See module docstring for dict structure.
 
    Raises
    ------
    FileNotFoundError
        If a geometry file cannot be found.
    """
    if isinstance(extensions, str):
        extensions = [extensions]
 
    return [
        _read_single_geom(folder, ext, params.get(fn))
        for fn, ext in enumerate(extensions)
    ]