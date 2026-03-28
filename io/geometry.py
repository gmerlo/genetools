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
    ``profiles``   dict: q  (global geometry only)
 
Example
-------
>>> from genetools.geometry import Geometry
>>> geoms = Geometry('/path/to/run/', ['_0001', '_0002'], params)
>>> J = geoms[0]['Jacobian']   # shape (nz,) for local, (nx, nz) for global
"""
 
import os
import re
import numpy as np
from scipy.interpolate import CubicSpline
 
 
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
 
def _read_global(fid, nx: int) -> dict:
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
 
    # z grid (same as coordinates.py)
    nz   = params['box']['nz0']
    npol = params['geometry'].get('n_pol', 1)
    edge = params['geometry'].get('edge_opt', 0)
 
    z = np.linspace(-np.pi * npol, np.pi * npol, nz + 1)[:-1]
 
    if edge != 0:
        k = np.arange(nz)
        z = np.sinh(
            (-np.pi + k * 2*np.pi / nz)
            * np.log(edge*np.pi + np.sqrt(edge**2 * np.pi**2 + 1)) / np.pi
        ) / edge
 
    # Local shear: d(gxy/gxx)/dz via cubic spline
    if nz > 1:
        try:
            ratio = gxy / gxx
            # CubicSpline expects y shape (n, ...) where n == len(x).
            # For local geometry ratio is (nz,); for global it's (nx, nz).
            if ratio.ndim == 2:
                # Global: transpose to (nz, nx), spline, transpose back
                cs   = CubicSpline(z, ratio.T)
                sloc = cs(z, 1).T   # first derivative, back to (nx, nz)
            else:
                cs   = CubicSpline(z, ratio)
                sloc = cs(z, 1)     # first derivative
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
    nz      = params['box']['nz0']
    Lref    = params.get('units', {}).get('Lref', 1.0)
    C_y     = geom['metric'].get('C_y', 1.0)
 
    A0 = (2*np.pi)**2 * abs(C_y) * Lref**2
 
    J   = geom['Jacobian']
    gxx = geom['metric'].get('gxx', np.ones_like(J))
 
    if x_local:
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
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Geometry file not found: {fpath}")
 
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
            geom = _read_global(fid, nx)
 
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