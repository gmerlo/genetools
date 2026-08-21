# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os

import h5py
import numpy as np

from genetools.io._zgrid import build_zgrid


# ----------------------------------------------------------------------
# Velocity-space grids shared by every geometry
# ----------------------------------------------------------------------
def _velocity_grids(params):
    """Return ``(vp, vp_weight, mu, mu_weight)`` from the namelist."""
    box = params["box"]
    vp = np.linspace(-box["lv"], box["lv"], box["nv0"])
    vp_weight = set_vp_weights(vp, params)

    nw = box["nw0"]
    mu_type = box["mu_grid_type"]
    if mu_type == "eq_vperp":
        deltamu = box["lw"] / nw**2
        idx = np.arange(1, nw + 1)
        mu = ((idx - 0.5) ** 2) * deltamu
        mu_weight = (2 * idx - 1) * deltamu
    elif mu_type == "gau_lag":
        # Gauss-Laguerre nodes scaled so the weights sum to lw, matching
        # GENE-3D's set_mu_coordinate_vars. Read from coord.h5 when available.
        mu, weights = np.polynomial.laguerre.laggauss(nw)
        weights = weights * np.exp(mu)
        mu = mu * (box["lw"] / np.sum(weights))
        mu_weight = weights * (box["lw"] / np.sum(weights))
    else:
        mu = np.array([])
        mu_weight = np.array([])
    return vp, vp_weight, mu, mu_weight


# ----------------------------------------------------------------------
# GENE-3D: real space in x, y and z
# ----------------------------------------------------------------------
def _coord_file(folder, ext):
    """Return the path of the GENE-3D ``coord<ext>.h5`` file, or ``None``."""
    if folder is None:
        return None
    path = os.path.join(str(folder), f"coord{ext}.h5")
    return path if os.path.isfile(path) else None


def _load_coord_h5(path):
    """Read whatever GENE-3D's ``coord<ext>.h5`` provides, keyed by name."""
    out = {}
    if not path:
        return out
    with h5py.File(path, "r") as f:
        coord = f.get("coord")
        if coord is None:                       # pragma: no cover - malformed
            return out
        for name in ("xval", "xval_a", "yval", "zval",
                     "vp", "vp_weight", "mu", "mu_weight"):
            if name in coord:
                out[name] = np.asarray(coord[name][...], dtype=np.float64)
    return out


def load_coord_xy_global(folder, ext, params):
    """
    Build coordinate arrays for a GENE-3D run (real space in x *and* y).

    GENE-3D writes its grids to ``coord<ext>.h5``, and those are preferred over
    reconstruction: the namelist alone is not quite enough to pin them down.
    The radial spacing depends on ``rad_bc_type`` (``lx/nx0`` for a periodic
    boundary, ``lx/(nx0-1)`` otherwise), and the parallel grid is shifted by
    half a cell when ``nz0`` is odd — both easy to get wrong, and both already
    settled in the file.

    Unlike the spectral cases there is no ``nky0`` or ``kymin`` anywhere in the
    parameters file, so the binormal wavenumbers are the FFT grid of ``yval``:
    ``ky`` is returned in FFT order (signed, length ``ny0``) so ``ky[j]``
    lines up with ``np.fft.fft(var, axis=y)[j]``, and ``ky_pos`` holds just the
    non-negative half.
    """
    box = params["box"]
    geom = params["geometry"]
    nonlocal_x = params.get("nonlocal_x", {})

    nx = int(box["nx0"])
    ny = int(box["ny0"])
    nz = int(box["nz0"])
    lx = float(box["lx"])
    ly = float(box["ly"])
    x0 = float(box["x0"])
    rhostar = float(geom["rhostar"])
    n_pol = int(geom.get("n_pol", 1) or 1)

    stored = _load_coord_h5(_coord_file(folder, ext)) if folder is not None else {}

    # --- x ------------------------------------------------------------
    if "xval" in stored and stored["xval"].size == nx:
        x = stored["xval"]
    else:
        rad_bc_type = int(nonlocal_x.get("rad_bc_type", 0) or 0)
        dx_ = lx / nx if rad_bc_type == 0 else lx / (nx - 1)
        x = -0.5 * lx + np.arange(nx) * dx_ + x0 / rhostar
    x_over_a = (stored["xval_a"] if "xval_a" in stored
                and stored["xval_a"].size == nx else x * rhostar)
    dx = x[1] - x[0] if nx > 1 else 0.0

    # --- y ------------------------------------------------------------
    # GENE-3D's yval runs 0 .. ly-dy, not -ly/2 .. ly/2.
    y = (stored["yval"] if "yval" in stored and stored["yval"].size == ny
         else np.arange(ny) * (ly / ny))
    dy = y[1] - y[0] if ny > 1 else 0.0

    # --- z ------------------------------------------------------------
    if "zval" in stored and stored["zval"].size == nz:
        z = stored["zval"]
    else:
        dz_ = 2.0 * np.pi * n_pol / nz
        z = -np.pi * n_pol + np.arange(nz) * dz_
        if nz % 2 != 0:
            z = z + dz_ / 2.0
    dz = z[1] - z[0] if nz > 1 else 0.0

    # --- wavenumbers (FFT grids; nothing spectral is stored on disk) ---
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=(lx / nx))
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=(ly / ny))

    # --- velocity space ------------------------------------------------
    # Reconstruct, then let the file win wherever it has an answer.
    vp, vp_weight, mu, mu_weight = _velocity_grids(params)
    vp = stored.get("vp", vp)
    vp_weight = stored.get("vp_weight", vp_weight)
    mu = stored.get("mu", mu)
    mu_weight = stored.get("mu_weight", mu_weight)

    return {
        "x": x,
        "x_o_a": x_over_a,
        "kx": kx,
        "kx_2": kx[: nx // 2 + 1],
        "dx": dx,
        "y": y,
        "dy": dy,
        "ny": ny,
        "ky": ky,
        # Non-negative half in FFT order. For even ny the Nyquist bin at
        # index ny//2 is signed negative by fftfreq, so it is excluded.
        "ky_pos": ky[: (ny + 1) // 2],
        "z": z,
        "dz": dz,
        "n_pol": n_pol,
        "mu": mu,
        "mu_weight": mu_weight,
        "vp": vp,
        "vp_weight": vp_weight,
    }


# ----------------------------------------------------------------------
# Velocity–parallel weights
# ----------------------------------------------------------------------
def set_vp_weights(vp, params):
    """
    Compute velocity-parallel quadrature weights.

    Applies GENE's endpoint corrections if:
      - collision operator is linear
      - not using arakawa_zv
      - nv0 > 8
    """
    dv = vp[1] - vp[0]
    weights = np.full_like(vp, dv)

    general = params["general"]
    box = params["box"]

    use_endpoint_corr = (
        general["collision_op"] not in {"nonlin", "sugama", "exact"}
        and not general["arakawa_zv"]
        and box["nv0"] > 8
    )

    if use_endpoint_corr:
        w = np.array([17, 59, 43, 49]) * dv / 48.0
        weights[:4] = w
        weights[-4:] = w[::-1]  # symmetric

    return weights


# ----------------------------------------------------------------------
# Load coordinates for a single run
# ----------------------------------------------------------------------
def load_coord_single_run(folder, file_number, params):
    """
    Generate coordinate arrays (x,y,z,kx,ky,vp,mu) for a single GENE run.

    GENE-3D runs are real-space in x and y and carry no ``nky0``/``kymin`` at
    all, so they take a separate path (:func:`load_coord_xy_global`) rather
    than a branch inside the spectral construction below.
    """
    if params.get("info", {}).get("geometry_kind") == "xy_global":
        return load_coord_xy_global(folder, file_number, params)

    box = params["box"]
    geom = params["geometry"]
    general = params["general"]

    # ==============================================================
    # X / KX
    # ==============================================================
    if general["x_local"]:
        nx = box["nx0"]
        lx = box["lx"]

        if nx == 1:
            kx = np.array([box.get("kx_center", 0)])
            kx_2 = []
            x = np.array([0.0])
            dx = 0.0
            x_over_a = []
        else:
            kxmin = 2 * np.pi / lx
            half = nx // 2

            kx_pos = np.arange(half + 1)
            if nx % 2 == 0:
                kx_modes = np.concatenate((kx_pos, -kx_pos[1:-1][::-1]))
            else:
                kx_modes = np.concatenate((kx_pos, -kx_pos[1:][::-1]))

            kx = kx_modes * kxmin
            kx_2 = kx[:half + 1]

            x = np.linspace(-lx / 2, lx / 2, nx + 1)[:-1]
            dx = x[1] - x[0]
            x_over_a = []
    else:
        # Non-local geometry definitions
        try:
            rhostar = geom["rhostar"]
            nx = box["nx0"]
            x_over_a = np.linspace(
                box["x0"] - box["lx"] * rhostar / 2,
                box["x0"] + box["lx"] * rhostar / 2,
                nx
            )
            x = x_over_a / rhostar
            dx = x[1] - x[0] if nx > 1 else 0
            kx = []
            kx_2 = []
        except Exception:
            # Fallback (GENE legacy)
            nx = box["nx0"]
            x = np.arange(1, nx + 1)
            x_over_a = x
            kx = []
            kx_2 = []
            dx = 1.0

    # ==============================================================
    # KY / Y
    # ==============================================================
    nky = box["nky0"]
    kymin = box["kymin"]

    ky = np.array([kymin]) if nky == 1 else kymin * np.arange(nky)
    Ly = 2 * np.pi / kymin
    ny = 2 * nky if nky > 1 else 50

    y = np.linspace(-Ly / 2, Ly / 2, ny + 1)[:-1]
    dy = y[1] - y[0]

    # ==============================================================
    # Z (ballooning coordinate)
    # ==============================================================
    nz = box["nz0"]
    npol = geom["n_pol"]
    edge = geom["edge_opt"]

    z = build_zgrid(nz, npol, edge)
    dz = z[1] - z[0] if nz > 1 else 0

    # ==============================================================
    # VP
    # ==============================================================
    vp = np.linspace(-box["lv"], box["lv"], box["nv0"])
    vp_weight = set_vp_weights(vp, params)

    # ==============================================================
    # MU
    # ==============================================================
    nw = box["nw0"]
    mu_type = box["mu_grid_type"]

    if mu_type == "gau_lag":
        mu = np.array([])
        mu_weight = np.array([])
        # TODO: mu, mu_weight = roots_laguerre(nw)
    elif mu_type == "eq_vperp":
        deltamu = box["lw"] / nw**2
        idx = np.arange(1, nw + 1)
        mu = ((idx - 0.5)**2) * deltamu
        mu_weight = (2 * idx - 1) * deltamu
    else:
        mu = np.array([])
        mu_weight = np.array([])

    return {
        "x": x,
        "x_o_a": x_over_a,
        "kx": kx,
        "kx_2": kx_2,
        "dx": dx,
        "y": y,
        "dy": dy,
        "ky": ky,
        "z": z,
        "dz": dz,
        "mu": mu,
        "mu_weight": mu_weight,
        "vp": vp,
        "vp_weight": vp_weight,
    }


# ----------------------------------------------------------------------
# Multi-run interface
# ----------------------------------------------------------------------
def Coordinates(folder, file_number, parameters):
    """
    Load coordinates for one or multiple GENE runs.
    """
    return [load_coord_single_run(folder, ext, parameters.get(fn))
            for fn, ext in enumerate(file_number)]
