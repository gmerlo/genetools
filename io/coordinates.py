import numpy as np
#from scipy.special import roots_laguerre


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
    """
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

            kx = (kx_modes * kxmin).reshape(-1)
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

    z = np.linspace(-np.pi * npol, np.pi * npol, nz + 1)[:-1]
    dz = z[1] - z[0] if nz > 1 else 0

    if edge != 0:
        logterm = np.log(edge * np.pi + np.sqrt((edge * np.pi)**2 + 1)) / np.pi
        arg = (-np.pi + 2 * np.pi * np.arange(nz) / nz) * logterm
        z = np.sinh(arg) / edge

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
