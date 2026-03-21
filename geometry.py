import os
import re
import numpy as np


def Geometry(folder, extensions, params):
    """
    Reads geometry for *every* extension supplied.
    Returns a list of geometry dictionaries.
    """

    if isinstance(extensions, str):
        extensions = [extensions]     # normalize to list

    results = []

    for ir, extension in enumerate(extensions):
        geom_dict = _read_single_geom(folder, extension, params.get(ir))
        results.append(geom_dict)

    return results



def _read_single_geom(folder, extension, params):

    # ------------------- build filename -------------------
    try:
        geom_type = params["geometry"]["magn_geometry"]
    except KeyError:
        raise KeyError("params['geometry']['magn_geometry'] missing")

    fname = os.path.join(folder, geom_type + extension)
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Geometry file not found: {fname}")

    # ------------------- read file ------------------------
    with open(fname, "r") as f:
        content = f.read()

    # ---------------------------------------------------------
    # 1) Extract namelist up to "/"
    # ---------------------------------------------------------
    namelist_text = content.split("/", 1)[0]

    tmp_geom = {}
    for key, value in re.findall(r'(\w+)\s*=\s*([^\s]+)', namelist_text):
        try:
            tmp_geom[key.lower()] = float(value)
        except ValueError:
            tmp_geom[key.lower()] = value.strip()
    
    # ---------------------------------------------------------
    # 2) Extract all numeric values
    # ---------------------------------------------------------
    numbers = np.array(
        re.findall(
            r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?'
            r'|[-+]?\d+(?:[eE][-+]?\d+)?',
            content.split("/", 1)[1]
        ), dtype=float
    )

    # ---------------------------------------------------------
    # 3) LOCAL GEOMETRY
    # ---------------------------------------------------------
    x_local = params["general"].get("x_local", False)

    if x_local:
        needed_cols = 16

        if len(numbers) % needed_cols != 0:
            raise ValueError(
                f"Local geometry expects {needed_cols} columns but got {len(numbers)} numbers"
            )

        coeffs = numbers.reshape(-1, needed_cols)

        # metric components
        gxx, gxy, gxz = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]
        gyy, gyz, gzz = coeffs[:, 3], coeffs[:, 4], coeffs[:, 5]

        Bfield, dBdx, dBdy, dBdz = coeffs[:, 6], coeffs[:, 7], coeffs[:, 8], coeffs[:, 9]
        Jacobian = coeffs[:, 10]

        R, Phi, Z = coeffs[:, 11], coeffs[:, 12], coeffs[:, 13]
        dxdR, dxdZ = coeffs[:, 14], coeffs[:, 15]

        # Derived defaults
        C_xy = tmp_geom.get(
            "cxy",
            np.sqrt(Bfield[0] ** 2 / (gxx[0] * gyy[0] - gxy[0] ** 2))
        )
        C_y = tmp_geom.get("cy", 1.0)

        metric = dict(
            C_xy=C_xy, C_y=C_y,
            gxx=gxx, gxy=gxy, gxz=gxz,
            gyy=gyy, gyz=gyz, gzz=gzz,
            dxdR=dxdR, dxdZ=dxdZ
        )

        shape = dict(gR=R, gZ=Z, gPhi=Phi)

        return dict(
            kind=geom_type,
            Bfield=Bfield,
            Jacobian=Jacobian,
            dBdx=dBdx, dBdy=dBdy, dBdz=dBdz,
            metric=metric,
            shape=shape,
            local=dict(
                q0=tmp_geom.get("q0"),
                shat=tmp_geom.get("shat"),
                trpeps=tmp_geom.get("trpeps"),
                gridpoints=tmp_geom.get("gridpoints")
            ),
            dxdR=dxdR,
            dxdZ=dxdZ
        )

    # ---------------------------------------------------------
    # 4) GLOBAL GEOMETRY
    # ---------------------------------------------------------
    try:
        x_res = params["box"]["nx0"]
    except KeyError:
        raise KeyError("params['box']['nx0'] missing for global geometry")

    if len(numbers) % x_res != 0:
        raise ValueError(
            f"Global geometry expects multiples of x_res={x_res}, but got {len(numbers)} numbers"
        )

    n_arrays = len(numbers) // x_res
    arrays = numbers.reshape(n_arrays, x_res)

    labels = [
        "geo_R", "geo_Z", "geo_phi",
        "Bfield", "jacobian",
        "dBdx", "dBdy", "dBdz",
        "geo_c1", "geo_c2"
    ]

    if n_arrays < len(labels):
        raise ValueError(
            f"Expected >= {len(labels)} geometry arrays but got {n_arrays}"
        )

    geom_arrays = {lbl: arrays[i] for i, lbl in enumerate(labels)}

    shape = dict(
        gR=geom_arrays["geo_R"],
        gZ=geom_arrays["geo_Z"],
        gPhi=geom_arrays["geo_phi"]
    )

    return dict(
        kind=geom_type,
        Bfield=geom_arrays["Bfield"],
        Jacobian=geom_arrays["jacobian"],
        dBdx=geom_arrays["dBdx"],
        dBdy=geom_arrays["dBdy"],
        dBdz=geom_arrays["dBdz"],
        metric={},
        shape=shape,
        local=dict(
            q0=tmp_geom.get("q0"),
            shat=tmp_geom.get("shat"),
            trpeps=tmp_geom.get("trpeps"),
            gridpoints=tmp_geom.get("gridpoints")
        ),
        dxdR=geom_arrays["geo_c1"],
        dxdZ=geom_arrays["geo_c2"]
    )

