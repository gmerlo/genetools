"""
Synthetic GENE-3D run directories for tests.

There is no GENE-3D output in this repository, so the readers are exercised
against directories built here to match what ``gene3d-dev/src`` actually
writes. The layout below is transcribed from the Fortran, and the details that
are easy to get wrong are called out where they are reproduced:

* **futils axis order.** ``putarr`` hands HDF5 a Fortran-ordered array, so a
  ``(nx0, ny0, nz0)`` field lands on disk as a ``(nz0, ny0, nx0)`` dataset.
* **futils precision.** ``creatf(..., 's')`` stores 32-bit reals, so
  ``field``/``mom``/``vsp``/``srcmom`` are ``float32`` no matter what
  ``PRECISION`` says; ``coord``/geometry/``profiles`` use ``'d'`` and are
  ``float64``.
* **Real data.** GENE-3D is real-space in x *and* y, so field and moment
  snapshots are real, not complex.
* **Snapshot groups.** Each variable is a group of ``%010d``-named datasets
  alongside one extendible ``time`` dataset.
* **``n_moms`` is wrong on purpose.** ``parameters`` reports ``n_moms = 6``
  (``par_in``) while ten moment datasets are written (``diag_3d``). Readers
  must introspect the file rather than trust the namelist.
"""

from pathlib import Path

import h5py
import numpy as np

# GENE-3D's diag_3d.F90 mom_label, in writing order.
MOM_LABELS = ["n", "u_par", "T_par", "T_per", "Gamma_es", "Gamma_em",
              "Q_es", "Q_em", "q_par", "q_perp"]

# diag_3d.F90 field_label, truncated to n_fields by the caller.
FIELD_LABELS = ["phi", "A_par", "B_par"]

# diag_3d.F90 vsp_label.
VSP_LABELS = ["G_es", "G_em", "Q_ese", "Q_eme", "<f_>"]

# diag_3d.F90 srcmom_label_1 x srcmom_label_2 — six datasets, not the nine the
# GUI's variable map expects (there are no f0_term_* moments in GENE-3D).
SRCMOM_LABELS = [f"{a}_{b}" for a in ("ck_heat", "ck_part")
                 for b in ("M00", "M10", "M22")]

GEOM_METRIC = ["g^xx", "g^xy", "g^xz", "g^yy", "g^yz", "g^zz"]
GEOM_BFIELD = ["Bfield", "dBdx", "dBdy", "dBdz", "Jacobian", "K_x", "K_y"]


# ---------------------------------------------------------------------------
# Grids — mirroring gene3d-dev/src/coordinates.F90
# ---------------------------------------------------------------------------

def gene3d_grids(nx0, ny0, nz0, lx, ly, x0, rhostar, n_pol=1,
                 rad_bc_type=0):
    """
    Return the ``(xval, xval_a, yval, zval)`` grids GENE-3D would write.

    Transcribed from ``set_x_coordinate_vars`` / ``set_y_coordinate_vars`` /
    ``set_z_coordinate_vars``: the radial spacing depends on the radial
    boundary condition, ``yval`` starts at zero rather than being centred, and
    ``zval`` picks up a half-cell shift when ``nz0`` is odd.
    """
    dx = lx / nx0 if rad_bc_type == 0 else lx / (nx0 - 1)
    xval = -0.5 * lx + np.arange(nx0) * dx + x0 / rhostar
    xval_a = xval * rhostar

    yval = np.arange(ny0) * (ly / ny0)

    dz = 2.0 * np.pi * n_pol / nz0
    zval = -np.pi * n_pol + np.arange(nz0) * dz
    if nz0 % 2 != 0:
        zval = zval + dz / 2.0

    return xval, xval_a, yval, zval


# ---------------------------------------------------------------------------
# parameters file — mirroring gene3d-dev/src/parameters_IO.F90
# ---------------------------------------------------------------------------

def gene3d_parameters(nx0=6, ny0=8, nz0=4, nv0=8, nw0=4, n_spec=2,
                      n_fields=2, lx=60.0, ly=80.0, x0=0.5,
                      rhostar=0.01, minor_r=1.0, major_R=3.0,
                      species=("ions", "electrons"),
                      magn_geometry="circular", n_pol=1,
                      norm_flux_projection=True, radial_dependence=True,
                      nonlinear=True, beta=0.001):
    """
    Return the text of a GENE-3D ``parameters`` file.

    Note where things live: ``x_local``/``y_local``/``write_h5``/``nrgcols``
    and ``ly`` are in ``&info``, not ``&general``/``&in_out``/``&box``, and
    ``&box`` carries ``ny0`` with no ``nky0`` or ``kymin`` at all.
    """
    spec_blocks = []
    for i, name in enumerate(species[:n_spec]):
        charge = -1 if name.startswith("e") else 1
        mass = 0.0002723 if name.startswith("e") else 1.0
        spec_blocks.append(
            "&species\n"
            f"name   = '{name}'\n"
            "prof_type = 2\n"
            "kappa_T   =    6.960000\n"
            "LT_center =   0.5000000\n"
            "LT_width  =   0.2000000\n"
            "kappa_n   =    2.230000\n"
            "Ln_center =   0.5000000\n"
            "Ln_width  =   0.2000000\n"
            f"mass   =   {mass}\n"
            "temp   =    1.000000\n"
            "dens   =    1.000000\n"
            f"charge = {charge}\n"
            "/\n")

    return f"""&parallelization
n_procs_s =   1
/

&box
n_spec = {n_spec}
nx0    = {nx0}
ny0   = {ny0}
nz0    = {nz0}
nv0    = {nv0}
nw0    = {nw0}

lv    =  3.00
lw    =  9.00
lx    = {lx}
lx_a    = {lx * rhostar}
x0    = {x0}
n0_global =      1
/

&in_out
diagdir = './'

read_checkpoint  = F
write_checkpoint = T

istep_field  =     10
istep_mom    =     10
istep_nrg    =     10
istep_prof   =     10
istep_vsp    =      0
istep_srcmom =      0
/

&general
nonlinear = {'T' if nonlinear else 'F'}
comp_type = 'IV'
calc_dt = T
beta       =   {beta}
debye2     =   0.0000000
collision_op = 'none'
/

&nonlocal_x
rad_bc_type =  0
z_bc_type =  1
/

&geometry
magn_geometry = '{magn_geometry}'
major_R  =   {major_R}
minor_r  =   {minor_r}
dpdx_term = 'full_drift'
rhostar  =   {rhostar}
alpha  =   0.0000000
radial_dependence   = {'T' if radial_dependence else 'F'}
binormal_dependence   = F
coord_type  = 1
norm_flux_projection  = {'T' if norm_flux_projection else 'F'}
n_pol = {n_pol}
/

{''.join(spec_blocks)}
&info
probdir = './'
q0     =   1.4000000
sign_Bpol_CW  =   1.0000000
init_time =     0.0000
n_fields = {n_fields}
n_moms   =  6
lx = {lx}
ly = {ly}
PRECISION  = DOUBLE
x_local   = F
y_local   = F
write_h5  = T
nrgcols   =  8
/

&units
Bref =   2.0000000
Tref =   1.0000000
nref =   1.0000000
Lref =   {major_R}
mref =   2.0000000
/
"""


# ---------------------------------------------------------------------------
# HDF5 writers — mirroring futils putarr/creatd/append
# ---------------------------------------------------------------------------

def _put3d(group, name, arr):
    """Store an ``(nx, ny, nz)`` array the way ``futils.putarr`` would."""
    group.create_dataset(name, data=np.asarray(arr).T)


def _write_snapshots(path, prefix, labels, times, arrays, dtype=np.float32):
    """
    Write one GENE-3D snapshot file.

    *arrays* is indexed ``[label][it]`` and holds ``(nx, ny, nz)`` arrays; each
    is transposed on the way in, matching ``putarr``. ``time`` is a plain 1-D
    dataset here — ``creatd`` makes it extendible, which a reader never has to
    care about.
    """
    with h5py.File(path, "w") as f:
        f.attrs["prec"] = "s" if dtype == np.float32 else "d"
        grp = f.create_group(prefix)
        grp.create_dataset("time", data=np.asarray(times, dtype=np.float64))
        for label in labels:
            sub = grp.create_group(label)
            for it in range(len(times)):
                dset = sub.create_dataset(
                    f"{it:010d}",
                    data=np.asarray(arrays[label][it], dtype=dtype).T)
                dset.attrs["time"] = float(times[it])


def _field_values(shape, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal(shape)


def _smooth_periodic(shape, seed, n_modes=3):
    """
    A smooth field, periodic in y, band-limited well below Nyquist.

    Random noise is fine for shape and dtype checks, but a flux identity has to
    be tested on something whose y-derivative is represented exactly by
    ``i k_y`` on the discrete grid — so the field is built from a handful of
    resolved binormal modes.
    """
    nx, ny, nz = shape
    rng = np.random.default_rng(seed)
    y = 2.0 * np.pi * np.arange(ny) / ny
    out = np.zeros(shape)
    for m in range(1, n_modes + 1):
        amp = rng.standard_normal((nx, 1, nz))
        phase = rng.uniform(0, 2 * np.pi, size=(nx, 1, nz))
        out = out + amp * np.cos(m * y[np.newaxis, :, np.newaxis] + phase)
    return out


def _dy_spectral(var, ly):
    """
    Exact y-derivative of a band-limited field, via its Fourier representation.

    This is how ``i k_y`` acts on the discrete grid, so a flux built with it and
    a flux built from the ky spectrum agree to round-off rather than to the
    accuracy of a finite-difference stencil.
    """
    ny = var.shape[1]
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    return np.real(np.fft.ifft(
        1j * ky[np.newaxis, :, np.newaxis] * np.fft.fft(var, axis=1), axis=1))


def _physical_fields_and_moments(shape, ly, geometry, n0, T0, field_labels,
                                 species, n_times, norm_flux_projection):
    """
    Build fields and moments that satisfy GENE-3D's own flux definitions.

    ``diag_3d.F90`` computes, with ``flux_geomfac`` from ``geometry.F90``::

        Gamma_es = -n            * dphi/dy   * flux_geomfac
        Q_es     = -[n_0 (T_par/2 + T_per) + 3/2 T_0 n] * dphi/dy * flux_geomfac
        Gamma_em = +u_par        * dA_par/dy * flux_geomfac
        Q_em     = +(q_par + q_perp)         * dA_par/dy * flux_geomfac

    Generating the primitive quantities and *deriving* the fluxes makes the
    relationship between them exactly the one the code produces, so a
    diagnostic that rebuilds a flux from ``phi`` and the moments can be checked
    against the flux GENE-3D wrote.
    """
    fgf = 1.0 / geometry["C_xy"][:, np.newaxis, np.newaxis]
    if norm_flux_projection:
        fgf = fgf / np.sqrt(geometry["g^xx"])

    fields = {lbl: [] for lbl in field_labels}
    moments = {sp: {lbl: [] for lbl in MOM_LABELS} for sp in species}

    nx_ = n0[:, np.newaxis, np.newaxis]
    tx_ = T0[:, np.newaxis, np.newaxis]

    for it in range(n_times):
        phi = _smooth_periodic(shape, 2000 + it)
        dphi_dy = _dy_spectral(phi, ly)
        fields["phi"].append(phi)

        a_par = None
        if "A_par" in fields:
            a_par = _smooth_periodic(shape, 3000 + it)
            fields["A_par"].append(a_par)
        if "B_par" in fields:
            fields["B_par"].append(_smooth_periodic(shape, 3500 + it))
        da_dy = _dy_spectral(a_par, ly) if a_par is not None else None

        for s, sp in enumerate(species):
            base = 4000 + 1000 * s + it
            n = _smooth_periodic(shape, base + 1)
            u_par = _smooth_periodic(shape, base + 2)
            t_par = _smooth_periodic(shape, base + 3)
            t_per = _smooth_periodic(shape, base + 4)
            q_par = _smooth_periodic(shape, base + 5)
            q_perp = _smooth_periodic(shape, base + 6)

            energy = (0.5 * t_par + t_per) * nx_ + 1.5 * n * tx_
            gamma_es = -n * dphi_dy * fgf
            q_es = -energy * dphi_dy * fgf
            if da_dy is not None:
                gamma_em = u_par * da_dy * fgf
                q_em = (q_par + q_perp) * da_dy * fgf
            else:
                gamma_em = np.zeros(shape)
                q_em = np.zeros(shape)

            for label, arr in (("n", n), ("u_par", u_par), ("T_par", t_par),
                               ("T_per", t_per), ("Gamma_es", gamma_es),
                               ("Gamma_em", gamma_em), ("Q_es", q_es),
                               ("Q_em", q_em), ("q_par", q_par),
                               ("q_perp", q_perp)):
                moments[sp][label].append(arr)

    return fields, moments


# ---------------------------------------------------------------------------
# Whole-run builder
# ---------------------------------------------------------------------------

class Gene3DRun:
    """Paths and ground-truth arrays for one synthetic GENE-3D run."""

    def __init__(self, folder, ext, nx0, ny0, nz0, n_fields, species,
                 times, fields, moments, grids, geometry, profiles):
        self.folder = Path(folder)
        self.ext = ext
        self.nx0, self.ny0, self.nz0 = nx0, ny0, nz0
        self.n_fields = n_fields
        self.species = list(species)
        self.times = np.asarray(times)
        self.fields = fields          # {label: [ (nx,ny,nz), ... ]}
        self.moments = moments        # {spec: {label: [ (nx,ny,nz), ... ]}}
        self.grids = grids            # {xval, xval_a, yval, zval, vp, mu, ...}
        self.geometry = geometry      # {name: array}
        self.profiles = profiles      # {spec: {T, n, omt, omn, x_o_a, ...}}

    @property
    def field_labels(self):
        return FIELD_LABELS[:self.n_fields]


def make_gene3d_run(tmp_path, ext=".dat", nx0=6, ny0=8, nz0=4, nv0=8, nw0=4,
                    n_fields=2, n_times=3, species=("ions", "electrons"),
                    magn_geometry="circular", lx=60.0, ly=80.0, x0=0.5,
                    rhostar=0.01, minor_r=1.0, major_R=3.0, n_pol=1,
                    write_vsp=False, write_srcmom=False,
                    write_profile_diag=False, nonlinear=True,
                    norm_flux_projection=True, physical=False):
    """
    Build a complete synthetic GENE-3D run directory under *tmp_path*.

    Returns a :class:`Gene3DRun` carrying the ground-truth arrays so tests can
    assert on values, not just shapes.

    With ``physical=True`` the fields and moments are smooth and the fluxes are
    *derived* from them exactly as ``diag_3d.F90`` does, so a diagnostic that
    rebuilds a flux from ``phi`` and the moments can be checked against the flux
    GENE-3D itself would have written. The default fills every array with
    independent noise, which is cheaper and enough for shape, dtype and
    plumbing tests.
    """
    folder = Path(tmp_path)
    folder.mkdir(parents=True, exist_ok=True)
    species = list(species)
    n_spec = len(species)
    shape = (nx0, ny0, nz0)

    # --- parameters -------------------------------------------------------
    (folder / f"parameters{ext}").write_text(gene3d_parameters(
        nx0=nx0, ny0=ny0, nz0=nz0, nv0=nv0, nw0=nw0, n_spec=n_spec,
        n_fields=n_fields, lx=lx, ly=ly, x0=x0, rhostar=rhostar,
        minor_r=minor_r, major_R=major_R, species=species,
        magn_geometry=magn_geometry, n_pol=n_pol, nonlinear=nonlinear,
        norm_flux_projection=norm_flux_projection))

    # --- coord<ext>.h5 ----------------------------------------------------
    xval, xval_a, yval, zval = gene3d_grids(
        nx0, ny0, nz0, lx, ly, x0, rhostar, n_pol=n_pol)
    vp = np.linspace(-3.0, 3.0, nv0)
    vp_weight = np.full(nv0, vp[1] - vp[0])
    mu, mu_weight = np.polynomial.laguerre.laggauss(nw0)
    grids = {"xval": xval, "xval_a": xval_a, "yval": yval, "zval": zval,
             "vp": vp, "vp_weight": vp_weight, "mu": mu,
             "mu_weight": mu_weight}
    with h5py.File(folder / f"coord{ext}.h5", "w") as f:
        f.attrs["prec"] = "d"
        grp = f.create_group("coord")
        for name, arr in grids.items():
            grp.create_dataset(name, data=np.asarray(arr, dtype=np.float64))

    # --- <magn_geometry><ext>.h5 -----------------------------------------
    rng = np.random.default_rng(7)

    def _metric_field(positive):
        """
        An (nx, ny, nz) array with no y dependence.

        GENE-3D's metric is y-independent today — ``geometry.F90`` gathers only
        over z and warns that the y gathering still has to be added "once we
        really have y-dep. metrics". Diagnostics that average over x and z at
        fixed ky rely on that, so the fixture has to reproduce it or it would
        test against a case the code cannot produce.
        """
        base = 1.0 + 0.1 * rng.standard_normal((nx0, 1, nz0))
        if positive:
            base = np.abs(base) + 0.5
        return np.broadcast_to(base, shape).copy()

    geometry = {}
    for name in GEOM_METRIC:
        geometry[name] = _metric_field(name in ("g^xx", "g^yy", "g^zz"))
    for name in GEOM_BFIELD:
        geometry[name] = _metric_field(name in ("Bfield", "Jacobian"))
    C_y = 0.5 + 0.01 * np.arange(nx0)
    C_xy = 0.8 + 0.01 * np.arange(nx0)
    q_prof = 1.4 + 0.5 * np.linspace(0, 1, nx0)
    dpdx_pm_arr = np.zeros(nx0)
    # dVdx and sqrtgxx_fs exactly as geometry.F90 computes them, so a reader
    # that trusts the file agrees with one that recomputes from the metric.
    avg_jaco_yz = geometry["Jacobian"].mean(axis=(1, 2))
    dVdx = (2.0 * np.pi) ** 2 * C_y * n_pol * avg_jaco_yz
    area_fs = ((2.0 * np.pi) ** 2 * C_y * n_pol
               * (np.sqrt(geometry["g^xx"]) * geometry["Jacobian"]).mean(axis=(1, 2)))
    sqrtgxx_fs = area_fs / dVdx
    gxx_fs = ((2.0 * np.pi) ** 2 * C_y * n_pol
              * (geometry["g^xx"] * geometry["Jacobian"]).mean(axis=(1, 2))) / dVdx
    geometry.update({"C_y": C_y, "C_xy": C_xy, "q_prof": q_prof,
                     "dVdx": dVdx, "area_fs": area_fs,
                     "sqrtgxx_fs": sqrtgxx_fs, "gxx_fs": gxx_fs,
                     "dpdx_pm_arr": dpdx_pm_arr})

    with h5py.File(folder / f"{magn_geometry}{ext}.h5", "w") as f:
        f.attrs["prec"] = "d"
        prof = f.create_group("profile")
        prof.create_dataset("q_prof", data=q_prof)
        prof.create_dataset("dpdx_pm_arr", data=dpdx_pm_arr)
        prof.create_dataset("gxx_fs", data=gxx_fs)
        prof.create_dataset("sqrtgxx_fs", data=sqrtgxx_fs)
        prof.create_dataset("dVdx", data=dVdx)
        prof.create_dataset("xval_a", data=xval_a)
        metric = f.create_group("metric")
        for name in GEOM_METRIC:
            _put3d(metric, name, geometry[name])
        metric.create_dataset("C_y", data=C_y)
        metric.create_dataset("C_xy", data=C_xy)
        bterms = f.create_group("Bfield_terms")
        for name in GEOM_BFIELD:
            _put3d(bterms, name, geometry[name])
        pars = f.create_group("parameters")
        for name, value in (("beta", 0.001), ("q0", 1.4), ("minor_r", minor_r),
                            ("major_R", major_R), ("Bref", 2.0),
                            ("Lref", major_R)):
            pars.create_dataset(name, data=np.array([value]))
        cart = f.create_group("cart_coords")
        for name in ("x", "y", "z"):
            _put3d(cart, name, rng.standard_normal(shape))

    # --- background profiles (needed by the physical flux generator) ------
    T_bg = 1.0 + 0.5 * (1.0 - xval_a)
    n_bg = 1.0 + 0.3 * (1.0 - xval_a)

    # --- field<ext>.h5 and mom_<spec><ext>.h5 -----------------------------
    times = np.arange(n_times, dtype=float) * 10.0
    field_labels = FIELD_LABELS[:n_fields]
    if physical:
        fields, moments = _physical_fields_and_moments(
            shape, ly, geometry, n_bg, T_bg, {lbl: [] for lbl in field_labels},
            species, n_times, norm_flux_projection)
    else:
        fields = {lbl: [_field_values(shape, 100 + 10 * i + it)
                        for it in range(n_times)]
                  for i, lbl in enumerate(field_labels)}
        moments = {
            spec: {lbl: [_field_values(shape, 500 + 100 * s + 10 * i + it)
                         for it in range(n_times)]
                   for i, lbl in enumerate(MOM_LABELS)}
            for s, spec in enumerate(species)}

    _write_snapshots(folder / f"field{ext}.h5", "field", field_labels,
                     times, fields)
    for spec in species:
        _write_snapshots(folder / f"mom_{spec}{ext}.h5", f"mom_{spec}",
                         MOM_LABELS, times, moments[spec])

    # --- vsp / srcmom (optional) -----------------------------------------
    if write_vsp:
        vsp_shape = (nz0, nv0, nw0, n_spec)
        vsp = {lbl: [_field_values(vsp_shape, 900 + 10 * i + it)
                     for it in range(n_times)]
               for i, lbl in enumerate(VSP_LABELS)}
        _write_snapshots(folder / f"vsp{ext}.h5", "vsp", VSP_LABELS,
                         times, vsp)
    if write_srcmom:
        for s, spec in enumerate(species):
            src = {lbl: [_field_values((nx0,), 1300 + 100 * s + 10 * i + it)
                         for it in range(n_times)]
                   for i, lbl in enumerate(SRCMOM_LABELS)}
            _write_snapshots(folder / f"srcmom_{spec}{ext}.h5",
                             f"srcmom_{spec}", SRCMOM_LABELS, times, src)

    # --- profiles_<spec><ext> (+ .h5) ------------------------------------
    profiles = {}
    for s, spec in enumerate(species):
        T, n = T_bg, n_bg
        omt = -np.gradient(np.log(T), xval_a)
        omn = -np.gradient(np.log(n), xval_a)
        profiles[spec] = {"x_o_a": xval_a, "x_o_rho_ref": xval,
                          "T": T, "n": n, "omt": omt, "omn": omn}
        header = ("#   x/a             x/rho_ref           T"
                  "                   n                   omt"
                  "                 omn\n")
        body = "".join(
            f"{xval_a[i]:20.10e}{xval[i]:20.10e}{T[i]:20.10e}"
            f"{n[i]:20.10e}{omt[i]:20.10e}{omn[i]:20.10e}\n"
            for i in range(nx0))
        (folder / f"profiles_{spec}{ext}").write_text(
            header + "#      0.000000\n" + body)
        with h5py.File(folder / f"profiles_{spec}{ext}.h5", "w") as f:
            f.attrs["prec"] = "d"
            pos = f.create_group("position")
            pos.create_dataset("x_o_a", data=xval_a)
            pos.create_dataset("x_o_rho_ref", data=xval)
            tmp = f.create_group("temp")
            tmp.create_dataset("T", data=T)
            tmp.create_dataset("omt", data=omt)
            dns = f.create_group("density")
            dns.create_dataset("n", data=n)
            dns.create_dataset("omn", data=omn)

    # --- nrg<ext> (8 columns, GENE-3D order) -----------------------------
    rng_nrg = np.random.default_rng(3)
    nrg_lines = []
    for it in range(n_times):
        nrg_lines.append(f"{times[it]:13.6f}")
        for _ in species:
            vals = rng_nrg.random(8)
            nrg_lines.append("".join(f"{v:16.8e}" for v in vals))
    (folder / f"nrg{ext}").write_text("\n".join(nrg_lines) + "\n")

    # --- profile_<spec><ext> (GENE-3D 8-column flavour) ------------------
    if write_profile_diag:
        for spec in species:
            blocks = [("#   x/a             x/rho_ref       T/Tref          "
                       "n/nref             omt             omn            "
                       "Gamma            Q         ")]
            for it in range(n_times):
                blocks.append(f"#{times[it]:14.6f}     0")
                prof = profiles[spec]
                for i in range(nx0):
                    blocks.append(
                        f"{prof['x_o_a'][i]:16.6e}{prof['x_o_rho_ref'][i]:16.6e}"
                        f"{prof['T'][i]:16.6e}{prof['n'][i]:16.6e}"
                        f"{prof['omt'][i]:16.6e}{prof['omn'][i]:16.6e}"
                        f"{0.1 * (it + 1):16.6e}{0.2 * (it + 1):16.6e}")
                blocks.append("")
                blocks.append("")
            (folder / f"profile_{spec}{ext}").write_text("\n".join(blocks) + "\n")

    return Gene3DRun(folder, ext, nx0, ny0, nz0, n_fields, species,
                     times, fields, moments, grids, geometry, profiles)
