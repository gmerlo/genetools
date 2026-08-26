# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
gene_fixture.py — synthetic regular-GENE runs (flux tube and x-global).

No x-global run with output exists in the repo, so every x-global code path was
verified by reading it rather than running it. Two of the bugs fixed in
`shearingrate` — the ExB sign and the whole-array Jacobian normalisation — were
hidden precisely because the x-global unit fixtures used a uniform Jacobian and
``C_xy = 1``. This builds real run directories instead.

Only what a given diagnostic needs is written, so the builders are additive:
:func:`make_xglobal_run` lays down the parameter file, and helpers add the
output files on top.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

#: GENE's own source-moment datasets: `diag_df.F90` src_label x mom_label.
#: Nine, where GENE-3D writes six — regular GENE adds `f0_term`.
SRCMOM_LABELS = [f"{a}_{b}" for a in ("ck_heat", "ck_part", "f0_term")
                 for b in ("M00", "M10", "M22")]

#: GENE's velocity-space datasets (`diag_vsp.F90` vsp_label). GENE-3D writes the
#: same five with `Q_ese`/`Q_eme` in place of `Q_es`/`Q_em`.
VSP_LABELS = ["G_es", "G_em", "Q_es", "Q_em", "<f_>"]


def make_xglobal_run(tmp_path, ext=".dat", nx0=12, nky0=4, nz0=8,
                     species=("ions", "electrons"), x0=0.5, lx=40.0):
    """
    Write an x-global run directory and return its folder.

    ``x_local=F`` with ``y_local=T`` is what makes `Params.geometry_kind()`
    report ``x_global``; the velocity-grid keys are needed because
    `Coordinates` builds the ``v_par``/``mu`` grids unconditionally.
    """
    folder = Path(tmp_path)
    folder.mkdir(parents=True, exist_ok=True)
    spec_blocks = "\n".join(textwrap.dedent(f"""\
        &species
         name = '{name}'
         omt = 6.0
         omn = 2.0
         mass = {1.0 if i == 0 else 0.00027}
         charge = {1 if i == 0 else -1}
         temp = 1.0
         dens = 1.0
        /
        """) for i, name in enumerate(species))

    (folder / f"parameters{ext}").write_text(textwrap.dedent(f"""\
        &parallelization
         n_procs_s = 1
        /
        &box
         nx0 = {nx0}
         nky0 = {nky0}
         nz0 = {nz0}
         nv0 = 8
         nw0 = 4
         lx = {lx}
         lv = 3.0
         lw = 9.0
         kymin = 0.05
         x0 = {x0}
        /
        &in_out
         diagdir = './'
         write_h5 = T
        /
        &general
         x_local = F
         y_local = T
         nonlinear = T
        /
        &geometry
         magn_geometry = 'circular'
         trpeps = 0.18
         major_R = 3.0
         minor_r = 1.0
         q0 = 1.4
         shat = 0.8
         rhostar = 0.002
         n_pol = 1
         edge_opt = 0
        /
        &units
         Tref = 1.0
         nref = 1.0
         Bref = 2.0
         mref = 2.0
         Lref = 3.0
        /
        """) + spec_blocks)
    (folder / f"nrg{ext}").touch()
    return folder


def write_srcmom(folder, ext=".dat", nx0=12, n_times=4,
                 species=("ions", "electrons"), labels=None, seed=0):
    """
    Write GENE-style ``srcmom_<species><ext>.h5`` files.

    Layout from `diag_df.F90`: one group per ``<src>_<mom>`` pair holding
    ``%010d`` snapshot datasets of shape ``(nx0,)``, beside an extendible
    ``time`` dataset — the same futils shape `H5Reader` already reads for
    GENE-3D, so the reader needs no new branch.
    """
    import h5py
    labels = list(labels or SRCMOM_LABELS)
    rng = np.random.default_rng(seed)
    times = np.arange(n_times, dtype=float) * 5.0
    truth = {}
    for name in species:
        path = Path(folder) / f"srcmom_{name}{ext}.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group(f"srcmom_{name}")
            grp.create_dataset("time", data=times, maxshape=(None,))
            per = {}
            for label in labels:
                sub = grp.create_group(label)
                arrs = rng.normal(size=(n_times, nx0))
                for it in range(n_times):
                    sub.create_dataset(f"{it:010d}", data=arrs[it])
                per[label] = arrs
            truth[name] = per
    return times, truth


def make_fluxtube_run(tmp_path, ext=".dat", nz0=16, nv0=8, nw0=4,
                      species=("ions", "electrons")):
    """
    Write a flux-tube run with a real ``tracer_efit`` geometry file.

    `MINIMAL_PARAMS` carries no ``&geometry`` namelist, so it can only be used
    to check that something refuses — loading a geometry from it raises. This
    writes the synthetic geometry the io tests use, so 1-D coefficient paths are
    genuinely exercised.
    """
    folder = Path(tmp_path)
    folder.mkdir(parents=True, exist_ok=True)
    spec_blocks = "\n".join(textwrap.dedent(f"""\
        &species
         name = '{name}'
         omt = 6.0
         omn = 2.0
         mass = {1.0 if i == 0 else 0.00027}
         charge = {1 if i == 0 else -1}
         temp = 1.0
         dens = 1.0
        /
        """) for i, name in enumerate(species))

    (folder / f"parameters{ext}").write_text(textwrap.dedent(f"""\
        &box
         nx0 = 4
         nky0 = 2
         nz0 = {nz0}
         nv0 = {nv0}
         nw0 = {nw0}
         lx = 1.0
         lv = 3.0
         lw = 9.0
         kymin = 0.1
        /
        &in_out
         diagdir = './'
         write_h5 = T
        /
        &general
         x_local = T
         y_local = T
        /
        &geometry
         magn_geometry = 'tracer_efit'
         n_pol = 1
         edge_opt = 0
        /
        &units
         Lref = 1.0
         Tref = 1.0
         nref = 1.0
         mref = 2.0
         Bref = 2.0
        /
        """) + spec_blocks)
    (folder / f"nrg{ext}").touch()

    rng = np.random.default_rng(0)
    # Columns: gxx gxy gxz gyy gyz gzz B dBdx dBdy dBdz J R Phi Z dxdR dxdZ.
    # All positive so the metric operations stay well-defined.
    data = rng.uniform(0.5, 1.5, size=(nz0, 16))
    rows = "\n".join("  ".join(f"{v:.8e}" for v in row) for row in data)
    (folder / f"tracer_efit{ext}").write_text(
        "&tracer_efit\n q0=1.4\n shat=0.8\n trpeps=0.18\n"
        " cxy=0.5\n cy=1.0\n/\n" + rows + "\n")
    return folder


def write_vsp(folder, ext="", nz0=16, nv0=8, nw0=4, n_spec=2, n_times=3,
              labels=None, seed=1):
    """
    Write a GENE-style ``vsp<ext>.h5``.

    From `diag_vsp.F90`: one group per label under ``/vsp``, holding ``%010d``
    snapshots of the Fortran array ``(nz0, nv0, nw0, n_spec)``. futils stores it
    reversed, which is what `H5Reader` undoes — so the fixture writes the
    reversed shape, exactly as the code would.
    """
    import h5py
    labels = list(labels or VSP_LABELS)
    rng = np.random.default_rng(seed)
    times = np.arange(n_times, dtype=float) * 2.5
    truth = {}
    with h5py.File(Path(folder) / f"vsp{ext}.h5", "w") as f:
        grp = f.create_group("vsp")
        grp.create_dataset("time", data=times, maxshape=(None,))
        for label in labels:
            sub = grp.create_group(label)
            arrs = rng.normal(size=(n_times, nz0, nv0, nw0, n_spec))
            for it in range(n_times):
                # Reverse every axis: putarr hands HDF5 a Fortran-ordered array.
                sub.create_dataset(f"{it:010d}", data=arrs[it].T)
            truth[label] = arrs
    return times, truth
