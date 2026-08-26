# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
xglobal_fixture.py — synthetic x-global (``x_local=F``, ``y_local=T``) runs.

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
