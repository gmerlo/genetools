#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
scan_summary.py — summarise a GENE linear parameter scan.

Reads ``scan.log``, which holds one row per run: the run number, the value of
every scanned parameter, and the eigenvalue GENE found. The scan is over ky and
possibly further parameters; this groups the runs by everything *except* ky, so
each remaining parameter combination gets its own figure of growth rate and
frequency against ky.

For every run it then opens that run's ``nrg`` file and reports two flux ratios:

    Q_es(electrons) / Q_es(ions)        — how the heat flux splits by species
    Q_em(electrons) / Q_es(electrons)   — how electromagnetic the transport is

Both are read from the last time step by default (a converged linear run has
settled onto its eigenmode by then); ``--navg`` averages the ratio over the last
N steps instead, which is what you want if the mode is still competing.

Usage
-----
    python scan_summary.py scanfiles0000
    python scan_summary.py scanfiles0000 --save figs/ --no-show
    python scan_summary.py scanfiles0000/scan.log --navg 5 --csv ratios.csv

``PATH`` is the scan directory or the scan.log itself.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np

#: Column positions in a GENE ``nrg`` row (diag.F90:nrg_label1). GENE writes ten
#: columns and GENE-3D the first eight, so these indices hold for both.
NRG_Q_ES, NRG_Q_EM = 6, 7

#: scan.log column names that mean "the binormal wavenumber", lower-cased.
KY_NAMES = ("kymin", "ky", "ky0", "kymin0", "kmin")

#: Header fragments identifying the eigenvalue columns when they are named.
GAMMA_NAMES = ("growth", "gamma", "grrate", "growthrate")
OMEGA_NAMES = ("freq", "omega", "frequency")

_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")


def _floats(text: str) -> list:
    """Every number in *text*, tolerating Fortran's ``D`` exponent."""
    return [float(m.group().replace("D", "E").replace("d", "e"))
            for m in _NUMBER.finditer(text)]


def _clean_header(name: str) -> str:
    """
    Reduce one scan.log header field to the parameter's name.

    GENE writes each column as ``<name> <scan index>`` and hangs the eigenvalue
    marker off the *last* one with no separator of its own, so a two-parameter
    scan's header reads::

        #Run  | x0        1  | kymin     1  /Eigenvalue1
        0001  | 9.750000e-01 | 5.000000e-02 |  0.0670 -1.0260

    — three header fields for four data columns, and neither name usable as
    written. So: cut at the ``/`` that starts the marker, then drop the trailing
    whitespace-separated integer, which is the scan index rather than part of
    the name. A name that ends in a digit with nothing between (``omn1``) keeps
    it, which is why the separator is required.

    Returns ``''`` for a field that was only the marker; the caller falls back
    to a positional name there.
    """
    name = name.strip().lstrip("#").strip()
    name = name.split("/", 1)[0]
    return re.sub(r"\s+\d+\s*$", "", name.strip()).strip()


# ---------------------------------------------------------------------------
# scan.log
# ---------------------------------------------------------------------------

def parse_scan_log(path: str) -> tuple:
    """
    Parse ``scan.log`` into ``(entries, param_names)``.

    Each entry is ``{'run': '0001', 'params': {name: value}, 'gamma': float,
    'omega': float}``. GENE writes the file either pipe-separated (the usual
    ``#Run |kymin |/EV`` form, where the trailing field holds the eigenvalue
    pair) or as plain whitespace-separated columns; both are handled, and the
    eigenvalue is taken from named columns when they exist and from the last
    two numbers on the row when they do not.

    Raises
    ------
    ValueError
        If no data row can be interpreted — the message quotes the header, so
        an unrecognised variant can be reported rather than guessed at.
    """
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")

    header = next((ln for ln in lines if ln.lstrip().startswith("#")), None)
    body = [ln for ln in lines if not ln.lstrip().startswith("#")]
    piped = "|" in (header or "") or any("|" in ln for ln in body)

    def split(line):
        return [f.strip() for f in (line.split("|") if piped else line.split())]

    names = [_clean_header(f) for f in split(header)] if header else []

    entries = []
    for line in body:
        fields = split(line)
        if len(fields) < 2:
            continue
        run = fields[0].strip()
        if not run:
            continue
        run = run.split()[0]

        params, evs = {}, []
        for i, field in enumerate(fields[1:], start=1):
            nums = _floats(field)
            if not nums:
                continue
            # The eigenvalue block has no header field of its own, so the
            # data row runs past the header; name those columns positionally.
            name = names[i] if i < len(names) and names[i] else f"col{i}"
            if len(nums) > 1:
                # A field holding several numbers is the eigenvalue block.
                evs.extend(nums)
            else:
                params[name] = nums[0]
        entries.append({"run": run, "params": params, "_evs": evs})

    if not entries:
        raise ValueError(
            f"no data rows parsed from {path}; header was:\n  {header!r}")

    # Eigenvalues: prefer explicitly named columns, else the trailing numbers.
    def named(keys):
        for name in list(entries[0]["params"]):
            if any(k in name.lower() for k in keys):
                return name
        return None

    g_col, w_col = named(GAMMA_NAMES), named(OMEGA_NAMES)
    for e in entries:
        if g_col and w_col:
            e["gamma"] = e["params"].pop(g_col)
            e["omega"] = e["params"].pop(w_col)
        elif len(e["_evs"]) >= 2:
            e["gamma"], e["omega"] = e["_evs"][0], e["_evs"][1]
        else:
            e["gamma"] = e["omega"] = np.nan
        e.pop("_evs")

    param_names = list(entries[0]["params"])
    return entries, param_names


def identify_ky(param_names: list, override: str = None) -> str:
    """Return the name of the ky column, or raise listing what is available."""
    if override:
        if override not in param_names:
            raise ValueError(f"--ky-col {override!r} not among {param_names}")
        return override
    for name in param_names:
        if name.lower() in KY_NAMES:
            return name
    raise ValueError(
        f"no ky column found among {param_names}; pass --ky-col to name it")


# ---------------------------------------------------------------------------
# Per-run files
# ---------------------------------------------------------------------------

def find_run_file(scan_dir: str, stem: str, run: str) -> str:
    """
    Locate one run's *stem* file, trying the layouts GENE scans produce.

    A scan writes ``nrg_0001`` beside ``scan.log``; a hand-built scan often
    puts each run in its own ``0001/`` directory with the plain name instead.
    """
    candidates = [
        f"{stem}_{run}", f"{stem}{run}", f"{stem}.dat_{run}",
        os.path.join(run, f"{stem}.dat"), os.path.join(run, stem),
        f"{stem}.dat", stem,
    ]
    for rel in candidates:
        path = os.path.join(scan_dir, rel)
        if os.path.isfile(path):
            return path
    return None


def read_nrg(path: str) -> tuple:
    """
    Read a GENE ``nrg`` file.

    Returns ``(times, data)`` with *data* shaped ``(n_times, n_species,
    n_columns)``. The species count is discovered from the file: a line holding
    a single number starts a new time block, and the rows that follow it are one
    species each.
    """
    times, blocks, current = [], [], []
    with open(path) as fh:
        for line in fh:
            vals = _floats(line)
            if not vals:
                continue
            if len(vals) == 1:
                if current:
                    blocks.append(current)
                current = []
                times.append(vals[0])
            else:
                current.append(vals)
    if current:
        blocks.append(current)
    if not blocks:
        raise ValueError(f"no data blocks in {path}")

    width = min(len(row) for blk in blocks for row in blk)
    n_spec = min(len(blk) for blk in blocks)
    data = np.array([[row[:width] for row in blk[:n_spec]] for blk in blocks],
                    dtype=float)
    return np.asarray(times[:len(blocks)], dtype=float), data


def species_order(scan_dir: str, run: str) -> list:
    """
    Species names in the order ``nrg`` writes them, from the parameters file.

    Returns a list of ``(name, charge)``. Empty when no parameters file is
    found, in which case the caller falls back to positional guessing.
    """
    path = find_run_file(scan_dir, "parameters", run)
    if path is None:
        return []
    out, name, charge = [], None, None
    with open(path) as fh:
        in_block = False
        for line in fh:
            s = line.strip()
            low = s.lower()
            if low.startswith("&species"):
                in_block, name, charge = True, None, None
                continue
            if in_block and s.startswith("/"):
                out.append((name or f"spec{len(out)}", charge))
                in_block = False
                continue
            if not in_block:
                continue
            if low.startswith("name"):
                m = re.search(r"['\"]([^'\"]*)['\"]", s)
                if m:
                    name = m.group(1)
            elif low.startswith("charge"):
                nums = _floats(s.split("=", 1)[-1])
                if nums:
                    charge = nums[0]
    return out


def electron_ion_indices(species: list) -> tuple:
    """
    Return ``(i_electron, i_ion)`` positions in the nrg species ordering.

    Charge decides: electrons are the negative one. Falls back to the name when
    a parameters file gives no charge, and to (0, 1) reversed-by-name order when
    there is no parameters file at all.
    """
    if not species:
        return None, None
    i_e = next((i for i, (_, q) in enumerate(species)
                if q is not None and q < 0), None)
    if i_e is None:
        i_e = next((i for i, (n, _) in enumerate(species)
                    if n and n.lower().startswith("e")), None)
    if i_e is None:
        return None, None
    i_i = next((i for i in range(len(species)) if i != i_e), None)
    return i_e, i_i


def flux_ratios(path: str, species: list, navg: int) -> dict:
    """
    ``Q_es(e)/Q_es(i)`` and ``Q_em(e)/Q_es(e)`` from one run's nrg file.

    The ratio is formed per time step and then averaged over the last *navg*
    steps. Averaging the ratio rather than the fluxes is what makes this
    meaningful for a linear run, where both fluxes grow exponentially and a mean
    of either alone is dominated by the final step.
    """
    times, data = read_nrg(path)
    i_e, i_i = electron_ion_indices(species)
    if i_e is None or data.shape[1] < 2:
        return {"q_es_e_over_i": np.nan, "q_em_over_es_e": np.nan,
                "n_species": data.shape[1], "t_last": times[-1]}

    n = max(1, min(navg, data.shape[0]))
    qes_e = data[-n:, i_e, NRG_Q_ES]
    qes_i = data[-n:, i_i, NRG_Q_ES]
    qem_e = data[-n:, i_e, NRG_Q_EM]

    with np.errstate(divide="ignore", invalid="ignore"):
        r_spec = np.where(qes_i != 0, qes_e / qes_i, np.nan)
        r_em = np.where(qes_e != 0, qem_e / qes_e, np.nan)
    return {"q_es_e_over_i": float(np.nanmean(r_spec)),
            "q_em_over_es_e": float(np.nanmean(r_em)),
            "n_species": data.shape[1], "t_last": float(times[-1])}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def collect_rows(scan_dir, entries, param_names, ky_col, navg=1,
                 ratios=True):
    """
    One row per run: its parameters, its eigenvalue and its flux ratios.

    Returns ``(rows, missing)`` — *missing* naming the runs whose ``nrg`` file
    could not be found, whose ratios come back as NaN rather than as a guess.
    Shared by the command line and :func:`load_scan` so the two cannot report
    different numbers for the same scan.
    """
    rows, missing = [], []
    for e in entries:
        row = {"run": e["run"], "ky": e["params"].get(ky_col, np.nan),
               "gamma": e["gamma"], "omega": e["omega"]}
        row.update({p: e["params"].get(p) for p in param_names if p != ky_col})
        row.update({"q_es_e_over_i": np.nan, "q_em_over_es_e": np.nan})
        if ratios:
            nrg = find_run_file(scan_dir, "nrg", e["run"])
            if nrg is None:
                missing.append(e["run"])
            else:
                row.update(flux_ratios(nrg, species_order(scan_dir, e["run"]),
                                       navg))
        row["_key"] = group_key(e, ky_col, param_names)
        rows.append(row)
    return rows, missing


def load_scan(path, navg: int = 1, ky_col: str = None, ratios: bool = True):
    """
    Read a whole scan into one :class:`pandas.DataFrame` — the notebook entry.

    *path* is the scan directory or its ``scan.log``. Every scanned parameter
    becomes a column, alongside ``gamma``, ``omega`` and the two flux ratios.
    Sorted by the non-ky parameters and then by ky, so plotting a group needs no
    further sorting.

    ``df.attrs`` carries what a plot needs to label itself: ``ky`` (the name of
    the ky column, always stored as ``'ky'``), ``group_by`` (the other scanned
    parameters) and ``missing`` (runs with no ``nrg``).

    pandas is imported here rather than at module scope: the command line does
    not need it.
    """
    import pandas as pd

    path = os.path.abspath(path)
    log_path = os.path.join(path, "scan.log") if os.path.isdir(path) else path
    scan_dir = os.path.dirname(log_path) or "."
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"no scan.log at {log_path}")

    entries, param_names = parse_scan_log(log_path)
    ky_col = identify_ky(param_names, ky_col)
    others = [p for p in param_names if p != ky_col]

    rows, missing = collect_rows(scan_dir, entries, param_names, ky_col,
                                 navg=navg, ratios=ratios)
    for row in rows:
        row.pop("_key", None)

    # The column keeps the scan's own name for ky, so the table reads the way
    # scan.log does; df.attrs["ky"] says which column that is.
    df = pd.DataFrame(rows).rename(columns={"ky": ky_col,
                                            "q_es_e_over_i": "Qes_e/Qes_i",
                                            "q_em_over_es_e": "Qem/Qes_e"})
    df = df.sort_values(others + [ky_col]).reset_index(drop=True)
    df.attrs.update(ky=ky_col, group_by=others, missing=missing,
                    scan_dir=scan_dir)
    return df


def group_key(entry: dict, ky_col: str, param_names: list) -> tuple:
    """The scanned parameters other than ky, as a hashable key."""
    return tuple((p, entry["params"].get(p))
                 for p in param_names if p != ky_col)


def label_of(key: tuple) -> str:
    """Readable label for a group key; empty when ky is the only scan axis."""
    return ", ".join(f"{p} = {v:g}" for p, v in key if v is not None)


def plot_group(key, rows, ky_col, plt, ratios=True):
    """
    Two subplots — growth rate and frequency against ky — plus the flux ratios.

    The ratios share the ky axis, so they are drawn as a second row rather than
    a separate figure: reading gamma(ky) against Q_em/Q_es(ky) is the point of
    computing them.
    """
    rows = sorted(rows, key=lambda r: r["ky"])
    ky = np.array([r["ky"] for r in rows])
    nrow = 2 if ratios else 1
    fig, axes = plt.subplots(nrow, 2, figsize=(11, 4.0 * nrow), squeeze=False)

    axes[0][0].plot(ky, [r["gamma"] for r in rows], "o-")
    axes[0][0].set_ylabel(r"$\gamma\;[c_s/L_{ref}]$")
    axes[0][1].plot(ky, [r["omega"] for r in rows], "s-", color="C1")
    axes[0][1].set_ylabel(r"$\omega\;[c_s/L_{ref}]$")
    axes[0][1].axhline(0.0, color="0.6", lw=0.8)

    if ratios:
        axes[1][0].plot(ky, [r["q_es_e_over_i"] for r in rows], "^-",
                        color="C2")
        axes[1][0].set_ylabel(r"$Q_{es}^{e}/Q_{es}^{i}$")
        axes[1][1].plot(ky, [r["q_em_over_es_e"] for r in rows], "v-",
                        color="C3")
        axes[1][1].set_ylabel(r"$Q_{em}^{e}/Q_{es}^{e}$")
        axes[1][1].axhline(0.0, color="0.6", lw=0.8)

    for row in axes:
        for ax in row:
            ax.set_xlabel(rf"${ky_col}$" if ky_col.startswith("k")
                          else ky_col)
            ax.grid(True, alpha=0.3)
    title = label_of(key)
    fig.suptitle(title if title else "linear scan")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Summarise a GENE linear scan: gamma and omega against ky, "
                    "plus the electron/ion and EM/ES heat-flux ratios.")
    p.add_argument("path", help="Scan directory, or the scan.log file itself.")
    p.add_argument("--ky-col", default=None,
                   help="Name of the ky column in scan.log (auto-detected).")
    p.add_argument("--navg", type=int, default=1,
                   help="Average the flux ratios over the last N nrg steps "
                        "(default 1: the final step).")
    p.add_argument("--no-ratios", action="store_true",
                   help="Skip the nrg files; plot gamma and omega only.")
    p.add_argument("--csv", default=None, metavar="FILE",
                   help="Write the per-run table to this CSV.")
    p.add_argument("--save", default=None, metavar="DIR",
                   help="Save figures as PNG into this directory.")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open plot windows.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if os.path.isdir(args.path):
        scan_dir, log_path = args.path, os.path.join(args.path, "scan.log")
    else:
        log_path = args.path
        scan_dir = os.path.dirname(os.path.abspath(log_path)) or "."
    if not os.path.isfile(log_path):
        print(f"no scan.log at {log_path}", file=sys.stderr)
        return 1

    try:
        entries, param_names = parse_scan_log(log_path)
        ky_col = identify_ky(param_names, args.ky_col)
    except ValueError as exc:
        print(f"could not read {log_path}: {exc}", file=sys.stderr)
        return 1
    print(f"{len(entries)} runs, scanned parameters: {param_names} "
          f"(ky column: {ky_col})")

    rows, missing = collect_rows(scan_dir, entries, param_names, ky_col,
                                 navg=args.navg, ratios=not args.no_ratios)
    if missing:
        print(f"  no nrg file for runs: {', '.join(missing)}", file=sys.stderr)

    # --- table ---------------------------------------------------------
    others = [p for p in param_names if p != ky_col]
    head = (["run", ky_col] + others
            + ["gamma", "omega", "Qes_e/Qes_i", "Qem/Qes_e"])
    widths = [max(len(h), 11) for h in head]
    print("  ".join(h.rjust(w) for h, w in zip(head, widths)))
    for r in sorted(rows, key=lambda r: (r["_key"], r["ky"])):
        cells = ([r["run"], f"{r['ky']:.4g}"]
                 + [f"{r.get(p):.4g}" if r.get(p) is not None else "-"
                    for p in others]
                 + [f"{r['gamma']:.4g}", f"{r['omega']:.4g}",
                    f"{r['q_es_e_over_i']:.4g}", f"{r['q_em_over_es_e']:.4g}"])
        print("  ".join(c.rjust(w) for c, w in zip(cells, widths)))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[h for h in head])
            w.writeheader()
            for r in sorted(rows, key=lambda r: (r["_key"], r["ky"])):
                w.writerow({"run": r["run"], ky_col: r["ky"],
                            **{p: r.get(p) for p in others},
                            "gamma": r["gamma"], "omega": r["omega"],
                            "Qes_e/Qes_i": r["q_es_e_over_i"],
                            "Qem/Qes_e": r["q_em_over_es_e"]})
        print(f"wrote {args.csv}")

    # --- figures -------------------------------------------------------
    import matplotlib
    if args.save or args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for r in rows:
        groups[r["_key"]].append(r)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
    for key, grp in groups.items():
        fig = plot_group(key, grp, ky_col, plt, ratios=not args.no_ratios)
        if args.save:
            tag = (re.sub(r"[^\w.=-]+", "_", label_of(key)) or "scan")
            out = os.path.join(args.save, f"scan_{tag}.png")
            fig.savefig(out, dpi=140)
            print(f"wrote {out}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
