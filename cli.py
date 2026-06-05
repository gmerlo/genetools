# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
cli.py — flag-style command-line interface for genetools.

Usage
-----
    genetools [PATH] --DIAG [options]

Examples
--------
    genetools /run --nrg
    genetools /run --spectra --t 500 2000 --save spectra.png
    genetools /run --contours --field 0 --ifft xy
    genetools . --profiles --no-show

One diagnostic flag is required. ``PATH`` defaults to the current directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Diagnostic flag -> Run accessor attribute.
DIAGNOSTICS = {
    "nrg": "nrg",
    "spectra": "spectra",
    "profiles": "profiles",
    "fluxes2d": "fluxes2d",
    "shearing": "shearing",
    "contours": "contours",
    "ballooning": "ballooning",
    "growthrate": "growthrate",
    "amplitude": "amplitude",
    "zonal": "zonal",
    "profile_diag": "profile_diag",
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the genetools argument parser."""
    p = argparse.ArgumentParser(
        prog="genetools",
        description="Post-process a GENE run from the command line.",
    )
    p.add_argument("path", nargs="?", default=".",
                   help="GENE run directory (default: current directory).")
    p.add_argument("--runpath", default=None,
                   help="GENE run directory (overrides the positional PATH).")

    group = p.add_argument_group("diagnostics (choose one)")
    diag = group.add_mutually_exclusive_group(required=True)
    for flag in DIAGNOSTICS:
        diag.add_argument(f"--{flag.replace('_', '-')}", dest=flag,
                          action="store_true",
                          help=f"Run the {flag} diagnostic.")

    p.add_argument("--t", nargs=2, type=float, metavar=("START", "STOP"),
                   default=None, help="Time window (two floats).")
    p.add_argument("--species", nargs="+", default=None,
                   help="Restrict to these species (default: all).")
    p.add_argument("--ext", nargs="+", default=None,
                   help="Segment suffix(es), e.g. _0002 .dat (default: all).")
    p.add_argument("--save", default=None, metavar="FILE",
                   help="Save figure(s); a directory auto-names PNG files.")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open a plot window (headless).")

    # Diagnostic-specific options
    p.add_argument("--ky", type=float, default=None,
                   help="ky mode (ballooning).")
    p.add_argument("--field", type=int, default=0,
                   help="Field/moment index (contours).")
    p.add_argument("--ifft", default=None, choices=[None, "x", "y", "xy"],
                   help="Inverse FFT axes (contours).")
    return p


def _selected_diagnostic(args) -> str:
    for flag in DIAGNOSTICS:
        if getattr(args, flag):
            return flag
    raise SystemExit("No diagnostic selected.")  # pragma: no cover (argparse guards)


def _plot_kwargs(name: str, args) -> dict:
    """Build the per-diagnostic plot kwargs from parsed args."""
    t = tuple(args.t) if args.t is not None else None
    kw = {"t": t}
    if name == "contours":
        kw["field"] = args.field
        kw["ifft"] = args.ifft
        if args.species:
            kw["species"] = args.species[0]
    elif name == "ballooning":
        kw["ky"] = args.ky
    return kw


def main(argv=None) -> int:
    """Entry point for the ``genetools`` console script."""
    args = build_parser().parse_args(argv)
    name = _selected_diagnostic(args)

    headless = args.no_show or args.save is not None
    import matplotlib
    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The existing diagnostics call plt.show() internally; intercept it so the
    # CLI controls whether windows open or figures are saved.
    orig_show = plt.show
    plt.show = lambda *a, **k: None
    plt.close("all")  # start clean so --save captures only this diagnostic

    from .run import Run

    try:
        path = args.runpath or args.path
        run = Run(path, ext=args.ext)

        accessor = DIAGNOSTICS[name]
        diag = getattr(run, accessor)
        kwargs = _plot_kwargs(name, args)
        if name == "ballooning":  # parametrized accessor
            ky = kwargs.pop("ky")
            if ky is None:
                print("genetools: --ballooning requires --ky VALUE",
                      file=sys.stderr)
                return 2
            diag = diag(ky=ky)
        diag.plot(**kwargs)
    except Exception as exc:  # surface a clean message, not a traceback
        print(f"genetools: error: {exc}", file=sys.stderr)
        return 1

    if args.save is not None:
        _save_figures(plt, args.save, name)
    elif not args.no_show:
        plt.show = orig_show
        orig_show()
    plt.close("all")
    return 0


def _save_figures(plt, save, name: str) -> None:
    """Save every open matplotlib figure to *save* (file or directory)."""
    nums = plt.get_fignums()
    if not nums:
        print(f"{name}: no figures were produced.", file=sys.stderr)
        return
    out = Path(save)
    # Treat as a directory if it exists as one, ends with '/', or has no file
    # extension (so `--save outdir` works even when outdir doesn't exist yet).
    is_dir = out.is_dir() or save.endswith("/") or out.suffix == ""
    if is_dir:
        out.mkdir(parents=True, exist_ok=True)
    for i, num in enumerate(nums):
        fig = plt.figure(num)
        if is_dir:
            fpath = out / f"{name}_{i}.png"
        elif len(nums) > 1:
            suffix = out.suffix or ".png"
            fpath = out.with_name(f"{out.stem}_{i}{suffix}")
        else:
            fpath = out if out.suffix else out.with_suffix(".png")
        fig.savefig(fpath, dpi=120, bbox_inches="tight")
        print(f"Saved {fpath}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
