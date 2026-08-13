#!/usr/bin/env python3
"""
Plot a 2-D slice of a GENE checkpoint (the distribution function g).

Standalone: numpy + matplotlib (adios2 only for .bp checkpoints).

Two on-disk layouts are supported, both per src/checkpoint.F90:

MPI-IO (``chpt_fmt = 1``) — a flat binary file:
    6 chars   precision, 'DOUBLE' or 'SINGLE'
    2 reals   time, dt
    6 int32   nx0, nky0, nz0, nv0, nw0, n_spec
    complex   g(nx0, nky0, nz0, nv0, nw0, n_spec), Fortran order

ADIOS2 (``chpt_fmt = 3``) — a ``.bp`` directory holding variables ``g_``,
``time``, ``dt``, ``nx0``…``n_spec`` and ``precision``; the last step is used.

Only the requested slice is read in either case — the flat file is
memory-mapped and the BP variable is read as a hyperslab — because a
checkpoint is far too large to load whole.

Usage:
    ckpt_slice.py checkpoint                 # |g|(v, mu), other dims at mid-grid
    ckpt_slice.py checkpoint.bp z,v          # any two of: x,ky,z,v,w,s
    ckpt_slice.py checkpoint v,w fig.png
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

DIMS = ("x", "ky", "z", "v", "w", "s")
SIZE_VARS = ("nx0", "nky0", "nz0", "nv0", "nw0", "n_spec")


def _fixed_index(shape, axes):
    """Full slices on *axes*, mid-grid on every other dimension."""
    idx = [n // 2 for n in shape]
    for a in axes:
        idx[a] = slice(None)
    return idx


def _slice_raw(path, axes):
    """Slice a flat MPI-IO checkpoint via memmap."""
    with open(path, "rb") as f:
        prec = f.read(6).decode("ascii", "replace").strip().upper()
        single = prec.startswith("SINGLE")
        real = np.dtype("<f4" if single else "<f8")
        cpx = np.dtype("<c8" if single else "<c16")
        time, dt = np.fromfile(f, real, 2)
        shape = tuple(int(n) for n in np.fromfile(f, np.int32, 6))
    offset = 6 + 2 * real.itemsize + 6 * 4

    need = offset + int(np.prod(shape, dtype=np.int64)) * cpx.itemsize
    if not prec.startswith(("DOUBLE", "SINGLE")) or os.path.getsize(path) < need:
        sys.exit(f"{path}: not an MPI-IO checkpoint (prec={prec!r}, dims={shape}, "
                 f"needs {need} bytes, file has {os.path.getsize(path)}).")

    idx = _fixed_index(shape, axes)
    g = np.memmap(path, dtype=cpx, mode="r", offset=offset, shape=shape, order="F")
    sl = np.abs(np.asarray(g[tuple(idx)]))
    del g
    return prec, float(time), float(dt), shape, idx, sl


def _patch_adios2_complex():
    """
    Teach the adios2 Python bindings about single-precision complex.

    ADIOS2 names the type of a ``complex(4)`` array ``'float complex'``, but
    the translation table in ``adios2/stream.py`` (still true in 2.12) lists
    only the legacy spelling ``'complex'`` next to ``'double complex'``, so
    reading a single-precision checkpoint dies with ``KeyError: 'float
    complex'``. Fill in the missing entries, leaving everything else alone.
    """
    try:
        from adios2 import stream as _stream
    except ImportError:
        return                                  # older bindings: no such module
    original = getattr(_stream, "type_adios_to_numpy", None)
    if original is None:
        return
    try:
        original("float complex")
        return                                  # already handled upstream
    except KeyError:
        pass

    extra = {"float complex": np.complex64, "double complex": np.complex128}

    def translate(name):
        return extra[name] if name in extra else original(name)

    _stream.type_adios_to_numpy = translate


def _bp_last_step(path):
    """
    Yield a ``read`` callable positioned on the checkpoint's last step.

    adios2 2.10 rewrote the Python bindings: the module-level ``open()`` was
    replaced by ``Stream``. Both spellings are handled so the script works
    whichever version is installed.
    """
    import adios2
    _patch_adios2_complex()

    if hasattr(adios2, "Stream"):                       # adios2 >= 2.10
        with adios2.Stream(path, "r") as s:
            last = s.num_steps() - 1
            for _ in s.steps():
                if s.current_step() == last:
                    yield s.read, s.available_variables()
                    return
    elif hasattr(adios2, "open"):                       # adios2 < 2.10
        with adios2.open(path, "r") as fh:
            last = sum(1 for _ in fh) - 1
        with adios2.open(path, "r") as fh:
            for i, step in enumerate(fh):
                if i == last:
                    yield step.read, step.available_variables()
                    return
    else:
        sys.exit("installed adios2 exposes neither Stream nor open()")


def _slice_bp(path, axes):
    """Slice an ADIOS2 checkpoint, reading only the needed hyperslab."""
    for read, variables in _bp_last_step(path):
        scalar = lambda name: np.asarray(read(name)).ravel()[0]
        shape = tuple(int(scalar(v)) for v in SIZE_VARS)
        time, dt = float(scalar("time")), float(scalar("dt"))
        prec = str(scalar("precision")).strip().upper()

        idx = _fixed_index(shape, axes)
        start = [0 if isinstance(i_, slice) else int(i_) for i_ in idx]
        count = [n if isinstance(i_, slice) else 1 for i_, n in zip(idx, shape)]

        # ADIOS2 may report a Fortran-written array with reversed dims;
        # match the selection to whatever the file actually declares.
        declared = tuple(int(n) for n in variables["g_"]["Shape"].split(","))
        flipped = declared == tuple(reversed(shape))
        if flipped:
            start, count = start[::-1], count[::-1]
        elif declared != shape:
            sys.exit(f"{path}: 'g_' has shape {declared}, expected {shape} "
                     "— unrecognised checkpoint layout.")

        raw = np.asarray(read("g_", start, count))
        sl = np.abs(raw.reshape(count, order="C").squeeze())
        return prec, time, dt, shape, idx, sl.T if flipped else sl

    sys.exit(f"{path}: BP checkpoint holds no steps.")


def main(path, plane="v,w", save=None):
    axes = [DIMS.index(d.strip()) for d in plane.split(",")]
    if len(axes) != 2 or len(set(axes)) != 2:
        sys.exit(f"give two distinct dimensions from {','.join(DIMS)}")

    is_bp = path.rstrip("/").endswith(".bp") or os.path.isdir(path)
    prec, time, dt, shape, idx, sl = (
        _slice_bp(path, axes) if is_bp else _slice_raw(path, axes))

    if axes[0] > axes[1]:                 # keep the requested order on the axes
        sl = sl.T

    fixed = ", ".join(f"{DIMS[i]}={idx[i]}" for i in range(6)
                      if not isinstance(idx[i], slice))
    print(f"{prec} checkpoint  t={time:g}  dt={dt:g}  dims={dict(zip(DIMS, shape))}")
    print(f"slice |g|({DIMS[axes[0]]}, {DIMS[axes[1]]}) at {fixed}: "
          f"shape={sl.shape} max={sl.max():.6g}")

    positive = sl[sl > 0]
    norm = LogNorm(positive.min(), positive.max()) if positive.size else None
    fig, ax = plt.subplots(figsize=(6, 4.6), constrained_layout=True)
    im = ax.imshow(sl.T, origin="lower", aspect="auto", norm=norm, cmap="viridis")
    fig.colorbar(im, ax=ax, label="$|g|$")
    ax.set_xlabel(DIMS[axes[0]])
    ax.set_ylabel(DIMS[axes[1]])
    ax.set_title(f"$|g|$ at t = {time:g}   ({fixed})")
    fig.savefig(save, dpi=150) if save else plt.show()
    return sl


if __name__ == "__main__":
    main(*sys.argv[1:4])
