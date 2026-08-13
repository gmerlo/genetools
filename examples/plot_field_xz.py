#!/usr/bin/env python3
"""
Plot the last stored field snapshot f(x, z) on the y = 0 plane of a GENE run.

Standalone (numpy + matplotlib; adios2 needed for .bp files only). Reads the
``field<ext>`` file — Fortran unformatted binary (little-endian; records
larger than 2 GiB, whose sign-flagged markers split them into subrecords, are
stitched back together) or ADIOS2 BP — takes the last complete time step,
inverse-FFTs ky -> y only and slices the y = 0 plane.

Usage:  plot_field_xz.py /path/to/run[/field_0002[.bp]] [fig.png]
"""
import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt


def _bp_last_step(path):
    """Yield a ``read`` callable positioned on the file's last step.

    adios2 2.10 replaced the module-level ``open()`` with ``Stream``, and its
    type table still omits ``'float complex'``, so single-precision output
    fails with a KeyError until the missing entries are supplied.
    """
    import adios2
    try:
        from adios2 import stream as _stream
        _orig = _stream.type_adios_to_numpy
        try:
            _orig("float complex")
        except KeyError:
            _extra = {"float complex": np.complex64,
                      "double complex": np.complex128}
            _stream.type_adios_to_numpy = (
                lambda n: _extra[n] if n in _extra else _orig(n))
    except ImportError:
        pass

    if hasattr(adios2, "Stream"):                       # adios2 >= 2.10
        with adios2.Stream(path, "r") as s:
            last = s.num_steps() - 1
            for _ in s.steps():
                if s.current_step() == last:
                    yield s.read
                    return
    else:                                               # adios2 < 2.10
        with adios2.open(path, "r") as fh:
            last = sum(1 for _ in fh) - 1
        with adios2.open(path, "r") as fh:
            for i, step in enumerate(fh):
                if i == last:
                    yield step.read
                    return


def read_last(field, p):
    """Return (time, [complex array (nx, nky, nz) per field]) of the last step."""
    shape = (p["nx"], p["nky"], p["nz"])
    names = ("phi", "A_par", "B_par")[:p["n_fields"]]

    if field.endswith(".bp"):
        for read in _bp_last_step(field):
            time = float(np.asarray(read("time")).ravel()[0])
            return time, [np.asarray(read(v)).reshape(shape, order="F")
                          for v in names]
        sys.exit(f"{field}: empty BP file")

    size = os.path.getsize(field)
    with open(field, "rb") as f:

        def record():
            """Payload byte ranges [(offset, nbytes), ...] of the next record.

            A negative leading marker means the record continues in another
            subrecord (Fortran convention for records > 2**31-1 bytes).
            Returns None on EOF or truncation.
            """
            parts, more = [], True
            while more:
                hdr = f.read(4)
                if len(hdr) < 4:
                    return None
                n = int(np.frombuffer(hdr, "<i4")[0])
                more, n = n < 0, abs(n)
                if f.tell() + n + 4 > size:
                    return None
                parts.append((f.tell(), n))
                f.seek(n + 4, 1)  # payload + trailing marker
            return parts

        def payload(parts, dtype):
            chunks = []
            for off, n in parts:
                f.seek(off)
                chunks.append(np.fromfile(f, np.uint8, n))
            raw = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            return raw.view(dtype)

        # Skim markers only (no data) to locate the last complete step:
        # each step is 1 time record + n_fields array records.
        last = None
        while True:
            recs = [record() for _ in range(1 + p["n_fields"])]
            if None in recs:
                break
            last = recs
        if last is None:
            sys.exit(f"{field}: no complete time step found")

        time = float(payload(last[0], p["real"])[0])
        return time, [payload(r, p["cpx"]).reshape(shape, order="F") for r in last[1:]]


def main(path, save=None):
    field = sorted(glob.glob(os.path.join(path, "field*")))[-1] if os.path.isdir(path) else path
    folder, name = os.path.split(field)
    ext = name[len("field"):-3] if name.endswith(".bp") else name[len("field"):]
    text = open(os.path.join(folder, "parameters" + ext)).read()
    get = lambda key, cast=float: cast(
        re.search(rf"^\s*{key}\s*=\s*([^\s!/]+)", text, re.M | re.I).group(1))
    single = re.search(r"PRECISION\s*=\s*SINGLE", text, re.I)
    p = {"nx": get("nx0", int), "nky": get("nky0", int), "nz": get("nz0", int),
         "n_fields": get("n_fields", int), "lx": get("lx"),
         "real": np.dtype("<f4" if single else "<f8"),
         "cpx": np.dtype("<c8" if single else "<c16")}
    time, arrays = read_last(field, p)

    x = np.linspace(-p["lx"] / 2, p["lx"] / 2, p["nx"], endpoint=False)
    z = np.linspace(-np.pi, np.pi, p["nz"], endpoint=False)
    nf = p["n_fields"]
    fig, axes = plt.subplots(1, nf, figsize=(5.2 * nf, 4.2), sharey=True,
                             constrained_layout=True, squeeze=False)
    for ax, arr, label in zip(axes[0], arrays, (r"$\phi$", r"$A_\parallel$", r"$B_\parallel$")):
        nky = p["nky"]
        fxz = (np.fft.irfft(arr, n=2 * nky, axis=1) * 2 * nky)[:, 0, :]  # ky -> y, slice y = 0
        vmax = np.abs(fxz).max() or 1.0
        pc = ax.pcolormesh(x, z, fxz.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
        fig.colorbar(pc, ax=ax, pad=0.02)
        ax.set_title(f"{label}(x, z, y=0)")
        ax.set_xlabel("$x$")
    axes[0, 0].set_ylabel("$z$")
    fig.suptitle(f"{name}   —   t = {time:g}")
    fig.savefig(save, dpi=150) if save else plt.show()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
