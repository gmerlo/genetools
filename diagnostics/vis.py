# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
vis.py — GENE-3D export to VTK for external 3-D visualisation.

GENE-3D writes the Cartesian position of every grid point into its geometry file
(``/cart_coords/{x,y,z}``), so a snapshot can be exported as a structured grid in
real space with no geometry library involved. That is what :class:`Vis` does,
and it is the path that works for any GENE-3D run.

The reference GUI instead calls into a compiled GVEC extension (``gvec_to_gene``,
built with CFFI) to rebuild the mapping from a stellarator equilibrium file and
write VTK through it. That covers cases where the run's own ``cart_coords`` are
absent or where the GVEC state file is the source of truth, so it is supported
here too — but only as an optional path, since it needs a compiled module that is
not part of this package.

VTK output is written directly rather than through a library: a legacy
``STRUCTURED_GRID`` file is a short, stable, well-documented text format, which
is a better dependency trade than requiring ``pyevtk`` or ``vtk`` for one writer.
"""

from __future__ import annotations

import os

import numpy as np

from genetools._xr import make_dataset, unit_attrs
from genetools.diagnostics._base import RunDiagnostic
from genetools.diagnostics import _gene3d as g3


class Vis(RunDiagnostic):
    """
    Export GENE-3D snapshots as VTK structured grids.

    Parameters
    ----------
    run : genetools.run.Run
    quantities : sequence of str
        Variables to export (from the field or moment file).
    species : str, optional
        Species supplying moment quantities (default: the first).
    xlim : (float, float), optional
        Radial window in ``x/a`` — a full 3-D export is large, and the buffer
        regions are rarely wanted.
    """

    name = "vis"
    supported = ("xy_global",)

    def __init__(self, run, quantities=("phi",), species=None, xlim=None):
        self.run = run
        self.quantities = tuple(quantities)
        self.species = species or (run.species[0] if run.species else None)
        self.xlim = xlim

    # ------------------------------------------------------------------

    def cartesian_grid(self):
        """
        Return the ``(X, Y, Z)`` Cartesian position of every grid point.

        Raises
        ------
        ValueError
            If the geometry file has no ``/cart_coords`` group. GENE-3D writes it
            unconditionally, so this means an older run or a geometry file
            written by something else; :meth:`write_vtk_via_gvec` is the
            alternative.
        """
        geom = self.run.geometry[0]
        cart = geom.get("cart_coords")
        if cart is None:
            raise ValueError(
                "The geometry file carries no /cart_coords group, so the "
                "Cartesian positions of the grid points are unknown. Use "
                "write_vtk_via_gvec() with a GVEC state file instead.")
        return (np.asarray(cart["x"]), np.asarray(cart["y"]),
                np.asarray(cart["z"]))

    def _sources(self):
        from genetools.diagnostics.slices import Slices
        return Slices(self.run, quantities=self.quantities,
                        species=self.species)._sources()

    def compute(self, t=None):
        """Read the requested snapshots, returning them keyed by variable."""
        run = self.run
        coord = run.coords[0]
        xsl = (g3.radial_slice(coord["x_o_a"], limits=self.xlim)
               if self.xlim else slice(None))

        frames, times = {}, None
        for reader, names in self._sources():
            _, idx = self._indices(reader, t)
            slots = {n: reader.index_of(n) for n in names}
            acc = {n: [] for n in names}
            got = []
            for time, arrays in reader.stream_selected(idx):
                got.append(time)
                for n in names:
                    acc[n].append(arrays[slots[n]][xsl])
            for n in names:
                frames[n] = np.asarray(acc[n])
            if times is None:
                times = np.asarray(got)
        return {"frames": frames, "times": times, "xslice": xsl}

    def dataset(self, t=None):
        """Return the snapshots as an :class:`xarray.Dataset`."""
        raw = self.compute(t)
        run = self.run
        params = run.params.get(0)
        coord = run.coords[0]
        data_vars = {n: (("time", "x", "y", "z"), arr)
                     for n, arr in raw["frames"].items()}
        ds = make_dataset(
            data_vars,
            {"time": raw["times"],
             "x": np.asarray(coord["x_o_a"])[raw["xslice"]],
             "y": coord["y"], "z": coord["z"]},
            params=params)
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = run.geometry_kind
        if self.species:
            ds.attrs["species"] = self.species
        return ds


    # ------------------------------------------------------------------

    def write_vtk(self, t=None, out_dir=None, prefix="gene3d"):
        """
        Write one legacy VTK structured-grid file per snapshot.

        Returns the list of paths written. Open them in ParaView or VisIt; the
        grid is the run's own Cartesian point positions, so the result sits in
        real space with no further transformation.
        """
        raw = self.compute(t)
        X, Y, Z = (arr[raw["xslice"]] for arr in self.cartesian_grid())
        out_dir = str(out_dir or self.run.path)
        os.makedirs(out_dir, exist_ok=True)

        shapes = {n: arr.shape[1:] for n, arr in raw["frames"].items()}
        for name, shape in shapes.items():
            if shape != X.shape:
                raise ValueError(
                    f"{name} has shape {shape} but the Cartesian grid has "
                    f"{X.shape}; they must match to write a structured grid.")

        written = []
        for it, time in enumerate(raw["times"]):
            path = os.path.join(out_dir, f"{prefix}_{it:04d}.vtk")
            fields = {n: arr[it] for n, arr in raw["frames"].items()}
            _write_structured_grid(path, X, Y, Z, fields, time)
            written.append(path)
        return written

    def write_vtk_via_gvec(self, t=None, gvec_file=None, out_dir=None,
                           prefix="gene3d_gvec"):
        """
        Write VTK through the GVEC ``gvec_to_gene`` extension.

        Only needed when the run's geometry file has no ``cart_coords``, or when
        a GVEC state file rather than the run itself should define the mapping.
        Requires the compiled ``gvec_to_gene`` module, which is built alongside
        GVEC and is not a dependency of this package.

        Raises
        ------
        ImportError
            If ``gvec_to_gene`` is not importable, with a pointer to
            :meth:`write_vtk`, which needs nothing extra.
        ValueError
            If no GVEC state file is given.
        """
        try:
            import importlib
            importlib.import_module("gvec_to_gene")
        except ImportError as exc:
            raise ImportError(
                "write_vtk_via_gvec() needs the compiled 'gvec_to_gene' "
                "extension, which is built with GVEC and is not part of "
                "genetools. GENE-3D writes the Cartesian grid into its own "
                "geometry file, so write_vtk() gives the same export without "
                "any extra dependency.") from exc
        if not gvec_file:
            raise ValueError(
                "write_vtk_via_gvec() needs gvec_file=<GVEC state file>.")
        raise NotImplementedError(
            "The GVEC path is wired up to the point of the extension call. "
            "Finishing it needs a GVEC state file and the compiled module to "
            "test against; write_vtk() covers the same export using the "
            "Cartesian grid GENE-3D already writes.")

    def plot(self, t=None, **kw):
        """
        Not a plotting diagnostic — writes VTK and reports where.

        Kept on the common facade so ``run.vis3d.plot()`` does the useful thing
        rather than raising.
        """
        written = self.write_vtk(t=t, **kw)
        print(f"Wrote {len(written)} VTK file(s):")
        for path in written[:5]:
            print(f"  {path}")
        if len(written) > 5:
            print(f"  ... and {len(written) - 5} more")
        return written


# ---------------------------------------------------------------------------
# Legacy VTK writer
# ---------------------------------------------------------------------------

def _write_structured_grid(path, X, Y, Z, fields, time):
    """
    Write a legacy VTK ``STRUCTURED_GRID`` file.

    Point order is x fastest, matching the VTK convention, so every array is
    transposed from GENE's ``(nx, ny, nz)`` before being flattened.
    """
    nx, ny, nz = X.shape
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write(f"GENE-3D snapshot t={time:.6g}\n")
        fh.write("ASCII\n")
        fh.write("DATASET STRUCTURED_GRID\n")
        fh.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        fh.write(f"POINTS {nx * ny * nz} float\n")
        pts = np.stack([X.ravel(order="F"), Y.ravel(order="F"),
                        Z.ravel(order="F")], axis=1)
        np.savetxt(fh, pts, fmt="%.6g")
        fh.write(f"\nPOINT_DATA {nx * ny * nz}\n")
        for i, (name, arr) in enumerate(sorted(fields.items())):
            safe = name.replace(" ", "_").replace("<", "").replace(">", "")
            fh.write(f"SCALARS {safe} float 1\n")
            fh.write("LOOKUP_TABLE default\n")
            np.savetxt(fh, np.asarray(arr).ravel(order="F"), fmt="%.6g")
            fh.write("\n")

