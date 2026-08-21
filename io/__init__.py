# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
genetools.io — File I/O, geometry, and coordinates for GENE simulations.

Classes
-------
BinaryReader
    Stream field/moment data from Fortran unformatted binary files.
BPReader
    Stream field/moment data from ADIOS2 BP files.
H5Reader
    Stream field/moment data from GENE and GENE-3D HDF5 files.
MultiSegmentReader
    Stitch multiple run-segment readers into a single transparent reader.
Params
    Load and merge GENE Fortran-90 namelist parameter files.

Functions
---------
set_runs(folder, exclude=None)
    Scan a GENE run directory and return sorted run suffix strings.
Geometry(folder, extensions, params)
    Load magnetic geometry for one or more run segments.
Coordinates(folder, extensions, params)
    Build coordinate arrays (x, kx, ky, z, vp, mu) for one or more segments.
"""

from .data import BinaryReader, BPReader, H5Reader, MultiSegmentReader
from .params import Params
from .utils import set_runs
from .geometry import Geometry
from .coordinates import Coordinates
from .profiles_loader import load_equilibrium_profiles, EquilibriumProfiles

__all__ = [
    "BinaryReader",
    "BPReader",
    "H5Reader",
    "MultiSegmentReader",
    "Params",
    "set_runs",
    "Geometry",
    "Coordinates",
    "load_equilibrium_profiles",
    "EquilibriumProfiles",
]
