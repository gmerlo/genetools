"""
genetools — Post-processing toolkit for GENE gyrokinetic simulations.

Modules
-------
params
    Load and parse GENE Fortran-90 namelist parameter files.
data
    Stream field and moment data from binary or ADIOS2 BP files.
nrg
    Read and plot energy/flux diagnostic files.
utils
    File-system helpers for GENE run directories.
geometry
    Parse geometry files (local and global).
spectra
    Compute and store time-averaged flux spectra.
coordinates
    Build coordinate arrays (kx, ky, z).
contours
    Contour and 2-D visualisations.

Quick start
-----------
>>> from genetools.params import Params
>>> from genetools.nrg import NrgReader
>>> params = Params('/path/to/run/').get()
>>> reader = NrgReader('/path/to/run/', params)
>>> times, data = reader.read_all()
>>> reader.plot()
"""

from .params import Params
from .data import BinaryReader, BPReader
from .nrg import NrgReader
from .utils import set_runs

__all__ = [
    "Params",
    "BinaryReader",
    "BPReader",
    "NrgReader",
    "set_runs",
]
