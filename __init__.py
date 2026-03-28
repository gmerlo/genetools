
"""
genetools — Post-processing toolkit for GENE gyrokinetic simulations.

Sub-packages
------------
genetools.io
    File I/O, geometry, and coordinates.
genetools.diagnostics
    Diagnostics: nrg, contours, shearing rate, flux spectra.

Quick start
-----------
>>> from genetools.io import Params, set_runs, BinaryReader, MultiSegmentReader
>>> from genetools.io import Geometry, Coordinates
>>> from genetools.diagnostics import NrgReader, Contours, ShearingRate, Spectra

>>> folder = '/path/to/run/'
>>> runs   = set_runs(folder)
>>> params = Params(folder, runs)
>>> geom   = Geometry(folder, runs, params)
>>> coord  = Coordinates(folder, runs, params)

>>> # NRG diagnostics
>>> NrgReader(folder, params.get(0)).plot()

>>> # Field contours
>>> field_reader = MultiSegmentReader([
...     BinaryReader('field', folder, ext, params.get(fn))
...     for fn, ext in enumerate(runs)
... ])
>>> Contours().plot_timeseries_2d(field_reader, t_start=10., t_stop=2000.,
...     field=0, ifft='xy', params=params.get(0))

Backward-compatible flat imports
---------------------------------
The original flat module names are preserved via the sub-package structure.
You can import everything from the top level:

>>> from genetools import BinaryReader, Params, Geometry, Coordinates
>>> from genetools import NrgReader, Contours, ShearingRate, Spectra
"""

from . import io
from . import diagnostics

# Flat convenience imports — mirrors original API
from .io import (
    BinaryReader,
    BPReader,
    MultiSegmentReader,
    Params,
    set_runs,
    Geometry,
    Coordinates,
    load_equilibrium_profiles,
    EquilibriumProfiles,
)
from .diagnostics import (
    NrgReader,
    Contours,
    ShearingRate,
    Spectra,
    Profiles,
    Fluxes2D,
    SpectraGlobal,
)

__version__ = "0.2.0"

__all__ = [
    "io",
    "diagnostics",
    "BinaryReader",
    "BPReader",
    "MultiSegmentReader",
    "Params",
    "set_runs",
    "Geometry",
    "Coordinates",
    "NrgReader",
    "Contours",
    "ShearingRate",
    "Spectra",
    "Profiles",
    "Fluxes2D",
    "SpectraGlobal",
    "load_equilibrium_profiles",
    "EquilibriumProfiles",
]