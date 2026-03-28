"""
genetools.diagnostics — Post-processing diagnostics for GENE simulations.
 
Classes
-------
NrgReader
    Read and plot energy/flux diagnostic files (``nrg*``).
Contours
    Stream and plot 2D field/moment slice plots.
ShearingRate
    Compute, cache, and plot ExB shearing rate diagnostics.
Spectra
    Compute and store time-averaged flux spectra (kx, ky, z).
"""
 
from .nrg import NrgReader
from .contours import Contours
from .shearingrate import ShearingRate
from .spectra import Spectra
from .profiles import Profiles
from .fluxes2d import Fluxes2D
from .spectra_global import SpectraGlobal

__all__ = [
    "NrgReader",
    "Contours",
    "ShearingRate",
    "Spectra",
    "Profiles",
    "Fluxes2D",
    "SpectraGlobal",
]
