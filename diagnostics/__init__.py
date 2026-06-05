# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
from .ballooning import Ballooning
from .growthrate import GrowthRate
from .amplitude import AmplitudeSpectra
from .zonal import Zonal
from .profile_diag import ProfileDiag

__all__ = [
    "NrgReader",
    "Contours",
    "ShearingRate",
    "Spectra",
    "Profiles",
    "Fluxes2D",
    "SpectraGlobal",
    "Ballooning",
    "GrowthRate",
    "AmplitudeSpectra",
    "Zonal",
    "ProfileDiag",
]
