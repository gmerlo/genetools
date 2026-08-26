# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
genetools.diagnostics — post-processing diagnostics for GENE simulations.

One class per diagnostic, each handling every geometry it supports internally:
flux tube (spectral x and y), x-global (real-space x), and GENE-3D
(``xy_global``, real space in x *and* y). :class:`~genetools.run.Run` exposes
them all and none of them needs a geometry argument —
:attr:`~genetools.run.Run.geometry_kind` decides.

Diagnostics with no meaning outside GENE-3D declare ``supported`` and refuse on
construction rather than silently reducing the data some other way.

Shared machinery
----------------
``_base.py``    ``CachingDiagnostic`` (HDF5 persistence, time windows, pairing
                field with moment snapshots by time value) and
                ``RunDiagnostic`` (the Run-native surface: ``.data``,
                ``.plot(t=...)``, ``.save()``).
``_gene3d.py``  GENE-3D-specific physics: ``flux_geomfac``, the ExB and
                magnetic-flutter velocities, Jacobian-weighted reductions, and
                the flux-consistency check against the fluxes GENE-3D writes
                itself.
"""

from .nrg import NrgReader
from .contours import Contours
from .shearingrate import ShearingRate
from .spectra import Spectra
from .spectra_global import SpectraGlobal
from .profiles import Profiles
from .fluxes2d import Fluxes2D
from .ballooning import Ballooning
from .growthrate import GrowthRate
from .amplitude import AmplitudeSpectra
from .zonal import Zonal
from .profile_diag import ProfileDiag
from .timetraces import TimeTraces
from .gam import Gam
from .chi import ChiGradient
from .omega import Omega
from .geometry_plots import GeometryPlots
from .velocity import SrcMom, VspSlice
from .planes import Planes
from .vis import Vis

__all__ = [
    # All geometries
    "NrgReader",
    "Contours",
    "ShearingRate",
    "Spectra",
    "SpectraGlobal",
    "Profiles",
    "Fluxes2D",
    "Ballooning",
    "GrowthRate",
    "AmplitudeSpectra",
    "Zonal",
    "ProfileDiag",
    # GENE-3D only (for now)
    "TimeTraces",
    "Gam",
    "ChiGradient",
    "Omega",
    "GeometryPlots",
    "SrcMom",
    "VspSlice",
    "Planes",
    "Vis",
]
