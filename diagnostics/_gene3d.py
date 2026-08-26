# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
_gene3d.py — physics primitives specific to GENE-3D geometry.

GENE-3D data is real and real-space in all three coordinates, ``(nx, ny, nz)``,
with a Jacobian of the same shape. Three consequences shape everything here:

* **Averages are Jacobian-weighted in y as well as z.** A flux-surface average
  is a weighted mean over both, not a sum over ky times a z-average.
* **There is no Hermitian symmetry to exploit.** The spectral-y cases store only
  non-negative ky and weight ``ky > 0`` by two. GENE-3D stores the full
  real-space y direction, so an FFT of it already contains both signs and no
  such factor applies anywhere.
* **Wavenumbers are derived, not stored.** ``ky`` is the FFT grid of ``yval``;
  the parameters file has no ``kymin`` at all.

Only GENE-3D-specific things live here. Time averaging, time-window handling and
the pairing of field with moment snapshots are general and come from
:mod:`genetools.diagnostics._base`.

Flux conventions
----------------
Nothing below is a convention this module chose. GENE-3D computes its own fluxes
in ``diag_3d.F90``::

    Gamma_es = -n     * dphi/dy   * flux_geomfac
    Gamma_em = +u_par * dA_par/dy * flux_geomfac

and ``geometry.F90`` defines

    flux_geomfac = 1 / C_xy                    (norm_flux_projection = F)
                 = 1 / (C_xy * sqrt(g^xx))     (norm_flux_projection = T)

So there is **no** ``1/Bref`` factor, and the magnetic-flutter velocity carries
the **opposite** sign to the ExB one. Because GENE-3D writes ``Gamma_es`` and
``Q_es`` to the moment file, this is checkable rather than assumed — see
:func:`check_flux_consistency`.
"""

from __future__ import annotations

import warnings

import numpy as np


def _as_radial(value):
    """Broadcast a scalar or radial profile against an ``(nx, ny, nz)`` array."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return arr
    return arr[:, np.newaxis, np.newaxis]


# ---------------------------------------------------------------------------
# ExB velocity convention
# ---------------------------------------------------------------------------
#
# GENE-3D fixes this itself, so nothing here is a free choice. Its own flux
# diagnostic (``diag_3d.F90``) computes
#
#     Gamma_es = -n * dphi/dy * flux_geomfac
#     Gamma_em = +u_par * dA_par/dy * flux_geomfac
#
# and ``flux_geomfac`` (``geometry.F90``) is
#
#     1 / C_xy                        (norm_flux_projection = F)
#     1 / (C_xy * sqrt(g^xx))         (norm_flux_projection = T)
#
# So: there is **no** ``1/Bref`` factor, and the magnetic-flutter velocity has
# the **opposite** sign to the ExB one. The reference GUI gets both of these
# wrong in one branch or the other — its x-global branch divides by ``Bref``,
# its GENE-3D branch flips the flutter sign — and neither applies the
# ``sqrt(g^xx)`` projection at all, even though the same flag controls which
# flux-surface area the result has to be multiplied by to give a total.
#
# Because GENE-3D writes ``Gamma_es``/``Q_es`` directly, this is checkable
# rather than assumed: :func:`check_flux_consistency` compares the ky-summed
# spectrum against the code's own flux, and any surviving normalisation error
# shows up as a constant ratio.

def flux_geomfac(geom, params) -> np.ndarray:
    """
    Return GENE-3D's ``flux_geomfac``, the geometric factor in every flux.

    ``1/C_xy``, and additionally ``1/sqrt(g^xx)`` when ``norm_flux_projection``
    is set — in which case the flux is projected onto the physical surface
    normal and the matching area is the ``sqrt(g^xx)``-weighted one
    (``geom['area']['Area']``) rather than ``dVdx``.

    Returned y-averaged and broadcast back over y: an average over x and z at
    fixed ky needs a weight independent of y, and GENE-3D's metric has no y
    dependence today (see :func:`jacobian_yz`).
    """
    C_xy = np.asarray(geom["metric"]["C_xy"], dtype=float)
    fac = 1.0 / _as_radial(C_xy)
    if params.get("geometry", {}).get("norm_flux_projection", False):
        gxx = np.asarray(geom["metric"]["gxx"], dtype=float)
        gxx_y = np.broadcast_to(gxx.mean(axis=1)[:, np.newaxis, :], gxx.shape)
        fac = fac / np.sqrt(gxx_y)
    return np.asarray(fac)


# ---------------------------------------------------------------------------
# Fourier transforms
# ---------------------------------------------------------------------------

def to_ky(var: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Transform a real-space y axis to ky, normalised so mode amplitudes are
    physical (``fft`` divided by the number of points).

    The result spans the full signed ky range in FFT order, matching the ``ky``
    array built by :func:`~genetools.io.coordinates.load_coord_xy_global`.
    """
    n = var.shape[axis]
    return np.fft.fft(var, axis=axis) / n

def to_kx(var: np.ndarray, axis: int = 0) -> np.ndarray:
    """As :func:`to_ky`, for the radial axis."""
    n = var.shape[axis]
    return np.fft.fft(var, axis=axis) / n


# ---------------------------------------------------------------------------
# Jacobian weights
# ---------------------------------------------------------------------------

def jacobian_yz(J: np.ndarray) -> np.ndarray:
    """
    Return the Jacobian averaged over y, broadcast back to ``(nx, ny, nz)``.

    Averages over x and z at fixed ky need a weight that does not itself depend
    on y — otherwise the weighting mixes binormal modes. GENE-3D's metric is
    currently y-independent (``geometry.F90`` notes the gathering in y still has
    to be added "once we really have y-dep. metrics"), so this is exact today
    and stays well-defined if that changes.
    """
    return np.broadcast_to(J.mean(axis=1)[:, np.newaxis, :], J.shape)

def flux_surface_average(var: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Jacobian-weighted average over y and z, giving a radial profile."""
    return np.average(var, weights=J, axis=(1, 2))

def xz_average(var: np.ndarray, J: np.ndarray, xslice=slice(None)):
    """
    Jacobian-weighted average over x and z, giving a ky spectrum.

    *xslice* restricts the radial range — useful for excluding the buffer
    regions, where the Krook operators make the fluxes unphysical.
    """
    weights = jacobian_yz(J)[xslice]
    return np.average(var[xslice], weights=weights, axis=(0, 2))

def index_window(values, limits, n=None) -> slice:
    """
    Return the index slice of *values* covering *limits*, inclusive.

    The nearest grid point to each bound is used. Equal bounds select that one
    point rather than widening to two: asking for the plane at ``z = 0`` must not
    average in a neighbour.

    Parameters
    ----------
    values : array
        Monotonic coordinate grid.
    limits : (float, float) or None
        Inclusive bounds in the units of *values*. ``None`` selects everything.
    n : int, optional
        Expected length of the axis being sliced. When given and it disagrees
        with ``len(values)`` the whole axis is returned, so a window in a real
        coordinate is ignored rather than misapplied to a transformed axis.
    """
    if limits is None:
        return slice(None)
    arr = np.asarray(values, dtype=float)
    if n is not None and arr.size != n:
        return slice(None)
    lo, hi = float(limits[0]), float(limits[1])
    i0 = int(np.argmin(np.abs(arr - lo)))
    i1 = int(np.argmin(np.abs(arr - hi)))
    if i1 < i0:
        i0, i1 = i1, i0
    return slice(i0, i1 + 1)


def radial_slice(x_o_a, limits=None, buffer_frac=None) -> slice:
    """
    Return a radial slice from *limits* in ``x/a``, or an inner-region fraction.

    Parameters
    ----------
    x_o_a : array
        Radial grid in ``x/a``.
    limits : (float, float), optional
        Inclusive bounds in ``x/a``. Nearest grid points are used.
    buffer_frac : float, optional
        Used only when *limits* is ``None``: trim this fraction of the grid from
        each end. The reference GUI hard-codes ``nx0/10``, which is what
        ``buffer_frac=0.1`` reproduces.
    """
    x = np.asarray(x_o_a, dtype=float)
    n = x.size
    if limits is not None:
        return index_window(x, limits)
    if buffer_frac:
        cut = int(n * float(buffer_frac))
        if 2 * cut < n:
            return slice(cut, n - cut)
    return slice(None)


# ---------------------------------------------------------------------------
# ExB and magnetic-flutter velocities
# ---------------------------------------------------------------------------

def exb_velocity_ky(phi, ky, geomfac) -> np.ndarray:
    """
    Radial ExB drift velocity per binormal mode, ``v_E^x(x, ky, z)``.

    *phi* is the real-space potential ``(nx, ny, nz)``; it is transformed to ky
    here. This is the ky representation of GENE-3D's own
    ``v_E^x = -flux_geomfac * dphi/dy``, so with ``d/dy -> i k_y``::

        v_E^x(k_y) = -i k_y phi(k_y) * flux_geomfac
    """
    return (-1j * ky[np.newaxis, :, np.newaxis] * to_ky(phi)) * geomfac

def flutter_velocity_ky(a_par, ky, geomfac) -> np.ndarray:
    """
    Radial magnetic-flutter velocity per binormal mode, ``B_x/B``.

    Note the sign: GENE-3D builds its electromagnetic fluxes as
    ``+u_par * dA_par/dy * flux_geomfac``, opposite to the electrostatic case,
    so this is ``+i k_y A_par(k_y) * flux_geomfac``. Carrying the ExB sign over
    here (as one of the reference GUI's two branches does) flips the sign of
    every electromagnetic flux.
    """
    return (1j * ky[np.newaxis, :, np.newaxis] * to_ky(a_par)) * geomfac


# ---------------------------------------------------------------------------
# Cross-check against the fluxes GENE-3D computes itself
# ---------------------------------------------------------------------------

def check_flux_consistency(spectrum_sum, code_flux, label, tol=0.05):
    """
    Warn when a reconstructed flux disagrees with the one GENE-3D wrote.

    GENE-3D computes ``Gamma_es`` and ``Q_es`` internally and writes them to the
    moment file, so any flux rebuilt from ``phi`` and the moments can be checked
    against them. A near-constant ratio other than one is the signature of a
    normalisation choice (see :data:`EXB_DIVIDE_BY_BREF`); a scattered ratio
    points at something else entirely.

    Returns the ratio ``spectrum_sum / code_flux``, or ``nan`` if the reference
    is zero.
    """
    ref = float(np.real(code_flux))
    got = float(np.real(spectrum_sum))
    if ref == 0.0:
        return float("nan")
    ratio = got / ref
    if abs(ratio - 1.0) > tol:
        warnings.warn(
            f"{label}: ky-summed flux is {ratio:.4g}x the value GENE-3D wrote "
            f"({got:.6g} vs {ref:.6g}). The ExB velocity normalisation is "
            "the usual cause — see flux_geomfac in "
            "genetools.diagnostics.gene3d._common.",
            RuntimeWarning, stacklevel=3)
    return ratio


# ---------------------------------------------------------------------------
# Reader helpers
# ---------------------------------------------------------------------------

def pick(reader, arrays, name):
    """Return the array called *name* from one streamed snapshot."""
    return arrays[reader.index_of(name)]

def has_var(reader, name) -> bool:
    """Whether *reader* exposes a variable called *name*."""
    return name in reader.var_names
