# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
planes.py — GENE-3D field-aligned to straight-field-line remapping.

GENE-3D's binormal coordinate ``y`` is field-aligned: a point at fixed ``y``
follows the field line as ``z`` advances. To see a turbulence structure the way a
diagnostic looking into the machine would, the data has to be resampled onto a
grid of geometric angles, and that is what this does. For a point at
``(theta, phi)`` on flux surface ``x``, the field-aligned coordinate is

    y = sign_Bpol * |C_y| / rhostar * ( q(x) * theta - phi )

taken modulo the box length ``ly``. The wrap is the crux: a global run resolves
only one binormal box, and the same field line re-enters it many times as
``theta`` and ``phi`` sweep a full torus, so the modulo is what makes the map
single-valued rather than an extrapolation off the end of the grid.

Resampling in ``z`` and ``y`` is done by interpolation. The ``y`` interpolation is
periodic — the box is periodic in ``y``, so a point that wraps past the last grid
cell must come back to the first, not be linearly extrapolated past it (which is
what the reference GUI does, via ``fill_value="extrapolate"``).

The mode numbers ``(n, m)`` of the remapped plane follow from a 2-D FFT, with
``n`` scaled by ``n0_global`` because the toroidal grid spans only
``2 pi / n0_global``.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from genetools._xr import make_dataset, unit_attrs
from genetools.diagnostics._base import RunDiagnostic
from genetools.diagnostics import _gene3d as g3


class Planes(RunDiagnostic):
    """
    GENE-3D data remapped onto geometric ``(theta, phi)`` angles.

    Parameters
    ----------
    run : genetools.run.Run
    quantities : sequence of str
        Variables to remap (from the field or moment file).
    species : str, optional
        Species supplying moment quantities (default: the first).
    n_theta, n_phi : int
        Resolution of the output angular grid.
    rms : bool
        Remap the root-mean-square over ``y`` instead of the field itself, which
        is what shows the envelope of the turbulence rather than one realisation.
    x_avg : bool
        Average over the radial window before remapping, giving one plane
        instead of a radially resolved stack.
    xlim : (float, float), optional
        Radial window in ``x/a``.
    t_avg : bool
        Average over time before remapping.
    """

    name = "planes"
    supported = ("xy_global",)

    def __init__(self, run, quantities=("phi",), species=None,
                 n_theta=128, n_phi=64, rms=False, x_avg=False,
                 xlim=None, t_avg=True):
        super().__init__(run)
        self.quantities = tuple(quantities)
        self.species = species or (run.species[0] if run.species else None)
        self.n_theta = int(n_theta)
        self.n_phi = int(n_phi)
        self.rms = bool(rms)
        self.x_avg = bool(x_avg)
        self.xlim = xlim
        self.t_avg = bool(t_avg)
        self._cache = {}

    # ------------------------------------------------------------------
    # Grids
    # ------------------------------------------------------------------

    @property
    def theta(self):
        return np.linspace(-np.pi, np.pi, self.n_theta, endpoint=False)

    @property
    def phi(self):
        n0 = float(self.run.params.get(0)["box"].get("n0_global", 1) or 1)
        return np.linspace(0.0, 2.0 * np.pi / n0, self.n_phi, endpoint=False)

    # ------------------------------------------------------------------


    def compute(self, t=None):
        """Stream the requested variables and remap each onto the angle grid."""
        key = tuple(t) if isinstance(t, (tuple, list)) else t
        if key in self._cache:
            return self._cache[key]

        run = self.run
        params = run.params.get(0)
        geom = run.geometry[0]
        coord = run.coords[0]
        J = geom["Jacobian"]

        q_prof = np.asarray((geom.get("profiles") or {}).get("q"), dtype=float)
        if q_prof.ndim != 1 or q_prof.size != J.shape[0]:
            raise ValueError(
                "The safety-factor profile q(x) is needed to map onto geometric "
                "angles, but the geometry file does not provide it as a radial "
                "profile.")
        C_y = np.atleast_1d(np.asarray(geom["metric"]["C_y"], dtype=float))
        rhostar = float(params["geometry"]["rhostar"])
        sign = float(params["info"].get("sign_bpol_cw",
                                        params["info"].get("sign_Bpol_CW", 1.0)))
        ly = float(params["box"]["ly"])
        y = np.asarray(coord["y"], dtype=float)
        z = np.asarray(coord["z"], dtype=float)
        x_o_a = np.asarray(coord["x_o_a"], dtype=float)
        xsl = g3.radial_slice(x_o_a, limits=self.xlim) if self.xlim else slice(None)

        theta, phi = self.theta, self.phi

        planes, times = {}, None
        for reader, names in self._sources(self.quantities,
                                        self.species):
            _, idx = self._indices(reader, t)
            slots = {n: reader.index_of(n) for n in names}
            acc = {n: [] for n in names}
            got = []
            for time, arrays in reader.stream_selected(idx):
                got.append(time)
                for n in names:
                    acc[n].append(arrays[slots[n]])
            if times is None:
                times = np.asarray(got)

            for n in names:
                stack = np.asarray(acc[n])
                if self.rms:
                    stack = np.sqrt((stack ** 2).mean(axis=2, keepdims=True))
                    stack = np.repeat(stack, y.size, axis=2)
                if self.t_avg:
                    stack = self._time_average(stack, times)[np.newaxis, ...]
                frames = []
                for frame in stack:
                    resampled = _resample_z(frame, z, theta)
                    if self.x_avg:
                        weights = J[xsl].mean(axis=(1, 2))
                        radial = np.average(resampled[xsl], axis=0,
                                            weights=weights)[np.newaxis, ...]
                        q_used = np.array([np.average(q_prof[xsl],
                                                      weights=weights)])
                    else:
                        radial = resampled[xsl]
                        q_used = q_prof[xsl]
                    frames.append(_resample_y(radial, y, ly, q_used, C_y,
                                              rhostar, sign, theta, phi))
                planes[n] = np.asarray(frames)

        alpha = _alpha_map(q_prof[xsl] if not self.x_avg
                           else np.array([q_prof[xsl].mean()]),
                           theta, phi, sign)

        result = {"planes": planes, "theta": theta, "phi": phi,
                  "times": times, "alpha": alpha,
                  "x_o_a": (np.array([x_o_a[xsl].mean()]) if self.x_avg
                            else x_o_a[xsl])}
        self._cache[key] = result
        return result

    # ------------------------------------------------------------------

    def dataset(self, t=None):
        """Return the remapped planes as an :class:`xarray.Dataset`."""
        raw = self.compute(t)
        params = self.run.params.get(0)
        # The toroidal angle is `varphi`, not `phi`: `phi` is the
        # electrostatic potential, and a coordinate sharing a data variable's
        # name makes the Dataset unbuildable.
        dims = ("time", "x", "varphi", "theta")
        data_vars = {name: (dims, arr) for name, arr in raw["planes"].items()}
        data_vars["alpha"] = (("x", "varphi", "theta"), raw["alpha"])
        ds = make_dataset(
            data_vars,
            {"time": raw["times"] if not self.t_avg else np.array([np.nan]),
             "x": raw["x_o_a"], "varphi": raw["phi"], "theta": raw["theta"]},
            params=params)
        if self.t_avg and "time" in ds.dims:
            ds = ds.isel(time=0, drop=True)
        ds["alpha"].attrs["long_name"] = "field-line label q*theta - phi"
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = self.run.geometry_kind
        ds.attrs["rms"] = int(self.rms)
        ds.attrs["n0_global"] = float(params["box"].get("n0_global", 1) or 1)
        if self.species:
            ds.attrs["species"] = self.species
        return ds



    def mode_spectrum(self, t=None, quantity=None):
        """
        Return the ``(n, m)`` mode spectrum of a remapped plane.

        Returns ``(n, m, power)`` where *power* is ``|FFT2|^2`` summed over the
        radial window. ``n`` is scaled by ``n0_global`` because the toroidal grid
        covers only ``2 pi / n0_global``. The toroidal angle is named ``varphi``
        in the dataset to leave ``phi`` free for the potential.
        """
        ds = self.dataset(t)
        name = quantity or self.quantities[0]
        arr = np.asarray(ds[name])
        if arr.ndim == 4:                       # time not averaged out
            arr = arr.mean(axis=0)
        power = np.abs(np.fft.fft2(arr, axes=(-2, -1))) ** 2
        power = power.sum(axis=0) if power.ndim == 3 else power
        n0 = float(ds.attrs["n0_global"])
        n = np.fft.fftfreq(self.n_phi) * self.n_phi * n0
        m = np.fft.fftfreq(self.n_theta) * self.n_theta
        return n, m, power

    # ------------------------------------------------------------------

    def plot(self, t=None, fft=False, **kw):
        """Plot each remapped plane, with the field-line labels overlaid."""
        ds = self.dataset(t)
        theta = np.asarray(ds["theta"])
        varphi = np.asarray(ds["varphi"])

        for name in self.quantities:
            if name not in ds:
                continue
            da = ds[name]
            if "time" in da.dims:
                da = self._t_average(da)
            plane = np.asarray(da.mean("x") if "x" in da.dims else da)
            alpha = np.asarray(ds["alpha"].mean("x"))

            fig, ax = plt.subplots(figsize=(7.5, 5))
            mesh = ax.pcolormesh(varphi, theta, plane.T, shading="nearest")
            ax.contour(varphi, theta, alpha.T, levels=8, colors="w",
                       linewidths=0.5, alpha=0.6)
            ax.set_xlabel(r"$\varphi$")
            ax.set_ylabel(r"$\theta$")
            ax.set_title(("RMS of " if self.rms else "") + name)
            fig.colorbar(mesh, ax=ax)
            fig.tight_layout()

            if fft:
                n, m, power = self.mode_spectrum(t, quantity=name)
                fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
                axes[0].pcolormesh(np.fft.fftshift(n), np.fft.fftshift(m),
                                   np.fft.fftshift(power).T, shading="nearest")
                axes[0].set_xlabel("$n$")
                axes[0].set_ylabel("$m$")
                axes[0].set_title("$|f_{n,m}|^2$")
                pn = power.sum(axis=1)
                pm = power.sum(axis=0)
                axes[1].plot(np.fft.fftshift(n), np.fft.fftshift(pn))
                axes[1].set_xlabel("$n$")
                axes[1].set_title(f"peak |n| = {abs(n[int(np.argmax(pn))]):.0f}")
                axes[2].plot(np.fft.fftshift(m), np.fft.fftshift(pm))
                axes[2].set_xlabel("$m$")
                axes[2].set_title(f"peak |m| = {abs(m[int(np.argmax(pm))]):.0f}")
                for ax_ in axes[1:]:
                    ax_.grid(True, alpha=0.3)
                fig2.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def _resample_z(var, z, theta):
    """
    Interpolate ``(nx, ny, nz)`` from the ``z`` grid onto *theta*.

    ``z`` is periodic over the simulation domain, so the interpolation wraps
    rather than clamping at the ends.
    """
    period = z[-1] - z[0] + (z[1] - z[0]) if z.size > 1 else 2 * np.pi
    z_ext = np.concatenate((z, [z[0] + period]))
    var_ext = np.concatenate((var, var[:, :, :1]), axis=2)
    out = np.empty(var.shape[:2] + (theta.size,))
    for i in range(var.shape[0]):
        for j in range(var.shape[1]):
            out[i, j, :] = np.interp(theta, z_ext, var_ext[i, j, :])
    return out


def _resample_y(var, y, ly, q_prof, C_y, rhostar, sign, theta, phi):
    """
    Interpolate the field-aligned ``y`` onto geometric ``(phi, theta)``.

    ``var`` is ``(nx, ny, n_theta)``; the result is ``(nx, n_phi, n_theta)``.
    Interpolation in ``y`` is periodic: the box is periodic in ``y``, and the
    ``modulo`` below deliberately wraps points round it many times.
    """
    nx = var.shape[0]
    y_ext = np.concatenate((y, [y[0] + ly]))
    out = np.empty((nx, phi.size, theta.size))
    for i in range(nx):
        cy = abs(C_y[i] if C_y.size == nx else C_y[0])
        q = q_prof[i] if np.size(q_prof) == nx else q_prof[0]
        # y for every (phi, theta) on this surface, wrapped into [0, ly).
        y_target = np.mod(
            sign * cy / rhostar
            * (q * sign * theta[np.newaxis, :] - phi[:, np.newaxis]), ly)
        for k in range(theta.size):
            col = np.concatenate((var[i, :, k], var[i, :1, k]))
            out[i, :, k] = np.interp(y_target[:, k], y_ext, col)
    return out


def _alpha_map(q_prof, theta, phi, sign):
    """Field-line label ``q*theta - phi`` on the angular grid."""
    q = np.atleast_1d(np.asarray(q_prof, dtype=float))
    return (q[:, np.newaxis, np.newaxis]
            * sign * theta[np.newaxis, np.newaxis, :]
            - phi[np.newaxis, :, np.newaxis])

