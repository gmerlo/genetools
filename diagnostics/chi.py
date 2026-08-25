# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
chi.py — GENE-3D heat diffusivity against the driving gradient.

The flux-gradient relation, which is what a global run is usually run to
measure::

    chi = Q / ( <g^xx> * n * T * omt )

with everything on the right-hand side taken self-consistently from the
simulation rather than from the input file: ``n`` and ``T`` are the *evolved*
total profiles (background plus flux-surface-averaged perturbation) and ``omt``
their logarithmic gradient, so as the profiles relax the diagnostic follows them.
That is the whole point — in a global run the gradient is an output, not an
input, and plotting ``chi`` against it traces out the transport response.

The pieces are reused rather than recomputed: ``Q`` comes from
:class:`~genetools.diagnostics.fluxes2d.Fluxes2D` and the profiles from
:class:`~genetools.diagnostics.profiles.Profiles`, so all three
diagnostics necessarily agree about normalisation.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from genetools._xr import make_dataset, unit_attrs
from genetools.diagnostics._base import RunDiagnostic
from genetools.diagnostics import _gene3d as g3
from genetools.diagnostics.fluxes2d import Fluxes2D
from genetools.diagnostics.profiles import Profiles


class ChiGradient(RunDiagnostic):
    """
    Heat diffusivity and the driving gradient, per species and over time.

    Parameters
    ----------
    run : genetools.run.Run
    x_avg_lims : (float, float), optional
        Radial averaging range in ``x/a`` for the scalar traces. Defaults to
        trimming 10% from each end.
    buffer_frac : float
        Fraction trimmed from each radial end when *x_avg_lims* is unset.
    """

    name = "chi"
    supported = ("xy_global",)

    def __init__(self, run, x_avg_lims=None, buffer_frac=0.1):
        super().__init__(run)
        self.x_avg_lims = x_avg_lims
        self.buffer_frac = buffer_frac
        self._fluxes = Fluxes2D(run)
        self._profiles = Profiles(run)
        self._cache = {}

    # ------------------------------------------------------------------

    def _geom_factor(self):
        """``<g^xx>`` over the flux surface, the metric factor in chi."""
        geom = self.run.geometry[0]
        return g3.flux_surface_average(geom["metric"]["gxx"], geom["Jacobian"])

    def compute(self, t=None):
        """Return the radial and scalar chi traces per species."""
        key = (tuple(t) if isinstance(t, (tuple, list)) else t,
               tuple(self.x_avg_lims) if self.x_avg_lims else None)
        if key in self._cache:
            return self._cache[key]

        flux_ds = self._fluxes.dataset(t)
        prof_ds = self._profiles.dataset(t)
        geom_fac = self._geom_factor()
        x_o_a = np.asarray(prof_ds["x"], dtype=float)
        xsl = g3.radial_slice(x_o_a, limits=self.x_avg_lims,
                             buffer_frac=self.buffer_frac)

        # The two datasets are streamed from the same moment files, so their
        # time axes match; guard rather than assume.
        n_t = min(flux_ds.sizes["time"], prof_ds.sizes["time"])
        times = np.asarray(prof_ds["time"])[:n_t]

        per = {}
        for name in prof_ds["species"].values:
            name = str(name)
            Q = np.asarray(flux_ds["Q_total"].sel(species=name))[:n_t]
            T = np.asarray(prof_ds["T"].sel(species=name))[:n_t]
            n = np.asarray(prof_ds["n"].sel(species=name))[:n_t]
            omt = np.asarray(prof_ds["omt"].sel(species=name))[:n_t]
            omn = np.asarray(prof_ds["omn"].sel(species=name))[:n_t]

            denom = geom_fac[np.newaxis, :] * n * T * omt
            with np.errstate(divide="ignore", invalid="ignore"):
                chi = np.where(denom != 0, Q / denom, np.nan)

            # Scalar traces: average numerator and denominator separately over
            # the window before dividing. Averaging the ratio instead lets a
            # near-zero local gradient dominate the result.
            num_avg = Q[:, xsl].mean(axis=1)
            den_avg = denom[:, xsl].mean(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                chi_avg = np.where(den_avg != 0, num_avg / den_avg, np.nan)

            per[name] = {
                "chi": chi, "Q": Q, "denominator": denom,
                "omt": omt, "omn": omn,
                "chi_avg": chi_avg,
                "omt_avg": omt[:, xsl].mean(axis=1),
                "omn_avg": omn[:, xsl].mean(axis=1),
            }

        result = {"species": per, "times": times, "x_o_a": x_o_a,
                  "geom_factor": geom_fac, "xslice": xsl}
        self._cache[key] = result
        return result

    # ------------------------------------------------------------------

    def _chi_reference(self):
        """
        SI conversion for chi, in m^2/s.

        From ``Q = -n chi dT/dx``: ``chi_ref = Q_gb * L_ref / p_ref``, which is
        the usual ``c_ref rho_ref^2 / L_ref``.
        """
        units = self.run.params.get(0).get("units", {}) or {}
        Qgb, Lref, pref = units.get("Qgb"), units.get("Lref"), units.get("pref")
        if None in (Qgb, Lref, pref) or pref == 0:
            return None
        return float(Qgb) * float(Lref) / float(pref)

    def dataset(self, t=None):
        """Return chi and the gradients as an :class:`xarray.Dataset`."""
        raw = self.compute(t)
        params = self.run.params.get(0)
        names = list(raw["species"])

        def stack(key):
            return np.stack([raw["species"][n][key] for n in names], axis=0)

        data_vars = {
            "chi": (("species", "time", "x"), stack("chi")),
            "omt": (("species", "time", "x"), stack("omt")),
            "omn": (("species", "time", "x"), stack("omn")),
            "chi_avg": (("species", "time"), stack("chi_avg")),
            "omt_avg": (("species", "time"), stack("omt_avg")),
            "omn_avg": (("species", "time"), stack("omn_avg")),
        }
        ds = make_dataset(data_vars, {"x": raw["x_o_a"], "time": raw["times"]},
                          species=names, params=params)
        ds = ds.assign(geom_factor=("x", raw["geom_factor"]))
        ds["geom_factor"].attrs["long_name"] = "<g^xx> over the flux surface"
        for name in ("chi", "chi_avg"):
            ds[name].attrs["units"] = "chi_gb (normalised)"
        for name in ("omt", "omt_avg"):
            ds[name].attrs["units"] = "a/L_T"
        for name in ("omn", "omn_avg"):
            ds[name].attrs["units"] = "a/L_n"

        chi_ref = self._chi_reference()
        if chi_ref is not None:
            for name in ("chi", "chi_avg"):
                ds[name + "_SI"] = ds[name] * chi_ref
                ds[name + "_SI"].attrs["units"] = "m^2 s^-1"
            ds.attrs["chi_ref"] = chi_ref

        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = self.run.geometry_kind
        x = raw["x_o_a"][raw["xslice"]]
        ds.attrs["x_avg_range"] = [float(x[0]), float(x[-1])]
        return ds



    # ------------------------------------------------------------------

    def plot(self, t=None, si=False, **kw):
        """Radial chi, its time trace, and chi against the driving gradient."""
        ds = self.dataset(t)
        key = "chi_SI" if (si and "chi_SI" in ds) else "chi"
        key_avg = "chi_avg_SI" if (si and "chi_avg_SI" in ds) else "chi_avg"
        x = np.asarray(ds["x"])
        times = np.asarray(ds["time"])

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for name in ds["species"].values:
            axes[0].plot(x, np.asarray(self._t_average(ds[key].sel(species=name))),
                         label=str(name))
            axes[1].plot(times, np.asarray(ds[key_avg].sel(species=name)),
                         label=str(name))
            axes[2].plot(np.asarray(ds["omt_avg"].sel(species=name)),
                         np.asarray(ds[key_avg].sel(species=name)),
                         "o-", ms=3, label=str(name))

        axes[0].set_xlabel(r"$x/a$")
        axes[0].set_title(r"$\chi(x)$, time averaged")
        axes[1].set_xlabel(r"$t\;[L_{\rm ref}/c_{\rm ref}]$")
        axes[1].set_title(rf"$\chi$ over $x/a \in "
                          rf"[{ds.attrs['x_avg_range'][0]:.2f},"
                          rf"{ds.attrs['x_avg_range'][1]:.2f}]$")
        axes[2].set_xlabel(r"$a/L_T$")
        axes[2].set_title(r"$\chi$ vs. driving gradient")
        for ax in axes:
            ax.set_ylabel(ds[key].attrs.get("units", ""))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        plt.show()
