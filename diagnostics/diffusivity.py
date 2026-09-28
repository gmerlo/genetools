# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
diffusivity.py — heat and particle diffusivity against the background profiles.

The flux-gradient relation with the gradient taken from the **equilibrium**
profiles the run was set up with, rather than from the evolved ones::

    chi = Q     / ( <g^xx> n T omt )
    D   = Gamma / ( <g^xx> n omn   )

Both in gyro-Bohm units, whose reference is the same for the two:
``c_ref rho_ref^2 / L_ref``.

Why the background, and how this differs from `chi.py`
-----------------------------------------------------
:class:`~genetools.diagnostics.chi.ChiGradient` divides by the *evolved*
profiles — background plus the flux-surface-averaged perturbation — which is
what you want to trace out a transport response as the profiles relax, and it
is GENE-3D only because it needs the `Profiles` diagnostic.

This one divides by the profiles in ``profiles_<species>``, fixed for the whole
run. That is the right choice when the question is "what transport coefficient
does this flux correspond to at the gradient I asked for", when the run is too
short to have relaxed, or simply when the code did not write evolved profiles.
It also needs nothing but the flux cache and the profile file, so it works on
any global geometry.

Units of the profile file
-------------------------
`profiles_<species>` is read verbatim by
:func:`~genetools.io.profiles_loader.load_equilibrium_profiles`, which
normalises nothing, and the rest of the package treats its ``T`` and ``n`` as
**physical** — ``Fluxes2D.build_prefactors`` divides them by ``spec.temp*Tref``
and ``spec.dens*nref`` to get maps of order one. The same convention is used
here, so ``n_hat = n/nref`` and ``T_hat = T/Tref`` carry the species factor
already. Both are put in the dataset, and a warning fires if either strays far
from order one — a profile file in some other normalisation would otherwise
rescale every diffusivity silently.
"""

from __future__ import annotations

import warnings

import numpy as np
import matplotlib.pyplot as plt

from genetools._xr import make_dataset, unit_attrs
from genetools.diagnostics._base import RunDiagnostic
from genetools.diagnostics import _gene3d as g3
from genetools.diagnostics.fluxes2d import Fluxes2D

#: Warn when the normalised background strays this far from unity — the
#: signature of a profile file in an unexpected normalisation.
_SANE = (1e-2, 1e2)


class Diffusivity(RunDiagnostic):
    """
    Heat and particle diffusivity from the background profiles.

    Parameters
    ----------
    run : genetools.run.Run
    x_avg_lims : (float, float), optional
        Radial range in ``x/a`` for the single-number averages. Default: the
        whole domain minus *buffer_frac* at each end.
    buffer_frac : float
        Fraction trimmed from each radial end when *x_avg_lims* is unset
        (default 0.1 — the Krook buffers, where the flux is unphysical).
    """

    name = "diffusivity"
    supported = ("x_global", "y_global", "xy_global")

    def __init__(self, run, x_avg_lims=None, buffer_frac=0.1):
        super().__init__(run)
        self.x_avg_lims = x_avg_lims
        self.buffer_frac = buffer_frac
        self._fluxes = Fluxes2D(run)
        self._cache = {}

    # ------------------------------------------------------------------

    def _geom_factor(self):
        """
        ``<g^xx>`` on each flux surface — the metric factor in the denominator.

        Dispatches on rank, like `GeometryPlots`: a flux tube writes ``(nz,)``,
        an x-global run ``(nx, nz)`` and GENE-3D ``(nx, ny, nz)``. The x-global
        average must be **per flux surface** (``axis=-1``); normalising the
        Jacobian over the whole array instead makes every surface average about
        ``1/nx`` too small, x-dependently — the bug that hid in `shearingrate`.
        """
        geom = self.geom
        gxx, J = geom["metric"]["gxx"], geom["Jacobian"]
        gxx, J = np.asarray(gxx, dtype=float), np.asarray(J, dtype=float)
        if gxx.ndim == 3:
            return g3.flux_surface_average(gxx, J)
        if gxx.ndim == 2:
            return (gxx * J).sum(axis=-1) / J.sum(axis=-1)
        return float((gxx * J).sum() / J.sum())

    def _flux_profiles(self, t):
        """
        Time-averaged ``(Q_total, Gamma_total)`` per species, on the x grid.

        The two geometry paths of `Fluxes2D` differ in both naming and whether
        they keep a time axis, so they are reconciled here rather than in the
        caller: GENE-3D gives ``Q_total``/``Gamma_total`` over ``(species,
        time, x)``, the spectral cache gives ``Qes_x``/``Qem_x`` already
        averaged.
        """
        ds = self._fluxes.dataset(t)
        if not ds.data_vars:
            raise ValueError(
                "no flux data in the requested window; run fluxes2d first or "
                "widen t.")
        out = {}
        for name in ds["species"].values:
            name = str(name)
            if "Q_total" in ds:                       # GENE-3D
                q = ds["Q_total"].sel(species=name)
                g = ds["Gamma_total"].sel(species=name)
                if "time" in q.dims:
                    q, g = self._t_average(q), self._t_average(g)
                q, g = np.asarray(q), np.asarray(g)
            else:                                     # spectral cache
                q = sum(np.asarray(ds[k].sel(species=name))
                        for k in ("Qes_x", "Qem_x") if k in ds)
                g = sum(np.asarray(ds[k].sel(species=name))
                        for k in ("Ges_x", "Gem_x") if k in ds)
            out[name] = (np.asarray(q, dtype=float), np.asarray(g, dtype=float))
        return out, np.asarray(ds["x"], dtype=float)

    def _background(self, name):
        """
        ``(n_hat, T_hat, omt, omn)`` from the equilibrium profile file.

        ``n_hat``/``T_hat`` are in reference units and already carry the species
        factor, since the file is written per species — see the module
        docstring on the normalisation.
        """
        profiles = self.run.eq_profiles
        if not profiles or name not in profiles:
            raise KeyError(
                f"no equilibrium profile for {name!r}; this diagnostic divides "
                "by the background gradient, so profiles_<species> is required")
        ep = profiles[name]
        units = self.params.get("units", {}) or {}
        nref = float(units.get("nref", 1.0)) or 1.0
        Tref = float(units.get("Tref", 1.0)) or 1.0
        n_hat = np.asarray(ep["n"], dtype=float) / nref
        T_hat = np.asarray(ep["T"], dtype=float) / Tref
        for label, arr in (("n", n_hat), ("T", T_hat)):
            typical = float(np.nanmedian(np.abs(arr)))
            if typical and not (_SANE[0] < typical < _SANE[1]):
                warnings.warn(
                    f"{name}: background {label} normalises to {typical:.3g}, "
                    "far from order one — profiles_<species> is probably not "
                    "in the physical units the rest of the package assumes, "
                    "and every diffusivity here is rescaled by that factor.",
                    RuntimeWarning, stacklevel=3)
        return (n_hat, T_hat,
                np.asarray(ep["omt"], dtype=float),
                np.asarray(ep["omn"], dtype=float))

    def compute(self, t=None):
        """Radial ``chi`` and ``D`` profiles per species, plus their averages."""
        key = (self._key(t),
               tuple(self.x_avg_lims) if self.x_avg_lims else None,
               self.buffer_frac)
        if key in self._cache:
            return self._cache[key]

        fluxes, x = self._flux_profiles(t)
        x_o_a = np.asarray(self.coord["x_o_a"], dtype=float)
        geom_fac = np.asarray(self._geom_factor(), dtype=float)
        xsl = g3.radial_slice(x_o_a, limits=self.x_avg_lims,
                              buffer_frac=self.buffer_frac)

        per = {}
        for name, (Q, Gamma) in fluxes.items():
            n_hat, T_hat, omt, omn = self._background(name)
            if n_hat.size != Q.size:
                raise ValueError(
                    f"{name}: the profile file has {n_hat.size} radial points "
                    f"but the flux profile has {Q.size}")
            den_chi = geom_fac * n_hat * T_hat * omt
            den_d = geom_fac * n_hat * omn
            with np.errstate(divide="ignore", invalid="ignore"):
                chi = np.where(den_chi != 0, Q / den_chi, np.nan)
                D = np.where(den_d != 0, Gamma / den_d, np.nan)

            # Average numerator and denominator separately before dividing:
            # averaging the ratio lets one near-zero local gradient dominate.
            def ratio(num, den):
                num_avg, den_avg = num[xsl].mean(), den[xsl].mean()
                return float(num_avg / den_avg) if den_avg else np.nan

            per[name] = {"chi": chi, "D": D, "Q": Q, "Gamma": Gamma,
                         "n": n_hat, "T": T_hat, "omt": omt, "omn": omn,
                         "chi_avg": ratio(Q, den_chi),
                         "D_avg": ratio(Gamma, den_d)}

        result = {"species": per, "x": x, "x_o_a": x_o_a,
                  "geom_factor": geom_fac, "xslice": xsl}
        self._cache[key] = result
        return result

    # ------------------------------------------------------------------

    def _reference(self):
        """
        SI conversion for both diffusivities, in m^2/s.

        From ``Q = -n chi dT/dx``: ``Qgb * Lref / pref``, the usual
        ``c_ref rho_ref^2 / L_ref``. ``Gamma = -D dn/dx`` gives
        ``Ggb * Lref / nref``, which is the same number — so one reference
        converts both.
        """
        units = self.params.get("units", {}) or {}
        Qgb, Lref, pref = (units.get("Qgb"), units.get("Lref"),
                           units.get("pref"))
        if None in (Qgb, Lref, pref) or not pref:
            return None
        return float(Qgb) * float(Lref) / float(pref)

    def dataset(self, t=None):
        """``chi`` and ``D`` against ``x``, with the background they divide by."""
        raw = self.compute(t)
        names = list(raw["species"])
        stack = lambda key: np.stack([raw["species"][n][key] for n in names])
        data_vars = {k: (("species", "x"), stack(k))
                     for k in ("chi", "D", "Q", "Gamma", "n", "T", "omt", "omn")}
        data_vars["chi_avg"] = (("species",),
                                np.array([raw["species"][n]["chi_avg"]
                                          for n in names]))
        data_vars["D_avg"] = (("species",),
                              np.array([raw["species"][n]["D_avg"]
                                        for n in names]))
        ds = make_dataset(data_vars, {"x": raw["x_o_a"]}, species=names,
                          params=self.params)
        ds.attrs.update(unit_attrs(self.params))
        ds.attrs["geometry_kind"] = self.geometry_kind
        ds.attrs["gradient_source"] = "background (profiles_<species>)"
        ref = self._reference()
        if ref:
            ds.attrs["chi_ref_SI"] = ref
            ds["chi"].attrs["units"] = "chi_gB"
            ds["D"].attrs["units"] = "chi_gB"
        lo, hi = raw["x_o_a"][raw["xslice"]][[0, -1]]
        ds.attrs["x_avg_range"] = [float(lo), float(hi)]
        return ds

    def plot(self, t=None, si=False, trim=True, **kw):
        """
        ``chi`` and ``D`` against radius, with the gradients that set them.

        Two rows: the diffusivities on top, the background gradients that
        divide into them below — because a diffusivity spiking where ``omt``
        passes through zero is an artefact of the denominator, and the two
        panels have to be read together to see that.

        *trim* (default) draws only the retained radial window. The buffer
        regions are where the Krook operators hold the profiles, so the flux
        there is not transport and the diffusivity built from it is meaningless
        — and being the largest numbers on the axis, they set the y-scale and
        flatten everything that is physical. ``trim=False`` draws the whole
        domain, with the excluded ends shaded.
        """
        ds = self.dataset(t)
        x = np.asarray(ds["x"])
        lo_t, hi_t = ds.attrs.get("x_avg_range", (x[0], x[-1]))
        keep = ((x >= lo_t) & (x <= hi_t)) if trim else np.ones(x.size, bool)
        x = x[keep]
        ref = ds.attrs.get("chi_ref_SI") if si else None
        scale = ref if ref else 1.0
        unit = r"$\;[\mathrm{m^2/s}]$" if ref else r"$/\chi_{gB}$"
        if si and not ref:
            print("diffusivity: no SI reference available; plotting gyro-Bohm.")

        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True,
                                 squeeze=False)
        for name in ds["species"].values:
            name = str(name)
            pick = lambda v: np.asarray(ds[v].sel(species=name))[keep]
            axes[0][0].plot(x, pick("chi") * scale, label=name)
            axes[0][1].plot(x, pick("D") * scale, label=name)
            axes[1][0].plot(x, pick("omt"), label=name)
            axes[1][1].plot(x, pick("omn"), label=name)
        axes[0][0].set_ylabel(r"$\chi$" + unit)
        axes[0][1].set_ylabel(r"$D$" + unit)
        axes[1][0].set_ylabel(r"$\omega_T = -L_{ref}\,\partial_x \ln T$")
        axes[1][1].set_ylabel(r"$\omega_n = -L_{ref}\,\partial_x \ln n$")
        full = np.asarray(ds["x"])
        for row in axes:
            for ax in row:
                ax.set_xlabel(r"$x/a$")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                if not trim:
                    # Whole domain drawn: show which ends are excluded.
                    ax.axvspan(full[0], lo_t, color="0.9", zorder=0)
                    ax.axvspan(hi_t, full[-1], color="0.9", zorder=0)
        shown = "buffers excluded" if trim else "buffers shaded"
        fig.suptitle("Diffusivity from the background profiles "
                     f"($x/a \\in [{lo_t:.2f}, {hi_t:.2f}]$, {shown})")
        fig.tight_layout()
        plt.show()
        return [fig]
