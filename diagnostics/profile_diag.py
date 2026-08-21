# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
profile_diag.py — GENE ``profile_<species>`` radial profile diagnostic.

Reads the ASCII radial-profile output written by the profile diagnostic, which
exists for **global, nonlinear** runs. Each file is a gnuplot-block time series:
a header line, then per-output-time blocks of ``# <time> <block_nr>`` followed by
``nx`` rows, separated by two blank lines. A ragged ``#time averaged profiles``
trailing block (if present) is ignored — the time average is computed here
instead.

**The two codes write different files under the same name.** GENE's
``diag_df.F90`` writes thirteen columns; GENE-3D's ``diag_3d.F90`` writes eight
and stops after the turbulent fluxes. Reading a GENE-3D file with GENE's layout
would mis-slice every column past the sixth, so the column set is chosen from
the run's geometry:

    GENE      x/a, x/rho_ref, T, n, omt, omn, Gamma, Q, Pi,
              Gamma_neo, Q_neo, Pi_neo, j_boot
    GENE-3D   x/a, x/rho_ref, T, n, omt, omn, Gamma, Q

GENE-3D additionally writes ``flux_profile_<species>``, carrying the same fluxes
already time-averaged; :meth:`ProfileDiag.flux_profiles` reads it.

The diagnostic returns an ``xarray.Dataset`` with dims ``(species, time, x)``,
normalised variables plus SI-converted companions (``*_SI``) for the quantities
with a well-defined reference (T, n, and the gyro-Bohm fluxes). ``j_boot`` is
left normalised (its SI normalisation is not applied here).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

from genetools.diagnostics._base import RunDiagnostic

# Columns 0,1 are always the coordinates; the rest are the variables.
#: GENE (``diag_df.F90``) — thirteen columns.
_VAR_COLS_GENE = ["T", "n", "omt", "omn", "Gamma", "Q", "Pi",
                  "Gamma_neo", "Q_neo", "Pi_neo", "j_boot"]
#: GENE-3D (``diag_3d.F90``) — eight columns, no neoclassical terms.
_VAR_COLS_3D = ["T", "n", "omt", "omn", "Gamma", "Q"]


def _columns(is_3d: bool):
    """Return ``(variable names, expected column count)`` for the geometry."""
    names = _VAR_COLS_3D if is_3d else _VAR_COLS_GENE
    return names, 2 + len(names)

# Normalised-unit labels (for plot axes / attrs).
_NORM_UNITS = {
    "T": "T_ref", "n": "n_ref", "omt": "R/L_T", "omn": "R/L_n",
    "Gamma": "Gamma_gb", "Q": "Q_gb", "Pi": "Pi_gb",
    "Gamma_neo": "Gamma_gb", "Q_neo": "Q_gb", "Pi_neo": "Pi_gb",
    "j_boot": "normalised",
}

# SI conversion: variable -> (reference key in params['units'], SI unit label).
_SI = {
    "T": ("Tref", "keV"),
    "n": ("nref", "1e19 m^-3"),
    "Gamma": ("Ggb", "m^-2 s^-1"),
    "Q": ("Qgb", "W m^-2"),
    "Pi": ("Pgb", "N m^-1"),
    "Gamma_neo": ("Ggb", "m^-2 s^-1"),
    "Q_neo": ("Qgb", "W m^-2"),
    "Pi_neo": ("Pgb", "N m^-1"),
}


def _parse_profile_file(path: str, ncol: int):
    """
    Parse one ``profile_<species>`` file.

    Parameters
    ----------
    path : str
    ncol : int
        Expected column count — 13 for GENE, 8 for GENE-3D.

    Returns
    -------
    times : np.ndarray, shape (n_times,)
    data  : np.ndarray, shape (n_times, nx, ncol)
    """
    times, blocks, current = [], [], None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if "time averaged" in s.lower():
                    break  # ragged trailing block — stop here
                toks = s.lstrip("#").split()
                try:
                    t = float(toks[0])           # "# <time> <block_nr>"
                except (ValueError, IndexError):
                    continue                     # column-name header line
                if current is not None:
                    blocks.append(current)
                times.append(t)
                current = []
            elif current is not None:
                current.append([float(v) for v in s.split()])
    if current:
        blocks.append(current)
    if not blocks:
        raise ValueError(f"No data blocks parsed from {path}")

    nx = len(blocks[0])
    if any(len(b) != nx for b in blocks):
        raise ValueError(f"Inconsistent radial grid size across blocks in {path}")
    data = np.asarray(blocks, dtype=float)       # (n_times, nx, ncol)
    if data.shape[2] < ncol:
        raise ValueError(
            f"{path}: expected {ncol} columns, found {data.shape[2]}. GENE "
            "writes 13 and GENE-3D 8; check that the run's geometry is being "
            "detected correctly.")
    return np.asarray(times, dtype=float), data


def _dedup_later_wins(times: np.ndarray, data: np.ndarray):
    """Sort by time and drop duplicate times, keeping the later occurrence."""
    order = np.argsort(times, kind="stable")
    times, data = times[order], data[order]
    rev = np.arange(len(times) - 1, -1, -1)
    _, first = np.unique(np.round(times[rev], 9), return_index=True)
    keep = np.sort(rev[first])
    return times[keep], data[keep]


class ProfileDiag(RunDiagnostic):
    """``profile_<species>`` radial profile diagnostic (global nonlinear runs)."""

    name = "profile_diag"

    @property
    def var_cols(self):
        """Variable names in this run's ``profile_<species>`` files."""
        return _columns(self.is_3d)[0]

    # ------------------------------------------------------------------

    def _load_species(self, species, prefix="profile", ncol=None):
        """Return (times, data) for one species, merged across all segments."""
        if ncol is None:
            ncol = _columns(self.is_3d)[1]
        all_t, all_d = [], []
        for ext in self.run.extensions:
            path = f"{self.run._folder}{prefix}_{species}{ext}"
            if os.path.exists(path):
                t, d = _parse_profile_file(path, ncol)
                all_t.append(t)
                all_d.append(d)
        if not all_t:
            raise FileNotFoundError(
                f"No '{prefix}_{species}<ext>' files in {self.run._folder} — "
                "these are written only for global nonlinear runs "
                "(istep_prof > 0).")
        times = np.concatenate(all_t)
        data = np.concatenate(all_d, axis=0)
        return _dedup_later_wins(times, data)

    def compute(self, t=None):
        """
        Parse all species; cache aligned (species, time, x) arrays.

        *t* is accepted for a uniform facade and ignored — the whole file is
        parsed and the window is applied when averaging or plotting.
        """
        if self._cache:
            return self._cache
        species = list(self.run.species)
        per = {sp: self._load_species(sp) for sp in species}

        t0, d0 = per[species[0]]
        nt, nx = d0.shape[0], d0.shape[1]
        for sp in species[1:]:
            ts, ds = per[sp]
            if ds.shape[0] != nt or ds.shape[1] != nx:
                raise ValueError(
                    "profile_ files have mismatched time/x grids across species "
                    f"('{species[0]}': {(nt, nx)} vs '{sp}': {ds.shape[:2]}). "
                    "Scope to a consistent subset with Run(path, ext=[...]).")

        variables = {}
        for vi, name in enumerate(self.var_cols):
            col = 2 + vi
            variables[name] = np.stack([per[sp][1][:, :, col] for sp in species],
                                       axis=0)             # (species, time, x)

        self._cache = {
            "species": species,
            "time": t0,
            "x_a": d0[0, :, 0],
            "x_rho_ref": d0[0, :, 1],
            "vars": variables,
        }
        return self._cache

    # ------------------------------------------------------------------

    def dataset(self, t=None):
        """Return an ``xarray.Dataset`` (dims species, time, x) of all quantities."""
        import xarray as xr

        c = self.compute()
        units = self.run.params.get(0).get("units", {}) or {}

        ds = xr.Dataset(coords={"species": c["species"], "time": c["time"],
                                "x": c["x_a"]})
        ds = ds.assign_coords(x_rho_ref=("x", c["x_rho_ref"]))
        for name, arr in c["vars"].items():
            ds[name] = (("species", "time", "x"), arr)
            ds[name].attrs["units"] = _NORM_UNITS.get(name, "")
            if name in _SI:
                ref_key, si_units = _SI[name]
                ref = units.get(ref_key)
                if ref is not None and np.isscalar(ref):
                    ds[f"{name}_SI"] = (("species", "time", "x"), arr * float(ref))
                    ds[f"{name}_SI"].attrs["units"] = si_units
        for k in ("Tref", "nref", "Qgb", "Ggb", "Pgb"):
            if k in units and np.isscalar(units[k]):
                ds.attrs[k] = float(units[k])
        ds.attrs["geometry_kind"] = self.geometry_kind
        ds.attrs["n_columns"] = _columns(self.is_3d)[1]
        return ds

    def flux_profiles(self):
        """
        Read GENE-3D's ``flux_profile_<species>`` — its own averaged fluxes.

        GENE-3D writes these already time-averaged and converted to SI, so they
        are an independent check on the conversion applied here. Returns
        ``{species: {'x_o_a', 'Gamma', 'Q'}}``, or ``{}`` when absent.
        """
        out = {}
        for sp in self.run.species:
            try:
                _, data = self._load_species(sp, prefix="flux_profile", ncol=3)
            except FileNotFoundError:
                continue
            out[sp] = {"x_o_a": data[-1, :, 0], "Gamma": data[-1, :, 1],
                       "Q": data[-1, :, 2]}
        return out

    def _averaged(self, t=None):
        """Return the time-averaged Dataset (dims species, x) over window *t*."""
        ds = self.dataset()
        if t is not None:
            tt = ds["time"].values
            a, b = (t if isinstance(t, (tuple, list)) else (t, t))
            mask = np.ones(tt.shape, bool)
            if a is not None:
                mask &= tt >= a
            if b is not None:
                mask &= tt <= b
            ds = ds.isel(time=np.where(mask)[0])
        tt = ds["time"].values
        if tt.size <= 1:
            avg = ds.isel(time=0)
        else:
            span = float(tt[-1] - tt[0])
            avg = ds.integrate("time") / span if span > 0 else ds.isel(time=0)
        return avg, ds

    def _si_available(self) -> bool:
        """True if the parameter file provides real reference units (not the
        all-1.0 defaults), so an SI conversion is physically meaningful."""
        units = self.run.params.get(0).get("units", {}) or {}
        return any(float(units.get(k, 1.0)) != 1.0
                   for k in ("Lref", "Bref", "Tref", "nref", "mref"))

    def plot(self, t=None, si=None, **kw):
        """
        Plot time-averaged radial profiles (8 panels), species overlaid.

        Always plots gyro-Bohm (normalised) units, and *additionally* plots the
        SI version when the parameter file provides reference units. Override
        with ``si=False`` (GB only) or ``si=True`` (SI only).

        Returns a single figure, or a list ``[gb_fig, si_fig]`` when both are
        drawn.
        """
        avg, ds = self._averaged(t)
        si_ok = self._si_available()

        show_gb = si in (None, False)
        show_si = si is True or (si is None and si_ok)
        if si is True and not si_ok:
            warnings.warn("profile_diag: SI requested but the parameter file has "
                          "no reference units; plotting gyro-Bohm instead.")
            show_gb, show_si = True, False

        figs = []
        if show_gb:
            figs.append(self._plot_panels(avg, ds, si=False))
        if show_si:
            figs.append(self._plot_panels(avg, ds, si=True))
        plt.show()
        return figs[0] if len(figs) == 1 else figs

    def _plot_panels(self, avg, ds, si: bool):
        """
        Draw one figure in gyro-Bohm (si=False) or SI (si=True).

        Only panels the run actually has are drawn: a GENE-3D file carries
        neither the momentum flux nor the neoclassical terms, so those axes are
        dropped rather than left blank.
        """
        all_panels = [("T", r"$T$"), ("n", r"$n$"),
                      ("omt", r"$R/L_T$"), ("omn", r"$R/L_n$"),
                      ("Gamma", r"$\Gamma$"), ("Q", r"$Q$"),
                      ("Pi", r"$\Pi$"), ("j_boot", r"$j_{\rm boot}$")]
        panels = [(k, ttl) for k, ttl in all_panels if k in avg]
        neo_of = {"Gamma": "Gamma_neo", "Q": "Q_neo", "Pi": "Pi_neo"}
        x = ds["x"].values

        def pick(key):
            return f"{key}_SI" if (si and f"{key}_SI" in avg) else key

        ncol = 4 if len(panels) > 4 else max(len(panels), 1)
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.5 * nrow),
                                 sharex=True, squeeze=False)
        for ax, (key, title) in zip(axes.flat, panels):
            k = pick(key)
            for sp in ds["species"].values:
                ax.plot(x, avg[k].sel(species=sp).values, label=str(sp))
                neo = neo_of.get(key)
                if neo and neo in avg:
                    kn = pick(neo)
                    ax.plot(x, avg[kn].sel(species=sp).values, ls="--",
                            label=f"{sp} (neo)")
            ax.set_title(title)
            ax.set_ylabel(ds[k].attrs.get("units", ""))
            ax.grid(True)
        for ax in axes.flat[len(panels):]:
            ax.set_visible(False)
        for ax in axes[-1, :]:
            ax.set_xlabel(r"$x/a$")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("Radial profiles (profile_<species>)"
                     + (" [SI]" if si else " [gyro-Bohm]"))
        fig.tight_layout()
        return fig
