# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
contours.py — 2-D cuts and 1-D line profiles of GENE field and moment data.

One class, one interface, every geometry. A snapshot has six projections — the
three 2-D planes (``xy``, ``xz``, ``yz``) and the three 1-D lines (``x``, ``y``,
``z``) — and ``reductions`` picks them; the default is the two cuts almost
always wanted, ``xy`` and ``xz``.

The same call means the same thing whatever the run is::

    run.contours.plot(quantities=("phi",), reductions="all")

What differs between the codes is only where the data starts. GENE-3D stores x
and y in real space; a flux tube stores both spectrally; an x-global run stores
x real and y spectral. ``x_fourier`` and ``y_fourier`` name the view you want,
not the transform needed to get there, so ``x_fourier=False`` means "show me
real x" and this module inverse-transforms a flux tube to honour it, exactly as
``x_fourier=True`` forward-transforms GENE-3D. Both default to ``False``: a
contour plot means real space.

Cut, not average
----------------
Every coordinate a reduction drops is held at the grid point nearest zero unless
you pass ``ylim``/``zlim``, in which case it is averaged over that range. So
``xy`` is the plane at ``z = 0`` (the outboard midplane on a standard grid) and
``xz`` the plane at ``y = 0``. A bare average over the whole of ``z`` would
smear the outboard and inboard sides of a global run together, which is why it
is not the default; ``zlim=(z[0], z[-1])`` asks for it explicitly.

``x`` is the exception: it defaults to the *whole* range, because it is the axis
the common planes keep, and a global run's radial grid never passes through
zero, so "nearest zero" would silently mean "the innermost surface". Pass
``xlim`` to restrict it.

Limits are read on the axis you are looking at: with ``y_fourier=True``,
``ylim`` is a ky range, and the default cut lands on ky = 0.

Averages are plain means, not Jacobian-weighted: a cut is a picture of the field
on the grid, and weighting it by the volume element would show the metric as
much as the turbulence. The flux and profile diagnostics, which do need the
volume element, weight explicitly.

Axis conventions
----------------
A real-space axis reconstructed from modes comes out of the transform starting
at zero, so this module uses the 0-based box grid for it (``y`` everywhere, and
``x`` for a flux tube), which is also the grid GENE-3D writes. The genuinely
radial geometries keep ``x/a``. Fourier axes are ``fftshift``ed for display when
they are stored in FFT order, and the data is shifted with them.

The real-space y grid has ``2 * nky0`` points: GENE's ``discretization.F90``
sets ``ly0da = 2 * nky0`` (and ``3 * nky0`` with the 3/2 dealiasing rule), so
``ny0 = 2 * nky0`` and the Nyquist bin GENE does not store is reconstructed as
zero, which is what ``np.fft.irfft`` does when handed the shorter input.

Memory
------
One snapshot at a time is read, transformed and reduced; only the reductions are
kept, so the time axis costs nothing. A transformed snapshot is the same size as
the stored one it came from — a flux tube's ``(nx, nky, nz)`` complex64 array
and its ``(nx, 2*nky, nz)`` real counterpart hold the same number of bytes.
"""

from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from genetools._xr import make_dataset, unit_attrs
from genetools.diagnostics._base import RunDiagnostic
from genetools.diagnostics import _gene3d as c

#: 2-D planes -> the array axis averaged (or cut) away.
_PLANES = {"xy": 2, "xz": 1, "yz": 0}
#: 1-D lines -> the two array axes averaged (or cut) away.
_LINES = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
#: Every reduction, planes first.
_ALL = tuple(_PLANES) + tuple(_LINES)
#: The two cuts drawn unless asked otherwise.
_DEFAULT_REDUCTIONS = ("xy", "xz")

_AXIS_LABELS = {
    "x": r"$x/a$", "x_box": r"$x/\rho_{\rm ref}$",
    "y": r"$y/\rho_{\rm ref}$", "z": r"$z/\pi$",
    "kx": r"$k_x \rho_{\rm ref}$", "ky": r"$k_y \rho_{\rm ref}$",
}

#: Options removed when the two geometry paths were merged, and what replaced
#: them. Named explicitly so a script written against the old spectral-only
#: interface fails with the new spelling rather than a bare TypeError.
_RETIRED = {
    "field": "quantities=('phi',) — variables are named, not indexed",
    "ifft": "x_fourier / y_fourier — name the view, not the transform",
    "iz": "zlim=(z0, z0) — cut at a coordinate, not an index",
    "iy": "ylim=(y0, y0) — cut at a coordinate, not an index",
    "show_xz": "reductions=('xy', 'xz')",
    "max_loads": "n_max",
    "del_zonal": "no replacement; filter the dataset from .dataset() instead",
    "zero_range": "no replacement; filter the dataset from .dataset() instead",
}

#: The reduction options, frozen so it can key the snapshot cache.
_Selection = namedtuple(
    "_Selection",
    "quantities species x_fourier y_fourier square xlim ylim zlim t_avg")


def _inverse_x(arr, axis=0):
    """Inverse transform a spectral radial axis, with GENE's normalisation."""
    return np.fft.ifft(arr, axis=axis) * arr.shape[axis]


def _inverse_y(arr, axis=1):
    """
    Inverse transform a one-sided ky axis to real y, GENE's normalisation.

    The output has ``2 * nky`` points: GENE stores ``nky0 = ny0 / 2`` modes and
    leaves the Nyquist bin out, which ``irfft`` supplies as zero.
    """
    ny = 2 * arr.shape[axis]
    return np.fft.irfft(arr, n=ny, axis=axis) * ny


class Contours(RunDiagnostic):
    """
    2-D cuts and 1-D line profiles of GENE field and moment data.

    Selection happens at call time, not construction, so ``run.contours`` can
    stay a plain attribute: every option below is a keyword to :meth:`compute`,
    :meth:`dataset` or :meth:`plot`, and each call is independent of the last.

    Parameters
    ----------
    run : genetools.run.Run, optional
        ``None`` gives a detached instance exposing only the pure helpers.
    cmap : str, optional
        Matplotlib colormap (default 'bwr').

    Call options
    ------------
    quantities : sequence of str
        Variable names from the field or moment file. Default ``('phi',)``.
    species : str
        Species whose moment file supplies any moment quantities. Defaults to
        the first species.
    reductions : sequence of str or 'all'
        Any of ``xy``, ``xz``, ``yz``, ``x``, ``y``, ``z``. Default
        ``('xy', 'xz')``.
    x_fourier, y_fourier : bool
        View that direction in Fourier space. Both default to ``False``, i.e.
        real space, whatever the run stores.
    square : bool
        Reduce ``|f|^2`` rather than ``f``.
    xlim, ylim, zlim : (float, float)
        Range of the dropped coordinate to average over, on the displayed axis.
        ``y`` and ``z`` default to a cut at the grid point nearest zero; ``x``
        defaults to the whole range.
    t_avg : bool
        Average over time instead of keeping the time axis.
    n_max : int
        :meth:`plot` only — at most this many times are drawn, evenly spaced
        through the window, so a long run does not open hundreds of axes.
    """

    name = "contours"

    def __init__(self, run=None, cmap="bwr"):
        super().__init__(run)
        self.cmap = cmap

    # ------------------------------------------------------------------
    # Option handling
    # ------------------------------------------------------------------

    @staticmethod
    def _at(values, target=0.0):
        """
        Return equal bounds on the grid point of *values* nearest *target*.

        Equal bounds make :func:`~genetools.diagnostics._gene3d.index_window`
        pick exactly that index, so the reduction is a cut and not an average.
        """
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return None
        v = float(arr[int(np.argmin(np.abs(arr - target)))])
        return (v, v)

    def _select(self, kw):
        """
        Build the frozen selection for *kw*.

        Built from the defaults and *this* call's arguments only. Carrying the
        previous call's arguments forward would make an earlier ``zlim`` leak
        into a later plot that never asked for it.
        """
        retired = sorted(set(kw) & set(_RETIRED))
        if retired:
            raise TypeError(
                "contour option(s) " + ", ".join(repr(r) for r in retired)
                + " were removed when the geometry paths were unified; use "
                + "; ".join(f"{r} -> {_RETIRED[r]}" for r in retired))
        unknown = set(kw) - set(_Selection._fields)
        if unknown:
            raise TypeError(
                f"unknown contour option(s) {sorted(unknown)}; "
                f"expected any of {list(_Selection._fields)}")
        return _Selection(
            quantities=tuple(kw.get("quantities", ("phi",))),
            species=kw.get("species") or self._default_species(),
            x_fourier=bool(kw.get("x_fourier", False)),
            y_fourier=bool(kw.get("y_fourier", False)),
            square=bool(kw.get("square", False)),
            xlim=self._as_limits(kw.get("xlim")),
            ylim=self._as_limits(kw.get("ylim")),
            zlim=self._as_limits(kw.get("zlim")),
            t_avg=bool(kw.get("t_avg", False)),
        )

    def _default_species(self):
        """
        The species whose moment file supplies moment quantities.

        The first one, or ``None`` on a detached instance — which is what makes
        the option handling usable without a run, as the class docstring
        promises of the pure helpers.
        """
        run = getattr(self, "run", None)
        names = getattr(run, "species", None) if run is not None else None
        return names[0] if names else None

    @staticmethod
    def _as_limits(value):
        """Normalise a limit pair to a hashable tuple, or ``None``."""
        return None if value is None else (float(value[0]), float(value[1]))

    @staticmethod
    def _reductions(value):
        """Normalise the ``reductions`` argument to a validated tuple."""
        if value is None:
            return ()
        if isinstance(value, str):
            if value == "all":
                return _ALL
            value = (value,)
        out = tuple(value)
        bad = [r for r in out if r not in _ALL]
        if bad:
            raise ValueError(
                f"unknown reduction(s) {bad}; expected any of {list(_ALL)}")
        return out

    def _split_kw(self, kw):
        """Split the call keywords into ``(reductions, selection)``."""
        kw = dict(kw)
        reductions = self._reductions(
            kw.pop("reductions", _DEFAULT_REDUCTIONS))
        return reductions, self._select(kw)

    @staticmethod
    def _x_axis(sel):
        return "kx" if sel.x_fourier else "x"

    @staticmethod
    def _y_axis(sel):
        return "ky" if sel.y_fourier else "y"

    # ------------------------------------------------------------------
    # Where the data starts, and how to get it onto the requested axes
    # ------------------------------------------------------------------

    @property
    def _x_is_spectral(self) -> bool:
        """Whether the run stores its radial axis as kx (flux tube only)."""
        return self.geometry_kind == "flux_tube"

    @property
    def _y_is_spectral(self) -> bool:
        """Whether the run stores its binormal axis as ky (everything but 3-D)."""
        return not self.is_3d

    @staticmethod
    def _fft_ordered(values) -> bool:
        """
        Whether *values* are in numpy's ``fftfreq`` order — 0, positive, negative.

        A one-sided ky axis is already ascending and must not be shifted; a
        signed kx or a GENE-3D ky is stored wrapped and must be.
        """
        v = np.asarray(values, dtype=float)
        return bool(v.size > 2 and v[0] == 0.0 and v[1] > 0.0 and v.min() < 0.0)

    def _box_grid(self, k_values, n):
        """
        The 0-based real-space grid dual to a spectral axis of *n* points.

        ``L = 2 pi / k_min`` from the smallest positive wavenumber. This is the
        grid the inverse transform actually produces — it starts at zero, not at
        ``-L/2`` — and it is also the convention GENE-3D writes.
        """
        k = np.asarray(k_values, dtype=float)
        positive = k[k > 0]
        if positive.size == 0 or n <= 0:
            return np.arange(max(n, 0), dtype=float)
        L = 2.0 * np.pi / float(positive.min())
        return np.arange(n, dtype=float) * (L / n)

    def _axes(self, sel, coord, shape):
        """
        Displayed axis values and their labels, plus which need an fftshift.

        Returns ``(values, labels, shift)``, each keyed ``'x'``, ``'y'``, ``'z'``.
        *shape* is the shape the snapshot will have once transformed, which is
        what fixes the length of a reconstructed real-space axis.
        """
        nx, ny, nz = shape
        values, labels, shift = {}, {}, {}

        if sel.x_fourier:
            kx = np.asarray(coord["kx"], dtype=float)[:nx]
            shift["x"] = self._fft_ordered(kx)
            values["x"] = np.fft.fftshift(kx) if shift["x"] else kx
            labels["x"] = "kx"
        elif self._x_is_spectral:
            # Reconstructed from kx: a box coordinate, not a radial one.
            values["x"] = self._box_grid(coord["kx"], nx)
            labels["x"], shift["x"] = "x_box", False
        else:
            values["x"] = np.asarray(coord["x_o_a"], dtype=float)[:nx]
            labels["x"], shift["x"] = "x", False

        if sel.y_fourier:
            ky = np.asarray(coord["ky"], dtype=float)[:ny]
            shift["y"] = self._fft_ordered(ky)
            values["y"] = np.fft.fftshift(ky) if shift["y"] else ky
            labels["y"] = "ky"
        elif self._y_is_spectral:
            values["y"] = self._box_grid(coord["ky"], ny)
            labels["y"], shift["y"] = "y", False
        else:
            values["y"] = np.asarray(coord["y"], dtype=float)[:ny]
            labels["y"], shift["y"] = "y", False

        values["z"] = np.asarray(coord["z"], dtype=float)[:nz]
        labels["z"], shift["z"] = "z", False
        return values, labels, shift

    def _to_axes(self, sel, arr, shift):
        """
        Bring one stored snapshot onto the axes *sel* asks for.

        The order of the y inverse is not free. GENE stores only ``ky >= 0``
        because the *real-space* field is real, and that gives
        ``f(-kx, -ky) = conj(f(kx, ky))`` — a relation linking the two
        wavenumbers. At fixed ``kx`` the ky axis is therefore **not** a
        Hermitian one-sided spectrum, and an ``irfft`` along it is meaningless.
        Only once x is back in real space does ``f(x, -ky) = conj(f(x, ky))``
        hold and the real transform apply. So a flux tube is inverted in x
        first, and a view that wants kx *and* real y goes back out to kx
        afterwards rather than taking a shortcut that would be wrong.

        A view that matches what the run already stores costs nothing: no
        round trip is made just to return to the representation on disk.

        The magnitude is taken once at the end rather than after each axis, so
        ``x_fourier`` and ``y_fourier`` together mean ``|FFT_xy(f)|`` and not
        ``FFT_y(|FFT_x(f)|)``.
        """
        out = arr
        if self._y_is_spectral and not sel.y_fourier:
            # Real y wanted: x must be real for the Hermitian inverse to hold.
            if self._x_is_spectral:
                out = _inverse_x(out)
            out = _inverse_y(out)
            if sel.x_fourier:
                out = c.to_kx(out)
        else:
            if self._x_is_spectral and not sel.x_fourier:
                out = _inverse_x(out)
            elif not self._x_is_spectral and sel.x_fourier:
                out = c.to_kx(out)
            if not self._y_is_spectral and sel.y_fourier:
                out = c.to_ky(out)

        if sel.x_fourier or sel.y_fourier:
            out = np.abs(out)
        elif np.iscomplexobj(out):
            out = out.real
        if sel.square:
            out = np.abs(out) ** 2

        for axis, name in ((0, "x"), (1, "y")):
            if shift.get(name):
                out = np.fft.fftshift(out, axes=axis)
        return out

    def _transformed_shape(self, sel, shape):
        """Shape a stored snapshot will have once brought onto the axes."""
        nx, nj, nz = shape
        ny = 2 * nj if (self._y_is_spectral and not sel.y_fourier) else nj
        return nx, ny, nz

    # ------------------------------------------------------------------
    # The reduction engine
    # ------------------------------------------------------------------

    def _limits(self, sel, values):
        """
        The bounds each coordinate is reduced over, resolved on the shown axes.

        ``y`` and ``z`` fall back to a cut at the grid point nearest zero; ``x``
        falls back to its whole range. See the module docstring for why x is
        the exception.
        """
        out = {}
        for name in "xyz":
            given = getattr(sel, f"{name}lim")
            if given is not None:
                out[name] = given
            elif name == "x":
                out[name] = None
            else:
                out[name] = self._at(values[name])
        return out

    @staticmethod
    def _windows(limits, values, shape):
        """Index slices for the three coordinates, from the resolved limits."""
        return tuple(c.index_window(values[name], limits[name], n)
                     for name, n in zip("xyz", shape))

    def compute(self, t=None, **kw):
        """
        Stream the requested variables and build every reduction.

        All six reductions are built — they are cheap means over an array
        already in memory — so ``reductions`` only selects what :meth:`dataset`
        returns and one streaming pass serves any choice of them.

        Works for every geometry: the snapshot is brought onto the requested
        axes first (:meth:`_to_axes`), and everything after that is the same
        slice-and-mean whatever the run stores.
        """
        _, sel = self._split_kw(kw)
        key = (self._key(t), sel)
        if key in self._cache:
            return self._cache[key]

        coord = self.coord
        sources = self._sources(sel.quantities, sel.species)
        if len(sources) > 1:
            # More than one file per time step: only the times all of them have.
            times, index_of = self._common_indices([r for r, _ in sources], t)
            per_reader = {id(r): index_of[id(r)] for r, _ in sources}
        else:
            reader = sources[0][0]
            all_times, idx = self._indices(reader, t)
            times = np.asarray(all_times)[idx]
            per_reader = {id(reader): idx}

        acc, axes = {}, None
        for reader, names in sources:
            slots = {n: reader.index_of(n) for n in names}
            for _, arrays in reader.stream_selected(per_reader[id(reader)],
                                                    variables=names):
                for n in names:
                    stored = arrays[slots[n]]
                    if axes is None:
                        shape = self._transformed_shape(sel, stored.shape)
                        values, labels, shift = self._axes(sel, coord, shape)
                        limits = self._limits(sel, values)
                        axes = (values, labels, shift, limits)
                    values, labels, shift, limits = axes
                    var = self._to_axes(sel, stored, shift)
                    xsl, ysl, zsl = self._windows(limits, values, var.shape)
                    store = acc.setdefault(n, {})
                    for plane, axis in _PLANES.items():
                        sub = _apply(var, xsl, ysl, zsl, keep=plane)
                        store.setdefault(plane, []).append(sub.mean(axis=axis))
                    for line, drop in _LINES.items():
                        sub = _apply(var, xsl, ysl, zsl, keep=line)
                        store.setdefault(line, []).append(sub.mean(axis=drop))

        if axes is None:
            raise ValueError("No snapshot found in the requested time window.")
        values, labels, shift, limits = axes

        times = np.asarray(times, dtype=float)
        reduced = {}
        for name, store in acc.items():
            reduced[name] = {}
            for red, stack in store.items():
                arr = np.asarray(stack)
                reduced[name][red] = (self._time_average(arr, times)
                                      if sel.t_avg else arr)

        result = {"reduced": reduced, "times": times, "coord": coord,
                  "selection": sel, "values": values, "labels": labels,
                  "limits": limits}
        self._cache[key] = result
        return result

    def dataset(self, t=None, **kw):
        """
        The requested reductions as an :class:`xarray.Dataset`, any geometry.

        Dimension names follow the view: the radial axis is ``x`` or ``kx`` and
        the binormal one ``y`` or ``ky``, so a script can tell from the dataset
        what it is looking at without knowing the run.
        """
        reductions, sel = self._split_kw(kw)
        raw = self.compute(t, **kw)
        values, labels = raw["values"], raw["labels"]
        rename = {"x": self._x_axis(sel), "y": self._y_axis(sel), "z": "z"}

        data_vars, candidates = {}, {}
        for name, store in raw["reduced"].items():
            for red in reductions:
                if red not in store:
                    continue
                dims = tuple(rename[ch] for ch in red)
                if not sel.t_avg:
                    dims = ("time",) + dims
                data_vars[f"{name}_{red}"] = (dims, np.asarray(store[red]))
        for ch, axis_name in rename.items():
            candidates[axis_name] = values[ch]
        if not sel.t_avg:
            candidates["time"] = raw["times"]

        params = self.params
        ds = make_dataset(data_vars, candidates, params=params)
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = self.geometry_kind
        ds.attrs["x_fourier"] = int(sel.x_fourier)
        ds.attrs["y_fourier"] = int(sel.y_fourier)
        ds.attrs["squared"] = int(sel.square)
        # The label carries what the dimension name cannot: whether a real x is
        # a radial coordinate (x/a) or a flux tube's box coordinate.
        for ch, axis_name in rename.items():
            ds.attrs[f"{axis_name}_label"] = labels[ch]
        if sel.species:
            ds.attrs["species"] = sel.species
        return ds

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, t=None, n_max=4, **kw):
        """
        Plot the requested reductions, one row each and one column per time.

        ``x`` runs horizontally in every reduction that has it, so the rows
        share a radial axis and read together.
        """
        reductions, sel = self._split_kw(kw)
        raw = self.compute(t, **kw)
        ds = self.dataset(t, **kw)
        quantities = [q for q in sel.quantities
                      if any(f"{q}_{r}" in ds for r in reductions)]
        if not quantities:
            raise ValueError("No reduction available to plot.")

        fixed = self._fixed_values(raw["limits"])
        labels = {self._x_axis(sel): raw["labels"]["x"],
                  self._y_axis(sel): raw["labels"]["y"], "z": "z"}
        figs = []
        for name in quantities:
            reds = [r for r in reductions if f"{name}_{r}" in ds]
            frames, n_total = self._pick_times(ds[f"{name}_{reds[0]}"], n_max)
            if n_total > len(frames):
                print(f"contours: showing {len(frames)} of {n_total} times.")

            fig, axes = plt.subplots(len(reds), len(frames),
                                     figsize=(4.4 * len(frames),
                                              3.5 * len(reds)),
                                     squeeze=False)
            for row, red in enumerate(reds):
                da = ds[f"{name}_{red}"]
                # One symmetric scale per row, so the time panels are comparable
                # instead of each being self-normalised.
                vmax = float(np.max(np.abs(np.asarray(da)))) or 1.0
                for col, (time, index) in enumerate(frames):
                    ax = axes[row][col]
                    frame = da if index is None else da.isel(time=index)
                    if len(red) == 2:
                        self._draw_plane(fig, ax, ds, frame, vmax, labels)
                    else:
                        self._draw_line(ax, ds, frame, vmax, labels)
                    ax.set_title(self._panel_title(red, fixed, time),
                                 fontsize=9)
            fig.suptitle(self._title(name, sel))
            fig.tight_layout()
            figs.append(fig)
        plt.show()
        return figs

    def _draw_plane(self, fig, ax, ds, frame, vmax, labels):
        """pcolormesh of a 2-D reduction, with the radial axis horizontal."""
        dims = frame.dims
        # dims[0] is the radial axis: put it horizontal.
        h, v = np.asarray(ds[dims[0]]), np.asarray(ds[dims[1]])
        mesh = ax.pcolormesh(h, v, np.asarray(frame).T, shading="nearest",
                             cmap=self.cmap, vmin=-vmax, vmax=vmax)
        ax.set_xlabel(_AXIS_LABELS.get(labels.get(dims[0], dims[0]), dims[0]))
        ax.set_ylabel(_AXIS_LABELS.get(labels.get(dims[1], dims[1]), dims[1]))
        fig.colorbar(mesh, ax=ax)

    @staticmethod
    def _draw_line(ax, ds, frame, vmax, labels):
        """Line plot of a 1-D reduction, on the scale shared across the row."""
        dim = frame.dims[0]
        ax.plot(np.asarray(ds[dim]), np.asarray(frame))
        ax.set_xlabel(_AXIS_LABELS.get(labels.get(dim, dim), dim))
        ax.set_ylim(-vmax, vmax)
        ax.grid(True, alpha=0.3)

    def _title(self, name, sel):
        """Figure title: the quantity, its species and whether it is squared."""
        moment = name not in self.run.field.var_names
        base = f"{name}" + (f" ({sel.species})" if sel.species and moment
                            else "")
        return f"|{base}|²" if sel.square else base

    @staticmethod
    def _panel_title(red, fixed, time):
        """Panel title: the reduction, any coordinate it holds fixed, the time."""
        bits = [red]
        for held in "xyz":
            if held not in red and held in fixed:
                bits.append(f"{held}={fixed[held]:.3g}")
        if time is not None:
            bits.append(f"t={time:.4g}")
        return "  ".join(bits)

    @staticmethod
    def _fixed_values(limits):
        """The coordinates this selection cuts at, for the panel titles."""
        out = {}
        for held in "xyz":
            bounds = limits.get(held)
            if bounds and bounds[0] == bounds[1]:
                out[held] = float(bounds[0])
        return out

    @staticmethod
    def _pick_times(da, n_max):
        """Return ``([(time, index), ...], n_total)`` evenly spaced in time."""
        if "time" not in da.dims:
            return [(None, None)], 1
        n_total = da.sizes["time"]
        picks = np.unique(np.linspace(0, n_total - 1,
                                      min(n_max, n_total)).astype(int))
        return [(float(da["time"].values[i]), int(i)) for i in picks], n_total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply(var, xsl, ysl, zsl, keep):
    """Apply the index windows, leaving the axes named in *keep* unrestricted."""
    sx = slice(None) if "x" in keep else xsl
    sy = slice(None) if "y" in keep else ysl
    sz = slice(None) if "z" in keep else zsl
    return var[sx, sy, sz]
