# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
contours.py — 2-D cuts and 1-D line profiles of GENE field and moment data.

One class covers every geometry and every projection of a snapshot: the three
2-D planes (``xy``, ``xz``, ``yz``) and the three 1-D lines (``x``, ``y``,
``z``). Which ones you get is the ``reductions`` argument; the default is the
two cuts almost always wanted, ``xy`` and ``xz``.

Cut, not average
----------------
Every coordinate a reduction drops is held at the grid point nearest zero unless
you pass ``xlim``/``ylim``/``zlim``, in which case it is averaged over that
range. So ``xy`` defaults to the plane at ``z = 0`` (the outboard midplane on a
standard grid) and ``xz`` to the plane at ``y = 0``. A bare average over the
whole of ``z`` would smear the outboard and inboard sides of a global run
together, which is why it is not the default. ``zlim=(z[0], z[-1])`` asks for it
explicitly.

Averages are plain means, not Jacobian-weighted: a cut is a picture of the field
on the grid, and weighting it by the volume element would show the metric as
much as the turbulence. The flux and profile diagnostics, which do need the
volume element, weight explicitly.

Geometry paths
--------------
GENE-3D is real space in x and y, so a reduction is a slice-and-mean of the
stored array and either horizontal direction can be viewed in Fourier space
instead (``x_fourier``, ``y_fourier``).

The spectral geometries are plot-only and stream: slice the 3-D array *before*
any transform, so every IFFT runs on a 2-D array; use ``irfft`` rather than a
Hermitian mirror; downcast to float32 first; and buffer one snapshot at a time.
That is what makes them usable on field files larger than memory (up to
1536 x 700 x 128), and it is also why they have no ``.dataset()`` — the frames
are drawn and discarded. Axes show physical units when coordinates are
available:

  x-axis XY: x  if IFFT along x (or global),  kx otherwise
  y-axis XY: y  if IFFT along y,               ky otherwise
  y-axis XZ: same rule as XY x-axis
  x-axis XZ: z always

``iy`` always indexes the stored ky dimension, before any y-transform.
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
    "x": r"$x/a$", "y": r"$y/\rho_{\rm ref}$", "z": r"$z/\pi$",
    "kx": r"$k_x \rho_{\rm ref}$", "ky": r"$k_y \rho_{\rm ref}$",
}

#: The reduction options, frozen so it can key the snapshot cache.
_Selection = namedtuple(
    "_Selection",
    "quantities species x_fourier y_fourier square xlim ylim zlim t_avg")


def _ifft_x_2d(f2d, nx):
    """IFFT along axis 0 with GENE normalisation (multiply by nx)."""
    return np.fft.ifft(f2d, axis=0) * nx


def _irfft_y_2d(f2d, ny_full):
    """Real IFFT along axis 1 from one-sided spectrum, GENE normalisation."""
    return np.fft.irfft(f2d, n=ny_full, axis=1) * ny_full


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

    Call options (GENE-3D)
    ----------------------
    quantities : sequence of str
        Variable names from the field or moment file. Default ``('phi',)``.
    species : str
        Species whose moment file supplies any moment quantities. Defaults to
        the first species.
    reductions : sequence of str or 'all'
        Any of ``xy``, ``xz``, ``yz``, ``x``, ``y``, ``z``. Default
        ``('xy', 'xz')``.
    x_fourier, y_fourier : bool
        View that direction in Fourier space (``|FFT|``).
    square : bool
        Reduce ``|f|^2`` rather than ``f``.
    xlim, ylim, zlim : (float, float)
        Average the dropped coordinate over this range instead of cutting at
        zero. ``xlim`` is in ``x/a``, ``ylim`` in ``y/rho_ref``, ``zlim`` in
        ``z``.
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
        Build the frozen selection for *kw*, filling in the cut-at-zero defaults.

        Built from the defaults and *this* call's arguments only. Carrying the
        previous call's arguments forward would make an earlier ``zlim`` leak
        into a later plot that never asked for it.
        """
        unknown = set(kw) - set(_Selection._fields)
        if unknown:
            raise TypeError(
                f"unknown contour option(s) {sorted(unknown)}; "
                f"expected any of {list(_Selection._fields)}")
        coord = self.coord
        return _Selection(
            quantities=tuple(kw.get("quantities", ("phi",))),
            species=kw.get("species") or (self.run.species[0]
                                          if self.run.species else None),
            x_fourier=bool(kw.get("x_fourier", False)),
            y_fourier=bool(kw.get("y_fourier", False)),
            square=bool(kw.get("square", False)),
            xlim=kw.get("xlim"),
            ylim=(kw["ylim"] if "ylim" in kw
                  else self._at(coord["y"])),
            zlim=(kw["zlim"] if "zlim" in kw
                  else self._at(coord["z"])),
            t_avg=bool(kw.get("t_avg", False)),
        )

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
    # GENE-3D: the reduction engine
    # ------------------------------------------------------------------

    def _windows(self, sel, coord, shape):
        """Index slices for the three coordinates, from the requested limits."""
        nx, ny, nz = shape
        return (c.index_window(coord["x_o_a"], sel.xlim, nx),
                c.index_window(coord["y"], sel.ylim, ny),
                c.index_window(coord["z"], sel.zlim, nz))

    def _transform(self, sel, var):
        """Apply the requested Fourier views, then ``|.|`` or ``|.|^2``."""
        out = var
        if sel.x_fourier:
            out = np.fft.fftshift(np.abs(c.to_kx(out)), axes=0)
        if sel.y_fourier:
            out = np.fft.fftshift(np.abs(c.to_ky(out)), axes=1)
        if sel.square:
            out = np.abs(out) ** 2
        return out

    def compute(self, t=None, **kw):
        """
        GENE-3D only: stream the requested variables and build every reduction.

        All six reductions are built — they are cheap means over an array already
        in memory — so ``reductions`` only selects what :meth:`dataset` returns
        and one streaming pass serves any choice of them.
        """
        self._require("xy_global")
        _, sel = self._split_kw(kw)
        key = (self._key(t), sel)
        if key in self._cache:
            return self._cache[key]

        coord = self.coord
        acc, times = {}, None
        for reader, names in self._sources(sel.quantities, sel.species):
            _, idx = self._indices(reader, t)
            slots = {n: reader.index_of(n) for n in names}
            got = []
            for time, arrays in reader.stream_selected(idx):
                got.append(time)
                for n in names:
                    var = self._transform(sel, arrays[slots[n]])
                    xsl, ysl, zsl = self._windows(sel, coord, var.shape)
                    store = acc.setdefault(n, {})
                    for plane, axis in _PLANES.items():
                        sub = _apply(var, xsl, ysl, zsl, keep=plane)
                        store.setdefault(plane, []).append(sub.mean(axis=axis))
                    for line, axes in _LINES.items():
                        sub = _apply(var, xsl, ysl, zsl, keep=line)
                        store.setdefault(line, []).append(sub.mean(axis=axes))
            if times is None:
                times = np.asarray(got)

        reduced = {}
        for name, store in acc.items():
            reduced[name] = {}
            for red, stack in store.items():
                arr = np.asarray(stack)
                reduced[name][red] = (self._time_average(arr, times)
                                      if sel.t_avg else arr)

        result = {"reduced": reduced, "times": times, "coord": coord,
                  "selection": sel}
        self._cache[key] = result
        return result

    def _axis_values(self, sel, coord):
        """Coordinate values for each named axis, honouring the Fourier views."""
        return {
            "x": (np.fft.fftshift(np.asarray(coord["kx"])) if sel.x_fourier
                  else np.asarray(coord["x_o_a"])),
            "y": (np.fft.fftshift(np.asarray(coord["ky"])) if sel.y_fourier
                  else np.asarray(coord["y"])),
            "z": np.asarray(coord["z"]),
        }

    def dataset(self, t=None, **kw):
        """
        GENE-3D only: the requested reductions as an :class:`xarray.Dataset`.

        The spectral paths are plot-only — they stream, draw and discard, which
        is what makes them usable on runs whose field file does not fit in
        memory.
        """
        self._require("xy_global")
        reductions, sel = self._split_kw(kw)
        raw = self.compute(t, **kw)
        coord = raw["coord"]
        axis_vals = self._axis_values(sel, coord)
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
            candidates[axis_name] = axis_vals[ch]
        if not sel.t_avg:
            candidates["time"] = raw["times"]

        params = self.params
        ds = make_dataset(data_vars, candidates, params=params)
        ds.attrs.update(unit_attrs(params))
        ds.attrs["geometry_kind"] = self.geometry_kind
        ds.attrs["x_fourier"] = int(sel.x_fourier)
        ds.attrs["y_fourier"] = int(sel.y_fourier)
        ds.attrs["squared"] = int(sel.square)
        if sel.species:
            ds.attrs["species"] = sel.species
        return ds

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, t=None, **kw):
        """
        Plot the requested reductions.

        GENE-3D takes the call options listed in the class docstring; the
        spectral geometries take the arguments of :meth:`plot_timeseries_2d`.
        """
        if self.is_3d:
            return self._plot_3d(t, **kw)
        return self._plot_spectral(t, **kw)

    def _plot_spectral(self, t, **kw):
        a, b = self._bounds(t)
        species = kw.pop("species", None)
        reader = self.run.field if species is None else self.run.mom(species)
        return self.plot_timeseries_2d(
            reader, a, b, params_list=self.params, coords=self.coord,
            species=species, **kw)

    def _plot_3d(self, t=None, n_max=4, **kw):
        """
        Plot the requested reductions, one row each and one column per time.

        ``x`` runs horizontally in every reduction that has it, so the rows share
        a radial axis and read together.
        """
        reductions, sel = self._split_kw(kw)
        ds = self.dataset(t, **kw)
        quantities = [q for q in sel.quantities
                      if any(f"{q}_{r}" in ds for r in reductions)]
        if not quantities:
            raise ValueError("No reduction available to plot.")

        fixed = self._fixed_values(sel)
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
                        self._draw_plane(fig, ax, ds, frame, vmax)
                    else:
                        self._draw_line(ax, ds, frame, vmax)
                    ax.set_title(self._panel_title(red, fixed, time),
                                 fontsize=9)
            fig.suptitle(self._title(name, sel))
            fig.tight_layout()
            figs.append(fig)
        plt.show()
        return figs

    def _draw_plane(self, fig, ax, ds, frame, vmax):
        """pcolormesh of a 2-D reduction, with the radial axis horizontal."""
        dims = frame.dims
        # dims[0] is the radial axis: put it horizontal.
        h, v = np.asarray(ds[dims[0]]), np.asarray(ds[dims[1]])
        mesh = ax.pcolormesh(h, v, np.asarray(frame).T, shading="nearest",
                             cmap=self.cmap, vmin=-vmax, vmax=vmax)
        ax.set_xlabel(_AXIS_LABELS.get(dims[0], dims[0]))
        ax.set_ylabel(_AXIS_LABELS.get(dims[1], dims[1]))
        fig.colorbar(mesh, ax=ax)

    @staticmethod
    def _draw_line(ax, ds, frame, vmax):
        """Line plot of a 1-D reduction, on the scale shared across the row."""
        dim = frame.dims[0]
        ax.plot(np.asarray(ds[dim]), np.asarray(frame))
        ax.set_xlabel(_AXIS_LABELS.get(dim, dim))
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
    def _fixed_values(sel):
        """The coordinates this selection cuts at, for the panel titles."""
        out = {}
        for held in "xyz":
            limits = getattr(sel, f"{held}lim")
            if limits and limits[0] == limits[1]:
                out[held] = float(limits[0])
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

    # ------------------------------------------------------------------
    # Spectral geometries: streaming plot-only path
    # ------------------------------------------------------------------

    def select_indices(self, reader, t_start, t_stop, max_loads):
        """Return downsampled iteration indices within the time window."""
        times = reader.read_all_times()
        mask  = (times >= t_start) & (times <= t_stop)
        idx   = np.where(mask)[0]
        if len(idx) == 0:
            print("No data found in the selected time interval.")
            return []
        if len(idx) > max_loads:
            stride = max(1, len(idx) // max_loads)
            idx    = idx[::stride][:max_loads]
        return idx.tolist()

    @staticmethod
    def _resolve_ifft(ifft_option, x_local):
        """Restrict ifft option for global geometry (x already real)."""
        if x_local:
            return ifft_option
        _map = {"xy": "y", "x": None, "y": "y", None: None}
        effective = _map.get(ifft_option, ifft_option)
        if effective != ifft_option:
            print(f"  [Contours] x_local=False: IFFT along x skipped "
                  f"('{ifft_option}' -> '{effective}')")
        return effective

    @staticmethod
    def _get_axes(coord, effective_ifft, x_local, nky=None):
        """
        Return (x_ax, y_ax, z_ax, x_label, y_label, z_label).

        x -> real (x) if ifft includes x or global, else spectral (kx).
        y -> real (y) if ifft includes y,            else spectral (ky).
        z -> always real (z).
        """
        if effective_ifft in ("x", "xy") or not x_local:
            x_ax, x_label = np.asarray(coord["x"]),  "x  [rho_ref]"
        else:
            x_ax, x_label = np.asarray(coord["kx"]), "kx [rho_ref]"

        if effective_ifft in ("y", "xy"):
            # Compute y-axis matching the irfft output size: ny_full = 2*(nky-1)
            ky_arr = np.asarray(coord["ky"])
            _nky = nky if nky is not None else len(ky_arr)
            ny_full = 2 * (_nky - 1) if _nky > 1 else 1
            kymin = float(ky_arr[0]) if len(ky_arr) > 0 else 1.0
            Ly = 2 * np.pi / kymin if kymin > 0 else 1.0
            y_ax = np.linspace(-Ly / 2, Ly / 2, ny_full, endpoint=False)
            y_label = "y  [rho_ref]"
        else:
            y_ax, y_label = np.asarray(coord["ky"]), "ky [rho_ref]"

        z_ax, z_label = np.asarray(coord["z"]), "z  [pi]"

        return x_ax, y_ax, z_ax, x_label, y_label, z_label

    def _compute_slices(self, field_3d, effective_ifft,
                        iz, iy, del_zonal, zero_range, nky):
        """
        Extract XY and XZ 2D slices with IFFT applied on 2D only.

        The 3D array is never copied. Mode filters are applied to a 2D
        copy of the z-slice only when needed. Both outputs are float32.

        Parameters
        ----------
        field_3d : np.ndarray  (nx, nky, nz) complex
        effective_ifft : str or None
        iz : int   z-index for XY slice
        iy : int   ky-index for XZ slice (always pre-y-IFFT index)
        del_zonal, zero_range : filter parameters
        nky : int  number of stored ky modes

        Returns
        -------
        f_xy : np.ndarray (nx, ny_real)  float32
        f_xz : np.ndarray (nx, nz)       float32
        """
        nx      = field_3d.shape[0]
        ny_full = 2 * (nky - 1) if nky > 1 else 0

        # ── XY slice — extract z first, transform in 2D ────────────────
        f_xy = field_3d[:, :, iz]               # view (nx, nky), no copy
        f_xy = f_xy.astype(np.complex64, copy=False)  # downcast, may be view

        if del_zonal or zero_range is not None:
            f_xy = f_xy.copy()                  # only copy needed here
            if del_zonal:
                f_xy[:, 0] = 0.0
            if zero_range is not None:
                f_xy[:, :zero_range] = 0.0

        if effective_ifft in ("x", "xy"):
            f_xy = _ifft_x_2d(f_xy, nx)         # (nx, nky) complex

        if effective_ifft in ("y", "xy") and ny_full > 0:
            f_xy = _irfft_y_2d(f_xy, ny_full).astype(np.float32)
        else:
            f_xy = f_xy.real.astype(np.float32)

        # ── XZ slice — extract ky first, transform in 2D ───────────────
        f_xz = field_3d[:, iy, :]               # view (nx, nz), no copy
        f_xz = f_xz.astype(np.complex64, copy=False)

        if effective_ifft in ("x", "xy"):
            f_xz = _ifft_x_2d(f_xz, nx).real.astype(np.float32)
        else:
            f_xz = f_xz.real.astype(np.float32)

        return f_xy, f_xz

    def plot_timeseries_2d(
        self,
        reader,
        t_start,
        t_stop,
        field=0,
        max_loads=9,
        iz=None,
        iy=None,
        ifft=None,
        del_zonal=False,
        zero_range=None,
        params_list=None,
        coords=None,
        show_xz=True,
        species=None,
    ):
        """
        Stream selected time steps and plot 2D XY and XZ slices.

        Parameters
        ----------
        reader
            BinaryReader, BPReader, or MultiSegmentReader.
        t_start, t_stop : float
            Time window.
        field : int, optional
            Field/moment index. Fields: 0=phi,1=A_par,2=B_par.
            Moments: 0=n,1=T_par,2=T_perp,3=q_par,4=q_perp,5=u_par.
        max_loads : int, optional
            Max snapshots (default 9).
        iz : int, optional
            z-index for XY slice (default nz//2).
        iy : int, optional
            ky-index for XZ slice (default nky//2). Always indexes the
            stored ky dimension regardless of ifft.
        ifft : str or None, optional
            None | 'x' | 'y' | 'xy'. Auto-restricted for global geometry.
        del_zonal : bool, optional
            Zero ky=0 before transforming (default False).
        zero_range : int or None, optional
            Zero ky=0..N-1 before transforming (default None).
        params_list : list of dict, or dict, optional
            Per-segment parameter dicts. Single dict accepted for
            single-segment readers.
        coords : list of dict, or dict, optional
            Per-segment coordinate dicts from Coordinates().
            Single dict accepted. Enables physical axis labels.
        show_xz : bool, optional
            Show XZ figure (default True).
        species : str, optional
            Species name appended to subplot titles, e.g. 'ions'.
        """
        # Normalise to lists
        if params_list is not None and not isinstance(params_list, list):
            params_list = [params_list]
        if coords is not None and not isinstance(coords, list):
            coords = [coords]

        def _p(seg):
            return params_list[seg] if params_list else None

        def _c(seg):
            return coords[seg] if coords else None

        # Default slice indices
        if iz is None:
            iz = reader.nk // 2
        if iy is None:
            iy = reader.nj // 2

        indices = self.select_indices(reader, t_start, t_stop, max_loads)
        
        if not indices:
            return

        n_plots = len(indices)
        ncols   = min(3, n_plots)
        nrows   = int(np.ceil(n_plots / ncols))

        fig_xy, axes_xy = plt.subplots(nrows, ncols,
                                        figsize=(5*ncols, 4*nrows),
                                        squeeze=False)
        axes_xy = axes_xy.reshape(-1)

        if show_xz:
            fig_xz, axes_xz = plt.subplots(nrows, ncols,
                                            figsize=(5*ncols, 4*nrows),
                                            squeeze=False)
            axes_xz = axes_xz.reshape(-1)

        sp_str = f"  [{species}]" if species else ""

        for plot_idx, (t, arrays, seg_idx) in enumerate(
                reader.stream_selected_with_seg(indices)):

            p     = _p(seg_idx)
            coord = _c(seg_idx)

            x_local = p.get("general", {}).get("x_local", True) if p else True
            nky     = p["box"]["nky0"] if p else reader.nj

            effective_ifft = self._resolve_ifft(ifft, x_local)

            if coord is not None:
                x_ax, y_ax, z_ax, x_label, y_label, z_label = \
                    self._get_axes(coord, effective_ifft, x_local, nky=nky)
            else:
                x_ax = y_ax = z_ax = None
                x_label, y_label, z_label = "x index", "y index", "z index"

            f_xy, f_xz = self._compute_slices(
                arrays[field], effective_ifft, iz, iy,
                del_zonal, zero_range, nky,
            )

            # XY subplot
            ax = axes_xy[plot_idx]
            if x_ax is not None:
                nx_s, ny_s = f_xy.shape
                im = ax.pcolormesh(x_ax[:nx_s], y_ax[:ny_s], f_xy.T,
                                   cmap=self.cmap, shading="auto")
            else:
                im = ax.imshow(f_xy.T, origin="lower", aspect="auto",
                               cmap=self.cmap)
            ax.set_title(f"t={t:.2f}  z={iz}{sp_str}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            fig_xy.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # XZ subplot
            if show_xz:
                ax = axes_xz[plot_idx]
                if x_ax is not None:
                    nx_s, nz_s = f_xz.shape
                    im = ax.pcolormesh(x_ax[:nx_s], z_ax[:nz_s], f_xz.T,
                                       cmap=self.cmap, shading="auto")
                else:
                    im = ax.imshow(f_xz.T, origin="lower", aspect="auto",
                                   cmap=self.cmap)
                ax.set_title(f"t={t:.2f}  ky={iy}{sp_str}")
                ax.set_xlabel(x_label)
                ax.set_ylabel(z_label)
                fig_xz.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for ax in axes_xy[n_plots:]:
            ax.axis("off")
        fig_xy.suptitle("XY slices", y=1.01)
        fig_xy.tight_layout()

        if show_xz:
            for ax in axes_xz[n_plots:]:
                ax.axis("off")
            fig_xz.suptitle("XZ slices", y=1.01)
            fig_xz.tight_layout()

        plt.show()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply(var, xsl, ysl, zsl, keep):
    """Apply the index windows, leaving the axes named in *keep* unrestricted."""
    sx = slice(None) if "x" in keep else xsl
    sy = slice(None) if "y" in keep else ysl
    sz = slice(None) if "z" in keep else zsl
    return var[sx, sy, sz]
