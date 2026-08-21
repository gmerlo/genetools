# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
contours.py — Streaming 2D slice plotter for GENE field and moment data.

Design for large arrays (up to 1536 x 700 x 128)
--------------------------------------------------
1.  Slice before IFFT — extract z-slice (XY) or ky-slice (XZ) from the
    3D array before any transform.  All IFFTs operate on 2D arrays.
2.  irfft instead of mirror+ifft — avoids the Hermitian mirror allocation
    and returns a real array directly.
3.  No unnecessary copies — mode filters applied to 2D slice copy only.
4.  float32 downcast before IFFT — halves memory, sufficient for plotting.
5.  stream_selected_with_seg — buffers at most one result at a time.

Axis selection
--------------
When coords is provided axes show physical units:
  x-axis XY: x  if IFFT along x (or global),  kx otherwise
  y-axis XY: y  if IFFT along y,               ky otherwise
  y-axis XZ: same rule as XY x-axis
  x-axis XZ: z always

iy always indexes the stored ky dimension (before y-transform).
"""

import numpy as np
import matplotlib.pyplot as plt

from genetools.diagnostics._base import RunDiagnostic

#: Axis labels shared with the slice diagnostic.
_AXIS_LABELS = {
    "x": r"$x/a$", "y": r"$y/\rho_{\rm ref}$", "z": r"$z/\pi$",
    "kx": r"$k_x \rho_{\rm ref}$", "ky": r"$k_y \rho_{\rm ref}$",
}


def _ifft_x_2d(f2d, nx):
    """IFFT along axis 0 with GENE normalisation (multiply by nx)."""
    return np.fft.ifft(f2d, axis=0) * nx


def _irfft_y_2d(f2d, ny_full):
    """Real IFFT along axis 1 from one-sided spectrum, GENE normalisation."""
    return np.fft.irfft(f2d, n=ny_full, axis=1) * ny_full


class Contours(RunDiagnostic):
    """
    Streaming 2-D slice plotter for GENE field and moment data.

    For the spectral geometries this streams the ``(x, y)`` and ``(x, z)``
    slices, inverse-transforming whichever perpendicular directions are
    spectral. GENE-3D is already real space in both, so its contour is the
    z-averaged ``xy`` plane and comes from
    :class:`~genetools.diagnostics.slices.Slices`, which owns the reduction
    machinery — this class just selects the ``xy`` product of it.

    Parameters
    ----------
    run : genetools.run.Run, optional
        ``None`` gives a detached instance exposing only the pure helpers.
    cmap : str, optional
        Matplotlib colormap (default 'bwr').
    """

    name = "contours"

    def __init__(self, run=None, cmap="bwr"):
        super().__init__(run)
        self.cmap = cmap
        self._slices = None
        self._slice_kw = {}

    # ------------------------------------------------------------------
    # GENE-3D: delegate the reduction, keep the xy product
    # ------------------------------------------------------------------

    #: Planes drawn, in order, and the coordinate each one holds fixed.
    _PLANES = (("xy", "z"), ("xz", "y"))

    @staticmethod
    def _at(values, target=0.0):
        """
        Return equal bounds on the grid point of *values* nearest *target*.

        Equal bounds make :func:`~genetools.diagnostics.slices._index_window`
        pick exactly that index, so the plane is a slice and not an average.
        """
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return None
        v = float(arr[int(np.argmin(np.abs(arr - target)))])
        return (v, v)

    def _reducer(self, kw):
        """
        Return a :class:`~genetools.diagnostics.slices.Slices` for *kw*.

        The defaults are the two cuts almost always wanted: ``xy`` at the
        parallel position nearest ``z = 0`` (the outboard midplane on a standard
        grid) and ``xz`` at ``y = 0``. A bare reduction averages over the whole
        of ``z`` instead, which smears the outboard and inboard sides of a global
        run together. Pass ``zlim``/``ylim`` to average over a range, or to cut
        elsewhere.

        Rebuilt only when the selection changes, so repeated plots of the same
        thing reuse the streamed data.
        """
        from genetools.diagnostics.slices import Slices
        # Built from the defaults and *this* call's arguments only. Carrying the
        # previous call's arguments forward would make an earlier `zlim` leak
        # into a later plot that never asked for it — the cached reducer below
        # is the only thing that persists.
        merged = {"quantities": ("phi",)}
        merged.update(kw)
        merged.setdefault("zlim", self._at(self.coord["z"]))
        merged.setdefault("ylim", self._at(self.coord["y"]))
        if self._slices is None or merged != self._slice_kw:
            self._slice_kw = merged
            self._slices = Slices(self.run, **merged)
        return self._slices

    def compute(self, t=None, **kw):
        """GENE-3D only: stream and reduce, returning the raw reductions."""
        self._require("xy_global")
        return self._reducer(kw).compute(t)

    def dataset(self, t=None, **kw):
        """
        GENE-3D only: the ``xy`` and ``xz`` cuts as an :class:`xarray.Dataset`.

        The spectral paths are plot-only — they stream, draw and discard, which
        is what makes them usable on runs whose field file does not fit in
        memory.
        """
        self._require("xy_global")
        ds = self._reducer(kw).dataset(t)
        keep = [n for n in ds.data_vars
                if any(n.endswith("_" + p) for p, _ in self._PLANES)]
        return ds[keep]

    def plot(self, t=None, **kw):
        """
        Plot the 2-D contours.

        GENE-3D takes ``quantities``, ``species``, ``x_fourier``, ``y_fourier``,
        ``square``, ``zlim`` and ``n_max``; the spectral geometries take the
        arguments of :meth:`plot_timeseries_2d`.
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

    def _plot_3d(self, t, n_max=4, **kw):
        """
        Plot the ``xy`` and ``xz`` cuts, one column per output time.

        ``x`` runs horizontally in both, so the two rows share a radial axis and
        read together. At most *n_max* times are drawn, evenly spaced through the
        window, so a long run does not open hundreds of axes.
        """
        ds = self.dataset(t, **kw)
        quantities = sorted({n.rsplit("_", 1)[0] for n in ds.data_vars})
        if not quantities:
            raise ValueError("No slice available to plot.")

        held_of = dict(self._PLANES)
        fixed = self._fixed_values()
        figs = []
        for name in quantities:
            planes = [p for p, _ in self._PLANES if f"{name}_{p}" in ds]
            frames, n_total = self._pick_times(ds[f"{name}_{planes[0]}"], n_max)
            if n_total > len(frames):
                print(f"contours: showing {len(frames)} of {n_total} times.")

            fig, axes = plt.subplots(len(planes), len(frames),
                                     figsize=(4.4 * len(frames),
                                              3.5 * len(planes)),
                                     squeeze=False)
            for row, plane in enumerate(planes):
                da = ds[f"{name}_{plane}"]
                # One symmetric colour scale per plane, so the time panels are
                # comparable instead of each being self-normalised.
                vmax = float(np.max(np.abs(np.asarray(da)))) or 1.0
                for col, (time, index) in enumerate(frames):
                    ax = axes[row][col]
                    frame = da if index is None else da.isel(time=index)
                    dims = frame.dims
                    # dims[0] is the radial axis: put it horizontal.
                    h, v = np.asarray(ds[dims[0]]), np.asarray(ds[dims[1]])
                    mesh = ax.pcolormesh(h, v, np.asarray(frame).T,
                                         shading="nearest", cmap=self.cmap,
                                         vmin=-vmax, vmax=vmax)
                    ax.set_xlabel(_AXIS_LABELS.get(dims[0], dims[0]))
                    ax.set_ylabel(_AXIS_LABELS.get(dims[1], dims[1]))
                    bits = [plane]
                    held = fixed.get(held_of[plane])
                    if held is not None:
                        bits.append(f"{held_of[plane]}={held:.3g}")
                    if time is not None:
                        bits.append(f"t={time:.4g}")
                    ax.set_title("  ".join(bits), fontsize=9)
                    fig.colorbar(mesh, ax=ax)
            fig.suptitle(name)
            fig.tight_layout()
            figs.append(fig)
        plt.show()
        return figs

    def _fixed_values(self):
        """The coordinate values the cuts hold fixed, for the panel titles."""
        out = {}
        for _, held in self._PLANES:
            limits = self._slice_kw.get(f"{held}lim")
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
