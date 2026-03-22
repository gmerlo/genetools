"""
contours.py — Streaming 2D slice plotter for GENE field and moment data.

The key method is :meth:`Contours.plot_timeseries_2d`, which streams a
selection of time snapshots from a reader, optionally transforms to real
space via IFFT, and produces XY and XZ slice plots.

Geometry-aware IFFT
-------------------
Whether an IFFT is needed along x depends on the simulation geometry:

* **Local** (``x_local=True``)  — x is spectral (kx space).
  ``ifft='xy'`` transforms both x and y to real space.
* **Global** (``x_local=False``) — x is already real-space.
  ``ifft='xy'`` or ``ifft='x'`` silently degrades to ``ifft='y'`` only,
  because applying IFFT along an already-real axis is meaningless.

Pass the ``params`` dictionary to :meth:`plot_timeseries_2d` and this is
handled automatically.

Example
-------
>>> from contours import Contours
>>> plotter = Contours()

>>> # Local geometry — full real-space transform
>>> plotter.plot_timeseries_2d(field_readers[-1], 10.5, 2850.,
...     field=0, max_loads=6, ifft='xy', params=params.get(0))

>>> # Global geometry — x already real, only y transformed
>>> plotter.plot_timeseries_2d(field_readers[-1], 10.5, 2850.,
...     field=0, max_loads=6, ifft='xy', params=params.get(0))
... # ifft is automatically downgraded to 'y' since x_local=False
"""

import numpy as np
import matplotlib.pyplot as plt


class Contours:
    """
    Streaming time-series plotter for GENE field and moment data.

    Parameters
    ----------
    cmap : str, optional
        Matplotlib colormap for 2D plots (default ``'bwr'``).
    """

    def __init__(self, cmap: str = "bwr"):
        self.cmap = cmap

    # ------------------------------------------------------------------
    # Time index selection
    # ------------------------------------------------------------------

    def select_indices(self, reader, t_start: float, t_stop: float,
                       max_loads: int) -> list:
        """
        Return a downsampled list of iteration indices within the time window.

        Parameters
        ----------
        reader : BinaryReader or BPReader
            Data reader with a ``read_all_times()`` method.
        t_start, t_stop : float
            Time window boundaries.
        max_loads : int
            Maximum number of snapshots to return.

        Returns
        -------
        list of int
            Zero-based iteration indices, at most *max_loads* entries.
        """
        times = reader.read_all_times()
        mask  = (times >= t_start) & (times <= t_stop)
        idx   = np.where(mask)[0]

        if len(idx) == 0:
            print("⚠  No data found in the selected time interval.")
            return []

        if len(idx) > max_loads:
            stride = max(1, len(idx) // max_loads)
            idx    = idx[::stride][:max_loads]

        return idx.tolist()

    # ------------------------------------------------------------------
    # Geometry-aware IFFT
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_ifft(ifft_option, x_local: bool):
        """
        Return the effective IFFT option given the simulation geometry.

        In global geometry (``x_local=False``) the x direction is already
        in real space, so any request to transform x is silently dropped:

        * ``'xy'`` → ``'y'``
        * ``'x'``  → ``None``
        * ``'y'``  → ``'y'``   (unchanged)
        * ``None`` → ``None``  (unchanged)

        Parameters
        ----------
        ifft_option : str or None
            Requested IFFT axes.
        x_local : bool
            ``True`` for local (spectral-x) geometry.

        Returns
        -------
        str or None
            Effective IFFT option to apply.
        """
        if x_local:
            return ifft_option          # local: honour request as-is

        # Global: x is already real — strip the x component
        _map = {"xy": "y", "x": None, "y": "y", None: None}
        effective = _map.get(ifft_option, ifft_option)

        if effective != ifft_option:
            print(
                f"  [Contours] x_local=False → IFFT along x skipped "
                f"('{ifft_option}' → '{effective}')"
            )
        return effective

    def apply_ifft(self, field: np.ndarray, ifft_option, x_local: bool,
                   del_zonal: bool, zero_range) -> np.ndarray:
        """
        Apply geometry-aware IFFT and optional mode filtering.

        Parameters
        ----------
        field : np.ndarray
            Complex array of shape ``(nx, nky, nz)``.
        ifft_option : str or None
            Requested transform axes: ``None``, ``'x'``, ``'y'``, ``'xy'``.
        x_local : bool
            Whether the simulation uses local (spectral-x) geometry.
        del_zonal : bool
            If ``True``, zero out the ky=0 (zonal) component before
            transforming.
        zero_range : int or None
            If set, zero out modes ``ky = 0 .. zero_range-1`` before
            transforming.

        Returns
        -------
        np.ndarray
            Transformed field array.
        """
        effective = self._resolve_ifft(ifft_option, x_local)

        # Apply mode filters only if we'll be transforming y
        if del_zonal or zero_range is not None:
            f = field.copy()
            if del_zonal:
                f[:, 0, :] = 0.0
            if zero_range is not None:
                f[:, :zero_range, :] = 0.0
        else:
            f = field

        # Transform x (local geometry only — already guarded by _resolve_ifft)
        if effective in ("x", "xy"):
            f = np.fft.ifft(f, axis=0)

        # Transform y — reconstruct full spectrum then IFFT
        if effective in ("y", "xy"):
            # GENE stores only non-negative ky; mirror to get full spectrum
            f_full = np.concatenate([f, np.conj(f[:, -2:0:-1, :])], axis=1)
            f = np.fft.ifft(f_full, axis=1)

        return f

    # ------------------------------------------------------------------
    # Main plotting method
    # ------------------------------------------------------------------

    def plot_timeseries_2d(
        self,
        reader,
        t_start: float,
        t_stop: float,
        field: int = 0,
        max_loads: int = 9,
        iz: int = None,
        iy: int = None,
        ifft=None,
        del_zonal: bool = False,
        zero_range=None,
        params: dict = None,
        show_xz: bool = False,
    ) -> None:
        """
        Stream selected time steps and plot 2D XY and XZ slices.

        Produces two figures:

        * **XY figure** — slice at z-index *iz* (default: midplane ``nz//2``)
        * **XZ figure** — slice at y-index *iy* (default: ``nky//2``)

        Each figure contains up to *max_loads* subplots with individual
        colour scales.

        Parameters
        ----------
        reader : BinaryReader or BPReader
            Data reader for the desired run segment.
        t_start, t_stop : float
            Time window to plot.
        field : int, optional
            Field or moment index to plot.

            Fields:  0=φ, 1=A∥, 2=B∥

            Moments: 0=n, 1=T∥, 2=T⊥, 3=q∥, 4=q⊥, 5=u∥
        max_loads : int, optional
            Maximum number of time snapshots to plot (default 9).
        iz : int, optional
            z-index for the XY slice (default ``nz//2``).
        iy : int, optional
            y-index for the XZ slice (default ``nky//2``).
        ifft : str or None, optional
            IFFT axes: ``None`` | ``'x'`` | ``'y'`` | ``'xy'``.
            Automatically restricted based on geometry when *params* is
            provided.
        del_zonal : bool, optional
            Zero out ky=0 before transforming (default ``False``).
        zero_range : int or None, optional
            Zero out ky=0..N-1 before transforming (default ``None``).
        params : dict, optional
            Parameter dictionary for this run segment.  When provided,
            ``params['general']['x_local']`` is used to automatically
            restrict the IFFT axes for global geometry.  If omitted,
            local geometry is assumed.
        show_xz : bool, optional
            Whether to produce the XZ figure (default ``True``).
            Set to ``False`` to show only the XY slices.
        """
        # Resolve geometry flag
        x_local = params.get("general", {}).get("x_local", True)

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

        for plot_idx, (t, arrays) in enumerate(reader.stream_selected(indices)):
            f = self.apply_ifft(arrays[field], ifft, x_local,
                                del_zonal, zero_range)

            # ── XY slice (fixed z) ─────────────────────────────────────
            f2d_xy = f[:, :, iz].real
            ax     = axes_xy[plot_idx]
            im     = ax.imshow(f2d_xy.T, origin="lower", aspect="auto",
                               cmap=self.cmap)
            ax.set_title(f"t={t:.2f},  z={iz}")
            ax.set_xlabel("y index" if not x_local else "Y index")
            ax.set_ylabel("x / x index")
            fig_xy.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # ── XZ slice (fixed y) — optional ──────────────────────────
            if show_xz:
                f2d_xz = f[:, iy, :].real
                ax     = axes_xz[plot_idx]
                im     = ax.imshow(f2d_xz.T, origin="lower", aspect="auto",
                                   cmap=self.cmap)
                ax.set_title(f"t={t:.2f},  y={iy}")
                ax.set_xlabel("z index")
                ax.set_ylabel("x / x index")
                fig_xz.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused axes
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