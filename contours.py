import numpy as np
import matplotlib.pyplot as plt

class Contours:
    """
    Streaming time-series plotter for FortranReader data.
    
    Supports:
      - selecting time interval automatically
      - plotting 2D slices at depth iz
      - optional IFFT along x, y, or both
    """

    def __init__(self, field=0, iz_default=0, cmap="bwr"):
        self.field = field
        self.cmap = cmap

    # ---------------------------------------------------------------

    def select_indices(self, reader, t_start, t_stop, max_loads):
        times = reader.read_all_times()

        mask = (times >= t_start) & (times <= t_stop)
        idx = np.where(mask)[0]

        if len(idx) == 0:
            print("⚠ No data in selected time interval.")
            return []

        # Downsample
        if len(idx) > max_loads:
            stride = max(1, len(idx) // max_loads)
            idx = idx[::stride]
            idx = idx[:max_loads]

        return idx.tolist()

    # ---------------------------------------------------------------
    # ----------- FFT helpers --------------------------------------
    # ---------------------------------------------------------------

    def apply_ifft(self, field, ifft_option, del_zonal, zero_range):
        """
        field: 3D array (nx, ny_half, nz)
        ifft_option: None, "x", "y", or "xy"
        """
        # Only copy if we need to modify the array
        if del_zonal or zero_range is not None:
            f = np.array(field, copy=True)
            if del_zonal:
                f[:, 0, :] = 0
            if zero_range is not None:
                f[:, 0:zero_range, :] = 0
        else:
            f = field  # No modification needed, use original array

        # Apply IFFT along requested axes
        if ifft_option in ("x", "xy"):
            f = np.fft.ifft(f, axis=0)
        if ifft_option in ("y", "xy"):
            f_full = np.concatenate([f, np.conj(f[:, -2:0:-1, :])], axis=1)
            f = np.fft.ifft(f_full, axis=1)

        return f


   
    def plot_timeseries_2d(self, reader, t_start, t_stop, field=None, max_loads=9, iz=None, iy=None, ifft=None,
                          del_zonal=False, zero_range=None):
        """
        Stream & plot selected times as subplots in two figures:
          - XY slices at iz (x-y plane)
          - XZ slices at iy (x-z plane)

        Each figure has up to max_loads subplots (time steps).
        Each subplot has its own color scale.
        """
        if iz is None:
            iz = reader.nk // 2
        if iy is None:
            iy = reader.nj // 2

        indices = self.select_indices(reader, t_start, t_stop, max_loads)
        if len(indices) == 0:
            return

        n_plots = len(indices)
        ncols = min(3, n_plots)
        nrows = int(np.ceil(n_plots / ncols))

        # Create figures with subplots
        fig_xy, axes_xy = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        fig_xz, axes_xz = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))

        axes_xy = np.array(axes_xy).reshape(-1)
        axes_xz = np.array(axes_xz).reshape(-1)

        # Plot each time step
        for idx, (t, arrays) in enumerate(reader.stream_selected(indices)):                      
            f = self.apply_ifft(arrays[self.field if field is None else field], ifft, del_zonal, zero_range)

            # XY slice
            f2d_xy = f[:, :, iz].real
            ax_xy = axes_xy[idx]
            im_xy = ax_xy.imshow(f2d_xy.T, origin='lower', aspect='auto', cmap=self.cmap)
            ax_xy.set_title(f"t={t:.2f}, z={iz}")
            ax_xy.set_xlabel("Y index")
            ax_xy.set_ylabel("X index")
            fig_xy.colorbar(im_xy, ax=ax_xy, fraction=0.046, pad=0.04)

            # XZ slice
            f2d_xz = f[:, iy, :].real
            ax_xz = axes_xz[idx]
            im_xz = ax_xz.imshow(f2d_xz.T, origin='lower', aspect='auto', cmap=self.cmap)
            ax_xz.set_title(f"t={t:.2f}, y={iy}")
            ax_xz.set_xlabel("Z index")
            ax_xz.set_ylabel("X index")
            fig_xz.colorbar(im_xz, ax=ax_xz, fraction=0.046, pad=0.04)
            
        # Hide unused axes
        for ax in axes_xy[n_plots:]:
            ax.axis('off')
        for ax in axes_xz[n_plots:]:
            ax.axis('off')

        fig_xy.tight_layout()
        fig_xz.tight_layout()
        plt.show()
