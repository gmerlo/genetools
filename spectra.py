import numpy as np
import matplotlib.pyplot as plt
import os 
import h5py
# Try to import numba
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = False
except ImportError:
    NUMBA_AVAILABLE = False

class Spectra:
    def __init__(self, outfile="flux_spectra.h5"):
        self.outfile = outfile

    # -------------------- SYNC INDICES --------------------
    def sync_indices(self, fld_reader, mom_readers, t_start, t_stop, params):
        """Synchronize field & moment indices and skip already computed times."""
        istep_fld = int(params["in_out"]["istep_field"])
        istep_mom = int(params["in_out"]["istep_mom"])
        if istep_fld <= 0 or istep_mom <= 0:
            raise ValueError("istep_field and istep_mom must be positive integers")

        # LCM stride
        L = int(np.lcm(istep_fld, istep_mom))
        stride_fld = L // istep_fld
        stride_mom = L // istep_mom

        times_fld = fld_reader.read_all_times()
        idx_fld = np.where((times_fld >= t_start) & (times_fld <= t_stop))[0][::stride_fld]

        times_mom = mom_readers[0].read_all_times()
        idx_mom = np.where((times_mom >= t_start) & (times_mom <= t_stop))[0][::stride_mom]

        # Already saved times
        if os.path.exists(self.outfile):
            with h5py.File(self.outfile, "r") as f:
                saved_times = f["time"][...] if "time" in f else np.array([], dtype=np.float32)
        else:
            saved_times = np.array([], dtype=np.float32)

        # Compare in float32 precision
        def is_saved(t):
            t32 = np.float32(t)
            tol = max(1e-6, abs(t32)*1e-6)
            return np.any(np.abs(saved_times - t32) <= tol)

        idx_fld_filtered = [i for i in idx_fld if not is_saved(times_fld[i])]
        idx_mom_filtered = [i for i in idx_mom if not is_saved(times_mom[i])]
        return idx_fld_filtered, idx_mom_filtered

    # -------------------- COMPUTE SPECTRA --------------------
    def compute_spectra(self, fields, moments, coords, geom, params):
        """Compute reduced flux spectra along kx, ky, z for all species."""
        species = params["species"]
        ky = coords["ky"]
        kx = coords["kx"]
        z  = coords["z"]
        results = []

        for i_sp, sp in enumerate(species):
            mom = moments[i_sp]
            n0, T0, q0 = sp['dens'], sp['temp'], sp['charge']

            # Electrostatic
            G_es = self.averages(-1j*ky[np.newaxis,:,np.newaxis]*fields[0] * mom[0]*n0, geom)
            Q_es = self.averages(-1j*ky[np.newaxis,:,np.newaxis]*fields[0] * (0.5*mom[1]+mom[2]+1.5*mom[0])*n0*T0, geom)

            # Electromagnetic (if present)
            n_fields = fields.shape[0]
            if n_fields > 1:
                B_x = 1j*ky[np.newaxis,:,np.newaxis]*fields[1]
                tmp1 = B_x * mom[5]*n0
                tmp2 = B_x * (mom[3]+mom[4])*n0*T0
                if n_fields>2:
                    dBpar_dy = 1j*ky[np.newaxis,:,np.newaxis]*fields[2]/geom['Bfield']
                    tmp1 += dBpar_dy * mom[6]*n0*T0/q0
                    tmp2 += dBpar_dy * (mom[7]+mom[8])*n0*T0**2/q0
                G_em = self.averages(tmp1, geom)
                Q_em = self.averages(tmp2, geom)
            else:
                G_em = Q_em = (None, None, None)

            results.append([Q_es, Q_em, G_es, G_em])
        return results

    # -------------------- AVERAGES --------------------
    @staticmethod
    def averages(flux, geom):
        if flux is None:
            return (None, None, None)
        flux[:,1:,:] *= 2
        J = geom['Jacobian'][np.newaxis,np.newaxis,:] / np.sum(geom['Jacobian'])
        sum_z = np.sum(flux.real*J, axis=(0,1))
        avg_z = np.sum(flux.real*J, axis=2)
        sp_ky = np.sum(avg_z, axis=0)
        tmp = np.sum(avg_z, axis=1)
        nx = tmp.shape[0]
        nx2 = nx//2+1
        sp_kx = np.zeros(nx2)
        sp_kx[0] = tmp[0]
        if nx>1:
            if nx%2==1:
                sp_kx[1:nx2] = tmp[1:nx2]+tmp[-1:nx2-1:-1]
            else:
                sp_kx[1:nx2-1] = tmp[1:nx2-1]+tmp[-1:nx2-1:-1]
                sp_kx[nx2-1] = tmp[nx2-1]
        return sp_kx, sp_ky, sum_z

    # -------------------- APPEND TO HDF5 --------------------
    def append_fluxes_piecewise(self, fluxes, coords, species_names, time_value):
        kx = coords["kx"]
        ky = coords["ky"]
        z  = coords["z"]

        file_exists = os.path.exists(self.outfile)

        with h5py.File(self.outfile, "a") as f:

            # File creation
            if not file_exists:
                f.create_dataset("kx", data=kx)
                f.create_dataset("ky", data=ky)
                f.create_dataset("z",  data=z)
                f.create_dataset("time", data=np.array([time_value], np.float32),
                                 maxshape=(None,))

                for i_sp, name in enumerate(species_names):
                    Q_es, Q_em, G_es, G_em = fluxes[i_sp]
                    for label, flux in zip(["Q_es","Q_em","G_es","G_em"], [Q_es,Q_em,G_es,G_em]):
                        for axis_name, arr in zip(["kx","ky","z"], flux):
                            dsname = f"{name}_{label}_{axis_name}"
                            if arr is None:
                                f.create_dataset(dsname, data=np.zeros((1,0)), maxshape=(None,0))
                            else:
                                f.create_dataset(dsname, data=arr[np.newaxis,:], maxshape=(None, arr.size))
                return

            # Append mode
            tds = f["time"]
            tds.resize((tds.shape[0]+1,))
            tds[-1] = np.float32(time_value)

            for i_sp, name in enumerate(species_names):
                Q_es, Q_em, G_es, G_em = fluxes[i_sp]
                for label, flux in zip(["Q_es","Q_em","G_es","G_em"], [Q_es,Q_em,G_es,G_em]):
                    for axis_name, arr in zip(["kx","ky","z"], flux):
                        dsname = f"{name}_{label}_{axis_name}"
                        dset = f[dsname]
                        if arr is None:
                            dset.resize((dset.shape[0]+1, 0))
                        else:
                            arr = np.asarray(arr, float)
                            dset.resize((dset.shape[0]+1, dset.shape[1]))
                            dset[-1,:] = arr

    # -------------------- COMPUTE MISSING --------------------
    def compute_missing(self, field_iters, mom_iters, coords_list, geoms_list, params_list, t_start, t_stop):
        species_names = [sp['name'] for sp in params_list['species']]

        for fld_reader, mom_readers, coords, geom, params in zip(field_iters, mom_iters, coords_list, geoms_list, [params_list]*len(field_iters)):
            idx_fld, idx_mom = self.sync_indices(fld_reader, mom_readers, t_start, t_stop, params)
            if len(idx_fld)==0 or len(idx_mom)==0:
                continue
            it_field = fld_reader.stream_selected(idx_fld)
            it_moms  = [r.stream_selected(idx_mom) for r in mom_readers]

            for tm, fields in it_field:
                moments_data = [next(it)[1] for it in it_moms]
                fluxes = self.compute_spectra(fields, moments_data, coords, geom, params)
                self.append_fluxes_piecewise(fluxes, coords, species_names, tm)

    # -------------------- LOAD AND TIME AVERAGE --------------------
    def load_time_average(self):
        with h5py.File(self.outfile,"r") as f:
            time = f["time"][...]
            sorted_idx = np.argsort(time)
            time = time[sorted_idx]
            flux_avg = {}
            for key in f.keys():
                if key=="time" or key in ["kx","ky","z"]:
                    continue
                data = f[key][...]
                if data.shape[0]==1:
                    flux_avg[key] = data[0]
                    continue
                # trapezoidal average
                y_sorted = data[sorted_idx]
                flux_avg[key] = np.trapz(y_sorted, x=time, axis=0)/(time[-1]-time[0])
        return flux_avg

    # -------------------- PLOT SUBPLOTS --------------------
    def plot(self, field_iters, mom_iters, coords_list, geoms_list, params_list, t_start, t_stop, log_mode="loglog"):
        # Step 1: compute missing times
        self.compute_missing(field_iters, mom_iters, coords_list, geoms_list, params_list, t_start, t_stop)

        # Step 2: load all data and compute trapezoidal time averages
        flux_avg = self.load_time_average()

        # Step 3: load coordinates
        with h5py.File(self.outfile,"r") as f:
            kx = f["kx"][...]
            ky = f["ky"][...]
            z  = f["z"][...]
            species_names = list({name.split("_")[0] for name in f.keys() if "_" in name})

        # Step 4: plot 2x2 spectra + z-profiles for each species
        for sp in species_names:
            fig, axes = plt.subplots(2,2,figsize=(12,8))
            axes = axes.flatten()
            for i, label in enumerate(["Q_es","Q_em","G_es","G_em"]):
                # kx plot
                key = f"{sp}_{label}_kx"
                if key in flux_avg: axes[i].plot(kx, flux_avg[key], label="kx")
                # ky plot
                key = f"{sp}_{label}_ky"
                if key in flux_avg: axes[i].plot(ky, flux_avg[key], label="ky")
                axes[i].set_xlabel("k")
                axes[i].set_ylabel(label)
                axes[i].set_title(sp)
                axes[i].legend()
                axes[i].grid(True)
            plt.tight_layout()
            plt.show()

        # Plot z-profiles
        fig, axes = plt.subplots(len(species_names),1,figsize=(8,3*len(species_names)),sharex=True)
        if len(species_names)==1: axes=[axes]
        for i, sp in enumerate(species_names):
            for label in ["Q_es","Q_em","G_es","G_em"]:
                key = f"{sp}_{label}_z"
                if key in flux_avg: axes[i].plot(z, flux_avg[key], label=label)
            axes[i].set_ylabel("Flux")
            axes[i].set_title(sp)
            axes[i].legend()
        axes[-1].set_xlabel("z/a")
        plt.tight_layout()
        plt.show()
                
        
        
        
    
    
class Spectra1:
    """
    Compute flux spectra from GENE-like data with global time-averaging,
    streaming, plotting (kx, ky, z-profiles), and optional Numba acceleration.
    """

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs
        self.outfile='flux_spectra.h5'

    # ---------------------------------------------------------------
    def sync_indices0(self, fld_reader, mom_readers, t_start, t_stop, params):
        """Synchronize field and moment indices using LCM of time steps."""
        istep_fld = int(params["in_out"]["istep_field"])
        istep_mom = int(params["in_out"]["istep_mom"])
        if istep_fld <= 0 or istep_mom <= 0:
            raise ValueError("istep_field and istep_mom must be positive integers")

        L = int(np.lcm(istep_fld, istep_mom))
        stride_fld = L // istep_fld
        stride_mom = L // istep_mom

        times_fld = fld_reader.read_all_times()
        mask_fld = (times_fld >= t_start) & (times_fld <= t_stop)
        idx_fld = np.where(mask_fld)[0][::stride_fld]

        times_mom = mom_readers[0].read_all_times()
        mask_mom = (times_mom >= t_start) & (times_mom <= t_stop)
        idx_mom = np.where(mask_mom)[0][::stride_mom]

        return idx_fld.tolist(), idx_mom.tolist()
    
    
    def sync_indices(self, fld_reader, mom_readers, t_start, t_stop, params):
        """
        Synchronize field & moment indices using the LCM stride.
        Then remove any timestep already present in the output .h5 file.

        Time values in the file are stored as float32.
        Tolerance is chosen based on float32 precision.
        """
        istep_fld = int(params["in_out"]["istep_field"])
        istep_mom = int(params["in_out"]["istep_mom"])
        if istep_fld <= 0 or istep_mom <= 0:
            raise ValueError("istep_field and istep_mom must be positive integers")

        # ---- LCM stride selection -----------------------------------------
        L = int(np.lcm(istep_fld, istep_mom))
        stride_fld = L // istep_fld
        stride_mom = L // istep_mom

        times_fld = fld_reader.read_all_times()
        mask_fld = (times_fld >= t_start) & (times_fld <= t_stop)
        idx_fld = np.where(mask_fld)[0][::stride_fld]

        times_mom = mom_readers[0].read_all_times()
        mask_mom = (times_mom >= t_start) & (times_mom <= t_stop)
        idx_mom = np.where(mask_mom)[0][::stride_mom]

        # ---- Load saved times (float32) -----------------------------------
        if os.path.exists(self.outfile):
            with h5py.File(self.outfile, "r") as f:
                saved_times = f["time"][...].astype(np.float32) if "time" in f else np.array([], dtype=np.float32)
        else:
            saved_times = np.array([], dtype=np.float32)

        # Nothing saved → return all candidate indices
        if saved_times.size == 0:
            return idx_fld.tolist(), idx_mom.tolist()

        # ---- Float32-aware comparison --------------------------------------
        def is_saved(t):
            # t is python float or float64 → cast to float32 to compare in same precision
            t32 = np.float32(t)
            # tolerance for float32: about 1e-6 relative
            tol = max(1e-6, abs(t32) * 1e-6)
            # vectorized check
            return np.any(np.abs(saved_times - t32) <= tol)

        idx_fld_filtered = [i for i in idx_fld if not is_saved(times_fld[i])]
        idx_mom_filtered = [i for i in idx_mom if not is_saved(times_mom[i])]

        return idx_fld_filtered, idx_mom_filtered


    # ---------------------------------------------------------------
    def compute_spectra(self, fields, moments, coord, geom, params):
        """Compute ES/EM flux spectra for all species."""
        n_species = len(moments)
        n_fields = params["info"]["n_fields"]

        # Compute vExB, B_x, dBpar/dy with optional numba
        ky = coord['ky']
        if NUMBA_AVAILABLE:
            vE_x_conj, B_x_conj, dBpar_dy_conj = self._compute_vExB_numba(ky, fields, geom['Bfield'], n_fields)
        else:
            vE_x_conj, B_x_conj, dBpar_dy_conj = self._compute_vExB_numpy(ky, fields, geom['Bfield'], n_fields)

        results = []
        for i_sp in range(n_species):
            mom = moments[i_sp]
            spec = params["species"][i_sp]
            n0, T0, q0 = spec['dens'], spec['temp'], spec['charge']

            # Electrostatic
            G_es = self.averages(vE_x_conj * mom[0] * n0, geom)
            Q_es = self.averages(vE_x_conj * (0.5*mom[1] + mom[2] + 1.5*mom[0]) * n0*T0, geom)

            # Electromagnetic
            if n_fields > 1:
                tmp1 = B_x_conj * mom[5] * n0
                tmp2 = B_x_conj * (mom[3] + mom[4]) * n0*T0
                if n_fields > 2:
                    tmp1 += dBpar_dy_conj * mom[6] * n0*T0/q0
                    tmp2 += dBpar_dy_conj * (mom[7]+mom[8]) * n0*T0**2/q0
                G_em = self.averages(tmp1, geom)
                Q_em = self.averages(tmp2, geom)
            else:
                G_em, Q_em = (None,None,None), (None,None,None)

            results.append([Q_es, Q_em, G_es, G_em])
        return results

    # ---------------------------------------------------------------
    @staticmethod
    def _compute_vExB_numpy(ky, fields, Bfield, n_fields):
        """Compute vE_x, B_x, dBpar/dy using numpy."""
        vE_x = np.conj(-1j * ky[np.newaxis,:,np.newaxis] * fields[0])
        B_x = np.conj(1j * ky[np.newaxis,:,np.newaxis] * fields[1]) if n_fields>1 else None
        dBpar_dy = (-np.conj(1j * ky[np.newaxis,:,np.newaxis] * fields[2] / Bfield[np.newaxis,np.newaxis,:])
                     if n_fields>2 else None)
        return vE_x, B_x, dBpar_dy

    # ---------------------------------------------------------------
    if NUMBA_AVAILABLE:
        @staticmethod
        @njit(parallel=True)
        def _compute_vExB_numba(ky, fields, Bfield, n_fields):
            nx, ny, nz = fields[0].shape
            vE_x = np.empty((nx, ny, nz), dtype=np.complex128)
            B_x = np.empty_like(vE_x) if n_fields>1 else None
            dBpar_dy = np.empty_like(vE_x) if n_fields>2 else None
            for i in prange(nx):
                for j in range(ny):
                    for k in range(nz):
                        vE_x[i,j,k] = -1j * ky[j] * fields[0][i,j,k]
                        if n_fields>1:
                            B_x[i,j,k] = 1j * ky[j] * fields[1][i,j,k]
                        if n_fields>2:
                            dBpar_dy[i,j,k] = 1j * ky[j] * fields[2][i,j,k] / Bfield[k]
            return np.conj(vE_x), np.conj(B_x) if B_x is not None else None, -np.conj(dBpar_dy) if dBpar_dy is not None else None

    # ---------------------------------------------------------------
    @staticmethod
    def averages(flux, geom):
        """Weighted averages along z, summed over x and y."""
        if flux is None:
            return (None,None,None)
        flux[:,1:,:] *= 2
        J = geom['Jacobian'][np.newaxis,np.newaxis,:] / np.sum(geom['Jacobian'])
        sum_z = np.sum(flux.real * J, axis=(0,1))
        avg_z = np.sum(flux.real * J, axis=2)
        sp_ky = np.sum(avg_z, axis=0)
        tmp = np.sum(avg_z, axis=1)
        nx = tmp.shape[0]
        nx2 = nx//2+1
        sp_kx = np.zeros(nx2)
        sp_kx[0] = tmp[0]
        if nx>1:
            if nx%2==1:
                sp_kx[1:nx2] = tmp[1:nx2]+tmp[-1:nx2-1:-1]
            else:
                sp_kx[1:nx2-1] = tmp[1:nx2-1]+tmp[-1:nx2-1:-1]
                sp_kx[nx2-1] = tmp[nx2-1]
        return sp_kx, sp_ky, sum_z

    # ---------------------------------------------------------------
    def compute_flux_time(self, field_iters, mom_iters, coords, geoms, params_list, t_start, t_stop):
        """
        Compute **global time-weighted average** over multiple iterators.
        field_iters, mom_iters, coords, geoms are lists of same length.
        """
        fluxes_avg = None
        total_time = 0.0
        t_prev = None
        data_started = False
        species_names=[sp['name'] for sp in params_list[0]['species']]

        for fld_reader, mom_readers, coord, geom, params in zip(field_iters, mom_iters, coords, geoms, params_list):
            idx_fld, idx_mom = self.sync_indices(fld_reader, mom_readers, t_start, t_stop, params)
            if len(idx_fld)==0 or len(idx_mom)==0:
                continue  # skip this iterator entirely

            it_field = fld_reader.stream_selected(idx_fld)
            it_moms = [r.stream_selected(idx_mom) for r in mom_readers]

            for tm, fields in it_field:
                moments_data = [next(it)[1] for it in it_moms]
                print(f"time={tm:.4f}")
                fluxes = self.compute_spectra(fields, moments_data, coord, geom, params)

                if not data_started:
                    fluxes_avg = fluxes
                    t_prev = tm
                    data_started = True
                    self.append_fluxes_piecewise(fluxes_avg, coord, species_names, tm)
                    continue

                dt = tm - t_prev
                if dt <= 0.0:
                    continue
                t_prev = tm

                # weighted running average
                for sp in range(len(fluxes)):
                    for i_flux in range(4):
                        f_new = fluxes[sp][i_flux]
                        f_old = fluxes_avg[sp][i_flux]
                        if f_new is not None:
                            fluxes_avg[sp][i_flux] = tuple(
                                (f_old[j]*total_time + f_new[j]*dt)/(total_time+dt)
                                for j in range(3)
                            )
                total_time += dt              
           
                self.append_fluxes_piecewise(fluxes_avg, coord, species_names, tm)

        if not data_started:
            return None  # no data processed

        
        return fluxes_avg
    
    # ---------------------------------------------------------------
    def plot(self, field_iters, mom_iters, params, coords, geoms,
             t_start, t_stop, log_mode="loglog", ky_split=None):
        """
        Compute global time-averaged fluxes and immediately plot spectra and z-profiles.
        """

        fluxes_avg = self.compute_flux_time(field_iters, mom_iters, coords, geoms, params.tolist(), t_start, t_stop)
        if fluxes_avg is None:
            print("No data to process in the selected time interval.")
            return
        
        species_names=[sp['name'] for sp in params.get(0)['species']]
        

        # Use first coord for kx/ky axes
        self.plot_fluxes(fluxes_avg, coords[0], species_names=species_names, log_mode=log_mode, ky_split=ky_split)
        # Use first coord for z-axis
        self.plot_z_profiles(fluxes_avg, coords[0], species_names=species_names)
    

    # ---------------------------------------------------------------
    @staticmethod
    def plot_fluxes(fluxes_avg, coord, species_names=None, log_mode="loglog", ky_split=None):
        """Plot kx, ky spectra per species with log/log or lin/log options."""
        k_axes = [coord['kx'], coord['ky']]
        row_scales = [("log","log"), ("linear","linear")] if log_mode=="loglog" else [("log","linear"),("linear","linear")]
        axis_titles = [r"$k_x\rho_{ref}$", r"$k_y\rho_{ref}$"]

        for sp_idx, flux_list in enumerate(fluxes_avg):
            name = species_names[sp_idx] if species_names else f"Species {sp_idx+1}"
            Q_es, Q_em, G_es, G_em = flux_list
            
            # inside plot_fluxes
            if ky_split is not None and Q_es[1] is not None:
                ky = coord['ky']
                idx_split = np.argmin(np.abs(ky - ky_split))
                Q_ky = np.zeros_like(ky)
                if Q_es[1] is not None: Q_ky += Q_es[1]
                if Q_em[1] is not None: Q_ky += Q_em[1]
                sum_below = np.sum(Q_ky[:idx_split])
                sum_above = np.sum(Q_ky[idx_split:])
                print(f"{name}: Q flux below ky={ky_split}: {sum_below:.4e} above: {sum_above:.4e}")

            fig, axes = plt.subplots(2,2,figsize=(12,8))
            axes = axes.flatten()

            for row,(xscale,yscale) in enumerate(row_scales):
                for col in range(2):
                    ax = axes[row*2+col]
                    k_axis = k_axes[col]

                    for flux,label in [(Q_es,r"$Q_{es}$"),(Q_em,r"$Q_{em}$"),(G_es,r"$\Gamma_{es}$"),(G_em,r"$\Gamma_{em}$")]:
                        if flux[col] is not None:
                            y = np.asarray(flux[col],dtype=float)
                            x = np.asarray(k_axis[:y.size],dtype=float)

                            if row==0 and log_mode=="linlog":
                                y = y*x

                            if row==0 and log_mode=="loglog":
                                mask = y>0
                                if not np.any(mask):
                                    continue
                                x_plot, y_plot = x[mask], y[mask]
                            else:
                                x_plot, y_plot = x, y

                            ax.plot(x_plot, y_plot, label=label)

                    ax.set_xscale(xscale if row==0 else "linear")
                    ax.set_yscale(yscale if row==0 else "linear")
                    ax.set_xlabel(axis_titles[col])
                    ax.set_ylabel("Flux")
                    ax.set_title(name)
                    ax.legend()

            plt.tight_layout()
            plt.show()

    # ---------------------------------------------------------------
    @staticmethod
    def plot_z_profiles(fluxes_avg, coord, species_names=None):
        """Plot sum over x,y of flux*J as function of z for each species."""
        n_species = len(fluxes_avg)
        z = coord["z"]

        fig, axes = plt.subplots(n_species,1,figsize=(8,3*n_species),sharex=True)
        if n_species==1:
            axes=[axes]

        for i_sp, flux_list in enumerate(fluxes_avg):
            name = species_names[i_sp] if species_names else f"Species {i_sp+1}"
            Q_es_z = flux_list[0][2]
            Q_em_z = flux_list[1][2] if flux_list[1][2] is not None else None
            G_es_z = flux_list[2][2]
            G_em_z = flux_list[3][2] if flux_list[3][2] is not None else None

            ax = axes[i_sp]
            if Q_es_z is not None: ax.plot(z, Q_es_z, label=r"$Q_{es}$")
            if Q_em_z is not None: ax.plot(z, Q_em_z, label=r"$Q_{em}$")
            if G_es_z is not None: ax.plot(z, G_es_z, label=r"$\Gamma_{es}$")
            if G_em_z is not None: ax.plot(z, G_em_z, label=r"$\Gamma_{em}$")

            ax.set_ylabel(r"Flux")
            ax.set_title(name)
            ax.legend()

        axes[-1].set_xlabel("z/a")
        plt.tight_layout()
        plt.show()
        

    def append_fluxes_piecewise(self, fluxes, coords, species_names, time_value):
        """
        Save/append flux data in *separate datasets* per species per flux part:

            <species>_<Q/G>_<es/em>_{kx,ky,z}

        Each dataset is 1D and grows with time.

        'time' dataset also grows.
        """

        kx = np.asarray(coords["kx"])
        ky = np.asarray(coords["ky"])
        z  = np.asarray(coords["z"])

        file_exists = os.path.exists(self.outfile)

        with h5py.File(self.outfile, "a") as f:

            # ---------------------------------------------------------
            # File creation: write coordinates, create time dataset and all flux datasets
            # ---------------------------------------------------------
            if not file_exists:
                # Axes
                f.create_dataset("kx", data=kx)
                f.create_dataset("ky", data=ky)
                f.create_dataset("z",  data=z)

                # Time dataset (growing)
                f.create_dataset("time", data=np.array([time_value], float),
                                 maxshape=(None,))

                # Create the empty datasets for flux pieces
                for i_sp, name in enumerate(species_names):

                    Q_es, Q_em, G_es, G_em = fluxes[i_sp]

                    for label, flux in zip(
                            ["Q_es", "Q_em", "G_es", "G_em"],
                            [Q_es,   Q_em,   G_es,   G_em]):

                        for axis_name, arr in zip(["kx", "ky", "z"], flux):
                            dsname = f"{name}_{label}_{axis_name}"

                            if arr is None:
                                # Make empty dataset with no maxshape
                                f.create_dataset(dsname, data=np.zeros((1, 0)),
                                                 maxshape=(None, 0))
                            else:
                                arr = np.asarray(arr, float)
                                f.create_dataset(
                                    dsname,
                                    data=arr[np.newaxis, :],   # first row
                                    maxshape=(None, arr.size)
                                )

                print(f"[HDF5] Created new streaming file: {self.outfile}")
                return

            # ---------------------------------------------------------
            # APPEND MODE
            # ---------------------------------------------------------
            for i_sp, name in enumerate(species_names):
                Q_es, Q_em, G_es, G_em = fluxes[i_sp]

                for label, flux in zip(
                        ["Q_es", "Q_em", "G_es", "G_em"],
                        [Q_es,   Q_em,   G_es,   G_em]):

                    for axis_name, arr in zip(["kx", "ky", "z"], flux):

                        dsname = f"{name}_{label}_{axis_name}"
                        dset = f[dsname]

                        if arr is None:
                            # Embedded empty dataset: just append a row of size 0
                            dset.resize((dset.shape[0] + 1, 0))
                            continue

                        arr = np.asarray(arr, float)
                        # Resize to add a new row
                        dset.resize((dset.shape[0] + 1, dset.shape[1]))
                        dset[-1, :] = arr
                        
            tds = f["time"]
            tds.resize((tds.shape[0] + 1,))
            tds[-1] = np.float32(time_value)          


class Spectra0:
    """
    Compute flux spectra from GENE-like data with global time-averaging,
    streaming, and plotting (kx, ky, and z-profiles).
    """

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs  # reserved for future parallelization

    # ---------------------------------------------------------------
    def sync_indices(self, fld_reader, mom_readers, t_start, t_stop, params):
        """
        Compute indices for field and moment readers by LCM of time steps.
        Returns lists of indices (may be empty).
        """
        istep_fld = int(params["in_out"]["istep_field"])
        istep_mom = int(params["in_out"]["istep_mom"])
        if istep_fld <= 0 or istep_mom <= 0:
            raise ValueError("istep_field and istep_mom must be positive integers")

        L = int(np.lcm(istep_fld, istep_mom))
        stride_fld = L // istep_fld
        stride_mom = L // istep_mom

        times_fld = fld_reader.read_all_times()
        mask_fld = (times_fld >= t_start) & (times_fld <= t_stop)
        idx_fld = np.where(mask_fld)[0][::stride_fld]

        times_mom = mom_readers[0].read_all_times()
        mask_mom = (times_mom >= t_start) & (times_mom <= t_stop)
        idx_mom = np.where(mask_mom)[0][::stride_mom]

        return idx_fld.tolist(), idx_mom.tolist()

    # ---------------------------------------------------------------
    def compute_spectra(self, fields, moments, coord, geom, params):
        """
        Compute ES/EM flux spectra for all species.
        Returns list of [Q_es, Q_em, G_es, G_em] per species,
        each as tuple (sp_kx, sp_ky, sum_z).
        """
        n_species = len(moments)
        n_fields = params["info"]["n_fields"]

        ky = coord['ky'][np.newaxis, :, np.newaxis]          # (1, ny, 1)
        inv_B = 1 / geom['Bfield'][np.newaxis, np.newaxis, :]  # (1,1,nz)

        vE_x_conj = np.conj(-1j * ky * fields[0])
        B_x_conj = np.conj(1j * ky * fields[1]) if n_fields > 1 else None
        dBpar_dy_conj = (-np.conj(1j * ky * fields[2] * inv_B) if n_fields > 2 else None)

        def compute_for_species(i_sp):
            mom = moments[i_sp]
            spec = params["species"][i_sp]
            n0, T0, q0 = spec['dens'], spec['temp'], spec['charge']

            # Electrostatic
            G_es = self.averages(vE_x_conj * mom[0] * n0, geom)
            Q_es = self.averages(vE_x_conj * (0.5*mom[1] + mom[2] + 1.5*mom[0]) * n0*T0, geom)

            # Electromagnetic
            if n_fields > 1:
                tmp1 = B_x_conj * mom[5] * n0
                tmp2 = B_x_conj * (mom[3] + mom[4]) * n0 * T0
                if n_fields > 2:
                    tmp1 += dBpar_dy_conj * mom[6] * n0*T0/q0
                    tmp2 += dBpar_dy_conj * (mom[7]+mom[8]) * n0*T0**2/q0
                G_em = self.averages(tmp1, geom)
                Q_em = self.averages(tmp2, geom)
            else:
                G_em, Q_em = (None,None,None), (None,None,None)

            return [Q_es, Q_em, G_es, G_em]

        return [compute_for_species(i) for i in range(n_species)]

    # ---------------------------------------------------------------
    @staticmethod
    def averages(flux, geom):
        """
        Weighted averages along z, then summed over x and y.
        Returns (sp_kx, sp_ky, sum_z)
        sum_z = sum_x,y(flux*J)
        """
        if flux is None:
            return (None,None,None)

        flux[:,1:,:] *= 2  # double positive ky modes
        J = geom['Jacobian'][np.newaxis, np.newaxis, :] / np.sum(geom['Jacobian'])

        sum_z = np.sum(flux.real * J, axis=(0,1))  # sum over x,y
        avg_z = np.sum(flux.real * J, axis=2)      # average along z

        sp_ky = np.sum(avg_z, axis=0)

        tmp = np.sum(avg_z, axis=1)
        nx = tmp.shape[0]
        nx2 = nx//2+1
        sp_kx = np.zeros(nx2)
        sp_kx[0] = tmp[0]
        if nx>1:
            if nx%2==1:
                sp_kx[1:nx2] = tmp[1:nx2]+tmp[-1:nx2-1:-1]
            else:
                sp_kx[1:nx2-1] = tmp[1:nx2-1]+tmp[-1:nx2-1:-1]
                sp_kx[nx2-1] = tmp[nx2-1]

        return sp_kx, sp_ky, sum_z



