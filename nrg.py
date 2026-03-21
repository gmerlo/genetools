import os
import glob
import numpy as np
import matplotlib.pyplot as plt

class NrgReader:
    """
    Reads 'nrg' files from a folder with repeated blocks of:
      - 1 scalar (time)
      - n_rows_per_block rows, each with n_cols_per_row floats
    Concatenates all blocks across files in ascending order of first scalar in each file.
    Reshapes data into (n_rows_per_block, n_cols_per_row, n_times)
    """

    def __init__(self, folder, params):
        self.folder = folder
        self.n_rows_per_block = params["box"]["n_spec"]
        self.n_cols_per_row = params["info"]["nrgcols"]
        self.nrg_files = self._detect_files()
        self.specnames = [d["name"] for d in params["species"]]
        self.times = None
        self.data = None

    def _detect_files(self):
        """Detect all nrg files in the folder"""
        pattern = os.path.join(self.folder, "nrg*")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No 'nrg' files found in {self.folder}")
        return files

    def read_all(self):
        """Read all files and concatenate blocks in ascending order of first time in each file"""
        file_blocks = []

        for fname in self.nrg_files:
            times_block, data_block = self._read_file(fname)
            file_blocks.append((times_block[0], times_block, data_block))

        # Sort files by first time of each file
        file_blocks.sort(key=lambda x: x[0])

        # Concatenate all blocks
        all_times_blocks = [b[1] for b in file_blocks]
        all_data_blocks = [b[2] for b in file_blocks]

        times_array = np.concatenate(all_times_blocks)  # 1D array: repeated times per row
        data_array_flat = np.concatenate(all_data_blocks, axis=0)  # (n_blocks*n_rows_per_block, n_cols_per_row)

        n_times = len(times_array) // self.n_rows_per_block
        # Reshape data into (n_rows_per_block, n_cols_per_row, n_times)
        data_array = data_array_flat.reshape((n_times, self.n_rows_per_block, self.n_cols_per_row))
        data_array = np.transpose(data_array, (1, 2, 0))  # (n_rows_per_block, n_cols_per_row, n_times)

        self.times = times_array[::self.n_rows_per_block]  # pick one per block
        self.data = data_array
        return self.times, self.data

    def _read_file(self, filepath):
        """
        Read a single nrg file with repeated blocks.
        Returns:
            times: 1D array of times (one per row)
            data: 2D array of shape (n_blocks * n_rows_per_block, n_cols_per_row)
        """
        times = []
        data_rows = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            # Read scalar (time)
            time_val = float(lines[i].strip())
            times.append(time_val)
            i += 1

            # Read next n_rows_per_block rows of n_cols_per_row floats
            block_numbers = []
            for _ in range(self.n_rows_per_block):
                if i >= len(lines):
                    raise ValueError(f"File {filepath} ends unexpectedly in a block.")
                row_floats = [float(x) for x in lines[i].strip().split()]
                if len(row_floats) != self.n_cols_per_row:
                    raise ValueError(
                        f"Expected {self.n_cols_per_row} floats per row, got {len(row_floats)} in {filepath}"
                    )
                block_numbers.append(row_floats)
                i += 1

            data_rows.extend(block_numbers)

        times_array = np.repeat(np.array(times), self.n_rows_per_block)
        data_array = np.array(data_rows)  # shape: (n_blocks*n_rows_per_block, n_cols_per_row)
        return times_array, data_array
    

    def plot_fluxes(self, first_row_titles=None):
        """
        Plot a grid of subplots:
        - Rows correspond to column pairs:
            1st row: columns 7 & 8
            2nd row: columns 4 & 5
            3rd row: columns 9 & 10
        - Columns correspond to each row per block
        - Solid line: first column in pair (blue)
        - Dashed line: second column in pair (magenta)
        """
        if self.times is None or self.data is None:
            raise ValueError("Data not loaded. Call read_all() first.")

        col_pairs = [(6, 7), (4, 5), (8, 9)] 
        col_pairs = col_pairs[0:2]
        n_rows = self.data.shape[0]

        ylabels = [r"$Q [Q_{GB}]$", r"$\Gamma [\Gamma_{GB}]$", r"$\Pi$  [\Pi_{GB}]$"]
        
        if first_row_titles is None:
            first_row_titles = [f"s" for s in self.specnames]
        if len(first_row_titles) != n_rows:
            raise ValueError("first_row_titles must be a list of n_rows strings")

        # Create 3xN grid with shared x-axis per column
        fig, axes = plt.subplots(len(col_pairs), n_rows, figsize=(4 * n_rows, 3 *len(col_pairs)),
                                 sharex='col', squeeze=False)

        for row_idx, (col1, col2) in enumerate(col_pairs):
            for block_row_idx in range(n_rows):
                ax = axes[row_idx, block_row_idx]
                # Plot first column (solid blue)
                ax.plot(self.times, self.data[block_row_idx, col1, :], color='b', linestyle='-')
                # Plot second column (dashed magenta)
                ax.plot(self.times, self.data[block_row_idx, col2, :], color='m', linestyle='--')
                ax.grid(True)

                # Set title only for first row
                if row_idx == 0:
                    ax.set_title(first_row_titles[block_row_idx])

                # Set y-axis label for first column of each row
                if block_row_idx == 0:
                    ax.set_ylabel(ylabels[row_idx])

                # Set x-axis label only for bottom row
                if row_idx == len(col_pairs) - 1:
                    ax.set_xlabel(r"$t~c_{ref}/L_{ref}$")

        plt.tight_layout()
        plt.show()


    def plot_fluctuations(self,lbls=None):
        """
        Plot each row of the data as a separate subplot.
        Each subplot shows the first four columns with distinct solid colors:
          - Column 1: blue
          - Column 2: magenta
          - Column 3: green
          - Column 4: red
        Adds a single legend for all subplots.
        """
        if self.times is None or self.data is None:
            raise ValueError("Data not loaded. Call read_all() first.")

        n_rows = self.data.shape[0]
        colors = ['b', 'm', 'g', 'r']
        labels = [r'$n$', r'$T_{\|}$', r'$T_{\perp}$', r'$u_{\|}$']

        fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3 * n_rows), sharex=True, squeeze=False)

        for row_idx in range(n_rows):
            ax = axes[row_idx, 0]
            for col_idx in range(4):
                ax.plot(self.times, self.data[row_idx, col_idx, :], color=colors[col_idx], linestyle='-', label=labels[col_idx] )
            if lbls is not None:
                ax.set_ylabel(lbls[row_idx])
            ax.grid(True)

        axes[-1, 0].set_xlabel(r"$t~c_{ref}/L_{ref}$")

        # Add a single legend above the first subplot
        axes[0, 0].legend(loc='upper right')

        plt.tight_layout()
        plt.show(self.specnames)
      
    
    def plot(self):
        if self.times is None or self.data is None:
            self.read_all()
        self.plot_fluxes(self.specnames)
        self.plot_fluctuations(self.specnames)
        



