"""
nrg.py — Reader and plotter for GENE energy/flux diagnostic files (``nrg*``).

GENE writes ``nrg`` files with a repeating block structure::

    <time_value>
    <row_0: n_cols floats>
    <row_1: n_cols floats>
    ...
    <row_{n_spec-1}: n_cols floats>
    <time_value>
    ...

Each row corresponds to one plasma species and contains time-integrated
diagnostic quantities (particle number, parallel/perpendicular temperature,
parallel velocity, and flux quantities).

Column index reference (0-based)
---------------------------------
0  n        particle number / density fluctuation
1  T_par    parallel temperature fluctuation
2  T_perp   perpendicular temperature fluctuation
3  u_par    parallel velocity fluctuation
4  …
5  …
6  Q        heat flux (electrostatic)
7  Q        heat flux (electromagnetic)
8  Γ        particle flux (electrostatic)
9  Γ        particle flux (electromagnetic)

Usage
-----
>>> from genetools.nrg import NrgReader
>>> from genetools.params import Params
>>> params = Params('/run/').get()
>>> reader = NrgReader('/run/', params)
>>> times, data = reader.read_all()   # data shape: (n_spec, n_cols, n_times)
>>> reader.plot()
"""

import os
import glob

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Column index constants (0-based) — kept as named values so changes are
# made in exactly one place.
# ---------------------------------------------------------------------------

# Fluctuation quantity columns
_COL_N = 0
_COL_T_PAR = 1
_COL_T_PERP = 2
_COL_U_PAR = 3

# Flux quantity column pairs (col_electrostatic, col_electromagnetic)
_FLUX_HEAT = (6, 7)        # Heat flux Q
_FLUX_PARTICLE = (8, 9)    # Particle flux Γ

# Flux column pairs used for plotting (ordered: heat first, then particle)
_FLUX_COL_PAIRS = [_FLUX_HEAT, _FLUX_PARTICLE]

# y-axis labels for each flux row
_FLUX_YLABELS = [
    r"$Q\,[Q_{\rm GB}]$",
    r"$\Gamma\,[\Gamma_{\rm GB}]$",
]

# Column colours and labels for fluctuation plots
_FLUCTUATION_COLORS = ["b", "m", "g", "r"]
_FLUCTUATION_LABELS = [r"$n$", r"$T_{\|}$", r"$T_{\perp}$", r"$u_{\|}$"]
_FLUCTUATION_COLS = [_COL_N, _COL_T_PAR, _COL_T_PERP, _COL_U_PAR]


class NrgReader:
    """
    Read and visualise GENE energy diagnostic (``nrg``) files.

    Files are detected automatically from *folder* and sorted by the first
    simulation time they contain before concatenation.

    Parameters
    ----------
    folder : str
        Run directory containing ``nrg*`` files.
    params : dict
        Parameter dictionary for this run (from :class:`~genetools.params.Params`).

    Attributes
    ----------
    times : np.ndarray or None
        1-D array of simulation times after :meth:`read_all` is called.
    data : np.ndarray or None
        3-D array of shape ``(n_species, n_cols, n_times)`` after
        :meth:`read_all` is called.
    """

    def __init__(self, folder: str, params: dict):
        self.folder = folder
        self.n_rows_per_block: int = params["box"]["n_spec"]
        self.n_cols_per_row: int = params["info"]["nrgcols"]
        self.specnames: list = [d["name"] for d in params["species"]]
        self.nrg_files: list = self._detect_files()
        self.times = None
        self.data = None

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _detect_files(self) -> list:
        """Return a list of all ``nrg*`` files found in :attr:`folder`."""
        pattern = os.path.join(self.folder, "nrg*")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No 'nrg' files found in '{self.folder}'")
        return files

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read_all(self) -> tuple:
        """
        Read all ``nrg`` files, sort by first timestamp, and concatenate.

        Returns
        -------
        times : np.ndarray
            Shape ``(n_times,)``.
        data : np.ndarray
            Shape ``(n_species, n_cols, n_times)``.
        """
        file_blocks = []
        for fname in self.nrg_files:
            times_block, data_block = self._read_file(fname)
            file_blocks.append((times_block[0], times_block, data_block))

        # Sort files by their first time stamp
        file_blocks.sort(key=lambda x: x[0])

        times_array = np.concatenate([b[1] for b in file_blocks])
        data_flat = np.concatenate([b[2] for b in file_blocks], axis=0)

        n_times = len(times_array) // self.n_rows_per_block

        # Reshape to (n_times, n_species, n_cols), then transpose
        data_array = data_flat.reshape((n_times, self.n_rows_per_block, self.n_cols_per_row))
        data_array = np.transpose(data_array, (1, 2, 0))  # → (n_species, n_cols, n_times)

        self.times = times_array[:: self.n_rows_per_block]
        self.data = data_array
        return self.times, self.data

    def _read_file(self, filepath: str) -> tuple:
        """
        Parse a single ``nrg`` file using :func:`numpy.loadtxt` for speed.

        The file alternates between single-value time lines and multi-value
        data rows.  We read the entire file at once and split based on column
        count.

        Parameters
        ----------
        filepath : str
            Full path to the ``nrg`` file.

        Returns
        -------
        times_array : np.ndarray
            Shape ``(n_blocks * n_rows_per_block,)`` — time repeated per row.
        data_array : np.ndarray
            Shape ``(n_blocks * n_rows_per_block, n_cols_per_row)``.
        """
        # Read all lines and classify by token count.
        # Time lines have exactly 1 token; data lines have n_cols_per_row tokens.
        # Using explicit line iteration is the only reliable approach because
        # np.loadtxt rejects files where the column count changes between rows.
        time_values = []
        data_rows = []

        with open(filepath, "r") as fh:
            for lineno, line in enumerate(fh, 1):
                tokens = line.split()
                if not tokens:
                    continue  # skip blank lines
                if len(tokens) == 1:
                    time_values.append(float(tokens[0]))
                elif len(tokens) == self.n_cols_per_row:
                    data_rows.append([float(t) for t in tokens])
                else:
                    raise ValueError(
                        f"{filepath} line {lineno}: expected 1 or "
                        f"{self.n_cols_per_row} tokens, got {len(tokens)}"
                    )

        time_values = np.array(time_values, dtype=float)
        data_rows = np.array(data_rows, dtype=float)  # (n_blocks*n_spec, n_cols)

        # Validate
        expected_data_rows = len(time_values) * self.n_rows_per_block
        if len(data_rows) != expected_data_rows:
            raise ValueError(
                f"{filepath}: expected {expected_data_rows} data rows "
                f"but found {len(data_rows)}"
            )

        times_array = np.repeat(time_values, self.n_rows_per_block)
        return times_array, data_rows

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _require_data(self) -> None:
        """Raise if :meth:`read_all` has not been called yet."""
        if self.times is None or self.data is None:
            raise ValueError("No data loaded — call read_all() first.")

    def plot_fluxes(self, titles=None) -> None:
        """
        Plot heat and particle flux time traces for each species.

        Creates a grid of subplots with rows for each flux type (heat flux Q,
        particle flux Γ) and columns for each species.  Electrostatic
        contributions are plotted as solid blue lines; electromagnetic as
        dashed magenta lines.

        Parameters
        ----------
        titles : list of str, optional
            Column headers (one per species).  Defaults to :attr:`specnames`.
        """
        self._require_data()
        n_species = self.data.shape[0]

        if titles is None:
            titles = self.specnames
        if len(titles) != n_species:
            raise ValueError(
                f"Expected {n_species} titles, got {len(titles)}"
            )

        n_rows = len(_FLUX_COL_PAIRS)
        fig, axes = plt.subplots(
            n_rows, n_species,
            figsize=(4 * n_species, 3 * n_rows),
            sharex="col",
            squeeze=False,
        )

        for row_idx, (col_es, col_em) in enumerate(_FLUX_COL_PAIRS):
            for sp_idx in range(n_species):
                ax = axes[row_idx, sp_idx]
                ax.plot(self.times, self.data[sp_idx, col_es, :], color="b", linestyle="-",
                        label="ES")
                ax.plot(self.times, self.data[sp_idx, col_em, :], color="m", linestyle="--",
                        label="EM")
                ax.grid(True)
                if row_idx == 0:
                    ax.set_title(titles[sp_idx])
                    ax.legend(fontsize=8)
                if sp_idx == 0:
                    ax.set_ylabel(_FLUX_YLABELS[row_idx])
                if row_idx == n_rows - 1:
                    ax.set_xlabel(r"$t\;c_{\rm ref}/L_{\rm ref}$")

        plt.tight_layout()
        plt.show()

    def plot_fluctuations(self, ylabels=None) -> None:
        """
        Plot density and temperature fluctuation time traces for each species.

        Each species gets its own subplot showing n, T_∥, T_⊥, and u_∥.

        Parameters
        ----------
        ylabels : list of str, optional
            y-axis labels (one per species).  Defaults to :attr:`specnames`.
        """
        self._require_data()
        n_species = self.data.shape[0]

        fig, axes = plt.subplots(
            n_species, 1,
            figsize=(8, 3 * n_species),
            sharex=True,
            squeeze=False,
        )

        for sp_idx in range(n_species):
            ax = axes[sp_idx, 0]
            for col_idx, (col, color, label) in enumerate(
                zip(_FLUCTUATION_COLS, _FLUCTUATION_COLORS, _FLUCTUATION_LABELS)
            ):
                ax.plot(self.times, self.data[sp_idx, col, :],
                        color=color, linestyle="-", label=label)
            if ylabels is not None and sp_idx < len(ylabels):
                ax.set_ylabel(ylabels[sp_idx])
            ax.grid(True)

        axes[0, 0].legend(loc="upper right")
        axes[-1, 0].set_xlabel(r"$t\;c_{\rm ref}/L_{\rm ref}$")
        plt.tight_layout()
        plt.show()

    def plot(self) -> None:
        """
        Convenience method: load data (if not already loaded) and show all plots.

        Calls :meth:`read_all` if needed, then :meth:`plot_fluxes` and
        :meth:`plot_fluctuations` with species names as labels.
        """
        if self.times is None or self.data is None:
            self.read_all()
        self.plot_fluxes(self.specnames)
        self.plot_fluctuations(self.specnames)
