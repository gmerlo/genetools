"""
params.py — GENE namelist parameter loader.

Reads one or more Fortran-90 namelist ``parameters`` files from a GENE run
directory, merges each file with a set of hard-coded defaults, and exposes
the result as plain Python dictionaries.

Typical usage
-------------
>>> from genetools.params import Params
>>> p = Params('/path/to/run/')          # single run
>>> d = p.get(0)                         # dict for first (only) file
>>> p.show()                             # pretty-print all loaded params

Multiple extensions (e.g. restart files)
----------------------------------------
>>> p = Params('/path/to/run/', extensions=['_0001', '_0002'])
>>> d0, d1 = p.tolist()
"""

from pathlib import Path
import copy
import f90nml


class Params:
    """
    Load and merge GENE simulation parameters from namelist files.

    The loader applies a set of physics defaults (precision, geometry flags,
    unit conversions, etc.) and then overlays the values found in each file.
    Derived unit quantities (``cref``, ``rhoref``, …) are computed once from
    the defaults and updated if the file overrides any base unit.

    Parameters
    ----------
    folder_or_file : str or path-like
        Path to a run folder (all ``parameters*`` files matching *extensions*
        are loaded) or a single parameter file.
    extensions : str or list of str, optional
        File-name suffix(es) to match, e.g. ``'_0001'`` or
        ``['_0001', '_0002']``.  Ignored when *folder_or_file* is a file.
        Defaults to ``['']`` (matches ``parameters`` with no suffix).
    """

    # ------------------------------------------------------------------
    # Hard-coded physics defaults
    # ------------------------------------------------------------------
    _DEFAULTS = {
        "info": {
            "precision": "DOUBLE",
            "endiannes": "LITTLE",
        },
        "general": {
            "x_local": True,
            "y_local": True,
            "diag_trap_levels": 0,
            "Bpar": False,
            "collision_op": "none",
            "coll": 0,
            "tau": 1,
            "zeff": 1,
            "arakawa_zv": True,
            "include_f0_contr": False,
            "trap_pass": False,
        },
        "in_out": {
            "write_std": True,
            "write_h5": False,
            "write_bp": False,
        },
        "nonlocal_x": {
            "ck_heat": 0,
            "ck_part": 0,
            "reset_limit": 0,
            "l_buff": 0,
            "u_buff": 0,
        },
        "geometry": {
            "minor_r": None,
            "n_pol": 1,
            "edge_opt": 0,
            "sign_Ip_CW": 1,
            "sign_Bt_CW": 1,
            "sign_Omega_CW": 1,
        },
        "units": {
            "Lref": 1.0,
            "Bref": 1.0,
            "nref": 1.0,
            "Tref": 1.0,
            "mref": 1.0,
            "omegatorref": 0.0,
            "qe": 1.602e-19,
            "mp": 1.6726219e-27,
        },
        "box": {
            "lx": 0,
            "mu_grid_type": "gau_lag",
            "ky0_ind": 0,
            "n_spec": 1,
        },
        "species": {
            "charge": 1.0,
            "mass": 1.0,
            "temp": 1.0,
            "src_prof_type": 0,
            "src_amp": 0.0,
        },
        "bsgrid": {
            "is_bsg": False,
        },
    }

    def __init__(self, folder_or_file, extensions=None):
        self.folder_or_file = Path(folder_or_file)
        self.extensions = extensions
        self.param_files = []   # list of Path objects
        self.params_list = []   # list of dicts, one per file

        self.param_files = self._find_files()
        for f in self.param_files:
            self.params_list.append(self._load_file(f))

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _find_files(self) -> list:
        """Return a sorted list of parameter file paths to load."""
        p = self.folder_or_file
        if p.is_file():
            return [p]
        if p.is_dir():
            exts = (
                [self.extensions]
                if isinstance(self.extensions, str)
                else (self.extensions or [""])
            )
            files = []
            for ext in exts:
                pattern = f"parameters*{ext}"
                matched = sorted(p.glob(pattern))
                if not matched:
                    raise FileNotFoundError(
                        f"No parameter files found matching '{pattern}' in '{p}'"
                    )
                files.extend(matched)
            return files
        raise FileNotFoundError(f"Path not found: {p}")

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _load_file(self, filepath: Path) -> dict:
        """
        Parse one Fortran namelist parameter file and merge with defaults.

        Lines beginning with ``FC`` (GENE's special continuation marker) are
        commented out before parsing so that ``f90nml`` does not choke on them.

        Parameters
        ----------
        filepath : Path
            Absolute path to the parameter file.

        Returns
        -------
        dict
            Nested dictionary of simulation parameters with defaults applied.
        """
        content = filepath.read_text()
        # Comment out GENE-specific "FC" continuation lines
        content = "\n".join(
            line if not line.startswith("FC") else "! " + line[3:]
            for line in content.splitlines()
        )

        nml = f90nml.reads(content)
        param_dict = {}

        for group_name, values in nml.items():
            if group_name not in self._DEFAULTS and group_name != "species":
                # Unknown group — pass through as-is
                param_dict[group_name] = copy.deepcopy(values)
                continue

            default = copy.deepcopy(self._DEFAULTS.get(group_name, {}))
            for k, v in values.items():
                default[k] = v

            if group_name == "species":
                param_dict.setdefault("species", []).append(default)
            else:
                param_dict[group_name] = default

        # Compute derived unit quantities from the (possibly updated) units block
        if "units" in param_dict:
            self._compute_derived_units(param_dict["units"])

        return param_dict

    @staticmethod
    def _compute_derived_units(u: dict) -> None:
        """
        Add derived reference quantities to the *units* dict in-place.

        Computed quantities
        -------------------
        cref      : thermal velocity  ``sqrt(T_ref * e / (m_ref * m_p))``  [m/s]
        Oref      : cyclotron frequency ``q_e * B_ref / (m_ref * m_p)``    [rad/s]
        pref      : pressure  ``n_ref * T_ref``                             [Pa]
        rhoref    : Larmor radius ``c_ref / Omega_ref``                     [m]
        rho_starref : ``rho_ref / L_ref``                                   [–]
        Ggb       : gyro-Bohm particle flux normalisation                   [m^-2 s^-1]
        Qgb       : gyro-Bohm heat flux normalisation                       [W m^-2]
        Pgb       : gyro-Bohm momentum flux normalisation                   [kg m^-1 s^-2]
        """
        qe = u["qe"]
        mp = u["mp"]
        u["cref"] = (u["Tref"] * 1e3 * qe / (u["mref"] * mp)) ** 0.5
        u["Oref"] = qe * u["Bref"] / (u["mref"] * mp)
        u["pref"] = u["nref"] * 1e19 * u["Tref"] * 1e3 * qe
        u["rhoref"] = u["cref"] / u["Oref"]
        u["rho_starref"] = u["rhoref"] / u["Lref"]
        u["Ggb"] = u["cref"] * u["nref"] * u["rho_starref"] ** 2
        u["Qgb"] = u["cref"] * u["pref"] * u["rho_starref"] ** 2
        u["Pgb"] = u["cref"] ** 2 * u["mref"] * u["nref"] * u["rho_starref"] ** 2

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, index: int = 0) -> dict:
        """
        Return the parameter dictionary for file *index*.

        Parameters
        ----------
        index : int, optional
            Zero-based index into the list of loaded files (default ``0``).
        """
        return self.params_list[index]

    def tolist(self) -> list:
        """Return the full list of parameter dictionaries (one per file)."""
        return self.params_list

    def show(self) -> None:
        """Pretty-print all loaded parameter dictionaries to stdout."""
        for i, p in enumerate(self.params_list):
            print(f"\n=== Parameter set {i} (from {self.param_files[i]}) ===")
            self._print_dict(p)

    def _print_dict(self, d: dict, indent: int = 0) -> None:
        """Recursively pretty-print a nested dict."""
        for k, v in d.items():
            if isinstance(v, dict):
                print(" " * indent + f"{k}:")
                self._print_dict(v, indent + 4)
            elif isinstance(v, list):
                print(" " * indent + f"{k}: [")
                for item in v:
                    if isinstance(item, dict):
                        self._print_dict(item, indent + 4)
                        print(" " * (indent + 4) + "---")
                    else:
                        print(" " * (indent + 4) + str(item))
                print(" " * indent + "]")
            else:
                print(" " * indent + f"{k}: {v}")
