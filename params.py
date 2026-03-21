from pathlib import Path
import f90nml
import copy

class Params:
    def __init__(self, folder_or_file, extensions=None):
        """
        Load parameters from a folder or single file.
        - folder_or_file: folder path or file path
        - extensions: string or list of extensions to match; one dict per extension/file
        """
        self.folder_or_file = Path(folder_or_file)
        self.extensions = extensions
        self.param_files = []  # list of Path objects
        self.params_list = []  # list of dictionaries, one per file/extension

        # Hardcoded defaults for a single parameter dictionary
        self._defaults = {
            "info": {"precision": "DOUBLE",
                     "endiannes": "LITTLE"},
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
                "trap_pass": False
            },
            "in_out": {"write_std": True,
                       "write_h5": False,
                       "write_bp": False},
            "nonlocal_x": {"ck_heat": 0,
                           "ck_part": 0,
                           "reset_limit": 0,
                           "l_buff": 0,
                           "u_buff": 0},
            "geometry": {"minor_r": None,
                         "n_pol": 1,
                         "edge_opt": 0,
                         "sign_Ip_CW": 1,
                         "sign_Bt_CW": 1,
                         "sign_Omega_CW": 1},
            "units": {"Lref": 1.0,
                      "Bref": 1.0,
                      "nref": 1.0,
                      "Tref": 1.0,
                      "mref": 1.0,
                      "omegatorref": 0.0,
                      "qe": 1.602e-19, "mp":
                      1.6726219e-27},
            "box": {"lx": 0,
                    "mu_grid_type": "gau_lag",
                    "ky0_ind": 0,
                    "n_spec": 1},
            "species": {"charge": 1.0,
                         "mass": 1.0,
                         "temp": 1.0,
                         "src_prof_type": 0,
                         "src_amp": 0.0},
            "bsgrid": {"is_bsg": False}
        }

        # Compute derived units fields
        u = self._defaults["units"]
        u["cref"] = (u["Tref"]*1e3*u["qe"]/(u["mref"]*u["mp"]))**0.5
        u["Oref"] = u["qe"]*u["Bref"]/(u["mref"]*u["mp"])
        u["pref"] = u["nref"]*1e19*u["Tref"]*1e3*u["qe"]
        u["rhoref"] = u["cref"]/u["Oref"]
        u["rho_starref"] = u["rhoref"]/u["Lref"]
        u["Ggb"] = u["cref"]*u["nref"]*u["rho_starref"]**2
        u["Qgb"] = u["cref"]*u["pref"]*u["rho_starref"]**2
        u["Pgb"] = u["cref"]**2*u["mref"]*u["nref"]*u["rho_starref"]**2

        # Find files to load
        self.param_files = self._find_files()

        # Load each file/extension into a separate dictionary
        for f in self.param_files:
            param_dict = self._load_file(f)
            self.params_list.append(param_dict)

    def _find_files(self):
        if self.folder_or_file.is_dir():
            exts = [self.extensions] if isinstance(self.extensions, str) else self.extensions or [""]
            files = []
            for ext in exts:
                pattern = f"parameters*{ext}"
                files.extend(sorted(self.folder_or_file.glob(pattern)))
            if not files:
                raise FileNotFoundError(f"No parameter files found with pattern {pattern}")
            return files
        elif self.folder_or_file.is_file():
            return [self.folder_or_file]
        else:
            raise FileNotFoundError(f"{self.folder_or_file} not found")

  
    def _load_file(self, filepath):
        """Load one parameter file into a dictionary, merging with defaults."""

        # Read and preprocess file
        content = filepath.read_text()
        content = "\n".join(
            line if not line.startswith("FC") else "! " + line[3:]
            for line in content.splitlines()
        )

        # Parse namelist
        nml = f90nml.reads(content)

        param_dict={}
        
        # Merge namelist content
        for group_name, values in nml.items():

            # Case 1: group not in defaults → create new dict
            if group_name not in self._defaults and group_name is not "species":
                param_dict[group_name] = copy.deepcopy(values)
                continue

            # Case 2: group exists in defaults
            default = copy.deepcopy(self._defaults[group_name])

            for k, v in values.items():
                default[k] = v
                
            if group_name == "species":        
                if group_name in param_dict:
                    param_dict[group_name].append(default)
                else:
                    param_dict[group_name] = [default]
            else:
                param_dict[group_name] = default

        return param_dict
            
        
    def show(self):
        """Pretty print all loaded parameter dictionaries"""
        for i, p in enumerate(self.params_list):
            print(f"\n=== Parameter {i} (from {self.param_files[i]}) ===")
            self._print_dict(p)

    def _print_dict(self, d, indent=0):
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

    def get(self, index=0):
        """Return the dictionary for a given index"""
        return self.params_list[index]
    
    def tolist(self):
        """Return the dictionary for a given index"""
        return self.params_list

