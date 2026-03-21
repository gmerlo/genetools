import os
import glob
import re


def set_runs(folder, exclude=None):
    """
    Scan a GENE folder and return sorted list of run suffix strings.

    Example return:
        ["0001", "0002", "0040", "0100", ".dat"]

    - Numeric suffixes sorted ascending
    - ".dat" always at end
    - Exclusion list applied to suffix strings
    """
    exclude = exclude or []

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder '{folder}' not found.")

    # Detect HDF5 mode → skip every other file
    use_h5 = bool(glob.glob(os.path.join(folder, "all_params*.h5")))
    stride = 2 if use_h5 else 1

    # Collect nrg files
    nrg_files = sorted(glob.glob(os.path.join(folder, "nrg*")))
    if not nrg_files:
        return []

    numeric_suffixes = []
    has_dat = False

    for i in range(0, len(nrg_files), stride):
        fname = os.path.basename(nrg_files[i])

        # .dat case → mark presence but sort later
        if fname.endswith(".dat"):
            has_dat = True
            continue

        # Extract 4-digit suffix
        m = re.search(r"(\d{4})$", fname)
        if m:
            numeric_suffixes.append(int(m.group(1)))

    # Apply exclusions (convert numeric excludes to ints if needed)
    cleaned = []
    for s in numeric_suffixes:
        s_str = f"{s:04d}"
        if s_str not in exclude:
            cleaned.append(s)

    numeric_suffixes = cleaned

    # Sort numerically then convert back to 4-digit zero padded
    suffix_list = [f"_{n:04d}" for n in sorted(numeric_suffixes)]

    # Add ".dat" last (unless excluded)
    if has_dat and ".dat" not in exclude:
        suffix_list.append(".dat")

    return suffix_list

