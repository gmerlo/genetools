"""
utils.py — File-system utilities for GENE run directories.

Functions
---------
set_runs(folder, exclude=None)
    Scan a GENE run folder and return sorted run suffix strings.
"""

import re
from pathlib import Path


def set_runs(folder, exclude=None) -> list:
    """
    Scan a GENE run folder and return a sorted list of run suffix strings.

    GENE writes one set of output files per simulation segment, each
    identified by a 4-digit zero-padded suffix (e.g. ``_0001``, ``_0042``)
    or the special suffix ``'.dat'`` for the initial / unsuffixed run.

    Detection is based on ``nrg*`` files.  When GENE's HDF5 mode is active
    (``all_params*.h5`` files exist), every other ``nrg`` file is a
    duplicate and is skipped (stride 2).

    Parameters
    ----------
    folder : str or path-like
        Path to the GENE run directory.
    exclude : list of str, optional
        Suffix strings to omit from the result, e.g. ``['_0001', '.dat']``.

    Returns
    -------
    list of str
        Sorted suffix strings such as ``['_0001', '_0002', '_0040', '.dat']``.
        Numeric suffixes are sorted numerically; ``'.dat'`` is always last.

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist or contains no ``nrg*`` files.

    Examples
    --------
    >>> from genetools.utils import set_runs
    >>> set_runs('/path/to/run/')
    ['_0001', '_0002', '_0003', '.dat']
    >>> set_runs('/path/to/run/', exclude=['_0001'])
    ['_0002', '_0003', '.dat']
    """
    exclude = set(exclude or [])
    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder '{folder}' not found.")

    # HDF5 mode → every other file is a duplicate
    use_h5 = bool(list(folder.glob("all_params*.h5")))
    stride = 2 if use_h5 else 1

    nrg_files = sorted(folder.glob("nrg*"))
    if not nrg_files:
        raise FileNotFoundError(f"No 'nrg' files found in '{folder}'.")

    numeric_suffixes = []
    has_dat = False

    for nrg_file in nrg_files[::stride]:
        name = nrg_file.name
        if name.endswith(".dat"):
            has_dat = True
            continue
        m = re.search(r"(\d{4})$", name)
        if m:
            numeric_suffixes.append(int(m.group(1)))

    # Apply exclusions and sort
    suffix_list = [
        f"_{n:04d}"
        for n in sorted(numeric_suffixes)
        if f"_{n:04d}" not in exclude
    ]

    if has_dat and ".dat" not in exclude:
        suffix_list.append(".dat")

    return suffix_list
