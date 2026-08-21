# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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

    Detection is based on ``nrg*`` files.  A run with HDF5 output writes both
    ``nrg<suffix>`` and ``nrg<suffix>.h5``; both name the same segment and are
    reported once.  Suffixes are returned exactly as they appear on disk, so
    non-standard widths (``_1``, as GENE-3D scans produce) survive.

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

    nrg_files = sorted(folder.glob("nrg*"))
    if not nrg_files:
        raise FileNotFoundError(f"No 'nrg' files found in '{folder}'.")

    # Suffix string -> sort key. Using a dict collapses the HDF5 twins
    # (``nrg_0001`` and ``nrg_0001.h5`` are one segment) without having to
    # guess at a stride, and keeping the literal suffix means a segment named
    # ``_1`` stays ``_1`` rather than being reformatted into a name that no
    # file on disk actually has.
    keys = {}
    for nrg_file in nrg_files:
        if not nrg_file.is_file():
            continue
        name = nrg_file.name
        if name.endswith(".h5"):
            name = name[: -len(".h5")]
        suffix = name[len("nrg"):]
        if suffix == ".dat":
            keys[suffix] = (1, 0)          # the unsuffixed run sorts last
            continue
        m = re.fullmatch(r"_(\d+)", suffix)
        if m:
            keys[suffix] = (0, int(m.group(1)))

    return [suffix for suffix in sorted(keys, key=keys.__getitem__)
            if suffix not in exclude]
