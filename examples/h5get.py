#!/usr/bin/env python3
"""
Open an HDF5 file and extract one dataset.

Standalone: h5py + numpy only.

Usage:
    h5get.py FILE                    # list every dataset with shape and dtype
    h5get.py FILE DATASET            # print a summary of that dataset
    h5get.py FILE DATASET out.npy    # ... and save it
"""
import sys

import h5py
import numpy as np


def main(path, name=None, save=None):
    def show(key, obj):
        # visititems stops as soon as the callback returns non-None, so this
        # must fall off the end rather than return a value.
        if isinstance(obj, h5py.Dataset):
            print(f"{key:40s} {str(obj.shape):18s} {obj.dtype}")

    with h5py.File(path, "r") as f:
        if name is None:
            f.visititems(show)
            return None
        if name not in f:
            sys.exit(f"'{name}' not in {path} — run without a dataset name to list.")
        dset = f[name]
        data = dset[...]                      # read into memory
        attrs = dict(dset.attrs)

    print(f"{name}: shape={data.shape} dtype={data.dtype}")
    if attrs:
        print("attrs:", attrs)
    if np.issubdtype(data.dtype, np.number) and data.size:
        print(f"min={np.nanmin(data):.6g}  max={np.nanmax(data):.6g}  "
              f"mean={np.nanmean(data):.6g}")
    if save:
        np.save(save, data)
        print(f"saved {save}")
    return data


if __name__ == "__main__":
    main(*sys.argv[1:4])
