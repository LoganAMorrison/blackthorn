from typing import Dict, Optional
from pathlib import Path
from matplotlib.ticker import LogLocator, NullFormatter

import h5py


def add_to_h5py_dataset(dataset, dictionary: Dict, path: Optional[str] = None):
    """Add a nested dictionary to the hdf5 dataset."""
    for k, v in dictionary.items():
        if path is not None:
            newpath = "/".join([path, k])
        else:
            newpath = f"{k}"
        if isinstance(v, dict):
            add_to_h5py_dataset(dataset, v, newpath)
        else:
            # at the bottom
            dataset.create_dataset(newpath, data=v)


def write_data_to_h5py(prefix: Path, filename: str, dataset: Dict, overwrite: bool):
    """Write the dataset to the file."""
    fname = Path(prefix, filename).with_suffix(".hdf5")
    if fname.exists and not overwrite:
        raise ValueError("File exists and overwrite is set to false.")
    with h5py.File(fname, "w") as f:
        add_to_h5py_dataset(f, dataset)


def configure_ticks(axis, minor_ticks="both"):
    axis.tick_params(axis="both", which="both", direction="in", width=0.9, labelsize=12)
    if minor_ticks in ["y", "both"]:
        axis.yaxis.set_minor_locator(
            LogLocator(base=10, subs=[i * 0.1 for i in range(1, 10)], numticks=100)
        )
        axis.yaxis.set_minor_formatter(NullFormatter())
    if minor_ticks in ["x", "both"]:
        axis.xaxis.set_minor_locator(
            LogLocator(base=10, subs=[i * 0.1 for i in range(1, 10)], numticks=100)
        )
        axis.xaxis.set_minor_formatter(NullFormatter())


def add_xy_grid(axis, alpha=0.5):
    axis.grid(True, axis="y", which="major", alpha=alpha)
    axis.grid(True, axis="x", which="major", alpha=alpha)
