"""
This file is used to compute and save the branching fractions for the MeV- and
GeV-scale RH neutrino models.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from rich.progress import Progress
from utils import write_data_to_h5py

from blackthorn import Gen, RhNeutrinoMeV, RhNeutrinoGeV

RealArray = npt.NDArray[np.float64]

RESULTS_DIR = Path(__file__).parent.joinpath("results")
KEY_GENS = [("fst", Gen.Fst), ("snd", Gen.Snd), ("trd", Gen.Trd)]


def __generate_branching_ratio_dict(
    mns: RealArray, theta: float, gen: Gen, mev: bool, progress, task
):
    """
    Create dictionary of branching fractions
    """
    RhNeutrino = RhNeutrinoMeV if mev else RhNeutrinoGeV

    # Create dummy model to get all the keys
    keys = RhNeutrino(mns[-1], theta, gen).branching_fractions().keys()

    brs = {key: np.zeros_like(mns) for key in keys}
    for i, mn in enumerate(mns):
        model = RhNeutrino(mn, theta, gen)
        for key, val in model.branching_fractions().items():
            brs[key][i] = val
        progress.update(task, advance=1)

    return {
        "masses": mns,
        "branching_fractions": brs,
    }


def generate_br_dict_mev(mns: RealArray, theta: float, gen: Gen, progress, task):
    return __generate_branching_ratio_dict(mns, theta, gen, True, progress, task)


def generate_br_dict_gev(mns: RealArray, theta: float, gen: Gen, progress, task):
    return __generate_branching_ratio_dict(mns, theta, gen, False, progress, task)


if __name__ == "__main__":

    dataset = {
        "mev": {
            "fst": {},
            "snd": {},
            "trd": {},
        },
        "gev": {
            "fst": {},
            "snd": {},
            "trd": {},
        },
    }

    theta = 1e-3
    npts = 250

    with Progress() as progress:
        mns_mev = np.geomspace(1e-4, 0.5, npts)
        mns_gev = np.geomspace(5.0, 1e3, npts)
        task_mev = progress.add_task("[red]MeV...", total=3 * npts)
        task_gev = progress.add_task("[cyan]GeV...", total=3 * npts)

        for kgen, gen in KEY_GENS:
            dataset["mev"][kgen] = generate_br_dict_mev(
                mns_mev, theta, gen, progress, task_mev
            )
            dataset["gev"][kgen] = generate_br_dict_gev(
                mns_gev, theta, gen, progress, task_gev
            )

    write_data_to_h5py(RESULTS_DIR, "brs2", dataset, overwrite=True)
