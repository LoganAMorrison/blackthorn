import warnings
import pathlib
import argparse

import pandas as pd
import numpy as np
from rich.progress import Progress


from ..constants import Gen
from .constrainers import RhNeutrinoConstrainer

RESULTS_DIR = pathlib.Path(__file__).parent.absolute().joinpath("..", "results")


def constrain_ann(mx, fname):
    constrainer = RhNeutrinoConstrainer()
    mns = np.geomspace(1e-6, mx / 2.0, 250)

    constraints_ = dict()
    constraints_["masses"] = mns
    for key, gen in [("fst", Gen.Fst), ("snd", Gen.Snd), ("trd", Gen.Trd)]:
        with Progress() as progress:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                constraints_[key] = constrainer.constrain(
                    mns,
                    mx=1.0,
                    theta=1e-3,
                    gen=gen,
                    progress=progress,
                    sigma=2.0,
                    method="chi2",
                )

    ann_constraints_low = dict()
    for key, gen in [("fst", Gen.Fst), ("snd", Gen.Snd), ("trd", Gen.Trd)]:
        for k, v in constraints_[key].items():
            ann_constraints_low[f"{k}_{key}"] = v

    ann_constraints_low["masses"] = mns
    del constraints_

    df = pd.DataFrame(ann_constraints_low)
    df.to_csv(RESULTS_DIR.joinpath(fname))


def constrain_dec(fname):
    constrainer = RhNeutrinoConstrainer()
    mns = np.geomspace(1e-6, 1e8, 250)

    constraints_ = dict()
    constraints_["masses"] = mns
    for key, gen in [("fst", Gen.Fst), ("snd", Gen.Snd), ("trd", Gen.Trd)]:
        with Progress() as progress:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                constraints_[key] = constrainer.constrain(
                    mns,
                    theta=1e-3,
                    gen=gen,
                    progress=progress,
                    sigma=2.0,
                    method="chi2",
                )

    constraints = dict()
    for key, gen in [("fst", Gen.Fst), ("snd", Gen.Snd), ("trd", Gen.Trd)]:
        for k, v in constraints_[key].items():
            constraints[f"{k}_{key}"] = v

    constraints["masses"] = mns
    del constraints_

    df = pd.DataFrame(constraints)
    df.to_csv(RESULTS_DIR.joinpath(fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="constrain",
        description="Compute constraints",
    )
    parser.add_argument("mx", help="Dark matter mass in GeV")
    parser.add_argument(
        "min_mass", help="Minimum RH neutrino mass in GeV", type=float, default=1e-6
    )
    parser.add_argument(
        "max_mass", help="Maximum RH neutrino mass in GeV", type=float, default=1e8
    )
    parser.add_argument(
        "num_masses",
        help="Number of RH neutrino masses",
        type=int,
        default=250,
    )

    parser.parse_args()
