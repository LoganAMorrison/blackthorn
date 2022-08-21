"""
Script used to compute and save the spectra for the MeV- and
GeV-scale RH neutrino models.
"""

from pathlib import Path
import json
import argparse
from typing import List, Union, Optional

import numpy as np
import numpy.typing as npt
from rich.progress import Progress, TaskID
from utils import write_data_to_h5py

from blackthorn import Gen, RhNeutrinoMeV, RhNeutrinoGeV, RhNeutrinoTeV

RealArray = npt.NDArray[np.float64]

GeV = 1.0
MeV = 1e-3
keV = 1e-6

ModelType = Union[RhNeutrinoMeV, RhNeutrinoGeV, RhNeutrinoTeV]

GENERATIONS = [Gen.Fst, Gen.Snd, Gen.Trd]


LOW_MASSES = [10e-6, 2e-3, 10e-3, 100e-3, 150e-3, 250e-3]
MID_MASSES = [10e-6, 100e-3, 500e-3, 50.0, 150.0, 500.0]
HIGH_MASSES = [10.0, 50.0, 300.0, 1e3, 1e5, 1e7]


def generate_spectrum(
    xs: RealArray,
    mn: float,
    theta: float,
    gen: Gen,
    eps: float,
    mx: float,
):
    if mn < 1.0:
        RhNeutrino = RhNeutrinoMeV
    elif mn <= 1e3:
        RhNeutrino = RhNeutrinoGeV
    else:
        RhNeutrino = RhNeutrinoTeV

    model = RhNeutrino(mn, theta, gen)
    beta = np.sqrt(1.0 - (2 * mn / mx) ** 2)
    if mn == mx / 2.0:
        return model.dndx_photon(xs).convolve(eps)(xs)
    else:
        return model.dndx_photon(xs).boost(beta).convolve(eps)(xs)


class BranchingFractions:
    def __init__(self, config_file):
        with open(config_file, "r") as f:
            config = json.load(f)

        self.dataset = {}
        self.tasks = {}
        self._validate_and_add(config)

    def _validate_and_add(self, config):
        if not isinstance(config, dict):
            raise ValueError()

        outfile = config.get("outfile", None)
        datasets = config.get("datasets", None)

        if outfile is None:
            raise ValueError("Configuration file must contain 'outfile'.")
        elif not isinstance(outfile, str):
            raise ValueError("'outfile' must be a string.")

        if datasets is None:
            raise ValueError("Configuration file must contain 'datasets'.")
        elif not isinstance(datasets, list):
            raise ValueError("'datasets' must be a list.")

        for dset in datasets:
            self._validate_dataset_and_add(dset)

    def _validate_dataset_and_add(self, dataset, path: List = []):
        if len(path) > 0:
            msg = "/".join(path)
            info = f"from '{msg}' "
        else:
            info = " "

        if not isinstance(dataset, dict):
            raise ValueError(f"Invalid entry: Dataset {info}must be an object.")

        name = dataset.get("name", None)
        if name is None:
            raise ValueError(f"Missing entry: Dataset {info}missing 'name'.")
        else:
            subpath = [*path, name]

        if "datasets" in dataset:
            if not isinstance(dataset["datasets"], list):
                raise ValueError("Invalid entry: 'datasets' must be a list.")
            for sdset in dataset["datasets"]:
                self._validate_dataset_and_add(sdset, path=subpath)
        else:
            self._validate_entry(dataset, "min-mass", float, path=subpath)
            self._validate_entry(dataset, "max-mass", float, path=subpath)
            self._validate_entry(dataset, "num-mass", int, path=subpath)
            self._validate_entry(dataset, "gen", int, path=subpath)
            self._validate_entry(dataset, "theta", float, path=subpath)
            self._validate_entry(dataset, "catagorize", bool, path=subpath)
            self._validate_entry(dataset, "progress", bool, path=subpath)

            self._add_dataset(
                subpath,
                dataset["min-mass"],
                dataset["max-mass"],
                dataset["num-mass"],
                dataset["gen"],
                dataset["theta"],
                dataset["catagorize"],
                dataset["progress"],
            )

    def _validate_entry(self, dataset, name, type=None, path=None):
        if len(path) > 0:
            msg = "/".join(path)
            info = f"from '{msg}' "
        else:
            info = " "

        if name not in dataset:
            raise ValueError(f"Missing entry: Dataset {info}missing entry '{name}'.")
        if type is not None:
            if not isinstance(dataset[name], type):
                raise ValueError(
                    f"Invalid entry: Dataset {info}has invalid entry {name}."
                    + f" Entry must be {type}."
                )

    def _int_to_gen(self, i):
        if i == 1:
            return Gen.Fst
        if i == 2:
            return Gen.Snd
        if i == 3:
            return Gen.Trd
        raise ValueError("Invalid generation. Value must be 1, 2, or 3.")

    def _add_dataset(
        self, path, min_mass, max_mass, num_mass, gen, theta, catagorize, progress
    ):
        key = "/".join(path)
        masses = np.geomspace(min_mass, max_mass, num_mass)
        self.tasks[key] = progress
        self.dataset[key] = {
            "masses": masses,
            "gen": self._int_to_gen(gen),
            "theta": theta,
            "branching_fractions": dict(),
            "catagorize": catagorize,
        }

    def _generate(
        self, path, progress: Optional[Progress] = None, task: Optional[TaskID] = None
    ):
        masses = self.dataset[path]["masses"]
        min_mass = np.min(masses)
        max_mass = np.max(masses)

        if max_mass < 1.0:
            RhNeutrino = RhNeutrinoMeV
        elif min_mass >= 5.0:
            RhNeutrino = RhNeutrinoGeV
        else:
            raise ValueError("Masses must be between [0, 1.0] or [5.0, inf].")

        theta = self.dataset[path]["theta"]
        gen = self.dataset[path]["gen"]

        if progress is not None and task is not None:

            def update():
                progress.update(task, advance=1)

        else:

            def update():
                pass

        keys = RhNeutrino(masses[-1], theta, gen).branching_fractions().keys()
        brs = {key: np.zeros_like(masses) for key in keys}
        for i, mn in enumerate(masses):
            model = RhNeutrino(mn, theta, gen)
            values = model.branching_fractions()
            update()
            for key, val in values.items():
                brs[key][i] = val

        self.dataset[path]["branching_fractions"] = brs

    def generate(self):
        if len(self.tasks.keys()) > 0:
            with Progress() as progress:
                for path in self.dataset.keys():
                    if path in self.tasks:
                        total = len(self.dataset[path]["masses"])
                        task = progress.add_task(f"{path}...", total=total)
                    else:
                        task = None
                    self._generate(path, progress, task)
        else:
            for path in self.dataset.keys():
                self._generate(path)


def add_subprogram_spectrum(subparsers):
    # Branching fractions data generator
    spectrum = subparsers.add_parser("spectrum", help="Generate spectrum data.")
    spectrum.add_argument(
        "spec_config", help="Path to configuration file specifying all options."
    )


def add_subprogram_branching_fractions(subparsers):
    # Branching fractions data generator
    branching_fractions = subparsers.add_parser(
        "branching_fractions", help="Generate branching ratio data."
    )
    branching_fractions.add_argument(
        "bf_config", help="Path to configuration file specifying all options."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate", description="Generate data from the blackthorn package."
    )
    subparsers = parser.add_subparsers(help="subcommand help")

    add_subprogram_spectrum(subparsers)
    add_subprogram_branching_fractions(subparsers)

    args = parser.parse_args()

    if "bf_config" in args:
        print(args.bf_config)
        generator = BranchingFractions(args.bf_config)
        generator.generate()
    else:
        print("not ready.")
