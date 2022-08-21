import argparse
import pathlib
import warnings
from typing import NamedTuple, Optional
import json

import numpy as np
from rich.progress import Progress

from blackthorn.constraints.constrainers import RhNeutrinoConstrainer
from blackthorn.constants import Gen


generate_constraints = argparse.ArgumentParser(
    prog="generate-constraints",
    description="Generate constraints from RH-neutrino decays "
    "or DM annihilations into RH-neutrinos",
)

generate_constraints.add_argument(
    "--min-mass", help="Minimum mass of the RH-neutrino", type=float, required=True
)
generate_constraints.add_argument(
    "--max-mass", help="Maximum mass of the RH-neutrino", type=float, required=True
)
generate_constraints.add_argument(
    "-n",
    "--num-masses",
    help="Maximum mass of the RH-neutrino",
    type=int,
    required=True,
)
generate_constraints.add_argument(
    "-g",
    "--generation",
    help="Generation of the RH-Neutrino",
    choices=[1, 2, 3],
    type=int,
    nargs="?",
    required=True,
)
generate_constraints.add_argument(
    "-o",
    "--output",
    help="File name where constraints should be stored",
    type=str,
    required=True,
)
generate_constraints.add_argument("--dm-mass", help="Mass of the DM", type=float)


class Arguments(NamedTuple):
    min_mass: float
    max_mass: float
    num_masses: int
    generation: Gen
    dm_mass: Optional[float]


def _generate_constraints(args: Arguments):
    constrainer = RhNeutrinoConstrainer()
    mns = np.geomspace(args.min_mass, args.max_mass, args.num_masses)

    constraints = dict()
    constraints["masses"] = mns

    if args.generation == Gen.Fst:
        constraints["generation"] = 1
    elif args.generation == Gen.Fst:
        constraints["generation"] = 2
    else:
        constraints["generation"] = 3

    with Progress() as progress:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            constraints = constrainer.constrain(
                mns,
                mx=args.dm_mass,
                theta=1e-3,
                gen=args.generation,
                progress=progress,
                sigma=2.0,
                method="chi2",
            )

    constraints["masses"] = mns

    for key, value in constraints.items():
        constraints[key] = list(value)

    return constraints


if __name__ == "__main__":
    args = generate_constraints.parse_args()

    output = pathlib.Path(args.output)
    if output.exists():
        raise ValueError(f"file {str(output)} already exists.")

    with open(output, "w") as f:
        if hasattr(args.generation, "__len__"):
            constraints = []
            for gen in args.generation:
                arguments = Arguments(
                    min_mass=args.min_mass,
                    max_mass=args.max_mass,
                    num_masses=args.num_masses,
                    generation=[Gen.Fst, Gen.Snd, Gen.Trd][args.generation],
                    dm_mass=args.dm_mass,
                )
                constraints.append(_generate_constraints(arguments))
            json.dump(constraints, f)
        else:
            arguments = Arguments(
                min_mass=args.min_mass,
                max_mass=args.max_mass,
                num_masses=args.num_masses,
                generation=[Gen.Fst, Gen.Snd, Gen.Trd][args.generation],
                dm_mass=args.dm_mass,
            )
            constraints = _generate_constraints(arguments)
            json.dump(constraints, f)
