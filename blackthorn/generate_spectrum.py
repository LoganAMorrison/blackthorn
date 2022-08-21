import argparse
import pathlib
from typing import NamedTuple, Optional, List, Union
import json

import numpy as np

from blackthorn.models.mev import RhNeutrinoMeV
from blackthorn.models.gev import RhNeutrinoGeV
from blackthorn.models.tev import RhNeutrinoTeV
from blackthorn.fields import (
    Photon,
    Positron,
    ElectronNeutrino,
    MuonNeutrino,
    TauNeutrino,
)
from blackthorn.constants import Gen
from blackthorn.spectrum_utils import Spectrum

THETA = 1e-3


generate_spectrum = argparse.ArgumentParser(
    prog="generate-spectrum",
    description="Generate spectra from RH-neutrino decays "
    "or DM annihilations into RH-neutrinos",
)

generate_spectrum.add_argument("mass", help="Mass of the RH-neutrino", type=float)
generate_spectrum.add_argument(
    "generation",
    help="Generation of the RH-neutrino",
    choices=[1, 2, 3],
    nargs="?",
    type=int,
)
generate_spectrum.add_argument(
    "output",
    help="File name where constraints should be stored",
    type=str,
)
generate_spectrum.add_argument(
    "--all",
    help="Flag to enable photon, positron and neutrino spectra generation",
    action="store_true",
)
generate_spectrum.add_argument(
    "--neutrino", help="Flag to enable neutrino spectra generation", action="store_true"
)
generate_spectrum.add_argument(
    "--photon", help="Flag to enable photon spectra generation", action="store_true"
)
generate_spectrum.add_argument(
    "--positron", help="Flag to enable positron spectra generation", action="store_true"
)
generate_spectrum.add_argument(
    "--scale",
    help="Scale of the x values ('log' or 'linear'). Default is 'log'",
    type=str,
    choices=["log", "linear"],
    default="log",
)
generate_spectrum.add_argument(
    "--x-min", help="Minimum value of x. Default is 1e-4", type=float, default=1e-4
)
generate_spectrum.add_argument(
    "--x-max", help="Maximum value of x. Default is 1", type=float, default=1.0
)
generate_spectrum.add_argument(
    "--n", help="Number of x values. Default is 100", type=float, default=100
)
generate_spectrum.add_argument(
    "--eps",
    help="Energy resolution. Default is 0.05 (10%)",
    type=float,
    default=0.05,
)
generate_spectrum.add_argument("--dm-mass", help="Mass of the dark-matter", type=float)


class Arguments(NamedTuple):
    mass: float
    photon: bool
    positron: bool
    neutrino: bool
    generation: Gen
    x_min: float
    x_max: float
    num_xs: int
    scale: str
    dm_mass: Optional[float]
    eps: float

    @classmethod
    def from_parser(cls, args) -> Union["Arguments", List["Arguments"]]:
        if hasattr(args.generation, "__len__"):
            arguments = []
            for gen in args.generation:
                arguments.append(
                    cls(
                        mass=args.mass,
                        x_min=args.x_min,
                        x_max=args.x_max,
                        num_xs=args.n,
                        photon=args.photon or args.all,
                        positron=args.positron or args.all,
                        neutrino=args.neutrino or args.all,
                        generation=[Gen.Fst, Gen.Snd, Gen.Trd][gen],
                        dm_mass=args.dm_mass,
                        eps=args.eps,
                        scale=args.scale,
                    )
                )
            return arguments
        else:
            return cls(
                mass=args.mass,
                x_min=args.x_min,
                x_max=args.x_max,
                num_xs=args.n,
                photon=args.photon,
                positron=args.positron,
                neutrino=args.neutrino,
                generation=[Gen.Fst, Gen.Snd, Gen.Trd][args.generation],
                dm_mass=args.dm_mass,
                eps=args.eps,
                scale=args.scale,
            )


def _get_model(mass, gen):
    if mass < 5.0:
        return RhNeutrinoMeV(mass, THETA, gen)
    elif mass < 1e3:
        return RhNeutrinoGeV(mass, THETA, gen)
    return RhNeutrinoTeV(mass, THETA, gen)


def _generate_spectrum(args: Arguments):
    mass = args.mass
    model = _get_model(mass, args.generation)
    if args.scale == "log":
        x = np.geomspace(args.x_min, args.x_max, args.num_xs)
    elif args.scale == "linear":
        x = np.linspace(args.x_min, args.x_max, args.num_xs)
    else:
        raise ValueError(f"Invalid scale {args.scale}")

    if args.dm_mass is not None:
        energy = args.dm_mass / 2.0
        beta = np.sqrt(1.0 - energy / mass)
    else:
        beta = None

    def postprocess(spectrum: Spectrum) -> List[float]:
        if beta is not None:
            spectrum = spectrum.boost(beta)
        spectrum = spectrum.convolve(args.eps)
        return list(spectrum.dndx)

    spectra = {"x": list(x)}

    if args.generation == Gen.Fst:
        spectra["generation"] = 1  # type: ignore
    elif args.generation == Gen.Fst:
        spectra["generation"] = 2  # type: ignore
    else:
        spectra["generation"] = 3  # type: ignore

    if args.photon:
        spectra["photon"] = postprocess(model.dndx(x, Photon))
    if args.positron:
        spectra["positron"] = postprocess(model.dndx(x, Positron))
    if args.neutrino:
        spectra["electronNeutrino"] = postprocess(model.dndx(x, ElectronNeutrino))
        spectra["muonNeutrino"] = postprocess(model.dndx(x, MuonNeutrino))
        spectra["tauNeutrino"] = postprocess(model.dndx(x, TauNeutrino))

    return spectra


if __name__ == "__main__":
    args = generate_spectrum.parse_args()

    output = pathlib.Path(args.output)
    if output.exists():
        raise ValueError(f"file {str(output)} already exists.")

    if not (args.photon or args.positron or args.neutrino):
        raise ValueError("No final state(s) specfied.")

    arguments = Arguments.from_parser(args)

    with open(output, "w") as f:
        if isinstance(arguments, list):
            constraints = []
            for args in arguments:
                constraints.append(_generate_spectrum(args))
            json.dump(constraints, f, indent=2)
        else:
            constraints = _generate_spectrum(arguments)
            json.dump(constraints, f, indent=2)
