import pathlib
from typing import Optional, List, TypedDict
import json

import numpy as np

from cleo.commands.command import Command
from cleo.helpers import option
from rich.prompt import Confirm

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
from blackthorn.cli.base import CliConsoleControl

CONSOLE = CliConsoleControl()
THETA = 1e-3
INT_TO_GEN = {
    "1": Gen.Fst,
    "2": Gen.Snd,
    "3": Gen.Trd,
}


def _get_model(mass, gen):
    if mass < 5.0:
        return RhNeutrinoMeV(mass, THETA, gen)
    elif mass < 1e3:
        return RhNeutrinoGeV(mass, THETA, gen)
    return RhNeutrinoTeV(mass, THETA, gen)


class ParsedArguments(TypedDict):
    mass: float
    xmin: float
    xmax: float
    num_x: int
    generations: List[Gen]
    output: pathlib.Path
    dm_mass: Optional[float]
    eps: float
    scale: str
    photon: bool
    positron: bool
    neutrino: bool


class GenerateSpectra(Command):
    name: str = "spectra"
    description: str = "Generate spectra from RH neutrino decays"
    arguments = []
    options = [
        option(
            long_name="mass",
            description="Mass of the RH neutrino in GeV",
            flag=False,
            value_required=True,
        ),
        option(
            long_name="xmin",
            description="Minimum value of x=2E/sqrt(s)",
            flag=False,
            value_required=True,
        ),
        option(
            long_name="xmax",
            description="Maximum value of x=2E/sqrt(s)",
            flag=False,
            value_required=True,
        ),
        option(
            long_name="num-xs",
            description="Number of x=2E/sqrt(s) values",
            flag=False,
            value_required=True,
        ),
        option(
            short_name="g",
            long_name="generation",
            description="Generation(s) of the RH neutrino",
            flag=False,
            value_required=True,
            multiple=True,
        ),
        option(
            short_name="o",
            long_name="output",
            description="File where results should be written to",
            flag=False,
            value_required=True,
        ),
        option(
            long_name="dm-mass",
            description="Mass of the dark matter in GeV."
            + " If specified, annihilation spectra are generated.",
            flag=False,
            value_required=True,
        ),
        option(
            long_name="eps",
            description="Energy resolution. Default is 0.05 (5%)",
            flag=False,
            value_required=True,
            default=0.05,
        ),
        option(
            long_name="scale",
            description="Scale of the x values ('log' or 'linear'). Default is 'log'",
            flag=False,
            value_required=True,
            default="log",
        ),
        option(
            long_name="photon",
            description="Enable photon spectra generation",
            flag=True,
        ),
        option(
            long_name="positron",
            description="Enable positron spectra generation",
            flag=True,
        ),
        option(
            long_name="neutrino",
            description="Enable neutrino spectra generation",
            flag=True,
        ),
    ]

    def __parse_output_file(self, output: str):
        path = pathlib.Path(output).absolute()
        if path.exists():
            override = Confirm.ask(
                "[yellow bold]File[/yellow bold]"
                + f" [blue underline]{str(output)}[/blue underline]"
                + " [yellow bold]already exists. Override?[/yellow bold]",
            )
            if not override:
                CONSOLE.print_error("Aborting")
                raise RuntimeError()

        return path

    def __try_parse_float(self, name) -> float:
        try:
            value = float(self.option(name))
        except ValueError as e:
            CONSOLE.print_error(f"Error parsing {name}: {str(e)}")
            raise e
        except TypeError as e:
            CONSOLE.print_error(f"Invalid type for {name}. Expected a float.")
            raise e

        return value

    def __try_parse_int(self, name) -> int:
        try:
            value = int(self.option(name))
        except ValueError as e:
            CONSOLE.print_error(f"Error parsing {name}: {str(e)}")
            raise e
        except TypeError as e:
            CONSOLE.print_error(f"Invalid type for {name}. Expected an int.")
            raise e

        return value

    def __try_parse_str(self, name) -> str:
        try:
            value = str(self.option(name))
        except ValueError as e:
            CONSOLE.print_error(f"Error parsing {name}: {str(e)}")
            raise e
        except TypeError as e:
            CONSOLE.print_error(f"Invalid type for {name}. Expected a string.")
            raise e

        return value

    def __parse_arguments(self) -> ParsedArguments:
        mass = self.__try_parse_float("mass")
        xmin = self.__try_parse_float("xmin")
        xmax = self.__try_parse_float("xmax")
        eps = self.__try_parse_float("eps")
        n = self.__try_parse_int("num-xs")
        photon = self.option("photon")
        positron = self.option("positron")
        neutrino = self.option("neutrino")
        scale = self.__try_parse_str("scale")

        if self.option("dm-mass") is not None:
            dm_mass = self.__try_parse_float("dm-mass")
        else:
            dm_mass = None

        if scale not in ["log", "linear"]:
            CONSOLE.print_error(
                f"Invalid value for 'scale': {scale}." + "Use 'log' or 'linear'."
            )
            raise ValueError()

        try:
            generations = list(map(lambda g: INT_TO_GEN[g], self.option("generation")))
        except ValueError as e:
            CONSOLE.print_error("Error parsing generation")
            raise e

        path = self.__parse_output_file(self.option("output"))

        return {
            "mass": mass,
            "xmin": xmin,
            "xmax": xmax,
            "num_x": n,
            "generations": generations,
            "output": path,
            "dm_mass": dm_mass,
            "eps": eps,
            "scale": scale,
            "photon": photon,
            "positron": positron,
            "neutrino": neutrino,
        }

    def _generate_spectrum(self, x, gen: Gen, args: ParsedArguments):
        mass = args["mass"]
        model = _get_model(mass, gen)

        if args["dm_mass"] is not None:
            energy = args["dm_mass"] / 2.0
            beta = np.sqrt(1.0 - energy / mass)
        else:
            beta = None

        def postprocess(spectrum: Spectrum) -> List[float]:
            if beta is not None:
                spectrum = spectrum.boost(beta)
            spectrum = spectrum.convolve(args["eps"])
            return list(spectrum.dndx)

        spectra = {}

        if gen == Gen.Fst:
            spectra["generation"] = 1  # type: ignore
        elif gen == Gen.Fst:
            spectra["generation"] = 2  # type: ignore
        else:
            spectra["generation"] = 3  # type: ignore

        if args["photon"]:
            spectra["photon"] = postprocess(model.dndx(x, Photon))
        if args["positron"]:
            spectra["positron"] = postprocess(model.dndx(x, Positron))
        if args["neutrino"]:
            spectra["electron_neutrino"] = postprocess(model.dndx(x, ElectronNeutrino))
            spectra["muon_neutrino"] = postprocess(model.dndx(x, MuonNeutrino))
            spectra["tau_neutrino"] = postprocess(model.dndx(x, TauNeutrino))

        return spectra

    def handle(self):
        args = self.__parse_arguments()
        CONSOLE.print_args(args)  # type: ignore
        generations = args["generations"]
        output = args["output"]

        xmin = args["xmin"]
        xmax = args["xmax"]
        n = args["num_x"]
        scale = args["scale"]
        if scale == "log":
            x = np.geomspace(xmin, xmax, n)
        elif scale == "linear":
            x = np.linspace(xmin, xmax, n)
        else:
            raise ValueError(f"Invalid scale {scale}")

        if output is None:
            return

        cont = CONSOLE.ask_yes_no("Continue?")
        if not cont:
            CONSOLE.print_error("Aborting")
            raise RuntimeError()

        spectra = {"x": list(x), "spectra": []}
        for gen in generations:
            spectra["spectra"].append(self._generate_spectrum(x, gen, args))

        with open(output, "w") as f:
            json.dump(spectra, f, indent=2)
