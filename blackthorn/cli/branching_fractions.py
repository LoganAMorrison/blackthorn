import pathlib
import warnings
from typing import List, TypedDict
import json

import numpy as np

from cleo.commands.command import Command
from cleo.helpers import option

# from cleo.helpers import argument

from rich.prompt import Confirm
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)


from blackthorn.models import RhNeutrino
from blackthorn.constants import Gen
from blackthorn.cli.base import CliConsoleControl

CONSOLE = CliConsoleControl()

INT_TO_GEN = {
    "1": Gen.Fst,
    "2": Gen.Snd,
    "3": Gen.Trd,
}


class ParsedArguments(TypedDict):
    min_mass: float
    max_mass: float
    num_masses: int
    generations: List[Gen]
    output: pathlib.Path


class BfArguments(TypedDict):
    min_mass: float
    max_mass: float
    num_masses: int
    generation: Gen


class GenerateBranchingFractions(Command):
    name: str = "bf"
    description: str = "Generate branching fractions for RH neutrino decays"
    arguments = []
    options = [
        option(
            long_name="min-mass",
            description="Minimum RH neutrino mass",
            flag=False,
            value_required=True,
        ),
        option(
            long_name="max-mass",
            description="Maximum RH neutrino mass",
            flag=False,
            value_required=True,
        ),
        option(
            short_name="N",
            long_name="num-masses",
            description="Number of masses between min and max mass",
            flag=False,
            value_required=True,
        ),
        option(
            short_name="o",
            long_name="output",
            description="File where results should be written to",
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

    def __parse_arguments(self) -> ParsedArguments:
        try:
            min_mass = float(self.option("min-mass"))
        except ValueError as e:
            CONSOLE.print_error(f"Error parsing min-mass: {str(e)}")
            raise e

        try:
            max_mass = float(self.option("max-mass"))
        except ValueError as e:
            CONSOLE.print_error(f"Error parsing max-mass: {str(e)}")
            raise e

        try:
            max_mass = float(self.option("max-mass"))
        except ValueError as e:
            CONSOLE.print_error(f"Error parsing max-mass: {str(e)}")
            raise e

        try:
            num_masses = int(self.option("num-masses"))
        except ValueError as e:
            CONSOLE.print_error(f"Error parsing num-masses: {str(e)}")
            raise e

        try:
            generations = list(map(lambda g: INT_TO_GEN[g], self.option("generation")))
        except ValueError as e:
            CONSOLE.print_error("Error parsing generation")
            raise e

        path = self.__parse_output_file(self.option("output"))

        return {
            "min_mass": min_mass,
            "max_mass": max_mass,
            "num_masses": num_masses,
            "generations": generations,
            "output": path,
        }

    def _generate_branching_fractions(self, generation: Gen, masses: np.ndarray):
        intgen = int(generation) + 1
        model = RhNeutrino(masses[0], generation, 1e-3)

        branching_fractions = dict()

        def format_mass(m):
            return "{:.2e} GeV".format(m)

        with CONSOLE.progress(["mass"]) as progress:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                task = progress.add_task(
                    f"[blue] Branching fractions (Gen={intgen})",
                    total=len(masses),
                    mass=format_mass(masses[0]),
                )

                for i, mass in enumerate(masses):
                    model.mass = mass
                    bfs = model.branching_fractions()

                    for key, val in bfs.items():
                        if key not in branching_fractions:
                            branching_fractions[key] = np.zeros_like(  # type: ignore
                                masses
                            )

                        branching_fractions[key][i] = val  # type: ignore

                    progress.update(
                        task, advance=1, refresh=True, mass=format_mass(mass)
                    )

        # Convert numpy to list to make JSON serializable
        for key, val in branching_fractions.items():
            if hasattr(val, "__len__"):
                branching_fractions[key] = list(val)  # type: ignore

        return branching_fractions

    def handle(self):
        args = self.__parse_arguments()
        CONSOLE.print_args(args)  # type: ignore
        min_mass = args["min_mass"]
        max_mass = args["max_mass"]
        num_masses = args["num_masses"]
        generations = args["generations"]
        output = args["output"]

        if output is None:
            return

        cont = CONSOLE.ask_yes_no("Continue?")
        if not cont:
            CONSOLE.print_error("Aborting")
            raise RuntimeError()

        masses = np.geomspace(min_mass, max_mass, num_masses)
        branching_fractions = {"masses": list(masses), "branching_fractions": []}
        for gen in generations:
            branching_fractions["branching_fractions"].append(
                self._generate_branching_fractions(gen, masses)
            )

        with open(output, "w") as f:
            json.dump(branching_fractions, f, indent=2)
