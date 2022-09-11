import argparse
import pathlib
import warnings
from typing import NamedTuple, Optional, Tuple, List
import json

import numpy as np
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

from rich.prompt import Prompt, FloatPrompt

from blackthorn.cli.prompt import RangedFloatPrompt, IntListPrompt, RangedIntPrompt
from blackthorn.models import RhNeutrino
from blackthorn.constants import Gen

CONSOLE = Console()


generate_branching_fractions = argparse.ArgumentParser(
    prog="generate-branching-ratios",
    description="Generate branching ratios for RH-neutrino decays.",
)

generate_branching_fractions.add_argument(
    "--min-mass",
    help="Minimum mass of the RH-neutrino",
    type=float,
)
generate_branching_fractions.add_argument(
    "--max-mass",
    help="Maximum mass of the RH-neutrino",
    type=float,
)
generate_branching_fractions.add_argument(
    "-n",
    "--num-masses",
    help="Maximum mass of the RH-neutrino",
    type=int,
)
generate_branching_fractions.add_argument(
    "-g",
    "--generation",
    help="Generation of the RH-Neutrino",
    choices=[1, 2, 3],
    type=int,
    nargs="?",
)
generate_branching_fractions.add_argument(
    "-o",
    "--output",
    help="File name where constraints should be stored",
    type=str,
)


class Arguments(NamedTuple):
    min_mass: float
    max_mass: float
    num_masses: int
    generation: Gen


def validate_float(
    value: Optional[float],
    name: str,
    min_value: float = -np.inf,
    max_value: float = np.inf,
    default: Optional[float] = None,
    ask_message: Optional[str] = None,
) -> float:
    def valid(val):
        if val is not None:
            return min_value < val < max_value
        return False

    def report_invalid_range(val):
        CONSOLE.print(
            f"[red]Invalid value {val}. Must be in range [{min_value},{max_value}]"
        )

    if value is not None:
        if valid(value):
            return value

    if ask_message is None:
        ask_message = f"Value for {name}"

    while True:
        value = FloatPrompt.ask(ask_message, default=default)

        if valid(value) and value is not None:
            return float(value)
        else:
            report_invalid_range(value)


def validate_str(
    value: Optional[str],
    name: str,
    default: Optional[str] = None,
    ask_message: Optional[str] = None,
) -> str:

    if value is not None:
        return value

    if ask_message is None:
        ask_message = f"Value for {name}"

    while True:
        value = Prompt.ask(ask_message, default=default)

        if value is not None:
            return value
        else:
            CONSOLE.print("{name} is required.")


def prompt_int_list(
    value: Optional[str],
    name: str,
    default: Optional[str] = None,
    ask_message: Optional[str] = None,
) -> str:

    if value is not None:
        return value

    if ask_message is None:
        ask_message = f"Value for {name}"

    while True:
        value = Prompt.ask(ask_message, default=default)

        if value is not None:
            return value
        else:
            CONSOLE.print("{name} is required.")


def _generate_branching_fractions(args: Arguments):
    masses = np.geomspace(args.min_mass, args.max_mass, args.num_masses)
    model = RhNeutrino(masses[0], args.generation, 1e-3)

    branching_fractions = dict(masses=masses, generation=int(args.generation))

    columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        TextColumn("mass: {task.fields[mass]}"),
    ]

    with Progress(*columns, console=CONSOLE) as progress:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            task = progress.add_task(
                "[blue] Branching fractions", total=args.num_masses, mass=masses[0]
            )

            for i, mass in enumerate(masses):
                model.mass = mass
                bfs = model.branching_fractions()

                for key, val in bfs.items():
                    if key not in branching_fractions:
                        branching_fractions[key] = np.zeros_like(masses)

                    branching_fractions[key][i] = val  # type: ignore

                progress.update(task, advance=1, refresh=True, mass=mass)

    # Convert numpy to list to make JSON serializable
    for key, val in branching_fractions.items():
        if hasattr(val, "__len__"):
            branching_fractions[key] = list(val)  # type: ignore

    return branching_fractions


def get_int(value: Optional[int], name: str, range: Tuple[int, int]) -> int:
    RangedIntPrompt.value_range = range
    if value is None:
        val = RangedIntPrompt.ask(f"Enter {name}")
    else:
        val = value
    return val


def get_int_list(
    value: Optional[List[int]], name: str, range: Tuple[int, int]
) -> List[int]:
    IntListPrompt.value_range = range
    if value is None:
        val = IntListPrompt.ask(f"Enter {name}")
    else:
        val = value
    return val


def get_float(value: Optional[float], name: str, range: Tuple[float, float]) -> float:
    RangedFloatPrompt.value_range = range
    if value is None:
        val = RangedFloatPrompt.ask(f"Enter {name}")
    else:
        val = value
    return val


def get_str(value: Optional[str], name: str) -> str:
    if value is None:
        val = Prompt.ask(f"Enter {name}")
    else:
        val = value
    return val


def main():
    args = generate_branching_fractions.parse_args()

    min_mass = get_float(args.min_mass, "minimum mass (GeV)", (0, np.inf))
    max_mass = get_float(args.max_mass, "maximum mass (GeV)", (0, np.inf))
    num_masses = get_int(args.num_masses, "Number of masses", (0, int(1e6)))
    generations = get_int_list(args.generation, "generations (comma separated)", (0, 4))
    output = pathlib.Path(get_str(args.output, "path to output file"))

    if output.exists():
        override = Confirm.ask(
            "[yellow bold]File[/yellow bold]"
            + f" [blue underline]{str(output)}[/blue underline]"
            + " [yellow bold]already exists. Override?[/yellow bold]",
        )
        if not override:
            CONSOLE.print("[red bold] Aborting")
            return

    cont = Confirm.ask("Continue?")
    if not cont:
        CONSOLE.print("[red bold] Aborting")
        return

    CONSOLE.print("[bold]Parameters:")
    CONSOLE.print(f"  [blue]min_mass = {min_mass}")
    CONSOLE.print(f"  [blue]max_mass = {max_mass}")
    CONSOLE.print(f"  [blue]num_masses = {num_masses}")
    CONSOLE.print(f"  [blue]generation = {generations}")
    CONSOLE.print(f"  [blue]output = {output}")

    with open(output, "w") as f:
        branching_fractions = []
        for gen in generations:  # type: ignore
            arguments = Arguments(
                min_mass=min_mass,
                max_mass=max_mass,
                num_masses=num_masses,
                generation=[Gen.Fst, Gen.Snd, Gen.Trd][gen - 1],
            )
            branching_fractions.append(_generate_branching_fractions(arguments))
        json.dump(branching_fractions, f, indent=2)


if __name__ == "__main__":
    main()
