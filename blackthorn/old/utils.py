from typing import Dict, Union, List, Tuple

import numpy as np

from .rh_neutrino import Gen
from .fields import UpQuark, CharmQuark, TopQuark
from .fields import DownQuark, StrangeQuark, BottomQuark
from .fields import Electron, Muon, Tau
from .fields import ElectronNeutrino, MuonNeutrino, TauNeutrino

LEPTON_STR_GEN: List[Tuple[str, Gen]] = [
    ("e", Gen.Fst),
    ("mu", Gen.Snd),
    ("tau", Gen.Trd),
]
UP_QUARK_STR_GEN: List[Tuple[str, Gen]] = [
    ("u", Gen.Fst),
    ("c", Gen.Snd),
    ("t", Gen.Trd),
]
DOWN_QUARK_STR_GEN: List[Tuple[str, Gen]] = [
    ("d", Gen.Fst),
    ("s", Gen.Snd),
    ("b", Gen.Trd),
]

UpQuarkType = Union[UpQuark, CharmQuark, TopQuark]
DownQuarkType = Union[DownQuark, StrangeQuark, BottomQuark]
ChargedLeptonType = Union[Electron, Muon, Tau]
NeutrinoType = Union[ElectronNeutrino, MuonNeutrino, TauNeutrino]

UP_QUARKS: Tuple[UpQuark, CharmQuark, TopQuark] = (UpQuark(), CharmQuark(), TopQuark())
DOWN_QUARKS: Tuple[DownQuark, StrangeQuark, BottomQuark] = (
    DownQuark(),
    StrangeQuark(),
    BottomQuark(),
)
CHARGED_LEPTONS: Tuple[Electron, Muon, Tau] = (Electron(), Muon(), Tau())
NEUTRINOS: Tuple[ElectronNeutrino, MuonNeutrino, TauNeutrino] = (
    ElectronNeutrino(),
    MuonNeutrino(),
    TauNeutrino(),
)

PDG_TO_NAME_DICT: Dict[int, str] = {
    # quarks
    1: "d",
    2: "u",
    3: "s",
    4: "c",
    5: "b",
    6: "t",
    # Leptons
    11: "e",
    12: "ve",
    13: "mu",
    14: "vmu",
    15: "tau",
    16: "vtau",
    # Bosons
    21: "g",
    22: "a",
    23: "z",
    24: "w",
    25: "h",
    # Anti quarks
    -1: "d~",
    -2: "u~",
    -3: "s~",
    -4: "c~",
    -5: "b~",
    -6: "t~",
    # Anti charged lepton
    -11: "e~",
    -13: "mu~",
    -15: "tau~",
    # Anti bosons
    -24: "w~",
}


def gen_to_index(gen: Gen) -> int:
    if gen == Gen.Fst:
        return 0
    elif gen == Gen.Snd:
        return 1
    else:
        return 2


def gen_to_up_quark(gen: Gen) -> UpQuarkType:
    return UP_QUARKS[gen_to_index(gen)]


def gen_to_down_quark(gen: Gen) -> DownQuarkType:
    return DOWN_QUARKS[gen_to_index(gen)]


def gen_to_charged_lepton(gen: Gen) -> ChargedLeptonType:
    return CHARGED_LEPTONS[gen_to_index(gen)]


def gen_to_neutrino(gen: Gen) -> NeutrinoType:
    return NEUTRINOS[gen_to_index(gen)]


def kallen_lambda(a: float, b: float, c: float) -> float:
    return a**2 + b**2 + c**2 - 2 * (a * b + a * c + b * c)


def energies_two_body(q: float, m1: float, m2: float) -> Tuple[float, float]:
    e1 = (q**2 + m1**2 - m2**2) / (2 * q)
    e2 = (q**2 - m1**2 + m2**2) / (2 * q)
    return (e1, e2)


def pdg_to_name(pdg: int) -> str:
    name = PDG_TO_NAME_DICT.get(pdg)
    if name is None:
        raise ValueError(f"Invalid PDG code {pdg}.")
    return name


TEX_DICT = {
    "v": r"$\nu$",
    "vi": r"$\nu_{i}$",
    "vj": r"$\nu_{j}$",
    "ve": r"$\nu_{e}$",
    "vmu": r"$\nu_{\mu}$",
    "vtau": r"$\nu_{\tau}$",
    "ell": r"$\ell^{\pm}$",
    "elli": r"$\ell_{i}^{\pm}$",
    "ellj": r"$\ell_{j}^{\pm}$",
    "e": r"$e^{\pm}$",
    "mu": r"$\mu^{\pm}$",
    "tau": r"$\tau^{\pm}$",
    "ellbar": r"$\ell^{\mp}$",
    "ellibar": r"$\ell_{i}^{\mp}$",
    "elljbar": r"$\ell_{j}^{\mp}$",
    "ebar": r"$e^{\mp}$",
    "mubar": r"$\mu^{\mp}$",
    "taubar": r"$\tau^{\mp}$",
    "u": r"$u$",
    "ui": r"$u_{i}$",
    "uj": r"$u_{j}$",
    "c": r"$c$",
    "t": r"$t$",
    "ubar": r"$\bar{u}$",
    "uibar": r"$\bar{u}_{i}$",
    "ujbar": r"$\bar{u}_{j}$",
    "cbar": r"$\bar{c}$",
    "tbar": r"$\bar{t}$",
    "d": r"$d$",
    "di": r"$d_{i}$",
    "dj": r"$d_{j}$",
    "s": r"$s$",
    "b": r"$b$",
    "dbar": r"$\bar{d}$",
    "dibar": r"$\bar{d}_{i}$",
    "djbar": r"$\bar{d}_{j}$",
    "sbar": r"$\bar{s}$",
    "bbar": r"$\bar{b}$",
    "h": r"$H$",
    "z": r"$Z$",
    "w": r"$W^{\pm}$",
    "a": r"$\gamma$",
    "wbar": r"$W^{\mp}$",
    "pi": r"$\pi^{\pm}$",
    "pibar": r"$\pi^{\mp}$",
    "k": r"$K^{\pm}$",
    "eta": r"$\eta$",
    "pi0": r"$\pi^{0}$",
    "k0": r"$K^{0}$",
    "k0bar": r"$\bar{K}^{0}$",
}


def state_to_latex(state: str) -> str:
    """
    Convert a string with space-separated states into a LaTeX equivalent.
    """
    return " + ".join([TEX_DICT[s] for s in state.split(" ")])
