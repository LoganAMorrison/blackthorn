from .fields import BottomQuark, CharmQuark, DownQuark, Electron, ElectronNeutrino, Muon, MuonNeutrino, StrangeQuark, Tau, TauNeutrino, TopQuark, UpQuark
from .rh_neutrino import Gen
from typing import Any, List, Tuple, Union

LEPTON_STR_GEN: List[Tuple[str, Gen]]
UP_QUARK_STR_GEN: List[Tuple[str, Gen]]
DOWN_QUARK_STR_GEN: List[Tuple[str, Gen]]
UpQuarkType = Union[UpQuark, CharmQuark, TopQuark]
DownQuarkType = Union[DownQuark, StrangeQuark, BottomQuark]
ChargedLeptonType = Union[Electron, Muon, Tau]
NeutrinoType = Union[ElectronNeutrino, MuonNeutrino, TauNeutrino]
UP_QUARKS: Tuple[UpQuark, CharmQuark, TopQuark]
DOWN_QUARKS: Tuple[DownQuark, StrangeQuark, BottomQuark]
CHARGED_LEPTONS: Tuple[Electron, Muon, Tau]
NEUTRINOS: Tuple[ElectronNeutrino, MuonNeutrino, TauNeutrino]

def gen_to_index(gen: Gen) -> int: ...
def gen_to_up_quark(gen: Gen) -> UpQuarkType: ...
def gen_to_down_quark(gen: Gen) -> DownQuarkType: ...
def gen_to_charged_lepton(gen: Gen) -> ChargedLeptonType: ...
def gen_to_neutrino(gen: Gen) -> NeutrinoType: ...
def kallen_lambda(a: float, b: float, c: float) -> float: ...
def energies_two_body(q: float, m1: float, m2: float) -> Tuple[float, float]: ...

TEX_DICT: Any

def state_to_latex(state: str) -> str: ...
