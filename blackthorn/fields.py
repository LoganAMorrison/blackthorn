import dataclasses

from .constants import Gen


@dataclasses.dataclass(frozen=True)
class QuantumField:
    mass: float
    pdg: int
    width: float
    charge: float


@dataclasses.dataclass(frozen=True)
class ChargedLepton(QuantumField):
    gen: Gen


Electron = ChargedLepton(
    mass=0.5109989461e-3, pdg=11, gen=Gen.Fst, width=0.0, charge=-1.0
)
Positron = ChargedLepton(
    mass=0.5109989461e-3, pdg=-11, gen=Gen.Fst, width=0.0, charge=1.0
)
Muon = ChargedLepton(mass=105.6583745e-3, pdg=13, gen=Gen.Snd, width=0.0, charge=-1.0)
AntiMuon = ChargedLepton(
    mass=105.6583745e-3, pdg=-13, gen=Gen.Snd, width=0.0, charge=1.0
)
Tau = ChargedLepton(mass=1.77686, pdg=15, gen=Gen.Trd, width=0.0, charge=-1.0)
AntiTau = ChargedLepton(mass=1.77686, pdg=-15, gen=Gen.Trd, width=0.0, charge=-1.0)


@dataclasses.dataclass(frozen=True)
class Neutrino(QuantumField):
    gen: Gen


ElectronNeutrino = Neutrino(
    pdg=12,
    mass=0.0,
    gen=Gen.Fst,
    width=0.0,
    charge=0.0,
)
MuonNeutrino = Neutrino(
    pdg=14,
    mass=0.0,
    gen=Gen.Snd,
    width=0.0,
    charge=0.0,
)
TauNeutrino = Neutrino(
    pdg=16,
    mass=0.0,
    gen=Gen.Trd,
    width=0.0,
    charge=0.0,
)


@dataclasses.dataclass(frozen=True)
class UpTypeQuark(QuantumField):
    gen: Gen


UpQuark = UpTypeQuark(
    pdg=2,
    mass=2.16e-3,
    gen=Gen.Fst,
    width=0.0,
    charge=2.0 / 3.0,
)
CharmQuark = UpTypeQuark(
    pdg=4,
    mass=1.27,
    gen=Gen.Snd,
    width=0.0,
    charge=2.0 / 3.0,
)
TopQuark = UpTypeQuark(
    pdg=6,
    mass=172.9,
    gen=Gen.Trd,
    width=0.0,
    charge=2.0 / 3.0,
)


@dataclasses.dataclass(frozen=True)
class DownTypeQuark(QuantumField):
    gen: Gen


DownQuark = DownTypeQuark(
    pdg=1,
    mass=4.67e-3,
    gen=Gen.Fst,
    width=0.0,
    charge=-1.0 / 3.0,
)
StrangeQuark = DownTypeQuark(
    pdg=3,
    mass=95.0e-3,
    gen=Gen.Snd,
    width=0.0,
    charge=-1.0 / 3.0,
)
BottomQuark = DownTypeQuark(
    pdg=5,
    mass=4.18,
    gen=Gen.Trd,
    width=0.0,
    charge=-1.0 / 3.0,
)


@dataclasses.dataclass(frozen=True)
class VectorBoson(QuantumField):
    pass


Gluon = QuantumField(
    pdg=21,
    mass=0.0,
    width=0.0,
    charge=0.0,
)
Photon = QuantumField(
    pdg=22,
    mass=0.0,
    width=0.0,
    charge=0.0,
)
ZBoson = QuantumField(
    pdg=23,
    mass=91.18760,
    width=2.49520,
    charge=0.0,
)
WBoson = QuantumField(
    pdg=24,
    mass=80.385003,
    width=2.08500,
    charge=1.0,
)


@dataclasses.dataclass(frozen=True)
class ScalarBoson(QuantumField):
    vev: float


Higgs = ScalarBoson(
    pdg=25,
    mass=125.00,
    width=0.00374,
    vev=246.21965,
    charge=0.0,
)


@dataclasses.dataclass(frozen=True)
class NeutralMeson(QuantumField):
    decay_constant: float


NeutralPion = NeutralMeson(
    mass=0.1349768,
    pdg=111,
    width=0.0,
    decay_constant=0.091924,
    charge=0.0,
)
Eta = NeutralMeson(
    mass=0.547862,
    pdg=211,
    width=0.0,
    decay_constant=0.0,
    charge=0.0,
)
NeutralKaon = NeutralMeson(
    mass=0.497611,
    pdg=311,
    width=0.0,
    decay_constant=0.0,
    charge=0.0,
)
ShortKaon = NeutralMeson(
    mass=NeutralKaon.mass,
    pdg=311,
    width=0.0,
    decay_constant=0.0,
    charge=0.0,
)
LongKaon = NeutralMeson(
    mass=NeutralKaon.mass,
    pdg=130,
    width=0.0,
    decay_constant=0.0,
    charge=0.0,
)


@dataclasses.dataclass(frozen=True)
class ChargedMeson(QuantumField):
    decay_constant: float


ChargedPion = ChargedMeson(
    mass=0.13957039,
    pdg=211,
    width=0.0,
    decay_constant=0.092214,
    charge=1.0,
)
ChargedKaon = ChargedMeson(
    mass=0.493677,
    pdg=321,
    width=0.0,
    decay_constant=0.0,
    charge=1.0,
)
