"""Module for describing quantum fields and all Standard model fields."""

import dataclasses

from .constants import Gen


@dataclasses.dataclass(frozen=True)
class QuantumField:
    """Dataclass for quantum fields.

    Attributes
    ----------
    name: str
        Name of the field.
    mass: float
        Mass of the field in GeV.
    pdg: int
        Particle Data Group number.
    width: float
        Decay width of the field in GeV.
    charge: float
        Electric charge of the field in units of the electron charge.
    spin: int
        Spin of the particle in units of hbar/2. This is twice the
        normal spin, i.e. the electron value will be 1 and the photon 2.
    """

    name: str
    mass: float
    pdg: int
    width: float
    charge: float
    spin: int


@dataclasses.dataclass(frozen=True)
class ChargedLepton(QuantumField):
    """Charged lepton fields (electron, muon, tau)."""

    gen: Gen


Electron = ChargedLepton(
    name="e",
    mass=0.5109989461e-3,
    pdg=11,
    gen=Gen.Fst,
    width=0.0,
    charge=-1.0,
    spin=1,
)
Positron = ChargedLepton(
    name="ebar",
    mass=0.5109989461e-3,
    pdg=-11,
    gen=Gen.Fst,
    width=0.0,
    charge=1.0,
    spin=1,
)
Muon = ChargedLepton(
    name="mu",
    mass=105.6583745e-3,
    pdg=13,
    gen=Gen.Snd,
    width=0.0,
    charge=-1.0,
    spin=1,
)
AntiMuon = ChargedLepton(
    name="mubar",
    mass=105.6583745e-3,
    pdg=-13,
    gen=Gen.Snd,
    width=0.0,
    charge=1.0,
    spin=1,
)
Tau = ChargedLepton(
    name="tau",
    mass=1.77686,
    pdg=15,
    gen=Gen.Trd,
    width=0.0,
    charge=-1.0,
    spin=1,
)
AntiTau = ChargedLepton(
    name="taubar",
    mass=1.77686,
    pdg=-15,
    gen=Gen.Trd,
    width=0.0,
    charge=-1.0,
    spin=1,
)


@dataclasses.dataclass(frozen=True)
class Neutrino(QuantumField):
    """Neutral lepton fields."""

    gen: Gen


ElectronNeutrino = Neutrino(
    name="ve",
    pdg=12,
    mass=0.0,
    gen=Gen.Fst,
    width=0.0,
    charge=0.0,
    spin=1,
)
MuonNeutrino = Neutrino(
    name="vmu",
    pdg=14,
    mass=0.0,
    gen=Gen.Snd,
    width=0.0,
    charge=0.0,
    spin=1,
)
TauNeutrino = Neutrino(
    name="vtau",
    pdg=16,
    mass=0.0,
    gen=Gen.Trd,
    width=0.0,
    charge=0.0,
    spin=1,
)


@dataclasses.dataclass(frozen=True)
class UpTypeQuark(QuantumField):
    """Up-type quark field (u, c, t)."""

    gen: Gen


UpQuark = UpTypeQuark(
    name="u",
    pdg=2,
    mass=2.16e-3,
    gen=Gen.Fst,
    width=0.0,
    charge=2.0 / 3.0,
    spin=1,
)
CharmQuark = UpTypeQuark(
    name="c",
    pdg=4,
    mass=1.27,
    gen=Gen.Snd,
    width=0.0,
    charge=2.0 / 3.0,
    spin=1,
)
TopQuark = UpTypeQuark(
    name="t",
    pdg=6,
    mass=172.9,
    gen=Gen.Trd,
    width=0.0,
    charge=2.0 / 3.0,
    spin=1,
)


@dataclasses.dataclass(frozen=True)
class DownTypeQuark(QuantumField):
    """Down-type quark field (d, s, b)."""

    gen: Gen


DownQuark = DownTypeQuark(
    name="d",
    pdg=1,
    mass=4.67e-3,
    gen=Gen.Fst,
    width=0.0,
    charge=-1.0 / 3.0,
    spin=1,
)
StrangeQuark = DownTypeQuark(
    name="s",
    pdg=3,
    mass=95.0e-3,
    gen=Gen.Snd,
    width=0.0,
    charge=-1.0 / 3.0,
    spin=1,
)
BottomQuark = DownTypeQuark(
    name="b",
    pdg=5,
    mass=4.18,
    gen=Gen.Trd,
    width=0.0,
    charge=-1.0 / 3.0,
    spin=1,
)


@dataclasses.dataclass(frozen=True)
class VectorBoson(QuantumField):
    """Standard model gauge-bosons."""


Gluon = VectorBoson(
    name="g",
    pdg=21,
    mass=0.0,
    width=0.0,
    charge=0.0,
    spin=2,
)
Photon = VectorBoson(
    name="a",
    pdg=22,
    mass=0.0,
    width=0.0,
    charge=0.0,
    spin=2,
)
ZBoson = VectorBoson(
    name="z",
    pdg=23,
    mass=91.18760,
    width=2.49520,
    charge=0.0,
    spin=2,
)
WBoson = VectorBoson(
    name="w",
    pdg=24,
    mass=80.385003,
    width=2.08500,
    charge=1.0,
    spin=2,
)


@dataclasses.dataclass(frozen=True)
class ScalarBoson(QuantumField):
    """Standard model scalar-bosons."""

    vev: float


Higgs = ScalarBoson(
    name="h",
    pdg=25,
    mass=125.00,
    width=0.00374,
    vev=246.21965,
    charge=0.0,
    spin=0,
)


@dataclasses.dataclass(frozen=True)
class NeutralMeson(QuantumField):
    """Standard model neutral psuedo-scalar mesons."""

    decay_constant: float


NeutralPion = NeutralMeson(
    name="pi0",
    mass=0.1349768,
    pdg=111,
    width=0.0,
    decay_constant=0.091924,
    charge=0.0,
    spin=0,
)
Eta = NeutralMeson(
    name="eta",
    mass=0.547862,
    pdg=211,
    width=0.0,
    decay_constant=0.0,
    charge=0.0,
    spin=0,
)
NeutralKaon = NeutralMeson(
    name="k0",
    mass=0.497611,
    pdg=311,
    width=0.0,
    decay_constant=0.0,
    charge=0.0,
    spin=0,
)
ShortKaon = NeutralMeson(
    name="kS",
    mass=NeutralKaon.mass,
    pdg=311,
    width=0.0,
    decay_constant=0.0,
    charge=0.0,
    spin=0,
)
LongKaon = NeutralMeson(
    name="kL",
    mass=NeutralKaon.mass,
    pdg=130,
    width=0.0,
    decay_constant=0.0,
    charge=0.0,
    spin=0,
)


@dataclasses.dataclass(frozen=True)
class ChargedMeson(QuantumField):
    """Standard model charged psuedo-scalar mesons."""

    decay_constant: float


ChargedPion = ChargedMeson(
    name="pi",
    mass=0.13957039,
    pdg=211,
    width=0.0,
    decay_constant=0.092214,
    charge=1.0,
    spin=0,
)
ChargedKaon = ChargedMeson(
    name="k",
    mass=0.493677,
    pdg=321,
    width=0.0,
    decay_constant=0.0,
    charge=1.0,
    spin=0,
)
