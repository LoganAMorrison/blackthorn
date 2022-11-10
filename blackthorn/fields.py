"""Module for describing quantum fields and all Standard model fields."""

import dataclasses
from typing import Optional

from .constants import CKM, Gen

# pylint: disable=too-many-instance-attributes


def _anti_same(obj, field: dataclasses.Field):
    return getattr(obj, field.name)


def _anti_name(obj, field: dataclasses.Field):
    return getattr(obj, field.name) + "bar"


def _anti_negate(obj, field: dataclasses.Field):
    return -getattr(obj, field.name)


def _anti_tex(obj, field: dataclasses.Field):
    if field.name == "tex":
        return getattr(obj, "anti_tex")
    return getattr(obj, "tex")


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
    color: int
        Representation under SU(3). If 1, field is a singlet, if 3, field belongs
        to fundamental representation and if 8, the adjoint representation.
    weak_isospin: float
        Eigenvalue of the third generator of weak-isospin. If field is in the
        upper part of a doublet, the value is 1/2. If in the lower part, -1/2.
        Otherwise, 0.
    """

    name: str = dataclasses.field(metadata=dict(anti_fn=_anti_name))
    mass: float = dataclasses.field(metadata=dict(anti_fn=_anti_same))
    pdg: int = dataclasses.field(metadata=dict(anti_fn=_anti_negate))
    width: float = dataclasses.field(metadata=dict(anti_fn=_anti_same))
    charge: float = dataclasses.field(metadata=dict(anti_fn=_anti_negate))
    spin: int = dataclasses.field(metadata=dict(anti_fn=_anti_same))
    color: int = dataclasses.field(metadata=dict(anti_fn=_anti_same))
    weak_isospin: float = dataclasses.field(metadata=dict(anti_fn=_anti_same))
    self_conjugate: bool = dataclasses.field(metadata=dict(anti_fn=_anti_same))
    tex: str = dataclasses.field(metadata=dict(anti_fn=_anti_tex))
    anti_tex: str = dataclasses.field(metadata=dict(anti_fn=_anti_tex))

    def anti(self, name: Optional[str] = None):
        """Generate the anti-particle."""
        if self.self_conjugate:
            return self

        attrs = {}
        for field in dataclasses.fields(self):
            attrs[field.name] = field.metadata["anti_fn"](self, field)

        if name is not None:
            attrs["name"] = name

        return self.__class__(**attrs)


@dataclasses.dataclass(frozen=True)
class ChargedLepton(QuantumField):
    """Charged lepton fields (electron, muon, tau)."""

    gen: Gen = dataclasses.field(metadata=dict(anti_fn=_anti_same))

    @classmethod
    def from_gen(cls, gen: Gen):
        """Return the charged lepton for the given generation."""

        if gen == gen.Fst:
            return Electron
        if gen == gen.Snd:
            return Muon
        return Tau

    def anti(self, name: Optional[str] = None):
        """Generate the anti-particle."""
        if self.self_conjugate:
            return self

        if name is None:
            name = self.name + "bar"

        return self.__class__(
            name=name,
            mass=self.mass,
            pdg=-self.pdg,
            gen=self.gen,
            width=self.width,
            charge=-self.charge,
            spin=self.spin,
            color=self.color,
            weak_isospin=self.weak_isospin,
            tex=self.anti_tex,
            anti_tex=self.tex,
            self_conjugate=self.self_conjugate,
        )


Electron = ChargedLepton(
    name="e",
    mass=0.5109989461e-3,
    pdg=11,
    gen=Gen.Fst,
    width=0.0,
    charge=-1.0,
    spin=1,
    color=1,
    weak_isospin=-0.5,
    tex=r"$e^{-}$",
    anti_tex=r"$e^{+}$",
    self_conjugate=False,
)
Muon = ChargedLepton(
    name="mu",
    mass=105.6583745e-3,
    pdg=13,
    gen=Gen.Snd,
    width=0.0,
    charge=-1.0,
    spin=1,
    color=1,
    weak_isospin=-0.5,
    tex=r"$\mu^{-}$",
    anti_tex=r"$\mu^{+}$",
    self_conjugate=False,
)
Tau = ChargedLepton(
    name="tau",
    mass=1.77686,
    pdg=15,
    gen=Gen.Trd,
    width=0.0,
    charge=-1.0,
    spin=1,
    color=1,
    weak_isospin=-0.5,
    tex=r"$\tau^{-}$",
    anti_tex=r"$\tau^{+}$",
    self_conjugate=False,
)

Positron = Electron.anti()
AntiMuon = Muon.anti()
AntiTau = Tau.anti()


@dataclasses.dataclass(frozen=True)
class Neutrino(QuantumField):
    """Neutral lepton fields."""

    gen: Gen = dataclasses.field(metadata=dict(anti_fn=_anti_same))

    @classmethod
    def from_gen(cls, gen: Gen):
        if gen == Gen.Fst:
            return ElectronNeutrino
        if gen == Gen.Snd:
            return MuonNeutrino
        return TauNeutrino



ElectronNeutrino = Neutrino(
    name="ve",
    pdg=12,
    mass=0.0,
    gen=Gen.Fst,
    width=0.0,
    charge=0.0,
    spin=1,
    color=1,
    weak_isospin=0.5,
    tex=r"$\nu_{e}$",
    anti_tex=r"$\nu_{e}$",
    self_conjugate=True,
)
MuonNeutrino = Neutrino(
    name="vmu",
    pdg=14,
    mass=0.0,
    gen=Gen.Snd,
    width=0.0,
    charge=0.0,
    spin=1,
    color=1,
    weak_isospin=0.5,
    tex=r"$\nu_{\mu}$",
    anti_tex=r"$\nu_{\mu}$",
    self_conjugate=True,
)
TauNeutrino = Neutrino(
    name="vtau",
    pdg=16,
    mass=0.0,
    gen=Gen.Trd,
    width=0.0,
    charge=0.0,
    spin=1,
    color=1,
    weak_isospin=0.5,
    tex=r"$\nu_{\tau}$",
    anti_tex=r"$\nu_{\tau}$",
    self_conjugate=True,
)


@dataclasses.dataclass(frozen=True)
class UpTypeQuark(QuantumField):
    """Up-type quark field (u, c, t)."""

    gen: Gen = dataclasses.field(metadata=dict(anti_fn=_anti_same))

    @classmethod
    def from_gen(cls, gen: Gen):
        """Return the up-type quark for the given generation."""

        if gen == gen.Fst:
            return UpQuark
        if gen == gen.Snd:
            return DownQuark
        return TopQuark


UpQuark = UpTypeQuark(
    name="u",
    pdg=2,
    mass=2.16e-3,
    gen=Gen.Fst,
    width=0.0,
    charge=2.0 / 3.0,
    spin=1,
    color=3,
    weak_isospin=0.5,
    tex=r"$u$",
    anti_tex=r"$\bar{u}$",
    self_conjugate=False,
)
CharmQuark = UpTypeQuark(
    name="c",
    pdg=4,
    mass=1.27,
    gen=Gen.Snd,
    width=0.0,
    charge=2.0 / 3.0,
    spin=1,
    color=3,
    weak_isospin=0.5,
    tex=r"$c$",
    anti_tex=r"$\bar{c}$",
    self_conjugate=False,
)
TopQuark = UpTypeQuark(
    name="t",
    pdg=6,
    mass=172.9,
    gen=Gen.Trd,
    width=0.0,
    charge=2.0 / 3.0,
    spin=1,
    color=3,
    weak_isospin=0.5,
    tex=r"$t$",
    anti_tex=r"$\bar{t}$",
    self_conjugate=False,
)

AntiUpQuark = UpQuark.anti()
AntiCharmQuark = CharmQuark.anti()
AntiTopQuark = TopQuark.anti()


@dataclasses.dataclass(frozen=True)
class DownTypeQuark(QuantumField):
    """Down-type quark field (d, s, b)."""

    gen: Gen = dataclasses.field(metadata=dict(anti_fn=_anti_same))

    @classmethod
    def from_gen(cls, gen: Gen):
        """Return the down-type quark for the given generation."""

        if gen == gen.Fst:
            return DownQuark
        if gen == gen.Snd:
            return StrangeQuark
        return BottomQuark


DownQuark = DownTypeQuark(
    name="d",
    pdg=1,
    mass=4.67e-3,
    gen=Gen.Fst,
    width=0.0,
    charge=-1.0 / 3.0,
    spin=1,
    color=3,
    weak_isospin=-0.5,
    tex=r"$d$",
    anti_tex=r"$\bar{d}$",
    self_conjugate=False,
)
StrangeQuark = DownTypeQuark(
    name="s",
    pdg=3,
    mass=95.0e-3,
    gen=Gen.Snd,
    width=0.0,
    charge=-1.0 / 3.0,
    spin=1,
    color=3,
    weak_isospin=-0.5,
    tex=r"$s$",
    anti_tex=r"$\bar{s}$",
    self_conjugate=False,
)
BottomQuark = DownTypeQuark(
    name="b",
    pdg=5,
    mass=4.18,
    gen=Gen.Trd,
    width=0.0,
    charge=-1.0 / 3.0,
    spin=1,
    color=3,
    weak_isospin=-0.5,
    tex=r"$b$",
    anti_tex=r"$\bar{b}$",
    self_conjugate=False,
)

AntiDownQuark = DownQuark.anti()
AntiStrangeQuark = StrangeQuark.anti()
AntiBottomQuark = BottomQuark.anti()


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
    color=8,
    weak_isospin=0.0,
    tex=r"$g$",
    anti_tex=r"$g$",
    self_conjugate=True,
)
Photon = VectorBoson(
    name="a",
    pdg=22,
    mass=0.0,
    width=0.0,
    charge=0.0,
    spin=2,
    color=1,
    weak_isospin=0.0,
    tex=r"$\gamma$",
    anti_tex=r"$\gamma$",
    self_conjugate=True,
)
ZBoson = VectorBoson(
    name="z",
    pdg=23,
    mass=91.18760,
    width=2.49520,
    charge=0.0,
    spin=2,
    color=1,
    weak_isospin=0.0,
    tex=r"$Z$",
    anti_tex=r"$Z$",
    self_conjugate=True,
)
WBoson = VectorBoson(
    name="w",
    pdg=24,
    mass=80.385003,
    width=2.08500,
    charge=1.0,
    spin=2,
    color=1,
    weak_isospin=0.0,
    tex=r"$W^{+}$",
    anti_tex=r"$W^{-}$",
    self_conjugate=False,
)
AntiWBoson = WBoson.anti()


@dataclasses.dataclass(frozen=True)
class ScalarBoson(QuantumField):
    """Standard model scalar-bosons."""

    vev: float = dataclasses.field(metadata=dict(anti_fn=_anti_same))


Higgs = ScalarBoson(
    name="h",
    pdg=25,
    mass=125.00,
    width=0.00374,
    vev=246.21965,
    charge=0.0,
    spin=0,
    color=1,
    weak_isospin=-0.5,
    tex=r"$h$",
    anti_tex=r"$h$",
    self_conjugate=True,
)


@dataclasses.dataclass(frozen=True)
class NeutralMeson(QuantumField):
    """Standard model neutral psuedo-scalar mesons."""

    decay_constant: float = dataclasses.field(metadata=dict(anti_fn=_anti_same))


NeutralPion = NeutralMeson(
    name="pi0",
    mass=0.1349768,
    pdg=111,
    width=0.0,
    decay_constant=0.091924,
    charge=0.0,
    spin=0,
    color=1,
    weak_isospin=0.0,
    tex=r"$\pi^{0}$",
    anti_tex=r"$\pi^{0}$",
    self_conjugate=True,
)
Eta = NeutralMeson(
    name="eta",
    mass=0.547862,
    pdg=211,
    width=0.0,
    decay_constant=0.0,
    charge=0.0,
    spin=0,
    color=1,
    weak_isospin=0.0,
    tex=r"$\eta$",
    anti_tex=r"$\eta$",
    self_conjugate=True,
)
NeutralKaon = NeutralMeson(
    name="k0",
    mass=0.497611,
    pdg=311,
    width=0.0,
    decay_constant=0.0,
    charge=0.0,
    spin=0,
    color=1,
    weak_isospin=0.0,
    tex=r"$K^{0}$",
    anti_tex=r"$\bar{K}^{0}$",
    self_conjugate=False,
)
ShortKaon = NeutralMeson(
    name="kS",
    mass=NeutralKaon.mass,
    pdg=311,
    width=0.0,
    decay_constant=0.0,
    charge=0.0,
    spin=0,
    color=1,
    weak_isospin=0.0,
    tex=r"$K_{S}$",
    anti_tex=r"$K_{S}$",
    self_conjugate=True,
)
LongKaon = NeutralMeson(
    name="kL",
    mass=NeutralKaon.mass,
    pdg=130,
    width=0.0,
    decay_constant=0.0,
    charge=0.0,
    spin=0,
    color=1,
    weak_isospin=0.0,
    tex=r"$K_{L}$",
    anti_tex=r"$K_{L}$",
    self_conjugate=True,
)
AntiNeutralKaon = NeutralKaon.anti()


@dataclasses.dataclass(frozen=True)
class ChargedMeson(QuantumField):
    """Standard model charged psuedo-scalar mesons."""

    decay_constant: float = dataclasses.field(metadata=dict(anti_fn=_anti_same))
    ckm: complex = dataclasses.field(metadata=dict(anti_fn=_anti_same))


ChargedPion = ChargedMeson(
    name="pi",
    mass=0.13957039,
    pdg=211,
    width=0.0,
    decay_constant=0.092214,
    charge=1.0,
    spin=0,
    color=1,
    weak_isospin=0.0,
    ckm=CKM[Gen.Fst, Gen.Fst],
    tex=r"$\pi^{+}$",
    anti_tex=r"$\pi^{-}$",
    self_conjugate=False,
)
ChargedKaon = ChargedMeson(
    name="k",
    mass=0.493677,
    pdg=321,
    width=0.0,
    decay_constant=0.0,
    charge=1.0,
    spin=0,
    color=1,
    weak_isospin=0.0,
    ckm=CKM[Gen.Fst, Gen.Snd],
    tex=r"$K^{+}$",
    anti_tex=r"$K^{-}$",
    self_conjugate=False,
)

AntiChargedPion = ChargedPion.anti()
AntiChargedKaon = ChargedKaon.anti()
