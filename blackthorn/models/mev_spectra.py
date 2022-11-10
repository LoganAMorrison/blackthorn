from typing import List

import numpy as np
import numpy.typing as npt
from hazma import spectra
from hazma.spectra.altarelli_parisi import dnde_photon_ap_fermion, dnde_photon_ap_scalar

from ..constants import Gen
from .. import fields
from ..fields import ChargedKaon, ChargedPion, Electron, Eta, Muon, NeutralPion, Tau
from ..spectrum_utils import Spectrum, SpectrumLine
from .base import RhNeutrinoBase, DecaySpectrumTwoBody, DecaySpectrumThreeBody
from .msqrd import (
    msqrd_n_to_v_l_l,
    msqrd_n_to_v_pi_pi,
    msqrd_n_to_l_pi_pi0,
    msqrd_n_to_v_v_v,
)

RealArray = npt.NDArray[np.float_]

_lepton_masses = [Electron.mass, Muon.mass, Tau.mass]


class DecaySpectrumVPi0(DecaySpectrumTwoBody):
    """Class for RHN decays into a neutrino and neutral pion."""

    def __init__(self, model: RhNeutrinoBase, branching_ratio: float = 1):
        neutrino = fields.Neutrino.from_gen(model.gen)
        super().__init__(model, neutrino, fields.NeutralPion, branching_ratio)


class DecaySpectrumVEta(DecaySpectrumTwoBody):
    """Class for RHN decays into a neutrino and eta."""

    def __init__(self, model: RhNeutrinoBase, branching_ratio: float = 1):
        neutrino = fields.Neutrino.from_gen(model.gen)
        super().__init__(model, neutrino, fields.Eta, branching_ratio)


class DecaySpectrumLPi(DecaySpectrumTwoBody):
    """Class for RHN decays into a charged lepton and charged pion."""

    def __init__(self, model: RhNeutrinoBase, branching_ratio: float = 1):
        lepton = fields.ChargedLepton.from_gen(model.gen)
        super().__init__(model, lepton, fields.ChargedPion, branching_ratio)


class DecaySpectrumLK(DecaySpectrumTwoBody):
    """Class for RHN decays into a charged lepton and charged kaon."""

    def __init__(self, model: RhNeutrinoBase, branching_ratio: float = 1):
        lepton = fields.ChargedLepton.from_gen(model.gen)
        super().__init__(model, lepton, fields.ChargedKaon, branching_ratio)


class DecaySpectrumVA(DecaySpectrumTwoBody):
    """Class for RHN decays into a neutrino and photon."""

    def __init__(self, model: RhNeutrinoBase, branching_ratio: float = 1):
        lepton = fields.Neutrino.from_gen(model.gen)
        super().__init__(model, lepton, fields.Photon, branching_ratio)


class DecaySpectrumVLL(DecaySpectrumThreeBody):
    """Class for RHN decays into a neutrino and two charged leptons."""

    def __init__(
        self,
        model: RhNeutrinoBase,
        *,
        genv: Gen,
        genl1: Gen,
        genl2: Gen,
        npts: int,
        nbins: int,
        branching_fraction: float = 1,
    ):

        neutrino = fields.Neutrino.from_gen(genv)
        lepton1 = fields.ChargedLepton.from_gen(genl1)
        lepton2 = fields.ChargedLepton.from_gen(genl2)

        def msqrd(momenta):
            return msqrd_n_to_v_l_l(
                model, momenta=momenta, genv=genv, genl1=genl1, genl2=genl2
            )

        super().__init__(
            model,
            neutrino,
            lepton1,
            lepton2,
            msqrd,
            npts=npts,
            nbins=nbins,
            branching_fraction=branching_fraction,
        )


class DecaySpectrumVPiPi(DecaySpectrumThreeBody):
    """Class for RHN decays into a neutrino and two charged pions."""

    def __init__(
        self,
        model: RhNeutrinoBase,
        *,
        npts: int,
        nbins: int,
        branching_fraction: float = 1,
    ):

        neutrino = fields.Neutrino.from_gen(model.gen)
        pion = fields.ChargedPion

        def msqrd(momenta):
            return msqrd_n_to_v_pi_pi(model, momenta=momenta)

        super().__init__(
            model,
            neutrino,
            pion,
            pion,
            msqrd,
            npts=npts,
            nbins=nbins,
            branching_fraction=branching_fraction,
        )


class DecaySpectrumLPiPi0(DecaySpectrumThreeBody):
    """Class for RHN decays into a charged lepton and two pions."""

    def __init__(
        self,
        model: RhNeutrinoBase,
        *,
        npts: int,
        nbins: int,
        branching_fraction: float = 1,
    ):

        lepton = fields.ChargedLepton.from_gen(model.gen)
        charged_pion = fields.ChargedPion
        neutral_pion = fields.NeutralPion

        def msqrd(momenta):
            return msqrd_n_to_l_pi_pi0(model, momenta=momenta)

        super().__init__(
            model,
            lepton,
            charged_pion,
            neutral_pion,
            msqrd,
            npts=npts,
            nbins=nbins,
            branching_fraction=branching_fraction,
        )


class DecaySpectrumVVV(DecaySpectrumThreeBody):
    """Class for RHN decays into three neutrinos."""

    def __init__(
        self,
        model: RhNeutrinoBase,
        *,
        genv1: Gen,
        genv2: Gen,
        genv3: Gen,
        npts: int,
        nbins: int,
        branching_fraction: float = 1,
    ):

        nu1 = fields.Neutrino.from_gen(genv1)
        nu2 = fields.Neutrino.from_gen(genv2)
        nu3 = fields.Neutrino.from_gen(genv3)

        def msqrd(momenta):
            return msqrd_n_to_v_v_v(
                model, momenta=momenta, genv1=genv1, genv2=genv2, genv3=genv3
            )

        super().__init__(
            model,
            nu1,
            nu2,
            nu3,
            msqrd,
            npts=npts,
            nbins=nbins,
            branching_fraction=branching_fraction,
        )


def dndx_v_pi0(self: RhNeutrinoBase, x: RealArray, product: int) -> Spectrum:
    mvr = self.mass
    mpi0 = NeutralPion.mass
    epi0 = (mvr**2 + mpi0**2) / (2 * mvr)
    enu = (mvr**2 - mpi0**2) / (2 * mvr)
    dndx = np.zeros_like(x)
    lines: List[SpectrumLine] = []

    if product == 21:
        dndx += mvr / 2 * spectra.dnde_photon_neutral_pion(0.5 * mvr * x, epi0)
    elif (
        (product == 12 and self.gen == Gen.Fst)
        or (product == 14 and self.gen == Gen.Snd)
        or (product == 16 and self.gen == Gen.Trd)
    ):
        xloc = enu * 2 / mvr
        br = 0.0  # TODO
        lines.append(SpectrumLine(xloc=xloc, br=br))

    return Spectrum(x=x, dndx=dndx, lines=lines)


def dndx_v_eta(self: RhNeutrinoBase, x, product) -> Spectrum:
    mvr = self.mass
    meta = Eta.mass
    eeta = (mvr**2 + meta**2) / (2 * mvr)
    enu = (mvr**2 - meta**2) / (2 * mvr)
    dndx = np.zeros_like(x)
    lines: List[SpectrumLine] = []
    if product == 21:
        dndx += mvr / 2 * spectra.dnde_photon_eta(0.5 * mvr * x, eeta)
    elif (
        (product == 12 and self.gen == Gen.Fst)
        or (product == 14 and self.gen == Gen.Snd)
        or (product == 16 and self.gen == Gen.Trd)
    ):
        xloc = enu * 2 / mvr
        br = 0.0  # TODO
        lines.append(SpectrumLine(xloc=xloc, br=br))

    return Spectrum(x=x, dndx=dndx, lines=lines)


def dndx_l_pi(self: RhNeutrinoBase, x: RealArray, product) -> Spectrum:
    mvr = self.mass
    mpi = ChargedPion.mass
    ml = _lepton_masses[int(self.gen)]
    epi = (mvr**2 + mpi**2 - ml**2) / (2 * mvr)
    el = (mvr**2 - mpi**2 + ml**2) / (2 * mvr)
    e = mvr * x / 2

    dnde = np.zeros_like(x)
    lines: List[SpectrumLine] = []
    if product == 21:
        dnde = spectra.dnde_photon_charged_pion(0.5 * mvr * x, epi)
        if self.gen == Gen.Snd:
            dnde += spectra.dnde_photon_muon(0.5 * mvr * x, el)
        dnde += dnde_photon_ap_fermion(e=e, s=mvr**2, mass=ml, charge=-1)
        dnde += dnde_photon_ap_scalar(e=e, s=mvr**2, mass=mpi, charge=1)

    elif product in [12, 14, 16]:
        dnde += mvr / 2 * spectra.dnde_neutrino_charged_pion(e, epi)[self.gen]
        if self.gen == Gen.Snd:
            dnde += mvr / 2 * spectra.dnde_neutrino_muon(e, el)[self.gen]

    return Spectrum(x=x, dndx=mvr / 2 * dnde, lines=lines)


def dndx_l_k(self: RhNeutrinoBase, x: RealArray, product) -> Spectrum:
    mvr = self.mass
    mk = ChargedKaon.mass
    ml = _lepton_masses[int(self.gen)]
    ek = (mvr**2 + mk**2 - ml**2) / (2 * mvr)
    el = (mvr**2 - mk**2 + ml**2) / (2 * mvr)
    e = mvr * x / 2

    dnde = np.zeros_like(x)
    lines: List[SpectrumLine] = []
    if product == 21:
        dnde = spectra.dnde_photon_charged_kaon(e, ek)
        if self.gen == Gen.Snd:
            dnde += spectra.dnde_photon_muon(0.5 * mvr * x, el)
        dnde += dnde_photon_ap_fermion(e=e, s=mvr**2, mass=ml, charge=-1)
        dnde += dnde_photon_ap_scalar(e=e, s=mvr**2, mass=ek, charge=1)

    elif product in [12, 14, 16]:
        pass

    return Spectrum(x=x, dndx=mvr / 2 * dnde, lines=lines)


def dndx_v_a(self: RhNeutrinoBase, x, product, br) -> Spectrum:
    mvr = self.mass
    xloc = 1.0

    dnde = np.zeros_like(x)
    lines: List[SpectrumLine] = []
    if product == 21:
        lines.append(SpectrumLine(xloc=xloc, br=br))
    elif product == 12 and self.gen == Gen.Fst:
        lines.append(SpectrumLine(xloc=xloc, br=br))
    elif product == 14 and self.gen == Gen.Snd:
        lines.append(SpectrumLine(xloc=xloc, br=br))
    elif product == 16 and self.gen == Gen.Trd:
        lines.append(SpectrumLine(xloc=xloc, br=br))

    return Spectrum(x=x, dndx=mvr / 2 * dnde, lines=lines)


def dndx_v_l_l(
    self: RhNeutrinoBase, x: RealArray, product, *, genv: Gen, genl1: Gen, genl2: Gen
):
    raise NotImplementedError()


def dndx_v_v_v(
    self: RhNeutrinoBase, x: RealArray, product, *, genv1: Gen, genv2: Gen, genv3: Gen
):
    raise NotImplementedError()


def dndx_v_pi_pi(self: RhNeutrinoBase, x: RealArray, product):
    raise NotImplementedError()


def dndx_l_pi_pi0(self: RhNeutrinoBase, x: RealArray, product, *, npts: int = 10_000):
    raise NotImplementedError()
