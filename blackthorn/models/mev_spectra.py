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

_charged_leptons = [fields.Electron, fields.Muon, fields.Tau]
_neutrinos = [fields.ElectronNeutrino, fields.MuonNeutrino, fields.TauNeutrino]
_lepton_masses = [Electron.mass, Muon.mass, Tau.mass]


class DecaySpectrumVPi0(DecaySpectrumTwoBody):
    def __init__(self, model: RhNeutrinoBase, branching_ratio: float = 1):
        if model.gen == Gen.Fst:
            f1 = fields.ElectronNeutrino
        elif model.gen == Gen.Fst:
            f1 = fields.MuonNeutrino
        else:
            f1 = fields.TauNeutrino

        super().__init__(model, f1, fields.NeutralPion, branching_ratio)


class DecaySpectrumVEta(DecaySpectrumTwoBody):
    def __init__(self, model: RhNeutrinoBase, branching_ratio: float = 1):
        if model.gen == Gen.Fst:
            f1 = fields.ElectronNeutrino
        elif model.gen == Gen.Fst:
            f1 = fields.MuonNeutrino
        else:
            f1 = fields.TauNeutrino

        super().__init__(model, f1, fields.Eta, branching_ratio)


class DecaySpectrumLPi(DecaySpectrumTwoBody):
    def __init__(self, model: RhNeutrinoBase, branching_ratio: float = 1):
        if model.gen == Gen.Fst:
            f1 = fields.Electron
        elif model.gen == Gen.Fst:
            f1 = fields.Muon
        else:
            f1 = fields.Tau

        super().__init__(model, f1, fields.ChargedPion, branching_ratio)


class DecaySpectrumLK(DecaySpectrumTwoBody):
    def __init__(self, model: RhNeutrinoBase, branching_ratio: float = 1):
        if model.gen == Gen.Fst:
            f1 = fields.Electron
        elif model.gen == Gen.Fst:
            f1 = fields.Muon
        else:
            f1 = fields.Tau

        super().__init__(model, f1, fields.ChargedKaon, branching_ratio)


class DecaySpectrumVA(DecaySpectrumTwoBody):
    def __init__(self, model: RhNeutrinoBase, branching_ratio: float = 1):
        if model.gen == Gen.Fst:
            f1 = fields.ElectronNeutrino
        elif model.gen == Gen.Fst:
            f1 = fields.MuonNeutrino
        else:
            f1 = fields.TauNeutrino

        super().__init__(model, f1, fields.Photon, branching_ratio)


class DecaySpectrumVLL(DecaySpectrumThreeBody):
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

        f1 = _neutrinos[genv]
        f2 = _charged_leptons[genl1]
        f3 = _charged_leptons[genl2]

        def msqrd(momenta):
            return msqrd_n_to_v_l_l(
                model, momenta=momenta, genv=genv, genl1=genl1, genl2=genl2
            )

        super().__init__(
            model,
            f1,
            f2,
            f3,
            msqrd,
            npts=npts,
            nbins=nbins,
            branching_fraction=branching_fraction,
        )


class DecaySpectrumVPiPi(DecaySpectrumThreeBody):
    def __init__(
        self,
        model: RhNeutrinoBase,
        *,
        npts: int,
        nbins: int,
        branching_fraction: float = 1,
    ):

        f1 = _neutrinos[model.gen]
        f2 = fields.ChargedPion
        f3 = fields.NeutralPion

        def msqrd(momenta):
            return msqrd_n_to_v_pi_pi(model, momenta=momenta)

        super().__init__(
            model,
            f1,
            f2,
            f3,
            msqrd,
            npts=npts,
            nbins=nbins,
            branching_fraction=branching_fraction,
        )


class DecaySpectrumLPiPi0(DecaySpectrumThreeBody):
    def __init__(
        self,
        model: RhNeutrinoBase,
        *,
        npts: int,
        nbins: int,
        branching_fraction: float = 1,
    ):

        f1 = _neutrinos[model.gen]
        f2 = fields.ChargedPion
        f3 = fields.NeutralPion

        def msqrd(momenta):
            return msqrd_n_to_l_pi_pi0(model, momenta=momenta)

        super().__init__(
            model,
            f1,
            f2,
            f3,
            msqrd,
            npts=npts,
            nbins=nbins,
            branching_fraction=branching_fraction,
        )


class DecaySpectrumVVV(DecaySpectrumThreeBody):
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

        f1 = _neutrinos[genv1]
        f2 = _neutrinos[genv2]
        f3 = _neutrinos[genv3]

        def msqrd(momenta):
            return msqrd_n_to_v_v_v(
                model, momenta=momenta, genv1=genv1, genv2=genv2, genv3=genv3
            )

        super().__init__(
            model,
            f1,
            f2,
            f3,
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
