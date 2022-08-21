from typing import Union, Callable, TypeVar
import functools

import numpy as np
import numpy.typing as npt
from hazma import spectra
from hazma.spectra import altarelli_parisi

from .. import fields

RealArray = npt.NDArray[np.float64]
RealOrRealArray = Union[npt.NDArray[np.float64], float]


T = TypeVar("T", RealArray, float)
Dndx = Callable[[T, float], T]


def call_with_mev(fn):
    @functools.wraps(fn)
    def wrapper(product_energy, parent_energy, *args, **kwargs):
        return fn(product_energy * 1e3, parent_energy * 1e3, *args, **kwargs) * 1e3

    return wrapper


def dndx_photon_fsr_electron(e, cme):
    mass = fields.Electron.mass
    return altarelli_parisi.dnde_photon_ap_fermion(e, cme**2, mass=mass, charge=-1)


def dndx_photon_fsr_tau(e, cme):
    mass = fields.Tau.mass
    return altarelli_parisi.dnde_photon_ap_fermion(e, cme**2, mass=mass, charge=-1)


# ============================================================================
# ---- Muon ------------------------------------------------------------------
# ============================================================================


@call_with_mev
def dndx_photon_muon(e, emu):
    return spectra.dnde_photon_muon(e, emu)


@call_with_mev
def dndx_positron_muon(e, emu):
    return spectra.dnde_positron_muon(e, emu)


@call_with_mev
def dndx_neutrino_muon(e: RealArray, emu: float, flavor: str):
    return spectra.dnde_neutrino_muon(e, emu, flavor=flavor)


def dndx_photon_fsr_muon(e, cme):
    mass = fields.Muon.mass
    return altarelli_parisi.dnde_photon_ap_fermion(e, cme**2, mass=mass, charge=-1)


# ============================================================================
# ---- Charged Pion ----------------------------------------------------------
# ============================================================================


@call_with_mev
def dndx_photon_charged_pion(e, epi):
    return spectra.dnde_photon_charged_pion(e, epi)


@call_with_mev
def dndx_positron_charged_pion(e, epi):
    return spectra.dnde_positron_charged_pion(e, epi)


@call_with_mev
def dndx_neutrino_charged_pion(e: RealArray, epi: float, flavor: str):
    return spectra.dnde_neutrino_charged_pion(e, epi, flavor=flavor)


def dndx_photon_fsr_charged_pion(e, cme):
    mass = fields.ChargedPion.mass
    return altarelli_parisi.dnde_photon_ap_scalar(e, cme**2, mass=mass, charge=-1)


# ============================================================================
# ---- Neutral Pion ----------------------------------------------------------
# ============================================================================


@call_with_mev
def dndx_photon_neutral_pion(e, epi):
    return spectra.dnde_photon_neutral_pion(e, epi)


# ============================================================================
# ---- Charged Kaon ----------------------------------------------------------
# ============================================================================


@call_with_mev
def dndx_photon_charged_kaon(e, ek):
    return spectra.dnde_photon_charged_kaon(e, ek)


@call_with_mev
def dndx_positron_charged_kaon(e, ek):
    return spectra.dnde_positron_charged_kaon(e, ek)


@call_with_mev
def dndx_neutrino_charged_kaon(e: RealArray, ek: float, flavor: str):
    return spectra.dnde_neutrino_charged_kaon(e, ek, flavor=flavor)


def dndx_photon_fsr_charged_kaon(e, cme):
    mass = fields.ChargedKaon.mass
    return altarelli_parisi.dnde_photon_ap_scalar(e, cme**2, mass=mass, charge=-1)


# ============================================================================
# ---- Long Kaon -------------------------------------------------------------
# ============================================================================


@call_with_mev
def dndx_photon_long_kaon(e, ek):
    return spectra.dnde_photon_long_kaon(e, ek)


@call_with_mev
def dndx_positron_long_kaon(e, ek):
    return spectra.dnde_positron_long_kaon(e, ek)


@call_with_mev
def dndx_neutrino_long_kaon(e: RealArray, ek: float, flavor: str):
    return spectra.dnde_neutrino_long_kaon(e, ek, flavor=flavor)


# ============================================================================
# ---- Short Kaon -------------------------------------------------------------
# ============================================================================


@call_with_mev
def dndx_photon_short_kaon(e, ek):
    return spectra.dnde_photon_short_kaon(e, ek)


@call_with_mev
def dndx_positron_short_kaon(e, ek):
    return spectra.dnde_positron_short_kaon(e, ek)


@call_with_mev
def dndx_neutrino_short_kaon(e: RealArray, ek: float, flavor: str):
    return spectra.dnde_neutrino_short_kaon(e, ek, flavor=flavor)


# ============================================================================
# ---- Eta -------------------------------------------------------------------
# ============================================================================


@call_with_mev
def dndx_photon_eta(e, eeta):
    return spectra.dnde_photon_eta(e, eeta)


@call_with_mev
def dndx_positron_eta(e, eeta):
    return spectra.dnde_positron_eta(e, eeta)


@call_with_mev
def dndx_neutrino_eta(e: RealArray, eeta: float, flavor: str):
    return spectra.dnde_neutrino_eta(e, eeta, flavor=flavor)


# ============================================================================
# ---- Eta -------------------------------------------------------------------
# ============================================================================


dndx_photon = {
    fields.Muon.pdg: dndx_photon_muon,
    fields.ChargedPion.pdg: dndx_photon_charged_pion,
    fields.ChargedKaon.pdg: dndx_photon_charged_kaon,
    fields.LongKaon.pdg: dndx_photon_long_kaon,
    fields.ShortKaon.pdg: dndx_photon_short_kaon,
    fields.Eta.pdg: dndx_photon_eta,
}
dndx_positron = {
    fields.Muon.pdg: dndx_positron_muon,
    fields.ChargedPion.pdg: dndx_positron_charged_pion,
    fields.ChargedKaon.pdg: dndx_positron_charged_kaon,
    fields.LongKaon.pdg: dndx_positron_long_kaon,
    fields.ShortKaon.pdg: dndx_positron_short_kaon,
    fields.Eta.pdg: dndx_positron_eta,
}
dndx_neutrino = {
    fields.Muon.pdg: dndx_neutrino_muon,
    fields.ChargedPion.pdg: dndx_neutrino_charged_pion,
    fields.ChargedKaon.pdg: dndx_neutrino_charged_kaon,
    fields.LongKaon.pdg: dndx_neutrino_long_kaon,
    fields.ShortKaon.pdg: dndx_neutrino_short_kaon,
    fields.Eta.pdg: dndx_neutrino_eta,
}
dndx_electron_neutrino = {
    key: functools.partial(fn, flavor="e") for key, fn in dndx_neutrino.items()
}
dndx_muon_neutrino = {
    key: functools.partial(fn, flavor="mu") for key, fn in dndx_neutrino.items()
}
dndx_tau_neutrino = {
    key: functools.partial(fn, flavor="tau") for key, fn in dndx_neutrino.items()
}

dndx_spectra = {
    fields.Photon.pdg: dndx_photon,
    fields.Positron.pdg: dndx_positron,
    fields.ElectronNeutrino.pdg: dndx_electron_neutrino,
    fields.MuonNeutrino.pdg: dndx_muon_neutrino,
    fields.TauNeutrino.pdg: dndx_tau_neutrino,
}


class HazmaSpectra:
    energy_min: float = 0.0
    energy_max: float = 0.5 * 0.5
    xmin: float = 0.0
    xmax: float = 1.0

    def __init__(self):
        pass

    def _dndx(self, x, energy, final_state: int, product: int):
        return dndx_spectra[product][final_state](x, energy)
