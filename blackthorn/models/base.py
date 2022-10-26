"""
blah
"""
import abc
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
from hazma.phase_space import PhaseSpaceDistribution1D, Rambo
from helax.numpy.phase_space import PhaseSpace
from scipy import interpolate

from .. import fields, spectrum_utils
from ..constants import Gen
from ..fields import Higgs
from ..spectrum_utils import Spectrum, SpectrumLine, convolved_spectrum_fn
from .utils import energies_two_body_final_state

RealArray = npt.NDArray[np.float64]
RealOrComplex = Union[float, complex]
InvariantMassDistributions = Dict[Tuple[int, int], PhaseSpaceDistribution1D]

T = TypeVar("T", RealArray, float)
Dndx = Callable[[T, float], T]


def _convolve(x: RealArray, f: Dndx, dpdys: RealArray, ys: RealArray) -> RealArray:
    """
    Convolve function with distribution.
    """
    fs = np.array([p * f(x, y) for (p, y) in zip(dpdys, ys)])
    return np.trapz(fs, ys, axis=0)


def _convolve_dist(x: RealArray, f: Dndx, dist: PhaseSpaceDistribution1D) -> RealArray:
    """
    Convolve function with distribution.
    """
    fs = np.array([p * f(x, y) for (p, y) in zip(dist.probabilities, dist.bin_centers)])
    return np.trapz(fs, dist.bin_centers, axis=0)


class RhNeutrinoGeneralBase:
    def __init__(self, theta, alpha, beta, mass) -> None:
        vev = Higgs.vev
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        ct, st = np.cos(theta), np.sin(theta)

        y = np.sqrt(2) * mass * np.tan(theta) / vev
        ye = y * sa * cb
        ym = y * sa * sb
        yt = y * ca
        mu = mass * (1.0 - np.tan(theta) ** 2)

        mde = vev * ye / np.sqrt(2)
        mdm = vev * ym / np.sqrt(2)
        mdt = vev * yt / np.sqrt(2)

        im = 1.0j
        omega = np.array(
            [
                [ca * cb, -sb, im * cb * ct * sa, cb * sa * st],
                [ca * sb, cb, im * ct * sa * sb, sa * sb * st],
                [-sa, 0, im * ca * ct, ca * st],
                [0, 0, -im * st, ct],
            ]
        )

        kl = omega[:3, :3]
        kr = omega[:3, 3]

        mass_mat = np.array(
            [
                [0.0, 0.0, 0.0, mde],
                [0.0, 0.0, 0.0, mdm],
                [0.0, 0.0, 0.0, mdt],
                [mde, mdm, mdt, mu],
            ]
        )


class RhNeutrinoBase:
    def __init__(self, mass: float, theta: float, gen: Gen) -> None:
        self._mass = mass
        self._theta = theta
        self._gen = gen

        if gen == Gen.Fst:
            self._lepstr = "e"
            self._nustr = "ve"
        elif gen == Gen.Snd:
            self._lepstr = "mu"
            self._nustr = "vmu"
        else:
            self._lepstr = "tau"
            self._nustr = "vtau"

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def gen(self) -> Gen:
        return self._gen

    @mass.setter
    def mass(self, mass: float) -> None:
        self._mass = mass

    @theta.setter
    def theta(self, theta: float) -> None:
        self._theta = theta

    @gen.setter
    def gen(self, gen: Gen) -> None:
        self._gen = gen

    @abc.abstractmethod
    def dndx(self, x: RealArray, product: fields.QuantumField, **kwargs) -> Spectrum:
        pass

    @abc.abstractmethod
    def lines(self, product: fields.QuantumField, **kwargs) -> Dict[str, float]:
        pass

    def dnde(self, e: RealArray, product: fields.QuantumField, **kwargs) -> RealArray:
        cme = kwargs.get("cme")
        kwargs = {key: val for key, val in kwargs.items() if not key == "cme"}
        if cme is not None:
            mn = self.mass
            if cme < 2 * mn:
                return np.zeros_like(e)

            gamma = cme / (2.0 * mn)
            beta = np.sqrt(1.0 - gamma**-2)
            dndx = self.dndx(2 * e / self.mass, product, **kwargs)
            dndx2 = dndx.boost(beta=beta)
            return 2.0 / cme * dndx2.dndx

        return 2.0 / self.mass * self.dndx(2 * e / self.mass, product, **kwargs).dndx

    def total_conv_spectrum_fn(
        self,
        e_min,
        e_max,
        energy_res,
        product: fields.QuantumField,
        npts=1000,
        aeff: Optional[Callable] = None,
        cme: Optional[float] = None,
    ) -> Callable:

        products = [product]
        nus = [fields.ElectronNeutrino, fields.MuonNeutrino, fields.TauNeutrino]

        if self.mass < e_min:

            def f(e):
                return np.zeros_like(e)

            f.integral = lambda a, b: 0.0
            return f

        if product in nus:
            products = [
                fields.ElectronNeutrino,
                fields.MuonNeutrino,
                fields.TauNeutrino,
            ]
            pre = 1.0 / 3.0
        else:
            pre = 1.0

        def dnde(e):
            res = np.zeros_like(e)
            for product in products:
                res += self.dnde(e, product, cme=cme)
            return res * pre

        return convolved_spectrum_fn(
            e_min=e_min,
            e_max=e_max,
            energy_res=energy_res,
            spec_fn=dnde,
            lines=self.lines(product),
            n_pts=npts,
            aeff=aeff,
        )


class PartialWidth(abc.ABC):
    def __init__(self, phase_space: PhaseSpace, extra_factor: float = 1):
        self._phase_space = phase_space
        self._extra_factor = extra_factor

    def accessible(self) -> bool:
        cme = self._phase_space.cme
        fsp_masses = self._phase_space.masses
        return cme > np.sum(fsp_masses)

    def width(self, *, npts: int = 10_000) -> float:
        if not self.accessible():
            return 0.0
        return self._phase_space.decay_width(n=npts)[0] * self._extra_factor  # type: ignore

    def energy_distributions(
        self, *, npts: int = 10_000, nbins: int = 25
    ) -> List[PhaseSpaceDistribution1D]:
        if not self.accessible():
            return []

        return self._phase_space.energy_distributions(n=npts, nbins=nbins)

    def invariant_mass_distributions(
        self, *, npts: int = 10_000, nbins: int = 25
    ) -> Dict[Tuple[int, int], PhaseSpaceDistribution1D]:
        if not self.accessible():
            return dict()
        return self._phase_space.invariant_mass_distributions(n=npts, nbins=nbins)


class DecaySpectrum(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def dndx_photon(self, x: RealArray) -> Spectrum:
        pass

    @abc.abstractmethod
    def dndx_positron(self, x: RealArray) -> Spectrum:
        pass

    @abc.abstractmethod
    def dndx_neutrino(self, x: RealArray, genv: Gen) -> Spectrum:
        pass

    @abc.abstractmethod
    def dndx(
        self, x: RealArray, product: fields.QuantumField, fsr: bool = False
    ) -> Spectrum:
        pass


class DecaySpectrumTwoBody(DecaySpectrum):
    def __init__(
        self,
        model: RhNeutrinoBase,
        f1: fields.QuantumField,
        f2: fields.QuantumField,
        branching_fraction: float = 1.0,
    ):
        self._cme = model.mass
        self._gen = model.gen
        self._theta = model.theta
        self._br = branching_fraction

        self._f1 = f1
        self._f2 = f2

        self._e1, self._e2 = energies_two_body_final_state(self._cme, f1.mass, f2.mass)

    def _dndx_product(
        self, x: RealArray, product: fields.QuantumField, fsr: bool
    ) -> Spectrum:
        to_e = self._cme / 2.0
        e = to_e * x
        dndx = np.zeros_like(x)

        if self._cme < self._f1.mass + self._f2.mass:
            return Spectrum(x, dndx)

        dnde_fsr1: Union[Dndx, None] = None
        dnde_fsr2: Union[Dndx, None] = None

        if product == fields.Photon:
            _dnde_photon = spectrum_utils.FIELD_TO_DNDE_PHOTON
            _dnde_fsr = spectrum_utils.FIELD_TO_DNDE_PHOTON_FSR

            dndx1 = _dnde_photon.get(self._f1)
            dndx2 = _dnde_photon.get(self._f2)
            dnde_fsr1 = _dnde_fsr.get(self._f1)
            dnde_fsr2 = _dnde_fsr.get(self._f2)

        elif product == fields.Electron:
            _dnde_pos = spectrum_utils.FIELD_TO_DNDE_POSITRON
            dndx1 = _dnde_pos.get(self._f1)
            dndx2 = _dnde_pos.get(self._f2)

        elif product == fields.ElectronNeutrino:
            _dnde_nu = spectrum_utils.FIELD_TO_DNDE_ELECTRON_NEUTRINO
            dndx1 = _dnde_nu.get(self._f1)
            dndx2 = _dnde_nu.get(self._f2)

        elif product == fields.MuonNeutrino:
            _dnde_nu = spectrum_utils.FIELD_TO_DNDE_MUON_NEUTRINO
            dndx1 = _dnde_nu.get(self._f1)
            dndx2 = _dnde_nu.get(self._f2)

        elif product == fields.TauNeutrino:
            _dnde_nu = spectrum_utils.FIELD_TO_DNDE_TAU_NEUTRINO
            dndx1 = _dnde_nu.get(self._f1)
            dndx2 = _dnde_nu.get(self._f2)

        else:
            raise ValueError(f"Invalid product {product.pdg}")

        if dndx1 is not None:
            dndx += to_e * dndx1(e, self._e1)

        if dndx2 is not None:
            dndx += to_e * dndx2(e, self._e2)

        if dnde_fsr1 is not None and fsr:
            dndx += to_e * dnde_fsr1(e, self._cme**2)

        if dnde_fsr2 is not None and fsr:
            dndx += to_e * dnde_fsr2(e, self._cme**2)

        lines: List[SpectrumLine] = []
        if self._f1 == product:
            lines.append(SpectrumLine(xloc=self._e1 / to_e, br=self._br))

        if self._f2 == product:
            lines.append(SpectrumLine(xloc=self._e2 / to_e, br=self._br))

        return Spectrum(x=x, dndx=dndx, lines=lines)

    def dndx_photon(self, x: RealArray, fsr: bool = False) -> Spectrum:
        return self._dndx_product(x, fields.Photon, fsr=fsr)

    def dndx_positron(self, x: RealArray) -> Spectrum:
        return self._dndx_product(x, fields.Electron, fsr=False)

    def dndx_neutrino(self, x: RealArray, genv: Gen) -> Spectrum:
        if genv == Gen.Fst:
            return self._dndx_product(x, fields.ElectronNeutrino, fsr=False)
        elif genv == Gen.Fst:
            return self._dndx_product(x, fields.MuonNeutrino, fsr=False)
        return self._dndx_product(x, fields.TauNeutrino, fsr=False)

    def dndx(
        self, x: RealArray, product: fields.QuantumField, fsr: bool = False
    ) -> Spectrum:
        return self._dndx_product(x, product, fsr=fsr)


class DecaySpectrumThreeBody(DecaySpectrum):
    def __init__(
        self,
        model: RhNeutrinoBase,
        f1: fields.QuantumField,
        f2: fields.QuantumField,
        f3: fields.QuantumField,
        msqrd,
        branching_fraction: float = 1,
        npts: int = 50_000,
        nbins: int = 25,
    ):
        self._cme = model.mass
        self._gen = model.gen
        self._theta = model.theta
        self._br = branching_fraction

        self._f1 = f1
        self._f2 = f2
        self._f3 = f3

        self._phase_space = Rambo(
            self._cme, masses=[f1.mass, f2.mass, f3.mass], msqrd=msqrd
        )

        if self._cme < self._f1.mass + self._f2.mass + self._f3.mass:
            self._energy_distributions: List[PhaseSpaceDistribution1D] = list()
            self._invariant_mass_distributions: InvariantMassDistributions = dict()
        else:
            self._energy_distributions = self._phase_space.energy_distributions(
                n=npts, nbins=nbins
            )
            self._invariant_mass_distributions = (
                self._phase_space.invariant_mass_distributions(n=npts, nbins=nbins)
            )

    def _dndx_product(
        self, x: RealArray, product: fields.QuantumField, fsr: bool
    ) -> Spectrum:
        to_e = self._cme / 2.0
        e = to_e * x
        dnde = np.zeros_like(x)

        if self._cme < self._f1.mass + self._f2.mass + self._f3.mass:
            return Spectrum(x, dnde)

        dnde_fsr1: Union[Dndx, None] = None
        dnde_fsr2: Union[Dndx, None] = None
        dnde_fsr3: Union[Dndx, None] = None

        if product == fields.Photon:
            _dnde_photon = spectrum_utils.FIELD_TO_DNDE_PHOTON
            _dnde_fsr = spectrum_utils.FIELD_TO_DNDE_PHOTON_FSR

            dnde1 = _dnde_photon.get(self._f1)
            dnde2 = _dnde_photon.get(self._f2)
            dnde3 = _dnde_photon.get(self._f3)

            dnde_fsr1 = _dnde_fsr.get(self._f1)
            dnde_fsr2 = _dnde_fsr.get(self._f2)
            dnde_fsr3 = _dnde_fsr.get(self._f3)

        elif product == fields.Electron:
            _dnde_pos = spectrum_utils.FIELD_TO_DNDE_POSITRON
            dnde1 = _dnde_pos.get(self._f1)
            dnde2 = _dnde_pos.get(self._f2)
            dnde3 = _dnde_pos.get(self._f3)

        elif product == fields.ElectronNeutrino:
            _dnde_nu = spectrum_utils.FIELD_TO_DNDE_ELECTRON_NEUTRINO
            dnde1 = _dnde_nu.get(self._f1)
            dnde2 = _dnde_nu.get(self._f2)
            dnde3 = _dnde_nu.get(self._f3)

        elif product == fields.MuonNeutrino:
            _dnde_nu = spectrum_utils.FIELD_TO_DNDE_MUON_NEUTRINO
            dnde1 = _dnde_nu.get(self._f1)
            dnde2 = _dnde_nu.get(self._f2)
            dnde3 = _dnde_nu.get(self._f3)

        elif product == fields.TauNeutrino:
            _dnde_nu = spectrum_utils.FIELD_TO_DNDE_TAU_NEUTRINO
            dnde1 = _dnde_nu.get(self._f1)
            dnde2 = _dnde_nu.get(self._f2)
            dnde3 = _dnde_nu.get(self._f3)

        else:
            raise ValueError(f"Invalid product {product.pdg}")

        # Handle decay contributions
        for i, dndef in enumerate((dnde1, dnde2, dnde3)):
            if dndef is not None:
                dist = self._energy_distributions[i]
                dnde += _convolve_dist(e, dndef, dist)

        # Handle FSR contributions
        pair_iter = (((0, 1), (0, 2)), ((0, 1), (1, 2)), ((0, 2), (1, 2)))
        dnde_fsr_iter = (dnde_fsr1, dnde_fsr2, dnde_fsr3)
        for pairs, dnde_fsrf in zip(pair_iter, dnde_fsr_iter):
            if dnde_fsrf is not None and fsr:
                for pair in pairs:
                    mdist = self._invariant_mass_distributions[pair]
                    dnde += 0.5 * to_e * _convolve_dist(e, dnde_fsrf, mdist)

        # Handle cases where final-state particle == product
        for i, f in enumerate([self._f1, self._f2, self._f3]):
            if f == product:
                dnde += self._energy_distributions[i](e)

        dndx = to_e * dnde
        return Spectrum(x, dndx)

    def dndx_photon(self, x: RealArray, fsr: bool = False) -> Spectrum:
        return self._dndx_product(x, fields.Photon, fsr)

    def dndx_positron(self, x: RealArray) -> Spectrum:
        return self._dndx_product(x, fields.Electron, fsr=False)

    def dndx_neutrino(self, x: RealArray, genv: Gen) -> Spectrum:
        if genv == Gen.Fst:
            return self._dndx_product(x, fields.ElectronNeutrino, fsr=False)
        elif genv == Gen.Fst:
            return self._dndx_product(x, fields.MuonNeutrino, fsr=False)
        return self._dndx_product(x, fields.TauNeutrino, fsr=False)

    def dndx(
        self, x: RealArray, product: fields.QuantumField, fsr: bool = False
    ) -> Spectrum:
        return self._dndx_product(x, product, fsr=fsr)
