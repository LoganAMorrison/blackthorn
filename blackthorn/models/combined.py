"""Right-handed neutrino model valid for any mass."""

# pylint: disable=invalid-name

from typing import Dict, Optional, Union

import numpy as np

from blackthorn import fields
from blackthorn.constants import Gen
from blackthorn.spectrum_utils import HDMSpectra, PPPC4DMIDSpectra, Spectrum

from .base import RealArray
from .gev import RhNeutrinoGeV
from .mev import RhNeutrinoMeV
from .tev import RhNeutrinoTeV

RhNeutrinoModelType = Union[RhNeutrinoMeV, RhNeutrinoGeV, RhNeutrinoTeV]


class RhNeutrino:
    """RH neutrino model with any mass.

    The model assumes a single RH neutrino that mixes with the standard model
    through a Yukawa interaction with only one of the lepton doublets. Thus,
    the RH neutrino only mixes with a single active neutrino generation.

    Attributes
    ----------
    mass: float
        Mass of the RH neutrino in GeV.
    gen: Gen
        Generation of the RH neutrino.
    theta: float
        Mixing angle between RH neutrino and active neutrino.
    mass_break_points: tuple[float,float]
        Masses at which model switches from MeV -> GeV and from GeV -> TeV.
        Default is 5.0 GeV and 1 TeV.


    Methods
    -------
    partial_widths()
        Compute the partial decay widths of the RH neutrino.
    branching_fractions()
        Compute the partial decay branching fractions of the RH neutrino.
    dndx_components(x, product, npts=10000, nbins=25, apply_br=True)
        Compute the spectral components into a given product for all decay
        final states.
    dndx(x, product, npts=10000, nbins=25, apply_br=True)
        Compute the total spectrum into a given product.
    """

    def __init__(
        self,
        mass: float,
        gen: Gen,
        theta: float,
        mass_break_points: tuple[float, float] = (5.0, 1e3),
    ) -> None:
        """
        Parameters
        ----------
        mass: float
            Mass of the RH neutrino in GeV.
        gen: Gen
            Generation of the RH neutrino.
        theta: float
            Mixing angle between RH neutrino and active neutrino.
        """
        self.__mass = mass
        self.__gen = gen
        self.__theta = theta
        self.__mass_break_points = mass_break_points

        self.__model = self.__make_model()

    def __make_model(self):
        mass = self.__mass
        theta = self.__theta
        gen = self.__gen
        mev_gev_bp, gev_tev_bp = self.__mass_break_points

        if 0 < mass < mev_gev_bp:
            return RhNeutrinoMeV(mass, theta, gen)

        if mev_gev_bp <= mass < gev_tev_bp:
            return RhNeutrinoGeV(mass, theta, gen)

        if gev_tev_bp <= mass:
            return RhNeutrinoTeV(mass, theta, gen)

        raise ValueError(f"Invalid mass: {mass}.")

    @property
    def mass(self) -> float:
        """Mass of the RH-neutrino in GeV."""
        return self.__mass

    @mass.setter
    def mass(self, mass: float) -> None:
        self.__mass = mass
        self.__model = self.__make_model()

    @property
    def gen(self) -> Gen:
        """Generation of the RH-neutrino."""
        return self.__gen

    @gen.setter
    def gen(self, gen: Gen) -> None:
        self.__gen = gen
        self.__model = self.__make_model()

    @property
    def theta(self) -> float:
        """Mixing angle between RH-neutrino and active neutrino."""
        return self.__theta

    @theta.setter
    def theta(self, theta: float) -> None:
        self.__theta = theta
        self.__model = self.__make_model()

    def partial_widths(self) -> Dict[str, float]:
        r"""Compute the partial widths of the RH neutrino.

        Returns
        -------
        pws: Dict[str, float]
            Dictionary containing all the partial widths of the RH neutrino.
        """
        return self.__model.partial_widths()

    def branching_fractions(self) -> Dict[str, float]:
        r"""Compute the decay branching fractions of the RH neutrino.

        Returns
        -------
        brs: Dict[str, float]
            Dictionary containing all the branching fractions of the RH neutrino.
        """
        return self.__model.branching_fractions()

    def dndx_components(
        self,
        x: RealArray,
        product: fields.QuantumField,
        *,
        npts: int = 10_000,
        nbins: int = 25,
        apply_br: bool = True,
    ) -> Dict[str, Spectrum]:
        r"""Compute RH neutrino decay spectra into a given product.

        Parameters
        ----------
        x: np.ndarray[float]
            Scaled energy values x=2E/sqrt(s).
        product: QuantumField
            Product to compute spectrum of.
        npts: int, optional
            Number of values to use in generating energy spectra of final
            states containing more than 2 particles. Default is 10_000.
        nbins: int, optional
            Number of energy bins used to construct energy distributions of
            final state particles. Default is 25.
        apply_br: bool, optional
            If `True`, the branching fractions are applied to each spectrum
            component.

        Returns
        -------
        dndx: Spectrum
            Spectrum object containing the resulting differential energy spectrum.
        """
        if isinstance(self.__model, RhNeutrinoTeV):
            # TeV model has no 3-body final states. No need for additional args.
            return self.__model.dndx_components(x=x, product=product)

        return self.__model.dndx_components(
            x=x, product=product, npts=npts, nbins=nbins, apply_br=apply_br
        )

    def dndx(
        self,
        x: RealArray,
        product: fields.QuantumField,
        *,
        npts: int = 10_000,
        nbins: int = 25,
    ) -> Spectrum:
        r"""Compute the spectrum into a given product from RH neutrino decays.

        Parameters
        ----------
        x: np.ndarray[float]
            Scaled energy values x=2E/sqrt(s).
        product: QuantumField
            Product to compute spectrum of.
        npts: int, optional
            Number of values to use in generating energy spectra of final
            states containing more than 2 particles. Default is 10_000.
        nbins: int, optional
            Number of energy bins used to construct energy distributions of
            final state particles. Default is 25.

        Returns
        -------
        dndx: Spectrum
            Spectrum object containing the resulting differential energy spectrum.
        """
        components = self.dndx_components(
            x, product, npts=npts, nbins=nbins, apply_br=True
        )
        dndx = np.zeros_like(x)

        for spec in components.values():
            dndx += spec.dndx

        return Spectrum(x, dndx)

    def dndx_bottom_quark(
        self,
        x: RealArray,
        product: fields.QuantumField,
        cme: Optional[float] = None,
        eps: float = 0.1,
    ) -> Spectrum:
        r"""Compute the spectrum from a pair of bottom quarks.

        This method is used to compare to RH neutrino decay spectra. If the
        center-of-mass energy isn't specified, the RH neutrino mass is used. If
        the center-of-mass energy is below the half the bottom quark mass, the
        resulting spectrum is zero.

        Parameters
        ----------
        x: np.ndarray[float]
            Scaled energy values x=2E/sqrt(s).
        product: QuantumField
            Product to compute spectrum of.
        cme: float, optional
            Center-of-mass energy of the bottom quarks. If not specified, RH
            neutrino mass is used. Default is None.
        eps: float, optional
            Energy resolution fraction dE/E. Default is 0.1 (10%).

        Returns
        -------
        dndx: Spectrum
            Spectrum object containing the resulting differential energy spectrum.
        """
        if cme is None:
            cme = self.mass

        if isinstance(self.__model, RhNeutrinoTeV):
            dndx = HDMSpectra.dndx(x, cme, fields.BottomQuark.pdg, product.pdg)
            return Spectrum(x, dndx).convolve(eps)

        if isinstance(self.__model, RhNeutrinoGeV):
            dndx = PPPC4DMIDSpectra.dndx(x, cme, "b", product.pdg, single=False)
            return Spectrum(x, dndx).convolve(eps)

        return Spectrum(x, np.zeros_like(x))
