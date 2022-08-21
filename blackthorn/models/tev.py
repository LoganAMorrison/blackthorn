from typing import Dict, Optional, Union, Tuple
import warnings

import numpy as np

from ..constants import Gen
from ..spectrum_utils import HDMSpectra, Spectrum, SpectrumLine
from .. import fields
from .base import RhNeutrinoBase, RealArray

_leptons = [fields.Electron, fields.Muon, fields.Tau]
_neutrinos = [fields.ElectronNeutrino, fields.MuonNeutrino, fields.TauNeutrino]


class RhNeutrinoTeV(RhNeutrinoBase):
    from .gev_widths import width_n_to_h_v as __width_h_v
    from .gev_widths import width_n_to_w_l as __width_w_l
    from .gev_widths import width_n_to_z_v as __width_z_v

    def __init__(self, mass: float, theta: float, gen: Gen) -> None:
        super().__init__(mass, theta, gen)

    def width_v_h(self) -> float:
        return self.__width_h_v()

    def width_v_z(self) -> float:
        return self.__width_z_v()

    def width_l_w(self) -> float:
        return self.__width_w_l()

    def partial_widths(self) -> Dict[str, float]:
        vv = self._nustr
        ll = self._lepstr
        return {
            f"{vv} h": self.width_v_h(),
            f"{vv} z": self.width_v_z(),
            f"{ll} w": 2 * self.width_l_w(),
        }

    def branching_fractions(self) -> Dict[str, float]:
        pws = self.partial_widths()
        tot = sum(pws.values())

        bfs = {key: 0.0 for key in pws.keys()}

        if tot > 0:
            for key, val in pws.items():
                bfs[key] = val / tot

        return bfs

    def _dndx(
        self,
        x: RealArray,
        finalstates: Tuple[fields.QuantumField, fields.QuantumField],
        product: fields.QuantumField,
        delta: bool = False,
        interpolation: str = "cubic",
        include_cc: bool = False,
    ) -> RealArray:
        stable_pdg = [
            fields.Photon.pdg,
            fields.Electron.pdg,
            fields.Positron.pdg,
            fields.ElectronNeutrino.pdg,
            fields.MuonNeutrino.pdg,
            fields.TauNeutrino.pdg,
        ]
        assert product.pdg in stable_pdg, "Invalid product."

        dndx = HDMSpectra.dndx(
            x=x,
            cme=self.mass,
            final_state=finalstates[0].pdg,
            final_state_bar=finalstates[1].pdg,
            product=product.pdg,
            delta=delta,
            interpolation=interpolation,
        )
        if include_cc:
            dndx = HDMSpectra.dndx(
                x=x,
                cme=self.mass,
                final_state=-finalstates[0].pdg,
                final_state_bar=-finalstates[1].pdg,
                product=product.pdg,
                delta=delta,
                interpolation=interpolation,
            )

        return dndx

    def dndx_l_w(self, x: RealArray, product: fields.QuantumField) -> Spectrum:
        """
        Compute the spectrum from N -> l + w.

        Parameters
        ----------
        xs: np.ndarray
            Array of x = 2E/m.

        Returns
        -------
        dndx: np.ndarray
            Array containing decay spectrum from N -> l + w.
        """
        lep = _leptons[self.gen]
        w = fields.WBoson
        dndx = self._dndx(x, (lep, w), product, delta=True, include_cc=True)
        lines = [SpectrumLine(xloc=1, br=dndx[-1])]
        return Spectrum(x, dndx[:-1], lines=lines)

    def dndx_v_h(self, x: RealArray, product: fields.QuantumField) -> Spectrum:
        """
        Compute the spectrum from N -> nu + h.

        Parameters
        ----------
        xs: np.ndarray
            Array of x = 2E/m.

        Returns
        -------
        dndx: np.ndarray
            Array containing decay spectrum from N -> nu + h.
        """
        h = fields.Higgs
        if self.gen == Gen.Fst:
            nu = fields.ElectronNeutrino
        elif self.gen == Gen.Snd:
            nu = fields.MuonNeutrino
        else:
            nu = fields.TauNeutrino

        dndx = self._dndx(x, (h, nu), product, delta=True)
        lines = [SpectrumLine(xloc=1, br=dndx[-1])]
        return Spectrum(x, dndx[:-1], lines=lines)

    def dndx_v_z(self, x: RealArray, product: fields.QuantumField) -> Spectrum:
        """
        Compute the spectrum from N -> nu + Z.

        Parameters
        ----------
        xs: np.ndarray
            Array of x = 2E/m.

        Returns
        -------
        dndx: np.ndarray
            Array containing decay spectrum from N -> nu + Z.
        """
        z = fields.ZBoson
        if self.gen == Gen.Fst:
            nu = fields.ElectronNeutrino
        elif self.gen == Gen.Snd:
            nu = fields.MuonNeutrino
        else:
            nu = fields.TauNeutrino

        dndx = self._dndx(x, (z, nu), product, delta=True)
        lines = [SpectrumLine(xloc=1, br=dndx[-1])]
        return Spectrum(x, dndx[:-1], lines=lines)

    def dndx_components(
        self, x: RealArray, product: fields.QuantumField
    ) -> Dict[str, Spectrum]:
        """
        Compute all components of the decay spectrum of the right-handed neutrino.

        Parameters
        ----------
        xs: np.ndarray
            Array of x = 2E/m.

        Returns
        -------
        dndx: Dict[str, np.ndarry]
            Dictionary containing all decay spectrum of the right-handed neutrino.
        """
        ll = self._lepstr
        vv = self._nustr

        bfs = self.branching_fractions()

        lep = _leptons[self.gen]
        h = fields.Higgs
        w = fields.WBoson
        z = fields.ZBoson
        if self.gen == Gen.Fst:
            nu = fields.ElectronNeutrino
        elif self.gen == Gen.Snd:
            nu = fields.MuonNeutrino
        else:
            nu = fields.TauNeutrino

        spec = {}

        key = f"{vv} h"
        br = bfs[key]
        dndx = self._dndx(x, (h, nu), product, delta=True)
        spec[key] = Spectrum(
            x,
            br * dndx[:-1],
            lines=[SpectrumLine(xloc=1, br=br * dndx[-1])],
        )

        key = f"{vv} z"
        br = bfs[key]
        dndx = self._dndx(x, (z, nu), product, delta=True)
        spec[key] = Spectrum(
            x,
            br * dndx[:-1],
            lines=[SpectrumLine(xloc=1, br=br * dndx[-1])],
        )

        key = f"{ll} w"
        # 1/2 for single fs. The w^+ l^- and w^- l^+ are included by
        # using include_cc=True
        br = 0.5 * bfs[key]
        dndx = self._dndx(x, (w, lep), product, delta=True, include_cc=True)
        spec[key] = Spectrum(
            x,
            br * dndx[:-1],
            lines=[SpectrumLine(xloc=1, br=br * dndx[-1])],
        )

        return spec

    def dndx(self, x: RealArray, product: fields.QuantumField) -> Spectrum:
        """
        Compute total decay spectrum of the right-handed neutrino.

        Parameters
        ----------
        xs: np.ndarray
            Array of x = 2E/m.
        nbins: optional, int
            Number of bins to use in creating invariant-mass distribution for
            final-state down-quarks.
        nevents: optional, int
            Number of Monte-Carlo points to use in estimating invariant-mass
            distribution.

        Returns
        -------
        dndx: np.ndarray
            Array the total decay spectrum of the right-handed neutrino.
        """
        dndx = Spectrum(x, np.zeros_like(x))
        specs = self.dndx_components(x, product)
        for key, val in specs.items():
            if len(np.nonzero(val == np.nan)[0]) > 0:
                warnings.warn(f"NaN encountered in {key}. Setting to zero.")
                dndx += Spectrum(x, np.nan_to_num(val.dndx), lines=val._lines)
            else:
                dndx = dndx + val

        return dndx

    @staticmethod
    def dndx_bb(x, cme: float, product: int, eps=0.1):
        dndx = HDMSpectra.dndx(x, cme, fields.BottomQuark.pdg, product)
        return Spectrum(x, dndx).convolve(eps)
