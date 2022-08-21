from numbers import Real
from pathlib import Path
from typing import Dict, Union, List

# from HDMSpectra import HDMSpectra
import numpy as np
import numpy.typing as npt
from HDMSpectra.HDMSpectra import spec as hdm_spec

from . import fields
from .rh_neutrino import Gen
from .rh_neutrino import RhNeutrinoGeVCpp as _RhNeutrinoGeV
from .spectrum_utils import Spectrum, SpectrumLine, HDMSpectra

RealArray = npt.NDArray[np.float64]

THIS_DIR = Path(__file__).parent


class RhNeutrinoTeV:
    def __init__(self, mass: float, theta: float, gen: Gen) -> None:
        self._mass = mass
        self._theta = theta
        self._gen = gen
        self._model = _RhNeutrinoGeV(mass, theta, gen)
        self._lep: Union[fields.Electron, fields.Muon, fields.Tau]
        self._nu: Union[
            fields.ElectronNeutrino, fields.MuonNeutrino, fields.TauNeutrino
        ]

        if gen == Gen.Fst:
            self._lepstr = "e"
            self._nustr = "ve"
            self._lep = fields.Electron()
            self._nu = fields.ElectronNeutrino()
        elif gen == Gen.Snd:
            self._lepstr = "mu"
            self._nustr = "vmu"
            self._lep = fields.Muon()
            self._nu = fields.MuonNeutrino()
        else:
            self._lepstr = "tau"
            self._nustr = "vtau"
            self._lep = fields.Tau()
            self._nu = fields.TauNeutrino()

    @property
    def mass(self) -> float:
        return self._mass

    @mass.setter
    def mass(self, val: float) -> None:
        self._mass = val

    @property
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, val: float) -> None:
        self._theta = val

    @property
    def gen(self) -> Gen:
        return self._gen

    @gen.setter
    def gen(self, val: Gen) -> None:
        self._gen = val

    def width_v_h(self) -> float:
        return self._model.width_v_h()

    def width_v_z(self) -> float:
        return self._model.width_v_z()

    def width_l_w(self) -> float:
        return self._model.width_l_w()

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

    def dndx_v_h(self, x: RealArray, product: int):
        return HDMSpectra.dndx(
            x=x,
            cme=self._mass,
            final_state=fields.Higgs.pdg,
            product=product,
            final_state_bar=self._nu.pdg,
            single=False,
        )

    def dndx_v_z(self, x: RealArray, product: int):
        return HDMSpectra.dndx(
            x=x,
            cme=self._mass,
            final_state=fields.ZBoson.pdg,
            product=product,
            final_state_bar=self._nu.pdg,
            single=False,
        )

    def dndx_l_w(self, x: RealArray, product: int):
        return HDMSpectra.dndx(
            x=x,
            cme=self._mass,
            final_state=fields.WBoson.pdg,
            product=product,
            final_state_bar=self._lep.pdg,
            single=False,
        )

    def dndx_l_w_bar(self, x: RealArray, product: int):
        return HDMSpectra.dndx(
            x=x,
            cme=self._mass,
            final_state=-fields.WBoson.pdg,
            product=product,
            final_state_bar=-self._lep.pdg,
            single=False,
        )

    def _positron_lines(self, brs: Dict[str, float]) -> List[SpectrumLine]:
        lines = []
        if self.gen == Gen.Fst:
            me = fields.Electron.mass
            mw = fields.WBoson.mass
            mn = self.mass
            ep = (mn**2 + me**2 - mw**2) / (2.0 * mn)
            xp = 2 * ep / self.mass
            lines.append(SpectrumLine(xp, 0.5 * brs["e w"]))
        return lines

    def _neutrino_lines(self, brs: Dict[str, float], gen: Gen) -> List[SpectrumLine]:
        lines = []
        if self.gen == Gen:
            vv = self._nustr

            mh = fields.Higgs.mass
            mn = self.mass
            e1 = (mn**2 - mh**2) / (2.0 * mn)
            x1 = 2 * e1 / self.mass
            lines.append(SpectrumLine(x1, 0.5 * brs[f"{vv} h"]))

            mz = fields.ZBoson.mass
            e2 = (mn**2 - mz**2) / (2.0 * mn)
            x2 = 2 * e2 / self.mass
            lines.append(SpectrumLine(x2, 0.5 * brs[f"{vv} z"]))
        return lines

    def dndx_components(self, x: RealArray, product: int) -> Dict[str, RealArray]:
        valid_prod = [22, -11, 12, 14, 16]
        assert (
            product in valid_prod
        ), f"Invalid product {product}. Must be 22, -11, 12, 14, or 16."

        vv = self._nustr
        ll = self._lepstr
        brs = self.branching_fractions()

        def make_dndx(prod):
            dndx: Dict[str, RealArray] = dict()
            dndx[f"{vv} h"] = brs[f"{vv} h"] * self.dndx_v_h(x, product=prod)
            dndx[f"{vv} z"] = brs[f"{vv} z"] * self.dndx_v_z(x, product=prod)
            dndx[f"{ll} w"] = 0.5 * brs[f"{ll} w"] * self.dndx_l_w(x, product=prod)
            dndx[f"{ll}bar wbar"] = (
                0.5 * brs[f"{ll} w"] * self.dndx_l_w_bar(x, product=prod)
            )
            return dndx

        dndx = make_dndx(product)
        if product in [12, 14, 16]:
            dndx_nubar = make_dndx(-product)
            for key in dndx.keys():
                dndx[key] += dndx_nubar[key]

        return dndx

    def dndx(self, x: RealArray, product: int) -> Spectrum:
        dndx = sum(self.dndx_components(x, product).values())
        lines: List[SpectrumLine] = []

        if product == -11:
            brs = self.branching_fractions()
            lines = self._positron_lines(brs)
        elif product in [12, 14, 16]:
            brs = self.branching_fractions()
            if product == 12:
                lines = self._neutrino_lines(brs, Gen.Fst)
            elif product == 14:
                lines = self._neutrino_lines(brs, Gen.Snd)
            else:
                lines = self._neutrino_lines(brs, Gen.Trd)
            return Spectrum(x, dndx, lines=lines)

        return Spectrum(x, dndx, lines=lines)

    def dndx_bb(self, x, cme: float, product: int, eps=0.1):
        dndx = HDMSpectra.dndx_bb(x=x, cme=cme, product=product)
        return Spectrum(x, dndx).convolve(eps)
