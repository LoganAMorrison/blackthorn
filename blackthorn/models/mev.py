from typing import Dict

import numpy as np

from ..constants import Gen
from .base import RhNeutrinoBase, RealArray
from .utils import final_state_generations_n_to_three_leptons as _fs_three_lep_gens
from .utils import final_state_strings_n_to_three_leptons as _fs_three_lep_strs
from ..spectrum_utils import Spectrum
from .. import fields
from . import mev_spectra


class RhNeutrinoMeV(RhNeutrinoBase):

    from .common_widths import width_n_to_v_a as __width_v_a
    from .common_widths import width_n_to_v_l_l as __width_v_l_l
    from .common_widths import width_n_to_v_v_v as __width_v_v_v
    from .mev_widths import width_n_to_eta_v as __width_eta_v
    from .mev_widths import width_n_to_k_l as __width_k_l
    from .mev_widths import width_n_to_l_pi_pi0 as __width_l_pi_pi0
    from .mev_widths import width_n_to_pi0_v as __width_pi0_v
    from .mev_widths import width_n_to_pi_l as __width_pi_l
    from .mev_widths import width_n_to_v_pi_pi as __width_v_pi_pi

    def __init__(self, mass: float, theta: float, gen: Gen) -> None:
        super().__init__(mass, theta, gen)

    def width_v_pi0(self) -> float:
        return self.__width_pi0_v()

    def width_v_eta(self) -> float:
        return self.__width_eta_v()

    def width_l_pi(self) -> float:
        return self.__width_pi_l()

    def width_l_k(self) -> float:
        return self.__width_k_l()

    def width_v_a(self) -> float:
        return self.__width_v_a()

    def width_v_l_l(self, *, genv: Gen, genl1: Gen, genl2: Gen):
        return self.__width_v_l_l(genv=genv, genl1=genl1, genl2=genl2)

    def width_v_v_v(self, *, genv1: Gen, genv2: Gen, genv3: Gen):
        return self.__width_v_v_v(genv1=genv1, genv2=genv2, genv3=genv3)

    def width_v_pi_pi(self):
        return self.__width_v_pi_pi()

    def width_l_pi_pi0(self, *, npts: int = 10_000):
        return self.__width_l_pi_pi0(npts=npts)

    def partial_widths(self) -> Dict[str, float]:
        r"""Compute the partial widths of the RH neutrino.

        Returns
        -------
        pws: Dict[str, float]
            Dictionary containing all the partial widths of the RH neutrino.
        """
        ll = self._lepstr
        vv = self._nustr

        pws: Dict[str, float] = {}
        pws[f"{ll} k"] = 2 * self.width_l_k()
        pws[f"{ll} pi"] = 2 * self.width_l_pi()
        pws[f"{vv} pi0"] = self.width_v_pi0()
        pws[f"{vv} eta"] = self.width_v_eta()
        pws[f"{vv} a"] = self.width_v_a()
        pws[f"{vv} pi pibar"] = self.width_v_pi_pi()
        pws[f"{ll} pi pi0"] = 2 * self.width_l_pi_pi0()[0]

        # N -> v1 + l2 + lbar3
        gen_tups = _fs_three_lep_gens(self.gen)
        str_tups = _fs_three_lep_strs(self.gen)
        for gen_tup, str_tup in zip(gen_tups, str_tups):
            g1, g2, g3 = gen_tup
            s1, s2, s3 = str_tup
            key = " ".join(["v" + s1, s2, s3 + "bar"])
            pws[key] = self.width_v_l_l(genv=g1, genl1=g2, genl2=g3)[0]

        # N -> v1 + v2 + v3
        gen_tups = _fs_three_lep_gens(self.gen, unique=True)
        str_tups = _fs_three_lep_strs(self.gen, unique=True)
        for gen_tup, str_tup in zip(gen_tups, str_tups):
            g1, g2, g3 = gen_tup
            s1, s2, s3 = str_tup
            key = " ".join(["v" + s1, "v" + s2, "v" + s3])
            pws[key] = self.width_v_v_v(genv1=g1, genv2=g2, genv3=g3)[0]

        return pws

    def branching_fractions(self):
        pws = self.partial_widths()
        width = sum(pws.values())
        return {key: val / width for key, val in pws.items()}

    def dndx_v_pi0(self, x: RealArray, product: fields.QuantumField) -> Spectrum:
        dndx = mev_spectra.DecaySpectrumVPi0(self)
        return dndx.dndx(x, product)

    def dndx_v_eta(self, x: RealArray, product: fields.QuantumField) -> Spectrum:
        dndx = mev_spectra.DecaySpectrumVEta(self)
        return dndx.dndx(x, product)

    def dndx_l_pi(self, x: RealArray, product: fields.QuantumField) -> Spectrum:
        dndx = mev_spectra.DecaySpectrumLPi(self)
        return dndx.dndx(x, product, fsr=True)

    def dndx_l_k(self, x: RealArray, product: fields.QuantumField) -> Spectrum:
        dndx = mev_spectra.DecaySpectrumLK(self)
        return dndx.dndx(x, product, fsr=True)

    def dndx_v_a(self, x: RealArray, product: fields.QuantumField) -> Spectrum:
        dndx = mev_spectra.DecaySpectrumVA(self)
        return dndx.dndx(x, product)

    def dndx_v_l_l(
        self,
        x: RealArray,
        product: fields.QuantumField,
        *,
        genv: Gen,
        genl1: Gen,
        genl2: Gen,
        nbins: int = 25,
        npts: int = 10_000,
    ) -> Spectrum:
        dndx = mev_spectra.DecaySpectrumVLL(
            self, genv=genv, genl1=genl1, genl2=genl2, nbins=nbins, npts=npts
        )
        return dndx.dndx(x, product, fsr=True)

    def dndx_v_v_v(
        self,
        x: RealArray,
        product: fields.QuantumField,
        *,
        genv1: Gen,
        genv2: Gen,
        genv3: Gen,
        nbins: int = 25,
        npts: int = 10_000,
    ) -> Spectrum:
        dndx = mev_spectra.DecaySpectrumVVV(
            self, genv1=genv1, genv2=genv2, genv3=genv3, nbins=nbins, npts=npts
        )
        return dndx.dndx(x, product)

    def dndx_v_pi_pi(
        self,
        x: RealArray,
        product: fields.QuantumField,
        *,
        nbins: int = 25,
        npts: int = 10_000,
    ) -> Spectrum:
        dndx = mev_spectra.DecaySpectrumVPiPi(self, npts=npts, nbins=nbins)
        return dndx.dndx(x, product, fsr=True)

    def dndx_l_pi_pi0(
        self,
        x: RealArray,
        product: fields.QuantumField,
        nbins: int = 25,
        npts: int = 10_000,
    ) -> Spectrum:
        dndx = mev_spectra.DecaySpectrumLPiPi0(self, nbins=nbins, npts=npts)
        return dndx.dndx(x, product, fsr=True)

    def lines(self, product: fields.QuantumField):
        vv = self._nustr
        lines = dict()
        brs = self.branching_fractions()

        nu = product == fields.ElectronNeutrino and self.gen == Gen.Fst
        nu = (product == fields.MuonNeutrino and self.gen == Gen.Snd) and nu
        nu = (product == fields.TauNeutrino and self.gen == Gen.Trd) and nu

        if product == fields.Photon:
            key = f"{vv} a"
            lines[f"{vv} a"] = {"energy": self.mass / 2.0, "bf": brs[key]}

        elif product == fields.Electron and self.gen == Gen.Fst:
            me = fields.Electron.mass
            mpi = fields.ChargedPion.mass
            mk = fields.ChargedKaon.mass
            if self.mass > mpi:
                key = "pi e"
                lines[key] = {
                    "energy": (self.mass**2 + me**2 - mpi**2) / (2 * self.mass),
                    "bf": brs[key],
                }
            if self.mass > mk:
                key = "k e"
                lines[key] = {
                    "energy": (self.mass**2 + me**2 - mk**2) / (2 * self.mass),
                    "bf": brs[key],
                }

        elif nu:
            mpi0 = fields.NeutralPion.mass
            meta = fields.Eta.mass
            if self.mass > mpi0:
                key = f"{vv} pi0"
                lines[key] = {
                    "energy": (self.mass**2 - mpi0**2) / (2 * self.mass),
                    "bf": brs[key],
                }
            if self.mass > meta:
                key = f"{vv} eta"
                lines[f"{vv} eta"] = {
                    "energy": (self.mass**2 - meta**2) / (2 * self.mass),
                    "bf": brs[key],
                }
            key = f"{vv} a"
            lines[f"{vv} a"] = {
                "energy": self.mass / 2.0,
                "bf": brs[key],
            }

        return lines

    def dndx_components(
        self,
        x: RealArray,
        product: fields.QuantumField,
        *,
        npts: int = 10_000,
        nbins: int = 25,
        apply_br: bool = True,
    ) -> Dict[str, Spectrum]:
        r"""Compute the partial widths of the RH neutrino.

        Returns
        -------
        pws: Dict[str, float]
            Dictionary containing all the partial widths of the RH neutrino.
        """
        ll = self._lepstr
        vv = self._nustr

        dndxs: Dict[str, Spectrum] = {}
        dndxs[f"{ll} k"] = 2 * self.dndx_l_k(x, product)
        dndxs[f"{ll} pi"] = 2 * self.dndx_l_pi(x, product)
        dndxs[f"{vv} pi0"] = self.dndx_v_pi0(x, product)
        dndxs[f"{vv} eta"] = self.dndx_v_eta(x, product)
        dndxs[f"{vv} a"] = self.dndx_v_a(x, product)
        dndxs[f"{vv} pi pibar"] = self.dndx_v_pi_pi(x, product, npts=npts, nbins=nbins)
        dndxs[f"{ll} pi pi0"] = 2 * self.dndx_l_pi_pi0(
            x, product, npts=npts, nbins=nbins
        )

        # N -> v1 + l2 + lbar3
        gen_tups = _fs_three_lep_gens(self.gen)
        str_tups = _fs_three_lep_strs(self.gen)
        for gen_tup, str_tup in zip(gen_tups, str_tups):
            g1, g2, g3 = gen_tup
            s1, s2, s3 = str_tup
            key = " ".join(["v" + s1, s2, s3 + "bar"])
            pf = 2 if g2 != g3 else 1.0
            dndxs[key] = pf * self.dndx_v_l_l(
                x, product, genv=g1, genl1=g2, genl2=g3, npts=npts, nbins=nbins
            )

        # N -> v1 + v2 + v3
        gen_tups = _fs_three_lep_gens(self.gen, unique=True)
        str_tups = _fs_three_lep_strs(self.gen, unique=True)
        for gen_tup, str_tup in zip(gen_tups, str_tups):
            g1, g2, g3 = gen_tup
            s1, s2, s3 = str_tup
            key = " ".join(["v" + s1, "v" + s2, "v" + s3])
            dndxs[key] = self.dndx_v_v_v(
                x, product, genv1=g1, genv2=g2, genv3=g3, npts=npts, nbins=nbins
            )

        if apply_br:
            bfs = self.branching_fractions()
            for key, val in bfs.items():
                dndxs[key] = dndxs[key] * val

        if product == fields.Electron:
            for key in dndxs.keys():
                dndxs[key] = 0.5 * dndxs[key]

        return dndxs

    def dndx(self, x: RealArray, product: fields.QuantumField) -> Spectrum:
        r"""Compute the partial widths of the RH neutrino.

        Returns
        -------
        pws: Dict[str, float]
            Dictionary containing all the partial widths of the RH neutrino.
        """
        dndxs = self.dndx_components(x, product, apply_br=True)
        dndx = Spectrum(x, np.zeros_like(x))

        for _, val in dndxs.items():
            dndx._dndx += val._dndx
            for line in val._lines:
                dndx._lines.append(line)

        return dndx
