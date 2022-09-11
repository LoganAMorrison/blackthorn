import warnings
from typing import Dict, Optional, Union, Tuple

import numpy as np
import numpy.typing as npt

from .. import fields
from ..constants import Gen
from ..spectrum_utils import PPPC4DMIDSpectra, Spectrum
from .base import RhNeutrinoBase
from .common_widths import WidthVLL, WidthVVV
from .gev_widths import WidthLUD, WidthVDD, WidthVUU
from .utils import final_state_generations_n_to_three_leptons as _fs_three_lep_gens
from .utils import final_state_strings_n_to_three_leptons as _fs_three_lep_strs
from .utils import DOWN_QUARK_STR_GEN, LEPTON_STR_GEN, UP_QUARK_STR_GEN

RealArray = npt.NDArray[np.float64]

_leptons = [fields.Electron, fields.Muon, fields.Tau]


def _gen_to_up_quark(genu):
    if genu == Gen.Fst:
        return fields.UpQuark
    if genu == Gen.Snd:
        return fields.CharmQuark
    return fields.TopQuark


def _gen_to_down_quark(gend):
    if gend == Gen.Fst:
        return fields.DownQuark
    if gend == Gen.Snd:
        return fields.StrangeQuark
    return fields.BottomQuark


def _gen_to_charged_lepton(genl):
    if genl == Gen.Fst:
        return fields.Electron
    if genl == Gen.Snd:
        return fields.Muon
    return fields.Tau


class RhNeutrinoGeV(RhNeutrinoBase):
    from .gev_widths import width_n_to_h_v as __width_h_v
    from .gev_widths import width_n_to_l_u_d as __width_l_u_d
    from .gev_widths import width_n_to_v_d_d as __width_v_d_d
    from .gev_widths import width_n_to_v_u_u as __width_v_u_u
    from .gev_widths import width_n_to_w_l as __width_w_l
    from .gev_widths import width_n_to_z_v as __width_z_v

    def __init__(self, mass: float, theta: float, gen: Gen) -> None:
        super().__init__(mass, theta, gen)

    def width_v_h(self):
        return self.__width_h_v()

    def width_v_z(self):
        return self.__width_z_v()

    def width_l_w(self):
        return self.__width_w_l()

    def width_l_u_d(self, *, genu: Gen, gend: Gen, npts: int = 10_000) -> float:
        """
        Compute the partial decay with to l + u + dbar.
        """
        if self.mass > fields.WBoson.mass + _leptons[self.gen].mass:
            return 0.0
        # return self.__width_l_u_d(genu=genu, gend=gend)
        return WidthLUD(model=self, genu=genu, gend=gend).width(npts=npts)

    def width_v_d_d(self, *, gend: Gen, npts: int = 10_000) -> float:
        """
        Compute the partial decay with to v + d + dbar.
        """
        if self.mass > fields.ZBoson.mass:
            return 0.0
        # return self.__width_v_d_d(gend=gend)
        return WidthVDD(model=self, gend=gend).width(npts=npts)

    def width_v_u_u(self, *, genu: Gen, npts=10_000) -> float:
        """
        Compute the partial decay with to v + u + ubar.
        """
        if self.mass > fields.ZBoson.mass:
            return 0.0
        # return self.__width_v_u_u(genu=genu)
        return WidthVUU(model=self, genu=genu).width(npts=npts)

    def width_v_l_l(
        self, *, genv: Gen, genl1: Gen, genl2: Gen, npts: int = 10_000
    ) -> float:
        """
        Compute the partial decay with to v + l + lbar.
        """
        if self.mass > fields.ZBoson.mass:
            return 0.0
        return WidthVLL(model=self, genv=genv, genl1=genl1, genl2=genl2).width(
            npts=npts
        )

    def width_v_v_v(
        self, *, genv1: Gen, genv2: Gen, genv3: Gen, npts: int = 10_000
    ) -> float:
        if self.mass > fields.ZBoson.mass:
            return 0.0
        return WidthVVV(model=self, genv1=genv1, genv2=genv2, genv3=genv3).width(
            npts=npts
        )

    def partial_widths(self) -> Dict[str, float]:
        ll = self._lepstr
        vv = self._nustr

        qus = [("u", Gen.Fst), ("c", Gen.Snd), ("t", Gen.Trd)]
        qds = [("d", Gen.Fst), ("s", Gen.Snd), ("b", Gen.Trd)]

        pws: Dict[str, float] = {}
        pws[f"{vv} h"] = self.width_v_h()
        pws[f"{vv} z"] = self.width_v_z()
        pws[f"{ll} w"] = 2 * self.width_l_w()

        # N -> v + u + u
        for qu, g in qus:
            pws[f"{vv} {qu} {qu}bar"] = self.width_v_u_u(genu=g)
        # N -> v + d + d
        for qd, g in qds:
            pws[f"{vv} {qd} {qd}bar"] = self.width_v_d_d(gend=g)
        # N -> l + u + d
        for qu, gu in qus:
            for qd, gd in qds:
                pws[f"{ll} {qu} {qd}bar"] = 2 * self.width_l_u_d(genu=gu, gend=gd)

        # N -> v1 + l2 + lbar3
        gen_tups = _fs_three_lep_gens(self.gen)
        str_tups = _fs_three_lep_strs(self.gen)
        for gen_tup, str_tup in zip(gen_tups, str_tups):
            g1, g2, g3 = gen_tup
            s1, s2, s3 = str_tup
            key = " ".join(["v" + s1, s2, s3 + "bar"])
            pf = 1.0 if g2 == g3 else 2.0
            pws[key] = pf * self.width_v_l_l(genv=g1, genl1=g2, genl2=g3)

        # N -> v1 + v2 + v3
        gen_tups = _fs_three_lep_gens(self.gen, unique=True)
        str_tups = _fs_three_lep_strs(self.gen, unique=True)
        for gen_tup, str_tup in zip(gen_tups, str_tups):
            g1, g2, g3 = gen_tup
            s1, s2, s3 = str_tup
            key = " ".join(["v" + s1, "v" + s2, "v" + s3])
            pws[key] = self.width_v_v_v(genv1=g1, genv2=g2, genv3=g3)

        return pws

    def branching_fractions(self) -> Dict[str, float]:
        pws = self.partial_widths()
        width = sum(pws.values())
        return {key: val / width for key, val in pws.items()}

    def _pdg_to_string(self, pdg: int) -> str:
        valid_pdgs = [22, -11, 12, 14, 16]
        assert pdg in valid_pdgs, f"Invalid product {pdg}. Must be in {valid_pdgs}"

        if pdg == 22:
            return "photon"
        if pdg == -11:
            return "positron"
        if pdg == 12:
            return "ve"
        if pdg == 14:
            return "vmu"
        return "vtau"

    def _dndx_standard_model_particle(
        self,
        x: RealArray,
        finalstate: str,
        product: fields.QuantumField,
        cme: Optional[Union[float, np.ndarray]] = None,
        single: bool = False,
    ) -> RealArray:

        if cme is not None:
            cme_ = cme
        else:
            cme_ = self.mass

        if product.pdg == 11:
            pdg = -11
        else:
            pdg = product.pdg

        dndx = PPPC4DMIDSpectra.dndx(x, cme_, finalstate, pdg)

        if single:
            return dndx / 2.0
        return dndx

    def dndx_l_w(self, x: RealArray, product: fields.QuantumField) -> RealArray:
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
        dndx_l = self._dndx_standard_model_particle(
            x, self._lepstr, product, single=True
        )
        dndx_w = self._dndx_standard_model_particle(x, "W", product, single=True)
        return dndx_l + dndx_w

    def dndx_v_h(self, x: RealArray, product: fields.QuantumField) -> RealArray:
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
        return self._dndx_standard_model_particle(x, "Z", product, single=True)

    def dndx_v_z(self, x: RealArray, product: fields.QuantumField) -> RealArray:
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
        return self._dndx_standard_model_particle(x, "h", product, single=True)

    def _dndx_standard_model_particle_convolved(
        self,
        x: RealArray,
        finalstate: str,
        product: fields.QuantumField,
        invariant_masses: RealArray,
        probabilities: RealArray,
        single: bool = False,
    ):
        ms = invariant_masses
        ps = probabilities

        shape_ms = ms.shape
        shape_ps = ps.shape
        assert shape_ms == shape_ps, (
            f"Shapes of 'invariant_masses' {shape_ms} "
            + f"and 'probabilities' {shape_ps} must match."
        )

        # Leading dimension is over energies
        dndxs = self._dndx_standard_model_particle(x, finalstate, product, cme=ms)
        dndxs = ps[:, None] * dndxs
        # Integrate P(s) * dN/dx(x,s) * ds with s=invariant-mass
        result = np.trapz(dndxs, x=invariant_masses, axis=0)

        if single:
            return result / 2.0
        return result

    def dndx_v_u_u(
        self,
        x: RealArray,
        product: fields.QuantumField,
        genu: Gen,
        npts: int = 10_000,
        nbins: int = 25,
    ) -> RealArray:
        """
        Compute the decay spectrum from N -> nu + u + u.

        Parameters
        ----------
        xs: np.ndarray
            Array of x = 2E/m.
        genu: Gen
            Generation of up-quark.
        nbins: optional, int
            Number of bins to use in creating invariant-mass distribution for
            final-state up-quarks.
        nevents: optional, int
            Number of Monte-Carlo points to use in estimating invariant-mass
            distribution.

        Returns
        -------
        dndx: np.ndarray
            Array containing decay spectrum from N -> nu + u + u.
        """
        mu = _gen_to_up_quark(genu).mass
        dndx = np.zeros_like(x)
        if self.mass < 2 * mu:
            return dndx

        vuu = WidthVUU(model=self, genu=genu)
        ps, ms = vuu.invariant_mass_distributions(npts=npts, nbins=nbins)[(1, 2)]

        if genu == Gen.Fst:
            finalstate = "q"
        elif genu == Gen.Snd:
            finalstate = "c"
        else:
            finalstate = "t"

        return self._dndx_standard_model_particle_convolved(
            x,
            finalstate=finalstate,
            product=product,
            invariant_masses=ms,
            probabilities=ps,
        )

    def dndx_v_d_d(
        self,
        x: RealArray,
        product: fields.QuantumField,
        gend: Gen,
        npts: int = 10_000,
        nbins: int = 25,
    ) -> RealArray:
        """
        Compute the decay spectrum from N -> nu + d + d.

        Parameters
        ----------
        xs: np.ndarray
            Array of x = 2E/m.
        gend: Gen
            Generation of down-quark.
        nbins: optional, int
            Number of bins to use in creating invariant-mass distribution for
            final-state down-quarks.
        nevents: optional, int
            Number of Monte-Carlo points to use in estimating invariant-mass
            distribution.

        Returns
        -------
        dndx: np.ndarray
            Array containing decay spectrum from N -> nu + d + d.
        """
        md = _gen_to_down_quark(gend).mass
        dndx = np.zeros_like(x)
        if self.mass < 2 * md:
            return dndx

        vdd = WidthVDD(model=self, gend=gend)
        ps, ms = vdd.invariant_mass_distributions(npts=npts, nbins=nbins)[(1, 2)]

        if gend == Gen.Fst or gend == Gen.Snd:
            finalstate = "q"
        else:
            finalstate = "b"

        return self._dndx_standard_model_particle_convolved(
            x,
            finalstate=finalstate,
            product=product,
            invariant_masses=ms,
            probabilities=ps,
        )

    def dndx_v_l_l(
        self,
        x: RealArray,
        product: fields.QuantumField,
        genv: Gen,
        genl1: Gen,
        genl2: Gen,
        npts: int = 10_000,
        nbins: int = 25,
    ) -> RealArray:
        """
        Compute the decay spectrum from N -> nu + l + l.

        Parameters
        ----------
        xs: np.ndarray
            Array of x = 2E/m.
        genv: Gen
            Generation of neutrino.
        genl1: Gen
            Generation of first charged-lepton.
        genl2: Gen
            Generation of second charged-lepton.
        nbins: optional, int
            Number of bins to use in creating invariant-mass distribution for
            final-state down-quarks.
        nevents: optional, int
            Number of Monte-Carlo points to use in estimating invariant-mass
            distribution.

        Returns
        -------
        dndx: np.ndarray
            Array containing decay spectrum from N -> nu + l + l.
        """
        ml1 = _gen_to_charged_lepton(genl1).mass
        ml2 = _gen_to_charged_lepton(genl2).mass
        if self.mass < ml1 + ml2:
            return np.zeros_like(x)

        vll = WidthVLL(model=self, genv=genv, genl1=genl1, genl2=genl2)
        ps, ms = vll.invariant_mass_distributions(npts=npts, nbins=nbins)[(1, 2)]

        lstrs = {Gen.Fst: "e", Gen.Snd: "mu", Gen.Trd: "tau"}

        if genl1 == genl2:
            return self._dndx_standard_model_particle_convolved(
                x,
                finalstate=lstrs[genl1],
                product=product,
                invariant_masses=ms,
                probabilities=ps,
            )
        else:
            dndx1 = self._dndx_standard_model_particle_convolved(
                x,
                finalstate=lstrs[genl1],
                product=product,
                invariant_masses=ms,
                probabilities=ps,
                single=True,
            )
            dndx2 = self._dndx_standard_model_particle_convolved(
                x,
                finalstate=lstrs[genl2],
                product=product,
                invariant_masses=ms,
                probabilities=ps,
                single=True,
            )
            return dndx1 + dndx2

    def dndx_l_u_d(
        self,
        x: RealArray,
        product: fields.QuantumField,
        genu: Gen,
        gend: Gen,
        npts: int = 10_000,
        nbins: int = 25,
    ) -> RealArray:
        """
        Compute the decay spectrum from N -> l + u + d.

        Parameters
        ----------
        xs: np.ndarray
            Array of x = 2E/m.
        genu: Gen
            Generation of up-quark.
        gend: Gen
            Generation of down-quark.
        nbins: optional, int
            Number of bins to use in creating invariant-mass distribution for
            final-state down-quarks.
        nevents: optional, int
            Number of Monte-Carlo points to use in estimating invariant-mass
            distribution.

        Returns
        -------
        dndx: np.ndarray
            Array containing decay spectrum from N -> nu + l + l.
        """
        ml = _gen_to_charged_lepton(self.gen).mass
        mu = _gen_to_up_quark(genu).mass
        md = _gen_to_down_quark(gend).mass
        if self.mass < ml + mu + md:
            return np.zeros_like(x)

        lud = WidthLUD(model=self, genu=genu, gend=gend)
        dists = lud.invariant_mass_distributions(npts=npts, nbins=nbins)

        ps_lu, ms_lu = dists[(0, 1)]
        ps_ud, ms_ud = dists[(1, 2)]

        lstrs = {Gen.Fst: "e", Gen.Snd: "mu", Gen.Trd: "tau"}
        ustrs = {Gen.Fst: "q", Gen.Snd: "c", Gen.Trd: "t"}
        dstrs = {Gen.Fst: "q", Gen.Snd: "q", Gen.Trd: "b"}

        dndx_l = self._dndx_standard_model_particle_convolved(
            x,
            finalstate=lstrs[self.gen],
            product=product,
            invariant_masses=ms_lu,
            probabilities=ps_lu,
            single=True,
        )
        dndx_u = self._dndx_standard_model_particle_convolved(
            x,
            finalstate=ustrs[genu],
            product=product,
            invariant_masses=ms_ud,
            probabilities=ps_ud,
            single=True,
        )
        dndx_d = self._dndx_standard_model_particle_convolved(
            x,
            finalstate=dstrs[gend],
            product=product,
            invariant_masses=ms_ud,
            probabilities=ps_ud,
            single=True,
        )
        return dndx_l + dndx_u + dndx_d

    def dndx_components(
        self,
        x: RealArray,
        product: fields.QuantumField,
        *,
        npts: int = 10_000,
        nbins: int = 25,
        apply_br: bool = True,
    ) -> Dict[str, Spectrum]:
        """
        Compute all components of the decay spectrum of the right-handed neutrino.

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
        dndx: Dict[str, np.ndarry]
            Dictionary containing all decay spectrum of the right-handed neutrino.
        """
        genn = self.gen
        ll = self._lepstr
        vv = self._nustr

        bfs = self.branching_fractions()
        if not apply_br:
            bfs = {key: 1.0 if abs(val) > 0.0 else 0.0 for key, val in bfs.items()}

        spec = {}
        key = f"{vv} h"
        if bfs[key] > 0.0:
            spec[key] = bfs[key] * self.dndx_v_h(x, product)
        key = f"{vv} z"
        if bfs[key] > 0.0:
            spec[key] = bfs[key] * self.dndx_v_z(x, product)
        key = f"{ll} w"
        if bfs[key] > 0.0:
            spec[key] = 2 * bfs[key] * self.dndx_l_w(x, product)

        # N -> v + u + u
        for qu, g in UP_QUARK_STR_GEN:
            key = f"{vv} {qu} {qu}bar"
            if bfs[key] > 0.0:
                spec[f"{vv} {qu} {qu}bar"] = bfs[key] * self.dndx_v_u_u(
                    x, product, g, npts=npts, nbins=nbins
                )

        # N -> v + d + d
        for qd, g in DOWN_QUARK_STR_GEN:
            key = f"{vv} {qd} {qd}bar"
            if bfs[key] > 0.0:
                spec[f"{vv} {qd} {qd}bar"] = bfs[key] * self.dndx_v_d_d(
                    x, product, g, npts=npts, nbins=nbins
                )

        # N -> l + u + d
        for qu, gu in UP_QUARK_STR_GEN:
            for qd, gd in DOWN_QUARK_STR_GEN:
                key = f"{ll} {qu} {qd}bar"
                if bfs[key] > 0.0:
                    spec[f"{ll} {qu} {qd}bar"] = (
                        2
                        * bfs[key]
                        * self.dndx_l_u_d(x, product, gu, gd, npts=npts, nbins=nbins)
                    )

        # N -> l + u + d
        for lep, g in LEPTON_STR_GEN:
            if lep == ll:
                key = f"v{ll} {ll} {ll}bar"
                if bfs[key] > 0.0:
                    spec[key] = bfs[key] * self.dndx_v_l_l(
                        x, product, genn, genn, genn, npts=npts, nbins=nbins
                    )
            else:
                key = f"v{ll} {lep} {lep}bar"
                if bfs[key] > 0.0:
                    spec[key] = bfs[key] * self.dndx_v_l_l(
                        x, product, genn, g, g, npts=npts, nbins=nbins
                    )
                key = f"v{lep} {ll} {lep}bar"
                if bfs[key] > 0.0:
                    spec[key] = (
                        2
                        * bfs[key]
                        * self.dndx_v_l_l(
                            x, product, g, genn, g, npts=npts, nbins=nbins
                        )
                    )

        return {key: Spectrum(x, dndx) for key, dndx in spec}

    def dndx(self, x: RealArray, product: fields.QuantumField, **kwargs) -> Spectrum:
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
        dndx = np.zeros_like(x)
        specs = self.dndx_components(x, product, **kwargs)
        for key, val in specs.items():
            val = val.dndx
            if len(np.nonzero(val == np.nan)[0]) > 0:
                warnings.warn(f"NaN encountered in {key}. Setting to zero.")
                dndx += np.nan_to_num(val)
            else:
                dndx = dndx + val

        return Spectrum(x, dndx)

    @staticmethod
    def dndx_bb(x, cme: float, product: int, eps=0.1):
        dndx = PPPC4DMIDSpectra.dndx(x, cme, "b", product)
        return Spectrum(x, dndx).convolve(eps)
