import enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings

import h5py
import numpy as np
import numpy.typing as npt
from scipy import interpolate
from scipy import integrate

from . import fields
from .rh_neutrino import Gen
from .rh_neutrino import RhNeutrinoGeVCpp as _RhNeutrinoGeV
from .spectrum_utils import Spectrum, PPPC4DMIDSpectra
from .utils import (
    DOWN_QUARK_STR_GEN,
    LEPTON_STR_GEN,
    UP_QUARK_STR_GEN,
    gen_to_charged_lepton,
    gen_to_down_quark,
    gen_to_up_quark,
)

RealArray = npt.NDArray[np.float64]

THIS_DIR = Path(__file__).parent
PPPC4DMID_DFILE = THIS_DIR.joinpath("data").joinpath("PPPC4DMID2.hdf5")

PRODUCT_PDG_TO_NAME = {
    22: "photon",
    -11: "positron",
    12: "ve",
    14: "vmu",
    16: "vtau",
}
FINAL_STATES = [
    "W",
    "Z",
    "a",
    "b",
    "c",
    "e",
    "h",
    "mu",
    "q",
    "t",
    "tau",
    "ve",
    "vmu",
    "vtau",
]


class RhNeutrinoGeV:
    def __init__(self, mass: float, theta: float, gen: Gen) -> None:
        if mass < 5.0 or mass > 100_000:
            raise ValueError(
                f"Invalid mass: {mass}. Mass must be 5 GeV < mass < 100 TeV."
            )

        if gen == Gen.Fst:
            self._lepstr = "e"
            self._nustr = "ve"
        elif gen == Gen.Snd:
            self._lepstr = "mu"
            self._nustr = "vmu"
        else:
            self._lepstr = "tau"
            self._nustr = "vtau"

        self._model = _RhNeutrinoGeV(mass, theta, gen)
        self._datafile = THIS_DIR.joinpath("data").joinpath("PPPC4DMID2.hdf5")

    @property
    def mass(self) -> float:
        """
        Mass of the right-handed neutrino in GeV.
        """
        return self._model.mass

    @mass.setter
    def mass(self, mass: float) -> None:
        self._model = _RhNeutrinoGeV(mass, self.theta, self.gen)

    @property
    def theta(self) -> float:
        """
        Mixing angle between right-handed and left-handed neutrinos.
        """
        return self._model.theta

    @theta.setter
    def theta(self, theta: float) -> None:
        self._model = _RhNeutrinoGeV(self.mass, theta, self.gen)

    @property
    def gen(self) -> Gen:
        """
        Generation of the right-handed neutrino.
        """
        return self._model.gen

    @gen.setter
    def gen(self, gen: Gen) -> None:
        self._model = _RhNeutrinoGeV(self.mass, self.theta, gen)

    def width_l_w(self) -> float:
        """
        Compute the partial width for N -> l + w.

        Returns
        -------
        width: float
            Partial width for N -> l + w.
        """
        if self.mass < fields.WBoson.mass:
            return 0.0
        return self._model.width_l_w()

    def width_v_h(self) -> float:
        """
        Compute the partial width for N -> nu + h.

        Returns
        -------
        width: float
            Partial width for N -> nu + h.
        """
        if self.mass < fields.Higgs.mass:
            return 0.0
        return self._model.width_v_h()

    def width_v_z(self) -> float:
        """
        Compute the partial width for N -> nu + Z.

        Returns
        -------
        width: float
            Partial width for N -> nu + Z.
        """
        if self.mass < fields.ZBoson.mass:
            return 0.0
        return self._model.width_v_z()

    def width_v_u_u(self, genu: Gen, **kwargs) -> Tuple[float, float]:
        """
        Compute the partial width for N -> nu + u + u.

        Parameters
        ----------
        genu: Gen
            Generation of the up-quarks.
        nevents: optional, int
            Number of Monte-Carlo points to use in estimating width.
        batchsize: optional, int
            Number of points per parallel batch used to compute width.

        Returns
        -------
        width: float
            Partial width for N -> nu + u + u.
        error: float
            Error estimate.
        """
        mu = gen_to_up_quark(genu).mass
        if self.mass < 2 * mu:
            return (0.0, 0.0)
        nevents = kwargs.get("nevents", 10_000)
        batchsize = kwargs.get("batchsize", 100)
        return self._model.width_v_u_u(genu, nevents, batchsize)

    def width_v_d_d(self, gend: Gen, **kwargs) -> Tuple[float, float]:
        """
        Compute the partial width for N -> nu + d + d.

        Parameters
        ----------
        gend: Gen
            Generation of the down-quarks.
        nevents: optional, int
            Number of Monte-Carlo points to use in estimating width.
        batchsize: optional, int
            Number of points per parallel batch used to compute width.

        Returns
        -------
        width: float
            Partial width for N -> nu + d + d.
        error: float
            Error estimate.
        """
        md = gen_to_down_quark(gend).mass
        if self.mass < 2 * md:
            return (0.0, 0.0)
        nevents = kwargs.get("nevents", 10_000)
        batchsize = kwargs.get("batchsize", 100)
        return self._model.width_v_d_d(gend, nevents, batchsize)

    def width_l_u_d(self, genu: Gen, gend: Gen, **kwargs) -> Tuple[float, float]:
        """
        Compute the partial width for N -> l + u + d.

        Parameters
        ----------
        genu: Gen
            Generation of the up-quark.
        gend: Gen
            Generation of the down-quark.
        nevents: optional, int
            Number of Monte-Carlo points to use in estimating width.
        batchsize: optional, int
            Number of points per parallel batch used to compute width.

        Returns
        -------
        width: float
            Partial width for N -> l + u + d.
        error: float
            Error estimate.
        """
        ml = gen_to_charged_lepton(self.gen).mass
        mu = gen_to_up_quark(genu).mass
        md = gen_to_down_quark(gend).mass
        if self.mass < ml + mu + md:
            return (0.0, 0.0)
        nevents = kwargs.get("nevents", 10_000)
        batchsize = kwargs.get("batchsize", 100)
        return self._model.width_l_u_d(genu, gend, nevents, batchsize)

    def width_v_l_l(
        self, genv: Gen, genl1: Gen, genl2: Gen, **kwargs
    ) -> Tuple[float, float]:
        """
        Compute the partial width for N -> nu + l + l.

        Parameters
        ----------
        genv: Gen
            Generation of the final-state neutrino.
        genu: Gen
            Generation of the first lepton.
        gend: Gen
            Generation of the second lepton.
        nevents: optional, int
            Number of Monte-Carlo points to use in estimating width.
        batchsize: optional, int
            Number of points per parallel batch used to compute width.

        Returns
        -------
        width: float
            Partial width for N -> nu + l + l.
        error: float
            Error estimate.
        """
        ml1 = gen_to_charged_lepton(genl1).mass
        ml2 = gen_to_charged_lepton(genl2).mass
        if self.mass < ml1 + ml2:
            return (0.0, 0.0)
        nevents = kwargs.get("nevents", 10_000)
        batchsize = kwargs.get("batchsize", 100)
        return self._model.width_v_l_l(genv, genl1, genl2, nevents, batchsize)

    def width_v_v_v(
        self, genv1: Gen, genv2: Gen, genv3: Gen, **kwargs
    ) -> Tuple[float, float]:
        """
        Compute the partial width for N -> nu + nu + nu.

        Parameters
        ----------
        genv1: Gen
            Generation of the first final-state neutrino.
        genv2: Gen
            Generation of the second final-state neutrino.
        genv3: Gen
            Generation of the third final-state neutrino.
        nevents: optional, int
            Number of Monte-Carlo points to use in estimating width.
        batchsize: optional, int
            Number of points per parallel batch used to compute width.

        Returns
        -------
        width: float
            Partial width for N -> nu + nu + nu.
        error: float
            Error estimate.
        """
        nevents = kwargs.get("nevents", 10_000)
        batchsize = kwargs.get("batchsize", 100)
        return self._model.width_v_v_v(genv1, genv2, genv3, nevents, batchsize)

    def __bfs_vvv_vll(self) -> Dict[str, float]:
        brs: Dict[str, float] = {}

        genn = self.gen
        ll = self._lepstr

        for lep, g in LEPTON_STR_GEN:
            if lep == ll:
                brs[f"v{ll} {ll} {ll}bar"] = self.width_v_l_l(genn, genn, genn)[0]
            else:
                brs[f"v{ll} {lep} {lep}bar"] = self.width_v_l_l(genn, g, g)[0]
                brs[f"v{lep} {ll} {lep}bar"] = 2 * self.width_v_l_l(g, genn, g)[0]
        for lep, g in LEPTON_STR_GEN:
            if lep == ll:
                brs[f"v{ll} v{ll} v{ll}"] = self.width_v_v_v(genn, genn, genn)[0]
            else:
                brs[f"v{ll} v{lep} v{lep}"] = self.width_v_v_v(genn, g, g)[0]

        return brs

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
            pws[f"{vv} {qu} {qu}bar"] = self.width_v_u_u(g)[0]
        # N -> v + d + d
        for qd, g in qds:
            pws[f"{vv} {qd} {qd}bar"] = self.width_v_d_d(g)[0]
        # N -> l + u + d
        for qu, gu in qus:
            for qd, gd in qds:
                pws[f"{ll} {qu} {qd}bar"] = 2 * self.width_l_u_d(gu, gd)[0]

        pws = {**pws, **self.__bfs_vvv_vll()}

        return pws

    def branching_fractions(self) -> Dict[str, float]:
        """
        Compute the branching fractions for the given model.
        """
        pws = self.partial_widths()
        tot = sum(pws.values())

        bfs = {key: 0.0 for key in pws.keys()}

        if tot > 0:
            for key, val in pws.items():
                bfs[key] = val / tot

        return bfs

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
        product: int,
        cme: Optional[Union[float, np.ndarray]] = None,
        single: bool = False,
    ) -> RealArray:

        if cme is not None:
            cme_ = cme
        else:
            cme_ = self.mass

        dndx = PPPC4DMIDSpectra.dndx(x, cme_, finalstate, product)

        if single:
            return dndx / 2.0
        return dndx

    def dndx_l_w(self, x: RealArray, product: int) -> RealArray:
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

    def dndx_v_h(self, x: RealArray, product: int) -> RealArray:
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

    def dndx_v_z(self, x: RealArray, product: int) -> RealArray:
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
        product: int,
        invariant_masses: List[float],
        probabilities: List[float],
        single: bool = False,
    ):
        ms = np.array(invariant_masses)
        ps = np.array(probabilities)

        shape_ms = ms.shape
        shape_ps = ps.shape
        assert shape_ms == shape_ps, (
            f"Shapes of 'invariant_masses' {shape_ms} "
            + f"and 'probabilities' {shape_ps} must match."
        )

        # Leading dimension is over energies
        dndxs = self._dndx_standard_model_particle(x, finalstate, product, cme=ms)
        dndxs = np.expand_dims(ps, 1) * dndxs
        # Integrate P(s) * dN/dx(x,s) * ds with s=invariant-mass
        result = integrate.simpson(dndxs, x=invariant_masses, axis=0)

        if single:
            return result / 2.0
        return result

    def dndx_v_u_u(self, x: RealArray, product: int, genu: Gen, **kwargs) -> RealArray:
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
        mu = gen_to_up_quark(genu).mass
        dndx = np.zeros_like(x)
        if self.mass < 2 * mu:
            return dndx

        nbins = kwargs.get("nbins", 20)
        nevents = kwargs.get("nevents", 10_000)
        ms, ps = self._model.inv_mass_distribution_v_u_u(genu, nbins, nevents)

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

    def dndx_v_d_d(self, x: RealArray, product: int, gend: Gen, **kwargs) -> RealArray:
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
        md = gen_to_down_quark(gend).mass
        dndx = np.zeros_like(x)
        if self.mass < 2 * md:
            return dndx

        nbins = kwargs.get("nbins", 20)
        nevents = kwargs.get("nevents", 10_000)
        ms, ps = self._model.inv_mass_distribution_v_d_d(gend, nbins, nevents)

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
        self, x: RealArray, product: int, genv: Gen, genl1: Gen, genl2: Gen, **kwargs
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
        ml1 = gen_to_charged_lepton(genl1).mass
        ml2 = gen_to_charged_lepton(genl2).mass
        if self.mass < ml1 + ml2:
            return np.zeros_like(x)

        nbins = kwargs.get("nbins", 20)
        nevents = kwargs.get("nevents", 10_000)
        ms, ps = self._model.inv_mass_distribution_v_l_l(
            genv, genl1, genl2, nbins, nevents
        )
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
        self, x: RealArray, product: int, genu: Gen, gend: Gen, **kwargs
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
        ml = gen_to_charged_lepton(self.gen).mass
        mu = gen_to_up_quark(genu).mass
        md = gen_to_down_quark(gend).mass
        if self.mass < ml + mu + md:
            return np.zeros_like(x)

        nbins = kwargs.get("nbins", 20)
        nevents = kwargs.get("nevents", 10_000)
        ms_lu, ps_lu, ms_ud, ps_ud = self._model.inv_mass_distributions_l_u_d(
            genu, gend, nbins, nevents
        )

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
        self, x: RealArray, product: int, **kwargs
    ) -> Dict[str, RealArray]:
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
                    x, product, g, **kwargs
                )

        # N -> v + d + d
        for qd, g in DOWN_QUARK_STR_GEN:
            key = f"{vv} {qd} {qd}bar"
            if bfs[key] > 0.0:
                spec[f"{vv} {qd} {qd}bar"] = bfs[key] * self.dndx_v_d_d(
                    x, product, g, **kwargs
                )

        # N -> l + u + d
        for qu, gu in UP_QUARK_STR_GEN:
            for qd, gd in DOWN_QUARK_STR_GEN:
                key = f"{ll} {qu} {qd}bar"
                if bfs[key] > 0.0:
                    spec[f"{ll} {qu} {qd}bar"] = (
                        2 * bfs[key] * self.dndx_l_u_d(x, product, gu, gd, **kwargs)
                    )

        # N -> l + u + d
        for lep, g in LEPTON_STR_GEN:
            if lep == ll:
                key = f"v{ll} {ll} {ll}bar"
                if bfs[key] > 0.0:
                    spec[key] = bfs[key] * self.dndx_v_l_l(
                        x, product, genn, genn, genn, **kwargs
                    )
            else:
                key = f"v{ll} {lep} {lep}bar"
                if bfs[key] > 0.0:
                    spec[key] = bfs[key] * self.dndx_v_l_l(
                        x, product, genn, g, g, **kwargs
                    )
                key = f"v{lep} {ll} {lep}bar"
                if bfs[key] > 0.0:
                    spec[key] = (
                        2 * bfs[key] * self.dndx_v_l_l(x, product, g, genn, g, **kwargs)
                    )

        return spec

    def dndx(self, x: RealArray, product: int, **kwargs) -> Spectrum:
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
