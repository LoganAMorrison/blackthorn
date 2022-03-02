from typing import Dict, Union, Tuple, List
from pathlib import Path

# from HDMSpectra import HDMSpectra
import h5py
from HDMSpectra.HDMSpectra import spec as hdm_spec
import numpy as np
import numpy.typing as npt
from scipy.interpolate import UnivariateSpline

from .rh_neutrino import Gen
from .rh_neutrino import RhNeutrinoGeVCpp as _RhNeutrinoGeV
from .rh_neutrino import RhNeutrinoMeVCpp as _RhNeutrinoMeV
from .utils import LEPTON_STR_GEN, UP_QUARK_STR_GEN, DOWN_QUARK_STR_GEN
from .utils import gen_to_up_quark, gen_to_down_quark, gen_to_charged_lepton
from . import fields

from .spectrum_utils import Spectrum, SpectrumLine

RealArray = npt.NDArray[np.float64]

THIS_DIR = Path(__file__).parent


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
        self._datafile = THIS_DIR.joinpath("data").joinpath("PPPC4DMIDPhoton.hdf5")

        with h5py.File(self._datafile) as f:
            self._logms = f["photon"]["logms"][:]  # type: ignore
            self._logxs = f["photon"]["logxs"][:]  # type: ignore

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

    def __make_dndx_two_body(self, xs: RealArray, p1: str, p2: str):
        logmn = np.log10(self.mass)
        path1 = "photon/" + p1
        path2 = "photon/" + p2

        idxs = np.argwhere(self._logms > logmn)

        if len(idxs) == 0:
            return np.zeros_like(xs)
        else:
            idx: int = idxs[0][0]

        with h5py.File(self._datafile) as f:
            if idx == 0:
                data1 = f[path1][idx]  # type: ignore
                data2 = f[path2][idx]  # type: ignore
            else:
                data1 = (f[path1][idx] + f[path1][idx - 1]) / 2.0  # type: ignore
                data2 = (f[path2][idx] + f[path2][idx - 1]) / 2.0  # type: ignore

        spline1 = UnivariateSpline(self._logxs, data1, s=0, k=1)
        spline2 = UnivariateSpline(self._logxs, data2, s=0, k=1)
        dndx1 = 10 ** spline1(np.log10(xs))  # type: ignore
        dndx2 = 10 ** spline2(np.log10(xs))  # type: ignore

        return (dndx1 + dndx2) / (2 * xs * np.log(10.0))

    def __make_dndx_two_body_single(self, xs: RealArray, p1: str):
        logmn = np.log10(self.mass)
        path1 = "photon/" + p1

        idxs = np.argwhere(self._logms > logmn)

        if len(idxs) == 0:
            return np.zeros_like(xs)
        else:
            idx: int = idxs[0][0]

        with h5py.File(self._datafile) as f:
            if idx == 0:
                data = f[path1][idx]  # type: ignore
            else:
                data = (f[path1][idx] + f[path1][idx - 1]) / 2.0  # type: ignore

        spline = UnivariateSpline(self._logxs, data, s=0, k=1)
        return 10 ** spline(np.log10(xs)) / (2 * xs * np.log(10.0))  # type: ignore

    def dndx_l_w(self, xs: RealArray) -> RealArray:
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
        return self.__make_dndx_two_body(xs, self._lepstr, "w")

    def dndx_v_h(self, xs: RealArray) -> RealArray:
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
        return self.__make_dndx_two_body_single(xs, "h")

    def dndx_v_z(self, xs: RealArray) -> RealArray:
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
        return self.__make_dndx_two_body_single(xs, "z")

    def __make_dndx_three_body(
        self, xs: RealArray, path: str, ms: List[float], ps: List[float]
    ) -> RealArray:
        idxs = []
        for i, m in enumerate(ms):
            idxs_ = np.argwhere(self._logms > np.log10(m))
            if len(idxs_) != 0:
                idxs.append((i, idxs_[0][0]))

        datas = []
        with h5py.File(self._datafile) as f:
            for i, idx in idxs:
                if idx == 0:
                    d = f[path][idx]  # type: ignore
                else:
                    d = (f[path][idx] + f[path][idx - 1]) / 2.0  # type: ignore
                datas.append((ps[i], d))

        dndx = np.zeros_like(xs)
        logxs = np.log10(xs)
        for p, data in datas:
            spline = UnivariateSpline(self._logxs, data, s=0, k=1)
            dndx = dndx + p * 10 ** spline(logxs)  # type: ignore

        return dndx / (xs * np.log(10.0))

    def dndx_v_u_u(self, xs: RealArray, genu: Gen, **kwargs) -> RealArray:
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
        if self.mass < 2 * mu:
            return np.zeros_like(xs)

        nbins = kwargs.get("nbins", 20)
        nevents = kwargs.get("nevents", 10_000)
        ms, ps = self._model.inv_mass_distribution_v_u_u(genu, nbins, nevents)

        if genu == Gen.Fst:
            path = "photon/q"
        elif genu == Gen.Snd:
            path = "photon/c"
        else:
            path = "photon/t"

        return self.__make_dndx_three_body(xs, path, ms, ps)

    def dndx_v_d_d(self, xs: RealArray, gend: Gen, **kwargs) -> RealArray:
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
        if self.mass < 2 * md:
            return np.zeros_like(xs)

        nbins = kwargs.get("nbins", 20)
        nevents = kwargs.get("nevents", 10_000)
        ms, ps = self._model.inv_mass_distribution_v_d_d(gend, nbins, nevents)

        if gend == Gen.Fst or gend == Gen.Snd:
            path = "photon/q"
        else:
            path = "photon/b"

        return self.__make_dndx_three_body(xs, path, ms, ps)

    def dndx_v_l_l(
        self, xs: RealArray, genv: Gen, genl1: Gen, genl2: Gen, **kwargs
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
            return np.zeros_like(xs)

        nbins = kwargs.get("nbins", 20)
        nevents = kwargs.get("nevents", 10_000)
        ms, ps = self._model.inv_mass_distribution_v_l_l(
            genv, genl1, genl2, nbins, nevents
        )

        if genl1 == Gen.Fst:
            path1 = "photon/e"
        elif genl1 == Gen.Snd:
            path1 = "photon/mu"
        else:
            path1 = "photon/tau"

        if genl2 == Gen.Fst:
            path2 = "photon/e"
        elif genl2 == Gen.Snd:
            path2 = "photon/mu"
        else:
            path2 = "photon/tau"

        return (
            self.__make_dndx_three_body(xs, path1, ms, ps) / 2.0
            + self.__make_dndx_three_body(xs, path2, ms, ps) / 2.0
        )

    def dndx_l_u_d(self, xs: RealArray, genu: Gen, gend: Gen, **kwargs) -> RealArray:
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
            return np.zeros_like(xs)

        nbins = kwargs.get("nbins", 20)
        nevents = kwargs.get("nevents", 10_000)
        ms_lu, ps_lu, ms_ud, ps_ud = self._model.inv_mass_distributions_l_u_d(
            genu, gend, nbins, nevents
        )

        pathl = f"photon/{self._lepstr}"

        if genu == Gen.Fst:
            pathu = "photon/q"
        elif genu == Gen.Snd:
            pathu = "photon/c"
        else:
            pathu = "photon/t"

        if gend == Gen.Fst or gend == Gen.Snd:
            pathd = "photon/q"
        else:
            pathd = "photon/b"

        return (
            self.__make_dndx_three_body(xs, pathl, ms_lu, ps_lu) / 2.0
            + self.__make_dndx_three_body(xs, pathu, ms_ud, ps_ud) / 2.0
            + self.__make_dndx_three_body(xs, pathd, ms_ud, ps_ud) / 2.0
        )

    def dndx_photon_components(self, xs: RealArray, **kwargs) -> Dict[str, RealArray]:
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
            spec[key] = bfs[key] * self.dndx_v_h(xs)
        key = f"{vv} z"
        if bfs[key] > 0.0:
            spec[key] = bfs[key] * self.dndx_v_z(xs)
        key = f"{ll} w"
        if bfs[key] > 0.0:
            spec[key] = 2 * bfs[key] * self.dndx_l_w(xs)

        # N -> v + u + u
        for qu, g in UP_QUARK_STR_GEN:
            key = f"{vv} {qu} {qu}bar"
            if bfs[key] > 0.0:
                spec[f"{vv} {qu} {qu}bar"] = bfs[key] * self.dndx_v_u_u(xs, g, **kwargs)

        # N -> v + d + d
        for qd, g in DOWN_QUARK_STR_GEN:
            key = f"{vv} {qd} {qd}bar"
            if bfs[key] > 0.0:
                spec[f"{vv} {qd} {qd}bar"] = bfs[key] * self.dndx_v_d_d(xs, g, **kwargs)

        # N -> l + u + d
        for qu, gu in UP_QUARK_STR_GEN:
            for qd, gd in DOWN_QUARK_STR_GEN:
                key = f"{ll} {qu} {qd}bar"
                if bfs[key] > 0.0:
                    spec[f"{ll} {qu} {qd}bar"] = (
                        2 * bfs[key] * self.dndx_l_u_d(xs, gu, gd, **kwargs)
                    )

        # N -> l + u + d
        for lep, g in LEPTON_STR_GEN:
            if lep == ll:
                key = f"v{ll} {ll} {ll}bar"
                if bfs[key] > 0.0:
                    spec[key] = bfs[key] * self.dndx_v_l_l(
                        xs, genn, genn, genn, **kwargs
                    )
            else:
                key = f"v{ll} {lep} {lep}bar"
                if bfs[key] > 0.0:
                    spec[key] = bfs[key] * self.dndx_v_l_l(xs, genn, g, g, **kwargs)
                key = f"v{lep} {ll} {lep}bar"
                if bfs[key] > 0.0:
                    spec[key] = 2 * bfs[key] * self.dndx_v_l_l(xs, g, genn, g, **kwargs)

        return spec

    def dndx_photon(self, xs: RealArray, **kwargs) -> Spectrum:
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
        dndx = np.zeros_like(xs)
        specs = self.dndx_photon_components(xs, **kwargs)
        for _, val in specs.items():
            dndx = dndx + val

        return Spectrum(xs, dndx)


class RhNeutrinoMeV(_RhNeutrinoMeV):
    def __init__(self, mass: float, theta: float, gen: Gen) -> None:
        super().__init__(mass, theta, gen)
        if gen == Gen.Fst:
            self._lepstr = "e"
            self._nustr = "ve"
        elif gen == Gen.Snd:
            self._lepstr = "mu"
            self._nustr = "vmu"
        else:
            self._lepstr = "tau"
            self._nustr = "vtau"

    def __bfs_vvv_vll(self) -> Dict:
        brs = {}

        genn = self.gen
        ll = self._lepstr

        # N -> v + l1 + l2
        for lep, g in LEPTON_STR_GEN[:-1]:
            if lep == ll:
                brs[f"v{ll} {ll} {ll}bar"] = self.width_v_l_l(genn, genn, genn)[0]
            else:
                brs[f"v{ll} {lep} {lep}bar"] = self.width_v_l_l(genn, g, g)[0]
                brs[f"v{lep} {ll} {lep}bar"] = 2 * self.width_v_l_l(g, genn, g)[0]
        # N -> v1 + v2 + v3
        for lep, g in LEPTON_STR_GEN:
            if lep == ll:
                brs[f"v{ll} v{ll} v{ll}"] = self.width_v_v_v(genn, genn, genn)[0]
            else:
                brs[f"v{ll} v{lep} v{lep}"] = self.width_v_v_v(genn, g, g)[0]

        return brs

    def partial_widths(self) -> Dict[str, float]:
        ll = self._lepstr
        vv = self._nustr

        pws: Dict[str, float] = {}
        pws[f"{ll} k"] = 2 * self.width_l_k()
        pws[f"{ll} pi"] = 2 * self.width_l_pi()
        pws[f"{vv} pi0"] = self.width_v_pi0()
        pws[f"{vv} eta"] = self.width_v_eta()
        pws[f"{vv} a"] = self.width_v_a()
        pws[f"{vv} pi pibar"] = self.width_v_pi_pi()[0]
        pws[f"{ll} pi pi0"] = 2 * self.width_l_pi_pi0()[0]

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

    def dndx_photon_components(self, xs: RealArray) -> Dict[str, RealArray]:
        genn = self.gen
        ll = self._lepstr
        vv = self._nustr

        bfs = self.branching_fractions()

        spec = {}
        spec[f"{ll} pi"] = 2 * self.dndx_l_pi(xs)
        spec[f"{ll} k"] = 2 * self.dndx_l_k(xs)
        spec[f"{vv} pi0"] = self.dndx_v_pi0(xs)
        spec[f"{vv} pi pibar"] = self.dndx_v_pi_pi(xs)
        spec[f"{ll} pi pi0"] = 2 * self.dndx_l_pi_pi0(xs)

        # N -> l + u + d
        for lep, g in LEPTON_STR_GEN[:-1]:
            if lep == ll:
                spec[f"v{ll} {ll} {ll}bar"] = self.dndx_v_l_l(xs, genn, genn, genn)
            else:
                spec[f"v{ll} {lep} {lep}bar"] = self.dndx_v_l_l(xs, genn, g, g)
                spec[f"v{lep} {ll} {lep}bar"] = 2 * self.dndx_v_l_l(xs, g, genn, g)

        for key in spec.keys():
            spec[key] = spec[key] * bfs[key]

        return spec

    def dndx_photon(self, xs: RealArray) -> Spectrum:
        tot = sum(self.dndx_photon_components(xs).values())
        brs = self.branching_fractions()
        vv = self._nustr
        lines = [SpectrumLine(1.0, brs[f"{vv} a"])]
        return Spectrum(xs, tot, lines=lines)


class RhNeutrinoTeV:
    def __init__(self, mass: float, theta: float, gen: Gen) -> None:
        self._mass = mass
        self._theta = theta
        self._gen = gen
        self._model = _RhNeutrinoGeV(mass, theta, gen)
        self._data = THIS_DIR.joinpath("data").joinpath("HDMSpectra.hdf5")
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

    def dndx_v_h(self, xs, finalstate=fields.Photon.pdg):
        return hdm_spec(
            finalstate=finalstate,
            X=fields.Higgs.pdg,
            xvals=xs,
            mDM=self._mass,
            data=self._data,
            annihilation=False,
            Xbar=self._nu.pdg,
            delta=False,
            interpolation="cubic",
        )

    def dndx_v_z(self, xs, finalstate=fields.Photon.pdg):
        return hdm_spec(
            finalstate=finalstate,
            X=fields.ZBoson.pdg,
            xvals=xs,
            mDM=self._mass,
            data=self._data,
            annihilation=False,
            Xbar=self._nu.pdg,
            delta=False,
            interpolation="cubic",
        )

    def dndx_l_w(self, xs, finalstate=fields.Photon.pdg):
        return hdm_spec(
            finalstate=finalstate,
            X=fields.WBoson.pdg,
            xvals=xs,
            mDM=self._mass,
            data=self._data,
            annihilation=False,
            Xbar=self._lep.pdg,
            delta=False,
            interpolation="cubic",
        )

    def dndx_l_w_bar(self, xs, finalstate=fields.Photon.pdg):
        return hdm_spec(
            finalstate=finalstate,
            X=-fields.WBoson.pdg,
            xvals=xs,
            mDM=self._mass,
            data=self._data,
            annihilation=False,
            Xbar=-self._lep.pdg,
            delta=False,
            interpolation="cubic",
        )

    def dndx_photon_components(self, xs):
        finalstate = fields.Photon.pdg
        vv = self._nustr
        ll = self._lepstr
        brs = self.branching_fractions()
        return {
            f"{vv} h": brs[f"{vv} h"] * self.dndx_v_h(xs, finalstate=finalstate),
            f"{vv} z": brs[f"{vv} z"] * self.dndx_v_z(xs, finalstate=finalstate),
            f"{ll} w": 0.5 * brs[f"{ll} w"] * self.dndx_l_w(xs, finalstate=finalstate),
            f"{ll}bar wbar": 0.5
            * brs[f"{ll} w"]
            * self.dndx_l_w_bar(xs, finalstate=finalstate),
        }

    def dndx_photon(self, xs: RealArray) -> Spectrum:
        tot = sum(self.dndx_photon_components(xs).values())
        return Spectrum(xs, tot)
