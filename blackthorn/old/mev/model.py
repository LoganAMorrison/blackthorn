"""
This module contains the implementation of the MeV RHNeutrino class.

Final states:
    l_i + pi
    nu_i + pi0
    nu_j + l_n + l_m
    nu_i + nu_n + nu_m
    nu_i + a
"""

from typing import Callable, Dict, Tuple

# from HDMSpectra import HDMSpectra
import numpy as np
import numpy.typing as npt

from .rh_neutrino import Gen
from .rh_neutrino import RhNeutrinoMeVCpp as _RhNeutrinoMeV
from .spectrum_utils import Spectrum, SpectrumLine

RealArray = npt.NDArray[np.float64]

# Ni -> vj + lm + ln
#   i1 -> j + m + n
#   i1, i2, i3 in [1,2,3], i1!=i2, i1!=i3, i2!=i3
#   i1 == j, m == n => j==i1, m == i1, n == i1
#                   => j==i1, m == i2, n == i2
#                   => j==i1, m == i3, n == i3
#   i1 == m, j == n => m==i1, j==i1, n==i1
#                   => m==i1, j == i2, n == i2
#                   => m==i1, j == i3, n == i3
#   i1 == n, j == m => n==i1, j==i1, m==i1
#                   => n==i1, j==i2, m==i2
#                   => n==i1, j==i3, m==i3
#
#   (1, 1, 1)
#   (1, 2, 2)
#   (1, 3, 3)
#   (2, 1, 2)
#   (3, 1, 3)
#   (2, 2, 1)
#   (3, 3, 1)


def final_state_generations_n_to_three_leptons(gen_n: Gen, unique: bool = False):
    gen1 = gen_n
    gen2, gen3 = {Gen.Fst, Gen.Snd, Gen.Trd}.difference({gen_n})
    gens = [
        (gen1, gen1, gen1),
        (gen1, gen2, gen2),
        (gen1, gen3, gen3),
    ]

    if not unique:
        gens.append((gen2, gen1, gen2))
        gens.append((gen2, gen2, gen1))
        gens.append((gen3, gen1, gen3))
        gens.append((gen3, gen3, gen1))

    return gens


def final_state_strings_n_to_three_leptons(gen_n: Gen, unique: bool = False):
    strs = ["e", "mu", "tau"]

    def gen_tup_to_str_tup(tup):
        return tuple(map(lambda gen: strs[gen], tup))

    gen_tups = final_state_generations_n_to_three_leptons(gen_n, unique)
    return list(map(gen_tup_to_str_tup, gen_tups))


class RhNeutrinoMeV:
    def __init__(self, mass: float, theta: float, gen: Gen) -> None:
        self._model = _RhNeutrinoMeV(mass, theta, gen)

        if gen == Gen.Fst:
            self._lepstr = "e"
            self._nustr = "ve"
        elif gen == Gen.Snd:
            self._lepstr = "mu"
            self._nustr = "vmu"
        else:
            self._lepstr = "tau"
            self._nustr = "vtau"

        self._dndx_l_pi = {
            "photon": self._model.dndx_photon_l_pi,
            "positron": self._model.dndx_positron_l_pi,
            "neutrino": self._model.dndx_neutrino_l_pi,
        }
        self._dndx_l_k = {
            "photon": self._model.dndx_photon_l_k,
            "positron": self._model.dndx_positron_l_k,
            "neutrino": self._model.dndx_neutrino_l_k,
        }
        self._dndx_v_pi0 = {
            "photon": self._model.dndx_photon_v_pi0,
            "positron": self._model.dndx_positron_v_pi0,
            "neutrino": self._model.dndx_neutrino_v_pi0,
        }
        self._dndx_v_pi_pi = {
            "photon": self._model.dndx_photon_v_pi_pi,
            "positron": self._model.dndx_positron_v_pi_pi,
            "neutrino": self._model.dndx_neutrino_v_pi_pi,
        }
        self._dndx_v_l_l = {
            "photon": self._model.dndx_photon_v_l_l,
            "positron": self._model.dndx_positron_v_l_l,
            "neutrino": self._model.dndx_neutrino_v_l_l,
        }
        self._dndx_l_pi_pi0 = {
            "photon": self._model.dndx_photon_l_pi_pi0,
            "positron": self._model.dndx_positron_l_pi_pi0,
            "neutrino": self._model.dndx_neutrino_l_pi_pi0,
        }

    @property
    def mass(self) -> float:
        """Mass of the RH neutrino in GeV."""
        return self._model.mass

    @mass.setter
    def mass(self, mass: float) -> None:
        self._model = _RhNeutrinoMeV(mass, self._model.theta, self._model.gen)

    @property
    def theta(self) -> float:
        """Mixing angle between the RH and LH neutrino."""
        return self._model.theta

    @theta.setter
    def theta(self, theta: float) -> None:
        self._model = _RhNeutrinoMeV(self._model.mass, theta, self._model.gen)

    @property
    def gen(self) -> Gen:
        """Generation of the RH neutrino."""
        return self._model.gen

    @gen.setter
    def gen(self, gen: Gen) -> None:
        self._model = _RhNeutrinoMeV(self._model.mass, self._model.theta, gen)

    def width_l_pi(self) -> float:
        r"""Compute the partial width for N -> ℓ⁻ + π⁺.

        The final-state lepton is of the same generation as the RHN.

        Returns
        -------
        width: float
            Partial width for N -> l + pi.
        """
        return self._model.width_l_pi()

    def width_l_k(self) -> float:
        r"""Compute the partial width for N -> ℓ⁻ + k⁺.

        The final-state lepton is of the same generation as the RHN.

        Returns
        -------
        width: float
            Partial width for N -> k + pi.
        """
        return self._model.width_l_k()

    def width_v_pi0(self) -> float:
        r"""Compute the partial width for N -> ν + π⁰.

        The final-state lepton is of the same generation as the RHN.

        Returns
        -------
        width: float
            Partial width for N -> ν + π⁰.
        """
        return self._model.width_v_pi0()

    def width_v_a(self) -> float:
        r"""Compute the partial width for N -> ν + γ.

        The final-state lepton is of the same generation as the RHN.

        Returns
        -------
        width: float
            Partial width for N -> ν + γ.
        """
        return self._model.width_v_a()

    def width_v_eta(self) -> float:
        r"""Compute the partial width for N -> ν + η.

        The final-state lepton is of the same generation as the RHN.

        Returns
        -------
        width: float
            Partial width for N -> ν + η.
        """
        return self._model.width_v_eta()

    def width_v_pi_pi(self, *, nevents: int = 10_000) -> Tuple[float, float]:
        r"""Compute the partial width for N -> ν + π⁺ + π⁻.

        The final-state lepton is of the same generation as the RHN.

        Parameters
        ----------
        nevents: int, optional
            Number of points to use in Monte-Carl phase-space integration.

        Returns
        -------
        width: float
            Partial width for N -> ν + π⁺ + π⁻.
        """
        return self._model.width_v_pi_pi(nevents=nevents)

    def width_v_l_l(
        self,
        genv: Gen,
        genl1: Gen,
        genl2: Gen,
        *,
        nevents: int = 10_000,
    ) -> Tuple[float, float]:
        r"""Compute the partial width for N -> ν + ℓ⁺ + ℓ⁻.

        Parameters
        ----------
        genv: Gen
            Generation of the LH neutrino.
        genl1, genl2: Gen
            Generations of the final-state charged leptons.
        nevents: int, optional
            Number of points to use in Monte-Carl phase-space integration.

        Returns
        -------
        width: float
            Partial width for N -> ν + ℓ⁺ + ℓ⁻.
        """
        return self._model.width_v_l_l(genv, genl1, genl2, nevents=nevents)

    def width_l_pi_pi0(
        self, *, nevents: int = 10_000, batchsize: int = 1000
    ) -> Tuple[float, float]:
        r"""Compute the partial width for N -> ℓ⁻ + π⁺ + π⁰.

        Parameters
        ----------
        nevents: int, optional
            Number of points to use in Monte-Carl phase-space integration.

        Returns
        -------
        width: float
            Partial width for N -> ℓ⁻ + π⁺ + π⁰.
        """
        return self._model.width_l_pi_pi0(nevents=nevents, batchsize=batchsize)

    def width_v_v_v(
        self,
        gen1: Gen,
        gen2: Gen,
        gen3: Gen,
        *,
        nevents: int = 10_000,
        batchsize: int = 1000,
    ) -> Tuple[float, float]:
        r"""Compute the partial width for N -> ν₁ + ν₂ + ν₃.

        Parameters
        ----------
        gen1, gen2, gen3: Gen
            Generations of the final-state LH neutrinos.
        nevents: int, optional
            Number of points to use in Monte-Carl phase-space integration.

        Returns
        -------
        width: float
            Partial width for N -> ν₁ + ν₂ + ν₃.
        """
        return self._model.width_v_v_v(
            gen1, gen2, gen3, nevents=nevents, batchsize=batchsize
        )

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
        pws[f"{vv} pi pibar"] = self.width_v_pi_pi()[0]
        pws[f"{ll} pi pi0"] = 2 * self.width_l_pi_pi0()[0]

        # N -> v1 + l2 + lbar3
        gen_tups = final_state_generations_n_to_three_leptons(self.gen)
        str_tups = final_state_strings_n_to_three_leptons(self.gen)
        for gen_tup, str_tup in zip(gen_tups, str_tups):
            g1, g2, g3 = gen_tup
            s1, s2, s3 = str_tup
            key = " ".join([s1, s2, s3 + "bar"])
            pf = 1.0 if g2 == g3 else 2.0
            pws[key] = pf * self.width_v_l_l(g1, g2, g3)[0]

        # N -> v1 + v2 + v3
        gen_tups = final_state_generations_n_to_three_leptons(self.gen, unique=True)
        str_tups = final_state_strings_n_to_three_leptons(self.gen, unique=True)
        for gen_tup, str_tup in zip(gen_tups, str_tups):
            g1, g2, g3 = gen_tup
            s1, s2, s3 = str_tup
            key = " ".join(["v" + s1, "v" + s2, "v" + s3])
            pws[key] = self.width_v_v_v(g1, g2, g3)[0]

        return pws

    def branching_fractions(self) -> Dict[str, float]:
        r"""Compute the branching fractions of the RH neutrino.

        Returns
        -------
        bfs: Dict[str, float]
            Dictionary containing all the branching fractions of the RH neutrino.
        """

        pws = self.partial_widths()
        tot = sum(pws.values())

        bfs = {key: 0.0 for key in pws.keys()}

        if tot > 0:
            for key, val in pws.items():
                bfs[key] = val / tot

        return bfs

    def _dndx_dispatch(self, dndx: Dict[str, Callable], product: str, *args):
        if product in dndx:
            return dndx[product](*args)
        else:
            raise ValueError(
                f"Invalid product {product}. Must be 'photon', 'positron' or 'neutrino'."
            )

    def dndx_l_pi(self, x: RealArray, product: str, beta: float):
        r"""Compute the spectrum into a product from N -> ℓ⁻ + π⁺.

        The final-state lepton is of the same generation as the RH neutrino.

        Parameters
        ----------
        x : RealArray
            Values of x = 2E / m where the spectrum should be evaluated.
        product : str
            Product to compute spectrum for. Can be 'photon', 'positron' or 'neutrino'.
        beta : float
            Boost factor. Must be between 0 and 1.

        Returns
        -------
        dndx: RealArray
            Spectrum dN/dx from N -> ℓ⁻ + π⁺ into `product`.
        """
        return self._dndx_dispatch(self._dndx_l_pi, product, x, beta)

    def dndx_l_k(self, x: RealArray, product: str, beta: float):
        r"""Compute the spectrum into a product from N -> ℓ⁻ + K⁺.

        The final-state lepton is of the same generation as the RH neutrino.

        Parameters
        ----------
        x : RealArray
            Values of x = 2E / m where the spectrum should be evaluated.
        product : str
            Product to compute spectrum for. Can be 'photon', 'positron' or 'neutrino'.
        beta : float
            Boost factor. Must be between 0 and 1.

        Returns
        -------
        dndx: RealArray
            Spectrum dN/dx from N -> ℓ⁻ + K⁺ into `product`.
        """
        return self._dndx_dispatch(self._dndx_l_k, product, x, beta)

    def dndx_v_pi0(self, x: RealArray, product: str, beta: float):
        r"""Compute the spectrum into a product from N -> ν + π⁰.

        The final-state lepton is of the same generation as the RH neutrino.

        Parameters
        ----------
        x : RealArray
            Values of x = 2E / m where the spectrum should be evaluated.
        product : str
            Product to compute spectrum for. Can be 'photon', 'positron' or 'neutrino'.
        beta : float
            Boost factor. Must be between 0 and 1.

        Returns
        -------
        dndx: RealArray
            Spectrum dN/dx from N -> ν + π⁰ into `product`.
        """
        return self._dndx_dispatch(self._dndx_v_pi0, product, x, beta)

    def dndx_v_pi_pi(self, x: RealArray, product: str, beta: float):
        r"""Compute the spectrum into a product from N -> ν + π⁺ + π⁻.

        The final-state lepton is of the same generation as the RH neutrino.

        Parameters
        ----------
        x : RealArray
            Values of x = 2E / m where the spectrum should be evaluated.
        product : str
            Product to compute spectrum for. Can be 'photon', 'positron' or 'neutrino'.
        beta : float
            Boost factor. Must be between 0 and 1.

        Returns
        -------
        dndx: RealArray
            Spectrum dN/dx from N -> ν + π⁺ + π⁻ into `product`.
        """
        return self._dndx_dispatch(self._dndx_v_pi_pi, product, x, beta)

    def dndx_v_l_l(
        self, x: RealArray, product: str, beta: float, genv: Gen, genl1: Gen, genl2: Gen
    ):
        r"""Compute the spectrum into a product from N -> ν + ℓ⁺ + ℓ⁻.

        Parameters
        ----------
        x : RealArray
            Values of x = 2E / m where the spectrum should be evaluated.
        product : str
            Product to compute spectrum for. Can be 'photon', 'positron' or 'neutrino'.
        beta : float
            Boost factor. Must be between 0 and 1.
        genv: Gen
            Generation of the LH neutrino.
        genl1, genl2: Gen
            Generations of the final-state charged leptons.

        Returns
        -------
        dndx: RealArray
            Spectrum dN/dx from N -> ν + ℓ⁺ + ℓ⁻ into `product`.
        """
        return self._dndx_dispatch(
            self._dndx_v_l_l, product, x, beta, genv, genl1, genl2
        )

    def dndx_l_pi_pi0(self, x: RealArray, product: str, beta: float):
        r"""Compute the spectrum into a product from N -> ℓ⁻ + π⁺ + π⁰.

        The final-state lepton is of the same generation as the RH neutrino.

        Parameters
        ----------
        x : RealArray
            Values of x = 2E / m where the spectrum should be evaluated.
        product : str
            Product to compute spectrum for. Can be 'photon', 'positron' or 'neutrino'.
        beta : float
            Boost factor. Must be between 0 and 1.

        Returns
        -------
        dndx: RealArray
            Spectrum dN/dx from N -> ℓ⁻ + π⁺ + π⁰ into `product`.
        """
        return self._dndx_dispatch(self._dndx_l_pi_pi0, product, x, beta)

    def dndx_components(
        self, x: RealArray, product: str, beta: float
    ) -> Dict[str, RealArray]:
        r"""Compute all components of the spectrum into a product from the decay of the RH neutrino.

        Parameters
        ----------
        x : RealArray
            Values of x = 2E / m where the spectrum should be evaluated.
        product : str
            Product to compute spectrum for. Can be 'photon', 'positron' or 'neutrino'.
        beta : float
            Boost factor. Must be between 0 and 1.

        Returns
        -------
        dndx: dict[str, np.ndarray]
            Spectrum components dN/dx from the decay of the RH into `product`.
        """
        ll = self._lepstr
        vv = self._nustr

        bfs = self.branching_fractions()

        spec: Dict[str, np.ndarray] = dict()
        spec[f"{ll} pi"] = 2 * self.dndx_l_pi(x, product, beta)
        spec[f"{ll} k"] = 2 * self.dndx_l_k(x, product, beta)
        spec[f"{vv} pi0"] = self.dndx_v_pi0(x, product, beta)
        spec[f"{vv} pi pibar"] = self.dndx_v_pi_pi(x, product, beta)
        spec[f"{ll} pi pi0"] = 2 * self.dndx_l_pi_pi0(x, product, beta)

        # N -> v1 + l2 + lbar3
        gen_tups = final_state_generations_n_to_three_leptons(self.gen)
        str_tups = final_state_strings_n_to_three_leptons(self.gen)
        for gen_tup, str_tup in zip(gen_tups, str_tups):
            g1, g2, g3 = gen_tup
            s1, s2, s3 = str_tup
            key = " ".join([s1, s2, s3 + "bar"])
            pf = 1.0 if g2 == g3 else 2.0
            spec[key] = pf * self.dndx_v_l_l(x, product, beta, g1, g2, g3)

        for key in spec.keys():
            spec[key] = spec[key] * bfs[key]

        return spec

    def dndx(self, x: RealArray, product: str, beta: float) -> Spectrum:
        r"""Compute total spectrum into a product from the decay of the RH neutrino.

        Parameters
        ----------
        x : RealArray
            Values of x = 2E / m where the spectrum should be evaluated.
        product : str
            Product to compute spectrum for. Can be 'photon', 'positron' or 'neutrino'.
        beta : float
            Boost factor. Must be between 0 and 1.

        Returns
        -------
        dndx: np.ndarray
            Total spectrum dN/dx from the decay of the RH into `product`.
        """
        tot = sum(self.dndx_components(x, product, beta).values())
        brs = self.branching_fractions()
        vv = self._nustr
        lines = [SpectrumLine(1.0, brs[f"{vv} a"])]
        return Spectrum(x, tot, lines=lines)
