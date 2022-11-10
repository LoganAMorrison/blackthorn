"""Implementation of partial widths for MeV model."""

# pylint: disable=invalid-name

from typing import Tuple

import numpy as np
from helax.numpy.utils import kallen_lambda
from scipy import integrate

from ..constants import CKM, CW, GF, SW, Gen
from ..fields import (ChargedKaon, ChargedMeson, ChargedPion, Electron, Eta,
                      Muon, NeutralMeson, NeutralPion, Tau)
from .base import RhNeutrinoBase

_lepton_masses = [Electron.mass, Muon.mass, Tau.mass]


def _width_n_to_m_v(self: RhNeutrinoBase, neutral_meson: NeutralMeson) -> float:
    mn = self.mass
    xh = neutral_meson.mass / mn
    if 1 < xh:
        return 0.0

    fpi = neutral_meson.decay_constant
    return (
        GF**2
        * fpi**2
        * mn**3
        / (8 * np.pi * CW**2)
        * self.theta**2
        * (1 - xh**2) ** 2
    )


def width_n_to_pi0_v(self: RhNeutrinoBase) -> float:
    """Compute the partial width for N -> pi^0 + nu."""

    return _width_n_to_m_v(self, NeutralPion)


def width_n_to_eta_v(self: RhNeutrinoBase) -> float:
    """Compute the partial width for N -> eta + nu."""

    return _width_n_to_m_v(self, Eta) / 3.0


# def _width_n_to_m_l(self: RhNeutrinoBase, mm: float, ckm2: float) -> float:
#     fpi = NeutralPion.decay_constant
#     ml = _lepton_masses[int(self.gen)]
#     mn = self.mass

#     if mn < mm + ml:
#         return 0.0

#     return (
#         fpi**2
#         * GF**2
#         * ckm2
#         * np.sqrt(
#             ml**4 + (mm**2 - mn**2) ** 2 - 2 * ml**2 * (mm**2 + mn**2)
#         )
#         * (ml**4 - mm**2 * mn**2 + mn**4 - ml**2 * (mm**2 + 2 * mn**2))
#         * np.sin(self.theta) ** 2
#     ) / (8.0 * np.pi * mn**3)


def _width_n_to_m_l(self: RhNeutrinoBase, charged_meson: ChargedMeson) -> float:
    fpi = charged_meson.decay_constant
    xl = _lepton_masses[int(self.gen)] / self.mass
    xh = charged_meson.mass / self.mass
    mn = self.mass

    if 1 < xh + xl:
        return 0.0

    dyn = ((1.0 - xl**2) ** 2 - xh**2 * (1.0 + xl**2)) * np.sqrt(
        kallen_lambda(1.0, xh**2, xl**2)
    )

    pre = (
        GF**2
        * fpi**2
        * mn**3
        * self.theta**2
        * abs(charged_meson.ckm) ** 2
        / (8 * np.pi)
    )

    return pre * dyn


def width_n_to_pi_l(self: RhNeutrinoBase) -> float:
    """Compute the partial width for N -> pi^+ + l."""
    return _width_n_to_m_l(self, ChargedPion)


def width_n_to_k_l(self: RhNeutrinoBase) -> float:
    """Compute the partial width for N -> K^+ + l."""

    return _width_n_to_m_l(self, ChargedKaon)


def width_n_to_v_a(self: RhNeutrinoBase) -> float:
    return 0.0


# def width_n_to_v_pi_pi(self: RhNeutrinoBase) -> float:
#     """Compute the partial width for N -> nu + pi^+ + pi^-."""
#     mpi = ChargedPion.mass
#     mn = self.mass
#     genn = int(self.gen)
#     theta = self.theta

#     if mn < 2 * mpi:
#         return 0.0

#     mu2 = (mpi / mn) ** 2

#     ex = 24 * mu2 * (-1 + 2 * (-1 + mu2) * mu2**2) * np.arctanh(np.sqrt(1 - 4 * mu2))
#     ex = ex + np.sqrt(1 - 4 * mu2) * (1 + 2 * mu2 * (12 + mu2 * (-5 + 6 * mu2)))

#     kl = feynman_rules.pmns_left(theta, genn, genn)
#     kr = feynman_rules.pmns_right(theta, genn, genn)
#     kk = np.abs(np.conj(kl) * kr) ** 2

#     pre = (
#         kk
#         * (GF**2 * mn**5 * (-1 + 2 * SW**2) ** 2)
#         / (768 * CW**4 * np.pi**3)
#     )

#     return pre * ex


def width_n_to_v_pi_pi(self: RhNeutrinoBase) -> float:
    """Compute the partial width for N -> nu + pi^+ + pi^-."""

    mn = self.mass
    mu = ChargedPion.mass / self.mass

    if 1 < 2 * mu:
        return 0.0

    ex = (
        np.sqrt(1 - 4 * mu**2) * (1 + 24 * mu**2 - 10 * mu**4 + 12 * mu**6)
    ) / 2.0 - 12 * mu**2 * (-1 - 2 * mu**4 + 2 * mu**6) * np.log(
        (2 * mu) / (1 + np.sqrt(1 - 4 * mu**2))
    )

    pre = (
        GF**2
        * mn**5
        * (1 - 2 * SW**2) ** 2
        * self.theta**2
        / (384 * np.pi**3)
    )

    return pre * ex


# def width_n_to_l_pi_pi0(self: RhNeutrinoBase, *, npts: int) -> Tuple[float, float]:
#     """Compute the partial width for N -> l + pi^0 + pi^+."""

#     mpi = ChargedPion.mass
#     mpi0 = NeutralPion.mass
#     ml = _lepton_masses[int(self.gen)]

#     if self.mass < ml + mpi + mpi0:
#         return (0.0, 0.0)

#     def msqrd(momenta):
#         return msqrd_n_to_l_pi_pi0(self, momenta)

#     phase_space = PhaseSpace(self.mass, [ml, mpi, mpi0], msqrd=msqrd)
#     width, err = phase_space.decay_width(n=npts)
#     return float(width), float(err)


def width_n_to_l_pi_pi0(self: RhNeutrinoBase) -> Tuple[float, float]:
    """Compute the partial width for N -> l + pi^0 + pi^+."""

    mn = self.mass
    mpi = ChargedPion.mass
    ml = _lepton_masses[int(self.gen)]

    mu = mpi / self.mass
    xl = ml / self.mass

    if self.mass < ml + 2 * mpi:
        return (0.0, 0.0)

    def integrand(x):
        return (
            np.sqrt((-1 + x) ** 2 - 2 * (1 + x) * xl**2 + xl**4)
            * (-2 * x**2 + (1 - xl**2) ** 2 + x * (1 + xl**2))
            * (1 - (4 * mu**2) / x) ** 1.5
        )

    vud2 = abs(CKM[Gen.Fst, Gen.Snd]) ** 2
    pre = 2 * GF**2 * mn**5 * vud2 * self.theta**2 / (384 * np.pi**3)
    width, err = integrate.quad(integrand, 4 * mu**2, (1 - xl) ** 2)

    return float(pre * width), float(pre * err)
