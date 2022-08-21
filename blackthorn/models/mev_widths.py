from typing import Tuple

import numpy as np
from hazma.rambo import PhaseSpace

from ..constants import CKM_UD, CKM_US, CW, GF, SW
from ..fields import ChargedKaon, ChargedPion, Electron, Eta, Muon, NeutralPion, Tau
from . import feynman_rules
from .base import RhNeutrinoBase
from .msqrd import msqrd_n_to_l_pi_pi0

_lepton_masses = [Electron.mass, Muon.mass, Tau.mass]


def _width_n_to_m_v(self: RhNeutrinoBase, mm: float) -> float:
    if self.mass < mm:
        return 0.0

    fpi = NeutralPion.decay_constant
    return (
        fpi**2
        * GF**2
        * (self.mass**2 - mm**2)
        * (-(mm**2) + self.mass**2)
        * np.sin(2 * self.theta) ** 2
    ) / (32.0 * CW**4 * np.pi * self.mass)


def width_n_to_pi0_v(self: RhNeutrinoBase) -> float:
    return _width_n_to_m_v(self, NeutralPion.mass)


def width_n_to_eta_v(self: RhNeutrinoBase) -> float:
    return _width_n_to_m_v(self, Eta.mass) / 3.0


def _width_n_to_m_l(self: RhNeutrinoBase, mm: float, ckm2: float) -> float:
    fpi = NeutralPion.decay_constant
    ml = _lepton_masses[int(self.gen)]
    mn = self.mass

    if mn < mm + ml:
        return 0.0

    return (
        fpi**2
        * GF**2
        * ckm2
        * np.sqrt(
            ml**4 + (mm**2 - mn**2) ** 2 - 2 * ml**2 * (mm**2 + mn**2)
        )
        * (ml**4 - mm**2 * mn**2 + mn**4 - ml**2 * (mm**2 + 2 * mn**2))
        * np.sin(self.theta) ** 2
    ) / (8.0 * np.pi * mn**3)


def width_n_to_pi_l(self: RhNeutrinoBase) -> float:
    mpi = ChargedPion.mass
    return _width_n_to_m_l(self, mpi, abs(CKM_UD) ** 2)


def width_n_to_k_l(self: RhNeutrinoBase) -> float:
    mk = ChargedKaon.mass
    return _width_n_to_m_l(self, mk, abs(CKM_US) ** 2)


def width_n_to_v_a(self: RhNeutrinoBase) -> float:
    return 0.0


def width_n_to_v_pi_pi(self: RhNeutrinoBase) -> float:
    mpi = ChargedPion.mass
    mn = self.mass
    genn = int(self.gen)
    theta = self.theta

    if mn < 2 * mpi:
        return 0.0

    mu2 = (mpi / mn) ** 2

    ex = 24 * mu2 * (-1 + 2 * (-1 + mu2) * mu2**2) * np.arctanh(np.sqrt(1 - 4 * mu2))
    ex = ex + np.sqrt(1 - 4 * mu2) * (1 + 2 * mu2 * (12 + mu2 * (-5 + 6 * mu2)))

    kl = feynman_rules.pmns_left(theta, genn, genn)
    kr = feynman_rules.pmns_right(theta, genn, genn)
    kk = np.abs(np.conj(kl) * kr) ** 2

    pre = (
        kk
        * (GF**2 * mn**5 * (-1 + 2 * SW**2) ** 2)
        / (768 * CW**4 * np.pi**3)
    )

    return pre * ex


def width_n_to_l_pi_pi0(self: RhNeutrinoBase, *, npts: int) -> Tuple[float, float]:
    mpi = ChargedPion.mass
    mpi0 = NeutralPion.mass
    ml = _lepton_masses[int(self.gen)]

    if self.mass < ml + mpi + mpi0:
        return (0.0, 0.0)

    def msqrd(momenta):
        return msqrd_n_to_l_pi_pi0(self, momenta)

    phase_space = PhaseSpace(self.mass, [ml, mpi, mpi0], msqrd=msqrd)
    return phase_space.decay_width(n=npts)
