import math
from typing import Tuple

import numpy as np
import numpy.typing as npt
from hazma.phase_space import Rambo

from .. import fields
from ..constants import ALPHA_EM, GF, Gen
from .base import PartialWidth, RhNeutrinoBase
from .msqrd import msqrd_n_to_v_l_l, msqrd_n_to_v_v_v

RealArray = npt.NDArray[np.float64]
_lepton_masses = [fields.Electron.mass, fields.Muon.mass, fields.Tau.mass]


class WidthVLL(PartialWidth):
    def __init__(
        self, *, model: RhNeutrinoBase, genv: Gen, genl1: Gen, genl2: Gen
    ) -> None:
        ml1 = _lepton_masses[int(genl1)]
        ml2 = _lepton_masses[int(genl2)]
        self._mass = model.mass
        self._fsp_masses = [0.0, ml1, ml2]

        def msqrd(momenta):
            return msqrd_n_to_v_l_l(
                model, momenta, genv=int(genv), genl1=int(genl1), genl2=int(genl2)
            )

        phase_space = Rambo(model.mass, self._fsp_masses, msqrd=msqrd)
        super().__init__(phase_space)


class WidthVVV(PartialWidth):
    def __init__(
        self, *, model: RhNeutrinoBase, genv1: Gen, genv2: Gen, genv3: Gen
    ) -> None:
        self._mass = model.mass
        self._fsp_masses = [0.0, 0.0, 0.0]

        # len==1 => 3 same => S = 3!
        # len==2 => 2 same => S = 2!
        # len==3 => 0 same => S = 1!
        symmetry_factor = math.factorial(4 - len({genv1, genv2, genv3}))

        def msqrd(momenta):
            return msqrd_n_to_v_v_v(
                model, momenta, genv1=int(genv1), genv2=int(genv2), genv3=int(genv3)
            )

        phase_space = Rambo(cme=model.mass, masses=self._fsp_masses, msqrd=msqrd)
        super().__init__(phase_space, 1.0 / symmetry_factor)


def width_n_to_v_l_l(
    self: RhNeutrinoBase, *, genv: Gen, genl1: Gen, genl2: Gen, npts: int = 10_000
) -> Tuple[float, float]:
    ml1 = _lepton_masses[int(genl1)]
    ml2 = _lepton_masses[int(genl2)]

    if self.mass < ml1 + ml2:
        return (0.0, 0.0)

    def msqrd(momenta):
        return msqrd_n_to_v_l_l(
            self, momenta, genv=int(genv), genl1=int(genl1), genl2=int(genl2)
        )

    phase_space = Rambo(self.mass, [0.0, ml1, ml2], msqrd=msqrd)
    return phase_space.decay_width(n=npts)  # type: ignore


def width_n_to_v_v_v(
    self: RhNeutrinoBase, *, genv1: Gen, genv2: Gen, genv3: Gen, npts: int = 10_000
) -> Tuple[float, float]:
    sym_fact = 1.0 / math.factorial(4 - len({genv1, genv2, genv3}))

    def msqrd(momenta):
        return msqrd_n_to_v_v_v(
            self, momenta, genv1=int(genv1), genv2=int(genv2), genv3=int(genv3)
        )

    phase_space = Rambo(self.mass, [0.0, 0.0, 0.0], msqrd=msqrd)
    width, error = phase_space.decay_width(n=npts)
    return sym_fact * width, sym_fact * error  # type: ignore


def width_n_to_v_a(self: RhNeutrinoBase) -> float:
    return (
        9
        * ALPHA_EM
        * GF**2
        / (1024 * np.pi**4)
        * self.mass**5
        * np.sin(2 * self.theta) ** 2
    )
