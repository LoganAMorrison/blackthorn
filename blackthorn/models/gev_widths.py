from typing import Tuple

import numpy as np
import numpy.typing as npt
from hazma.phase_space import Rambo
from hazma.utils import kallen_lambda

from .. import fields
from ..constants import ALPHA_EM, CW, SW, Gen
from .base import PartialWidth, RhNeutrinoBase
from .msqrd import msqrd_n_to_l_u_d, msqrd_n_to_v_d_d, msqrd_n_to_v_u_u

RealArray = npt.NDArray[np.float64]

_lepton_masses = [fields.Electron.mass, fields.Muon.mass, fields.Tau.mass]
_up_quark_masses = [fields.UpQuark.mass, fields.CharmQuark.mass, fields.TopQuark.mass]
_down_quark_masses = [
    fields.DownQuark.mass,
    fields.StrangeQuark.mass,
    fields.BottomQuark.mass,
]


def width_n_to_h_v(self: RhNeutrinoBase) -> float:
    mh = fields.Higgs.mass
    vev = fields.Higgs.vev

    if self.mass < mh:
        return 0.0

    return (
        (self.mass**2 - mh**2)
        * np.cos(2 * self.theta) ** 2
        * (-(mh**2) + self.mass**2)
        * np.tan(self.theta) ** 2
    ) / (16.0 * np.pi * vev**2 * self.mass)


def width_n_to_z_v(self: RhNeutrinoBase) -> float:
    mvr = self.mass
    mz = fields.ZBoson.mass

    if self.mass < mz:
        return 0.0

    return (
        (mvr**4 + mvr**2 * mz**2 - 2 * mz**4)
        * ALPHA_EM
        * (mvr**2 - mz**2)
        * np.sin(2 * self.theta) ** 2
    ) / (64.0 * CW**2 * mvr**3 * mz**2 * SW**2)


def width_n_to_w_l(self: RhNeutrinoBase) -> float:
    mvr = self.mass
    mw = fields.WBoson.mass
    ml = _lepton_masses[int(self.gen)]

    if self.mass < mw + ml:
        return 0.0

    return (
        ALPHA_EM
        * np.sqrt(kallen_lambda(mvr**2, ml**2, mw**2))
        * (
            mvr**4
            + mvr**2 * mw**2
            - 2 * mw**4
            + (-2 * mvr**2 + mw**2) * ml**2
            + ml**4
        )
        * np.sin(self.theta) ** 2
    ) / (16.0 * mvr**3 * mw**2 * SW**2)


def width_n_to_v_u_u(
    self: RhNeutrinoBase, *, genu: Gen, npts: int = 10_000
) -> Tuple[float, float]:

    mu = _up_quark_masses[int(genu)]

    if self.mass < 2 * mu:
        return (0.0, 0.0)

    def msqrd(momenta):
        return msqrd_n_to_v_u_u(self, momenta=momenta, genu=int(genu))

    phase_space = Rambo(self.mass, [0.0, mu, mu], msqrd=msqrd)
    return phase_space.decay_width(n=npts)


def width_n_to_v_d_d(
    self: RhNeutrinoBase, *, gend: Gen, npts: int = 10_000
) -> Tuple[float, float]:

    md = _down_quark_masses[int(gend)]

    if self.mass < 2 * md:
        return (0.0, 0.0)

    def msqrd(momenta):
        return msqrd_n_to_v_d_d(self, momenta, gend=int(gend))

    phase_space = Rambo(self.mass, [0.0, md, md], msqrd=msqrd)
    return phase_space.decay_width(n=npts)


def width_n_to_l_u_d(
    self: RhNeutrinoBase, *, genu: Gen, gend: Gen, npts: int = 10_000
) -> Tuple[float, float]:
    ml = _lepton_masses[int(self.gen)]
    mu = _up_quark_masses[int(genu)]
    md = _down_quark_masses[int(gend)]

    if self.mass < ml + mu + md:
        return (0.0, 0.0)

    def msqrd(momenta):
        return msqrd_n_to_l_u_d(self, momenta, genu=int(genu), gend=int(gend))

    phase_space = Rambo(self.mass, [ml, mu, md], msqrd=msqrd)
    return phase_space.decay_width(n=npts)


class WidthVUU(PartialWidth):
    def __init__(self, *, model: RhNeutrinoBase, genu: Gen) -> None:
        self._mass = model.mass
        self._fsp_masses = [0.0, _up_quark_masses[genu], _up_quark_masses[genu]]

        def msqrd(momenta):
            return msqrd_n_to_v_u_u(model, momenta=momenta, genu=int(genu))

        phase_space = Rambo(model.mass, self._fsp_masses, msqrd=msqrd)
        super().__init__(phase_space)


class WidthVDD(PartialWidth):
    def __init__(self, *, model: RhNeutrinoBase, gend: Gen) -> None:
        self._mass = model.mass
        md = _down_quark_masses[gend]
        self._fsp_masses = [0.0, md, md]

        def msqrd(momenta):
            return msqrd_n_to_v_d_d(model, momenta=momenta, gend=int(gend))

        phase_space = Rambo(model.mass, self._fsp_masses, msqrd=msqrd)
        super().__init__(phase_space)


class WidthLUD(PartialWidth):
    def __init__(self, *, model: RhNeutrinoBase, genu: Gen, gend: Gen) -> None:
        self._mass = model.mass
        self._fsp_masses = [
            _lepton_masses[model.gen],
            _up_quark_masses[genu],
            _down_quark_masses[gend],
        ]

        def msqrd(momenta):
            return msqrd_n_to_l_u_d(model, momenta, genu=int(genu), gend=int(gend))

        phase_space = Rambo(model.mass, self._fsp_masses, msqrd=msqrd)
        super().__init__(phase_space)
