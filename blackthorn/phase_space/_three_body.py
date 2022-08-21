from typing import Sequence, Tuple, Optional, Dict
from abc import abstractmethod

import numpy as np
from scipy import integrate

from ..fields import QuantumField

from ._proto import SquaredMatrixElement, Distribution


class ThreeBodySquaredMatrixElement(SquaredMatrixElement):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, s, t, **kwargs) -> float:
        raise NotImplementedError()


def kallen_lambda(a, b, c):
    return a**2 + b**2 + c**2 - 2.0 * a * b - 2.0 * a * c - 2.0 * b * c


def energy_distributions_three_body_decay(
    cme,
    final_states: Sequence[QuantumField],
    nbins: int,
    msqrd: Optional[SquaredMatrixElement],
    k: int = 1,
) -> Tuple[Distribution, Distribution, Distribution]:
    assert (
        len(final_states) == 3
    ), f"`final_states` must have three fields. Found {len(final_states)}"

    if msqrd is not None:

        def msqrd_(s, t):  # type: ignore
            return msqrd(s, t)

    else:

        def msqrd_(s, t):
            return 1.0

    def ybounds(mu1, mu2, mu3, x):
        t1 = -((-2 + x) * (-1 + x - mu1**2 - mu2**2 + mu3**2))
        t2 = np.sqrt(kallen_lambda(1, mu1**2, 1 - x + mu1**2)) * np.sqrt(
            kallen_lambda(1 - x + mu1**2, mu2**2, mu3**2)
        )

        ymin = (t1 - t2) / (2.0 * (-1.0 + x - mu1**2))
        ymax = (t1 + t2) / (2.0 * (-1.0 + x - mu1**2))

        return ymin, ymax

    def xbounds(mu1, mu2, mu3):
        xmin = 2 * mu1
        xmax = mu1**2 + (1.0 - mu2 - mu3) * (1.0 + mu2 + mu3)
        return xmin, xmax

    def dist(mu1, mu2, mu3):
        def f(x, y):
            s = cme**2 * (1.0 - x + mu1**2)
            t = cme**2 * (1.0 - y + mu2**2)
            return msqrd_(s, t)

        def p(x):
            ybs = ybounds(mu1, mu2, mu3, x)
            return integrate.quad(lambda y: f(x, y), ybs[0], ybs[1])

        # x = 2E/cme
        xbs = xbounds(mu1, mu2, mu3)
        xs = np.linspace(xbs[0], xbs[1], nbins)
        dndx = np.array([p(x) for x in xs])

        energies = 0.5 * cme * xs
        probabilities = 2.0 / cme * np.array([p[0] for p in dndx])

        return Distribution.from_data(energies, probabilities, k=k)

    mu1 = final_states[0].mass / cme
    mu2 = final_states[1].mass / cme
    mu3 = final_states[2].mass / cme

    dist1 = dist(mu1, mu2, mu3)
    dist2 = dist(mu2, mu3, mu1)
    dist3 = dist(mu3, mu1, mu2)

    return dist1, dist2, dist3


def invariant_mass_distributions_three_body_decay(
    cme,
    final_states: Sequence[QuantumField],
    nbins: int,
    msqrd: Optional[SquaredMatrixElement],
    k: int = 1,
) -> Dict[Tuple[int, int], Distribution]:
    assert (
        len(final_states) == 3
    ), f"`final_states` must have three fields. Found {len(final_states)}"

    if msqrd is not None:

        def msqrd_(s, t):  # type: ignore
            return msqrd(s, t)

    else:

        def msqrd_(s, t):
            return 1.0

    def tbounds(m1, m2, m3, s):
        t1 = (
            (m2**2 + m3**2 - s) * s
            + m1**2 * (m2**2 - m3**2 + s)
            + cme**2 * (-(m2**2) + m3**2 + s)
        )
        t2 = np.sqrt(kallen_lambda(cme**2, m1**2, s)) * np.sqrt(
            kallen_lambda(m2**2, m3**2, s)
        )

        tmin = (t1 - t2) / (2.0 * s)
        tmax = (t1 + t2) / (2.0 * s)

        return tmin, tmax

    def sbounds(m1, m2, m3):
        smin = (m2 + m3) ** 2
        smax = (cme - m1) ** 2
        return smin, smax

    def dist(m1, m2, m3):
        def p(s):
            tbs = tbounds(m1, m2, m3, s)
            return integrate.quad(lambda t: msqrd_(s, t), tbs[0], tbs[1])

        sbs = sbounds(mu1, mu2, mu3)
        ss = np.linspace(sbs[0], sbs[1], nbins)
        dnds = np.array([p(s) for s in ss])

        ms = np.sqrt(ss)
        probabilities = np.array([2 * m * p[0] for (m, p) in zip(ms, dnds)])

        return Distribution.from_data(ms, probabilities, k=k)

    mu1 = final_states[0].mass / cme
    mu2 = final_states[1].mass / cme
    mu3 = final_states[2].mass / cme

    dist1 = dist(mu1, mu2, mu3)
    dist2 = dist(mu2, mu3, mu1)
    dist3 = dist(mu3, mu1, mu2)

    return {(1, 2): dist1, (0, 2): dist2, (0, 1): dist3}
