from typing import Protocol
from abc import abstractmethod

import numpy as np
from scipy import interpolate


class SquaredMatrixElement(Protocol):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class Distribution:
    def __init__(self, limits, dist_fn):
        self.limits = limits
        self._dist_fn = dist_fn

    @classmethod
    def from_data(cls, energies, probabilities, k: int = 1) -> "Distribution":
        es = energies
        ps = probabilities

        limits = np.min(es), np.max(es)

        norm = np.trapz(ps, es)
        if norm > 0.0:
            ps = ps / norm

        interp = interpolate.InterpolatedUnivariateSpline(es, ps, ext=1, k=k)
        return cls(limits, interp)

    def __call__(self, e):
        return self._dist_fn(e)
