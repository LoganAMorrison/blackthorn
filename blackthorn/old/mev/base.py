"""
blah
"""

from ..constants import Gen


class RhNeutrinoBase:
    def __init__(self, mass: float, theta: float, gen: Gen) -> None:
        self._mass = mass
        self._theta = theta
        self._gen = gen

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def gen(self) -> Gen:
        return self._gen

    @mass.setter
    def mass(self, mass: float) -> None:
        self._mass = mass

    @theta.setter
    def theta(self, theta: float) -> None:
        self._theta = theta

    @gen.setter
    def gen(self, gen: Gen) -> None:
        self._gen = gen
