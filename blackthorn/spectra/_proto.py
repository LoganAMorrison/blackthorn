from typing import Protocol
from abc import abstractmethod

import numpy as np


class SpectrumGeneratorProtocol(Protocol):
    xmin: float = 0.0
    xmax: float = 2.0
    energy_min: float = 0.0
    energy_max: float = np.inf

    @abstractmethod
    def _dndx(self, x, energy, final_state: int, product: int):
        raise NotImplementedError()
