from typing import Optional, Union, Dict
import pathlib

import numpy as np
import numpy.typing as npt
from HDMSpectra.HDMSpectra import spec as HDMSpectraSpec
from HDMSpectra.HDMSpectra import FF as HDMSpectraFF

from ._proto import SpectrumGeneratorProtocol


RealArray = npt.NDArray[np.float64]
RealOrRealArray = Union[npt.NDArray[np.float64], float]

THIS_DIR = pathlib.Path(__file__).parent
HDMSPECTRA_DFILE = THIS_DIR.joinpath("data").joinpath("HDMSpectra.hdf5")

PDG_TO_NAME = {
    # quarks
    1: "d",
    2: "u",
    3: "s",
    4: "c",
    5: "b",
    6: "t",
    # Leptons
    11: "e",
    12: "ve",
    13: "mu",
    14: "vmu",
    15: "tau",
    16: "vtau",
    # Bosons
    21: "g",
    22: "a",
    23: "Z",
    24: "W",
    25: "h",
}

NAME_TO_PDG: Dict[str, int] = {val: key for key, val in PDG_TO_NAME.items()}

PRODUCT_PDG_TO_NAME = {
    22: "photon",
    -11: "positron",
    12: "ve",
    14: "vmu",
    16: "vtau",
}

FINAL_STATES = [
    "W",
    "Z",
    "a",
    "b",
    "c",
    "e",
    "h",
    "mu",
    "q",
    "t",
    "tau",
    "ve",
    "vmu",
    "vtau",
]


class HDMSpectra(SpectrumGeneratorProtocol):
    energy_min: float = 0.5 * 1e3
    energy_max: float = 0.5 * 1.22091e19
    xmin: float = 1e-6
    xmax: float = 1.0

    def __init__(self):
        self._datafile = HDMSPECTRA_DFILE.as_posix()

    def _dndx(
        self,
        x: RealOrRealArray,
        energy: float,
        final_state: int,
        product: int,
        delta: bool = False,
        final_state_bar: Optional[int] = None,
        interpolation: str = "cubic",
    ) -> RealOrRealArray:
        if isinstance(x, float):
            x_ = np.array(x)
        else:
            x_ = x

        if final_state_bar is None:
            final_state_bar = -final_state

        mask = np.logical_and(x_ <= 1.0, x_ >= 1e-6)
        if delta:
            dndx = np.zeros((x_.shape[0] + 1,), dtype=x_.dtype)
            dndx_mask = np.append(mask, True)
        else:
            dndx = np.zeros_like(x_)
            dndx_mask = mask

        if np.any(mask):
            dndx[dndx_mask] = HDMSpectraSpec(
                X=final_state,
                finalstate=product,
                xvals=x_[mask],
                mDM=2 * energy,
                data=self._datafile,
                delta=delta,
                interpolation=interpolation,
                Xbar=final_state_bar,
            )

        if isinstance(x, float) and not delta:
            dndx = dndx[0]

        return dndx

    def fragmentation_function(
        self,
        x: RealOrRealArray,
        energy: float,
        final_state: int,
        product: int,
        delta: bool = False,
    ) -> RealOrRealArray:
        if isinstance(x, float):
            x_ = np.array(x)
        else:
            x_ = x

        mask = np.logical_and(x_ <= 1.0, x_ >= 1e-6)
        if delta:
            dndx = np.zeros((x_.shape[0] + 1,), dtype=x_.dtype)
            dndx_mask = np.append(mask, True)
        else:
            dndx = np.zeros_like(x_)
            dndx_mask = mask

        if np.any(mask):
            dndx[dndx_mask] = HDMSpectraFF(
                id_i=final_state,
                id_f=product,
                xvals=x_[mask],
                Qval=energy,
                data=self._datafile,
                delta=delta,
            )

        if isinstance(x, float) and not delta:
            dndx = dndx[0]

        return dndx

    def dndx_bb(
        self,
        x: RealArray,
        energy: float,
        product: int,
        delta: bool = False,
        interpolation="cubic",
    ):
        return self._dndx(
            x=x,
            energy=energy,
            final_state=NAME_TO_PDG["b"],
            product=product,
            delta=delta,
            interpolation=interpolation,
        )
