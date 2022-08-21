from typing import Union, Dict, Callable
import pathlib

from scipy import interpolate
import numpy as np
import numpy.typing as npt
import h5py

from ._proto import SpectrumGeneratorProtocol

RealArray = npt.NDArray[np.float64]
RealOrRealArray = Union[npt.NDArray[np.float64], float]

THIS_DIR = pathlib.Path(__file__).parent
PPPC4DMID_DFILE = THIS_DIR.joinpath("data").joinpath("PPPC4DMID.hdf5")

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


class PPPC4DMIDSpectra(SpectrumGeneratorProtocol):
    energy_min: float = 0.5 * 5.0
    energy_max: float = 0.5 * 1e5
    xmin: float = 1e-9
    xmax: float = 1.0

    def __init__(self):
        self._interpolators: Dict[int, Dict[str, Callable]] = dict()
        pars = {"kx": 1, "ky": 1, "s": 0.0}
        with h5py.File(PPPC4DMID_DFILE) as f:  # type: ignore
            for pdg in [22, -11, 12, 14, 16]:
                self._interpolators[pdg] = {}
                for fs in FINAL_STATES:
                    stable = PRODUCT_PDG_TO_NAME[pdg]
                    logm = f[stable][fs]["logM"][:]
                    logx = f[stable][fs]["logX"][:]
                    data = f[stable][fs]["data"][:, :].T
                    self._interpolators[pdg][fs] = interpolate.RectBivariateSpline(
                        logx, logm, data, **pars
                    )

    def _dndx(
        self,
        x: RealOrRealArray,
        cme: RealOrRealArray,
        final_state: str,
        product: int,
        single: bool = True,
    ) -> RealOrRealArray:
        assert product in self._interpolators, f"Invalid product {product}"
        interp_dict = self._interpolators[product]
        assert final_state in interp_dict, f"Invalid final state {final_state}"

        # cme = 2 * M
        logm = np.log10(cme / 2.0)
        logx = np.log10(x)
        interp = interp_dict[final_state]
        # Convert log10(dN / dlog10(x)) -> dN/dx
        ldndlx = np.squeeze(interp(logx, logm)).T
        dndx = (10**ldndlx) / (np.log(10.0) * x)

        if single:
            return dndx / 2.0
        return dndx

    def dndx_bb(self, x: RealArray, cme: float, product: int):
        return self._dndx(x, cme, "b", product, single=False)
