"""
Script used to compute and save the spectra for the MeV- and
GeV-scale RH neutrino models.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import h5py
import numpy.typing as npt
from rich.progress import Progress, TaskID
from scipy.interpolate import UnivariateSpline
from HDMSpectra import HDMSpectra

from blackthorn import Gen, RhNeutrinoMeV, RhNeutrinoGeV, RhNeutrinoTeV, Spectrum
from blackthorn import fields

RealArray = npt.NDArray[np.float64]

GeV = 1.0
MeV = 1e-3
keV = 1e-6

RESULTS_DIR = Path(__file__).parent.joinpath("..").joinpath("results")

GENERATIONS = [Gen.Fst, Gen.Snd, Gen.Trd]

LOW_MASSES = [10e-6, 2e-3, 10e-3, 100e-3, 150e-3, 250e-3]
MID_MASSES = [10e-6, 100e-3, 500e-3, 50.0, 150.0, 500.0]
HIGH_MASSES = [10.0, 50.0, 300.0, 1e3, 1e5, 1e7]


class SpectrumBB:
    def __init__(self):
        self._data_pppc4dmid = (
            Path(__file__)
            .parent.joinpath("..")
            .joinpath("blackthorn")
            .joinpath("data")
            .joinpath("PPPC4DMIDPhoton.hdf5")
        )
        self._data_hdmspectra = (
            Path(__file__)
            .parent.joinpath("..")
            .joinpath("blackthorn")
            .joinpath("data")
            .joinpath("HDMSpectra.hdf5")
        )

        with h5py.File(self._data_pppc4dmid) as f:
            self._logms = f["photon"]["logms"][:]  # type: ignore
            self._logxs = f["photon"]["logxs"][:]  # type: ignore

    def __make_dndx_two_body(self, eng: float, xs: RealArray):
        logmn = np.log10(eng)
        path = "photon/b"

        idxs = np.argwhere(self._logms > logmn)

        if len(idxs) == 0:
            return np.zeros_like(xs)
        else:
            idx: int = idxs[0][0]

        with h5py.File(self._data_pppc4dmid) as f:
            if idx == 0:
                data = f[path][idx]  # type: ignore
            else:
                data = (f[path][idx] + f[path][idx - 1]) / 2.0  # type: ignore

        spline = UnivariateSpline(self._logxs, data, s=0, k=1)
        dndx = 10 ** spline(np.log10(xs))  # type: ignore

        return dndx / (xs * np.log(10.0))

    def __dndx_decay_pppc4dmid(self, mx: float, xs: RealArray, eps) -> RealArray:
        dndx = Spectrum(xs, self.__make_dndx_two_body(mx, xs))
        return dndx.convolve(eps)(xs)  # type: ignore

    def __dndx_decay_hdmspectra(
        self, mx: float, xs: RealArray, eps: float
    ) -> RealArray:
        dndx_ = HDMSpectra.spec(
            finalstate=fields.Photon.pdg,
            X=fields.BottomQuark.pdg,
            xvals=xs,
            mDM=mx,
            data=self._data_hdmspectra,
            annihilation=False,
            Xbar=-fields.BottomQuark.pdg,
            delta=False,
            interpolation="cubic",
        )
        dndx = Spectrum(xs, dndx_)  # type: ignore
        return dndx.convolve(eps)(xs)  # type: ignore

    def dndx_decay(self, mx: float, xs: RealArray, eps: float) -> RealArray:
        if mx > 1e3:
            return self.__dndx_decay_hdmspectra(mx, xs, eps)
        return self.__dndx_decay_pppc4dmid(mx, xs, eps)

    def dndx_annihilation(self, mx: float, xs: RealArray, eps) -> RealArray:
        if mx > 1e3:
            return self.__dndx_decay_hdmspectra(2.0 * mx, xs, eps)
        return self.__dndx_decay_pppc4dmid(2.0 * mx, xs, eps)


def generate_spectrum(
    xs: RealArray,
    mn: float,
    theta: float,
    gen: Gen,
    eps: float,
    mx: float,
):
    if mn < 1.0:
        RhNeutrino = RhNeutrinoMeV
    elif mn <= 1e3:
        RhNeutrino = RhNeutrinoGeV
    else:
        RhNeutrino = RhNeutrinoTeV

    model = RhNeutrino(mn, theta, gen)
    beta = np.sqrt(1.0 - (2 * mn / mx) ** 2)
    if mn == mx / 2.0:
        return model.dndx_photon(xs).convolve(eps)(xs)
    else:
        return model.dndx_photon(xs).boost(beta).convolve(eps)(xs)


def generate_spectra_data(xs, eps=0.1, theta=1e-3):

    specbb = SpectrumBB()

    spectra = {
        Gen.Fst: {
            "low": {"mx": 1.0, "mns": [], "dndxs": []},
            "mid": {"mx": 1e3, "mns": [], "dndxs": [], "bb": None},
            "high": {"mx": 1e8, "mns": [], "dndxs": [], "bb": None},
        },
        Gen.Snd: {
            "low": {"mx": 1.0, "mns": [], "dndxs": []},
            "mid": {"mx": 1e3, "mns": [], "dndxs": [], "bb": None},
            "high": {"mx": 1e8, "mns": [], "dndxs": [], "bb": None},
        },
        Gen.Trd: {
            "low": {"mx": 1.0, "mns": [], "dndxs": []},
            "mid": {"mx": 1e3, "mns": [], "dndxs": [], "bb": None},
            "high": {"mx": 1e8, "mns": [], "dndxs": [], "bb": None},
        },
    }

    for gen in GENERATIONS:
        spectra[gen]["low"]["mns"] = [10e-6, 2e-3, 10e-3, 100e-3, 150e-3, 250e-3]
        spectra[gen]["mid"]["mns"] = [10e-6, 100e-3, 500e-3, 50.0, 150.0, 500.0]
        spectra[gen]["high"]["mns"] = [10.0, 50.0, 300.0, 1e3, 1e5, 1e7]

    gstrs = ["e", "mu", "tau"]

    with Progress() as progress:
        for i, (gen, spec_obj) in enumerate(zip(GENERATIONS, spectra.values())):
            for key in spec_obj.keys():
                mx = spec_obj[key]["mx"]
                mns = spec_obj[key]["mns"]
                task = progress.add_task(f"{gstrs[i]}-{key}", total=len(mns))
                spec_obj[key]["dndxs"] = []

                if "bb" in spec_obj[key]:
                    spec_obj[key]["bb"] = specbb.dndx_decay(mx, xs, eps)

                for mn in mns:
                    spec_obj[key]["dndxs"].append(
                        generate_spectrum(xs, mn, theta, gen, eps, mx)
                    )
                    progress.update(task, advance=1)

    return spectra


if __name__ == "__main__":

    theta = 1e-3
    eps = 0.1
    xs = np.geomspace(1e-6, 1.0, 200)

    data = generate_spectra_data(xs, eps, theta)
    files = [
        RESULTS_DIR.joinpath(name)
        for name in [
            "photon_dndx_e.hdf5",
            "photon_dndx_mu.hdf5",
            "photon_dndx_tau.hdf5",
        ]
    ]

    for gen, file in zip(GENERATIONS, files):
        with h5py.File(file, "w") as f:
            dset = data[gen]
            for key, val in dset.items():
                g = f.create_group(key)
                g.attrs["mx"] = val["mx"]
                g.create_dataset("xs", data=xs)
                g.create_dataset("masses", data=np.array(val["mns"]))
                g.create_dataset("dndxs", data=np.array(val["dndxs"]))

                if "bb" in val:
                    g.create_dataset("bb", data=np.array(val["bb"]))
