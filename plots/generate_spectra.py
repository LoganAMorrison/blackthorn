"""
Script used to compute and save the spectra for the MeV- and
GeV-scale RH neutrino models.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from rich.progress import Progress
from utils import write_data_to_h5py

from blackthorn import Gen, RhNeutrinoMeV, RhNeutrinoGeV, RhNeutrinoTeV

RealArray = npt.NDArray[np.float64]

GeV = 1.0
MeV = 1e-3
keV = 1e-6

GENERATIONS = [Gen.Fst, Gen.Snd, Gen.Trd]


LOW_MASSES = [10e-6, 2e-3, 10e-3, 100e-3, 150e-3, 250e-3]
MID_MASSES = [10e-6, 100e-3, 500e-3, 50.0, 150.0, 500.0]
HIGH_MASSES = [10.0, 50.0, 300.0, 1e3, 1e5, 1e7]


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


def generate_spectra_data():
    xs = np.geomspace(1e-6, 1.0, 200)
    eps = 0.1

    GSTRS = ["fst", "snd", "trd"]

    spectra = {
        "fst": {
            "low": {"mx": 1.0, "mns": [], "dndxs": []},
            "mid": {"mx": 1e3, "mns": [], "dndxs": []},
            "high": {"mx": 1e8, "mns": [], "dndxs": []},
        },
        "snd": {
            "low": {"mx": 1.0, "mns": [], "dndxs": []},
            "mid": {"mx": 1e3, "mns": [], "dndxs": []},
            "high": {"mx": 1e8, "mns": [], "dndxs": []},
        },
        "trd": {
            "low": {"mx": 1.0, "mns": [], "dndxs": []},
            "mid": {"mx": 1e3, "mns": [], "dndxs": []},
            "high": {"mx": 1e8, "mns": [], "dndxs": []},
        },
    }

    for gen in GSTRS:
        spectra[gen]["low"]["mns"] = [10e-6, 2e-3, 10e-3, 100e-3, 150e-3, 250e-3]
        spectra[gen]["mid"]["mns"] = [10e-6, 100e-3, 500e-3, 50.0, 150.0, 500.0]
        spectra[gen]["high"]["mns"] = [10.0, 50.0, 300.0, 1e3, 1e5, 1e7]

    for gen, spec_obj in zip(GENERATIONS, spectra.values()):
        for key in spec_obj.keys():
            mx = spec_obj[key]["mx"]
            mns = spec_obj[key]["mns"]
            spec_obj[key]["dndxs"] = [
                generate_spectrum(xs, mn, theta, gen, eps, mx) for mn in mns
            ]

    return spectra


if __name__ == "__main__":

    theta = 1e-3
    eps = 0.1

    # mx = 1 GeV
    mx = 1 * GeV
    xs = np.geomspace(1e-6, 1.0, 200)
    mns = [10 * keV, 2 * MeV, 10 * MeV, 100 * MeV, 150 * MeV, 250 * MeV]

    for mn in mns:
        spec = generate_spectra(xs, mn, theta, Gen.Fst, True, 0.1, mx)
