from pathlib import Path

import numpy as np
import numpy.typing as npt
import h5py
from scipy.interpolate import UnivariateSpline
from HDMSpectra import HDMSpectra

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

from blackthorn import Gen, RhNeutrinoMeV, RhNeutrinoGeV, RhNeutrinoTeV, Spectrum
from blackthorn import fields

RealArray = npt.NDArray[np.float64]

GeV = 1.0
MeV = 1e-3
keV = 1e-6


VIRIDIS = plt.get_cmap("viridis").colors


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


def generate_spectra(
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


def number_to_latex(x: float) -> str:
    lmn = np.log10(x)
    exponent = int(np.floor(lmn))
    ab = x / 10 ** exponent

    if exponent < -6:
        # 1.4e-7 => 1.4 * 10^{-1} keV
        exponent += 6
        return f"{ab}\\times 10^{{{exponent}}} \\ \\mathrm{{keV}}"
    elif -6 < exponent < -5:
        # 1.4e-6 => 1.4 keV
        return f"{ab}\\times 10^{{{exponent}}} \\ \\mathrm{{keV}}"
    elif -5 < exponent < -4:
        # 1.4e-5 => 14 keV
        return f"{ab*10}\\times \\ \\mathrm{{keV}}"
    elif -4 < exponent < -3:
        # 1.4e-4 => 140 keV
        return f"{ab*100}\\times \\ \\mathrm{{keV}}"
    elif -3 < exponent < -2:
        # 1.4e-3 => 1.4 MeV
        return f"{ab}\\times \\ \\mathrm{{MeV}}"
    elif -2 < exponent < -1:
        # 1.4e-2 => 14 MeV
        return f"{ab*10}\\times \\ \\mathrm{{MeV}}"
    elif -1 < exponent < 0:
        # 1.4e-1 => 140 MeV
        return f"{ab*100}\\times \\ \\mathrm{{MeV}}"
    elif 0 < exponent < -1:
        # 1.4 => 1.4 GeV
        return f"{ab}\\times \\mathrm{{GeV}}"
    else:
        return f"{ab}\\times 10^{{{exponent}}} \\ \\mathrm{{GeV}}"


def plot_dndx_vs_bb(gen, outfile, theta=1e-3, eps=0.1):
    specbb = SpectrumBB()
    xs = np.geomspace(1e-6, 1.0, 200)
    mxs = [100.0, 1e3, 1e8]

    spectra = [{"mx": mx, "half": None, "tenth": None, "bb": None} for mx in mxs]

    labels = {
        "half": [
            r"$m_{\chi} = 100 \ \mathrm{GeV}, m_{N} = \mathrm{m_{\chi}} / 2$",
            r"$m_{\chi} = 1 \ \mathrm{TeV}, m_{N} = \mathrm{m_{\chi}} / 2$",
            r"$m_{\chi} = 10^{8} \ \mathrm{GeV}, m_{N} = \mathrm{m_{\chi}} / 2$",
        ],
        "tenth": [
            r"$m_{\chi} = 100 \ \mathrm{GeV}, m_{N} = \mathrm{m_{\chi}} / 10$",
            r"$m_{\chi} = 1 \ \mathrm{TeV}, m_{N} = \mathrm{m_{\chi}} / 10$",
            r"$m_{\chi} = 10^{8} \ \mathrm{GeV}, m_{N} = \mathrm{m_{\chi}} / 10$",
        ],
    }

    for i, item in enumerate(spectra):
        mx = item["mx"]
        spectra[i]["half"] = generate_spectra(xs, mx / 2.0, theta, gen, eps, mx)
        spectra[i]["tenth"] = generate_spectra(xs, mx / 10.0, theta, gen, eps, mx)
        spectra[i]["bb"] = specbb.dndx_decay(mx, xs, eps)

    colors = ["firebrick", "steelblue", "goldenrod"]

    for i, item in enumerate(spectra):
        plt.plot(xs, item["half"] / item["bb"], label=labels["half"][i], c=colors[i])
        plt.plot(
            xs,
            item["tenth"] / item["bb"],
            label=labels["tenth"][i],
            ls="--",
            c=colors[i],
        )

    handles = [
        Line2D([], [], c="firebrick", label=r"$m_{\chi} = 100 \ \mathrm{GeV}$"),
        Line2D([], [], c="steelblue", label=r"$m_{\chi} = 10^{3} \ \mathrm{GeV}$"),
        Line2D([], [], c="goldenrod", label=r"$m_{\chi} = 10^{8} \ \mathrm{GeV}$"),
        Line2D([], [], c="k", label=r"$m_{N} = m_{\chi}/2$", ls="-"),
        Line2D([], [], c="k", label=r"$m_{N} = m_{\chi}/10$", ls="--"),
    ]

    plt.ylabel(r"$\frac{dN_{N}}{dx} / \frac{dN_{b\bar{b}}}{dx}$", fontsize=16)
    plt.xlabel(r"$x = 2E_{\gamma}/m_{\chi}$", fontsize=16)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([1e-3, 1.0])
    plt.ylim([1e-2, 1e3])
    plt.legend(handles=handles, frameon=False)
    plt.savefig(outfile)


def plot_dndx_vs_bb2(gen, outfile, theta=1e-3, eps=0.1):
    specbb = SpectrumBB()
    xs = np.geomspace(1e-6, 1.0, 200)
    mxs = [100.0, 1e3, 1e8]

    spectra = [
        {
            "mx": 100.0,
            "spectra": [
                {"ratio": 0.5, "spec": None},
                {"ratio": 1.0 / 5.0, "spec": None},
                {"ratio": 1.0 / 10.0, "spec": None},
            ],
        },
        {
            "mx": 1e3,
            "spectra": [
                {"ratio": 1.0 / 10.0, "spec": None},
                {"ratio": 1.0 / 50.0, "spec": None},
                {"ratio": 1.0 / 100.0, "spec": None},
            ],
        },
        {
            "mx": 1e8,
            "spectra": [
                {"ratio": 1e-1, "spec": None},
                {"ratio": 1e-2, "spec": None},
                {"ratio": 1e-3, "spec": None},
            ],
        },
    ]

    for i, item in enumerate(spectra):
        mx = item["mx"]
        for j, spec in item["spectra"]:
            r = spec["ratio"]
            # item[i]["spectra"]["spec"] = generate_spectra(xs, mx / 2.0, theta, gen, eps, mx)
        # spectra[i]["fifth"] = generate_spectra(xs, mx / 5.0, theta, gen, eps, mx)
        # spectra[i]["tenth"] = generate_spectra(xs, mx / 10.0, theta, gen, eps, mx)
        # spectra[i]["bb"] = specbb.dndx_decay(mx, xs, eps)

    titles = [
        r"$m_{\chi} = 100 \ \mathrm{GeV}$",
        r"$m_{\chi} = 10^{3} \ \mathrm{GeV}$",
        r"$m_{\chi} = 10^{8} \ \mathrm{GeV}$",
    ]
    confs = {
        "half": {
            "color": "firebrick",
            "linestyle": "-",
            "linewidth": 1,
            "label": r"$m_{N} = m_{\chi}/2$",
        },
        "fifth": {
            "color": "steelblue",
            "linestyle": "--",
            "linewidth": 1,
            "label": r"$m_{N} = m_{\chi}/5$",
        },
        "tenth": {
            "color": "goldenrod",
            "linestyle": "-.",
            "linewidth": 1,
            "label": r"$m_{N} = m_{\chi}/10$",
        },
    }

    _, axes = plt.subplots(
        nrows=1, ncols=3, dpi=150, figsize=(10, 3), sharey=True, sharex=True
    )

    for i in range(len(spectra)):
        dndx = spectra[i]
        for key, val in confs.items():
            axes[i].plot(xs, dndx[key] / dndx["bb"], **val)

    axes[0].legend(frameon=False)

    axes[0].set_ylabel(r"$\frac{dN_{N}}{dx} / \frac{dN_{b\bar{b}}}{dx}$", fontsize=16)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim([1e-3, 1.0])
    axes[0].set_ylim([1e-2, 1e3])

    for i, ax in enumerate(axes):
        ax.set_xlabel(r"$x = 2E_{\gamma}/m_{\chi}$", fontsize=16)
        ax.text(3e-2, 3e-2, titles[i], fontdict={"size": 14})

    plt.tight_layout()
    plt.savefig(outfile)


def plot_dndx(dataset, outfile):
    def set_xlabel(ax):
        ax.set_xlabel(r"$x = 2E_{\gamma}/\sqrt{s}$", fontdict={"size": 16})

    labels = [
        [
            r"$10 \ \mathrm{keV}$",
            r"$2 \ \mathrm{MeV}$",
            r"$10 \ \mathrm{MeV}$",
            r"$100 \ \mathrm{MeV}$",
            r"$150 \ \mathrm{MeV}$",
            r"$250 \ \mathrm{MeV}$",
        ],
        [
            r"$10 \ \mathrm{keV}$",
            r"$100 \ \mathrm{MeV}$",
            r"$500 \ \mathrm{MeV}$",
            r"$50 \ \mathrm{GeV}$",
            r"$150 \ \mathrm{GeV}$",
            r"$500 \ \mathrm{GeV}$",
        ],
        [
            r"$10 \ \mathrm{GeV}$",
            r"$50 \ \mathrm{GeV}$",
            r"$300 \ \mathrm{GeV}$",
            r"$10^{3} \ \mathrm{GeV}$",
            r"$10^{5} \ \mathrm{GeV}$",
            r"$10^{7} \ \mathrm{GeV}$",
        ],
    ]

    titles = [
        r"$m_{\chi} = 1 \ \mathrm{GeV}$",
        r"$m_{\chi} = 10^{3} \ \mathrm{GeV}$",
        r"$m_{\chi} = 10^{8} \ \mathrm{GeV}$",
    ]

    xlims = [
        (1e-3, 1.0),
        (1e-4, 1.0),
        (1e-4, 1.0),
    ]

    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(12, 4), sharey=True, sharex=False
    )
    fig.dpi = 150

    mx_key_idx = [(key, dataset[key].attrs["mx"]) for key in dataset.keys()]

    mx_key_idx.sort(key=lambda a: a[-1])
    print(mx_key_idx)

    cs = ["#003f5c", "#444e86", "#955196", "#dd5182", "#ff6e54", "#ffa600"]

    for i, (key, _) in enumerate(mx_key_idx):
        xs = dataset[key]["xs"][:]
        mns = dataset[key]["masses"][:]
        dndxs = dataset[key]["dndxs"][:]
        ax = axes[i]
        for j, (mn, dndx) in enumerate(zip(mns, dndxs)):
            label = None if i == 1 else labels[i][j]
            ax.plot(xs, xs ** 2 * dndx, label=label, color=cs[j])

        if "bb" in dataset[key].keys():
            bb = dataset[key]["bb"]
            ax.plot(xs, xs ** 2 * bb, color="k", ls="--")

        ax.set_title(titles[i], fontdict={"size": 16})
        ax.set_xlim(*xlims[i])
        ax.set_ylim(1e-5, 1)
        ax.set_yscale("log")
        ax.set_xscale("log")
        set_xlabel(ax)

    # ===============================
    # ---- Labels for 1 GeV Plot ----
    # ===============================

    ax: Axes = axes[0]

    lx, ly = -2.8, -0.6
    ax.text(
        10 ** lx,
        10 ** ly,
        r"$\bm{\mathrm{Photon}}$",
        fontdict={"size": 14, "color": "k"},
    )

    lx, ly = -2.5, -3.7
    ax.text(
        10 ** lx,
        10 ** ly,
        r"$500 \ \mathrm{GeV}$",
        fontdict={"size": 12, "color": cs[5]},
        rotation=25,
    )
    lx, ly = -2.5, -4.15
    ax.text(
        10 ** lx,
        10 ** ly,
        r"$150 \ \mathrm{GeV}$",
        fontdict={"size": 12, "color": cs[4]},
        rotation=25,
    )
    lx, ly = -1.2, -3.2
    ax.text(
        10 ** lx,
        10 ** ly,
        r"$50 \ \mathrm{GeV}$",
        fontdict={"size": 12, "color": cs[3]},
        rotation=23,
    )
    lx, ly = -1.7, -4
    ax.text(
        10 ** lx,
        10 ** ly,
        r"$500 \ \mathrm{MeV}$",
        fontdict={"size": 12, "color": cs[2]},
        rotation=25,
    )
    lx, ly = -1.8, -4.8
    ax.text(
        10 ** lx,
        10 ** ly,
        r"$100 \ \mathrm{MeV}$",
        fontdict={"size": 12, "color": cs[1]},
        rotation=30,
    )
    lx, ly = -1.2, -4.9
    ax.text(
        10 ** lx,
        10 ** ly,
        r"$10 \ \mathrm{keV}$",
        fontdict={"size": 12, "color": cs[0]},
        rotation=45,
    )

    # ==================================
    # ---- Labels for 1000 GeV Plot ----
    # ==================================

    ax: Axes = axes[1]
    lx, ly = -2.0, -0.8
    ax.text(10 ** lx, 10 ** ly, r"$b\bar{b}$", fontdict={"size": 16})
    lx, ly = -3.9, -3.4
    ax.text(
        10 ** lx,
        10 ** ly,
        r"$500 \ \mathrm{GeV}$",
        fontdict={"size": 12, "color": cs[-1]},
        rotation=50,
    )
    lx, ly = -3.8, -4.1
    ax.text(
        10 ** lx,
        10 ** ly,
        r"$150 \ \mathrm{GeV}$",
        fontdict={"size": 12, "color": cs[-2]},
        rotation=50,
    )
    lx, ly = -3.6, -4.6
    ax.text(
        10 ** lx,
        10 ** ly,
        r"$50 \ \mathrm{GeV}$",
        fontdict={"size": 12, "color": cs[-3]},
        rotation=50,
    )
    lx, ly = -3.3, -4.9
    ax.text(
        10 ** lx,
        10 ** ly,
        r"$500 \ \mathrm{MeV}$",
        fontdict={"size": 12, "color": cs[-4]},
        rotation=35,
    )
    lx, ly = -2.4, -4.7
    ax.text(
        10 ** lx,
        10 ** ly,
        r"$100 \ \mathrm{MeV}$",
        fontdict={"size": 12, "color": cs[-5]},
        rotation=35,
    )
    lx, ly = -1.1, -4.7
    ax.text(
        10 ** lx,
        10 ** ly,
        r"$10 \ \mathrm{keV}$",
        fontdict={"size": 12, "color": cs[-6]},
        rotation=52,
    )

    # =================================
    # ---- Labels for 1e8 GeV Plot ----
    # =================================

    lx, ly = -3.5, -1.5
    axes[2].text(10 ** lx, 10 ** ly, r"$b\bar{b}$", fontdict={"size": 16})

    axes[0].set_ylabel(r"$x^2 \dv*{N_{\gamma}}{x}$", fontdict={"size": 16})

    axes[2].legend(frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(outfile)


if __name__ == "__main__":
    this_dir = Path(__file__).parent
    result_dir = this_dir.joinpath("..").joinpath("results")
    figure_dir = Path(__file__).parent.joinpath("figures")

    # plot_dndx_vs_bb2(Gen.Fst, FIGURE_DIR.joinpath("ratio_e.pdf"))
    # plot_dndx_1gev(Gen.Fst, FIGURE_DIR.joinpath("dndx_e_1gev.pdf"))
    # plot_dndx_1e3gev(Gen.Fst, FIGURE_DIR.joinpath("dndx_e_1e3gev.pdf"))
    with h5py.File(result_dir.joinpath("photon_dndx_e.hdf5")) as f:
        plot_dndx(f, figure_dir.joinpath("photon_dndx_e.pdf"))

    # plt.savefig(FIGURE_DIR.joinpath("ratio_e.pdf"))
