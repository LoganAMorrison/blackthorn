from typing import Dict, Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import h5py

import utils
from blackthorn import Gen, RhNeutrinoMeV, RhNeutrinoGeV

RealArray = npt.NDArray[np.float64]

GOLDEN_RATIO = (1.0 + np.sqrt(5.0)) / 2.0

BRS_FILENAME = "brs"

LEPS = ["e", "mu", "tau"]
LEPBARS = ["ebar", "mubar", "taubar"]
NUS = ["nue", "numu", "nutau"]
US = ["u", "c", "t"]
UBARS = ["ubar", "cbar", "tbar"]
DS = ["d", "s", "b"]
DBARS = ["dbar", "sbar", "bbar"]

CATAGORIES = [
    "nu z",
    "nu h",
    "ell w",
    "ell ui uibar",
    "nu ui uibar",
    "nu di dibar",
    "nu li libar",
    "nu nu nu",
    "nu li li'bar",
]

CONFIG = {
    "mev": {
        "fst": {
            "ve a": {
                "x": 3e-4,
                "y": 8e-3,
                "label": r"$\nu_{e} + \gamma$",
                "color": "goldenrod",
            },
            "nu nu nu": {
                "x": 3e-4,
                "y": 6e-1,
                "label": r"$\nu + \nu + \nu$",
                "color": "steelblue",
            },
            "nu li libar": {
                "x": 2e-3,
                "y": 1e-1,
                "label": r"$\nu + \ell^{\mp}_{i} + \ell^{\pm}_{i}$",
                "color": "firebrick",
            },
        }
    }
}


def state_to_catagory(state: str):
    states = state.split(" ")
    cat = []

    for s in states:
        if s in ["ve", "vmu", "vtau"]:
            cat.append("nu")
        elif s in ["u", "c", "t"]:
            cat.append("ui")
        elif s in ["ubar", "cbar", "tbar"]:
            cat.append("uibar")
        elif s in ["d", "s", "b"]:
            cat.append("di")
        elif s in ["dbar", "sbar", "bbar"]:
            cat.append("dibar")
        elif s in ["e", "mu", "tau"]:
            cat.append("li")
        elif s in ["ebar", "mubar", "taubar"]:
            cat.append("libar")
        else:
            cat.append(s)
    return " ".join(cat)


def state_to_latex(state: str):

    if state == "ve h":
        return r"$\nu_{e} + h$"
    if state == "vmu h":
        return r"$\nu_{\mu} + h$"
    if state == "vtau h":
        return r"$\nu_{\tau} + h$"

    if state == "ve z":
        return r"$\nu_{e} + Z$"
    if state == "vmu z":
        return r"$\nu_{\mu} + Z$"
    if state == "vtau z":
        return r"$\nu_{\tau} + Z$"

    if state == "e w":
        return r"$e^{\mp} + W^{\pm}$"
    if state == "mu w":
        return r"$\mu^{\mp} + W^{\pm}$"
    if state == "tau w":
        return r"$\tau^{\mp} + W^{\pm}$"

    if state == "ve ui uibar":
        return r"$\nu_{e} + u_{i} + \bar{u}_{i}$"
    if state == "vmu ui uibar":
        return r"$\nu_{\mu} + u_{i} + \bar{u}_{i}$"
    if state == "vtau ui uibar":
        return r"$\nu_{\tau} + u_{i} + \bar{u}_{i}$"

    if state == "ve t tbar":
        return r"$\nu_{e} + t + \bar{t}$"
    if state == "vmu t tbar":
        return r"$\nu_{\mu} + t + \bar{t}$"
    if state == "vtau t tbar":
        return r"$\nu_{\tau} + t + \bar{t}$"

    if state == "ve di dibar":
        return r"$\nu_{e} + d_{i} + \bar{d}_{i}$"
    if state == "vmu di dibar":
        return r"$\nu_{\mu} + d_{i} + \bar{d}_{i}$"
    if state == "vtau di dibar":
        return r"$\nu_{\tau} + d_{i} + \bar{d}_{i}$"

    if state == "e ui djbar":
        return r"$e^{\mp} + u_{i} + \bar{d}_{j}$"
    if state == "mu ui djbar":
        return r"$\mu^{\mp} + u_{i} + \bar{d}_{j}$"
    if state == "tau ui djbar":
        return r"$\tau^{\mp} + u_{i} + \bar{d}_{j}$"

    if state == "nu li libar":
        return r"$\nu + \ell^{\pm}_{i} + \ell^{\mp}$"

    if state == "nu nu nu":
        return r"$\nu + \nu + \nu$"

    if state == "e pi":
        return r"$e^{\mp} + \pi^{\pm}$"
    if state == "mu pi":
        return r"$\mu^{\mp} + \pi^{\pm}$"
    if state == "tau pi":
        return r"$\tau^{\mp} + \pi^{\pm}$"

    if state == "ve pi0":
        return r"$\nu_{e} + \pi^{0}$"
    if state == "vmu pi0":
        return r"$\nu_{\mu} + \pi^{0}$"
    if state == "vtau pi0":
        return r"$\nu_{\tau} + \pi^{0}$"

    if state == "ve eta":
        return r"$\nu_{e} + \eta$"
    if state == "vmu eta":
        return r"$\nu_{\mu} + \eta$"
    if state == "vtau eta":
        return r"$\nu_{\tau} + \eta$"

    if state == "ve a":
        return r"$\nu_{e} + \gamma$"
    if state == "vmu a":
        return r"$\nu_{\mu} + \gamma$"
    if state == "vtau a":
        return r"$\nu_{\tau} + \gamma$"

    if state == "ve pi pibar":
        return r"$\nu_{e} + \pi^{\pm}+ \pi^{\mp}$"
    if state == "vmu pi pibar":
        return r"$\nu_{\mu} + \pi^{\pm}+ \pi^{\mp}$"
    if state == "vtau pi pibar":
        return r"$\nu_{\tau} + \pi^{\pm}+ \pi^{\mp}$"

    if state == "e pi pi0":
        return r"$e^{\mp} + \pi^{\pm} + \pi^{0}$"
    if state == "mu pi pi0":
        return r"$\mu^{\mp} + \pi^{\pm} + \pi^{0}$"
    if state == "tau pi pi0":
        return r"$\tau^{\mp} + \pi^{\pm} + \pi^{0}$"

    if state == "e k":
        return r"$e^{\mp} + K^{\pm}$"
    if state == "mu k":
        return r"$\mu^{\mp} + K^{\pm}$"
    if state == "tau k":
        return r"$\tau^{\mp} + K^{\pm}$"

    raise ValueError(f"Unknown state {state}")


def branching_fraction_catagorized_gev(data: Dict, gen: Gen, shape: Tuple[int, ...]):
    if gen == Gen.Fst:
        l0 = "e"
        l1 = "mu"
        l2 = "tau"
    elif gen == Gen.Snd:
        l0 = "mu"
        l1 = "e"
        l2 = "tau"
    else:
        l0 = "tau"
        l1 = "e"
        l2 = "mu"

    null = np.zeros(shape, dtype=np.float64)
    catagorized = {}

    catagorized[f"v{l0} h"] = data.get(f"v{l0} h", null)[:]
    catagorized[f"v{l0} z"] = data.get(f"v{l0} z", null)[:]
    catagorized[f"{l0} w"] = data.get(f"{l0} w", null)[:]

    # N -> v + u + u
    catagorized[f"v{l0} ui uibar"] = (
        data.get(f"v{l0} u ubar", null)[:] + data.get(f"v{l0} c cbar", null)[:]
    )
    catagorized[f"v{l0} t tbar"] = data.get(f"v{l0} t tbar", null)[:]
    # N -> v + d + d
    catagorized[f"v{l0} di dibar"] = (
        data.get(f"v{l0} d dbar", null)[:]
        + data.get(f"v{l0} s sbar", null)[:]
        + data.get(f"v{l0} b bbar", null)[:]
    )

    # N -> l + u + d
    catagorized[f"{l0} ui djbar"] = (
        data.get(f"{l0} u dbar", null)[:]
        + data.get(f"{l0} u sbar", null)[:]
        + data.get(f"{l0} u bbar", null)[:]
        + data.get(f"{l0} c dbar", null)[:]
        + data.get(f"{l0} c sbar", null)[:]
        + data.get(f"{l0} c bbar", null)[:]
        + data.get(f"{l0} t dbar", null)[:]
        + data.get(f"{l0} t sbar", null)[:]
        + data.get(f"{l0} t bbar", null)[:]
    )

    catagorized["nu li libar"] = (
        data.get(f"v{l0} {l0} {l0}bar", null)[:]
        + data.get(f"v{l0} {l1} {l1}bar", null)[:]
        + data.get(f"v{l0} {l2} {l2}bar", null)[:]
        + data.get(f"v{l1} {l0} {l1}bar", null)[:]
        + data.get(f"v{l2} {l0} {l2}bar", null)[:]
    )

    catagorized["nu nu nu"] = (
        data.get(f"v{l0} v{l0} v{l0}", null)[:]
        + data.get(f"v{l0} v{l1} v{l1}", null)[:]
        + data.get(f"v{l0} v{l2} v{l2}", null)[:]
    )

    return catagorized


def branching_fraction_catagorized_mev(data: Dict, gen: Gen, shape: Tuple[int, ...]):
    if gen == Gen.Fst:
        l0 = "e"
        l1 = "mu"
        l2 = "tau"
    elif gen == Gen.Snd:
        l0 = "mu"
        l1 = "e"
        l2 = "tau"
    else:
        l0 = "tau"
        l1 = "e"
        l2 = "mu"

    null = np.zeros(shape, dtype=np.float64)
    catagorized = {}

    catagorized[f"{l0} k"] = data.get(f"{l0} k", null)[:]
    catagorized[f"{l0} pi"] = data.get(f"{l0} pi", null)[:]
    catagorized[f"{l0} pi pi0"] = data.get(f"{l0} pi pi0", null)[:]
    catagorized[f"v{l0} pi0"] = data.get(f"v{l0} pi0", null)[:]
    catagorized[f"v{l0} eta"] = data.get(f"v{l0} eta", null)[:]
    catagorized[f"v{l0} a"] = data.get(f"v{l0} a", null)[:]
    catagorized[f"v{l0} pi pibar"] = data.get(f"v{l0} pi pibar", null)[:]

    catagorized["nu li libar"] = (
        data.get(f"v{l0} {l0} {l0}bar", null)[:]
        + data.get(f"v{l0} {l1} {l1}bar", null)[:]
        + data.get(f"v{l0} {l2} {l2}bar", null)[:]
        + data.get(f"v{l1} {l0} {l1}bar", null)[:]
        + data.get(f"v{l2} {l0} {l2}bar", null)[:]
    )

    catagorized["nu nu nu"] = (
        data.get(f"v{l0} v{l0} v{l0}", null)[:]
        + data.get(f"v{l0} v{l1} v{l1}", null)[:]
        + data.get(f"v{l0} v{l2} v{l2}", null)[:]
    )

    return catagorized


def gen_to_str(gen: Gen) -> str:
    if gen == Gen.Fst:
        return "fst"
    if gen == Gen.Snd:
        return "snd"
    return "trd"


line_styles = {
    "l k": {"color": "#48d1cc", "lw": 1},
    "l pi": {"color": "#5a7d9a", "lw": 1},
    "l pi pi0": {"color": "#9370db", "lw": 1},
    "nu pi0": {"color": "#b22222", "lw": 1},
    "nu eta": {"color": "#3cd371", "lw": 1},
    "nu a": {"color": "#daa520", "lw": 1},
    "nu pi pibar": {"color": "#c71585", "lw": 1},
    "nu li libar": {"color": "#ba55d3", "lw": 1},
    "nu nu nu": {"color": "#483d8b", "lw": 1},
    # GeV states
    "nu h": {"color": "#b22222", "lw": 1},
    "nu z": {"color": "#5a7d9a", "lw": 1},
    "l w": {"color": "#daa520", "lw": 1},
    "nu ui uibar": {"color": "#c71585", "lw": 1},
    "nu t tbar": {"color": "#48d1cc", "lw": 1},
    "nu di dibar": {"color": "#3cd371", "lw": 1},
    "l ui djbar": {"color": "#9370db", "lw": 1},
}


plot_config = {
    "mev": {
        Gen.Fst: {
            "e k": line_styles["l k"],
            "e pi": line_styles["l pi"],
            "e pi pi0": line_styles["l pi pi0"],
            "ve pi0": line_styles["nu pi0"],
            "ve eta": line_styles["nu eta"],
            "ve a": line_styles["nu a"],
            "ve pi pibar": line_styles["nu pi pibar"],
            "nu li libar": line_styles["nu li libar"],
            "nu nu nu": line_styles["nu nu nu"],
            "text": [
                {
                    "x": 2e-4,
                    "y": 7e-3,
                    "s": r"$\nu_{e} + \gamma$",
                    "fontdict": {"color": line_styles["nu a"]["color"], "size": 14},
                },
                {
                    "x": 2e-4,
                    "y": 5e-1,
                    "s": r"$\nu + \nu + \nu$",
                    "fontdict": {"color": line_styles["nu nu nu"]["color"], "size": 14},
                },
                {
                    "x": 2e-3,
                    "y": 1e-1,
                    "s": r"$\nu + \ell^{\pm}_{i} + \ell^{\mp}_{i}$",
                    "fontdict": {
                        "color": line_styles["nu li libar"]["color"],
                        "size": 14,
                    },
                },
                {
                    "x": 3e-2,
                    "y": 4e-2,
                    "s": r"$\nu_{e} + \pi^{0}$",
                    "fontdict": {"color": line_styles["nu pi0"]["color"], "size": 14},
                },
                {
                    "x": 9e-2,
                    "y": 6e-1,
                    "s": r"$e^{\mp} + \pi^{\pm}$",
                    "fontdict": {"color": line_styles["l pi"]["color"], "size": 14},
                },
            ],
            "xlims": [1e-4, 0.5],
            "ylims": [1e-3, 1e0],
            "title": {"label": "Electron", "fontdict": {"size": 16}},
        },
        Gen.Snd: {
            "mu k": line_styles["l k"],
            "mu pi": line_styles["l pi"],
            "mu pi pi0": line_styles["l pi pi0"],
            "vmu pi0": line_styles["nu pi0"],
            "vmu eta": line_styles["nu eta"],
            "vmu a": line_styles["nu a"],
            "vmu pi pibar": line_styles["nu pi pibar"],
            "nu li libar": line_styles["nu li libar"],
            "nu nu nu": line_styles["nu nu nu"],
            "text": [
                {
                    "x": 1e-2,
                    "y": 7e-3,
                    "s": r"$\nu_{\mu} + \gamma$",
                    "fontdict": {"color": line_styles["nu a"]["color"], "size": 14},
                },
                {
                    "x": 2e-3,
                    "y": 5e-1,
                    "s": r"$\nu + \nu + \nu$",
                    "fontdict": {"color": line_styles["nu nu nu"]["color"], "size": 14},
                },
                {
                    "x": 4e-3,
                    "y": 7e-2,
                    "s": r"$\nu + \ell^{\pm}_{i} + \ell^{\mp}_{i}$",
                    "fontdict": {
                        "color": line_styles["nu li libar"]["color"],
                        "size": 14,
                    },
                },
                {
                    "x": 4e-2,
                    "y": 1.5e-1,
                    "s": r"$\nu_{\mu} + \pi^{0}$",
                    "fontdict": {"color": line_styles["nu pi0"]["color"], "size": 14},
                },
                {
                    "x": 1.7e-1,
                    "y": 3e-3,
                    "s": r"$\mu^{\mp} + \pi^{\pm}$",
                    "fontdict": {"color": line_styles["l pi"]["color"], "size": 14},
                    "rotation": 90.0,
                },
            ],
            "xlims": [1e-3, 0.5],
            "ylims": [1e-3, 1e0],
            "title": {"label": "Muon", "fontdict": {"size": 16}},
        },
        Gen.Trd: {
            "tau k": line_styles["l k"],
            "tau pi": line_styles["l pi"],
            "tau pi pi0": line_styles["l pi pi0"],
            "vtau pi0": line_styles["nu pi0"],
            "vtau eta": line_styles["nu eta"],
            "vtau a": line_styles["nu a"],
            "vtau pi pibar": line_styles["nu pi pibar"],
            "nu li libar": line_styles["nu li libar"],
            "nu nu nu": line_styles["nu nu nu"],
            "text": [
                {
                    "x": 1e-2,
                    "y": 7e-3,
                    "s": r"$\nu_{\tau} + \gamma$",
                    "fontdict": {"color": line_styles["nu a"]["color"], "size": 14},
                },
                {
                    "x": 2e-3,
                    "y": 5e-1,
                    "s": r"$\nu + \nu + \nu$",
                    "fontdict": {"color": line_styles["nu nu nu"]["color"], "size": 14},
                },
                {
                    "x": 4e-3,
                    "y": 7e-2,
                    "s": r"$\nu + \ell^{\pm}_{i} + \ell^{\mp}_{i}$",
                    "fontdict": {
                        "color": line_styles["nu li libar"]["color"],
                        "size": 14,
                    },
                },
                {
                    "x": 4e-2,
                    "y": 1.5e-1,
                    "s": r"$\nu_{\tau} + \pi^{0}$",
                    "fontdict": {"color": line_styles["nu pi0"]["color"], "size": 14},
                },
            ],
            "xlims": [1e-3, 0.5],
            "ylims": [1e-3, 1.1e0],
            "title": {"label": "Tau", "fontdict": {"size": 16}},
        },
    },
    "gev": {
        Gen.Fst: {
            "ve h": line_styles["nu h"],
            "ve z": line_styles["nu z"],
            "e w": line_styles["l w"],
            "ve ui uibar": line_styles["nu ui uibar"],
            "ve t tbar": line_styles["nu t tbar"],
            "ve di dibar": line_styles["nu di dibar"],
            "e ui djbar": line_styles["l ui djbar"],
            "nu li libar": line_styles["nu li libar"],
            "nu nu nu": line_styles["nu nu nu"],
            "xlims": [5.0, 1e3],
            "ylims": [1e-2, 1e0],
            "title": {"label": "Electron", "fontdict": {"size": 16}},
            "text": [
                {
                    "x": 300,
                    "y": 6e-1,
                    "s": r"$e^{\mp} + W^{\pm}$",
                    "fontdict": {"color": line_styles["l w"]["color"], "size": 12},
                },
                {
                    "x": 130,
                    "y": 3.3e-1,
                    "s": r"$\nu_{e} + Z$",
                    "fontdict": {"color": line_styles["nu z"]["color"], "size": 12},
                },
                {
                    "x": 220,
                    "y": 8e-2,
                    "s": r"$\nu_{e} + h$",
                    "fontdict": {"color": line_styles["nu h"]["color"], "size": 12},
                },
                {
                    "x": 6,
                    "y": 6e-1,
                    "s": r"$e + u_{i} + \bar{d}_{j}$",
                    "fontdict": {
                        "color": line_styles["l ui djbar"]["color"],
                        "size": 12,
                    },
                },
                {
                    "x": 10,
                    "y": 1.5e-1,
                    "s": r"$\nu + \nu + \nu$",
                    "fontdict": {
                        "color": line_styles["nu nu nu"]["color"],
                        "size": 12,
                    },
                },
                {
                    "x": 6,
                    "y": 3.1e-1,
                    "s": r"$\nu + \ell_{i} + \bar{\ell}_{j}$",
                    "fontdict": {
                        "color": line_styles["nu li libar"]["color"],
                        "size": 12,
                    },
                },
                {
                    "x": 18,
                    "y": 8e-2,
                    "s": r"$\nu_{e} + d_{i} + d_{i}$",
                    "fontdict": {
                        "color": line_styles["nu di dibar"]["color"],
                        "size": 12,
                    },
                },
                {
                    "x": 18,
                    "y": 4e-2,
                    "s": r"$\nu_{e} + u_{i} + u_{i}$",
                    "fontdict": {
                        "color": line_styles["nu ui uibar"]["color"],
                        "size": 12,
                    },
                },
            ],
        },
        Gen.Snd: {
            "vmu h": line_styles["nu h"],
            "vmu z": line_styles["nu z"],
            "mu w": line_styles["l w"],
            "vmu ui uibar": line_styles["nu ui uibar"],
            "vmu t tbar": line_styles["nu t tbar"],
            "vmu di dibar": line_styles["nu di dibar"],
            "mu ui djbar": line_styles["l ui djbar"],
            "nu li libar": line_styles["nu li libar"],
            "nu nu nu": line_styles["nu nu nu"],
            "xlims": [5.0, 1e3],
            "ylims": [1e-2, 1e0],
            "title": {"label": "Muon", "fontdict": {"size": 16}},
            "text": [
                {
                    "x": 300,
                    "y": 6e-1,
                    "s": r"$\mu^{\mp} + W^{\pm}$",
                    "fontdict": {"color": line_styles["l w"]["color"], "size": 12},
                },
                {
                    "x": 130,
                    "y": 3.3e-1,
                    "s": r"$\nu_{\mu} + Z$",
                    "fontdict": {"color": line_styles["nu z"]["color"], "size": 12},
                },
                {
                    "x": 220,
                    "y": 8e-2,
                    "s": r"$\nu_{\mu} + h$",
                    "fontdict": {"color": line_styles["nu h"]["color"], "size": 12},
                },
                {
                    "x": 6,
                    "y": 6e-1,
                    "s": r"$\mu + u_{i} + \bar{d}_{j}$",
                    "fontdict": {
                        "color": line_styles["l ui djbar"]["color"],
                        "size": 12,
                    },
                },
                {
                    "x": 10,
                    "y": 1.5e-1,
                    "s": r"$\nu + \nu + \nu$",
                    "fontdict": {
                        "color": line_styles["nu nu nu"]["color"],
                        "size": 12,
                    },
                },
                {
                    "x": 6,
                    "y": 3.1e-1,
                    "s": r"$\nu + \ell_{i} + \bar{\ell}_{j}$",
                    "fontdict": {
                        "color": line_styles["nu li libar"]["color"],
                        "size": 12,
                    },
                },
                {
                    "x": 18,
                    "y": 8e-2,
                    "s": r"$\nu_{\mu} + d_{i} + d_{i}$",
                    "fontdict": {
                        "color": line_styles["nu di dibar"]["color"],
                        "size": 12,
                    },
                },
                {
                    "x": 18,
                    "y": 4e-2,
                    "s": r"$\nu_{\mu} + u_{i} + u_{i}$",
                    "fontdict": {
                        "color": line_styles["nu ui uibar"]["color"],
                        "size": 12,
                    },
                },
            ],
        },
        Gen.Trd: {
            "vtau h": line_styles["nu h"],
            "vtau z": line_styles["nu z"],
            "tau w": line_styles["l w"],
            "vtau ui uibar": line_styles["nu ui uibar"],
            "vtau t tbar": line_styles["nu t tbar"],
            "vtau di dibar": line_styles["nu di dibar"],
            "tau ui djbar": line_styles["l ui djbar"],
            "nu li libar": line_styles["nu li libar"],
            "nu nu nu": line_styles["nu nu nu"],
            "xlims": [5.0, 1e3],
            "ylims": [1e-2, 1.1e0],
            "title": {"label": "Tau", "fontdict": {"size": 16}},
            "text": [
                {
                    "x": 300,
                    "y": 6e-1,
                    "s": r"$\tau^{\mp} + W^{\pm}$",
                    "fontdict": {"color": line_styles["l w"]["color"], "size": 12},
                },
                {
                    "x": 130,
                    "y": 3.3e-1,
                    "s": r"$\nu_{\tau} + Z$",
                    "fontdict": {"color": line_styles["nu z"]["color"], "size": 12},
                },
                {
                    "x": 220,
                    "y": 8e-2,
                    "s": r"$\nu_{\tau} + h$",
                    "fontdict": {"color": line_styles["nu h"]["color"], "size": 12},
                },
                {
                    "x": 6,
                    "y": 5.4e-1,
                    "s": r"$\tau + u_{i} + \bar{d}_{j}$",
                    "fontdict": {
                        "color": line_styles["l ui djbar"]["color"],
                        "size": 12,
                    },
                },
                {
                    "x": 13,
                    "y": 1.5e-1,
                    "s": r"$\nu + \nu + \nu$",
                    "fontdict": {
                        "color": line_styles["nu nu nu"]["color"],
                        "size": 12,
                    },
                },
                {
                    "x": 6,
                    "y": 3.0e-1,
                    "s": r"$\nu + \ell_{i} + \bar{\ell}_{j}$",
                    "fontdict": {
                        "color": line_styles["nu li libar"]["color"],
                        "size": 12,
                    },
                },
                {
                    "x": 18,
                    "y": 8e-2,
                    "s": r"$\nu_{\tau} + d_{i} + d_{i}$",
                    "fontdict": {
                        "color": line_styles["nu di dibar"]["color"],
                        "size": 12,
                    },
                },
                {
                    "x": 18,
                    "y": 4e-2,
                    "s": r"$\nu_{\tau} + u_{i} + u_{i}$",
                    "fontdict": {
                        "color": line_styles["nu ui uibar"]["color"],
                        "size": 12,
                    },
                },
                # {
                #     "x": 360,
                #     "y": 5e-3,
                #     "s": r"$\nu_{\tau} + t + \bar{t}$",
                #     "fontdict": {
                #         "color": line_styles["nu t tbar"]["color"],
                #         "size": 12,
                #     },
                # },
            ],
        },
    },
}


def plot_branching_ratios_mev(datafile, gen: Gen, outfile):
    with h5py.File(datafile, "r") as f:
        gstr = gen_to_str(gen)
        data = f["mev"][gstr]  # type: ignore
        mns: RealArray = data["masses"][:]  # type: ignore
        brs_raw: Dict[str, RealArray] = data["branching_fractions"]  # type: ignore
        brs = branching_fraction_catagorized_mev(brs_raw, gen, (len(mns),))

    config = plot_config["mev"][gen]
    plt.figure(dpi=150, figsize=(6, 4))

    for key, val in brs.items():
        if np.max(val) > min(config["ylims"]):
            plt.plot(mns, val, label=state_to_latex(key), **config[key])

    for conf in config["text"]:
        plt.text(**conf)

    plt.title(**config["title"])
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(*config["ylims"])
    plt.xlim(*config["xlims"])
    plt.ylabel(r"$\mathrm{Branching} \ \mathrm{Ratio}$", fontsize=16)
    plt.xlabel(r"$m_{N} \ [\mathrm{GeV}]$", fontsize=16)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)


def plot_branching_ratios_mev_together(datafile, outfile):
    gens = [Gen.Fst, Gen.Snd, Gen.Trd]
    mns = dict()
    brs = dict()
    with h5py.File(datafile, "r") as f:
        for gen in gens:
            gstr = gen_to_str(gen)
            data = f["mev"][gstr]  # type: ignore
            mns[gen]: RealArray = data["masses"][:]  # type: ignore
            brs_raw: Dict[str, RealArray] = data["branching_fractions"]  # type: ignore
            brs[gen] = branching_fraction_catagorized_mev(
                brs_raw, gen, (len(mns[gen]),)
            )

    _, axes = plt.subplots(nrows=1, ncols=3, sharey=True, dpi=150, figsize=(12, 4))
    print(axes)
    for (gen, ax) in zip(gens, axes):

        config = plot_config["mev"][gen]

        for key, val in brs[gen].items():
            if np.max(val) > min(config["ylims"]):
                ax.plot(mns[gen], val, label=state_to_latex(key), **config[key])

        for conf in config["text"]:
            ax.text(**conf)

        ax.set_title(**config["title"])
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim(*config["ylims"])
        ax.set_xlim(*config["xlims"])
        # ax.set_ylabel(r"$\mathrm{Branching} \ \mathrm{Ratio}$", fontsize=16)
        ax.set_xlabel(r"$m_{N} \ [\mathrm{GeV}]$", fontsize=16)
        utils.configure_ticks(ax)

    axes[0].set_ylabel(r"$\mathrm{Branching} \ \mathrm{Ratio}$", fontsize=16)

    plt.tight_layout()
    plt.savefig(outfile)


def plot_branching_ratios_gev(datafile, gen: Gen, outfile):
    with h5py.File(datafile, "r") as f:
        gstr = gen_to_str(gen)
        data = f["gev"][gstr]  # type: ignore
        mns: RealArray = data["masses"][:]  # type: ignore
        brs_raw: Dict[str, RealArray] = data["branching_fractions"]  # type: ignore
        brs = branching_fraction_catagorized_gev(brs_raw, gen, (len(mns),))

    config = plot_config["gev"][gen]
    plt.figure(dpi=150, figsize=(6, 4))

    for key, val in brs.items():
        if np.max(val) > min(config["ylims"]):
            plt.plot(mns, val, label=state_to_latex(key), **config[key])

    for conf in config["text"]:
        plt.text(**conf)

    plt.title(**config["title"])
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(*config["ylims"])
    plt.xlim(*config["xlims"])
    plt.ylabel(r"$\mathrm{Branching} \ \mathrm{Ratio}$", fontsize=16)
    plt.xlabel(r"$m_{N} \ [\mathrm{GeV}]$", fontsize=16)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)


def plot_branching_ratios_gev_together(datafile, outfile):
    gens = [Gen.Fst, Gen.Snd, Gen.Trd]
    mns = dict()
    brs = dict()
    with h5py.File(datafile, "r") as f:
        for gen in gens:
            gstr = gen_to_str(gen)
            data = f["gev"][gstr]  # type: ignore
            mns[gen]: RealArray = data["masses"][:]  # type: ignore
            brs_raw: Dict[str, RealArray] = data["branching_fractions"]  # type: ignore
            brs[gen] = branching_fraction_catagorized_gev(
                brs_raw, gen, (len(mns[gen]),)
            )

    _, axes = plt.subplots(nrows=1, ncols=3, sharey=True, dpi=150, figsize=(12, 4))
    print(axes)
    for (gen, ax) in zip(gens, axes):

        config = plot_config["gev"][gen]

        for key, val in brs[gen].items():
            if np.max(val) > min(config["ylims"]):
                ax.plot(mns[gen], val, label=state_to_latex(key), **config[key])

        for conf in config["text"]:
            ax.text(**conf)

        ax.set_title(**config["title"])
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim(*config["ylims"])
        ax.set_xlim(*config["xlims"])
        ax.set_xlabel(r"$m_{N} \ [\mathrm{GeV}]$", fontsize=16)
        utils.configure_ticks(ax)

    axes[0].set_ylabel(r"$\mathrm{Branching} \ \mathrm{Ratio}$", fontsize=16)

    plt.tight_layout()
    plt.savefig(outfile)


if __name__ == "__main__":
    results_dir = Path(__file__).parent.joinpath("results")
    figures_dir = Path(__file__).parent.joinpath("figures")
    datafile = results_dir.joinpath("brs2").with_suffix(".hdf5")
    # overwrite = False
    # if not datafile.exists() or overwrite:
    #     generate_br_data(overwrite=overwrite)

    # plot_branching_ratios_mev_together(
    #     datafile, figures_dir.joinpath("brs_mev_all.pdf")
    # )
    # plot_branching_ratios_gev_together(
    #     datafile, figures_dir.joinpath("brs_gev_all.pdf")
    # )
    plot_branching_ratios_mev_gev_together(
        datafile, figures_dir.joinpath("brs_mev_gev_all.pdf")
    )
