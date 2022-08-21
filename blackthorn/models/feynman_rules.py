import numpy as np
from helax.vertices import VertexFFS, VertexFFV

from ..constants import CKM, CW, QE, SW
from ..fields import (
    BottomQuark,
    CharmQuark,
    DownQuark,
    Electron,
    Higgs,
    Muon,
    StrangeQuark,
    Tau,
    TopQuark,
    UpQuark,
    WBoson,
    ZBoson,
)

IM = 1.0j
SQRT_2 = np.sqrt(2.0)
PRE_WVL = QE / (SQRT_2 * SW)
PRE_ZVL = QE / (2.0 * SW * CW)


def _vertex_zff(t3, q):
    return VertexFFV(
        left=QE * (t3 - q * SW**2) / (SW * CW),
        right=-q * QE * SW / CW,
    )


VERTEX_ZLL = _vertex_zff(-0.5, -1.0)
VERTEX_ZUU = _vertex_zff(0.5, 2.0 / 3)
VERTEX_ZDD = _vertex_zff(-0.5, -1.0 / 3)


def _vertex_hff(mass):
    return VertexFFV(
        left=-mass / Higgs.vev,
        right=-mass / Higgs.vev,
    )


VERTEX_HLL = [_vertex_hff(p.mass) for p in [Electron, Muon, Tau]]
VERTEX_HUU = [_vertex_hff(p.mass) for p in [UpQuark, CharmQuark, TopQuark]]
VERTEX_HDD = [_vertex_hff(p.mass) for p in [DownQuark, StrangeQuark, BottomQuark]]


def pmns_left(theta, genn, genv):
    if genn == genv:
        return np.cos(theta) * IM
    else:
        return 1.0 + 0.0 * IM


def pmns_right(theta, genn, genv):
    if genn == genv:
        return np.sin(theta) + 0 * IM
    else:
        return 0 * IM


def kld_kl(theta, genn, genv1, genv2):
    if genv1 == genv2:
        if genn == genv1:
            return np.cos(theta) ** 2
        return 1.0
    return 0


def kld_kr(theta, genn, genv1):
    if genn == genv1:
        return IM * np.cos(theta) * np.sin(theta)
    return 0


def krd_kl(theta, genn, genv1):
    if genn == genv1:
        return -IM * np.cos(theta) * np.sin(theta)
    return 0


def kl_krc_mvr(theta, mass, genn, genv1):
    if genn == genv1:
        return -IM * mass * np.cos(theta) * np.sin(theta)
    return 0


def kr_klc_mvl(theta, mass, genn, genv1):
    if genn == genv1:
        return IM * mass * np.sin(theta) ** 2 * np.tan(theta)
    return 0


def krd_kl_mvl(theta, mass, genn, genv1):
    if genn == genv1:
        return -IM * mass * np.sin(theta) ** 2 * np.tan(theta)
    return 0


def kld_kr_mvr(theta, mass, genn, genv1):
    if genn == genv1:
        return IM * mass * np.sin(theta) * np.cos(theta)
    return 0


# ============================================================================
# ---- W-Lepton-Lepton -------------------------------------------------------
# ============================================================================


def vertex_wnl(theta, *, genn, genl, l_in: bool):
    """
    Vertex for W,N,L.
    """

    if genn == genl:
        left = QE / (SQRT_2 * SW) * np.sin(theta)
    else:
        left = 0.0
    return VertexFFV(left=left, right=0.0)


def vertex_wvl(theta, *, genn, genv, genl, l_in: bool):
    """
    Vertex for W,V,L.
    """
    if genv == genl:
        if genn == genv:
            s = -1.0 if l_in else 1.0
            left = s * IM * QE / (SQRT_2 * SW) * np.cos(theta)
        else:
            left = QE / (SQRT_2 * SW)
    else:
        left = 0.0
    return VertexFFV(left=left, right=0.0)


def vertex_wud(*, genu, gend, u_in: bool):
    """
    Vertex for W,V,L.
    """
    ckm = CKM[genu, gend]
    if u_in:
        ckm = np.conj(ckm)
    pre = QE / (SQRT_2 * SW)
    left = pre * ckm
    return VertexFFV(left=left, right=0.0)


# ============================================================================
# ---- Z-Lepton-Lepton -------------------------------------------------------
# ============================================================================


def vertex_znv(theta, *, genn, genv):
    """
    Vertex for Z,N,V.
    """
    if genn == genv:
        left = IM * QE * np.sin(2 * theta) / (4 * CW * SW)
        right = left
    else:
        left = 0.0
        right = 0.0

    return VertexFFV(left=left, right=right)


def vertex_zvv(theta, *, genn, genv1, genv2):
    if genv1 == genv2:
        left = QE / (2 * CW * SW)
        if genn == genv1:
            left = left * np.cos(theta) ** 2
        left = left
        right = -left
    else:
        left = 0.0
        right = 0.0

    return VertexFFV(left=left, right=right)


# ============================================================================
# ---- H-Lepton-Lepton -------------------------------------------------------
# ============================================================================


def vertex_hnv(theta, mass, *, genn, genv):
    """
    Vertex for Z,N,V.
    """
    vh = Higgs.vev
    if genn == genv:
        y = mass / vh
        st = np.sin(theta)
        ct = np.cos(theta)
        left = -IM * y * st / ct * (-(ct**2) + st**2)
        right = -IM * y * st / ct * (-(st**2) + ct**2)
    else:
        left = 0.0
        right = 0.0
    return VertexFFS(left=left, right=right)


def vertex_hvv(theta, mass, *, genn, genv1, genv2):
    """
    Vertex for Z,N,V.
    """
    vh = Higgs.vev
    if genn == genv1 == genv2:
        c = -2 * mass * np.sin(theta) ** 2 / vh
    else:
        c = 0.0
    return VertexFFS(left=c, right=c)
