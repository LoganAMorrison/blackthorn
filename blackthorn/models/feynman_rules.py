"""Feynman rules for the RHN model."""

import numpy as np
from helax.vertices import VertexFFS, VertexFFV

from ..constants import CKM, CW, QE, SW
from ..fields import (BottomQuark, CharmQuark, DownQuark, Electron, Higgs,
                      Muon, StrangeQuark, Tau, TopQuark, UpQuark)

IM = 1.0j
SQRT_2 = np.sqrt(2.0)
PRE_WVL = QE / (SQRT_2 * SW)
PRE_ZVL = QE / (2.0 * SW * CW)


def _vertex_zff(weak_isospin, charge):
    return VertexFFV(
        left=QE * (weak_isospin - charge * SW**2) / (SW * CW),
        right=-charge * QE * SW / CW,
    )


VERTEX_ZLL = _vertex_zff(-0.5, -1.0)
VERTEX_ZUU = _vertex_zff(0.5, 2.0 / 3)
VERTEX_ZDD = _vertex_zff(-0.5, -1.0 / 3)


def _vertex_hff(mass):
    return VertexFFS(
        left=-mass / Higgs.vev,
        right=-mass / Higgs.vev,
    )


VERTEX_HLL = [_vertex_hff(p.mass) for p in [Electron, Muon, Tau]]
VERTEX_HUU = [_vertex_hff(p.mass) for p in [UpQuark, CharmQuark, TopQuark]]
VERTEX_HDD = [_vertex_hff(p.mass) for p in [DownQuark, StrangeQuark, BottomQuark]]


def pmns_left(theta, genn, genv):
    """Compute the left-handed value of the non-zero PMNS matrix."""
    if genn == genv:
        return np.cos(theta) * IM
    return 1.0 + 0.0 * IM


def pmns_right(theta, genn, genv):
    """Compute the right-handed value of the non-zero PMNS matrix."""
    if genn == genv:
        return np.sin(theta) + 0 * IM
    return 0 * IM


def kld_kl(theta, genn, genv1, genv2):
    """Compute the KL^d * KL."""
    if genv1 == genv2:
        if genn == genv1:
            return np.cos(theta) ** 2
        return 1.0
    return 0


def kld_kr(theta, genn, genv1):
    """Compute the KL^d * KR."""
    if genn == genv1:
        return IM * np.cos(theta) * np.sin(theta)
    return 0


def krd_kl(theta, genn, genv1):
    """Compute the KR^d * KL."""
    if genn == genv1:
        return -IM * np.cos(theta) * np.sin(theta)
    return 0


def kl_krc_mvr(theta, mass, genn, genv1):
    """Compute the KL * KR^c * mvr."""
    if genn == genv1:
        return -IM * mass * np.cos(theta) * np.sin(theta)
    return 0


def kr_klc_mvl(theta, mass, genn, genv1):
    """Compute the KR * KL^c * mvl."""
    if genn == genv1:
        return IM * mass * np.sin(theta) ** 2 * np.tan(theta)
    return 0


def krd_kl_mvl(theta, mass, genn, genv1):
    """Compute the KR^d * KL * mvl."""
    if genn == genv1:
        return -IM * mass * np.sin(theta) ** 2 * np.tan(theta)
    return 0


def kld_kr_mvr(theta, mass, genn, genv1):
    """Compute the KL^d * KR * mvr."""
    if genn == genv1:
        return IM * mass * np.sin(theta) * np.cos(theta)
    return 0


# ============================================================================
# ---- W-Lepton-Lepton -------------------------------------------------------
# ============================================================================


def vertex_wnl(theta, *, genn, genl, _: bool = True):
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
            sgn = -1.0 if l_in else 1.0
            left = sgn * IM * QE / (SQRT_2 * SW) * np.cos(theta)
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
    """Vertex for Z,N,V."""
    if genn == genv:
        coupling = IM * QE * np.sin(2 * theta) / (4 * CW * SW)
    else:
        coupling = 0.0

    return VertexFFV(
        left=coupling,
        right=coupling,
    )


def vertex_zvv(theta, *, genn, genv1, genv2):
    """Vertex for Z and two neutrinos."""
    if genv1 == genv2:
        coupling = QE / (2 * CW * SW)
        if genn == genv1:
            coupling = coupling * np.cos(theta) ** 2
    else:
        coupling = 0.0

    return VertexFFV(
        left=coupling,
        right=-coupling,
    )


# ============================================================================
# ---- H-Lepton-Lepton -------------------------------------------------------
# ============================================================================


def vertex_hnv(theta, mass, *, genn, genv):
    """
    Vertex for Z,N,V.
    """
    if genn == genv:
        yuk = mass / Higgs.vev
        sint = np.sin(theta)
        cost = np.cos(theta)
        tant = np.tan(theta)
        left = IM * yuk * tant * (cost**2 - sint**2)
    else:
        left = 0.0
    return VertexFFS(left=left, right=-left)


def vertex_hvv(theta, mass, *, genn, genv1, genv2):
    """
    Vertex for Z,N,V.
    """
    if genn == genv1 == genv2:
        coupling = -2 * mass * np.sin(theta) ** 2 / Higgs.vev
    else:
        coupling = 0.0
    return VertexFFS(left=coupling, right=coupling)
