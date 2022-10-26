"""

"""
# TODO: l + pi + pi0 seems wrong.

from typing import List, Sequence, Union

# import jax
# import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from helax.numpy import amplitudes, wavefunctions
from helax.vertices import VertexFFS, VertexFFV

from .. import fields
from ..constants import CKM_UD, CW, GF, SW
from ..fields import (ChargedPion, Electron, Higgs, Muon, NeutralPion,
                      ScalarBoson, Tau, VectorBoson, WBoson, ZBoson)
from . import feynman_rules
from .base import RhNeutrinoBase

DiracWf = wavefunctions.DiracWf
VectorWf = wavefunctions.VectorWf
ScalarWf = wavefunctions.ScalarWf

Wavefunction = Union[ScalarWf, DiracWf, VectorWf]
Vertex = Union[VertexFFV, VertexFFS]

MASS_W = WBoson.mass
MASS_Z = ZBoson.mass
MASS_H = Higgs.mass
MASS_PI = ChargedPion.mass
MASS_PI0 = NeutralPion.mass

WIDTH_W = WBoson.width
WIDTH_Z = ZBoson.width
WIDTH_H = Higgs.width


lepton_masses = [Electron.mass, Muon.mass, Tau.mass]
up_quark_masses = [fields.UpQuark.mass, fields.CharmQuark.mass, fields.TopQuark.mass]
down_quark_masses = [
    fields.DownQuark.mass,
    fields.StrangeQuark.mass,
    fields.BottomQuark.mass,
]


def dirac_spinor(
    spinor_type: str, momentum: npt.NDArray, mass: float
) -> tuple[DiracWf, DiracWf]:
    """Generate spinor wavefunctions for all spins.

    Parameters
    ----------
    spinor_type: str
        Type of spinor to generate ("u", "v", "ubar" or "vbar".)
    momenta: array
        Momentum of the field.
    mass: float
        Mass of the field.

    Returns
    -------
    wavefunctions: tuple[DiracWf, DiracWf]
        Spinor wavefunctions with spin down and up.
    """
    if spinor_type == "u":
        spinor_fn = wavefunctions.spinor_u
    elif spinor_type == "ubar":
        spinor_fn = wavefunctions.spinor_ubar
    elif spinor_type == "v":
        spinor_fn = wavefunctions.spinor_v
    elif spinor_type == "vbar":
        spinor_fn = wavefunctions.spinor_vbar
    else:
        raise ValueError(f"Invalid spinor type {type}")

    return (spinor_fn(momentum, mass, -1), spinor_fn(momentum, mass, 1))


def charge_conjugate_spinors(psi: Sequence[DiracWf]) -> List[DiracWf]:
    """Charge conjugate a sequence of spinors."""
    return [wavefunctions.charge_conjugate(wf) for wf in psi]


def vector_current(
    vector: VectorBoson, vertex: VertexFFV, psi_out: DiracWf, psi_in: DiracWf
):
    """Compute the w-current from a pair of spinors.

    Parameters
    ----------
    vector: VectorBoson
        Vector boson to generate current of.
    vertex: VertexFFV
        The V-f-f vertex.
    psi_out: DiracWf
        Flow-out dirac wavefunction.
    psi_in: DiracWf
        Flow-in dirac wavefunction.

    Returns
    -------
    current: VectorWf
        Vector-boson wavefunction generated from the two fermions.
    """
    return amplitudes.current_ff_to_v(
        vertex, vector.mass, vector.width, psi_out, psi_in
    )


def scalar_current(
    scalar: ScalarBoson, vertex: VertexFFS, psi_out: DiracWf, psi_in: DiracWf
):
    """Compute the w-current from a pair of spinors.

    Parameters
    ----------
    scalar: ScalarBoson
        Scalar boson to generate current of.
    vertex: VertexFFV
        The S-f-f vertex.
    psi_out: DiracWf
        Flow-out dirac wavefunction.
    psi_in: DiracWf
        Flow-in dirac wavefunction.

    Returns
    -------
    current: ScalarWf
        Scalar-boson wavefunction generated from the two fermions.
    """
    return amplitudes.current_ff_to_s(
        vertex, scalar.mass, scalar.width, psi_out, psi_in
    )


def amplitude(vertex: Vertex, wavefuncs: Sequence[Wavefunction]):
    """Compute the amplitude given a vertex and wavefunctions.

    Parameters
    ----------
    vertex: Vertex
        The vertex joining all wavefunctions.
    wavefuncs: Sequence[Wavefunction]
        The wavefunctions to join.

    Returns
    -------
    amplitude: complex array
        The value(s) of the vertex.
    """

    if isinstance(vertex, VertexFFV):
        assert len(wavefuncs) == 3, f"Expected 3 wavefunctions for a {type(vertex)}."
        psi_out, psi_in, eps = wavefuncs

        assert isinstance(psi_out, DiracWf), "First wavefunction must be a DiracWf."
        assert isinstance(psi_in, DiracWf), "Second wavefunction must be a DiracWf."
        assert isinstance(eps, VectorWf), "Third wavefunction must be a Vector."

        assert psi_out.direction == -1, "First wavefunction must be a flow-out DiracWf"
        assert psi_in.direction == 1, "Second wavefunction must be a flow-in DiracWf"

        return amplitudes.amplitude_ffv(vertex, psi_out, psi_in, eps)

    if isinstance(vertex, VertexFFS):
        assert len(wavefuncs) == 3, f"Expected 3 wavefunctions for a {type(vertex)}."
        psi_out, psi_in, scalar = wavefuncs

        assert isinstance(psi_out, DiracWf), "First wavefunction must be a DiracWf."
        assert isinstance(psi_in, DiracWf), "Second wavefunction must be a DiracWf."
        assert isinstance(scalar, ScalarWf), "Third wavefunction must be a ScalarWf."

        assert psi_out.direction == -1, "First wavefunction must be a flow-out DiracWf"
        assert psi_in.direction == 1, "Second wavefunction must be a flow-in DiracWf"

        return amplitudes.amplitude_ffs(vertex, psi_out, psi_in, scalar)

    raise ValueError(f"Unexpected vertex type {type(vertex)}")


def msqrd_n_to_v_l_l(self: RhNeutrinoBase, momenta, *, genv, genl1, genl2):
    """Compute the squared matrix element N -> nu + l + lbar."""
    mass = self.mass
    theta = self.theta
    genn = self.gen

    pv = momenta[:, 0]
    pl1 = momenta[:, 1]
    pl2 = momenta[:, 2]
    pn = np.sum(momenta, axis=1)

    n_u = dirac_spinor("u", pn, mass)
    n_vbar = dirac_spinor("vbar", pn, mass)

    v_ubar = dirac_spinor("ubar", pv, 0.0)
    v_v = dirac_spinor("v", pv, 0.0)

    l1_ubar = dirac_spinor("ubar", pl1, lepton_masses[genl1])
    l2_v = dirac_spinor("v", pl2, lepton_masses[genl2])

    v_zll = feynman_rules.VERTEX_ZLL
    v_znv = feynman_rules.vertex_znv(theta, genn=genn, genv=genv)

    v_wnl1 = feynman_rules.vertex_wnl(theta, genn=genn, genl=genl1)
    v_wnl2 = feynman_rules.vertex_wnl(theta, genn=genn, genl=genl2)

    v_wvl1 = feynman_rules.vertex_wvl(
        theta, genn=genn, genv=genv, genl=genl1, l_in=True
    )
    v_wvl2 = feynman_rules.vertex_wvl(
        theta, genn=genn, genv=genv, genl=genl2, l_in=False
    )

    def diagram1(i_n, i_v, i_l1, i_l2):
        n_wf = n_u[i_n]
        v_wf = v_ubar[i_v]
        l1_wf = l1_ubar[i_l1]
        l2_wf = l2_v[i_l2]

        # (l1, l2), (v, n)
        return amplitude(
            v_zll, (l1_wf, l2_wf, vector_current(ZBoson, v_znv, v_wf, n_wf))
        )

    def diagram2(i_n, i_v, i_l1, i_l2):
        n_wf = n_u[i_n]
        v_wf = v_ubar[i_v]
        l1_wf = l1_ubar[i_l1]
        l2_wf = l2_v[i_l2]

        # (v, l2), (l1, n)
        return -amplitude(
            v_wvl2, (v_wf, l2_wf, vector_current(WBoson, v_wnl1, l1_wf, n_wf))
        )

    def diagram3(i_n, i_v, i_l1, i_l2):
        n_wf = n_vbar[i_n]
        v_wf = v_v[i_v]
        l1_wf = l1_ubar[i_l1]
        l2_wf = l2_v[i_l2]

        # (l1, v), (n, l2)
        return amplitude(
            v_wvl1, (l1_wf, v_wf, vector_current(WBoson, v_wnl2, n_wf, l2_wf))
        )

    def _msqrd(i_n, i_v, i_l1, i_l2):
        d1 = diagram1(i_n, i_v, i_l1, i_l2)
        d2 = diagram2(i_n, i_v, i_l1, i_l2)
        d3 = diagram3(i_n, i_v, i_l1, i_l2)
        return np.square(np.abs(d1 + d2 + d3))

    idxs = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], [0, 1])).T.reshape(-1, 4)
    res = sum(_msqrd(i_n, i_v, i_l1, i_l2) for (i_n, i_v, i_l1, i_l2) in idxs)

    return res / 2.0


def msqrd_n_to_v_v_v(self: RhNeutrinoBase, momenta, *, genv1, genv2, genv3):
    mass = self.mass
    genn = self.gen
    theta = self.theta

    pv1 = momenta[:, 0]
    pv2 = momenta[:, 1]
    pv3 = momenta[:, 2]
    pn = np.sum(momenta, axis=1)

    wf_i = dirac_spinor("u", pn, mass)
    wf_j_1 = dirac_spinor("ubar", pv1, 0.0)
    wf_k_1 = dirac_spinor("ubar", pv2, 0.0)
    wf_l_1 = dirac_spinor("v", pv3, 0.0)
    vji_1 = feynman_rules.vertex_znv(theta, genn=genn, genv=genv1)
    vkl_1 = feynman_rules.vertex_zvv(theta, genn=genn, genv1=genv2, genv2=genv3)

    wf_j_2 = dirac_spinor("ubar", pv2, 0.0)
    wf_k_2 = dirac_spinor("ubar", pv1, 0.0)
    wf_l_2 = dirac_spinor("v", pv3, 0.0)
    vji_2 = feynman_rules.vertex_znv(theta, genn=genn, genv=genv2)
    vkl_2 = feynman_rules.vertex_zvv(theta, genn=genn, genv1=genv1, genv2=genv3)

    wf_j_3 = dirac_spinor("ubar", pv3, 0.0)
    wf_k_3 = dirac_spinor("ubar", pv2, 0.0)
    wf_l_3 = dirac_spinor("v", pv1, 0.0)
    vji_3 = feynman_rules.vertex_znv(theta, genn=genn, genv=genv3)
    vkl_3 = feynman_rules.vertex_zvv(theta, genn=genn, genv1=genv2, genv2=genv1)

    def _amplitude_template(wfi, wfj, wfk, wfl, vji, vkl):
        wfz = vector_current(ZBoson, vji, wfj, wfi)
        return amplitude(vkl, [wfk, wfl, wfz])

    def _amplitude(ii, ij, ik, il):
        wfi = wf_i[ii]
        return (
            +_amplitude_template(wfi, wf_j_1[ij], wf_k_1[ik], wf_l_1[il], vji_1, vkl_1)
            - _amplitude_template(wfi, wf_j_2[ik], wf_k_2[ij], wf_l_2[il], vji_2, vkl_2)
            - _amplitude_template(wfi, wf_j_3[il], wf_k_3[ik], wf_l_3[ij], vji_3, vkl_3)
        )

    def _msqrd(ii, ij, ik, il):
        return np.square(np.abs(_amplitude(ii, ij, ik, il)))

    idxs = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], [0, 1])).T.reshape(-1, 4)
    res = sum(_msqrd(i_n, i_v1, i_v2, i_v3) for (i_n, i_v1, i_v2, i_v3) in idxs)

    return res / 2.0


def msqrd_n_to_v_u_u(self: RhNeutrinoBase, momenta, *, genu):
    genn = int(self.gen)

    pv = momenta[:, 0]
    pu1 = momenta[:, 1]
    pu2 = momenta[:, 2]
    pn = pv + pu1 + pu2

    mu = up_quark_masses[genu]

    n_wfs = dirac_spinor("u", pn, self.mass)
    v_wfs = dirac_spinor("ubar", pv, 0.0)
    u1_wfs = dirac_spinor("ubar", pu1, mu)
    u2_wfs = dirac_spinor("v", pu2, mu)

    v_znv = feynman_rules.vertex_znv(self.theta, genn=genn, genv=genn)
    v_zuu = feynman_rules.VERTEX_ZUU

    v_hnv = feynman_rules.vertex_hnv(self.theta, self.mass, genn=genn, genv=genn)
    v_huu = feynman_rules.VERTEX_HUU[genu]

    def _msqrd(i_n, i_v1, i_u1, i_u2):
        wfn = n_wfs[i_n]
        wfv = v_wfs[i_v1]
        wfu1 = u1_wfs[i_u1]
        wfu2 = u2_wfs[i_u2]

        wf_z = vector_current(ZBoson, v_znv, wfv, wfn)
        wf_h = scalar_current(Higgs, v_hnv, wfv, wfn)

        return np.square(
            np.abs(
                amplitude(v_zuu, (wfu1, wfu2, wf_z))
                + amplitude(v_huu, (wfu1, wfu2, wf_h))
            )
        )

    idxs = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], [0, 1])).T.reshape(-1, 4)
    res = sum(_msqrd(i_n, i_v1, i_v2, i_v3) for (i_n, i_v1, i_v2, i_v3) in idxs)

    return 3.0 * res / 2.0


def msqrd_n_to_v_d_d(self: RhNeutrinoBase, momenta, *, gend):
    genn = int(self.gen)

    pv = momenta[:, 0]
    pd1 = momenta[:, 1]
    pd2 = momenta[:, 2]
    pn = pv + pd1 + pd2

    md = down_quark_masses[gend]

    n_wfs = dirac_spinor("u", pn, self.mass)
    v_wfs = dirac_spinor("ubar", pv, 0.0)
    d1_wfs = dirac_spinor("ubar", pd1, md)
    d2_wfs = dirac_spinor("v", pd2, md)

    v_znv = feynman_rules.vertex_znv(self.theta, genn=genn, genv=genn)
    v_zdd = feynman_rules.VERTEX_ZDD

    v_hnv = feynman_rules.vertex_hnv(self.theta, self.mass, genn=genn, genv=genn)
    v_hdd = feynman_rules.VERTEX_HDD[gend]

    def _msqrd(i_n, i_v1, i_d1, i_d2):
        wfn = n_wfs[i_n]
        wfv = v_wfs[i_v1]
        wfd1 = d1_wfs[i_d1]
        wfd2 = d2_wfs[i_d2]

        wf_z = vector_current(ZBoson, v_znv, wfv, wfn)
        wf_h = scalar_current(Higgs, v_hnv, wfv, wfn)

        return np.square(
            np.abs(
                amplitude(v_zdd, (wfd1, wfd2, wf_z))
                + amplitude(v_hdd, (wfd1, wfd2, wf_h))
            )
        )

    idxs = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], [0, 1])).T.reshape(-1, 4)
    res = sum(_msqrd(i_n, i_v1, i_v2, i_v3) for (i_n, i_v1, i_v2, i_v3) in idxs)

    return 3.0 * res / 2.0


def msqrd_n_to_l_u_d(self: RhNeutrinoBase, momenta, *, genu, gend):
    mass = self.mass
    theta = self.theta
    genn = self.gen

    pl = momenta[:, 0]
    pu = momenta[:, 1]
    pd = momenta[:, 2]
    pn = np.sum(momenta, axis=1)

    n_wfs = dirac_spinor("u", pn, mass)
    l_wfs = dirac_spinor("ubar", pl, lepton_masses[genn])
    u_wfs = dirac_spinor("ubar", pu, up_quark_masses[genu])
    d_wfs = dirac_spinor("v", pd, down_quark_masses[gend])

    v_wnl = feynman_rules.vertex_wnl(theta, genn=genn, genl=genn, l_in=False)
    v_wud = feynman_rules.vertex_wud(genu=genu, gend=gend, u_in=False)

    def _msqrd(i_n, i_l, i_u, i_d):
        wfn = n_wfs[i_n]
        wfl = l_wfs[i_l]
        wfu = u_wfs[i_u]
        wfd = d_wfs[i_d]

        wf_z = vector_current(WBoson, v_wnl, wfl, wfn)
        return np.square(np.abs(amplitude(v_wud, (wfu, wfd, wf_z))))

    idxs = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], [0, 1])).T.reshape(-1, 4)
    res = sum(_msqrd(i_n, i_l, i_u, i_d) for (i_n, i_l, i_u, i_d) in idxs)
    return 3.0 * res / 2.0


def msqrd_n_to_l_pi_pi0(self: RhNeutrinoBase, momenta):
    mn = self.mass
    genn = self.gen
    theta = self.theta

    mup = MASS_PI / mn
    mu0 = MASS_PI0 / mn
    mul = lepton_masses[genn] / mn

    x = 2.0 * momenta[0, 0] / mn
    y = 2.0 * momenta[0, 1] / mn

    kr = feynman_rules.pmns_right(theta=theta, genn=genn, genv=genn)
    pre = 8 * GF**2 * mn**4 * np.abs(kr) ** 2 * np.abs(CKM_UD) ** 2

    ex = (
        1
        - mu0**2
        + mup**2
        + y * (-2 + mu0**2 - mup**2 + y)
        + x * (-0.75 + mul**2 / 4.0 - mup**2 + y)
    )

    return pre * ex


def msqrd_n_to_v_pi_pi(self: RhNeutrinoBase, momenta):
    mn = self.mass
    theta = self.theta
    genn = self.gen

    mup = MASS_PI / mn
    x = 2.0 * momenta[0, 0] / mn
    y = 2.0 * momenta[0, 1] / mn

    kl = feynman_rules.pmns_left(theta, genn, genn)
    kr = feynman_rules.pmns_right(theta, genn, genn)
    kk = np.abs(np.conj(kl) * kr) ** 2

    pre = 8.0 * GF**2 * mn**4 * (1 - 2 * SW**2) ** 2 * kk / CW**4
    expr = 1.0 + y * (-2.0 + y) + x * (-0.75 + y - mup**2)

    return pre * expr
