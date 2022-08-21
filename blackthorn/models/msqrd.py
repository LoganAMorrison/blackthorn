"""

"""
# TODO: l + pi + pi0 seems wrong.

import functools
from typing import Tuple

# import jax
# import jax.numpy as jnp
import numpy as np
import numpy as jnp
from helax.numpy import amplitudes, wavefunctions

from .base import RhNeutrinoBase
from .. import fields
from ..constants import CKM_UD, CW, GF, SW
from ..fields import (
    ChargedPion,
    Electron,
    Higgs,
    Muon,
    NeutralPion,
    Tau,
    WBoson,
    ZBoson,
)
from . import feynman_rules

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


# @functools.partial(jax.jit, static_argnums=(1,))
def spinor_u(momentum, mass):
    return (
        wavefunctions.spinor_u(momentum, mass, -1),
        wavefunctions.spinor_u(momentum, mass, 1),
    )


# @functools.partial(jax.jit, static_argnums=(1,))
def spinor_v(momentum, mass):
    return (
        wavefunctions.spinor_v(momentum, mass, -1),
        wavefunctions.spinor_v(momentum, mass, 1),
    )


# @functools.partial(jax.jit, static_argnums=(1,))
def spinor_ubar(momentum, mass):
    return (
        wavefunctions.spinor_ubar(momentum, mass, -1),
        wavefunctions.spinor_ubar(momentum, mass, 1),
    )


# @functools.partial(jax.jit, static_argnums=(1,))
def spinor_vbar(momentum, mass):
    return (
        wavefunctions.spinor_vbar(momentum, mass, -1),
        wavefunctions.spinor_vbar(momentum, mass, 1),
    )


# @jax.jit
def charge_conjugate(psi: wavefunctions.DiracWf):
    return wavefunctions.charge_conjugate(psi)


# @jax.jit
def charge_conjugate_spinors(psi: Tuple[wavefunctions.DiracWf, wavefunctions.DiracWf]):
    return (
        wavefunctions.charge_conjugate(psi[0]),
        wavefunctions.charge_conjugate(psi[1]),
    )


# @functools.partial(jax.jit, static_argnums=(0,))
def current_w(vertex, psi_out, psi_in):
    return amplitudes.current_ff_to_v(vertex, MASS_W, WIDTH_W, psi_out, psi_in)


# @functools.partial(jax.jit, static_argnums=(0,))
def current_z(vertex, psi_out, psi_in):
    return amplitudes.current_ff_to_v(vertex, MASS_Z, WIDTH_Z, psi_out, psi_in)


# @functools.partial(jax.jit, static_argnums=(0,))
def current_h(vertex, psi_out, psi_in):
    return amplitudes.current_ff_to_s(vertex, MASS_H, WIDTH_H, psi_out, psi_in)


# @functools.partial(jax.jit, static_argnums=(0,))
def amplitude_ffv(vertex, psi_out, psi_in, eps):
    return amplitudes.amplitude_ffv(vertex, psi_out, psi_in, eps)


# @functools.partial(jax.jit, static_argnums=(0,))
def amplitude_ffs(vertex, psi_out, psi_in, phi):
    return amplitudes.amplitude_ffs(vertex, psi_out, psi_in, phi)


def msqrd_n_to_v_l_l(self: RhNeutrinoBase, momenta, *, genv, genl1, genl2):
    mass = self.mass
    theta = self.theta
    genn = self.gen

    pv = momenta[:, 0]
    pl1 = momenta[:, 1]
    pl2 = momenta[:, 2]
    pn = jnp.sum(momenta, axis=1)

    n_u = spinor_u(pn, mass)
    n_vbar = spinor_vbar(pn, mass)  # charge_conjugate_spinors(n_u)

    v_ubar = spinor_ubar(pv, 0.0)
    v_v = spinor_v(pv, 0.0)  # charge_conjugate_spinors(v_ubar)

    l1_u = spinor_ubar(pl1, lepton_masses[genl1])
    l2_v = spinor_v(pl2, lepton_masses[genl2])

    v_zll = feynman_rules.VERTEX_ZLL
    v_znv = feynman_rules.vertex_znv(theta, genn=genn, genv=genv)

    v_wnl1 = feynman_rules.vertex_wnl(theta, genn=genn, genl=genl1, l_in=False)
    v_wnl2 = feynman_rules.vertex_wnl(theta, genn=genn, genl=genl2, l_in=True)

    v_wvl1 = feynman_rules.vertex_wvl(
        theta, genn=genn, genv=genv, genl=genl1, l_in=False
    )
    v_wvl2 = feynman_rules.vertex_wvl(
        theta, genn=genn, genv=genv, genl=genl2, l_in=True
    )

    def diagram1(i_n, i_v, i_l1, i_l2):
        n_wf = n_u[i_n]
        v_wf = v_ubar[i_v]
        l1_wf = l1_u[i_l1]
        l2_wf = l2_v[i_l2]

        z_wf = current_z(v_znv, v_wf, n_wf)
        return amplitude_ffv(v_zll, l1_wf, l2_wf, z_wf)

    def diagram2(i_n, i_v, i_l1, i_l2):
        n_wf = n_u[i_n]
        v_wf = v_ubar[i_v]
        l1_wf = l1_u[i_l1]
        l2_wf = l2_v[i_l2]

        w_wf = current_w(v_wnl1, l1_wf, n_wf)
        return -amplitude_ffv(v_wvl2, v_wf, l2_wf, w_wf)

    def diagram3(i_n, i_v, i_l1, i_l2):
        n_wf = n_vbar[i_n]
        v_wf = v_v[i_v]
        l1_wf = l1_u[i_l1]
        l2_wf = l2_v[i_l2]

        w_wf = current_w(v_wnl2, n_wf, l2_wf)
        return amplitude_ffv(v_wvl1, l1_wf, v_wf, w_wf)

    def _msqrd(i_n, i_v, i_l1, i_l2):
        return jnp.square(
            jnp.abs(
                diagram1(i_n, i_v, i_l1, i_l2)
                + diagram2(i_n, i_v, i_l1, i_l2)
                + diagram3(i_n, i_v, i_l1, i_l2)
            )
        )

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
    pn = jnp.sum(momenta, axis=1)

    n_wfs = spinor_u(pn, mass)
    n_wfs_r = (charge_conjugate(n_wfs[0]), charge_conjugate(n_wfs[1]))
    n_wfs_r = spinor_vbar(pn, mass)

    v1_wfs = spinor_ubar(pv1, 0.0)
    v1_wfs_r = (charge_conjugate(v1_wfs[0]), charge_conjugate(v1_wfs[1]))
    v1_wfs_r = spinor_v(pv1, 0.0)

    v2_wfs = spinor_u(pv2, 0.0)
    v3_wfs = spinor_v(pv3, 0.0)

    v_znv1 = feynman_rules.vertex_znv(theta, genn=genn, genv=genv1)
    v_znv2 = feynman_rules.vertex_znv(theta, genn=genn, genv=genv2)
    v_znv3 = feynman_rules.vertex_znv(theta, genn=genn, genv=genv3)

    v_zvv1 = feynman_rules.vertex_zvv(theta, genn=genn, genv1=genv2, genv2=genv3)
    v_zvv2 = feynman_rules.vertex_zvv(theta, genn=genn, genv1=genv1, genv2=genv3)
    v_zvv3 = feynman_rules.vertex_zvv(theta, genn=genn, genv1=genv1, genv2=genv2)

    v_hnv1 = feynman_rules.vertex_hnv(theta, mass, genn=genn, genv=genv1)
    v_hnv2 = feynman_rules.vertex_hnv(theta, mass, genn=genn, genv=genv2)
    v_hnv3 = feynman_rules.vertex_hnv(theta, mass, genn=genn, genv=genv3)

    v_hvv1 = feynman_rules.vertex_hvv(theta, mass, genn=genn, genv1=genv2, genv2=genv3)
    v_hvv2 = feynman_rules.vertex_hvv(theta, mass, genn=genn, genv1=genv1, genv2=genv3)
    v_hvv3 = feynman_rules.vertex_hvv(theta, mass, genn=genn, genv1=genv1, genv2=genv2)

    def _diagram_1(i_n, i_v1, i_v2, i_v3):
        v_znv = v_znv1
        v_zvv = v_zvv1
        v_hnv = v_hnv1
        v_hvv = v_hvv1

        wfn = n_wfs[i_n]
        wf1 = v1_wfs[i_v1]
        wf2 = v2_wfs[i_v2]
        wf3 = v3_wfs[i_v3]

        wf_z = current_z(v_znv, wf1, wfn)
        wf_h = current_h(v_hnv, wf1, wfn)

        return amplitude_ffv(v_zvv, wf2, wf3, wf_z) + amplitude_ffs(
            v_hvv, wf2, wf3, wf_h
        )

    def _diagram_2(i_n, i_v1, i_v2, i_v3):
        v_znv = v_znv2
        v_zvv = v_zvv2
        v_hnv = v_hnv2
        v_hvv = v_hvv2

        wfn = n_wfs[i_n]
        wf1 = v1_wfs[i_v1]
        wf2 = v2_wfs[i_v2]
        wf3 = v3_wfs[i_v3]

        wf_z = current_z(v_znv, wf2, wfn)
        wf_h = current_h(v_hnv, wf2, wfn)
        amp = amplitude_ffv(v_zvv, wf1, wf3, wf_z) + amplitude_ffs(
            v_hvv, wf1, wf3, wf_h
        )
        return -amp

    def _diagram_3(i_n, i_v1, i_v2, i_v3):
        v_znv = v_znv3
        v_zvv = v_zvv3
        v_hnv = v_hnv3
        v_hvv = v_hvv3

        wfn = n_wfs_r[i_n]
        wf1 = v1_wfs_r[i_v1]
        wf2 = v2_wfs[i_v2]
        wf3 = v3_wfs[i_v3]

        wf_z = current_z(v_znv, wf3, wfn)
        wf_h = current_h(v_hnv, wf3, wfn)
        return amplitude_ffv(v_zvv, wf2, wf1, wf_z) + amplitude_ffs(
            v_hvv, wf2, wf1, wf_h
        )

    def _msqrd(i_n, i_v1, i_v2, i_v3):
        return jnp.square(
            jnp.abs(
                _diagram_1(i_n, i_v1, i_v2, i_v3)
                + _diagram_2(i_n, i_v1, i_v2, i_v3)
                + _diagram_3(i_n, i_v1, i_v2, i_v3)
            )
        )

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

    n_wfs = spinor_u(pn, self.mass)
    v_wfs = spinor_ubar(pv, 0.0)
    u1_wfs = spinor_ubar(pu1, mu)
    u2_wfs = spinor_v(pu2, mu)

    v_znv = feynman_rules.vertex_znv(self.theta, genn=genn, genv=genn)
    v_zuu = feynman_rules.VERTEX_ZUU

    v_hnv = feynman_rules.vertex_hnv(self.theta, self.mass, genn=genn, genv=genn)
    v_huu = feynman_rules.VERTEX_HUU[genu]

    def _msqrd(i_n, i_v1, i_u1, i_u2):
        wfn = n_wfs[i_n]
        wfv = v_wfs[i_v1]
        wfu1 = u1_wfs[i_u1]
        wfu2 = u2_wfs[i_u2]

        wf_z = current_z(v_znv, wfv, wfn)
        wf_h = current_h(v_hnv, wfv, wfn)

        return jnp.square(
            jnp.abs(
                amplitude_ffv(v_zuu, wfu1, wfu2, wf_z)
                + amplitude_ffs(v_huu, wfu1, wfu2, wf_h)
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

    n_wfs = spinor_u(pn, self.mass)
    v_wfs = spinor_ubar(pv, 0.0)
    d1_wfs = spinor_ubar(pd1, md)
    d2_wfs = spinor_v(pd2, md)

    v_znv = feynman_rules.vertex_znv(self.theta, genn=genn, genv=genn)
    v_zdd = feynman_rules.VERTEX_ZDD

    v_hnv = feynman_rules.vertex_hnv(self.theta, self.mass, genn=genn, genv=genn)
    v_hdd = feynman_rules.VERTEX_HDD[gend]

    def _msqrd(i_n, i_v1, i_d1, i_d2):
        wfn = n_wfs[i_n]
        wfv = v_wfs[i_v1]
        wfd1 = d1_wfs[i_d1]
        wfd2 = d2_wfs[i_d2]

        wf_z = current_z(v_znv, wfv, wfn)
        wf_h = current_h(v_hnv, wfv, wfn)

        return jnp.square(
            jnp.abs(
                amplitude_ffv(v_zdd, wfd1, wfd2, wf_z)
                + amplitude_ffs(v_hdd, wfd1, wfd2, wf_h)
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
    pn = jnp.sum(momenta, axis=1)

    n_wfs = spinor_u(pn, mass)
    l_wfs = spinor_ubar(pl, lepton_masses[genn])
    u_wfs = spinor_ubar(pu, up_quark_masses[genu])
    d_wfs = spinor_v(pd, down_quark_masses[gend])

    v_wnl = feynman_rules.vertex_wnl(theta, genn=genn, genl=genn, l_in=False)
    v_wud = feynman_rules.vertex_wud(genu=genu, gend=gend, u_in=False)

    def _msqrd(i_n, i_l, i_u, i_d):
        wfn = n_wfs[i_n]
        wfl = l_wfs[i_l]
        wfu = u_wfs[i_u]
        wfd = d_wfs[i_d]

        wf_z = current_w(v_wnl, wfl, wfn)
        return jnp.square(jnp.abs(amplitude_ffv(v_wud, wfu, wfd, wf_z)))

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
