from pytest import approx
import numpy as np
from scipy import integrate

from blackthorn import fields
from blackthorn.fields import QuantumField
from blackthorn.constants import Gen, SW, GF, CKM
from blackthorn.models import RhNeutrinoMeV
from blackthorn.models.base import RhNeutrinoBase

charged_leptons = {
    Gen.Fst: fields.Electron,
    Gen.Snd: fields.Muon,
    Gen.Trd: fields.Tau,
}

MODEL = RhNeutrinoMeV(0.2, 1e-5, Gen.Fst)


def kallen_lambda(a, b, c):
    return a**2 + b**2 + c**2 - 2 * a * b - 2 * a * c - 2 * b * c


def width_n_to_v_f_f(model: RhNeutrinoBase, nu: fields.Neutrino, f: QuantumField):
    mn = model.mass
    theta = model.theta
    genn = model.gen
    x = f.mass / mn
    uu = 0.5 * np.tan(2 * theta)

    if mn < 2 * f.mass:
        return 0.0

    if isinstance(f, fields.UpTypeQuark):
        if not genn == nu.gen:
            return 0.0

        nz = 3.0
        c1 = 1 / 4 * (1 - 8 / 3 * SW**2 + 32 / 9 * SW**4)
        c2 = 1 / 3 * SW**2 * (4 / 3 * SW**2 - 1)

    elif isinstance(f, fields.DownTypeQuark):
        if not genn == nu.gen:
            return 0.0

        nz = 3.0
        c1 = 1 / 4 * (1 - 4 / 3 * SW**2 + 8 / 9 * SW**4)
        c2 = 1 / 6 * SW**2 * (2 / 3 * SW**2 - 1)

    elif isinstance(f, fields.ChargedLepton):
        if model.gen != nu.gen:
            return 0.0

        nz = 1.0
        if nu.gen != f.gen:
            c1 = 1 / 4 * (1 - 4 * SW**2 + 8 * SW**4)
            c2 = 1 / 2 * SW**2 * (2 * SW**2 - 1)
        elif nu.gen == nu.gen:
            c1 = 1 / 4 * (1 + 4 * SW**2 + 8 * SW**4)
            c2 = 1 / 2 * SW**2 * (2 * SW**2 + 1)
        else:
            raise ValueError()

    else:
        raise ValueError()

    sx = np.sqrt(1.0 - 4 * x**2)
    num = 1.0 - 3.0 * x**2 - (1.0 - x**2) * sx
    if num < 1e-10:
        lx = 4 * np.log(x) + 4 * x**2 + 6 * x**4
    else:
        lx = np.log(num / (x**2 * (1.0 + sx)))

    return (
        nz
        * GF**2
        * mn**5
        / (192 * np.pi**3)
        * uu**2
        * (
            c1
            * (
                (1 - 14 * x**2 - 2 * x**4 - 12 * x**6) * sx
                + 12 * x**4 * (x**4 - 1) * lx
            )
            + 4
            * c2
            * (
                x**2 * (2 + 10 * x**2 - 12 * x**4) * sx
                + 6 * x**4 * (1 - 2 * x**2 + 2 * x**4) * lx
            )
        )
    )


def width_n_to_vvv(model: RhNeutrinoBase, genv1: Gen, genv2: Gen):
    mn = model.mass
    theta = model.theta
    uu = 0.5 * np.tan(2 * theta)

    if genv1 == genv2:
        p = 2.0
    else:
        p = 1.0

    return p * GF**2 * mn**5 * uu**2 / (768 * np.pi**3)


def width_n_to_lud(model: RhNeutrinoBase, genl: Gen, u: QuantumField, d: QuantumField):
    mn = model.mass
    theta = model.theta
    uu = 0.5 * np.tan(2 * theta)

    if isinstance(u, fields.Neutrino):
        assert isinstance(d, fields.ChargedLepton)
        assert d.gen != genl
        nw = 1.0
    elif isinstance(u, fields.UpTypeQuark):
        assert isinstance(d, fields.DownTypeQuark)
        nw = 3 * np.abs(CKM[u.gen, d.gen]) ** 2
    else:
        raise ValueError("Invalid u")

    ml = charged_leptons[genl].mass
    mu = u.mass
    md = d.mass

    if mn < ml + mu + md:
        return 0.0

    xl = ml / mn
    xu = mu / mn
    xd = md / mn

    def integrand(x):
        return (
            1.0
            / x
            * (x - xl**2 - xd**2)
            * (1 + xu**2 - x)
            * np.sqrt(kallen_lambda(x, xl**2, xd**2) * kallen_lambda(1, x, xu**2))
        )

    lb = (xd + xl) ** 2
    ub = (1 - xu) ** 2
    integral = 12 * integrate.quad(integrand, lb, ub)[0]

    return nw * GF**2 * mn**5 * uu**2 * integral / (192 * np.pi**3)


def width_v_l_l(
    model: RhNeutrinoBase,
    nu: fields.Neutrino,
    l1: fields.ChargedLepton,
    l2: fields.ChargedLepton,
):
    gn = model.gen
    gv = nu.gen
    g1 = l1.gen
    g2 = l2.gen

    abab = gn == g1 and gv == g2 and gn != gv
    abba = gn == g2 and gv == g1 and gn != gv
    aabb = gn == gv and g1 == g2 and gn != g1
    aaaa = gn == gv and g1 == g2 and gn == g1

    if abab:
        return width_n_to_lud(model, g1, nu, l2)
    elif abba:
        return width_n_to_lud(model, g2, nu, l1)
    elif aabb:
        return width_n_to_v_f_f(model, nu, l1)
    elif aaaa:
        raise ValueError("Invalid combo")
    else:
        raise ValueError("Invalid combo")


def test_width_n_to_vll_aabb():
    genn = Gen.Fst
    nu = fields.ElectronNeutrino
    l1 = fields.Muon
    l2 = fields.Muon
    percent = 10.0

    model = RhNeutrinoMeV(0.3, 1e-5, genn)

    w_test = width_v_l_l(model, nu, l1, l2)
    w_helax = model.width_v_l_l(genv=nu.gen, genl1=l1.gen, genl2=l2.gen)
    frac_diff = 100 * (w_helax[0] - w_test) / w_test

    assert abs(frac_diff) <= percent


def test_width_n_to_vll_abab():
    genn = Gen.Fst
    nu = fields.MuonNeutrino
    l1 = fields.Electron
    l2 = fields.Muon
    percent = 10.0

    model = RhNeutrinoMeV(0.3, 1e-5, genn)

    w_test = width_v_l_l(model, nu, l1, l2)
    w_helax = model.width_v_l_l(genv=nu.gen, genl1=l1.gen, genl2=l2.gen)
    frac_diff = 100 * (w_helax[0] - w_test) / w_test

    assert abs(frac_diff) <= percent


def test_width_n_to_vll_abba():
    genn = Gen.Fst
    nu = fields.MuonNeutrino
    l1 = fields.Muon
    l2 = fields.Electron
    percent = 10.0

    model = RhNeutrinoMeV(0.3, 1e-5, genn)

    w_test = width_v_l_l(model, nu, l1, l2)
    w_helax = model.width_v_l_l(genv=nu.gen, genl1=l1.gen, genl2=l2.gen)
    frac_diff = 100 * (w_helax[0] - w_test) / w_test

    assert abs(frac_diff) <= percent


def test_width_n_to_vll_aaaa():
    genn = Gen.Fst
    nu = fields.MuonNeutrino
    l1 = fields.Muon
    l2 = fields.Electron
    percent = 10.0

    model = RhNeutrinoMeV(0.3, 1e-5, genn)

    w_test = width_v_l_l(model, nu, l1, l2)
    w_helax = model.width_v_l_l(genv=nu.gen, genl1=l1.gen, genl2=l2.gen)
    frac_diff = 100 * (w_helax[0] - w_test) / w_test

    assert abs(frac_diff) <= percent


def test_width_n_to_vvv_aabb():
    genn = Gen.Fst
    v1 = fields.ElectronNeutrino
    v2 = fields.MuonNeutrino
    v3 = fields.MuonNeutrino
    percent = 10.0

    model = RhNeutrinoMeV(0.3, 1e-5, genn)

    w_test = width_n_to_vvv(model, v1.gen, v2.gen)
    w_helax = model.width_v_v_v(genv1=v1.gen, genv2=v2.gen, genv3=v3.gen)
    frac_diff = 100 * (w_helax[0] - w_test) / w_test

    assert abs(frac_diff) <= percent


def test_width_n_to_vvv_aaaa():
    genn = Gen.Fst
    v1 = fields.ElectronNeutrino
    v2 = fields.ElectronNeutrino
    v3 = fields.ElectronNeutrino
    percent = 10.0

    model = RhNeutrinoMeV(0.1, 1e-5, genn)

    w_test = width_n_to_vvv(model, v1.gen, v2.gen)
    w_helax = model.width_v_v_v(genv1=v1.gen, genv2=v2.gen, genv3=v3.gen)
    frac_diff = 100 * (w_helax[0] - w_test) / w_test

    assert abs(frac_diff) <= percent
