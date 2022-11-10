"""Tests for common RHN decay widths."""

# pylint: disable=invalid-name,too-many-locals,too-many-arguments


import numpy as np
import pytest
from helax.numpy.lvector import lnorm_sqr
from helax.numpy.phase_space import PhaseSpace

from blackthorn import fields
from blackthorn.constants import GF, SW, Gen
from blackthorn.fields import ChargedLepton, DownTypeQuark, UpTypeQuark
from blackthorn.models import RhNeutrinoGeV, feynman_rules


def _approximate_width_n_to_vvv(model: RhNeutrinoGeV, genv: Gen):
    mn = model.mass
    theta = model.theta
    uu = 0.5 * np.tan(2 * theta)

    if model.gen == genv:
        p = 2.0
    else:
        p = 1.0

    return p * GF**2 * mn**5 * uu**2 / (768 * np.pi**3)


def _approximate_width_n_to_vll(model: RhNeutrinoGeV, genl: Gen):
    mn = model.mass
    theta = model.theta
    f = ChargedLepton.from_gen(genl)
    x = f.mass / mn
    uu = 0.5 * np.tan(2 * theta)

    if mn < 2 * f.mass:
        return 0.0

    p = 1.0 if genl == model.gen else -1.0
    c1a = 4.0 * p
    c1b = 8.0

    c2a = 1.0 / 2.0
    c2b = 2.0
    c2c = p

    nz = 1.0

    c1 = 1.0 / 4.0 * (1.0 + c1a * SW**2 + c1b * SW**4)
    c2 = c2a * SW**2 * (c2b * SW**2 + c2c)

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


def _analytic_msqrd_n_to_lud(
    s,
    t,
    model: RhNeutrinoGeV,
    genl: Gen,
    genu: Gen,
    gend: Gen,
    coup: complex,
):
    """Compute the analytic squared matrix element for N->l+u+d."""

    mr = model.mass
    ml = ChargedLepton.from_gen(genl).mass
    mu = UpTypeQuark.from_gen(genu).mass
    md = DownTypeQuark.from_gen(gend).mass
    MW = fields.WBoson.mass
    GW = fields.WBoson.width

    pre = 3.0 / 2.0 * abs(coup) ** 2

    return (
        pre
        * (
            -(mr**4 * mu**4)
            - 4 * mr**2 * mu**2 * MW**4
            - ml**4 * mu**2 * (mu**2 + 4 * MW**2 - s)
            + mr**4 * mu**2 * s
            + mr**2 * mu**4 * s
            + 4 * mr**2 * MW**4 * s
            + 4 * mu**2 * MW**4 * s
            - mr**2 * mu**2 * s**2
            - 4 * MW**4 * s**2
            - md**4
            * (
                ml**4
                + mr**2 * (mr**2 + 4 * MW**2 - s)
                - ml**2 * (2 * mr**2 + s)
            )
            - 4 * mr**2 * mu**2 * MW**2 * t
            + 4 * mr**2 * MW**4 * t
            + 4 * mu**2 * MW**4 * t
            - 8 * MW**4 * s * t
            - 4 * MW**4 * t**2
            + ml**2
            * (
                2 * mr**2 * (mu**4 - 2 * MW**4 + mu**2 * (2 * MW**2 - s))
                + mu**4 * (-4 * MW**2 + s)
                + 4 * MW**4 * (s + t)
                + mu**2 * (-(s**2) + 4 * MW**2 * (s + t))
            )
            + md**2
            * (
                ml**4 * (2 * mu**2 + s)
                + mr**4 * (2 * mu**2 - 4 * MW**2 + s)
                + 4 * MW**4 * (-(mu**2) + s + t)
                - ml**2
                * (
                    4 * MW**4
                    + s**2
                    + 2 * mr**2 * (2 * mu**2 - 2 * MW**2 + s)
                    + mu**2 * (-4 * MW**2 + 2 * s)
                    + 4 * MW**2 * t
                )
                + mr**2
                * (mu**2 * (4 * MW**2 - 2 * s) - s**2 + 4 * MW**2 * (s + t))
            )
        )
        / (MW**4 * (MW**4 + MW**2 * (GW**2 - 2 * s) + s**2))
    )


@pytest.mark.parametrize(
    ["mass", "theta", "genn", "genv"],
    [
        (0.1, 1e-3, Gen.Fst, Gen.Fst),
        (0.1, 1e-3, Gen.Fst, Gen.Snd),
        (1.0, 1e-3, Gen.Fst, Gen.Fst),
        (1.0, 1e-3, Gen.Fst, Gen.Snd),
        (10.0, 1e-3, Gen.Fst, Gen.Fst),
        (10.0, 1e-3, Gen.Fst, Gen.Snd),
    ],
)
def test_width_n_to_vvv(mass: float, theta: float, genn: Gen, genv: Gen):
    """Test the partial width into three neutrinos."""
    model = RhNeutrinoGeV(mass=mass, theta=theta, gen=genn)
    width = model.width_v_v_v(genv1=genn, genv2=genv, genv3=genv)
    approx = _approximate_width_n_to_vvv(model, genv=genv)

    assert width == pytest.approx(approx, rel=0.2)


@pytest.mark.parametrize(
    ["mass", "theta", "genn", "genl"],
    [
        (0.3, 1e-3, Gen.Fst, Gen.Fst),
        (0.3, 1e-3, Gen.Fst, Gen.Snd),
        (10.0, 1e-3, Gen.Fst, Gen.Fst),
        (10.0, 1e-3, Gen.Fst, Gen.Snd),
    ],
)
def test_width_n_to_vll(mass: float, theta: float, genn: Gen, genl: Gen):
    """Test the partial width into three neutrinos."""
    model = RhNeutrinoGeV(mass=mass, theta=theta, gen=genn)
    width = model.width_v_l_l(genv=genn, genl1=genl, genl2=genl)
    approx = _approximate_width_n_to_vll(model, genl=genl)

    assert width == pytest.approx(approx, rel=0.2)


@pytest.mark.parametrize(
    ["mass", "theta", "genn", "genu", "gend"],
    [
        (10.0, 1e-3, Gen.Fst, Gen.Fst, Gen.Fst),
        (80.0, 1e-3, Gen.Fst, Gen.Fst, Gen.Fst),
        (10.0, 1e-3, Gen.Fst, Gen.Fst, Gen.Snd),
        (80.0, 1e-3, Gen.Fst, Gen.Fst, Gen.Snd),
        (10.0, 1e-3, Gen.Fst, Gen.Snd, Gen.Fst),
        (80.0, 1e-3, Gen.Fst, Gen.Snd, Gen.Fst),
    ],
)
def test_width_n_to_lud(mass: float, theta: float, genn: Gen, genu: Gen, gend: Gen):
    """Test the partial width into three neutrinos."""

    model = RhNeutrinoGeV(mass=mass, theta=theta, gen=genn)

    num_phase_space_points = 50_000
    fr_nlw = feynman_rules.vertex_wnl(theta, genn=genn, genl=genn)
    fr_wud = feynman_rules.vertex_wud(genu=genu, gend=gend, u_in=False)
    coup = fr_nlw.left * fr_wud.left
    ml = ChargedLepton.from_gen(genn).mass
    mu = UpTypeQuark.from_gen(genu).mass
    md = DownTypeQuark.from_gen(gend).mass

    def semi_analytic_width():
        """Semi-analytic width for RHN -> l + u + d."""

        def msqrd(momenta):
            pl = momenta[:, 0]
            pu = momenta[:, 1]
            pd = momenta[:, 2]
            s = lnorm_sqr(pu + pd)
            t = lnorm_sqr(pl + pd)

            return _analytic_msqrd_n_to_lud(s, t, model, genn, genu, gend, coup)

        phase_space = PhaseSpace(
            cme=model.mass,
            masses=[ml, mu, md],
            msqrd=msqrd,
        )

        return phase_space.decay_width(n=num_phase_space_points)[0]

    width = float(
        model.width_l_u_d(
            genu=genu,
            gend=gend,
            npts=num_phase_space_points,
        )
    )
    approx = semi_analytic_width()

    assert width == pytest.approx(approx, rel=0.1)
