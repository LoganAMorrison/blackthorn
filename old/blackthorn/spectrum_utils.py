from typing import Optional, List, Union

from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson
import numpy as np


class SpectrumLine:
    def __init__(self, xloc, br, mass=0.0) -> None:
        self._xloc = xloc
        self._br = br
        if mass != 0.0:
            raise ValueError("Massless particles not yet implemented.")

    def boost(self, xs, beta: float):
        dndxs = np.zeros_like(xs)
        br = self._br
        if beta >= 0.0 or beta <= 1.0:
            x0 = self._xloc
            dndxs = np.where(
                np.logical_and(x0 * (1 - beta) < xs, xs < x0 * (1 + beta)),
                br / (2 * beta * x0),
                0.0,
            )
        return Spectrum(xs, dndxs)

    def convolve(self, xs, eps):
        x0 = self._xloc
        br = self._br
        xe = x0 * eps
        dndxs = br * np.exp(-0.5 * ((xs - x0) / xe) ** 2) / (np.sqrt(2 * np.pi) * xe)
        return Spectrum(xs, dndxs)


class Spectrum:
    def __init__(
        self, xs, dndxs, lines: Optional[Union[SpectrumLine, List[SpectrumLine]]] = None
    ):
        self._xs = xs
        self._dndxs = dndxs
        self._spline = self.__make_spline()

        self._lines: List[SpectrumLine]
        if lines is not None:
            if isinstance(lines, list):
                self._lines = lines
            else:
                self._lines = [lines]
        else:
            self._lines = []

    def __make_spline(self):
        spline = UnivariateSpline(self._xs, self._dndxs, s=0, k=1, ext=1)
        return spline

    def __call__(self, x):
        return self._spline(x)

    @property
    def xs(self):
        return self._xs

    @property
    def dndxs(self):
        return self._dndxs

    def boost(self, beta: float):
        if beta == 0.0:
            return Spectrum(self._xs, self._dndxs)

        g2 = 1 / (1 - beta ** 2)
        # y = log(w) = log(1 - beta * z)
        yl = np.log(1 - beta)
        yu = np.log(1 + beta)
        ys = np.linspace(yl, yu, 100)

        def integrate(x):
            integrands = self._spline(g2 * x * np.exp(ys))
            return simpson(integrands, ys) / (2 * beta)

        xmin = np.min(self._xs)
        arg_new_xmin = np.argmin(self._xs / (1 + beta) > xmin)
        new_xs = self._xs[arg_new_xmin:]
        new_dndxs = np.array([integrate(x) for x in new_xs])

        for line in self._lines:
            line_spec = line.boost(new_xs, beta)
            new_dndxs = new_dndxs + line_spec.dndxs

        return Spectrum(new_xs, new_dndxs)

    def convolve(self, eps, nsig=3):
        eps2 = eps ** 2
        norm = 1 / np.sqrt(2 * np.pi * eps2)

        w = nsig * eps
        ys = np.linspace(1 / (1 + w), 1 / (1 - w), 100)
        kernel = np.exp(-1 / (2 * eps2) * (1 - ys) ** 2) / ys

        def integrate(x):
            integrands = kernel * self._spline(x / ys)
            return norm * simpson(integrands, ys)

        xmin = np.min(self._xs)
        arg_new_xmin = np.argmin(self._xs / (1 + w) > xmin)
        new_xs = self._xs[arg_new_xmin:]
        new_dndxs = np.array([integrate(x) for x in new_xs])

        for line in self._lines:
            line_spec = line.convolve(new_xs, eps)
            new_dndxs = new_dndxs + line_spec.dndxs

        return Spectrum(new_xs, new_dndxs)


def boost(dndx_rf, xs, beta):
    if beta == 0.0:
        return dndx_rf(xs)

    g2 = 1 / (1 - beta ** 2)
    # xr = xl * g^2 * w, wmin = 1-beta, wmax = 1+beta
    xmin = np.min(xs) / (1 + beta)
    xmax = min(1.0, np.max(xs) / (1 - beta))
    xs_ = np.geomspace(xmin, xmax, len(xs) * 10)
    # y = log(w) = log(1 - beta * z)
    yl = np.log(1 - beta)
    yu = np.log(1 + beta)
    ys = np.linspace(yl, yu, 100)

    spline = UnivariateSpline(xs_, dndx_rf(xs_), s=0, k=1, ext=1)

    def integrate(x):
        integrands = spline(g2 * x * np.exp(ys))
        return simpson(integrands, ys) / (2 * beta)

    return np.array([integrate(x) for x in xs])


def convolve(xs, dndxs, eps, nsig=3):
    eps2 = eps ** 2
    norm = 1 / np.sqrt(2 * np.pi * eps2)

    spline = UnivariateSpline(xs, dndxs, s=0, k=1, ext=1)
    ys = np.linspace(-eps * nsig, eps * nsig, 100)
    kernel = np.exp(-1 / (2 * eps2) * (1 - ys) ** 2) / ys

    def integrate(x):
        integrands = kernel * spline(x / ys)
        return simpson(integrands, ys)

    return np.array([norm * integrate(x) for x in xs])
