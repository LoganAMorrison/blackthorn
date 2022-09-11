from typing import Optional, List, Union, Dict, Callable, overload, TypeVar
import pathlib
import dataclasses
import functools

from scipy import interpolate
from scipy.integrate import simpson, trapz
import numpy as np
import numpy.typing as npt
from hazma import spectra
from hazma.spectra import altarelli_parisi
import h5py

from . import fields

RealArray = npt.NDArray[np.float64]
RealOrRealArray = Union[npt.NDArray[np.float64], float]

THIS_DIR = pathlib.Path(__file__).parent
PPPC4DMID_DFILE = THIS_DIR.joinpath("data").joinpath("PPPC4DMID.hdf5")
HDMSPECTRA_DFILE = THIS_DIR.joinpath("data").joinpath("HDMSpectra.hdf5")

PDG_TO_NAME = {
    # quarks
    1: "d",
    2: "u",
    3: "s",
    4: "c",
    5: "b",
    6: "t",
    # Leptons
    11: "e",
    12: "ve",
    13: "mu",
    14: "vmu",
    15: "tau",
    16: "vtau",
    # Bosons
    21: "g",
    22: "a",
    23: "Z",
    24: "W",
    25: "h",
}

NAME_TO_PDG: Dict[str, int] = {val: key for key, val in PDG_TO_NAME.items()}

PRODUCT_PDG_TO_NAME = {
    22: "photon",
    -11: "positron",
    12: "ve",
    14: "vmu",
    16: "vtau",
}

FINAL_STATES = [
    "W",
    "Z",
    "a",
    "b",
    "c",
    "e",
    "h",
    "mu",
    "q",
    "t",
    "tau",
    "ve",
    "vmu",
    "vtau",
]


def _make_pppc4dmid_spectra():
    @dataclasses.dataclass
    class PPPC4DMIDSpectra:
        _interpolators: Dict[int, Dict[str, Callable]] = dataclasses.field(init=False)

        def __post_init__(self):
            self._interpolators = {}
            pars = {"kx": 1, "ky": 1, "s": 0.0}
            with h5py.File(PPPC4DMID_DFILE) as f:  # type: ignore
                for pdg in [22, -11, 12, 14, 16]:
                    self._interpolators[pdg] = {}
                    for fs in FINAL_STATES:
                        stable = PRODUCT_PDG_TO_NAME[pdg]
                        logm = f[stable][fs]["logM"][:]
                        logx = f[stable][fs]["logX"][:]
                        data = f[stable][fs]["data"][:, :].T
                        self._interpolators[pdg][fs] = interpolate.RectBivariateSpline(
                            logx, logm, data, **pars
                        )

        @overload
        def dndx(
            self,
            x: float,
            cme: float,
            final_state: str,
            product: int,
            single: bool = True,
        ) -> float:
            ...

        @overload
        def dndx(
            self,
            x: RealArray,
            cme: float,
            final_state: str,
            product: int,
            single: bool = True,
        ) -> RealArray:
            ...

        @overload
        def dndx(
            self,
            x: float,
            cme: RealArray,
            final_state: str,
            product: int,
            single: bool = True,
        ) -> RealArray:
            ...

        @overload
        def dndx(
            self,
            x: RealArray,
            cme: RealArray,
            final_state: str,
            product: int,
            single: bool = True,
        ) -> RealArray:
            ...

        def dndx(
            self,
            x: RealOrRealArray,
            cme: RealOrRealArray,
            final_state: str,
            product: int,
            single: bool = True,
        ) -> RealOrRealArray:
            assert product in self._interpolators, f"Invalid product {product}"
            interp_dict = self._interpolators[product]
            assert final_state in interp_dict, f"Invalid final state {final_state}"

            # if abs(product) == fields.Electron.pdg:
            #     mass = fields.Electron.mass
            # else:
            #     mass = 0.0

            # cme = 2 * M
            logm = np.log10(cme / 2.0)
            logx = np.log10(x)
            interp = interp_dict[final_state]
            # Convert log10(dN / dlog10(x)) -> dN/dx
            ldndlx = np.squeeze(interp(logx, logm)).T
            dndx = (10**ldndlx) / (np.log(10.0) * x)

            if single:
                return dndx / 2.0
            return dndx

        def dndx_bb(self, x: RealArray, cme: float, product: int):
            return self.dndx(x, cme, "b", product, single=False)

    return PPPC4DMIDSpectra()


def _make_hdmspectra():
    from HDMSpectra.HDMSpectra import spec as HDMSpectraSpec
    from HDMSpectra.HDMSpectra import FF as HDMSpectraFF

    datafile: pathlib.Path = THIS_DIR.joinpath("data").joinpath("HDMSpectra.hdf5")

    def spec(
        x,
        final_state,
        product,
        cme,
        delta: bool = False,
        interpolation: str = "cubic",
        final_state_bar: Optional[int] = None,
    ):
        if isinstance(x, float):
            x_ = np.array(x)
        else:
            x_ = x

        if final_state_bar is None:
            final_state_bar = -final_state

        mask = np.logical_and(x_ <= 1.0, x_ >= 1e-6)
        if delta:
            dndx = np.zeros((x_.shape[0] + 1,), dtype=x_.dtype)
            dndx_mask = np.append(mask, True)
        else:
            dndx = np.zeros_like(x_)
            dndx_mask = mask

        if np.any(mask):
            dndx[dndx_mask] = HDMSpectraSpec(
                X=final_state,
                finalstate=product,
                xvals=x_[mask],
                mDM=cme,
                data=datafile.as_posix(),
                delta=delta,
                interpolation=interpolation,
                Xbar=final_state_bar,
            )

        if isinstance(x, float) and not delta:
            dndx = dndx[0]

        return dndx

    def ff(x, final_state, product, cme, delta: bool = False):
        if isinstance(x, float):
            x_ = np.array(x)
        else:
            x_ = x

        mask = np.logical_and(x_ <= 1.0, x_ >= 1e-6)
        if delta:
            dndx = np.zeros((x_.shape[0] + 1,), dtype=x_.dtype)
            dndx_mask = np.append(mask, True)
        else:
            dndx = np.zeros_like(x_)
            dndx_mask = mask

        if np.any(mask):
            dndx[dndx_mask] = HDMSpectraFF(
                id_i=final_state,
                id_f=product,
                xvals=x_[mask],
                Qval=cme / 2.0,
                data=datafile.as_posix(),
                delta=delta,
            )

        if isinstance(x, float) and not delta:
            dndx = dndx[0]

        return dndx

    @dataclasses.dataclass
    class HDMSpectra:
        _datafile: pathlib.Path = THIS_DIR.joinpath("data").joinpath("HDMSpectra.hdf5")

        @overload
        def dndx(
            self,
            x: float,
            cme: float,
            final_state: int,
            product: int,
            delta: bool = False,
            final_state_bar: Optional[int] = None,
            interpolation: str = "cubic",
        ) -> float:
            ...

        @overload
        def dndx(
            self,
            x: RealArray,
            cme: float,
            final_state: int,
            product: int,
            delta: bool = False,
            final_state_bar: Optional[int] = None,
            interpolation: str = "cubic",
        ) -> RealArray:
            ...

        def dndx(
            self,
            x: RealOrRealArray,
            cme: float,
            final_state: int,
            product: int,
            delta: bool = False,
            final_state_bar: Optional[int] = None,
            interpolation: str = "cubic",
        ) -> RealOrRealArray:
            return spec(
                x=x,
                cme=cme,
                final_state=final_state,
                product=product,
                delta=delta,
                final_state_bar=final_state_bar,
                interpolation=interpolation,
            )

        @overload
        def fragmentation_function(
            self,
            x: float,
            cme: float,
            final_state: int,
            product: int,
            delta: bool = False,
        ) -> float:
            ...

        @overload
        def fragmentation_function(
            self,
            x: RealArray,
            cme: float,
            final_state: int,
            product: int,
            delta: bool = False,
        ) -> RealArray:
            ...

        def fragmentation_function(
            self,
            x: RealOrRealArray,
            cme: float,
            final_state: int,
            product: int,
            delta: bool = False,
        ) -> RealOrRealArray:
            return ff(
                x=x, cme=cme, final_state=final_state, product=product, delta=delta
            )

        def dndx_bb(
            self,
            x: RealArray,
            cme: float,
            product: int,
            delta: bool = False,
            interpolation="cubic",
        ):
            return spec(
                x=x,
                cme=cme,
                final_state=NAME_TO_PDG["b"],
                product=product,
                delta=delta,
                interpolation=interpolation,
            )

    return HDMSpectra()


PPPC4DMIDSpectra = _make_pppc4dmid_spectra()
HDMSpectra = _make_hdmspectra()


T = TypeVar("T", RealArray, float)
Dndx = Callable[[T, float], T]


def _call_with_mev(f, product_energy, parent_energy):
    return f(product_energy * 1e3, parent_energy * 1e3) * 1e3


def _dnde_photon_muon(e, emu):
    return _call_with_mev(spectra.dnde_photon_muon, e, emu)


def _dnde_photon_charged_pion(e, epi):
    return _call_with_mev(spectra.dnde_photon_charged_pion, e, epi)


def _dnde_photon_neutral_pion(e, epi):
    return _call_with_mev(spectra.dnde_photon_neutral_pion, e, epi)


def _dnde_photon_charged_kaon(e, ek):
    return _call_with_mev(spectra.dnde_photon_charged_kaon, e, ek)


def _dnde_photon_long_kaon(e, ek):
    return _call_with_mev(spectra.dnde_photon_long_kaon, e, ek)


def _dnde_photon_short_kaon(e, ek):
    return _call_with_mev(spectra.dnde_photon_short_kaon, e, ek)


def _dnde_photon_eta(e, eeta):
    return _call_with_mev(spectra.dnde_photon_eta, e, eeta)


FIELD_TO_DNDE_PHOTON: Dict[fields.QuantumField, Dndx] = {
    # Leptons
    fields.Muon: _dnde_photon_muon,
    # Mesons
    fields.ChargedPion: _dnde_photon_charged_pion,
    fields.NeutralPion: _dnde_photon_neutral_pion,
    fields.ChargedKaon: _dnde_photon_charged_kaon,
    fields.LongKaon: _dnde_photon_long_kaon,
    fields.ShortKaon: _dnde_photon_short_kaon,
    fields.Eta: _dnde_photon_eta,
}


def _dnde_photon_fsr_electron(e, cme):
    mass = fields.Electron.mass
    return altarelli_parisi.dnde_photon_ap_fermion(e, cme**2, mass=mass, charge=-1)


def _dnde_photon_fsr_muon(e, cme):
    mass = fields.Muon.mass
    return altarelli_parisi.dnde_photon_ap_fermion(e, cme**2, mass=mass, charge=-1)


def _dnde_photon_fsr_tau(e, cme):
    mass = fields.Tau.mass
    return altarelli_parisi.dnde_photon_ap_fermion(e, cme**2, mass=mass, charge=-1)


def _dnde_photon_fsr_charged_pion(e, cme):
    mass = fields.ChargedPion.mass
    return altarelli_parisi.dnde_photon_ap_scalar(e, cme**2, mass=mass, charge=-1)


def _dnde_photon_fsr_charged_kaon(e, cme):
    mass = fields.ChargedKaon.mass
    return altarelli_parisi.dnde_photon_ap_scalar(e, cme**2, mass=mass, charge=-1)


FIELD_TO_DNDE_PHOTON_FSR: Dict[fields.QuantumField, Dndx] = {
    # Leptons
    fields.Electron: _dnde_photon_fsr_electron,
    fields.Muon: _dnde_photon_fsr_muon,
    fields.Tau: _dnde_photon_fsr_tau,
    # Mesons
    fields.ChargedPion: _dnde_photon_fsr_charged_pion,
    fields.ChargedKaon: _dnde_photon_fsr_charged_kaon,
}


def _dnde_positron_muon(e, emu):
    return _call_with_mev(spectra.dnde_positron_muon, e, emu)


def _dnde_positron_charged_pion(e, epi):
    return _call_with_mev(spectra.dnde_positron_charged_pion, e, epi)


def _dnde_positron_charged_kaon(e, ek):
    return _call_with_mev(spectra.dnde_positron_charged_kaon, e, ek)


def _dnde_positron_long_kaon(e, ek):
    return _call_with_mev(spectra.dnde_positron_long_kaon, e, ek)


def _dnde_positron_short_kaon(e, ek):
    return _call_with_mev(spectra.dnde_positron_short_kaon, e, ek)


def _dnde_positron_eta(e, eeta):
    return _call_with_mev(spectra.dnde_positron_eta, e, eeta)


FIELD_TO_DNDE_POSITRON: Dict[fields.QuantumField, Dndx] = {
    # Leptons
    fields.Muon: _dnde_positron_muon,
    # Mesons
    fields.ChargedPion: _dnde_positron_charged_pion,
    fields.ChargedKaon: _dnde_positron_charged_kaon,
    fields.LongKaon: _dnde_positron_long_kaon,
    fields.ShortKaon: _dnde_positron_short_kaon,
    fields.Eta: _dnde_positron_eta,
}


def _dnde_electron_neutrino_muon(e: RealArray, emu: float):
    return _call_with_mev(
        functools.partial(spectra.dnde_neutrino_muon, flavor="e"), e, emu
    )


def _dnde_muon_neutrino_muon(e: RealArray, emu: float):
    return _call_with_mev(
        functools.partial(spectra.dnde_neutrino_muon, flavor="mu"), e, emu
    )


def _dnde_tau_neutrino_muon(e: RealArray, emu: float):
    return _call_with_mev(
        functools.partial(spectra.dnde_neutrino_muon, flavor="tau"), e, emu
    )


def _dnde_electron_neutrino_charged_pion(e: RealArray, epi: float):
    return _call_with_mev(
        functools.partial(spectra.dnde_neutrino_charged_pion, flavor="e"), e, epi
    )


def _dnde_muon_neutrino_charged_pion(e: RealArray, epi: float):
    return _call_with_mev(
        functools.partial(spectra.dnde_neutrino_charged_pion, flavor="mu"), e, epi
    )


def _dnde_tau_neutrino_charged_pion(e: RealArray, epi: float):
    return _call_with_mev(
        functools.partial(spectra.dnde_neutrino_charged_pion, flavor="tau"), e, epi
    )


FIELD_TO_DNDE_ELECTRON_NEUTRINO: Dict[fields.QuantumField, Dndx] = {
    # Leptons
    fields.Muon: _dnde_electron_neutrino_muon,  # type: ignore
    # Mesons
    fields.ChargedPion: _dnde_electron_neutrino_charged_pion,
}

FIELD_TO_DNDE_MUON_NEUTRINO: Dict[fields.QuantumField, Dndx] = {
    # Leptons
    fields.Muon: _dnde_muon_neutrino_muon,  # type: ignore
    # Mesons
    fields.ChargedPion: _dnde_muon_neutrino_charged_pion,
}

FIELD_TO_DNDE_TAU_NEUTRINO: Dict[fields.QuantumField, Dndx] = {
    # Leptons
    fields.Muon: _dnde_tau_neutrino_muon,  # type: ignore
    # Mesons
    fields.ChargedPion: _dnde_tau_neutrino_charged_pion,
}


class SpectrumLine:
    def __init__(self, xloc, br, mu=0.0) -> None:
        self._xloc = xloc
        self._br = br
        self._mu = mu

    def boost(self, xs, beta: float):
        dndxs = np.zeros_like(xs)
        br = self._br
        mu = self._mu
        x0 = self._xloc
        if (beta >= 0.0 or beta <= 1.0) and x0 > mu:
            xmin = x0 - beta * np.sqrt(x0**2 - mu**2)
            xmax = x0 + beta * np.sqrt(x0**2 - mu**2)
            cond = np.logical_and(xmin < xs, xs < xmax)
            dndxs = np.where(cond, br / (2 * beta * np.sqrt(x0**2 - mu**2)), 0.0)
        return Spectrum(xs, dndxs)

    def convolve(self, xs, eps):
        x0 = self._xloc
        br = self._br
        xe = x0 * eps
        dndxs = br * np.exp(-0.5 * ((xs - x0) / xe) ** 2) / (np.sqrt(2 * np.pi) * xe)
        return Spectrum(xs, dndxs)


class Spectrum:
    def __init__(
        self,
        x,
        dndx,
        mu: float = 0.0,
        lines: Optional[Union[SpectrumLine, List[SpectrumLine]]] = None,
    ):
        self._x = x
        self._dndx = dndx
        self._mu = mu
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
        spline = interpolate.InterpolatedUnivariateSpline(
            self._x, self._dndx, k=1, ext=1
        )
        return spline

    def __call__(self, x):
        return self._spline(x)

    def __mul__(self, x):
        assert isinstance(x, float) or isinstance(x, int)
        self._dndx *= x
        for line in self._lines:
            line._br *= x
        return self

    def __rmul__(self, x):
        assert isinstance(x, float) or isinstance(x, int)
        self._dndx *= x
        for line in self._lines:
            line._br *= x
        return self

    def __add__(self, other):
        self._dndx += other._dndx
        for line in other._lines:
            self._lines.append(line)
        return self

    def __radd__(self, other):
        self._dndx += other._dndx
        for line in other._lines:
            self._lines.append(line)
        return self

    @property
    def x(self):
        return self._x

    @property
    def dndx(self):
        return self._dndx

    def boost(self, beta: float, trim: bool = True) -> "Spectrum":
        if beta == 0.0:
            return self

        mu1 = self._mu
        x = self._x
        y1 = self._dndx / np.sqrt(x**2 - mu1**2) / (2.0 * beta)
        integrand = interpolate.InterpolatedUnivariateSpline(x, y1, k=1, ext=1)

        g2 = 1.0 / (1.0 - beta**2)
        mu2 = mu1 / np.sqrt(g2)
        x_minus = g2 * x * (1.0 - beta * np.sqrt(1.0 - (mu2 / x) ** 2))
        x_plus = g2 * x * (1.0 + beta * np.sqrt(1.0 - (mu2 / x) ** 2))
        dndx = np.array([integrand.integral(xm, xp) for xm, xp in zip(x_minus, x_plus)])

        if trim:
            idx_min = np.argmin(x_minus > np.min(x))
            x = x[idx_min:]
            dndx = dndx[idx_min:]

        for line in self._lines:
            line_spec = line.boost(x, beta)
            dndx += line_spec.dndx

        return Spectrum(x, dndx)

    def convolve(self, eps):
        eps2 = eps**2
        norm = 1 / (eps * np.sqrt(2 * np.pi))

        xp = np.expand_dims(self.x, 0)
        x = np.expand_dims(self.x, 1)
        kernel = (1.0 / xp) * np.exp(-0.5 / eps2 * (1.0 - x / xp) ** 2)
        integrands = np.expand_dims(self.dndx, 0) * kernel
        dndx = norm * simpson(integrands, self.x, axis=-1)

        for line in self._lines:
            line_spec = line.convolve(self._x, eps)
            dndx += line_spec.dndx

        return Spectrum(self._x, dndx)


def boost(dndx_rf, xs, beta):
    if beta == 0.0:
        return dndx_rf(xs)

    g2 = 1 / (1 - beta**2)
    # xr = xl * g^2 * w, wmin = 1-beta, wmax = 1+beta
    xmin = np.min(xs) / (1 + beta)
    xmax = min(1.0, np.max(xs) / (1 - beta))
    xs_ = np.geomspace(xmin, xmax, len(xs) * 10)
    # y = log(w) = log(1 - beta * z)
    yl = np.log(1 - beta)
    yu = np.log(1 + beta)
    ys = np.linspace(yl, yu, 100)

    spline = interpolate.UnivariateSpline(xs_, dndx_rf(xs_), s=0, k=1, ext=1)

    def integrate(x):
        integrands = spline(g2 * x * np.exp(ys))
        return simpson(integrands, ys) / (2 * beta)

    return np.array([integrate(x) for x in xs])


def convolve(xs, dndxs, eps, nsig=3):
    eps2 = eps**2
    norm = 1 / np.sqrt(2 * np.pi * eps2)

    spline = interpolate.UnivariateSpline(xs, dndxs, s=0, k=1, ext=1)
    ys = np.linspace(-eps * nsig, eps * nsig, 100)
    kernel = np.exp(-1 / (2 * eps2) * (1 - ys) ** 2) / ys

    def integrate(x):
        integrands = kernel * spline(x / ys)
        return simpson(integrands, ys)

    return np.array([norm * integrate(x) for x in xs])

    # def boost(self, beta: float) -> "Spectrum":
    #     if beta == 0.0:
    #         return self

    #     g2 = 1 / (1 - beta**2)
    #     # y = log(w) = log(1 - beta * z)
    #     yl = np.log(1 - beta)
    #     yu = np.log(1 + beta)
    #     ys = np.linspace(yl, yu, 100)

    #     def integrate(x):
    #         integrands = self._spline(g2 * x * np.exp(ys))
    #         return simpson(integrands, ys) / (2 * beta)

    #     xmin = np.min(self._x)
    #     arg_new_xmin = np.argmin(self._x / (1 + beta) > xmin)
    #     new_xs = self._x[arg_new_xmin:]
    #     new_dndxs = np.array([integrate(x) for x in new_xs])

    #     for line in self._lines:
    #         line_spec = line.boost(new_xs, beta)
    #         new_dndxs = new_dndxs + line_spec.dndx

    #     return Spectrum(new_xs, new_dndxs)

    # def convolve(self, eps, nsig=3):
    #     eps2 = eps**2
    #     norm = 1 / np.sqrt(2 * np.pi * eps2)

    #     w = nsig * eps
    #     ys = np.linspace(1 / (1 + w), 1 / (1 - w), 100)
    #     kernel = np.exp(-1 / (2 * eps2) * (1 - ys) ** 2) / ys

    #     def integrate(x):
    #         integrands = kernel * self._spline(x / ys)
    #         return norm * simpson(integrands, ys)

    #     xmin = np.min(self._x)
    #     arg_new_xmin = np.argmin(self._x / (1 + w) > xmin)
    #     new_xs = self._x[arg_new_xmin:]
    #     new_dndxs = np.array([integrate(x) for x in new_xs])

    #     for line in self._lines:
    #         line_spec = line.convolve(new_xs, eps)
    #         new_dndxs = new_dndxs + line_spec.dndx

    #     return Spectrum(new_xs, new_dndxs)


def spec_res_fn(ep, e, energy_res):
    """Get the spectral resolution function."""
    sigma = e * energy_res(e)

    if sigma == 0:
        if hasattr(ep, "__len__"):
            return np.zeros(ep.shape)
        else:
            return 0.0
    else:
        return (
            1.0
            / np.sqrt(2.0 * np.pi * sigma**2)
            * np.exp(-((ep - e) ** 2) / (2.0 * sigma**2))
        )


def convolved_spectrum_fn(
    e_min, e_max, energy_res, spec_fn=None, lines=None, n_pts=1000, aeff=None
):
    r"""
    Convolves a continuum and line spectrum with a detector's spectral
    resolution function.

    Parameters
    ----------
    e_min : float
        Lower bound of energy range over which to perform convolution.
    e_max : float
        Upper bound of energy range over which to perform convolution.
    energy_res : float -> float
        The detector's energy resolution (Delta E / E) as a function of
        photon energy in MeV.
    spec_fn : np.array -> np.array
        Continuum spectrum function.
    lines : dict
        Information about spectral lines.
    n_pts : float
        Number of points to use to create resulting interpolating function.

    Returns
    -------
    dnde_conv : InterpolatedUnivariateSpline
        An interpolator giving the DM annihilation spectrum as seen by the
        detector. Using photon energies outside the range [e_min, e_max] will
        produce a ``bounds_errors``.
    """
    es = np.geomspace(e_min, e_max, n_pts)
    dnde_conv = np.zeros(es.shape)

    # Pad energy grid to avoid edge effects
    es_padded = np.geomspace(0.1 * e_min, 10 * e_max, n_pts)
    if spec_fn is not None:
        dnde_src = spec_fn(es_padded)
        if not np.all(dnde_src == 0):

            def integral(e):
                """
                Performs the integration at given photon energy.
                """
                spec_res_fn_vals = spec_res_fn(es_padded, e, energy_res)
                if np.any(spec_res_fn_vals > 0.0):
                    integrand_vals = (
                        dnde_src * spec_res_fn_vals / trapz(spec_res_fn_vals, es_padded)
                    )

                    return trapz(integrand_vals, es_padded)
                else:
                    return 0.0

            dnde_conv += np.vectorize(integral)(es)

    # Line contribution
    if lines is not None:
        for line in lines.values():
            dnde_conv += line["bf"] * spec_res_fn(es, line["energy"], energy_res)

    if aeff is not None:
        dnde_conv *= aeff(es)

    return interpolate.InterpolatedUnivariateSpline(es, dnde_conv, k=1, ext=1)
