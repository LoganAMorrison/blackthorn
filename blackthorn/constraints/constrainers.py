import warnings
from copy import copy
from abc import ABC, abstractmethod
import pathlib
import dataclasses
from typing import Dict, List

import numpy as np
import numpy.typing as npt
from scipy.optimize import root_scalar, RootResults
from typing import Optional
from rich.progress import Progress
from scipy.stats import norm, chi2
from scipy import interpolate

from hazma.relic_density import relic_density
from hazma.parameters import omega_h2_cdm, sv_inv_MeV_to_cm3_per_s
import hazma.gamma_ray_parameters as grp
from hazma.flux_measurement import FluxMeasurement
from hazma.target_params import TargetParams

from blackthorn.models.base import RhNeutrinoBase
from blackthorn.models.mev import RhNeutrinoMeV
from blackthorn.models.gev import RhNeutrinoGeV
from blackthorn.models.tev import RhNeutrinoTeV
from blackthorn.constants import Gen
from blackthorn import fields

# from blackthorn.constraints import iterators

from .experiment_data import FermiDwarfObservation, FermiDwarfTarget

RealArray = npt.NDArray[np.floating]

THIS_DIR = pathlib.Path(__file__).parent.absolute()


ICECUBE_MEASUREMENT = FluxMeasurement.from_file(
    THIS_DIR.joinpath("..", "data", "icecube.csv"),
    target=TargetParams(J=1.695e28 * 1e-6, D=3.563e25 * 1e-3, dOmega=10.82),
    energy_res=np.vectorize(lambda _: 0.25),
)

HAWC_ENERGY_RES = np.genfromtxt(
    THIS_DIR.joinpath("..", "data", "hawc_res.csv"), delimiter=","
)
HAWC_MEASUREMENT_M31 = FluxMeasurement.from_file(
    THIS_DIR.joinpath("..", "data", "hawc_m31.csv"),
    energy_res=interpolate.InterpolatedUnivariateSpline(
        HAWC_ENERGY_RES.T[0], HAWC_ENERGY_RES.T[1], ext=1, k=1
    ),
    target=TargetParams(1.27e20, 1.56e20, dOmega=0.0597),
)


def __load_measurement(fname, diffuse: TargetParams, energy_res):
    mev_to_gev = 1e-3
    gev_to_mev = 1e3

    fname = THIS_DIR.joinpath("..", "data", fname)
    data = np.loadtxt(fname, delimiter=",")
    e_lows = data.T[0] * mev_to_gev
    e_highs = data.T[1] * mev_to_gev
    fluxes = data.T[2] * mev_to_gev
    upper_errors = data.T[3] * mev_to_gev
    lower_errors = data.T[4] * mev_to_gev

    assert diffuse.J is not None
    assert diffuse.D is not None

    target = TargetParams(
        J=diffuse.J * mev_to_gev**2,
        D=diffuse.D * mev_to_gev,
        dOmega=diffuse.dOmega,
    )

    def energy_res_(e):
        return energy_res(e * gev_to_mev)

    return FluxMeasurement(
        e_lows,
        e_highs,
        fluxes,
        upper_errors,
        lower_errors,
        energy_res=energy_res_,
        target=target,
    )


FERMI_MEASUREMENT = __load_measurement(
    "fermi_diffuse.csv", grp.fermi_diffuse_target, grp.energy_res_fermi
)
EGRET_MEASUREMENT = __load_measurement(
    "egret_diffuse.csv", grp.egret_diffuse_target, grp.energy_res_egret
)
COMPTEL_MEASUREMENT = __load_measurement(
    "comptel_diffuse.csv", grp.comptel_diffuse_target, grp.energy_res_comptel
)
INTEGRAL_MEASUREMENT = __load_measurement(
    "integral_diffuse.csv", grp.integral_diffuse_target, grp.energy_res_integral
)


@dataclasses.dataclass
class EventEstimates:
    measured: Dict[str, npt.NDArray]
    predicted: npt.NDArray


def sigmav(model, vx: float = 1e-3):
    """Compute <σv> for the given model."""
    cme = 2 * model.mx * (1.0 + 0.5 * vx**2)
    sig = model.annihilation_cross_sections(cme)["total"]
    return sig * vx * sv_inv_MeV_to_cm3_per_s


def max_bin_test(phi, measurement: FluxMeasurement, n_sigma):
    phi_max = (
        measurement.target.dOmega
        * (measurement.e_highs - measurement.e_lows)
        * (n_sigma * measurement.upper_errors + measurement.fluxes)
    )

    # Return the most stringent limit
    lims = np.ones_like(phi) * np.inf
    mask = phi > 0.0
    if np.any(mask):
        lims[mask] = phi_max[mask] / phi[mask]
    return np.min(lims)


def chi2_test(phi, measurement: FluxMeasurement, n_sigma: float):
    phi_obs = (
        measurement.target.dOmega
        * (measurement.e_highs - measurement.e_lows)
        * measurement.fluxes
    )
    # Errors on integrated fluxes
    phi_uncer = (
        measurement.target.dOmega
        * (measurement.e_highs - measurement.e_lows)
        * measurement.upper_errors
    )

    chi2_obs = np.sum(np.maximum(phi - phi_obs, 0) ** 2 / phi_uncer**2)

    if chi2_obs == 0:
        return np.inf

    # Convert n_sigma to chi^2 critical value
    p_val = norm.cdf(n_sigma)
    chi2_crit = chi2.ppf(p_val, df=len(phi))
    return np.sqrt(chi2_crit / chi2_obs)


def measurement_to_gev(measurement: FluxMeasurement, power: float = 2):
    measurement = grp.integral_diffuse

    measurement.e_lows = measurement.e_lows * 1e-3
    measurement.e_highs = measurement.e_highs * 1e-3
    measurement.fluxes = measurement.fluxes * 1e3
    measurement.upper_errors = measurement.upper_errors * 1e3
    measurement.lower_errors = measurement.lower_errors * 1e3
    measurement.target = TargetParams(
        # MeV^2 cm^-5 -> GeV^2 cm^-5
        J=measurement.target.J * 1e-6,
        # MeV cm^-2 -> GeV cm^-2.
        D=measurement.target.D * 1e-3,
        dOmega=measurement.target.dOmega,
        vx=measurement.target.vx,
    )

    def energy_res(e):
        return measurement.energy_res(e) * 1e-3

    measurement.energy_res = energy_res
    return measurement


def _compute_fluxes(
    model: RhNeutrinoBase,
    lower_energies,
    upper_energies,
    target,
    energy_res,
    product: fields.QuantumField,
    npts: int,
    **kwargs,
):
    mass = model.mass
    e_min, e_max = lower_energies[0], upper_energies[-1]

    mx = kwargs.get("mx")
    ann = mx is not None
    pre = 2.0 if ann else 1.0

    if ann:
        f_dm = 2.0
        dm_flux_factor = (
            target.J
            * target.dOmega
            / (2.0 * f_dm * mx**2 * 4.0 * np.pi)  # type: ignore
        )

        # TODO: this should depend on the target!
        cme = 2.0 * mx * (1.0 + 0.5 * target.vx**2)
        dnde_conv = model.total_conv_spectrum_fn(  # type: ignore
            e_min=e_min,
            e_max=e_max,
            energy_res=energy_res,
            product=product,
            npts=npts,
            aeff=kwargs.get("aeff"),
            cme=cme,
        )
    else:
        # Factor to convert dN/dE to Phi. Factor of 2 comes from DM not being
        # self-conjugate.
        dm_flux_factor = target.D * target.dOmega / (mass * 4.0 * np.pi)  # type: ignore

        dnde_conv = model.total_conv_spectrum_fn(
            e_min=e_min,
            e_max=e_max,
            energy_res=energy_res,
            product=product,
            npts=npts,
            aeff=kwargs.get("aeff"),
        )

    # Integrated flux (excluding <sigma v>) from DM processes in each bin
    phi = []
    for e_low, e_high in zip(lower_energies, upper_energies):
        phi_ = pre * dm_flux_factor * dnde_conv.integral(e_low, e_high)
        phi.append(phi_)

    return np.array(phi)


# ============================================================================
# ---- Abstract Constrainers -------------------------------------------------
# ============================================================================


class AbstractConstrainer(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def _constrain(self, model):
        pass

    def _iterate(self, fn, model_iterator, **kwargs):
        """
        Compute the constraints on the models.

        Parameters
        ----------
        model_iterator: iter
            Iterator over the dark matter models.
        progress: Optional[Progress]
            A `rich` progress object to track progress.

        Returns
        -------
        constraints: array-like
            Numpy array containing the constraints for each model.
        """
        progress = kwargs.get("progress")

        if progress is not None:
            task = progress.add_task(self.description, total=len(model_iterator))

        def progress_update():
            if progress is not None:
                progress.update(task, advance=1, refresh=True)

        results = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for model in model_iterator:
                results.append(fn(model, **kwargs))
                progress_update()
        return results

    def constrain(self, model_iterator, **kwargs):
        """
        Compute the constraints on the models.

        Parameters
        ----------
        model_iterator: iter
            Iterator over the dark matter models.
        progress: Optional[Progress]
            A `rich` progress object to track progress.

        Returns
        -------
        constraints: array-like
            Numpy array containing the constraints for each model.
        """

        def fn(model, *args, **kws):
            return self._constrain(model, *args, **kws)

        results = self._iterate(fn, model_iterator, **kwargs)
        return np.array(results)


class CompositeConstrainer(ABC):
    def __init__(self, constrainers=[]):
        self._constrainers = constrainers

    def add_constrainer(self, constrainer):
        self._constrainers.append(constrainer)

    def reset_constrainers(self):
        self._constrainers = []

    @property
    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError()

    def __len__(self):
        return len(self._constrainers)

    def _iterate(self, fn, model_iterator, **kwargs):
        progress = kwargs.get("progress")
        if progress is None:
            overall = None
        else:
            overall = progress.add_task(self.description, total=len(self))

        results = {}
        for constrainer in self._constrainers:
            name = constrainer.name
            results[name] = fn(constrainer, model_iterator, **kwargs)

            if progress is not None and overall is not None:
                progress.update(overall, advance=1, refresh=True)

        return results

    def _constrain(self, model_iterator, **kwargs):
        def fn(constrainer, *args, **kws):
            return constrainer.constrain(*args, **kws)

        return self._iterate(fn, model_iterator, **kwargs)


class ExistingTelescopeConstrainer(AbstractConstrainer):
    """
    Class for computing the constraints on the dark-matter models from an
    existing telescope.
    """

    def __init__(
        self,
        measurement,
        product: fields.QuantumField,
        sigma=2.0,
        method="1bin",
        npts=1000,
    ):
        super().__init__()
        self.measurement = measurement
        self.sigma = sigma
        self.method = method
        self.product = product
        self.npts = npts

        bin_widths = measurement.e_highs - measurement.e_lows
        scaling = measurement.target.dOmega * bin_widths
        self.centeral_phi = scaling * measurement.fluxes
        self.upper_uncertainty_phi = scaling * measurement.upper_errors
        self.lower_uncertainty_phi = scaling * measurement.lower_errors

    def _compute_fluxes(self, model: RhNeutrinoBase, **kwargs):
        mass = model.mass
        measurement = self.measurement
        e_min, e_max = self.measurement.e_lows[0], self.measurement.e_highs[-1]

        mx = kwargs.get("mx")
        ann = mx is not None
        pre = 2.0 if ann else 1.0

        if ann:
            f_dm = 2.0
            dm_flux_factor = (
                measurement.target.J
                * measurement.target.dOmega
                / (2.0 * f_dm * mx**2 * 4.0 * np.pi)  # type: ignore
            )

            # TODO: this should depend on the target!
            cme = 2.0 * mx * (1.0 + 0.5 * measurement.target.vx**2)
            dnde_conv = model.total_conv_spectrum_fn(  # type: ignore
                e_min=e_min,
                e_max=e_max,
                energy_res=measurement.energy_res,
                product=self.product,
                npts=self.npts,
                aeff=kwargs.get("aeff"),
                cme=cme,
            )
        else:
            # Factor to convert dN/dE to Phi. Factor of 2 comes from DM not being
            # self-conjugate.
            dm_flux_factor = (
                measurement.target.D
                * measurement.target.dOmega
                / (mass * 4.0 * np.pi)  # type: ignore
            )

            dnde_conv = model.total_conv_spectrum_fn(
                e_min=e_min,
                e_max=e_max,
                energy_res=self.measurement.energy_res,
                product=self.product,
                npts=self.npts,
                aeff=kwargs.get("aeff"),
            )

        # Integrated flux (excluding <sigma v>) from DM processes in each bin
        phi = []
        for e_low, e_high in zip(measurement.e_lows, measurement.e_highs):
            phi_ = pre * dm_flux_factor * dnde_conv.integral(e_low, e_high)
            phi.append(phi_)

        return np.array(phi)

    def _constrain(self, model: RhNeutrinoBase, **kwargs):
        r"""
        Determines the limit on :math:`<sigma v>` from gamma-ray data.

        We define a signal to be in conflict with the measured flux for bin
        :math:`i` for an experiment if

        .. math::

            \Phi_{\chi}^{(i)} > n_{\sigma} \sigma^{(i)} + \Phi^{(i)},

        where :math:`\Phi_\chi^{(i)}` is the integrated flux due to DM
        annihilations for the bin, :math:`\Phi^{(i)}` is the measured flux in
        the bin, :math:`\sigma^{(i)}` is size of the upper error bar for the
        bin and :math:`n_{\sigma} = 2` is the significance. The overall limit on
        :math:`\langle\sigma v\rangle` is computed by minimizing over the
        limits determined for each bin.

        Parameters
        ----------
        measurement : FluxMeasurement
            Information about the flux measurement and target.
        n_sigma : float
            See the notes for this function.

        Returns
        -------
        <sigma v>_tot : float
            Largest allowed thermally averaged total cross section in cm^3 / s

        """
        ann = kwargs.get("mx") is not None
        phi = self._compute_fluxes(model, **kwargs)
        power = 1.0 if ann else -1.0

        if self.method == "1bin":
            return max_bin_test(phi, self.measurement, self.sigma) ** power
        elif self.method == "chi2":
            return chi2_test(phi, self.measurement, self.sigma) ** power
        else:
            raise NotImplementedError()

    def compute_fluxes(self, model_iterator, **kwargs):
        fluxes = np.array(self._iterate(self._compute_fluxes, model_iterator, **kwargs))
        measured = {
            "centeral": self.centeral_phi,
            "lower_uncertainty": self.upper_uncertainty_phi,
            "upper_uncertainty": self.lower_uncertainty_phi,
        }
        return EventEstimates(measured, fluxes)


# ============================================================================
# ---- General Constrainers --------------------------------------------------
# ============================================================================


class CmbConstrainer(AbstractConstrainer):
    """
    Class for computing constraints on dark-matter models from CMB.
    """

    def __init__(self, x_kd=1e-6):
        super().__init__()
        self.x_kd = x_kd

    def _constrain(self, model):
        return model.cmb_limit(x_kd=self.x_kd)

    @property
    def description(self):
        return "[purple] CMB"

    @property
    def name(self):
        return "cmb"


class RelicDensityConstrainer(AbstractConstrainer):
    """
    Class for computing constraints on dark-matter models from relic-density.
    """

    def __init__(
        self,
        prop: str,
        prop_min: float,
        prop_max: float,
        vx: float = 1e-3,
        log: bool = True,
    ):
        """
        Create a constrainer object for constraining the dark-matter
        annihilation cross section by varying a specified property
        such that the model yields the correct relic-density.

        Parameters
        ----------
        prop: str
            String specifying the property to vary in order fix the
            dark-matter relic-density.
        prop_min: float
            Minimum value of the property.
        prop_max: float
            Maximum value of the property.
        vx: float, optional
            The dark-matter velocity used to compute the annihilation cross
            section. Default is 1e-3.
        log: bool, optional
            If true, the property is varied logarithmically.
        """
        super().__init__()
        self.prop = prop
        self.prop_min = prop_min
        self.prop_max = prop_max
        self.vx = vx
        self.log = log

    @property
    def description(self):
        return "[dark_violet] Relic Density"

    @property
    def name(self):
        return "relic-density"

    def _setprop(self, model, val):
        if self.log:
            setattr(model, self.prop, 10**val)
        else:
            setattr(model, self.prop, val)

    def _constrain(self, model):
        model_ = copy(model)
        lb = self.prop_min if not self.log else np.log10(self.prop_min)
        ub = self.prop_max if not self.log else np.log10(self.prop_max)

        def f(val):
            self._setprop(model_, val)
            return relic_density(model_, semi_analytic=True) - omega_h2_cdm

        try:
            root: RootResults = root_scalar(f, bracket=[lb, ub], method="brentq")
            if not root.converged:
                warnings.warn(f"root_scalar did not converge. Flag: {root.flag}")
            self._setprop(model_, root.root)
            return sigmav(model_, self.vx)
        except ValueError as e:
            warnings.warn(f"Error encountered: {e}. Returning nan", RuntimeWarning)
            return np.nan


# ============================================================================
# ---- Specific Constrainers -------------------------------------------------
# ============================================================================


class ComptelConstrainer(ExistingTelescopeConstrainer):
    def __init__(self, sigma: float = 2.0, method: str = "1bin", npts: int = 1000):
        super().__init__(
            measurement=COMPTEL_MEASUREMENT,
            product=fields.Photon,
            sigma=sigma,
            method=method,
            npts=npts,
        )

    @property
    def description(self):
        return "[deep_sky_blue2] COMPTEL"

    @property
    def name(self):
        return "comptel"


class EgretConstrainer(ExistingTelescopeConstrainer):
    def __init__(self, sigma: float = 2.0, method: str = "1bin", npts: int = 1000):
        super().__init__(
            measurement=EGRET_MEASUREMENT,
            product=fields.Photon,
            sigma=sigma,
            method=method,
            npts=npts,
        )

    @property
    def description(self):
        return "[deep_sky_blue1] EGRET"

    @property
    def name(self):
        return "egret"


class FermiConstrainer(ExistingTelescopeConstrainer):
    def __init__(self, sigma: float = 2.0, method: str = "1bin", npts: int = 1000):
        super().__init__(
            measurement=FERMI_MEASUREMENT,
            product=fields.Photon,
            sigma=sigma,
            method=method,
            npts=npts,
        )

    @property
    def description(self):
        return "[light_sea_green] Fermi"

    @property
    def name(self):
        return "fermi"


class IntegralConstrainer(ExistingTelescopeConstrainer):
    def __init__(self, sigma: float = 2.0, method: str = "1bin", npts: int = 1000):
        super().__init__(
            measurement=INTEGRAL_MEASUREMENT,
            product=fields.Photon,
            sigma=sigma,
            method=method,
            npts=npts,
        )

    @property
    def description(self):
        return "[dark_cyan] INTEGRAL"

    @property
    def name(self):
        return "integral"


class IceCubeConstrainer(ExistingTelescopeConstrainer):
    def __init__(self, sigma: float = 2.0, method: str = "1bin", npts: int = 1000):
        super().__init__(
            measurement=ICECUBE_MEASUREMENT,
            product=fields.MuonNeutrino,
            sigma=sigma,
            method=method,
            npts=npts,
        )

    @property
    def description(self):
        return "[dark_cyan] ICE-CUBE"

    @property
    def name(self):
        return "icecube"


class HawcConstrainer(ExistingTelescopeConstrainer):
    def __init__(self, sigma: float = 2.0, method: str = "1bin", npts: int = 1000):
        super().__init__(
            measurement=HAWC_MEASUREMENT_M31,
            product=fields.MuonNeutrino,
            sigma=sigma,
            method=method,
            npts=npts,
        )

    @property
    def description(self):
        return "[dark_cyan] HAWC"

    @property
    def name(self):
        return "hawc"


# ============================================================================
# ---- Fermi Dwarf Constrainers ----------------------------------------------
# ============================================================================


class FermiDracoConstrainer(AbstractConstrainer):
    def __init__(self, npts: int = 1000):
        obs = FermiDwarfObservation.load_target_data(
            FermiDwarfTarget.Draco, energy_units="GeV"
        )
        self.__npts = npts
        # This is from Hazma, 5 deg cone
        # 1 arcmin cone
        # J=3.418e30, D=5.949e25, dOmega=2.66e-7
        # 5 deg cone:
        # J=8.058e26, D=1.986e24, dOmega=0.0239
        dOmegaH = 0.0239
        self.__jfactor = 8.058e26 * 1e-6 * dOmegaH
        self.__dfactor = 1.986e24 * 1e-6 * dOmegaH

        lower_energies: List[float] = []
        upper_energies: List[float] = []
        diff_flux_upper_limits: List[float] = []

        for entry in obs.data:
            lower_energies.append(entry.lower_energy)
            upper_energies.append(entry.upper_energy)
            diff_flux_upper_limits.append(entry.flux_upper_limit)

        self.__lower_energies: RealArray = np.array(lower_energies)
        self.__upper_energies: RealArray = np.array(upper_energies)

        # We currently have E^2 dPhi/dE. Transform to dPhi/dE
        centers = np.sqrt(self.__lower_energies * self.__upper_energies)
        self.__diff_flux_upper_limits: RealArray = (
            np.array(diff_flux_upper_limits) / centers**2
        )

    @property
    def description(self):
        return "[light_sea_green] Fermi (Draco)"

    @property
    def name(self):
        return "fermi_draco"

    def _compute_fluxes(
        self,
        model: RhNeutrinoBase,
        energy_res,
        product: fields.QuantumField,
        npts: int,
        **kwargs,
    ):
        mass = model.mass
        e_min, e_max = self.__lower_energies[0], self.__upper_energies[-1]

        mx = kwargs.get("mx")
        ann = mx is not None
        pre = 2.0 if ann else 1.0

        if ann:
            f_dm = 2.0
            dm_flux_factor = self.__jfactor / (2.0 * f_dm * mx**2 * 4.0 * np.pi)

            cme = 2.0 * mx * (1.0 + 0.5 * 1e-6)
            dnde_conv = model.total_conv_spectrum_fn(
                e_min=e_min,
                e_max=e_max,
                energy_res=energy_res,
                product=product,
                npts=npts,
                # aeff=kwargs.get("aeff"),
                cme=cme,
            )
        else:
            # Factor to convert dN/dE to Phi. Factor of 2 comes from DM not being
            # self-conjugate.
            dm_flux_factor = self.__dfactor / (mass * 4.0 * np.pi)

            dnde_conv = model.total_conv_spectrum_fn(
                e_min=e_min,
                e_max=e_max,
                energy_res=energy_res,
                product=product,
                npts=npts,
                # aeff=kwargs.get("aeff"),
            )

        # Integrated flux (excluding <sigma v>) from DM processes in each bin
        phi = []
        for e_low, e_high in zip(self.__lower_energies, self.__upper_energies):
            phi_ = pre * dm_flux_factor * dnde_conv.integral(e_low, e_high)
            phi.append(phi_)

        return np.array(phi)

    def _constrain(self, model: RhNeutrinoBase, **kwargs):
        r"""
        Determines the limit on :math:`<sigma v>` from gamma-ray data.

        We define a signal to be in conflict with the measured flux for bin
        :math:`i` for an experiment if

        .. math::

            \Phi_{\chi}^{(i)} > n_{\sigma} \sigma^{(i)} + \Phi^{(i)},

        where :math:`\Phi_\chi^{(i)}` is the integrated flux due to DM
        annihilations for the bin, :math:`\Phi^{(i)}` is the measured flux in
        the bin, :math:`\sigma^{(i)}` is size of the upper error bar for the
        bin and :math:`n_{\sigma} = 2` is the significance. The overall limit on
        :math:`\langle\sigma v\rangle` is computed by minimizing over the
        limits determined for each bin.

        Parameters
        ----------
        measurement : FluxMeasurement
            Information about the flux measurement and target.
        n_sigma : float
            See the notes for this function.

        Returns
        -------
        <sigma v>_tot : float
            Largest allowed thermally averaged total cross section in cm^3 / s
        """
        lower = np.array(self.__lower_energies)
        upper = np.array(self.__upper_energies)

        # Estimated differential flux from Fermi
        de = upper - lower
        phi_max = self.__diff_flux_upper_limits * de

        def energy_res(e):
            # Hazma takes MeV and return 1/MeV
            return grp.energy_res_fermi(e * 1e3) * 1e3

        phi = self._compute_fluxes(
            model,
            energy_res,
            fields.Photon,
            self.__npts,
            **kwargs,
        )

        lims = np.ones_like(phi) * np.inf
        mask = phi > 0.0
        if np.any(mask):
            lims[mask] = phi_max[mask] / phi[mask]

        # If annihilation, return Tau
        power = 1.0 if kwargs.get("mx") is not None else -1.0

        return np.min(lims) ** power


class RhNeutrinoConstrainer(CompositeConstrainer):
    def __init__(self):
        super().__init__()

    @property
    def description(self) -> str:
        return "[bold blue] RH-Neutrino"

    def _update(self, model: RhNeutrinoBase, mass):
        model.mass = mass

    def _model_iterator(self, masses: npt.NDArray[np.float64], theta: float, gen: Gen):
        def get_model(mass):
            if mass < 5.0:
                return RhNeutrinoMeV(mass, theta, gen)
            elif mass < 1e3:
                return RhNeutrinoGeV(mass, theta, gen)
            return RhNeutrinoTeV(mass, theta, gen)

        models = [get_model(mass) for mass in masses]
        return models

    def constrain(
        self,
        masses: npt.NDArray[np.float64],
        progress: Optional[Progress] = None,
        **options,
    ):
        """
        Compute the constraints on a RH-Neutrino dark matter model. The observations
        used to constrain are: COMPTEL, EGRET, Fermi, INTEGRAL and GECCO.

        Parameters
        ----------
        mxs: ArrayLike
            Array of dark-matter masses.
        existing: bool, optional
            If true, constraints from existing telescopes are computed. Default is True.
        gecco: bool, optional
            If true, constraints from GECCO are computed. Default is True.
        progress: rich.progress.Progress or None, optional
            Rich progress-bar to display progress.
        options
            Options to pass to various constrainers. These options are listed below.
        sigma: float, optional
            Discovery threshold for GECCO corresponding to a singal-to-noise ratio
            greater than `sigma`. Default is 5.0.
        lepton: str, optional
            Lepton flavor the RH-neutrino mixes with. Can be "e" or "mu".
            Default is "e".

        Returns
        -------
        constraints: Dict
            Dictionary containing all constraints.
        """
        self.reset_constrainers()

        theta = options.get("theta", 1e-3)
        gen = options.get("gen", Gen.Fst)
        mx = options.get("mx")
        model_iterator = self._model_iterator(masses, theta=theta, gen=gen)

        # self.add_constrainer(FermiDracoConstrainer())
        self.add_constrainer(EgretConstrainer())
        self.add_constrainer(ComptelConstrainer())
        self.add_constrainer(IntegralConstrainer())
        self.add_constrainer(FermiConstrainer())
        self.add_constrainer(HawcConstrainer())
        self.add_constrainer(IceCubeConstrainer())

        return self._constrain(model_iterator, progress=progress, mx=mx)

    def compute_fluxes(
        self,
        masses: npt.NDArray[np.float64],
        progress: Optional[Progress] = None,
        **options,
    ):
        """
        Compute the constraints on a RH-Neutrino dark matter model. The observations
        used to constrain are: COMPTEL, EGRET, Fermi, INTEGRAL and GECCO.

        Parameters
        ----------
        mxs: ArrayLike
            Array of dark-matter masses.
        existing: bool, optional
            If true, constraints from existing telescopes are computed. Default is True.
        gecco: bool, optional
            If true, constraints from GECCO are computed. Default is True.
        progress: rich.progress.Progress or None, optional
            Rich progress-bar to display progress.
        options
            Options to pass to various constrainers. These options are listed below.
        sigma: float, optional
            Discovery threshold for GECCO corresponding to a singal-to-noise ratio
            greater than `sigma`. Default is 5.0.
        lepton: str, optional
            Lepton flavor the RH-neutrino mixes with. Can be "e" or "mu".
            Default is "e".

        Returns
        -------
        constraints: Dict
            Dictionary containing all constraints.
        """
        self.reset_constrainers()

        theta = options.get("theta", 1e-3)
        gen = options.get("gen", Gen.Fst)
        mx = options.get("mx")
        model_iterator = self._model_iterator(masses, theta=theta, gen=gen)

        self.add_constrainer(FermiDracoConstrainer())
        self.add_constrainer(EgretConstrainer())
        self.add_constrainer(ComptelConstrainer())
        self.add_constrainer(IntegralConstrainer())
        self.add_constrainer(FermiConstrainer())
        self.add_constrainer(HawcConstrainer())
        self.add_constrainer(IceCubeConstrainer())

        def fn(constrainer, *args):
            return constrainer.compute_fluxes(*args, mx=mx)

        return self._iterate(fn, model_iterator, progress=progress)
