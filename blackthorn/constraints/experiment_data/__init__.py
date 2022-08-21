import pathlib
import json
from typing import NamedTuple, List, Dict, Union
import enum
from dataclasses import dataclass

import jsonschema


THIS_DIR = pathlib.Path(__file__).parent.absolute()


class FermiDwarfTarget(enum.IntEnum):
    BootesI = 0
    BootesII = 1
    BootesIII = 2
    CanesVenaticiI = 3
    CanesVenaticiII = 4
    CanisMajor = 5
    Carina = 6
    ComaBerenices = 7
    Draco = 8
    Fornax = 9
    Hercules = 10
    LeoI = 11
    LeoII = 12
    LeoIV = 13
    LeoV = 14
    PiscesII = 15
    Sagittarius = 16
    Sculptor = 17
    Segue1 = 18
    Segue2 = 19
    Sextans = 20
    UrsaMajor_I = 21
    UrsaMajor_II = 22
    UrsaMinor = 23
    Willman1 = 24


_FERMI_DWARF_TARGET_NAMES = [
    "bootes_I",
    "bootes_II",
    "bootes_III",
    "canes_venatici_I",
    "canes_venatici_II",
    "canis_major",
    "carina",
    "coma_berenices",
    "draco",
    "fornax",
    "hercules",
    "leo_I",
    "leo_II",
    "leo_IV",
    "leo_V",
    "pisces_II",
    "sagittarius",
    "sculptor",
    "segue_1",
    "segue_2",
    "sextans",
    "ursa_major_I",
    "ursa_major_II",
    "ursa_minor",
    "willman_1",
]


fermi_observation_schema = {
    "$schema": "http://json-schema.org/schema",
    "title": "Schema for Fermi observation data of Dwarf Spheroidal Galaxies",
    "type": "object",
    "properties": {
        "galaxyName": {"type": "string", "description": "Name of the galaxy"},
        "id": {"type": "string", "description": "Identifier of the galaxy"},
        "galacticLongitude": {
            "type": "number",
            "description": "Longitude of the galaxy",
            "units": ["deg"],
        },
        "galacticLatitude": {
            "type": "number",
            "description": "Latitude of the galaxy",
            "units": ["deg"],
        },
        "angularSize": {
            "type": "number",
            "description": "Angular size of observation area",
            "units": ["sr"],
        },
        "distance": {
            "type": "number",
            "description": "Distance from Earth to the target",
            "units": ["kpc"],
        },
        "observedJFactor": {
            "type": "object",
            "description": "Observed value of the J-factor",
            "properties": {
                "mean": {"type": "number", "description": "Mean of the observation"},
                "uncertainty": {
                    "type": "number",
                    "description": "Uncertainty of the observation",
                },
            },
        },
        "data": {
            "type": "array",
            "description": "Upper limits and likelihoods from the Fermi collaboration",
            "items": {
                "type": "object",
                "properties": {
                    "lowerEnergy": {
                        "type": "number",
                        "description": "Lower energy value of the bin",
                        "units": "MeV",
                    },
                    "upperEnergy": {
                        "type": "number",
                        "description": "Upper energy value of the bin",
                        "units": "MeV",
                    },
                    "fluxUpperLimit": {
                        "type": "number",
                        "description": "Upper limit on the total flux",
                        "units": "MeV cm^-2 s^-1",
                    },
                    "differentialFluxUpperLimit": {
                        "type": "number",
                        "description": "Upper limit on the differential flux",
                        "units": "MeV cm^-2 s^-1",
                    },
                    "fluxLikelihoods": {
                        "type": "array",
                        "description": "List of the fluxes and"
                        " their corresponding likelihoods",
                        "items": {
                            "type": "object",
                            "properties": {
                                "flux": {
                                    "type": "number",
                                    "description": "Value of the flux",
                                    "units": "MeV cm^-2 s^-1",
                                },
                                "deltaLogLikelihood": {
                                    "type": "number",
                                    "description": "Log-likelihood of the flux",
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


class JFactorObservation(NamedTuple):
    """Observed value of the J-factor.

    Attributes
    ----------
    mean: float
        Mean value of the energy flux in units of [MeV cm^-2 s^-1].
    uncertainty: float
        Uncertainty of the energy flux in units of [MeV cm^-2 s^-1].
    """

    # Mean of the observation
    mean: float
    # Uncertainty of the observation
    uncertainty: float

    energy_units: str

    @classmethod
    def from_dict(cls, entry: Dict) -> "JFactorObservation":
        mean = entry["mean"]
        uncertainty = entry["uncertainty"]
        return cls(mean=mean, uncertainty=uncertainty, energy_units="MeV")

    def in_gev(self) -> "JFactorObservation":
        """Generate a new instance with energy units in GeV."""
        if self.energy_units == "GeV":
            return self

        return type(self)(
            mean=self.mean * 1e-3,
            uncertainty=self.uncertainty * 1e-3,
            energy_units="GeV",
        )


class FermiDwarfObservationLikelihood(NamedTuple):
    """Flux likelihood measurement.

    Attributes
    ----------
    flux: float
        Flux in units of [MeV cm^-2 s^-1].
    delta_log_likelihood: float
        Relative log-likelihood of the flux.
    """

    flux: float
    delta_log_likelihood: float
    energy_units: str

    @classmethod
    def from_dict(cls, item: Dict) -> "FermiDwarfObservationLikelihood":
        flux = item["flux"]
        delta_log_likelihood = item["deltaLogLikelihood"]
        return cls(
            flux=flux, delta_log_likelihood=delta_log_likelihood, energy_units="MeV"
        )

    @classmethod
    def from_dict_list(
        cls, entries: List[Dict]
    ) -> List["FermiDwarfObservationLikelihood"]:
        return [FermiDwarfObservationLikelihood.from_dict(entry) for entry in entries]

    def in_gev(self) -> "FermiDwarfObservationLikelihood":
        """Generate a new instance with energy units in GeV."""
        if self.energy_units == "GeV":
            return self
        return type(self)(
            flux=self.flux * 1e-3,
            delta_log_likelihood=self.delta_log_likelihood,
            energy_units="GeV",
        )


@dataclass
class FermiDwarfObservationEntry:
    """Bin measurement from Fermi.

    Attributes
    ----------
    lower_energy: float
        Lower energy value of the bin.
    upper_energy: float
        Upper energy value of the bin.
    flux_upper_limit: float
        Upper limit on the total flux in units of [MeV cm^-2 s^-1].
    differential_flux_upper_limit: float
        Upper limit on the differential flux in units of [MeV cm^-2 s^-1].
    likelihood: List[FermiDwarfObservationLikelihood]
        List of the fluxes and their corresponding likelihoods.
    """

    lower_energy: float
    upper_energy: float
    flux_upper_limit: float
    differential_flux_upper_limit: float
    likelihood: List[FermiDwarfObservationLikelihood]
    energy_units: str

    @classmethod
    def from_dict(cls, entry: Dict) -> "FermiDwarfObservationEntry":
        lower_energy = entry["lowerEnergy"]
        upper_energy = entry["upperEnergy"]
        flux_upper_limit = entry["fluxUpperLimit"]
        differential_flux_upper_limit = entry["differentialFluxUpperLimit"]
        likelihood = FermiDwarfObservationLikelihood.from_dict_list(
            entry["fluxLikelihoods"]
        )

        return cls(
            lower_energy=lower_energy,
            upper_energy=upper_energy,
            flux_upper_limit=flux_upper_limit,
            differential_flux_upper_limit=differential_flux_upper_limit,
            likelihood=likelihood,
            energy_units="MeV",
        )

    @classmethod
    def from_dict_list(cls, entries: List[Dict]) -> List["FermiDwarfObservationEntry"]:
        return [FermiDwarfObservationEntry.from_dict(entry) for entry in entries]

    def in_gev(self) -> "FermiDwarfObservationEntry":
        """Generate a new instance with energy units in GeV."""
        if self.energy_units == "GeV":
            return self

        return type(self)(
            lower_energy=self.lower_energy * 1e-3,
            upper_energy=self.upper_energy * 1e-3,
            flux_upper_limit=self.flux_upper_limit * 1e-3,
            differential_flux_upper_limit=self.differential_flux_upper_limit * 1e-3,
            likelihood=[lh.in_gev() for lh in self.likelihood],
            energy_units="GeV",
        )


@dataclass
class FermiDwarfObservation:
    """Full measurement provided by the Fermi collaboration.

    Attributes
    ----------
    galaxy_name: str
        Name of the dwarf spheroidal galaxy.
    galactic_longitude: float
        Galactic longitude of the galaxy [deg].
    galactic_latitude: float
        Galactic latitude of the galaxy [deg].
    angular_size: float
        Angular size of observation area [sr].
    distance: float
        Distance from Earth to the target [kpc].
    observed_jfactor: JFactorObservation
        Observed value of the J-factor
    data: List[FermiDwarfObservationEntry]
        Upper limits and likelihoods from the Fermi collaboration

    """

    galaxy_name: str
    galactic_longitude: float
    galactic_latitude: float
    angular_size: float
    distance: float
    observed_jfactor: JFactorObservation
    data: List[FermiDwarfObservationEntry]
    energy_units: str

    @classmethod
    def _from_json_file(cls, file: pathlib.Path) -> "FermiDwarfObservation":
        with open(file, "r") as f:
            json_data = json.load(f)

        jsonschema.validate(json_data, fermi_observation_schema)

        galaxy_name = json_data["galaxyName"]
        galactic_longitude = json_data["galacticLongitude"]
        galactic_latitude = json_data["galacticLatitude"]
        angular_size = json_data["angularSize"]
        distance = json_data["distance"]
        observed_jfactor = JFactorObservation.from_dict(json_data["observedJFactor"])
        data = FermiDwarfObservationEntry.from_dict_list(json_data["data"])

        return cls(
            galaxy_name=galaxy_name,
            galactic_longitude=galactic_longitude,
            galactic_latitude=galactic_latitude,
            angular_size=angular_size,
            distance=distance,
            observed_jfactor=observed_jfactor,
            data=data,
            energy_units="MeV",
        )

    @classmethod
    def load_target_data(
        cls, target: FermiDwarfTarget, energy_units: str = "GeV"
    ) -> "FermiDwarfObservation":
        """Load the data associated with the target from local file data."""
        name = _FERMI_DWARF_TARGET_NAMES[target]
        file = THIS_DIR.joinpath(name).with_suffix(".json")
        data = cls._from_json_file(file)

        if energy_units == "GeV":
            return data.in_gev()

        return data

    @classmethod
    def from_json_file(cls, file: Union[str, pathlib.Path]) -> "FermiDwarfObservation":
        """Load the data from the given file."""
        file = pathlib.Path(file).absolute()
        return cls._from_json_file(file)

    def in_gev(self) -> "FermiDwarfObservation":
        """Convert energy units to GeV."""
        if self.energy_units == "GeV":
            return self

        return type(self)(
            galaxy_name=self.galaxy_name,
            galactic_longitude=self.galactic_longitude,
            galactic_latitude=self.galactic_latitude,
            angular_size=self.angular_size,
            distance=self.distance,
            observed_jfactor=self.observed_jfactor.in_gev(),
            data=[datum.in_gev() for datum in self.data],
            energy_units="GeV",
        )
