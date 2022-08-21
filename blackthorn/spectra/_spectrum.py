from typing import Union, Optional, Sequence

import numpy as np
import numpy.typing as npt

from .. import fields
from ._hazma import HazmaSpectra
from ._pppc import PPPC4DMIDSpectra
from ._hdmspectra import HDMSpectra

RealArray = npt.NDArray[np.float64]
RealOrRealArray = Union[npt.NDArray[np.float64], float]


class SpectrumGenerator:
    def __init__(self) -> None:
        self._generators = {
            "hazma": HazmaSpectra(),
            "pppc": PPPC4DMIDSpectra(),
            "hdmspectra": HDMSpectra(),
        }

    def _default_generator(self, energy):
        if energy < 1.0:
            return self._generators["hazma"]
        if energy < 1e3:
            return self._generators["pppc"]
        return self._generators["hdmspectra"]

    def _filter_kwargs(self, **_):
        pass

    def _energy_distributions(self, cme, final_states: Sequence[fields.QuantumField]):
        if len(final_states) == 2:
            m1 = final_states[0].mass
            m2 = final_states[1].mass
            e1 = (cme**2 + m1**2 - m2**2) / (2 * cme)
            e2 = (cme**2 - m1**2 + m2**2) / (2 * cme)
            return [e1, e2]
        elif len(final_states) == 3:
            pass

    def dndx(
        self,
        x,
        cme: float,
        final_states: Sequence[fields.QuantumField],
        product: fields.QuantumField,
        generator: Optional[str] = None,
        **kwargs,
    ):
        """Compute the spectrum from a given final state into a specified product.

        Parameters
        ----------
        x: array
            Values of x = 2 E_gamma / cme.
        energy: float
            Energy of the final state particle.
        final_state: QuantumField
            Final state generating the spectrum.
        product: QuantumField
            Product to generate spectrum of.

        Returns
        -------
        dndx: array
            Product spectrum.
        """

        if generator is not None:
            dndx = self._generators[generator]
        else:
            dndx = self._default_generator(energy)

        return dndx(x, energy, final_state.pdg, product.pdg, **kwargs)

    def dndx_photon(self, x, energy: float, final_state: fields.QuantumField, **kwargs):
        """Compute the spectrum from a given final state into a photon.

        Parameters
        ----------
        x: array
            Values of x = 2 E_gamma / cme.
        energy: float
            Energy of the final state particle.
        final_state: QuantumField
            Final state generating the spectrum.

        Returns
        -------
        dndx: array
            Photon spectrum.
        """
        return self.dndx(x, energy, final_state, fields.Photon, **kwargs)

    def dndx_positron(
        self, x, energy: float, final_state: fields.QuantumField, **kwargs
    ):
        """Compute the spectrum from a given final state into a positron.

        Parameters
        ----------
        x: array
            Values of x = 2 E_pos / cme.
        energy: float
            Energy of the final state particle.
        final_state: QuantumField
            Final state generating the spectrum.

        Returns
        -------
        dndx: array
            Positron spectrum.
        """
        return self.dndx(x, energy, final_state, fields.Positron, **kwargs)

    def dndx_neutrino(
        self, x, energy: float, final_state: fields.QuantumField, flavor: str, **kwargs
    ):
        """Compute the spectrum from a given final state into a neutrino.

        Parameters
        ----------
        x: array
            Values of x = 2 Enu / cme.
        energy: float
            Energy of the final state particle.
        final_state: QuantumField
            Final state generating the spectrum.
        flavor: str
            Flavor of neutrino.

        Returns
        -------
        dndx: array
            Neutrino spectrum.
        """
        if flavor == "e":
            f = fields.ElectronNeutrino
        elif flavor == "mu":
            f = fields.MuonNeutrino
        elif flavor == "tau":
            f = fields.TauNeutrino
        else:
            raise ValueError(f"Invalid neutrino flavor {flavor}.")

        return self.dndx(x, energy, final_state, f, **kwargs)
