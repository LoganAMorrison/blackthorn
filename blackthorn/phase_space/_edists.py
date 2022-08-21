from typing import Sequence, Optional

from ..fields import QuantumField

from ._proto import SquaredMatrixElement
from ._three_body import energy_distributions_three_body_decay
from ._nbody import energy_distributions_nbody_decay


def energy_distributions_decay(
    cme,
    final_states: Sequence[QuantumField],
    nbins: int,
    msqrd: Optional[SquaredMatrixElement],
    k: int = 1,
    **kwargs
):
    if len(final_states) == 2:
        pass
    elif len(final_states) == 3:
        return energy_distributions_three_body_decay(cme, final_states, nbins, msqrd, k)
    else:
        npts = kwargs.get("npts", 1 << 15)
        return energy_distributions_nbody_decay(
            cme, final_states, nbins, msqrd, k, npts=npts
        )


def invariant_mass_distributions_decay(
    cme,
    final_states: Sequence[QuantumField],
    nbins: int,
    msqrd: Optional[SquaredMatrixElement],
    k: int = 1,
    **kwargs
):
    if len(final_states) == 2:
        pass
    elif len(final_states) == 3:
        return energy_distributions_three_body_decay(cme, final_states, nbins, msqrd, k)
    else:
        npts = kwargs.get("npts", 1 << 15)
        return energy_distributions_nbody_decay(
            cme, final_states, nbins, msqrd, k, npts=npts
        )
