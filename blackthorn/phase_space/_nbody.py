from typing import Sequence, List, Optional, Dict, Tuple

import numpy as np
from hazma.rambo import PhaseSpace

from ..fields import QuantumField

from ._proto import SquaredMatrixElement, EnergyDistribution


def energy_distributions_nbody_decay(
    cme,
    final_states: Sequence[QuantumField],
    nbins: int,
    msqrd: Optional[SquaredMatrixElement],
    k: int = 1,
    npts: int = 1 << 15,
) -> List[EnergyDistribution]:
    assert (
        len(final_states) > 3
    ), f"`final_states` must have more than 3 fields. Found {len(final_states)}"

    if msqrd is not None:

        def msqrd_(momenta):  # type: ignore
            return msqrd(momenta)

    else:

        def msqrd_(_):
            return 1.0

    masses = np.array([f.mass for f in final_states])
    phase_space = PhaseSpace(cme=cme, masses=masses, msqrd=msqrd_)
    dists = phase_space.energy_distributions(n=npts, nbins=nbins)

    return [EnergyDistribution.from_data(es, ps, k=k) for ps, es in dists]


def invariant_mass_distributions_nbody_decay(
    cme,
    final_states: Sequence[QuantumField],
    nbins: int,
    msqrd: Optional[SquaredMatrixElement],
    k: int = 1,
    npts: int = 1 << 15,
) -> Dict[Tuple[int, int], EnergyDistribution]:
    assert (
        len(final_states) > 3
    ), f"`final_states` must have more than 3 fields. Found {len(final_states)}"

    if msqrd is not None:

        def msqrd_(momenta):  # type: ignore
            return msqrd(momenta)

    else:

        def msqrd_(_):
            return 1.0

    masses = np.array([f.mass for f in final_states])
    phase_space = PhaseSpace(cme=cme, masses=masses, msqrd=msqrd_)
    dists = phase_space.invariant_mass_distributions(n=npts, nbins=nbins)

    return {
        key: EnergyDistribution.from_data(es, ps, k=k)
        for key, (ps, es) in dists.items()
    }
