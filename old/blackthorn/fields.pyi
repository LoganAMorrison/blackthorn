from typing import Any, ClassVar
from .rh_neutrino import Gen

class BottomQuark:
    gen: ClassVar[Gen] = ...  # read-only
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...

class ChargedKaon:
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...
    def dnde_photon(self, *args, **kwargs) -> Any: ...

class ChargedPion:
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...
    def dnde_photon(self, *args, **kwargs) -> Any: ...

class CharmQuark:
    gen: ClassVar[Gen] = ...  # read-only
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...

class DownQuark:
    gen: ClassVar[Gen] = ...  # read-only
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...

class Electron:
    gen: ClassVar[Gen] = ...  # read-only
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...

class ElectronNeutrino:
    gen: ClassVar[Gen] = ...  # read-only
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...

class Gluon:
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...

class Higgs:
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    vev: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...

class Muon:
    gen: ClassVar[Gen] = ...  # read-only
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...
    def dnde_photon(self, *args, **kwargs) -> Any: ...

class MuonNeutrino:
    gen: ClassVar[Gen] = ...  # read-only
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...

class NeutralPion:
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...
    def dnde_photon(self, *args, **kwargs) -> Any: ...

class Photon:
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...

class StrangeQuark:
    gen: ClassVar[Gen] = ...  # read-only
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...

class Tau:
    gen: ClassVar[Gen] = ...  # read-only
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...

class TauNeutrino:
    gen: ClassVar[Gen] = ...  # read-only
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...

class TopQuark:
    gen: ClassVar[Gen] = ...  # read-only
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...

class UpQuark:
    gen: ClassVar[Gen] = ...  # read-only
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...

class WBoson:
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...

class ZBoson:
    mass: ClassVar[float] = ...  # read-only
    pdg: ClassVar[int] = ...  # read-only
    def __init__(self) -> None: ...
