from typing import ClassVar, List, Tuple

from typing import overload
import numpy
Fst: Gen
Null: Gen
Snd: Gen
Trd: Gen

class Gen:
    __doc__: ClassVar[str] = ...  # read-only
    __members__: ClassVar[dict] = ...  # read-only
    Fst: ClassVar[Gen] = ...
    Null: ClassVar[Gen] = ...
    Snd: ClassVar[Gen] = ...
    Trd: ClassVar[Gen] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class RhNeutrinoGeVCpp:
    gen: Gen
    mass: float
    theta: float
    def __init__(self, arg0: float, arg1: float, arg2: Gen) -> None: ...
    def dndx_l_u_d(self, genu: Gen, gend: Gen, xmin: float = ..., xmax: float = ..., nbins: int = ..., nevents: int = ...) -> Tuple[List[float],List[float]]: ...
    def dndx_l_w(self, xmin: float = ..., xmax: float = ..., nbins: int = ..., nevents: int = ...) -> Tuple[List[float],List[float]]: ...
    def dndx_v_d_d(self, genu: Gen, xmin: float = ..., xmax: float = ..., nbins: int = ..., nevents: int = ...) -> Tuple[List[float],List[float]]: ...
    def dndx_v_h(self, xmin: float = ..., xmax: float = ..., nbins: int = ..., nevents: int = ...) -> Tuple[List[float],List[float]]: ...
    def dndx_v_l_l(self, genv: Gen, genl1: Gen, genl2: Gen, xmin: float = ..., xmax: float = ..., nbins: int = ..., nevents: int = ...) -> Tuple[List[float],List[float]]: ...
    def dndx_v_u_u(self, genu: Gen, xmin: float = ..., xmax: float = ..., nbins: int = ..., nevents: int = ...) -> Tuple[List[float],List[float]]: ...
    def dndx_v_z(self, xmin: float = ..., xmax: float = ..., nbins: int = ..., nevents: int = ...) -> Tuple[List[float],List[float]]: ...
    def inv_mass_distribution_v_d_d(self, gend: Gen, nbins: int, nevents: int) -> Tuple[List[float],List[float]]: ...
    def inv_mass_distribution_v_l_l(self, genv: Gen, genl1: Gen, genl2: Gen, nbins: int, nevents: int) -> Tuple[List[float],List[float]]: ...
    def inv_mass_distribution_v_u_u(self, genu: Gen, nbins: int, nevents: int) -> Tuple[List[float],List[float]]: ...
    def inv_mass_distributions_l_u_d(self, genu: Gen, gend: Gen, nbins: int, nevents: int) -> Tuple[List[float],List[float],List[float],List[float]]: ...
    def l_u_d_energy_distributions(self, genu: Gen, gend: Gen, nbins: List[int[3]], nevents: int) -> List[Tuple[List[float],List[float]]]: ...
    def v_d_d_energy_distributions(self, gend: Gen, nbins: List[int[3]], nevents: int) -> List[Tuple[List[float],List[float]]]: ...
    def v_l_l_energy_distributions(self, genv: Gen, genl1: Gen, genl2: Gen, nbins: List[int[3]], nevents: int) -> List[Tuple[List[float],List[float]]]: ...
    def v_u_u_energy_distributions(self, genu: Gen, nbins: List[int[3]], nevents: int) -> List[Tuple[List[float],List[float]]]: ...
    def width_l_u_d(self, genu: Gen, gend: Gen, nevents: int = ..., batchsize: int = ...) -> Tuple[float,float]: ...
    def width_l_w(self) -> float: ...
    def width_v_d_d(self, gend: Gen, nevents: int = ..., batchsize: int = ...) -> Tuple[float,float]: ...
    def width_v_h(self) -> float: ...
    def width_v_l_l(self, genv: Gen, genl1: Gen, genl2: Gen, nevents: int = ..., batchsize: int = ...) -> Tuple[float,float]: ...
    def width_v_u_u(self, genu: Gen, nevents: int = ..., batchsize: int = ...) -> Tuple[float,float]: ...
    @overload
    def width_v_v_v(self, genv1: Gen, genv2: Gen, genv3: Gen, nevents: int = ..., batchsize: int = ...) -> Tuple[float,float]: ...
    @overload
    def width_v_v_v(self, nevents: int = ..., batchsize: int = ...) -> Tuple[float,float]: ...
    def width_v_z(self) -> float: ...

class RhNeutrinoMeVCpp:
    gen: Gen
    mass: float
    theta: float
    def __init__(self, arg0: float, arg1: float, arg2: Gen) -> None: ...
    @overload
    def dndx_neutrino_l_k(self, x: numpy.ndarray[numpy.float64], beta: float) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_neutrino_l_k(self, x: float, beta: float) -> List[float[3]]: ...
    @overload
    def dndx_neutrino_l_pi(self, x: numpy.ndarray[numpy.float64], beta: float) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_neutrino_l_pi(self, x: float, beta: float) -> List[float[3]]: ...
    @overload
    def dndx_neutrino_l_pi_pi0(self, x: numpy.ndarray[numpy.float64], beta: float) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_neutrino_l_pi_pi0(self, x: float, beta: float) -> List[float[3]]: ...
    @overload
    def dndx_neutrino_v_l_l(self, x: numpy.ndarray[numpy.float64], beta: float, gv: Gen, gl1: Gen, gl2: Gen) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_neutrino_v_l_l(self, x: float, beta: float, gv: Gen, gl1: Gen, gl2: Gen) -> List[float[3]]: ...
    @overload
    def dndx_neutrino_v_pi0(self, x: numpy.ndarray[numpy.float64], beta: float) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_neutrino_v_pi0(self, x: float, beta: float) -> List[float[3]]: ...
    @overload
    def dndx_neutrino_v_pi_pi(self, x: numpy.ndarray[numpy.float64], beta: float) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_neutrino_v_pi_pi(self, x: float, beta: float) -> List[float[3]]: ...
    @overload
    def dndx_photon_l_k(self, x: numpy.ndarray[numpy.float64], beta: float) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_photon_l_k(self, x: float, beta: float) -> float: ...
    @overload
    def dndx_photon_l_pi(self, x: numpy.ndarray[numpy.float64], beta: float) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_photon_l_pi(self, x: float, beta: float) -> float: ...
    @overload
    def dndx_photon_l_pi_pi0(self, x: numpy.ndarray[numpy.float64], beta: float) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_photon_l_pi_pi0(self, x: float, beta: float) -> float: ...
    @overload
    def dndx_photon_v_l_l(self, x: numpy.ndarray[numpy.float64], beta: float, gv: Gen, gl1: Gen, gl2: Gen) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_photon_v_l_l(self, x: float, beta: float, gv: Gen, gl1: Gen, gl2: Gen) -> float: ...
    @overload
    def dndx_photon_v_pi0(self, x: numpy.ndarray[numpy.float64], beta: float) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_photon_v_pi0(self, x: float, beta: float) -> float: ...
    @overload
    def dndx_photon_v_pi_pi(self, x: numpy.ndarray[numpy.float64], beta: float) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_photon_v_pi_pi(self, x: float, beta: float) -> float: ...
    @overload
    def dndx_positron_l_k(self, x: numpy.ndarray[numpy.float64], beta: float) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_positron_l_k(self, x: float, beta: float) -> float: ...
    @overload
    def dndx_positron_l_pi(self, x: numpy.ndarray[numpy.float64], beta: float) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_positron_l_pi(self, x: float, beta: float) -> float: ...
    @overload
    def dndx_positron_l_pi_pi0(self, x: numpy.ndarray[numpy.float64], beta: float) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_positron_l_pi_pi0(self, x: float, beta: float) -> float: ...
    @overload
    def dndx_positron_v_l_l(self, x: numpy.ndarray[numpy.float64], beta: float, gv: Gen, gl1: Gen, gl2: Gen) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_positron_v_l_l(self, x: float, beta: float, gv: Gen, gl1: Gen, gl2: Gen) -> float: ...
    @overload
    def dndx_positron_v_pi0(self, x: numpy.ndarray[numpy.float64], beta: float) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_positron_v_pi0(self, x: float, beta: float) -> float: ...
    @overload
    def dndx_positron_v_pi_pi(self, x: numpy.ndarray[numpy.float64], beta: float) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def dndx_positron_v_pi_pi(self, x: float, beta: float) -> float: ...
    def width_l_k(self) -> float: ...
    def width_l_pi(self) -> float: ...
    def width_l_pi_pi0(self, nevents: int = ..., batchsize: int = ...) -> Tuple[float,float]: ...
    def width_v_a(self) -> float: ...
    def width_v_eta(self) -> float: ...
    def width_v_l_l(self, genv: Gen, genl1: Gen, genl2: Gen, nevents: int = ..., batchsize: int = ...) -> Tuple[float,float]: ...
    def width_v_pi0(self) -> float: ...
    def width_v_pi_pi(self, nevents: int = ..., batchsize: int = ...) -> Tuple[float,float]: ...
    @overload
    def width_v_v_v(self, genv1: Gen, genv2: Gen, genv3: Gen, nevents: int = ..., batchsize: int = ...) -> Tuple[float,float]: ...
    @overload
    def width_v_v_v(self, nevents: int = ..., batchsize: int = ...) -> Tuple[float,float]: ...