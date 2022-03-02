#include "blackthorn/Models/RhNeutrino.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace blackthorn::py_interface::mev {

namespace py = pybind11;

using RhnClassType = py::class_<RhNeutrinoMeV>;

auto config_class(RhnClassType *rhn) -> void;

auto add_width_l_pi(RhnClassType *rhn) -> void;
auto add_width_l_k(RhnClassType *rhn) -> void;
auto add_width_v_a(RhnClassType *rhn) -> void;
auto add_width_v_pi0(RhnClassType *rhn) -> void;
auto add_width_v_eta(RhnClassType *rhn) -> void;
auto add_width_v_l_l(RhnClassType *rhn) -> void;
auto add_width_v_v_v(RhnClassType *rhn) -> void;
auto add_width_v_pi_pi(RhnClassType *rhn) -> void;
auto add_width_l_pi_pi0(RhnClassType *rhn) -> void;

auto add_dndx_photon_l_pi(RhnClassType *rhn) -> void;
auto add_dndx_photon_l_k(RhnClassType *rhn) -> void;
auto add_dndx_photon_v_pi0(RhnClassType *rhn) -> void;
// auto add_dndx_v_eta(RhnClassType *rhn) -> void;
auto add_dndx_photon_v_l_l(RhnClassType *rhn) -> void;
auto add_dndx_photon_v_pi_pi(RhnClassType *rhn) -> void;
auto add_dndx_photon_l_pi_pi0(RhnClassType *rhn) -> void;

auto add_dndx_positron_l_pi(RhnClassType *rhn) -> void;
auto add_dndx_positron_l_k(RhnClassType *rhn) -> void;
auto add_dndx_positron_v_pi0(RhnClassType *rhn) -> void;
// auto add_dndx_v_eta(RhnClassType *rhn) -> void;
auto add_dndx_positron_v_l_l(RhnClassType *rhn) -> void;
auto add_dndx_positron_v_pi_pi(RhnClassType *rhn) -> void;
auto add_dndx_positron_l_pi_pi0(RhnClassType *rhn) -> void;

auto add_dndx_neutrino_l_pi(RhnClassType *rhn) -> void;
auto add_dndx_neutrino_l_k(RhnClassType *rhn) -> void;
auto add_dndx_neutrino_v_pi0(RhnClassType *rhn) -> void;
// auto add_dndx_v_eta(RhnClassType *rhn) -> void;
auto add_dndx_neutrino_v_l_l(RhnClassType *rhn) -> void;
auto add_dndx_neutrino_v_pi_pi(RhnClassType *rhn) -> void;
auto add_dndx_neutrino_l_pi_pi0(RhnClassType *rhn) -> void;

} // namespace blackthorn::py_interface::mev
