#include "blackthorn/Models/RhNeutrino.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace blackthorn::py_interface::gev {

namespace py = pybind11;

using RhnClassType = py::class_<RhNeutrinoGeV>;

auto config_class(RhnClassType *rhn) -> void;

auto add_constructor(RhnClassType *rhn) -> void;
auto add_width_v_h(RhnClassType *rhn) -> void;
auto add_width_v_z(RhnClassType *rhn) -> void;
auto add_width_l_w(RhnClassType *rhn) -> void;
auto add_width_v_u_u(RhnClassType *rhn) -> void;
auto add_width_v_d_d(RhnClassType *rhn) -> void;
auto add_width_l_u_d(RhnClassType *rhn) -> void;
auto add_width_v_l_l(RhnClassType *rhn) -> void;
auto add_width_v_v_v(RhnClassType *rhn) -> void;
auto add_dndx_v_h(RhnClassType *rhn) -> void;
auto add_dndx_v_z(RhnClassType *rhn) -> void;
auto add_dndx_l_w(RhnClassType *rhn) -> void;
auto add_dndx_v_u_u(RhnClassType *rhn) -> void;
auto add_dndx_v_d_d(RhnClassType *rhn) -> void;
auto add_dndx_l_u_d(RhnClassType *rhn) -> void;
auto add_dndx_v_l_l(RhnClassType *rhn) -> void;

} // namespace blackthorn::py_interface::gev
