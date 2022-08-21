#include "RhNeutrinoMeV.h"

namespace blackthorn::py_interface::mev {

using RhnClassType = py::class_<RhNeutrinoMeV>;

static auto add_constructor(RhnClassType *rhn) -> void;

auto config_class(py::class_<RhNeutrinoMeV> *rhn) -> void {
  rhn->def(py::init<double, double, Gen>(), "");

  // ---- Accessors ----

  rhn->def_property(
      "mass", [](const RhNeutrinoMeV &model) { return model.mass(); },
      [](RhNeutrinoMeV &model) { return model.mass(); });
  rhn->def_property(
      "theta", [](const RhNeutrinoMeV &model) { return model.theta(); },
      [](RhNeutrinoMeV &model) { return model.theta(); });
  rhn->def_property(
      "gen", [](const RhNeutrinoMeV &model) { return model.gen(); },
      [](RhNeutrinoMeV &model) { return model.gen(); });

  // ---- Widths ----
  add_width_l_pi(rhn);
  add_width_l_k(rhn);
  add_width_v_a(rhn);
  add_width_v_pi0(rhn);
  add_width_v_eta(rhn);
  add_width_v_l_l(rhn);
  add_width_v_v_v(rhn);
  add_width_v_pi_pi(rhn);
  add_width_l_pi_pi0(rhn);

  // ---- dndx_photon ----
  add_dndx_photon_l_pi(rhn);
  add_dndx_photon_l_k(rhn);
  add_dndx_photon_v_pi0(rhn);
  // add_dndx_photon_v_eta(rhn);
  add_dndx_photon_v_l_l(rhn);
  add_dndx_photon_v_pi_pi(rhn);
  add_dndx_photon_l_pi_pi0(rhn);

  // ---- dndx_neutrino ----
  add_dndx_positron_l_pi(rhn);
  add_dndx_positron_l_k(rhn);
  add_dndx_positron_v_pi0(rhn);
  // add_dndx_photon_v_eta(rhn);
  add_dndx_positron_v_l_l(rhn);
  add_dndx_positron_v_pi_pi(rhn);
  add_dndx_positron_l_pi_pi0(rhn);

  // ---- dndx_neutrino ----
  add_dndx_neutrino_l_pi(rhn);
  add_dndx_neutrino_l_k(rhn);
  add_dndx_neutrino_v_pi0(rhn);
  // add_dndx_photon_v_eta(rhn);
  add_dndx_neutrino_v_l_l(rhn);
  add_dndx_neutrino_v_pi_pi(rhn);
  add_dndx_neutrino_l_pi_pi0(rhn);
}

auto add_constructor(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Create a RhNeutrinoMeV object.

Parameters
----------
mass: float
  Mass of the right-handed neutrino.
theta: float
  Mixing angle between right-handed and active neutrino.
gen: Gen
  Generation of active neutrino the right-handed neturino mixes with.
)Doc";
  rhn->def(py::init<double, double, Gen>(), doc.c_str());
}

} // namespace blackthorn::py_interface::mev
