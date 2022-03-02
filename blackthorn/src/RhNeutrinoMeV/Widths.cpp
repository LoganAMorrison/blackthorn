#include "RhNeutrinoMeV.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace blackthorn::py_interface::mev {

auto add_width_l_pi(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the partial width for N -> l + pi.
)Doc";
  rhn->def("width_l_pi", &RhNeutrinoMeV::width_l_pi, doc.c_str());
}

auto add_width_l_k(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the partial width for N -> l + K.
)Doc";
  rhn->def("width_l_k", &RhNeutrinoMeV::width_l_k, doc.c_str());
}

auto add_width_v_a(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the partial width for N -> nu + photon.
)Doc";
  rhn->def("width_v_a", &RhNeutrinoMeV::width_v_a, doc.c_str());
}

auto add_width_v_pi0(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the partial width for N -> nu + pi0.
)Doc";
  rhn->def("width_v_pi0", &RhNeutrinoMeV::width_v_pi0, doc.c_str());
}

auto add_width_v_eta(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the partial width for N -> nu + eta.
)Doc";
  rhn->def("width_v_eta", &RhNeutrinoMeV::width_v_eta, doc.c_str());
}

auto add_width_v_pi_pi(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the partial width for N -> nu + pi + pi.

Parameters
----------
nevents: int, optional
  Number of events used to estimate width.
batchsize: int, optional
  Size of batches used to estimate widths.
)Doc";
  rhn->def("width_v_pi_pi", &RhNeutrinoMeV::width_v_pi_pi, doc.c_str(),
           py::arg("nevents") = RhNeutrinoMeV::DEFAULT_NEVENTS,
           py::arg("batchsize") = RhNeutrinoMeV::DEFAULT_BATCHSIZE);
}

auto add_width_l_pi_pi0(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the partial width for N -> l + pi + pi0.

Parameters
----------
nevents: int, optional
  Number of events used to estimate width.
batchsize: int, optional
  Size of batches used to estimate widths.
)Doc";
  rhn->def("width_l_pi_pi0", &RhNeutrinoMeV::width_l_pi_pi0, doc.c_str(),
           py::arg("nevents") = RhNeutrinoMeV::DEFAULT_NEVENTS,
           py::arg("batchsize") = RhNeutrinoMeV::DEFAULT_BATCHSIZE);
}

auto add_width_v_l_l(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the partial width for N -> nu + l + l.

Parameters
----------
genv: Gen
  Generation of the final-state neutrino.
genl1: Gen
  Generation of the first final-state lepton.
genl2: Gen
  Generation of the second final-state lepton.
nevents: int, optional
  Number of events used to estimate width.
batchsize: int, optional
  Size of batches used to estimate widths.
)Doc";
  rhn->def("width_v_l_l", &RhNeutrinoMeV::width_v_l_l, doc.c_str(),
           py::arg("genv"), py::arg("genl1"), py::arg("genl2"),
           py::arg("nevents") = RhNeutrinoMeV::DEFAULT_NEVENTS,
           py::arg("batchsize") = RhNeutrinoMeV::DEFAULT_BATCHSIZE);
}

auto add_width_v_v_v(RhnClassType *rhn) -> void {
  static std::string doc1 = R"Doc(
Compute the partial width for N -> nu + nu + nu.

Parameters
----------
genv1: Gen
  Generation of the 1st final-state neutrino.
genv2: Gen
  Generation of the 2nd final-state neutrino.
genv3: Gen
  Generation of the 3rd final-state neutrino.
nevents: int, optional
  Number of events used to estimate width.
batchsize: int, optional
  Size of batches used to estimate widths.
)Doc";

  rhn->def(
      "width_v_v_v",
      [](const RhNeutrinoMeV &model, Gen g1, Gen g2, Gen g3,
         size_t nevents = RhNeutrinoMeV::DEFAULT_NEVENTS,
         size_t batchsize = RhNeutrinoMeV::DEFAULT_BATCHSIZE) {
        return model.width_v_v_v(g1, g2, g3, nevents, batchsize);
      },
      doc1.c_str(), py::arg("genv1"), py::arg("genv2"), py::arg("genv3"),
      py::arg("nevents") = RhNeutrinoMeV::DEFAULT_NEVENTS,
      py::arg("batchsize") = RhNeutrinoMeV::DEFAULT_BATCHSIZE);

  static std::string doc2 = R"Doc(
Compute the partial width for N -> nu + nu + nu, summing over all final-state
neutrino permutations.

Parameters
----------
nevents: int, optional
  Number of events used to estimate width.
batchsize: int, optional
  Size of batches used to estimate widths.
)Doc";

  rhn->def(
      "width_v_v_v",
      [](const RhNeutrinoMeV &model,
         size_t nevents = RhNeutrinoMeV::DEFAULT_NEVENTS,
         size_t batchsize = RhNeutrinoMeV::DEFAULT_BATCHSIZE) {
        return model.width_v_v_v(nevents, batchsize);
      },
      doc2.c_str(), py::arg("nevents") = RhNeutrinoMeV::DEFAULT_NEVENTS,
      py::arg("batchsize") = RhNeutrinoMeV::DEFAULT_BATCHSIZE);
}
} // namespace blackthorn::py_interface::mev
