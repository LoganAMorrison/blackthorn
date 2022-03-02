#include "RhNeutrinoGeV.h"

namespace blackthorn::py_interface::gev {

auto add_width_v_h(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the partial width for N -> nu + H.
)Doc";
  rhn->def("width_v_h", &RhNeutrinoGeV::width_v_h, doc.c_str());
}

auto add_width_v_z(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the partial width for N -> nu + Z.
)Doc";
  rhn->def("width_v_z", &RhNeutrinoGeV::width_v_z, doc.c_str());
}

auto add_width_l_w(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the partial width for N -> l + W.
)Doc";
  rhn->def("width_l_w", &RhNeutrinoGeV::width_l_w, doc.c_str());
}

auto add_width_v_u_u(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the partial width for N -> nu + u + u.

Parameters
----------
genu: Gen
  Generation of the final-state up-quark.
nevents: int, optional
  Number of events used to estimate width.
batchsize: int, optional
  Size of batches used to estimate widths.
)Doc";
  rhn->def("width_v_u_u", &RhNeutrinoGeV::width_v_u_u, doc.c_str(),
           py::arg("genu"), py::arg("nevents") = RhNeutrinoGeV::DEFAULT_NEVENTS,
           py::arg("batchsize") = RhNeutrinoGeV::DEFAULT_BATCHSIZE);
}

auto add_width_v_d_d(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the partial width for N -> nu + d + d.

Parameters
----------
gend: Gen
  Generation of the final-state down-quark.
nevents: int, optional
  Number of events used to estimate width.
batchsize: int, optional
  Size of batches used to estimate widths.
)Doc";
  rhn->def("width_v_d_d", &RhNeutrinoGeV::width_v_d_d, doc.c_str(),
           py::arg("gend"), py::arg("nevents") = RhNeutrinoGeV::DEFAULT_NEVENTS,
           py::arg("batchsize") = RhNeutrinoGeV::DEFAULT_BATCHSIZE);
}

auto add_width_l_u_d(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the partial width for N -> l + u + d.

Parameters
----------
genu: Gen
  Generation of the final-state up-quark.
gend: Gen
  Generation of the final-state down-quark.
nevents: int, optional
  Number of events used to estimate width.
batchsize: int, optional
  Size of batches used to estimate widths.
)Doc";
  rhn->def("width_l_u_d", &RhNeutrinoGeV::width_l_u_d, doc.c_str(),
           py::arg("genu"), py::arg("gend"),
           py::arg("nevents") = RhNeutrinoGeV::DEFAULT_NEVENTS,
           py::arg("batchsize") = RhNeutrinoGeV::DEFAULT_BATCHSIZE);
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
  rhn->def("width_v_l_l", &RhNeutrinoGeV::width_v_l_l, doc.c_str(),
           py::arg("genv"), py::arg("genl1"), py::arg("genl2"),
           py::arg("nevents") = RhNeutrinoGeV::DEFAULT_NEVENTS,
           py::arg("batchsize") = RhNeutrinoGeV::DEFAULT_BATCHSIZE);
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
      [](const RhNeutrinoGeV &model, Gen g1, Gen g2, Gen g3,
         size_t nevents = RhNeutrinoGeV::DEFAULT_NEVENTS,
         size_t batchsize = RhNeutrinoGeV::DEFAULT_BATCHSIZE) {
        return model.width_v_v_v(g1, g2, g3, nevents, batchsize);
      },
      doc1.c_str(), py::arg("genv1"), py::arg("genv2"), py::arg("genv3"),
      py::arg("nevents") = RhNeutrinoGeV::DEFAULT_NEVENTS,
      py::arg("batchsize") = RhNeutrinoGeV::DEFAULT_BATCHSIZE);

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
      [](const RhNeutrinoGeV &model,
         size_t nevents = RhNeutrinoGeV::DEFAULT_NEVENTS,
         size_t batchsize = RhNeutrinoGeV::DEFAULT_BATCHSIZE) {
        return model.width_v_v_v(nevents, batchsize);
      },
      doc2.c_str(), py::arg("nevents") = RhNeutrinoGeV::DEFAULT_NEVENTS,
      py::arg("batchsize") = RhNeutrinoGeV::DEFAULT_BATCHSIZE);
}
} // namespace blackthorn::py_interface::gev
