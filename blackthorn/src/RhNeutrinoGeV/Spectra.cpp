#include "RhNeutrinoGeV.h"

namespace blackthorn::py_interface::gev {

static constexpr double default_xmin = 1e-6;
static constexpr double default_xmax = 1.0;
static constexpr unsigned int default_nevents = 10'000;
static constexpr unsigned int default_nbins = 100;

auto add_dndx_v_h(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the spectrum dN/dx from N -> nu + H.

Parameters
----------
xmin: float,optional
  Minimum value of x = 2*Egam/mn. Default is 1e-6.
xmax: float
  Maximum value of x = 2*Egam/mn. Default is 1.0.
nbins: int
  Number of bins x bins to use. Default is 100.
nevents: int, optional
  Number of Pythia events to use. Default is 10,000.
)Doc";
  rhn->def(
      "dndx_v_h",
      [](const RhNeutrinoGeV &model, double xmin = default_xmin,
         double xmax = default_xmax, unsigned int nbins = default_nbins,
         unsigned int nevents = default_nevents) {
        return model.dndx_v_h(xmin, xmax, nbins, nevents);
      },
      doc.c_str(), py::arg("xmin") = default_xmin,
      py::arg("xmax") = default_xmax, py::arg("nbins") = default_nbins,
      py::arg("nevents") = default_nevents);
}

auto add_dndx_v_z(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the spectrum dN/dx from N -> nu + Z.

Parameters
----------
xmin: float,optional
  Minimum value of x = 2*Egam/mn. Default is 1e-6.
xmax: float
  Maximum value of x = 2*Egam/mn. Default is 1.0.
nbins: int
  Number of bins x bins to use. Default is 100.
nevents: int, optional
  Number of Pythia events to use. Default is 10,000.
)Doc";
  rhn->def(
      "dndx_v_z",
      [](const RhNeutrinoGeV &model, double xmin = default_xmin,
         double xmax = default_xmax, unsigned int nbins = default_nbins,
         unsigned int nevents = default_nevents) {
        return model.dndx_v_z(xmin, xmax, nbins, nevents);
      },
      doc.c_str(), py::arg("xmin") = default_xmin,
      py::arg("xmax") = default_xmax, py::arg("nbins") = default_nbins,
      py::arg("nevents") = default_nevents);
}

auto add_dndx_l_w(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the spectrum dN/dx from N -> l + W.

Parameters
----------
xmin: float,optional
  Minimum value of x = 2*Egam/mn. Default is 1e-6.
xmax: float
  Maximum value of x = 2*Egam/mn. Default is 1.0.
nbins: int
  Number of bins x bins to use. Default is 100.
nevents: int, optional
  Number of Pythia events to use. Default is 10,000.
)Doc";
  rhn->def(
      "dndx_l_w",
      [](const RhNeutrinoGeV &model, double xmin = default_xmin,
         double xmax = default_xmax, unsigned int nbins = default_nbins,
         unsigned int nevents = default_nevents) {
        return model.dndx_l_w(xmin, xmax, nbins, nevents);
      },
      doc.c_str(), py::arg("xmin") = default_xmin,
      py::arg("xmax") = default_xmax, py::arg("nbins") = default_nbins,
      py::arg("nevents") = default_nevents);
}

auto add_dndx_v_u_u(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the spectrum dN/dx from N -> v + u + u.

Parameters
----------
genu: Gen
  Generation of final-state up-type-quark.
xmin: float,optional
  Minimum value of x = 2*Egam/mn. Default is 1e-6.
xmax: float
  Maximum value of x = 2*Egam/mn. Default is 1.0.
nbins: int
  Number of bins x bins to use. Default is 100.
nevents: int, optional
  Number of Pythia events to use. Default is 10,000.
)Doc";
  rhn->def(
      "dndx_v_u_u",
      [](const RhNeutrinoGeV &model, Gen genu, double xmin = 1e-6,
         double xmax = 1.0, unsigned int nbins = default_nbins,
         unsigned int nevents = default_nevents) {
        return model.dndx_v_u_u(xmin, xmax, nbins, genu, nevents);
      },
      doc.c_str(), py::arg("genu"), py::arg("xmin") = default_xmin,
      py::arg("xmax") = default_xmax, py::arg("nbins") = default_nbins,
      py::arg("nevents") = default_nevents);
}

auto add_dndx_v_d_d(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the spectrum dN/dx from N -> v + d + d.

Parameters
----------
gend: Gen
  Generation of final-state down-type-quark.
xmin: float,optional
  Minimum value of x = 2*Egam/mn. Default is 1e-6.
xmax: float
  Maximum value of x = 2*Egam/mn. Default is 1.0.
nbins: int
  Number of bins x bins to use. Default is 100.
nevents: int, optional
  Number of Pythia events to use. Default is 10,000.
)Doc";
  rhn->def(
      "dndx_v_d_d",
      [](const RhNeutrinoGeV &model, Gen gend, double xmin = 1e-6,
         double xmax = 1.0, unsigned int nbins = default_nbins,
         unsigned int nevents = default_nevents) {
        return model.dndx_v_d_d(xmin, xmax, nbins, gend, nevents);
      },
      doc.c_str(), py::arg("genu"), py::arg("xmin") = default_xmin,
      py::arg("xmax") = default_xmax, py::arg("nbins") = default_nbins,
      py::arg("nevents") = default_nevents);
}

auto add_dndx_l_u_d(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the spectrum dN/dx from N -> l + u + d.

Parameters
----------
genu: Gen
  Generation of final-state up-type-quark.
gend: Gen
  Generation of final-state down-type-quark.
xmin: float,optional
  Minimum value of x = 2*Egam/mn. Default is 1e-6.
xmax: float
  Maximum value of x = 2*Egam/mn. Default is 1.0.
nbins: int
  Number of bins x bins to use. Default is 100.
nevents: int, optional
  Number of Pythia events to use. Default is 10,000.
)Doc";
  rhn->def(
      "dndx_l_u_d",
      [](const RhNeutrinoGeV &model, Gen genu, Gen gend, double xmin = 1e-6,
         double xmax = 1.0, unsigned int nbins = default_nbins,
         unsigned int nevents = default_nevents) {
        return model.dndx_l_u_d(xmin, xmax, nbins, genu, gend, nevents);
      },
      doc.c_str(), py::arg("genu"), py::arg("gend"),
      py::arg("xmin") = default_xmin, py::arg("xmax") = default_xmax,
      py::arg("nbins") = default_nbins, py::arg("nevents") = default_nevents);
}

auto add_dndx_v_l_l(RhnClassType *rhn) -> void {

  static std::string doc = R"Doc(
Compute the spectrum dN/dx from N -> v + l + l.

Parameters
----------
genv: Gen
  Generation of final-state neutrino.
genl1: Gen
  Generation of 1st final-state lepton.
genl2: Gen
  Generation of 2nd final-state lepton.
xmin: float,optional
  Minimum value of x = 2*Egam/mn. Default is 1e-6.
xmax: float
  Maximum value of x = 2*Egam/mn. Default is 1.0.
nbins: int
  Number of bins x bins to use. Default is 100.
nevents: int, optional
  Number of Pythia events to use. Default is 10,000.
)Doc";
  rhn->def(
      "dndx_v_l_l",
      [](const RhNeutrinoGeV &model, Gen genv, Gen genl1, Gen genl2,
         double xmin = 1e-6, double xmax = 1.0,
         unsigned int nbins = default_nbins,
         unsigned int nevents = default_nevents) {
        return model.dndx_v_l_l(xmin, xmax, nbins, genv, genl1, genl2, nevents);
      },
      doc.c_str(), py::arg("genv"), py::arg("genl1"), py::arg("genl2"),
      py::arg("xmin") = default_xmin, py::arg("xmax") = default_xmax,
      py::arg("nbins") = default_nbins, py::arg("nevents") = default_nevents);
}

} // namespace blackthorn::py_interface::gev
