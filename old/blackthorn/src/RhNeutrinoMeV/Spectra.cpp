#include "RhNeutrinoMeV.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace blackthorn {
namespace py_interface {
namespace mev {

enum class SpectrumType { Photon, Positron, Neutrino };

template <SpectrumType Type>
static auto make_dndx_docstring(const std::string &process) -> std::string {
  std::string type;
  if (Type == SpectrumType::Photon) {
    type = "photon";
  } else if (Type == SpectrumType::Positron) {
    type = "positron";
  } else {
    type = "neutrino";
  }

  return ("Compute the " + type + " spectrum, dN/dx, from N -> " + process +
          "."
          "\n"
          "\n" +
          "Parameters" + "\n" + "----------" + "\n" +
          "x: Union[float, np.ndarray]" + "\n" + "    Value of x = 2 E / M" +
          "\n" +
          "beta: float"
          "\n" +
          "    Boost velocity of the RH neutrino." + "\n" + "\n" + "Returns" +
          "\n" + "-------" + "\n" + "dndx: Union[float, np.ndarray]" + "\n" +
          "    Decay spectrum." + "\n");
}

// =========================================================================
// ---- N -> l + pi --------------------------------------------------------
// =========================================================================

auto add_dndx_photon_l_pi(RhnClassType *rhn) -> void {
  static std::string doc = make_dndx_docstring<SpectrumType::Photon>("l + pi");
  rhn->def(
      "dndx_photon_l_pi",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x,
         double beta) { return model.dndx_photon_l_pi(x, beta); },
      doc.c_str(), py::arg("x"), py::arg("beta"));
  rhn->def(
      "dndx_photon_l_pi",
      [](const RhNeutrinoMeV &model, double x, double beta) {
        return model.dndx_photon_l_pi(x, beta);
      },
      doc.c_str(), py::arg("x"), py::arg("beta"));
}

auto add_dndx_positron_l_pi(RhnClassType *rhn) -> void {
  static std::string doc =
      make_dndx_docstring<SpectrumType::Positron>("l + pi");
  rhn->def(
      "dndx_positron_l_pi",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x,
         double beta) { return model.dndx_positron_l_pi(x, beta); },
      doc.c_str(), py::arg("x"), py::arg("beta"));
  rhn->def(
      "dndx_positron_l_pi",
      [](const RhNeutrinoMeV &model, double x, double beta) {
        return model.dndx_positron_l_pi(x, beta);
      },
      doc.c_str(), py::arg("x"), py::arg("beta"));
}

auto add_dndx_neutrino_l_pi(RhnClassType *rhn) -> void {
  static std::string doc =
      make_dndx_docstring<SpectrumType::Neutrino>("l + pi");
  rhn->def(
      "dndx_neutrino_l_pi",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x,
         double beta) { return model.dndx_neutrino_l_pi(x, beta); },
      doc.c_str(), py::arg("x"), py::arg("beta"));
  rhn->def(
      "dndx_neutrino_l_pi",
      [](const RhNeutrinoMeV &model, double x, double beta) {
        return model.dndx_neutrino_l_pi(x, beta);
      },
      doc.c_str(), py::arg("x"), py::arg("beta"));
}

// =========================================================================
// ---- N -> l + k ---------------------------------------------------------
// =========================================================================

auto add_dndx_photon_l_k(RhnClassType *rhn) -> void {
  static std::string doc = make_dndx_docstring<SpectrumType::Photon>("l + k");
  rhn->def(
      "dndx_photon_l_k",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x,
         double beta) { return model.dndx_photon_l_k(x, beta); },
      doc.c_str(), py::arg("x"), py::arg("beta"));
  rhn->def(
      "dndx_photon_l_k",
      [](const RhNeutrinoMeV &model, double x, double beta) {
        return model.dndx_photon_l_k(x, beta);
      },
      doc.c_str(), py::arg("x"), py::arg("beta"));
}

auto add_dndx_positron_l_k(RhnClassType *rhn) -> void {
  static std::string doc = make_dndx_docstring<SpectrumType::Positron>("l + k");
  rhn->def(
      "dndx_positron_l_k",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x,
         double beta) { return model.dndx_positron_l_k(x, beta); },
      doc.c_str(), py::arg("x"), py::arg("beta"));
  rhn->def(
      "dndx_positron_l_k",
      [](const RhNeutrinoMeV &model, double x, double beta) {
        return model.dndx_positron_l_k(x, beta);
      },
      doc.c_str(), py::arg("x"), py::arg("beta"));
}

auto add_dndx_neutrino_l_k(RhnClassType *rhn) -> void {
  static std::string doc = make_dndx_docstring<SpectrumType::Neutrino>("l + k");
  rhn->def(
      "dndx_neutrino_l_k",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x,
         double beta) { return model.dndx_neutrino_l_k(x, beta); },
      doc.c_str(), py::arg("x"), py::arg("beta"));
  rhn->def(
      "dndx_neutrino_l_k",
      [](const RhNeutrinoMeV &model, double x, double beta) {
        return model.dndx_neutrino_l_k(x, beta);
      },
      doc.c_str(), py::arg("x"), py::arg("beta"));
}

// =========================================================================
// ---- N -> v + pi0 -------------------------------------------------------
// =========================================================================

auto add_dndx_photon_v_pi0(RhnClassType *rhn) -> void {
  static std::string doc = make_dndx_docstring<SpectrumType::Photon>("l + pi0");
  rhn->def(
      "dndx_photon_v_pi0",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x,
         double beta) { return model.dndx_photon_v_pi0(x, beta); },
      doc.c_str(), py::arg("x"), py::arg("beta"));
  rhn->def(
      "dndx_photon_v_pi0",
      [](const RhNeutrinoMeV &model, double x, double beta) {
        return model.dndx_photon_v_pi0(x, beta);
      },
      doc.c_str(), py::arg("x"), py::arg("beta"));
}

auto add_dndx_positron_v_pi0(RhnClassType *rhn) -> void {
  static std::string doc =
      make_dndx_docstring<SpectrumType::Positron>("v + pi0");
  rhn->def(
      "dndx_positron_v_pi0",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x,
         double beta) { return model.dndx_positron_v_pi0(x, beta); },
      doc.c_str(), py::arg("x"), py::arg("beta"));
  rhn->def(
      "dndx_positron_v_pi0",
      [](const RhNeutrinoMeV &model, double x, double beta) {
        return model.dndx_positron_v_pi0(x, beta);
      },
      doc.c_str(), py::arg("x"), py::arg("beta"));
}

auto add_dndx_neutrino_v_pi0(RhnClassType *rhn) -> void {
  static std::string doc =
      make_dndx_docstring<SpectrumType::Neutrino>("v + pi0");
  rhn->def(
      "dndx_neutrino_v_pi0",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x,
         double beta) { return model.dndx_neutrino_v_pi0(x, beta); },
      doc.c_str(), py::arg("x"), py::arg("beta"));
  rhn->def(
      "dndx_neutrino_v_pi0",
      [](const RhNeutrinoMeV &model, double x, double beta) {
        return model.dndx_neutrino_v_pi0(x, beta);
      },
      doc.c_str(), py::arg("x"), py::arg("beta"));
}

// =========================================================================
// ---- N -> v + pi + pi ---------------------------------------------------
// =========================================================================

auto add_dndx_photon_v_pi_pi(RhnClassType *rhn) -> void {
  static std::string doc =
      make_dndx_docstring<SpectrumType::Photon>("v + pi + pi");
  rhn->def(
      "dndx_photon_v_pi_pi",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x,
         double beta) { return model.dndx_photon_v_pi_pi(x, beta); },
      doc.c_str(), py::arg("x"), py::arg("beta"));
  rhn->def(
      "dndx_photon_v_pi_pi",
      [](const RhNeutrinoMeV &model, double x, double beta) {
        return model.dndx_photon_v_pi_pi(x, beta);
      },
      doc.c_str(), py::arg("x"), py::arg("beta"));
}

auto add_dndx_positron_v_pi_pi(RhnClassType *rhn) -> void {
  static std::string doc =
      make_dndx_docstring<SpectrumType::Positron>("v + pi + pi");
  rhn->def(
      "dndx_positron_v_pi_pi",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x,
         double beta) { return model.dndx_positron_v_pi_pi(x, beta); },
      doc.c_str(), py::arg("x"), py::arg("beta"));
  rhn->def(
      "dndx_positron_v_pi_pi",
      [](const RhNeutrinoMeV &model, double x, double beta) {
        return model.dndx_positron_v_pi_pi(x, beta);
      },
      doc.c_str(), py::arg("x"), py::arg("beta"));
}

auto add_dndx_neutrino_v_pi_pi(RhnClassType *rhn) -> void {
  static std::string doc =
      make_dndx_docstring<SpectrumType::Neutrino>("v + pi + pi");
  rhn->def(
      "dndx_neutrino_v_pi_pi",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x,
         double beta) { return model.dndx_neutrino_v_pi_pi(x, beta); },
      doc.c_str(), py::arg("x"), py::arg("beta"));
  rhn->def(
      "dndx_neutrino_v_pi_pi",
      [](const RhNeutrinoMeV &model, double x, double beta) {
        return model.dndx_neutrino_v_pi_pi(x, beta);
      },
      doc.c_str(), py::arg("x"), py::arg("beta"));
}

// =========================================================================
// ---- N -> v + l + l -----------------------------------------------------
// =========================================================================

auto add_dndx_photon_v_l_l(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the photon spectrum dN/dx from N -> v + l + l.

Parameters
----------
x: np.ndarray
  Value of x = 2 E / mn
beta: float
  Boost velocity of the RH neutrino.
gv: Gen
  Generation of the active neutrino.
g1: Gen
  Generation of first lepton.
g2: Gen
  Generation of second lepton.

Returns
-------
dndx: np.ndarray
  Decay spectrum.
)Doc";
  rhn->def(
      "dndx_photon_v_l_l",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x, double beta,
         Gen gv, Gen gl1,
         Gen gl2) { return model.dndx_photon_v_l_l(x, beta, gv, gl1, gl2); },
      doc.c_str(), py::arg("x"), py::arg("beta"), py::arg("gv"), py::arg("gl1"),
      py::arg("gl2"));
  rhn->def(
      "dndx_photon_v_l_l",
      [](const RhNeutrinoMeV &model, double x, double beta, Gen gv, Gen gl1,
         Gen gl2) { return model.dndx_photon_v_l_l(x, beta, gv, gl1, gl2); },
      doc.c_str(), py::arg("x"), py::arg("beta"), py::arg("gv"), py::arg("gl1"),
      py::arg("gl2"));
}

auto add_dndx_positron_v_l_l(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the positron spectrum dN/dx from N -> v + l + l.

Parameters
----------
x: np.ndarray
  Value of x = 2 E / mn
beta: float
  Boost velocity of the RH neutrino.
gv: Gen
  Generation of the active neutrino.
g;1: Gen
  Generation of 1st lepton.
gl2: Gen
  Generation of 2nd lepton.

Returns
-------
dndx: np.ndarray
  Decay spectrum.
)Doc";
  rhn->def(
      "dndx_positron_v_l_l",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x, double beta,
         Gen gv, Gen gl1,
         Gen gl2) { return model.dndx_positron_v_l_l(x, beta, gv, gl1, gl2); },
      doc.c_str(), py::arg("x"), py::arg("beta"), py::arg("gv"), py::arg("gl1"),
      py::arg("gl2"));
  rhn->def(
      "dndx_positron_v_l_l",
      [](const RhNeutrinoMeV &model, double x, double beta, Gen gv, Gen gl1,
         Gen gl2) { return model.dndx_positron_v_l_l(x, beta, gv, gl1, gl2); },
      doc.c_str(), py::arg("x"), py::arg("beta"), py::arg("gv"), py::arg("gl1"),
      py::arg("gl2"));
}

auto add_dndx_neutrino_v_l_l(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Compute the neutrino spectrum dN/dx from N -> v + l + l.

Parameters
----------
x: np.ndarray
  Value of x = 2 E / mn
beta: float
  Boost velocity of the RH neutrino.
gv: Gen
  Generation of the active neutrino.
gl1: Gen
  Generation of first lepton.
gl2: Gen
  Generation of second lepton.

Returns
-------
dndx: np.ndarray
  Decay spectrum.
)Doc";
  rhn->def(
      "dndx_neutrino_v_l_l",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x, double beta,
         Gen gv, Gen g1,
         Gen g2) { return model.dndx_neutrino_v_l_l(x, beta, gv, g1, g2); },
      doc.c_str(), py::arg("x"), py::arg("beta"), py::arg("gv"), py::arg("gl1"),
      py::arg("gl2"));
  rhn->def(
      "dndx_neutrino_v_l_l",
      [](const RhNeutrinoMeV &model, double x, double beta, Gen gv, Gen g1,
         Gen g2) { return model.dndx_neutrino_v_l_l(x, beta, gv, g1, g2); },
      doc.c_str(), py::arg("x"), py::arg("beta"), py::arg("gv"), py::arg("gl1"),
      py::arg("gl2"));
}

// =========================================================================
// ---- N -> l + pi + pi0 --------------------------------------------------
// =========================================================================

auto add_dndx_photon_l_pi_pi0(RhnClassType *rhn) -> void {
  static std::string doc =
      make_dndx_docstring<SpectrumType::Photon>("l + pi + pi0");
  rhn->def(
      "dndx_photon_l_pi_pi0",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x,
         double beta) { return model.dndx_photon_l_pi_pi0(x, beta); },
      doc.c_str(), py::arg("x"), py::arg("beta"));
  rhn->def(
      "dndx_photon_l_pi_pi0",
      [](const RhNeutrinoMeV &model, double x, double beta) {
        return model.dndx_photon_l_pi_pi0(x, beta);
      },
      doc.c_str(), py::arg("x"), py::arg("beta"));
}

auto add_dndx_positron_l_pi_pi0(RhnClassType *rhn) -> void {
  static std::string doc =
      make_dndx_docstring<SpectrumType::Positron>("l + pi + pi0");
  rhn->def(
      "dndx_positron_l_pi_pi0",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x,
         double beta) { return model.dndx_positron_l_pi_pi0(x, beta); },
      doc.c_str(), py::arg("x"), py::arg("beta"));
  rhn->def(
      "dndx_positron_l_pi_pi0",
      [](const RhNeutrinoMeV &model, double x, double beta) {
        return model.dndx_positron_l_pi_pi0(x, beta);
      },
      doc.c_str(), py::arg("x"), py::arg("beta"));
}

auto add_dndx_neutrino_l_pi_pi0(RhnClassType *rhn) -> void {
  static std::string doc =
      make_dndx_docstring<SpectrumType::Neutrino>("l + pi + pi0");
  rhn->def(
      "dndx_neutrino_l_pi_pi0",
      [](const RhNeutrinoMeV &model, const py::array_t<double> &x,
         double beta) { return model.dndx_neutrino_l_pi_pi0(x, beta); },
      doc.c_str(), py::arg("x"), py::arg("beta"));
  rhn->def(
      "dndx_neutrino_l_pi_pi0",
      [](const RhNeutrinoMeV &model, double x, double beta) {
        return model.dndx_neutrino_l_pi_pi0(x, beta);
      },
      doc.c_str(), py::arg("x"), py::arg("beta"));
}

} // namespace mev
} // namespace py_interface
} // namespace blackthorn
