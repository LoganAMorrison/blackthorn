#include "RhNeutrinoGeV/RhNeutrinoGeV.h"
#include "RhNeutrinoMeV/RhNeutrinoMeV.h"
#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace blackthorn;
namespace py = pybind11;

PYBIND11_MODULE(rh_neutrino, m) { // NOLINT

  py::enum_<Gen>(m, "Gen")
      .value("Fst", Gen::Fst)
      .value("Snd", Gen::Snd)
      .value("Trd", Gen::Trd)
      .value("Null", Gen::Null)
      .export_values();

  auto rhngev = py::class_<RhNeutrinoGeV>(m, "RhNeutrinoGeVCpp");
  py_interface::gev::config_class(&rhngev);

  auto rhnmev = py::class_<RhNeutrinoMeV>(m, "RhNeutrinoMeVCpp");
  py_interface::mev::config_class(&rhnmev);
}
