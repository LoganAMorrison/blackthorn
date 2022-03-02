#ifndef BLACKTHORN_PY_INTERFACE_H
#define BLACKTHORN_PY_INTERFACE_H

#include "blackthorn/Models/RhNeutrino.h"
#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace blackthorn {

namespace py_interface {

namespace py = pybind11;

auto config_class(py::class_<RhNeutrinoGeV> *rhn) -> void;
auto config_class(py::class_<RhNeutrinoMeV> *rhn) -> void;

struct DocStringParam { // NOLINT
  std::string name;
  std::string type;
  std::string desc;
};

class DocString {
  std::string p_desc;
  std::vector<DocStringParam> p_params;
  std::vector<DocStringParam> p_return;

  auto create_param_list() const -> std::string {
    std::string plist = "Parameters\n----------\n";
    for (const auto &p : p_params) {
      plist = fmt::format("{}\n{}: {}\n\t{}", plist, p.name, p.type, p.desc);
    }
    return plist;
  }

  auto create_return() const -> std::string {
    std::string plist = "Parameters\n----------\n";
    for (const auto &p : p_return) {
      plist = fmt::format("{}\n{}: {}\n\t{}", plist, p.name, p.type, p.desc);
    }
    return plist;
  }

public:
  DocString() = default;

  auto desc(std::string desc) -> DocString {
    p_desc = std::move(desc);
    return *this;
  }
  auto add_return(DocStringParam param) -> DocString {
    p_return.emplace_back(std::move(param));
    return *this;
  }
  auto add_param(DocStringParam param) -> DocString {
    p_params.emplace_back(std::move(param));
    return *this;
  }

  auto create() -> std::string {
    return fmt::format("{}\n\n{}\n\n{}", p_desc, create_param_list(),
                       create_return());
  }
};

} // namespace py_interface
} // namespace blackthorn

#endif // BLACKTHORN_PY_INTERFACE_H
