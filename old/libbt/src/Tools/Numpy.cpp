#include "blackthorn/Tools.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <vector>

namespace blackthorn::tools {

auto get_buffer_and_check_dim(const py::array_t<double> &xs)
    -> py::buffer_info {
  py::buffer_info buf = xs.request();
  if (buf.ndim != 1) {
    throw std::runtime_error("Number of dimensions must be 1.");
  }
  return buf;
}

auto zeros_like(const py::array_t<double> &xs) -> py::array_t<double> {
  py::buffer_info buf_xs = get_buffer_and_check_dim(xs);

  auto dndx = py::array_t<double>(buf_xs.size);
  py::buffer_info buf_dndx = dndx.request();
  auto *ptr_dndx = static_cast<double *>(buf_dndx.ptr);

  for (size_t i = 0; i < buf_xs.shape[0]; ++i) { // NOLINT
    ptr_dndx[i] = 0.0;                           // NOLINT
  }
  return dndx;
}

auto zeros_like(const std::vector<double> &xs) -> std::vector<double> {
  return std::vector<double>(xs.size(), 0.0); // NOLINT
}

} // namespace blackthorn::tools
