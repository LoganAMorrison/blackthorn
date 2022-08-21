#include "blackthorn/Tools.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <vector>

namespace blackthorn::tools {

auto linspace(const double start, const double end, const size_t n)
    -> std::vector<double> {
  std::vector<double> lst(n);
  const double step = (end - start) / static_cast<double>(n - 1);
  std::generate(lst.begin(), lst.end(), [&, n = 0]() mutable {
    const double val = step * static_cast<double>(n) + start;
    n++;
    return val;
  });
  return lst;
}

auto logspace(const double start, const double end, const size_t n,
              const double base) -> std::vector<double> {
  std::vector<double> lst(n);
  const double step = (end - start) / static_cast<double>(n - 1);
  std::generate(lst.begin(), lst.end(), [&, nn = 0]() mutable {
    const double val = step * static_cast<double>(nn) + start;
    ++nn;
    return std::pow(base, val);
  });
  return lst;
}

auto geomspace(const double start, const double end, const size_t n)
    -> std::vector<double> {
  return logspace(log10(start), log10(end), n, 10.0);
}

} // namespace blackthorn::tools
