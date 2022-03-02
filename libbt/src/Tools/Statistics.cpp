#include "blackthorn/Tools.h"
#include <numeric>

namespace blackthorn::tools {

auto mean_sum_sqrs_welford(const std::vector<double> &wgts)
    -> std::tuple<double, double, double> {
  using tools::sqr;

  double mean = 0.0;
  double m2 = 0.0;
  double count = 0.0;

  for (const auto &wgt : wgts) {
    count += 1;
    const double delta = wgt - mean;
    mean += delta / count;
    const double delta2 = wgt - mean;
    m2 += delta * delta2;
  }

  return std::make_tuple(mean, m2, count);
}

auto mean_var_welford(const std::vector<double> &wgts)
    -> std::pair<double, double> {
  using tools::sqr;

  double mean = 0.0;
  double m2 = 0.0;
  double count = 0.0;

  for (const auto &wgt : wgts) {
    count += 1;
    const double delta = wgt - mean;
    mean += delta / count;
    const double delta2 = wgt - mean;
    m2 += delta * delta2;
  }

  return std::make_pair(mean, m2 / (count - 1));
}

auto mean_var(const std::vector<double> &wgts) -> std::pair<double, double> {
  using tools::sqr;
  const double inv_n = 1.0 / static_cast<double>(wgts.size());

  const double mean = inv_n * std::accumulate(wgts.begin(), wgts.end(), 0.0);

  auto var_reduce = [mean](double a, double b) { return a + sqr(b - mean); };
  const double var =
      inv_n * std::accumulate(wgts.begin(), wgts.end(), 0.0, var_reduce);

  return std::make_pair(mean, var);
}

} // namespace blackthorn::tools
