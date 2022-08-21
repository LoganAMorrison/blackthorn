#include "blackthorn/Spectra/Splitting.h"
#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/Tools.h"

namespace blackthorn {

auto kin_log(double x, double m, double s) -> double {
  const double f = s * tools::sqr(x / (2 * m));
  return log(f) + 2 * log(1 + sqrt(1 - 1 / f));
}

auto kernel_f_to_a(double x) -> double { return (1 + tools::sqr(1 - x)) / x; }

auto kernel_s_to_a(double x) -> double { return 2 * (1 - x) / x; }

auto kernel_v_to_a(double x) -> double {
  return 2 * (x * (1 - x) + x / (1 - x) + (1 - x) / x);
}

static auto kernel_pre_x_to_a(double x, double beta) -> double { // NOLINT
  const double b2 = beta * beta;
  if (x > 1 - exp(1) * b2) {
    return 0;
  }
  return StandardModel::alpha_em / (2 * M_PI) * (log((1 - x) / b2) - 1);
}

auto dndx_altarelli_parisi_f_to_a(double x, double beta) -> double {
  return kernel_f_to_a(x) * kernel_pre_x_to_a(x, beta);
}

auto dndx_altarelli_parisi_s_to_a(double x, double beta) -> double {
  return kernel_s_to_a(x) * kernel_pre_x_to_a(x, beta);
}

auto dndx_altarelli_parisi_v_to_a(double x, double beta) -> double {
  return kernel_v_to_a(x) * kernel_pre_x_to_a(x, beta);
}

auto dndx_altarelli_parisi_f_to_a(double x, double beta, double z) -> double {
  return kernel_f_to_a(x / z) * kernel_pre_x_to_a(x / z, beta / z);
}

auto dndx_altarelli_parisi_s_to_a(double x, double beta, double z) -> double {
  return kernel_s_to_a(x / z) * kernel_pre_x_to_a(x / z, beta / z);
}

auto dndx_altarelli_parisi_v_to_a(double x, double beta, double z) -> double {
  return kernel_v_to_a(x / z) * kernel_pre_x_to_a(x / z, beta / z);
}

// auto f(double x, double mu) -> double {
//   return StandardModel::alpha_em / (2 * M_PI) * split(x) *
//          (log((1 - x) / x) - 1);
// }

} // namespace blackthorn
