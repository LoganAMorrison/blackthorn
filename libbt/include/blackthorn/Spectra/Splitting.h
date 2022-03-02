#ifndef BLACKTHORN_SPECTRA_SPLITTING_H
#define BLACKTHORN_SPECTRA_SPLITTING_H

#include "blackthorn/Models/Particles.h"
#include "blackthorn/Models/StandardModel.h"
#include <vector>

namespace blackthorn {

template <class F> class dndx_fsr { // NOLINT

  static_assert(is_fermion<F>::value || is_scalar_boson<F>::value ||
                    is_vector_boson<F>::value,
                "Field must be a scalar,fermion, or vector.");

private:
  static constexpr double mass = field_attrs<F>::mass();
  static constexpr double Q = quantum_numbers<F>::charge();
  static constexpr double PRE = StandardModel::alpha_em / (2 * M_PI);

  static auto kernel(double x) -> double {
    if constexpr (is_fermion<F>::value) {
      return (1 + tools::sqr(1 - x)) / x;
    } else if constexpr (is_scalar_boson<F>::value) {
      return 2 * (1 - x) / x;
    } else {
      return 2 * (x * (1 - x) + x / (1 - x) + (1 - x) / x);
    }
  }

  static auto pre_factor(double x, double s) -> double {
    using tools::sqr;
    const double mu2 = sqr(mass) / s;
    const double xm = 1.0 - x;
    if (M_E * mu2 > xm) {
      return 0.0;
    }
    return StandardModel::alpha_em / (2 * M_PI) * sqr(Q) *
           (log(xm / mu2) - 1.0);
  }

public:
  /**
   * Compute the fsr spectrum using the altarelli-parisi equations.
   *
   * @param x Energy of the photon
   * @param s Square of center-of-mass energy
   */
  static auto photon_spectrum(double x, double s) -> double {
    if constexpr (Q == 0.0) {
      return 0.0;
    } else {
      return kernel(x) * pre_factor(x, s);
    }
  }
  /**
   * Compute the fsr spectrum using the altarelli-parisi equations.
   *
   * @param x Energy of the photon
   * @param s Square of center-of-mass energy
   */
  static auto photon_spectrum(const std::vector<double> &xs, double s)
      -> std::vector<double> {
    std::vector<double> spec(xs.size(), 0.0);
    std::transform(xs.cbegin(), xs.cend(), spec.begin(), [&s](double x) {
      return dndx_fsr<F>::photon_spectrum(x, s);
    });
    return spec;
  }
};

auto dndx_altarelli_parisi_f_to_a(double x, double beta) -> double;
auto dndx_altarelli_parisi_s_to_a(double x, double beta) -> double;
auto dndx_altarelli_parisi_v_to_a(double x, double beta) -> double;

auto dndx_altarelli_parisi_f_to_a(double x, double beta, double z) -> double;
auto dndx_altarelli_parisi_s_to_a(double x, double beta, double z) -> double;
auto dndx_altarelli_parisi_v_to_a(double x, double beta, double z) -> double;

} // namespace blackthorn

#endif // BLACKTHORN_SPECTRA_SPLITTING_H
