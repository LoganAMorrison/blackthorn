#include "blackthorn/Models/Particles.h"
#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/Tools.h"

namespace blackthorn {

/// Compute the radiative decay spectrum, dN/dE, from: P -> ℓ + ν + γ.
///
/// @tparam Meson The decaying meson (either `ChargedPion` or `ChargedKaon`.)
/// @tparam Lepton Final state lepton (either `Muon` or `Electron`.)
/// @param egam Energy of the final state lepton.
template <class Meson, class Lepton> auto dnde_x_to_lva(double egam) -> double {
  static constexpr double ALPHA = StandardModel::alpha_em;
  static constexpr double ml = Lepton::mass;
  static constexpr double mp = Meson::mass;
  static constexpr double f = Meson::decay_const / Meson::mass;
  static constexpr double FV = Meson::ff_vec;
  static constexpr double FA = Meson::ff_axi;
  static constexpr double slope = Meson::ff_vec_slope;
  static constexpr double eps = Meson::ff_eps;

  const double x = 2.0 * egam / mp;
  const double r = tools::sqr(ml / mp);
  const double rm = 1.0 - r;

  if (x < 0.0 || rm < x) {
    return 0.0;
  }

  // Vp(s) = Vp(0) * (1 + a * s), s = 1 - x
  const double vp = FV * (1.0 - slope * (1.0 - x));
  const double xm = 1.0 - x;

  const double pre = ALPHA / (M_PI * pow(rm, 2));

  const double coeff_poly = (r - xm) / (24. * pow(f, 2) * r * x * pow(xm, 2));
  const double poly =
      (pow(FA, 2) + pow(FV, 2)) * (2 + r - 2 * x) * pow(x, 4) * (r - xm) +
      12 * pow(f, 2) * r * xm * (pow(-2 + x, 2) - 4 * r * xm) +
      12 * f * r * pow(x, 2) * (FA * (1 + r - 2 * x) + FV * x) * xm * eps;

  const double coeff_log = 1 / (2.0 * f * x);
  const double log_term = (f * (2 - 2 * pow(r, 2) + 2 * r * x + (-2 + x) * x) +
                           pow(x, 2) * (2 * FA * r - FA * x + FV * x) * eps) *
                          log(xm / r);

  return pre * (coeff_poly * poly + coeff_log * log_term);
}

} // namespace blackthorn
