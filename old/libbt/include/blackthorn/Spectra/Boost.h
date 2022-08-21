#ifndef BLACKTHORN_SPECTRA_BOOST_H
#define BLACKTHORN_SPECTRA_BOOST_H

#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/PhaseSpace.h"
#include "blackthorn/Tools.h"
#include <algorithm>
#include <array>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <cmath>
#include <execution>

namespace blackthorn {

/**
 * Compute the boosted energy of a daugther particle when boosted from the
 * lab-frame to the rest-frame of the parent particle.
 *
 * @param ep energy of the parent particle in lab-frame
 * @param mp mass of the parent particle
 * @param ed energy of the daughter particle in lab-frame
 * @param mp mass of the daughter particle
 * @param zl cosine of the angle the daughter particle makes wrt z-axis in
 * lab-frame
 */
auto boost_eng(double ep, double mp, double ed, double md, double zl) -> double;

/**
 * Returns the Jacobian for boost integrals when boosting from the lab frame
 * to the parent particle's rest frame.
 *
 * @param ep energy of the parent particle in lab-frame
 * @param mp mass of the parent particle
 * @param ed energy of the daughter particle in lab-frame
 * @param mp mass of the daughter particle
 * @param zl cosine of the angle the daughter particle makes wrt z-axis in
 * lab-frame
 */
auto boost_jac(double ep, double mp, double ed, double md, double zl) -> double;

/**
 * Boost a δ-function spectrum centered at `e0` from the rest-frame of the
 * parent particle to the lab-frame.
 *
 * @param ep energy of the parent particle in lab-frame
 * @param mp mass of the parent particle
 * @param ed energy of the daughter particle in lab-frame
 * @param mp mass of the daughter particle
 * @param e0 center of the dirac-delta spectrum in rest-frame
 */
auto boost_delta_function(double e0, double e, double m, double beta) -> double;

/**
 * Boost the spectrum of a daughter particle in parent particles rest-frame into
 * the lab-frame.
 *
 * @param spec_rf unary function to compute spectrum in the rest-frame
 * @param ep energy of the parent particle in lab-frame
 * @param mp mass of the parent particle
 * @param ed energy of the daughter particle in lab-frame
 * @param mp mass of the daughter particle
 * @param ed_ul lower bound on the daughter energy. Default 0.
 * @param ed_ub upper bound on the daughter energy. Default +inf.
 */
template <class F>
auto boost_spectrum(F spec_rf, double ep,
                    double mp, // NOLINT
                    double ed, // NOLINT
                    double md, // NOLINT
                    double ed_lb = 0.0,
                    double ed_ub = std::numeric_limits<double>::infinity())
    -> double {
  using boost::math::quadrature::gauss_kronrod;
  static constexpr unsigned int GK_N = 15;
  static constexpr unsigned int GK_MAX_LIMIT = 7;
  if (ep < mp) {
    return 0.0;
  }
  // If we are sufficiently close to the parent's rest-frame, use the
  // rest-frame result.
  if (ep - mp < std::numeric_limits<double>::epsilon()) {
    return spec_rf(ed);
  }

  double l_ed_lb = ed_lb;
  if (l_ed_lb >= std::numeric_limits<double>::infinity()) {
    l_ed_lb = ep;
  }

  const auto integrand = [&](const double e) {
    return spec_rf(e) / sqrt(e * e - md * md);
  };

  const double beta = tools::beta(ep, mp);
  const double gamma = tools::gamma(ep, mp);
  const double k = sqrt(ed * ed - md * md);
  const double eminus = std::max(ed_lb, gamma * (ed - beta * k));
  const double eplus = std::min(ed_ub, gamma * (ed + beta * k));

  if (eminus > eplus) {
    return 0.0;
  }

  const double pre = 1.0 / (2.0 * gamma * beta);

  return pre * gauss_kronrod<double, GK_N>::integrate(integrand, eminus, eplus,
                                                      GK_MAX_LIMIT, 1e-6);
}

/**
 * Boost the spectrum dN/dx of a daughter particle in parent particles
 * rest-frame into the lab-frame.
 *
 * @param spec_rf unary function to compute spectrum in the rest-frame
 * @param x2 Boosted value of x: x2 = 2 * E / Q2
 * @param beta boost velocity
 * @param mu Scaled mass: mu = 2 * mass / Q1
 * @param x1_min Minimum value of x = 2 * E1 / Q1
 * @param x1_min Maximum value of x = 2 * E1 / Q1
 */
template <class F>
auto dndx_boost(F dndx_rf, double x2, double beta, double mu = 0.0, // NOLINT
                const double x1_min = 0.0,                          // NOLINT
                const double x1_max = std::numeric_limits<double>::infinity())
    -> double {
  using boost::math::quadrature::gauss_kronrod;
  using tools::sqr;
  static constexpr unsigned int GK_N = 15;
  static constexpr unsigned int GK_MAX_LIMIT = 7;

  if (beta < 0 || beta > 1.0) {
    return 0.0;
  }
  if (beta < std::numeric_limits<double>::epsilon()) {
    return dndx_rf(x2);
  }

  const double gamma = 1.0 / sqrt(1.0 - beta * beta);

  const auto integrand = [&dndx_rf, mu](double x1) {
    return dndx_rf(x1) / sqrt(x1 * x1 - mu * mu);
  };

  // Compute integration bounds
  const double k2 = sqrt(x2 * x2 - mu * mu);
  const double x1_minus = std::max(x1_min, sqr(gamma) * (x2 - beta * k2));
  const double x1_plus = std::min(x1_max, sqr(gamma) * (x2 + beta * k2));

  if (x1_minus > x1_minus) {
    return 0.0;
  }

  const double pre = 1.0 / (2.0 * beta);
  return pre * gauss_kronrod<double, GK_N>::integrate(
                   integrand, x1_minus, x1_plus, GK_MAX_LIMIT, 1e-6);
}

} // namespace blackthorn

#endif // BLACKTHORN_SPECTRA_BOOST_H
