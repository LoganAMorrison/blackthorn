#include "blackthorn/Tools.h"
#include <gsl/gsl_sf_dilog.h>

namespace blackthorn::tools {

/**
 * Compute the dilogarithm.
 */
auto dilog(const std::complex<double> &z) -> std::complex<double> {
  const double r = std::abs(z);
  const double theta = std::arg(z);

  gsl_sf_result res_re{};
  gsl_sf_result res_im{};
  gsl_sf_complex_dilog_e(r, theta, &res_re, &res_im);

  return std::complex{res_re.val, res_im.val};
}

/**
 * Compute the Scalar C0 function C0(s1, s12, s2; m1, m2, m3) w/ s1=s12=0 and
 * m2=m3.
 */
auto scalar_c0_1(double s2, double m1, double m2) -> std::complex<double> {
  using tools::kallen_lambda;

  return (pow(log((pow(m1, 2) - pow(m2, 2) - s2 +
                   std::sqrt(kallen_lambda(pow(m1, 2), pow(m2, 2), s2))) /
                  (pow(m1, 2) - pow(m2, 2) + s2 +
                   std::sqrt(kallen_lambda(pow(m1, 2), pow(m2, 2), s2)))),
              2) /
              2. +
          dilog(1 - pow(m1, 2) / pow(m2, 2)) +
          dilog((2 * s2) /
                (pow(m1, 2) - pow(m2, 2) + s2 +
                 std::sqrt(kallen_lambda(pow(m1, 2), pow(m2, 2), s2)))) -
          dilog((2 * s2) /
                (-pow(m1, 2) + pow(m2, 2) + s2 +
                 std::sqrt(kallen_lambda(pow(m1, 2), pow(m2, 2), s2))))) /
         s2;
}

/**
 * Compute the Scalar C0 function C0(s1, s12, s2; m1, m2, m3) w/ s1=s12=0 and
 * m1=m2.
 */
auto scalar_c0_2(double s2, double m1, double m3) -> std::complex<double> {
  using tools::kallen_lambda;
  return (pow(log((-pow(m1, 2) + pow(m3, 2) - s2 +
                   std::sqrt(kallen_lambda(pow(m1, 2), pow(m3, 2), s2))) /
                  (-pow(m1, 2) + pow(m3, 2) + s2 +
                   std::sqrt(kallen_lambda(pow(m1, 2), pow(m3, 2), s2)))),
              2) /
              2. +
          dilog(1 - pow(m3, 2) / pow(m1, 2)) -
          dilog((2 * s2) /
                (pow(m1, 2) - pow(m3, 2) + s2 +
                 std::sqrt(kallen_lambda(pow(m1, 2), pow(m3, 2), s2)))) +
          dilog((2 * s2) /
                (-pow(m1, 2) + pow(m3, 2) + s2 +
                 std::sqrt(kallen_lambda(pow(m1, 2), pow(m3, 2), s2))))) /
         s2;
}

} // namespace blackthorn::tools
