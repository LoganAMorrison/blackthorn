#include "blackthorn/Spectra/Boost.h"
#include "blackthorn/Tools.h"

namespace blackthorn {

/**
 * Compute the boosted energy of a daugther particle when boosted from the
 * lab-frame to the rest-frame of the parent particle.
 */
auto boost_eng(double ep, double mp, double ed, double md, double zl)
    -> double {
  const double b = tools::beta(ep, mp);
  const double g = tools::gamma(ep, mp);
  const double kt = sqrt(1 - tools::sqr(md / ed));
  return g * ed * (1 + kt * b * zl);
}

/**
 * Returns the Jacobian for boost integrals when boosting from the lab frame
 * to the parent particle's rest frame.
 *
 * # Notes
 *
 * The Jacobian is given by:
 *
 *     J = det({
 *              {    dER/dEL,    dER/dcostL }
 *              { dcostR/dEl, dcostR/dcostL }
 *         })
 *
 * where `ER` is the energy of the daughter particle in the parent particle's
 * rest-frame, `costR` is the cosine of the angle the daughter particle makes
 * w.r.t. the z-axis. The quantities with `L` are in the lab-frame.
 *
 * # Arguments
 *
 * - `ep`: Energy of the parent particle in lab-frame
 * - `mp`: Mass of the parent particle
 * - `ed`: Energy of the daughter particle in lab-frame
 * - `mp`: Mass of the daughter particle
 * - `cost`: Cosine of the angle the daughter particle makes wrt z-axis in
 * lab-frame
 */
auto boost_jac(double ep, double mp, double ed, double md, double zl)
    -> double {
  const double b = tools::beta(ep, mp);
  const double g = tools::gamma(ep, mp);
  const double kt = sqrt(1 - tools::sqr(md / ed));
  return kt / (g * (1.0 + b * kt * zl));
}

/**
 * Boost a δ-function spectrum centered at `e0` from the rest-frame of the
 * parent particle to the lab-frame.
 *
 * @param e0 center of the dirac-delta spectrum in rest-frame
 * @param e energy of the product in the lab frame.
 * @param m mass of the product
 * @param beta boost velocity of the decaying particle
 */
auto boost_delta_function(double e0, double e, double m, double beta)
    -> double {
  using tools::sqr;

  if (beta <= 0.0 || beta > 1.0) {
    return 0.0;
  }
  if (e < m) {
    return 0.0;
  }

  const double gamma = tools::gamma(beta);
  const double k = sqrt(sqr(e) - sqr(m));
  const double eminus = gamma * (e - beta * k);
  const double eplus = gamma * (e + beta * k);

  if (e0 <= eminus || eplus <= e0) {
    return 0.0;
  }

  const double k0 = sqrt(tools::sqr(e0) - tools::sqr(m));
  return 1.0 / (2 * gamma * beta * k0);
}

} // namespace blackthorn
