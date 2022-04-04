#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/Spectra/Boost.h"
#include "blackthorn/Spectra/Decay.h"
#include <boost/math/constants/constants.hpp>
#include <boost/math/tools/rational.hpp>
#include <cmath>
#include <dbg.h>
#include <gsl/gsl_sf_dilog.h>

namespace consts = boost::math::double_constants;

namespace blackthorn {

static constexpr double RATIO_ME_MMU = Electron::mass / Muon::mass;
// 1 / (1 - 8 r^2 + 8 r^6 - r^8 - 12 r^2 ln(r^2))
static constexpr double R_FACTOR = 1.0001870858234163;

static auto assert_beta_in_range(double beta) -> void {      // NOLINT
  assert((0.0 < beta && beta < 1.0) && "beta out of range"); // NOLINT
}

/// Helper class to compute the rest-frame and boosted photon spectra.
class dndx_photon {
  static constexpr double R = tools::sqr(Electron::mass / Muon::mass);
  static constexpr double EPS = std::numeric_limits<double>::epsilon();
  static constexpr double XMIN_RF = 0.0;
  static constexpr double XMAX_RF = 1 - R;
  static constexpr double PRE = StandardModel::alpha_em / (6.0 * M_PI);

  inline static auto rf_poly1(double x) -> double {
    using boost::math::tools::evaluate_polynomial;
    return evaluate_polynomial({-102.0, 148.0, -147.0, 156.0, -55.0}, x) / 6.0;
  }

  inline static auto rf_poly2(double x) -> double {
    using boost::math::tools::evaluate_polynomial;
    return evaluate_polynomial({6.0, -10.0, 12.0, -12.0, 4.0}, x);
  }

  inline static auto boost_poly1(double x) -> double {
    using boost::math::tools::evaluate_polynomial;
    return evaluate_polynomial({102.0, 0.0, -191.0, 92.0, -21.0}, x);
  }

  inline static auto boost_poly2(double x) -> double {
    using boost::math::tools::evaluate_polynomial;
    return evaluate_polynomial({-36.0, -72.0 - 60.0 * log(x), 72.0, -36.0, 8.0},
                               x);
  }

public:
  static auto rest_frame(double x) -> double {
    using boost::math::tools::evaluate_polynomial;
    if (x < XMIN_RF || XMAX_RF < x) {
      return 0.0;
    }
    return PRE * (rf_poly1(x) + rf_poly2(x) * log((1.0 - x) / R)) / x;
  }

  // Compute the spectrum dN/dx given a muon with velocity beta
  static auto boosted(double x2, double beta) -> double { // NOLINT
    using tools::sqr;
    assert_beta_in_range(beta);
    if (beta < std::numeric_limits<double>::epsilon()) {
      return rest_frame(x2);
    }
    if (x2 < 0.0) {
      throw std::domain_error("x must be greater than zero.");
    }
    if (beta < 0.0 || beta > 1.0) {
      throw std::domain_error("Invalid boost velocity. Must be 0 < beta < 1");
    }
    // if !(x2 < (1-r^2)(1+beta)), then integrand is zero
    if (x2 >= XMAX_RF * (1 + beta)) {
      return 0.0;
    }

    const double xm = x2 / (1.0 + beta);
    const double xp = std::min(XMAX_RF, x2 / (1.0 - beta));

    const double pre = PRE * 0.5 / beta;
    const double xmm = 1.0 - xm;
    const double xpm = 1.0 - xp;

    const double rp = (boost_poly1(xp) + boost_poly2(xp) * log(xpm / R)) / xp;
    const double rm = (boost_poly1(xm) + boost_poly2(xm) * log(xmm / R)) / xm;
    const double plm = gsl_sf_dilog(xmm);
    const double plp = gsl_sf_dilog(xpm);
    const double l1pm = log(xpm / xmm);
    const double l2pm = log(xp / xm);

    return pre * ((rp - rm) / 6.0 - 10.0 * (plp - plm) +
                  8.0 / 3.0 * (7.0 * log(xp / xm) + 4.0 * log(xpm / xmm)));
  }
};

auto decay_spectrum<Muon>::dnde_photon(double e, double muon_energy) // NOLINT
    -> double {
  if (muon_energy < Muon::mass) {
    return 0.0;
  }
  const double beta = tools::beta(muon_energy, Muon::mass);
  const double e_to_x = 2.0 / muon_energy;
  const double x = e * e_to_x;
  return dndx_photon::boosted(x, beta) * e_to_x;
}

// ===========================================================================
// ---- Positron Spectra -----------------------------------------------------
// ===========================================================================

/// Helper class to compute the rest-frame and boosted positron spectra.
struct dndx_positron { // NOLINT
  static constexpr double R = RATIO_ME_MMU;
  static constexpr double R2 = tools::sqr(RATIO_ME_MMU);
  static constexpr double R4 = tools::sqr(R2);
  static constexpr double XMIN_RF = 2 * R; // Ee == me
  static constexpr double XMAX_RF = 1.0 + R2;

  static auto rest_frame(double x) -> double {
    using std::fma;
    using tools::sqr;
    if (x < XMIN_RF || XMAX_RF < x) {
      return 0.0;
    }
    return 2 * R_FACTOR * sqrt(sqr(x) - 4 * R2) *
           fma(fma(-2.0, x, 3.0 + 3.0 * R2), x, -4 * R2);
  }

  static auto boosted(double x2, double beta) -> double {
    using tools::powi;
    using tools::sqr;

    if (beta < 0.0 || beta > 1.0) {
      return 0.0;
    }
    if (std::abs(1.0 - beta) < std::numeric_limits<double>::epsilon()) {
      return 5.0 / 6.0 * (1.0 + 4.0 * R + R2) * powi<4>(1 - R);
    }
    if (beta < std::numeric_limits<double>::epsilon()) {
      return rest_frame(x2);
    }

    const double pre = R_FACTOR / (2.0 * beta);
    const double gamma = tools::gamma(beta);
    const double mu2 = 2.0 * R / gamma; // mu2 = me / emu
    if (x2 < mu2) {
      return 0.0;
    }
    const double k2 = sqrt(sqr(x2) - sqr(mu2));
    const double xm = std::max(XMIN_RF, sqr(gamma) * (x2 - beta * k2));
    const double xp = std::min(XMAX_RF, sqr(gamma) * (x2 + beta * k2));
    if (xp < xm) {
      return 0.0;
    }

    static constexpr double B1 = 3.0 + 3.0 * R2;
    static constexpr double B2 = -8.0 * R2;
    const double pp = xp * std::fma(std::fma(-4.0 / 3.0, xp, B1), xp, B2);
    const double pm = xm * std::fma(std::fma(-4.0 / 3.0, xm, B1), xm, B2);
    return pre * (pp - pm);
  }
};

auto decay_spectrum<Muon>::dnde_positron(double e, double muon_energy) // NOLINT
    -> double {
  if (muon_energy < Muon::mass) {
    return 0.0;
  }

  const double e_to_x = 2.0 / muon_energy;
  const double x = e * e_to_x;

  if (muon_energy - Muon::mass < std::numeric_limits<double>::epsilon()) {
    return dndx_positron::rest_frame(x) * e_to_x;
  }

  const double beta = tools::beta(muon_energy, Muon::mass);
  return dndx_positron::boosted(x, beta) * e_to_x;
}

// ===========================================================================
// ---- Neutrino Spectra -----------------------------------------------------
// ===========================================================================

struct dndx_neutrino { // NOLINT
  static constexpr double R = RATIO_ME_MMU;
  static constexpr double R2 = tools::sqr(R);
  static constexpr double R4 = tools::sqr(R2);
  static constexpr double R6 = R4 * R2;
  static constexpr double XMIN_RF = 0.0;
  static constexpr double XMAX_RF = 1.0 - R2;

  /// Compute electron-neutrino spectrum in the muon rest-frame
  static auto rest_frame_electron(double x) {
    using tools::sqr;
    if (x < XMIN_RF || XMAX_RF < x) {
      return 0.0;
    }
    return R_FACTOR * 12.0 * sqr(x) * sqr(XMAX_RF - x) / (1.0 - x);
  }

  /// Compute muon-neutrino spectrum in the muon rest-frame
  static auto rest_frame_muon(double x) {
    using boost::math::tools::evaluate_polynomial;
    using tools::powi;
    using tools::sqr;

    static constexpr double C0 = 6.0 - 6.0 * R2 - 6.0 * R4 + 6.0 * R6;
    static constexpr double C1 = -22.0 + 18.0 * R2 + 6.0 * R4 - 2.0 * R6;
    static constexpr double C2 = 30.0 - 18.0 * R2;
    static constexpr double C3 = -18.0 + 6.0 * R2;

    if (x < XMIN_RF || XMAX_RF < x) {
      return 0.0;
    }

    const double poly = evaluate_polynomial({C0, C1, C2, C3, 4.0}, x);
    return R_FACTOR * sqr(x) * poly / powi<3>(1 - x);
  }

  /// Compute electron-neutrino spectrum give muon velocity
  static auto boosted_electron(double x2, double beta) {
    using tools::sqr;

    if (x2 < 0.0) {
      throw std::domain_error("x must be greater than zero.");
    }
    if (beta < 0.0 || beta > 1.0) {
      throw std::domain_error("Invalid boost velocity. Must be 0 < beta < 1");
    }

    if (beta < std::numeric_limits<double>::epsilon()) {
      return rest_frame_electron(x2);
    }

    if (x2 >= XMAX_RF * (1 + beta)) {
      return 0.0;
    }

    const double pre = R_FACTOR / (2.0 * beta);
    const double xm = x2 / (1.0 + beta);
    const double xp = std::min(XMAX_RF, x2 / (1.0 - beta));

    return 2.0 * pre *
           ((xm - xp) *
                (-3.0 * (xm + xp) + 2.0 * (3.0 * R4 + sqr(xm) + xm * xp +
                                           sqr(xp) + 3.0 * R2 * (xm + xp))) -
            6.0 * R4 * log((1.0 - xp) / (1.0 - xm)));
  }

  /// Compute muon-neutrino spectrum give muon velocity
  static auto boosted_muon(double x2, double beta) {
    using boost::math::tools::evaluate_polynomial;
    using std::fma;
    using tools::sqr;

    static constexpr double C0 = -5.0 + 9.0 * R2 - 18.0 * R4;
    static constexpr double C1 = 10.0 - 18.0 * R2 + 18.0 * R4 + 6.0 * R6;
    static constexpr double C2 = 4.0;
    static constexpr double C3 = -22.0 + 18.0 * R2;
    static constexpr double C4 = 17.0 - 9.0 * R2;
    static constexpr double C5 = -4.0;
    static constexpr double C6 = -2.0 * R4 * (3.0 - R2);

    if (x2 < 0.0) {
      throw std::domain_error("x must be greater than zero.");
    }
    if (beta < 0.0 || beta > 1.0) {
      throw std::domain_error("Invalid boost velocity. Must be 0 < beta < 1");
    }

    if (beta < std::numeric_limits<double>::epsilon()) {
      return rest_frame_electron(x2);
    }

    if (x2 >= XMAX_RF * (1 + beta)) {
      return 0.0;
    }

    const double pre = R_FACTOR / (2.0 * beta);
    const double xm = x2 / (1.0 + beta);
    const double xp = std::min(XMAX_RF, x2 / (1.0 - beta));
    const double xmm = 1.0 - xm;
    const double xpm = 1.0 - xp;

    const double rp =
        evaluate_polynomial({C0, C1, C2, C3, C4, C5}, xp) / (3.0 * sqr(xpm));
    const double rm =
        evaluate_polynomial({C0, C1, C2, C3, C4, C5}, xm) / (3.0 * sqr(xmm));
    return pre * (rp - rm + C6 * log(xpm / xmm));
  }
};

auto decay_spectrum<Muon>::dnde_neutrino(double e, double muon_energy) // NOLINT
    -> NeutrinoSpectrum<double> {
  using tools::powi;
  using tools::sqr;

  if (muon_energy < Muon::mass) {
    return {0.0, 0.0, 0.0};
  }

  const double e_to_x = 2.0 / muon_energy;
  const double x = e * e_to_x;

  if (muon_energy - Muon::mass < std::numeric_limits<double>::epsilon()) {
    return {dndx_neutrino::rest_frame_electron(x) * e_to_x,
            dndx_neutrino::rest_frame_muon(x) * e_to_x, 0.0};
  }

  const double beta = tools::beta(muon_energy, Muon::mass);
  return {dndx_neutrino::boosted_electron(x, beta) * e_to_x,
          dndx_neutrino::boosted_muon(x, beta) * e_to_x, 0.0};
}

auto decay_spectrum<Muon>::dnde_neutrino(double e, double muon_energy, // NOLINT
                                         Gen g) -> double {
  using tools::powi;
  using tools::sqr;

  if (muon_energy < Muon::mass) {
    return 0.0;
  }

  const double e_to_x = 2.0 / muon_energy;
  const double x = e * e_to_x;

  if (muon_energy - Muon::mass < std::numeric_limits<double>::epsilon()) {
    switch (g) {
    case Gen::Fst:
      return dndx_neutrino::rest_frame_electron(x) * e_to_x;
    case Gen::Snd:
      return dndx_neutrino::rest_frame_muon(x) * e_to_x;
    default:
      return 0.0;
    }
  }

  const double beta = tools::beta(muon_energy, Muon::mass);
  switch (g) {
  case Gen::Fst:
    return dndx_neutrino::boosted_electron(x, beta) * e_to_x;
  case Gen::Snd:
    return dndx_neutrino::boosted_muon(x, beta) * e_to_x;
  default:
    return 0.0;
  }
}

// ===========================================================================
// ---- Vectorized Versions --------------------------------------------------
// ===========================================================================

auto decay_spectrum<Muon>::dnde_photon(const std::vector<double> &egams,
                                       const double emu)
    -> std::vector<double> {
  const auto f = [&](double x) { return dnde_photon(x, emu); };
  return tools::vectorized_par(f, egams);
}

auto decay_spectrum<Muon>::dnde_photon(const py::array_t<double> &egams,
                                       double emu) -> py::array_t<double> {
  const auto f = [&](double x) { return dnde_photon(x, emu); };
  return tools::vectorized(f, egams);
}

auto decay_spectrum<Muon>::dnde_neutrino(const std::vector<double> &enus,
                                         const double emu)
    -> NeutrinoSpectrum<std::vector<double>> {
  std::vector<double> res_e(enus.size(), 0.0);
  std::vector<double> res_mu(enus.size(), 0.0);
  std::vector<double> res_tau(enus.size(), 0.0);
  std::transform(enus.begin(), enus.end(), res_e.begin(),
                 [&](double x) { return dnde_neutrino(x, emu, Gen::Fst); });
  std::transform(enus.begin(), enus.end(), res_e.begin(),
                 [&](double x) { return dnde_neutrino(x, emu, Gen::Snd); });
  return {res_e, res_mu, res_tau};
}

auto decay_spectrum<Muon>::dnde_positron(const std::vector<double> &eps,
                                         const double emu)
    -> std::vector<double> {
  const auto f = [&](double x) { return dnde_positron(x, emu); };
  return tools::vectorized(f, eps);
}

} // namespace blackthorn
