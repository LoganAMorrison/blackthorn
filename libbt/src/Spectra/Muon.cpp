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
    const double gamma2 = sqr(tools::gamma(beta));
    const double xm = std::max(XMIN_RF, gamma2 * x2 * (1.0 - beta));
    const double xp = std::min(XMAX_RF, gamma2 * x2 * (1.0 + beta));

    if (xp - xm < 0) {
      return 0.0;
    }

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
    using tools::powi;
    using tools::sqr;
    if (x < XMIN_RF || XMAX_RF < x) {
      return 0.0;
    }
    static constexpr double B1 = -18 + 6 * R2;
    static constexpr double B2 = 30 - 18 * R2;
    static constexpr double B3 = -22 + 18 * R2 + 6 * R4 - 2 * R6;
    static constexpr double B4 = 6 - 6 * R2 - 6 * R4 + 6 * R6;
    const double poly =
        std::fma(std::fma(std::fma(std::fma(4, x, B1), x, B2), x, B3), x, B3);

    return R_FACTOR * sqr(x) * poly / powi<3>(1 - x);
  }

  /// Compute electron-neutrino spectrum give muon velocity
  static auto boosted_electron(double x2, double beta) {
    using tools::sqr;

    if (beta < 0.0 || 1.0 < beta) {
      return 0.0;
    }
    if (beta < std::numeric_limits<double>::epsilon()) {
      return rest_frame_electron(x2);
    }

    if (x2 < XMIN_RF || XMAX_RF * sqrt((1.0 + beta) / (1.0 - beta)) <= x2) {
      return 0.0;
    }

    const double pre = R_FACTOR / (2.0 * beta);
    const double xm = std::max(XMIN_RF, x2 / (1.0 + beta));
    const double xp = std::min(XMAX_RF, x2 / (1.0 - beta));

    if (xm > xp) {
      return 0.0;
    }
    static constexpr double B1 = 3.0 - 6.0 * R2;
    static constexpr double B2 = -6.0 * R4;
    const double pp = xp * std::fma(std::fma(-2.0, xp, B1), xp, B2);
    const double pm = xm * std::fma(std::fma(-2.0, xm, B1), xm, B2);
    const double lpm = log((1 - xp) / (1 - xm));
    return 2 * pre * (pp - pm - 6.0 * R4 * lpm);
  }

  /// Compute muon-neutrino spectrum give muon velocity
  static auto boosted_muon(double x2, double beta) {
    using std::fma;
    using tools::sqr;

    if (beta < 0.0 || 1.0 < beta) {
      return 0.0;
    }
    if (beta < std::numeric_limits<double>::epsilon()) {
      return rest_frame_muon(x2);
    }

    if (x2 < XMIN_RF || XMAX_RF * sqrt((1.0 + beta) / (1.0 - beta)) < x2) {
      return 0.0;
    }

    const double pre = R_FACTOR / (2.0 * beta);
    const double xm = std::max(XMIN_RF, x2 / (1.0 + beta));
    const double xp = std::min(XMAX_RF, x2 / (1.0 - beta));

    if (xm > xp) {
      return 0.0;
    }

    static constexpr double B1 = 17 - 9 * R2;
    static constexpr double B2 = -22 + 18 * R2;
    static constexpr double B3 = 9 - 9 * R2 + 18 * R4;
    static constexpr double B4 = -18 * R4 + 6 * R6;
    static constexpr double C1 = 2 * R4 * (-3 + R2);

    const double np =
        xp * fma(fma(fma(fma(-4, xp, B1), xp, B2), xp, B3), xp, B4);
    const double nm =
        xm * fma(fma(fma(fma(-4, xm, B1), xm, B2), xm, B3), xm, B4);
    const double dp = fma(fma(3, xp, -6), xp, 3);
    const double dm = fma(fma(3, xm, -6), xm, 3);
    const double lpm = log((1 - xp) / (1 - xm));

    return pre * (np / dp - nm / dm + C1 * lpm);
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

// auto decay_spectrum<Muon>::dnde_photon(double egam, double emu) // NOLINT
//     -> double {
//   using tools::sqr;

//   if (emu < Muon::mass) {
//     return 0.0;
//   }

//   if (emu - Muon::mass < std::numeric_limits<double>::epsilon()) {
//     return 0.0;
//   }

//   const double gamma = emu / Muon::mass;
//   const double beta = sqrt(1.0 - sqr(Muon::mass / emu));

//   // Rescaled variables
//   const double y = 2 * egam / Muon::mass;
//   constexpr double r = sqr(Electron::mass / Muon::mass);
//   // Re-rescaled variables
//   const double x = y * gamma;

//   // Bounds check
//   if (x < 0.0 || x >= (1.0 - r) / (1.0 - beta)) {
//     return 0.0;
//   }

//   // Upper bound on 'angular' variable (w = 1 - beta * ctheta)
//   const double wp = (x < (1.0 - r) / (1.0 + beta)) ? 1.0 + beta : (1.0 - r) /
//   x; const double wm = 1.0 - beta;

//   const double xp = x * wp;
//   const double xm = x * wm;

//   // Polynomial contribution
//   double result =
//       ((xm - xp) *
//        (102.0 + xm * xp *
//                     (191.0 + 21.0 * sqr(xm) + xm * (-92.0 + 21.0 * xp) +
//                      xp * (-92.0 + 21.0 * xp))) /
//        (12.0 * xm * xp * beta));
//   // Logarithmic contributions
//   result += (9.0 + xm * (18.0 + xm * (-18.0 + (9.0 - 2.0 * xm) * xm))) *
//             log((1.0 - xm) / r) / (3.0 * xm * beta);
//   result += (-9.0 + xp * (-18.0 + xp * (18.0 + xp * (-9.0 + 2 * xp)))) *
//             log((1.0 - xp) / r) / (3.0 * xp * beta);
//   result += 5.0 / beta *
//             (log((1.0 - xm) / r) * log(xm) - log((1.0 - xp) / r) * log(xp));
//   result += 4.0 / (3.0 * beta) *
//             (4.0 * log((1.0 - xp) / (1.0 - xm)) + 7.0 * log(xp / xm));
//   // PolyLog terms
//   result += 5.0 / beta * (gsl_sf_dilog(1.0 - xm) - gsl_sf_dilog(1.0 - xp));

//   return result * StandardModel::alpha_em / (3.0 * M_PI * emu);
// }

} // namespace blackthorn
