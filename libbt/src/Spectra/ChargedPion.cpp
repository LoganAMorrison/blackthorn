#include "ChargedMesonRadiativeDecay.h"
#include "blackthorn/Models/Particles.h"
#include "blackthorn/Spectra/Boost.h"
#include "blackthorn/Spectra/Decay.h"

namespace blackthorn {

// /**
//  * Helper struct to compute pi -> l + nu + a or k -> l + nu + a
//  */
// template <class L> static auto dnde_x_to_lva(double egam) -> double {
//   static constexpr double ml = L::mass;
//   static constexpr double mp = ChargedPion::mass;
//   static constexpr double fp = ChargedPion::decay_const;
//   static constexpr double ff_vec = ChargedPion::ff_vec;
//   static constexpr double ff_axi = ChargedPion::ff_axi;
//   static constexpr double ff_vec_slope = ChargedPion::ff_vec_slope;
//   static constexpr double eps = 1.0;

//   const double x = 2.0 * egam / mp;
//   const double r = tools::sqr(ml / mp);

//   if (x < 0.0 || 1.0 - r < x) {
//     return 0.0;
//   }
//   const double vp = ff_vec * (1.0 - ff_vec_slope * (1.0 - x));
//   const double xp = 1.0 - x;
//   const double x2 = tools::sqr(x);

//   const double t2 = r + x - 1.0;
//   const double t3 = -2.0 * x;
//   const double t4 = x - 2.0;

//   return (StandardModel::alpha_em *
//           (t2 * (24.0 * tools::sqr(fp) * r * xp *
//                      (-4.0 * r * xp + tools::sqr(t4)) +
//                  pow(mp, 2) * t2 * (2 + r + t3) *
//                      (tools::sqr(ff_axi) + tools::sqr(vp)) * pow(x, 4) +
//                  12.0 * eps * fp * mp * r * xp * M_SQRT2 * x2 *
//                      (ff_axi * (1.0 + r + t3) + vp * x)) +
//            12.0 * fp * r * tools::sqr(xp) *
//                (fp * (-4.0 + 4.0 * tools::sqr(r) - 4.0 * r * x - 2.0 * t4 *
//                x) +
//                 eps * mp * M_SQRT2 * x2 * (-vp * x + ff_axi * (x - 2.0 * r)))
//                 *
//                log(r / xp))) /
//          (24. * tools::sqr(fp * (1.0 - r) * xp) * mp * M_PI * r * x);
// }

auto decay_spectrum<ChargedPion>::dnde_photon(const double egam,
                                              const double epi) -> double {
  using Self = ChargedPion;

  static constexpr double eng_mu_pi_rf =
      (tools::sqr(Self::mass) + tools::sqr(Muon::mass)) / (2.0 * Self::mass);

  if (epi < Self::mass) {
    return 0.0;
  }
  const auto spec_rf = [&](double eg) {
    // contribution: π -> e + νe + γ
    const double eva = Self::BR_PI_TO_E_NUE * dnde_x_to_lva<Self, Electron>(eg);

    // contribution: π -> μ + νμ + γ
    const double mva = Self::BR_PI_TO_MU_NUMU * dnde_x_to_lva<Self, Muon>(eg);

    // contribution: π -> [μ -> e + νe + νμ + γ] + νμ
    const double mv = Self::BR_PI_TO_MU_NUMU *
                      decay_spectrum<Muon>::dnde_photon(eg, eng_mu_pi_rf);

    return eva + mva + mv;
  };

  return boost_spectrum(spec_rf, epi, ChargedPion::mass, egam, 0.0);
}

auto decay_spectrum<ChargedPion>::dnde_neutrino(const double enu,
                                                const double epi)
    -> NeutrinoSpectrum<double> {
  using Self = ChargedPion;
  using tools::sqr;

  static constexpr double mpi = Self::mass;
  static constexpr double mmu = Muon::mass;
  static constexpr double me = Electron::mass;
  static constexpr double eng_mu_pi_rf = (sqr(mpi) + sqr(mmu)) / (2.0 * mpi);
  static constexpr double br_m = Self::BR_PI_TO_MU_NUMU;
  static constexpr double br_e = Self::BR_PI_TO_E_NUE;

  const double gamma = tools::gamma(epi, mpi);
  const double beta = tools::beta(epi, mpi);

  if (epi < mpi) {
    return {0.0, 0.0, 0.0};
  }
  const auto dnde_e_rf = [](double eg) -> double {
    return decay_spectrum<Muon>::dnde_neutrino(eg, eng_mu_pi_rf).electron;
  };
  const auto dnde_m_rf = [](double eg) -> double {
    return decay_spectrum<Muon>::dnde_neutrino(eg, eng_mu_pi_rf).muon;
  };

  // Neutrino energies form: π -> ℓ + νℓ
  const double e0_e = (sqr(mpi) - sqr(me)) / (2.0 * mpi);
  const double e0_m = (sqr(mpi) - sqr(mmu)) / (2.0 * mpi);

  // contributions from muon decay: π -> [μ -> e + νe + νμ + γ] + νμ
  const double dnde_e = br_m * boost_spectrum(dnde_e_rf, epi, mpi, enu, 0.0);
  const double dnde_m = br_m * boost_spectrum(dnde_m_rf, epi, mpi, enu, 0.0);

  // δ-function contribution from: π -> e + νe
  const double dnde_e_d = br_e * boost_delta_function(e0_e, enu, 0.0, beta);
  // δ-function contribution from: π -> μ + νμ
  const double dnde_m_d = br_m * boost_delta_function(e0_m, enu, 0.0, beta);

  return {dnde_e + dnde_e_d, dnde_m + dnde_m_d, 0.0};
}

auto decay_spectrum<ChargedPion>::dnde_positron(const double ep,
                                                const double epi) -> double {
  using Self = ChargedPion;
  using tools::sqr;

  static constexpr double mpi = Self::mass;
  static constexpr double mmu = Muon::mass;
  static constexpr double me = Electron::mass;
  static constexpr double eng_mu_pi_rf = (sqr(mpi) + sqr(mmu)) / (2.0 * mpi);
  static constexpr double br_m = Self::BR_PI_TO_MU_NUMU;
  static constexpr double br_e = Self::BR_PI_TO_E_NUE;

  const double gamma = tools::gamma(epi, mpi);
  const double beta = tools::beta(epi, mpi);

  if (epi < mpi) {
    return 0.0;
  }
  const auto dnde_rf = [](double e) -> double {
    return decay_spectrum<Muon>::dnde_positron(e, eng_mu_pi_rf);
  };

  // Neutrino energy from: π -> e + νe
  const double e0_e = (sqr(mpi) - sqr(me)) / (2.0 * mpi);

  // contributions from muon decay: π -> [μ -> e + νe + νμ + γ] + νμ
  const double dnde_m = br_m * boost_spectrum(dnde_rf, epi, mpi, ep, me);

  // δ-function contribution from: π -> e + νe
  const double dnde_e_d = br_e * boost_delta_function(e0_e, ep, me, beta);

  return dnde_m + dnde_e_d;
}

auto decay_spectrum<ChargedPion>::dnde_photon(const std::vector<double> &egams,
                                              const double epi)
    -> std::vector<double> {
  const auto f = [=](double x) { return dnde_photon(x, epi); };
  return tools::vectorized_par(f, egams);
}

auto decay_spectrum<ChargedPion>::dnde_photon(const py::array_t<double> &egams,
                                              double epi)
    -> py::array_t<double> {
  const auto f = [=](double x) { return dnde_photon(x, epi); };
  return tools::vectorized(f, egams);
}

auto decay_spectrum<ChargedPion>::dnde_neutrino(const std::vector<double> &enus,
                                                const double epi)
    -> NeutrinoSpectrum<std::vector<double>> {
  std::vector<double> res_e(enus.size(), 0.0);
  std::vector<double> res_mu(enus.size(), 0.0);
  std::vector<double> res_tau(enus.size(), 0.0);
  for (size_t i = 0; i < enus.size(); ++i) { // NOLINT
    const auto result = dnde_neutrino(enus[i], epi);
    res_e[i] = result.electron;
    res_mu[i] = result.muon;
  }
  return {res_e, res_mu, res_tau};
}

} // namespace blackthorn
