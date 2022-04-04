#include "ChargedMesonRadiativeDecay.h"
#include "blackthorn/Models/Particles.h"
#include "blackthorn/Spectra/Boost.h"
#include "blackthorn/Spectra/Decay.h"

namespace blackthorn {

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
    return decay_spectrum<Muon>::dnde_neutrino(eg, eng_mu_pi_rf, Gen::Fst);
  };
  const auto dnde_m_rf = [](double eg) -> double {
    return decay_spectrum<Muon>::dnde_neutrino(eg, eng_mu_pi_rf, Gen::Snd);
  };

  // Maximum photon energy from muon decay in pion rest frame
  const double emax = (1.0 - sqr(me / mmu)) * (1.0 + beta);

  // contributions from muon decay: π -> [μ -> e + νe + νμ + γ] + νμ
  const double decay_e =
      br_m * boost_spectrum(dnde_e_rf, epi, mpi, enu, 0.0, 0.0, emax);
  const double decay_m =
      br_m * boost_spectrum(dnde_m_rf, epi, mpi, enu, 0.0, 0.0, emax);

  // δ-function contribution from: π -> e + νe
  const double e0_e = tools::energy_one_cm(mpi, 0.0, me);
  const double delta_e = br_e * boost_delta_function(e0_e, enu, 0.0, beta);
  // δ-function contribution from: π -> μ + νμ
  const double e0_m = tools::energy_one_cm(mpi, 0.0, mmu);
  const double delta_m = br_m * boost_delta_function(e0_m, enu, 0.0, beta);

  return {decay_e + delta_e, decay_m + delta_m, 0.0};
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

auto decay_spectrum<ChargedPion>::dnde_positron(
    const std::vector<double> &positron_energies, const double pion_energy)
    -> std::vector<double> {
  const auto f = [pion_energy](double x) {
    return dnde_positron(x, pion_energy);
  };
  return tools::vectorized_par(f, positron_energies);
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
