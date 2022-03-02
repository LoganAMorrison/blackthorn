// This file contains the implementation of the gamma-ray spectrum from the
// decay of a RH neutrino through: N -> ν + π⁺ + π⁻. The FSR spectrum is
// implemented using the analytic matrix element for N -> ν + π⁺ + π⁻ + γ and
// RAMBO.

#include "blackthorn/Models/Particles.h"
#include "blackthorn/Models/RhNeutrino.h"
#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/Spectra.h"
#include "blackthorn/Spectra/Conv.h"
#include "blackthorn/Tools.h"

namespace blackthorn {

// Structure holding the configuration how total spectrum from N -> ν + π⁺ + π⁻
// should be computed.
struct VPiPiConfig { // NOLINT
  unsigned int nbins;
  size_t nevents_edist;
  size_t nevents_fsr;
};

// Squared matrix element for N -> ν + π⁺ + π⁻ + γ
static auto msqrd_v_pi_pi_a(const std::array<LVector<double>, 4> &ps, double M,
                            double theta) -> double {
  const auto pl = ps[0];
  const auto p1 = ps[1];
  const auto p2 = ps[2];
  const auto k = ps[3];
  const double M2 = M * M;
  const double x2 = dot(p1, pl) / M2;
  const double x3 = dot(p2, pl) / M2;
  const double x4 = dot(p1, k) / M2;
  const double x5 = dot(p2, k) / M2;
  const double x6 = dot(pl, k) / M2;
  const double bp2 = tools::sqr(ChargedPion::mass / M);
  constexpr double qe = StandardModel::qe;
  constexpr double cw = StandardModel::cw;
  constexpr double sw = StandardModel::sw;
  constexpr double gf = StandardModel::g_fermi;

  return (std::pow(qe, 2) * std::pow(gf, 2) * std::pow(M, 2) *
          std::pow(1 - 2 * std::pow(sw, 2), 2) *
          (4 * std::pow(bp2, 2) * std::pow(x4 + x5, 2) * (x2 + x3 + x6) +
           x4 * x5 *
               (2 * std::pow(x2, 2) * (-1 + 8 * x3 + 4 * x5 + 4 * x6) +
                (x3 + x6) * ((-1 + 2 * x5 + 2 * x6) * (-1 + 4 * x6) +
                             x3 * (-2 + 8 * x4 + 8 * x6) +
                             x4 * (-2 + 8 * x5 + 8 * x6)) +
                x2 * (1 + 16 * std::pow(x3, 2) - 2 * x5 - 8 * x6 +
                      16 * x5 * x6 + 16 * std::pow(x6, 2) +
                      4 * x3 * (-3 + 2 * x4 + 2 * x5 + 8 * x6) +
                      x4 * (-2 + 8 * x5 + 8 * x6))) +
           bp2 *
               (8 * std::pow(x2, 2) * x4 * x5 + 8 * std::pow(x3, 2) * x4 * x5 +
                x6 * (-std::pow(x5, 2) + std::pow(x4, 2) * (-1 + 8 * x5) +
                      2 * x4 * x5 * (-3 + 4 * x5 + 8 * x6)) +
                x3 * (std::pow(x4, 2) * (-1 + 8 * x5) +
                      std::pow(x5, 2) * (-1 + 8 * x6) +
                      2 * x4 * x5 * (-3 + 4 * x5 + 12.0 * x6)) +
                x2 * ((-1 + 8 * x3) * std::pow(x5, 2) +
                      std::pow(x4, 2) * (-1 + 8 * x3 + 8 * x5 + 8 * x6) +
                      2 * x4 * x5 * (-3 + 16 * x3 + 4 * x5 + 12.0 * x6)))) *
          std::pow(std::sin(2 * theta), 2)) /
         (std::pow(cw, 4) * std::pow(x4, 2) * std::pow(x5, 2));
}

// Construct the energy distributions of the final state particles from N -> ν +
// π⁺ + π⁻
static auto energy_distributions_v_pi_pi(const RhNeutrinoMeV &model,
                                         const VPiPiConfig &config)
    -> std::array<EnergyHist<LinAxis>, 3> {

  const auto msqrd = SquaredAmplitudeNToVPiPi(model);
  static constexpr std::array<double, 3> fsp_masses = {0.0, ChargedPion::mass,
                                                       ChargedPion::mass};
  const auto nbins = config.nbins;
  const auto nevents = config.nevents_edist;
  return energy_distributions_linear(msqrd, model.mass(), fsp_masses,
                                     {nbins, nbins, nbins}, nevents);
}

// Given the energy distribution of a final state pion, convolve the
// distribution with the pion decay spectrum.
// Computes:
//  ∫dEπ (dN/dE)(Eγ, Eπ) P(Eπ)
// where P(Eπ) is the probability of the pion having energy Eπ.
static auto convolve_pion_decay_spectrum(double eg,
                                         const EnergyHist<LinAxis> &edists)
    -> double {
  double val = 0;
  for (auto &&h : boost::histogram::indexed(edists)) { // NOLINT
    const double p = *h;
    const double epi = h.bin().center();
    const double de = h.bin().width();
    val += p * decay_spectrum<ChargedPion>::dnde_photon(eg, epi) * de;
  }
  return val;
}

// Compute the decay spectrum from final state pions in N -> ν + π⁺ + π⁻.
// Computes:
//  ∫dEπ₁ (dN/dE)(Eγ, Eπ₁) P(Eπ₁) + ∫dEπ₂ (dN/dE)(Eγ, Eπ₂) P(Eπ₂)
static auto convolved_pion_decay_spectra(
    double eg, const std::array<EnergyHist<LinAxis>, 3> &edists) -> double {
  return convolve_pion_decay_spectrum(eg, edists[1]) +
         convolve_pion_decay_spectrum(eg, edists[2]);
}

// Return a function that compute the FSR spectrum from N -> ν + π⁺ + π⁻.
static auto make_fsr_function(const RhNeutrinoMeV &model,
                              const VPiPiConfig &config)
    -> std::function<double(double)> {

  const double mass = model.mass();
  const double theta = model.theta();
  const double non_rad = model.width_v_pi_pi().first;
  const std::array<double, 3> fsp_masses = {0.0, ChargedPion::mass,
                                            ChargedPion::mass};

  auto msqrd_rad = [mass, theta](const std::array<LVector<double>, 4> &ps) {
    return msqrd_v_pi_pi_a(ps, mass, theta);
  };

  const auto nevents = config.nevents_fsr;
  auto fsr = [mass, msqrd_rad, fsp_masses, non_rad, nevents](double eg) {
    return photon_spectrum_rambo(msqrd_rad, eg, mass, fsp_masses, non_rad,
                                 nevents)
        .first;
  };

  return fsr;
}

// Return a function that computes the FSR and decay spectrum from N -> ν + π⁺ +
// π⁻. Note this computes dndx and not dnde. Factors are computed to do the
// conversion.
static auto make_total_dndx_spectrum_function(const RhNeutrinoMeV &model,
                                              const VPiPiConfig &config)
    -> std::function<double(double)> {
  const double to_x = model.mass() / 2;

  auto edists = energy_distributions_v_pi_pi(model, config);
  auto fsr = make_fsr_function(model, config);

  return [to_x, edists, fsr](double x) {
    const double eg = x * to_x;
    return to_x * (convolved_pion_decay_spectra(eg, edists) + fsr(eg));
  };
}

// Return a function that computes the boosted FSR and decay spectrum from N ->
// ν + π⁺ + π⁻. Note this computes dndx and not dnde. Factors are computed to do
// the conversion.
static auto
make_total_boosted_dndx_spectrum_function(const RhNeutrinoMeV &model, double e,
                                          const VPiPiConfig &config)
    -> std::function<double(double)> {
  const double beta = tools::beta(e, model.mass());
  auto f_dndx = make_total_dndx_spectrum_function(model, config);

  return [f_dndx, beta](double x) { return dndx_boost(f_dndx, x, beta); };
}

// ========================================================================
// ---- Functions to compute the total spectrum from N -> ν + π⁺ + π⁻. ----
// ========================================================================

auto RhNeutrinoMeV::dndx_v_pi_pi(const py::array_t<double> &xs) const
    -> py::array_t<double> {
  static constexpr unsigned int nbins = 20;
  static constexpr size_t nevents = 1000;
  static constexpr size_t nevents_fsr = 5000;
  const VPiPiConfig config{nbins, nevents, nevents_fsr};

  if (p_mass < 2 * ChargedPion::mass) {
    return tools::zeros_like(xs);
  }

  auto f_dndx = make_total_dndx_spectrum_function(*this, config);

  py::buffer_info buf_xs = tools::get_buffer_and_check_dim(xs);
  auto dndx = py::array_t<double>(buf_xs.size);
  py::buffer_info buf_dndx = dndx.request();

  auto *ptr_xs = static_cast<double *>(buf_xs.ptr);
  auto *ptr_dndx = static_cast<double *>(buf_dndx.ptr);

  const double to_x = p_mass / 2;

  double x = 0;
  double val = 0;
  for (size_t i = 0; i < buf_xs.shape[0]; ++i) { // NOLINT
    x = ptr_xs[i];                               // NOLINT
    ptr_dndx[i] = f_dndx(x);                     // NOLINT
  }
  return dndx;
}

auto RhNeutrinoMeV::dndx_v_pi_pi(const py::array_t<double> &xs, double e) const
    -> py::array_t<double> {
  static constexpr unsigned int nbins = 20;
  static constexpr size_t nevents = 1000;
  static constexpr size_t nevents_fsr = 5000;
  const VPiPiConfig config{nbins, nevents, nevents_fsr};

  if (p_mass < 2 * ChargedPion::mass) {
    return tools::zeros_like(xs);
  }

  const double beta = p_mass / e;
  auto f_dndx = make_total_boosted_dndx_spectrum_function(*this, e, config);

  py::buffer_info buf_xs = tools::get_buffer_and_check_dim(xs);
  auto dndx = py::array_t<double>(buf_xs.size);
  py::buffer_info buf_dndx = dndx.request();

  auto *ptr_xs = static_cast<double *>(buf_xs.ptr);
  auto *ptr_dndx = static_cast<double *>(buf_dndx.ptr);

  double x = 0;
  for (size_t i = 0; i < buf_xs.shape[0]; ++i) { // NOLINT
    x = ptr_xs[i];                               // NOLINT
    ptr_dndx[i] = dndx_boost(f_dndx, x, beta);   // NOLINT
  }
  return dndx;
}

} // namespace blackthorn
