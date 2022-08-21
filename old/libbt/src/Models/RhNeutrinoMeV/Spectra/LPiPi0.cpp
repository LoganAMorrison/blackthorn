// This file contains the implementation of the gamma-ray spectrum from the
// decay of a RH neutrino through: N -> ℓ⁻ + π⁺ + π⁰. The FSR spectrum is
// implemented using the analytic matrix element for N -> ℓ⁻ + π⁺ + π⁰ + γ and
// RAMBO.

#include "blackthorn/Models/Particles.h"
#include "blackthorn/Models/RhNeutrino.h"
#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/Spectra.h"
#include "blackthorn/Spectra/Conv.h"
#include "blackthorn/Tools.h"

namespace blackthorn {

// Structure holding the configuration how total spectrum from N -> ℓ⁻ + π⁺ + π⁰
// should be computed.
struct LPiPi0Config { // NOLINT
  unsigned int nbins;
  size_t nevents_edist;
  size_t nevents_fsr;
};

static auto msqrd_l_pi_pi0_a(const std::array<LVector<double>, 4> &ps, double M,
                             double theta, double ml) -> double { // NOLINT
  const auto pl = ps[0];
  const auto pp = ps[1];
  const auto p0 = ps[2];
  const auto k = ps[3];
  const double M2 = M * M;
  const double x2 = dot(pp, pl) / M2;
  const double x3 = dot(p0, pl) / M2;
  const double x4 = dot(pp, k) / M2;
  const double x5 = dot(p0, k) / M2;
  const double x6 = dot(pl, k) / M2;
  const double bp2 = tools::sqr(ChargedPion::mass / M);
  const double b02 = tools::sqr(NeutralPion::mass / M);
  const double bl2 = tools::sqr(ml / M);
  constexpr double qe = StandardModel::qe;
  constexpr double cw = StandardModel::cw;
  constexpr double sw = StandardModel::sw;
  constexpr double gf = StandardModel::g_fermi;
  const double vud2 = std::norm(StandardModel::ckm<Gen::Fst, Gen::Fst>());

  return (-2 * std::pow(qe, 2) * std::pow(gf, 2) * std::pow(M, 2) * vud2 *
          (-2 * std::pow(bl2, 3) * std::pow(x4, 2) -
           std::pow(bl2, 2) *
               (6.0 * std::pow(x4, 3) + 2 * bp2 * std::pow(x6, 2) -
                x4 * x6 * (4 * x2 + 3 * x6) +
                std::pow(x4, 2) * (-2 + 4 * b02 + 4 * bp2 + 6.0 * x2 +
                                   6.0 * x3 + 6.0 * x5 + 6.0 * x6)) +
           bl2 * (2 * std::pow(x4, 3) - 8 * b02 * std::pow(x4, 3) +
                  2 * std::pow(x4, 2) * x5 - 8 * bp2 * std::pow(x4, 2) * x5 -
                  16 * std::pow(x4, 3) * x5 + 12.0 * std::pow(x2, 2) * x4 * x6 +
                  2 * std::pow(x4, 2) * x6 - 8 * b02 * std::pow(x4, 2) * x6 -
                  2 * std::pow(x4, 3) * x6 - 10.0 * std::pow(x4, 2) * x5 * x6 +
                  2 * bp2 * std::pow(x6, 2) - 4 * b02 * bp2 * std::pow(x6, 2) -
                  4 * std::pow(bp2, 2) * std::pow(x6, 2) -
                  4 * x4 * std::pow(x6, 2) + 3 * b02 * x4 * std::pow(x6, 2) +
                  bp2 * x4 * std::pow(x6, 2) +
                  12.0 * std::pow(x4, 2) * std::pow(x6, 2) +
                  6.0 * x4 * x5 * std::pow(x6, 2) -
                  6.0 * bp2 * std::pow(x6, 3) + 14.0 * x4 * std::pow(x6, 3) -
                  2 * x3 *
                      (8 * std::pow(x4, 3) + 3 * bp2 * std::pow(x6, 2) -
                       3 * x4 * std::pow(x6, 2) +
                       std::pow(x4, 2) * (-1 + 4 * bp2 + 5.0 * x6)) -
                  2 * x2 *
                      (std::pow(x4, 2) *
                           (-1 + 4 * b02 + 8 * x3 + 8 * x5 - 6.0 * x6) +
                       3 * bp2 * std::pow(x6, 2) -
                       x4 * x6 *
                           (-2 + 4 * b02 + 4 * bp2 + 6.0 * x3 + 3 * x5 +
                            13.0 * x6))) +
           x6 * (-2 * std::pow(x4, 3) + 8 * b02 * std::pow(x4, 3) -
                 2 * std::pow(x4, 2) * x5 + 8 * bp2 * std::pow(x4, 2) * x5 +
                 16 * std::pow(x4, 3) * x5 + x4 * x6 - b02 * x4 * x6 -
                 3 * bp2 * x4 * x6 - 4 * b02 * bp2 * x4 * x6 +
                 4 * std::pow(bp2, 2) * x4 * x6 - 12.0 * std::pow(x4, 2) * x6 +
                 16 * b02 * std::pow(x4, 2) * x6 +
                 16 * bp2 * std::pow(x4, 2) * x6 + 16 * std::pow(x4, 3) * x6 -
                 2 * x4 * x5 * x6 + 8 * bp2 * x4 * x5 * x6 +
                 32 * std::pow(x4, 2) * x5 * x6 + 2 * bp2 * std::pow(x6, 2) -
                 8 * b02 * bp2 * std::pow(x6, 2) - 10.0 * x4 * std::pow(x6, 2) +
                 8 * b02 * x4 * std::pow(x6, 2) +
                 16 * bp2 * x4 * std::pow(x6, 2) +
                 32 * std::pow(x4, 2) * std::pow(x6, 2) +
                 16 * x4 * x5 * std::pow(x6, 2) + 16 * x4 * std::pow(x6, 3) +
                 4 * std::pow(x2, 2) * x4 *
                     (-1 + 4 * b02 + 8 * x3 + 4 * x5 + 4 * x6) +
                 2 * x3 *
                     (8 * std::pow(x4, 3) + bp2 * (1 - 4 * bp2 - 8 * x6) * x6 +
                      x4 * x6 * (-1 - 4 * bp2 + 8 * x6) +
                      std::pow(x4, 2) * (-1 + 4 * bp2 + 16 * x6)) +
                 2 * x2 *
                     ((1 - 4 * b02) * bp2 * x6 +
                      2 * std::pow(x4, 2) * (-1 + 4 * b02 + 8 * x5 + 8 * x6) +
                      x4 * x5 * (-1 + 4 * bp2 + 16 * x6) +
                      x4 * x6 * (-7.0 + 12.0 * b02 + 8 * bp2 + 16 * x6) +
                      x3 * (24.0 * std::pow(x4, 2) - 8 * bp2 * x6 +
                            x4 * (-2 + 8 * bp2 + 24.0 * x6))))) *
          std::pow(std::sin(theta), 2)) /
         (std::pow(x4, 2) * std::pow(x6, 2));
}

// Get the masses of the final state particles from N -> ℓ⁻ + π⁺ + π⁰.
static auto final_state_particle_masses(const RhNeutrinoMeV &model)
    -> std::array<double, 3> {
  return {StandardModel::charged_lepton_mass(model.gen()), ChargedPion::mass,
          NeutralPion::mass};
}

// Construct the energy distributions of the final state particles from N -> ℓ⁻
// + π⁺ + π⁰.
static auto energy_distributions_l_pi_pi0(const RhNeutrinoMeV &model,
                                          const LPiPi0Config &config)
    -> std::array<EnergyHist<LinAxis>, 3> {

  const auto msqrd = SquaredAmplitudeNToLPiPi0(model);
  const std::array<double, 3> fsp_masses = final_state_particle_masses(model);

  return energy_distributions_linear(msqrd, model.mass(), fsp_masses,
                                     {config.nbins, config.nbins, config.nbins},
                                     config.nevents_edist);
}

// Given the energy distribution of a final state particle (muon or charged
// pion), convolve the distribution with the particle's decay spectrum.
// Computes:
//  ∫dε (dN/dE)(Eγ, ε) P(ε)
// where P(ε) is the probability of the particle having energy ε.
template <class Particle>
static auto convolve_decay_spectrum(double eg, const EnergyHist<LinAxis> &edist)
    -> double {
  double val = 0;
  for (const auto &&h : boost::histogram::indexed(edist)) { // NOLINT
    const double p = *h;
    const double ep = h.bin().center();
    const double dep = h.bin().width();
    val += p * decay_spectrum<Particle>::dnde_photon(eg, ep) * dep;
  }
  return val;
}

// Compute the decay spectrum from final state pions in N -> ℓ⁻ + π⁺ + π⁰.
// Computes:
//  ∫dEπ⁺ (dN/dE)(Eγ, Eπ⁺) P(Eπ⁺)
//    + ∫dEπ⁰ (dN/dE)(Eγ, Eπ⁰) P(Eπ⁰)
//    + ∫dEℓ (dN/dE)(Eγ, Eℓ) P(Eℓ)
static auto
convolved_decay_spectra(double eg,
                        const std::array<EnergyHist<LinAxis>, 3> &edists,
                        Gen gen) -> double {

  double val = 0.0;
  if (gen == Gen::Snd) {
    val += convolve_decay_spectrum<Muon>(eg, edists[0]);
  }
  val += convolve_decay_spectrum<ChargedPion>(eg, edists[1]);
  val += convolve_decay_spectrum<NeutralPion>(eg, edists[2]);

  return val;
}

// Return a function that compute the FSR spectrum from N -> ℓ⁻ + π⁺ + π⁰..
static auto make_fsr_function(const RhNeutrinoMeV &model,
                              const LPiPi0Config &config)
    -> std::function<double(double)> {

  const double mass = model.mass();
  const double ml = StandardModel::charged_lepton_mass(model.gen());
  const double theta = model.theta();
  const double non_rad = model.width_l_pi_pi0().first;
  const std::array<double, 3> fsp_masses = final_state_particle_masses(model);

  auto msqrd_rad = [mass, theta, ml](const std::array<LVector<double>, 4> &ps) {
    return msqrd_l_pi_pi0_a(ps, mass, theta, ml);
  };

  const auto nevents = config.nevents_fsr;
  auto fsr = [mass, msqrd_rad = std::move(msqrd_rad), fsp_masses, non_rad,
              nevents](double eg) {
    return photon_spectrum_rambo(msqrd_rad, eg, mass, fsp_masses, non_rad,
                                 nevents)
        .first;
  };

  return fsr;
}

// Return a function that computes the FSR and decay spectrum from N -> ℓ⁻ + π⁺
// + π⁰. Note this computes dndx and not dnde. Factors are computed to do the
// conversion.
static auto make_total_dndx_spectrum_function(const RhNeutrinoMeV &model,
                                              const LPiPi0Config &config)
    -> std::function<double(double)> {
  const double to_x = model.mass() / 2;
  const Gen gen = model.gen();

  auto edists = energy_distributions_l_pi_pi0(model, config);
  auto fsr = make_fsr_function(model, config);

  return
      [edists = std::move(edists), to_x, gen, fsr = std::move(fsr)](double x) {
        const double eg = x * to_x;
        return to_x * (convolved_decay_spectra(eg, edists, gen) + fsr(eg));
      };
}

// Return a function that computes the boosted FSR and decay spectrum from N ->
// ℓ⁻ + π⁺ + π⁰. Note this computes dndx and not dnde. Factors are computed to
// do the conversion.
static auto
make_total_boosted_dndx_spectrum_function(const RhNeutrinoMeV &model, double e,
                                          const LPiPi0Config &config)
    -> std::function<double(double)> {
  const double beta = model.mass() / e;
  auto f_dndx = make_total_dndx_spectrum_function(model, config);

  return [f_dndx, beta](double x) { return dndx_boost(f_dndx, x, beta); };
}

// =========================================================================
// ---- Functions to compute the total spectrum from N -> ℓ⁻ + π⁺ + π⁰. ----
// =========================================================================

auto RhNeutrinoMeV::dndx_l_pi_pi0(const std::vector<double> &xs, double e) const
    -> std::vector<double> {
  using tools::vectorized_par;
  static constexpr unsigned int nbins = 25;
  static constexpr size_t nevents = 1000;
  static constexpr size_t nevents_rad = 1000;

  const LPiPi0Config config = {nbins, nevents, nevents_rad};

  constexpr double mpi = ChargedPion::mass;
  constexpr double mpi0 = NeutralPion::mass;
  const double ml = StandardModel::charged_lepton_mass(p_gen);

  std::vector<double> dndx(xs.size(), 0);

  if (p_mass < ml + mpi + mpi0) {
    return dndx;
  }

  auto f_dndx = make_total_boosted_dndx_spectrum_function(*this, e, config);
  return vectorized_par(f_dndx, xs);
}

auto RhNeutrinoMeV::dndx_l_pi_pi0(const py::array_t<double> &xs, double e) const
    -> py::array_t<double> {
  static constexpr unsigned int nbins = 25;
  static constexpr size_t nevents = 1000;
  static constexpr size_t nevents_rad = 1000;

  const LPiPi0Config config = {nbins, nevents, nevents_rad};

  constexpr double mpi = ChargedPion::mass;
  constexpr double mpi0 = NeutralPion::mass;
  const double ml = StandardModel::charged_lepton_mass(p_gen);

  if (p_mass < ml + mpi + mpi0) {
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

auto RhNeutrinoMeV::dndx_l_pi_pi0(const py::array_t<double> &xs) const
    -> py::array_t<double> {
  static constexpr unsigned int nbins = 25;
  static constexpr size_t nevents = 1000;
  static constexpr size_t nevents_rad = 1000;

  const LPiPi0Config config = {nbins, nevents, nevents_rad};

  constexpr double mpi = ChargedPion::mass;
  constexpr double mpi0 = NeutralPion::mass;
  const double ml = StandardModel::charged_lepton_mass(p_gen);

  if (p_mass < ml + mpi + mpi0) {
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

} // namespace blackthorn
