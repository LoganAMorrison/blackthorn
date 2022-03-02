// This file compute the gamma-ray spectrum from the decay of a RH neutrino into
// an active neutrino and two changed leptons. The matrix element for N -> ν +
// ℓ⁺ + ℓ⁻ + γ is sufficiently complicated that we instead use the
// Altarelli-Parisi approximation for FSR off the leptons.

#include "blackthorn/Models/RhNeutrino.h"
#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/Spectra.h"
#include "blackthorn/Tools.h"

namespace blackthorn {

struct ModelConfigVLL { // NOLINT
  Gen gv;
  Gen g1;
  Gen g2;
  unsigned int nbins;
  size_t nevents;
};

static auto dndx_fsr_ap(double e, double ecm, double mass) -> double {
  const double x = 2 * e / ecm;
  const double mu = mass / ecm;
  return StandardModel::alpha_em / (2.0 * M_PI) * (1.0 + tools::sqr(1.0 - x)) /
         x * (log((1 - x) / tools::sqr(mu)) - 1);
}

// Compute the masses of the final state particle mass from N -> ν + ℓ⁺ + ℓ⁻.
static auto final_state_particle_masses(Gen g1, Gen g2)
    -> std::array<double, 3> {
  const double ml1 = StandardModel::charged_lepton_mass(g1);
  const double ml2 = StandardModel::charged_lepton_mass(g2);
  return {0.0, ml1, ml2};
}

// Compute the sum of the final state particle mass from N -> ν + ℓ⁺ + ℓ⁻.
static auto sum_final_state_particle_masses(Gen g1, Gen g2) -> double {
  const double ml1 = StandardModel::charged_lepton_mass(g1);
  const double ml2 = StandardModel::charged_lepton_mass(g2);
  return ml1 + ml2;
}

// Compute the energy distributions of the final state charged leptons in the
// decay N -> ν + ℓ⁺ + ℓ⁻.
static auto energy_distributions_v_l_l(const RhNeutrinoMeV &model,
                                       const ModelConfigVLL &config)
    -> std::array<EnergyHist<LinAxis>, 3> {
  const auto msqrd =
      SquaredAmplitudeNToVLL(model, config.gv, config.g1, config.g2);
  const auto fsp_masses = final_state_particle_masses(config.g1, config.g2);
  const auto nbins = config.nbins;
  const auto nevents = config.nevents;
  return energy_distributions_linear(msqrd, model.mass(), fsp_masses,
                                     {nbins, nbins, nbins}, nevents);
}

// Compute the invariant mass distribution of the final state leptons in the
// decay N -> ν + ℓ⁺ + ℓ⁻.
static auto ll_invariant_mass_distribution(const RhNeutrinoMeV &model,
                                           const ModelConfigVLL &config)
    -> EnergyHist<LinAxis> {
  const auto msqrd =
      SquaredAmplitudeNToVLL(model, config.gv, config.g1, config.g2);
  const auto fsp_masses = final_state_particle_masses(config.g1, config.g2);
  const auto nbins = config.nbins;
  const auto nevents = config.nevents;
  return invariant_mass_distributions_linear<1, 2>(msqrd, model.mass(),
                                                   fsp_masses, nbins, nevents);
}

// Convolve the muon decay spectrum with its energy distribution. That is,
// computes
//     ∫dEμ (dN/dE)(Eγ, Eμ) P(Eμ)
// with P(Eμ) being the probability of a muon having energy Eμ.
static auto convolve_muon_spectrum(double eg, const EnergyHist<LinAxis> &edist)
    -> double {
  double val = 0.0;
  for (const auto &&h : boost::histogram::indexed(edist)) { // NOLINT
    const double p = *h;
    const double em = h.bin().center();
    const double dem = h.bin().width();
    val += p * decay_spectrum<Muon>::dnde_photon(eg, em) * dem;
  }
  return val;
}

// Construct function computing dN/dx from a lepton. Only the muon is taken to
// decay (tau is to heavy for this model).
static auto make_lepton_decay_function(Gen g, const EnergyHist<LinAxis> &edist)
    -> std::function<double(double)> {
  if (g == Gen::Snd) {
    return [edist](double eg) { return convolve_muon_spectrum(eg, edist); };
  }
  return [](double /*eg*/) { return 0.0; };
}

// Construct function computing dN/dx from decays of final state leptons from N
// -> ν + ℓ⁺ + ℓ⁻.
static auto make_dndx_decay_function(const RhNeutrinoMeV &model,
                                     const ModelConfigVLL &config)
    -> std::function<double(double)> {
  const auto edists = energy_distributions_v_l_l(model, config);
  std::function<double(double)> f_dnde1 =
      make_lepton_decay_function(config.g1, edists[1]);
  std::function<double(double)> f_dnde2 =
      make_lepton_decay_function(config.g2, edists[2]);
  const double to_x = model.mass() / 2.0;
  return [to_x, f_dnde1 = std::move(f_dnde1), f_dnde2 = std::move(f_dnde2)](
             double eg) { return to_x * (f_dnde1(eg) + f_dnde2(eg)); };
}

// Construct the FSR dN/dx from N -> ν + ℓ⁺ + ℓ⁻. To compute FSR, we use the
// Altarelli Parisi approximation for radiation off charged leptons. We convolve
// the AP function with the invariant mass distribution of the final state
// leptons. The invariant mass acts as the 'center-of-mass energy' of the ℓ⁺ +
// ℓ⁻ pair.
static auto make_dndx_fsr_function(const RhNeutrinoMeV &model,
                                   const ModelConfigVLL &config)
    -> std::function<double(double)> {
  const double m = model.mass();
  const double ml1 = StandardModel::charged_lepton_mass(config.g1);
  const double ml2 = StandardModel::charged_lepton_mass(config.g2);
  const double bl1 = ml1 / m;
  const double bl2 = ml2 / m;
  auto invmass_dist = ll_invariant_mass_distribution(model, config);

  return [dist = std::move(invmass_dist), m, bl1, bl2](double x) {
    double val = 0.0;
    for (const auto &&h : boost::histogram::indexed(dist)) { // NOLINT
      const double pz = *h * m;
      const double z = h.bin().center() / m;
      const double dz = h.bin().width() / m;
      val += pz * dndx_altarelli_parisi_f_to_a(x, bl1, z) * dz;
      val += pz * dndx_altarelli_parisi_f_to_a(x, bl2, z) * dz;
    }
    return val;
  };
}

// Construct the total dN/dx for (include decay and FSR) from N -> ν + ℓ⁺ + ℓ⁻
// in the RH neutrino rest frame.
static auto make_dndx_total_function(const RhNeutrinoMeV &model,
                                     const ModelConfigVLL &config)
    -> std::function<double(double)> {
  auto fsr = make_dndx_fsr_function(model, config);
  auto decay = make_dndx_decay_function(model, config);
  return [fsr = std::move(fsr), decay = std::move(decay)](double x) {
    return fsr(x) + decay(x);
  };
}

// Construct the total boosted dN/dx for (include decay and FSR) from N -> ν +
// ℓ⁺ + ℓ⁻.
static auto make_dndx_total_boosted_function(const RhNeutrinoMeV &model,
                                             const ModelConfigVLL &config,
                                             double e)
    -> std::function<double(double)> {
  const double beta = tools::beta(e, model.mass());
  auto dndx = make_dndx_total_function(model, config);
  return [beta, dndx = std::move(dndx)](double x) {
    return dndx_boost(dndx, x, beta);
  };
}

// Apply the function dndx over the array xs.
static auto
apply_dndx_over_numpy_array(const std::function<double(double)> &dndx,
                            const py::array_t<double> &xs)
    -> py::array_t<double> {
  py::buffer_info buf_xs = tools::get_buffer_and_check_dim(xs);
  auto result = py::array_t<double>(buf_xs.size);
  py::buffer_info buf_dndx = result.request();

  auto *ptr_xs = static_cast<double *>(buf_xs.ptr);
  auto *ptr_dndx = static_cast<double *>(buf_dndx.ptr);

  double x = 0;
  double val = 0;
  for (size_t i = 0; i < buf_xs.shape[0]; ++i) { // NOLINT
    ptr_dndx[i] = dndx(ptr_xs[i]);               // NOLINT
  }

  return result;
}

// ========================================================================
// ---- Functions to compute the total spectrum from N -> ν + ℓ⁺ + ℓ⁻. ----
// ========================================================================

auto RhNeutrinoMeV::dndx_v_l_l(const py::array_t<double> &xs, Gen gv, Gen g1,
                               Gen g2) const -> py::array_t<double> {
  static constexpr unsigned int nbins = 20;
  static constexpr size_t nevents = 1000;

  if (p_mass < sum_final_state_particle_masses(g1, g2)) {
    return tools::zeros_like(xs);
  }

  const ModelConfigVLL config{gv, g1, g2, nbins, nevents};
  const auto f_dndx = make_dndx_total_function(*this, config);
  return apply_dndx_over_numpy_array(f_dndx, xs);
}

auto RhNeutrinoMeV::dndx_v_l_l(const py::array_t<double> &xs, Gen gv, Gen g1,
                               Gen g2, double e) const -> py::array_t<double> {
  static constexpr unsigned int nbins = 20;
  static constexpr size_t nevents = 1000;

  if (p_mass < sum_final_state_particle_masses(g1, g2)) {
    return tools::zeros_like(xs);
  }

  const ModelConfigVLL config{gv, g1, g2, nbins, nevents};
  const auto f_dndx = make_dndx_total_boosted_function(*this, config, e);
  return apply_dndx_over_numpy_array(f_dndx, xs);
}

auto RhNeutrinoMeV::dndx_v_l_l(const std::vector<double> &xs, Gen gv, Gen g1,
                               Gen g2, double e) const -> std::vector<double> {
  static constexpr unsigned int nbins = 20;
  static constexpr size_t nevents = 1000;

  std::vector<double> result(xs.size(), 0.0);

  if (p_mass < sum_final_state_particle_masses(g1, g2)) {
    return result;
  }

  const ModelConfigVLL config{gv, g1, g2, nbins, nevents};
  const auto f_dndx = make_dndx_total_boosted_function(*this, config, e);

  for (size_t i = 0; i < xs.size(); ++i) {
    result[i] = f_dndx(xs[i]);
  }

  return result;
}

} // namespace blackthorn
