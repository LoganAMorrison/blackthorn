#include "blackthorn/Models/RhNeutrino.h"
#include "blackthorn/Spectra/Conv.h"
#include "blackthorn/Spectra/Pythia.h"
#include "blackthorn/Spectra/Splitting.h"

namespace blackthorn {

// ===========================================================================
// ---- N -> nu + H ----------------------------------------------------------
// ===========================================================================

auto RhNeutrinoGeV::dndx_v_h(double xmin, double xmax, unsigned int nbins,
                             unsigned int nevents) const
    -> PythiaSpectrum<2>::SpectrumType {
  PythiaSpectrum<2> ps{xmin, xmax, nbins, nevents};
  std::array<int, 2> pdgs = {StandardModel::neutrino_pdg(p_gen), Higgs::pdg};
  const auto msqrd = SquaredAmplitudeNToVH(*this);
  return ps.decay_spectrum(p_mass, Photon::pdg, pdgs, msqrd);
}

// ===========================================================================
// ---- N -> nu + Z ----------------------------------------------------------
// ===========================================================================

auto RhNeutrinoGeV::dndx_v_z(double xmin, double xmax, unsigned int nbins,
                             unsigned int nevents) const
    -> PythiaSpectrum<2>::SpectrumType {
  PythiaSpectrum<2> ps{xmin, xmax, nbins, nevents};
  std::array<int, 2> pdgs = {StandardModel::neutrino_pdg(p_gen), ZBoson::pdg};
  const auto msqrd = SquaredAmplitudeNToVZ(*this);
  return ps.decay_spectrum(p_mass, Photon::pdg, pdgs, msqrd);
}

// ===========================================================================
// ---- N -> l + W -----------------------------------------------------------
// ===========================================================================

static auto dndx_l_w_decay(const RhNeutrinoGeV &model, double xmin, double xmax,
                           unsigned int nbins, unsigned int nevents)
    -> PythiaSpectrum<2>::SpectrumType {
  PythiaSpectrum<2> ps{xmin, xmax, nbins, nevents};
  std::array<int, 2> pdgs = {StandardModel::charged_lepton_pdg(model.gen()),
                             WBoson::pdg};
  const auto msqrd = SquaredAmplitudeNToLW(model);
  return ps.decay_spectrum(model.mass(), Photon::pdg, pdgs, msqrd);
}

auto RhNeutrinoGeV::dndx_l_w_fsr(double x) const -> double {
  constexpr double mw = WBoson::mass;
  const double ml = StandardModel::charged_lepton_mass(p_gen);
  const double m = p_mass;

  const double bw = mw / m;
  const double bl = ml / m;
  const double bw2 = bw * bw;
  const double bl2 = bl * bl;
  const double bw4 = bw2 * bw2;
  const double bl4 = bl2 * bl2;

  const double sqrtfac =
      std::sqrt(std::pow(1 - x - bl2, 2) - 2 * (1 - x + bl2) * bw2 + bw4);

  const double bpoly1 = 1 + bl4 + bw2 - 2 * bw4 + bl2 * (-2 + bw2);

  const double pre =
      StandardModel::alpha_em /
      (2.0 * M_PI * x *
       std::sqrt(bl4 + std::pow(1 - bw2, 2) - 2 * bl2 * (1 + bw2)));

  const double rat = 4 * sqrtfac * (x * x - bpoly1) / bpoly1;

  const double wlog_pre = 2 * (1 - x - bl2 - bw2);
  const double llog_pre =
      2 * (1 - bl2 - bw2) + x * (2 - (x * (1 + bl2 + 2 * bw2)) / bpoly1);

  const double llog =
      log((1 - x + bl2 - bw2 + sqrtfac) / (1 - x + bl2 - bw2 - sqrtfac));
  const double wlog =
      log((1 - x - bl2 + bw2 - sqrtfac) / (1 - x - bl2 + bw2 + sqrtfac));

  return pre * (rat + llog_pre * llog + wlog_pre * wlog);
}

auto RhNeutrinoGeV::dndx_l_w(double xmin, double xmax, unsigned int nbins,
                             unsigned int nevents) const
    -> PythiaSpectrum<2>::SpectrumType {
  const double ml = StandardModel::charged_lepton_mass(p_gen);
  constexpr double mw = WBoson::mass;
  auto [xs, dndxs] = dndx_l_w_decay(*this, xmin, xmax, nbins, nevents);

  if (p_mass > ml + mw) {
    for (size_t i = 0; i < xs.size(); i++) { // NOLINT
      dndxs[i] += dndx_l_w_fsr(xs[i]);
    }
  }

  return std::make_pair(xs, dndxs);
}

// ===========================================================================
// ---- N -> nu + u + u ------------------------------------------------------
// ===========================================================================

static auto dndx_v_u_u_decay(const RhNeutrinoGeV &model, Gen genu, double xmin,
                             double xmax, unsigned int nbins,
                             unsigned int nevents)
    -> PythiaSpectrum<3>::SpectrumType {
  const int pdgv = StandardModel::neutrino_pdg(model.gen());
  const int pdgu = StandardModel::up_type_quark_pdg(genu);
  PythiaSpectrum<3> ps{xmin, xmax, nbins, nevents};
  std::array<int, 3> pdgs = {pdgv, pdgu, -pdgu};
  const auto msqrd = SquaredAmplitudeNToVUU(model, genu);
  return ps.decay_spectrum(model.mass(), Photon::pdg, pdgs, msqrd);
}

auto RhNeutrinoGeV::dndx_v_u_u(double xmin, double xmax, unsigned int nbins,
                               Gen genu, unsigned int nevents) const
    -> PythiaSpectrum<3>::SpectrumType {
  static constexpr unsigned int nbins_ = 20;
  static constexpr size_t nevents_ = 1000;

  auto [xs, dndxs] = dndx_v_u_u_decay(*this, genu, xmin, xmax, nbins, nevents);

  const auto mu = StandardModel::up_type_quark_mass(genu);
  const auto bu = mu / p_mass;
  const std::array<double, 3> fsp_masses = {0.0, mu, mu};

  const auto msqrd = SquaredAmplitudeNToVUU(*this, genu);
  auto invmass_dist = invariant_mass_distributions_linear<1, 2>(
      msqrd, p_mass, fsp_masses, nbins_, nevents_);

  double x = 0;
  double val = 0;
  for (size_t i = 0; i < xs.size(); ++i) { // NOLINT
    x = xs[i];                             // NOLINT
    val = 0;
    for (auto &&h : boost::histogram::indexed(invmass_dist)) { // NOLINT
      const double p = *h * p_mass;
      const double z = h.bin().center() / p_mass;
      const double dz = h.bin().width() / p_mass;
      val += 2 * p * dndx_altarelli_parisi_f_to_a(x, bu, z) * dz;
    }

    dndxs[i] += val; // NOLINT
  }

  return std::make_pair(xs, dndxs);
}

// ===========================================================================
// ---- N -> nu + d + d ------------------------------------------------------
// ===========================================================================

static auto dndx_v_d_d_decay(const RhNeutrinoGeV &model, Gen gend, double xmin,
                             double xmax, unsigned int nbins,
                             unsigned int nevents)
    -> PythiaSpectrum<3>::SpectrumType {
  const int pdgv = StandardModel::neutrino_pdg(model.gen());
  const int pdgd = StandardModel::down_type_quark_pdg(gend);
  PythiaSpectrum<3> ps{xmin, xmax, nbins, nevents};
  std::array<int, 3> pdgs = {pdgv, pdgd, -pdgd};
  const auto msqrd = SquaredAmplitudeNToVDD(model, gend);
  return ps.decay_spectrum(model.mass(), Photon::pdg, pdgs, msqrd);
}

static auto dndx_v_d_d_add_fsr(PythiaSpectrum<3>::SpectrumType *spec,
                               const RhNeutrinoGeV &model, Gen gend) -> void {
  static constexpr unsigned int nbins_ = 20;
  static constexpr size_t nevents_ = 1000;

  const auto msqrd = SquaredAmplitudeNToVDD(model, gend);
  const auto md = StandardModel::down_type_quark_mass(gend);
  const auto bd = md / model.mass();
  const std::array<double, 3> fsp_masses = {0.0, md, md};

  auto invmass_dist = invariant_mass_distributions_linear<1, 2>(
      msqrd, model.mass(), fsp_masses, nbins_, nevents_);

  double x = 0;
  double val = 0;
  for (size_t i = 0; i < spec->first.size(); ++i) { // NOLINT
    x = spec->first[i];                             // NOLINT
    val = 0;
    for (auto &&h : boost::histogram::indexed(invmass_dist)) { // NOLINT
      const double p = *h * model.mass();
      const double z = h.bin().center() / model.mass();
      const double dz = h.bin().width() / model.mass();
      val += 2 * p * dndx_altarelli_parisi_f_to_a(x, bd, z) * dz;
    }

    spec->second[i] += val; // NOLINT
  }
}

auto RhNeutrinoGeV::dndx_v_d_d(double xmin, double xmax, unsigned int nbins,
                               Gen gend, unsigned int nevents) const
    -> PythiaSpectrum<3>::SpectrumType {
  auto spec = dndx_v_d_d_decay(*this, gend, xmin, xmax, nbins, nevents);
  dndx_v_d_d_add_fsr(&spec, *this, gend);
  return spec;
}

// ===========================================================================
// ---- N -> l + u + d -------------------------------------------------------
// ===========================================================================

static auto dndx_l_u_d_decay(const RhNeutrinoGeV &model, Gen genu, Gen gend,
                             double xmin, double xmax, unsigned int nbins,
                             unsigned int nevents)
    -> PythiaSpectrum<3>::SpectrumType {
  PythiaSpectrum<3> ps{xmin, xmax, nbins, nevents};
  std::array<int, 3> pdgs = {StandardModel::charged_lepton_pdg(model.gen()),
                             StandardModel::up_type_quark_pdg(genu),
                             -StandardModel::down_type_quark_pdg(gend)};
  const auto msqrd = SquaredAmplitudeNToLUD(model, genu, gend);
  return ps.decay_spectrum(model.mass(), Photon::pdg, pdgs, msqrd);
}

static auto dndx_l_u_d_add_fsr(PythiaSpectrum<3>::SpectrumType *spec,
                               const RhNeutrinoGeV &model, Gen genu, Gen gend)
    -> void {
  static constexpr unsigned int nbins_ = 20;
  static constexpr unsigned int nevents_ = 1000;
  const auto msqrd = SquaredAmplitudeNToLUD(model, genu, gend);
  const double mn = model.mass();
  const double ml = StandardModel::charged_lepton_mass(model.gen());
  const double mu = StandardModel::up_type_quark_mass(genu);
  const double md = StandardModel::down_type_quark_mass(gend);
  const double bl = ml / mn;
  const double bu = mu / mn;
  const double bd = md / mn;
  constexpr double qu2 = tools::sqr(2.0 / 3.0);
  constexpr double qd2 = tools::sqr(1.0 / 3.0);
  std::array<double, 3> fsp_masses = {ml, mu, md};

  const auto edists = energy_distributions_linear(
      msqrd, model.mass(), fsp_masses, {nbins_, nbins_, nbins_}, nevents_);

  for (size_t i = 0; i < spec->first.size(); ++i) {
    const double x = spec->first[i];
    double fsr = 0.0;
    for (auto &&h : bh::indexed(edists[0])) { // NOLINT
      const double p = *h * mn;
      const double z = h.bin().center() / mn;
      const double dz = h.bin().width() / mn;
      fsr += p * dndx_altarelli_parisi_f_to_a(x, bl, z) * dz;
    }
    for (auto &&h : bh::indexed(edists[1])) { // NOLINT
      const double p = *h * qu2 * mn;
      const double z = h.bin().center() / mn;
      const double dz = h.bin().width() / mn;
      fsr += p * dndx_altarelli_parisi_f_to_a(x, bu, z) * dz;
    }
    for (auto &&h : bh::indexed(edists[2])) { // NOLINT
      const double p = *h * qd2 * mn;
      const double z = h.bin().center() / mn;
      const double dz = h.bin().width() / mn;
      fsr += p * dndx_altarelli_parisi_f_to_a(x, bd, z) * dz;
    }
    spec->second[i] += fsr;
  }
}

auto RhNeutrinoGeV::dndx_l_u_d(double xmin, double xmax, unsigned int nbins,
                               Gen genu, Gen gend, unsigned int nevents) const
    -> PythiaSpectrum<3>::SpectrumType {
  return dndx_l_u_d_decay(*this, genu, gend, xmin, xmax, nbins, nevents);
}

// ===========================================================================
// ---- N -> v + l + l -------------------------------------------------------
// ===========================================================================

auto RhNeutrinoGeV::dndx_v_l_l(double xmin, double xmax, unsigned int nbins,
                               Gen genv, Gen genl1, Gen genl2,
                               unsigned int nevents) const
    -> PythiaSpectrum<3>::SpectrumType {
  static constexpr unsigned int nbins_ = 20;
  static constexpr size_t nevents_ = 1000;

  PythiaSpectrum<3> ps{xmin, xmax, nbins, nevents};
  std::array<int, 3> pdgs = {StandardModel::neutrino_pdg(genv),
                             StandardModel::charged_lepton_pdg(genl1),
                             -StandardModel::charged_lepton_pdg(genl2)};
  const auto msqrd = SquaredAmplitudeNToVLL(*this, genv, genl1, genl2);
  auto [xs, dndxs] = ps.decay_spectrum(p_mass, Photon::pdg, pdgs, msqrd);

  const auto ml1 = StandardModel::charged_lepton_mass(genl1);
  const auto ml2 = StandardModel::charged_lepton_mass(genl2);
  const auto bl1 = ml1 / p_mass;
  const auto bl2 = ml2 / p_mass;
  const std::array<double, 3> fsp_masses = {0.0, ml1, ml2};

  auto invmass_dist = invariant_mass_distributions_linear<1, 2>(
      msqrd, p_mass, fsp_masses, nbins_, nevents_);

  double x = 0;
  double val = 0;
  for (size_t i = 0; i < xs.size(); ++i) { // NOLINT
    x = xs[i];                             // NOLINT
    val = 0;
    for (auto &&h : boost::histogram::indexed(invmass_dist)) { // NOLINT
      const double p = *h * p_mass;
      const double z = h.bin().center() / p_mass;
      const double dz = h.bin().width() / p_mass;
      val += p * dndx_altarelli_parisi_f_to_a(x, bl1, z) * dz;
      val += p * dndx_altarelli_parisi_f_to_a(x, bl2, z) * dz;
    }

    dndxs[i] = val; // NOLINT
  }

  return std::make_pair(xs, dndxs);
}

} // namespace blackthorn
