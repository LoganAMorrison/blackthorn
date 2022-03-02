#include "blackthorn/Models/RhNeutrino.h"

namespace blackthorn {

auto RhNeutrinoGeV::width_v_h() const -> double {
  if (p_mass < Higgs::mass) {
    return 0.0;
  }
  return (std::pow(std::pow(Higgs::mass, 2) - std::pow(mass(), 2), 2) *
          std::pow(std::tan(theta()), 2)) /
         (16. * M_PI * std::pow(Higgs::vev, 2) * mass());
}

auto RhNeutrinoGeV::width_v_z() const -> double {
  if (p_mass < ZBoson::mass) {
    return 0.0;
  }
  return (std::pow(StandardModel::qe, 2) *
          std::pow(std::pow(ZBoson::mass, 2) - std::pow(mass(), 2), 2) *
          (2 * std::pow(ZBoson::mass, 2) + std::pow(mass(), 2)) *
          std::pow(std::sin(2 * theta()), 2)) /
         (256. * std::pow(StandardModel::cw, 2) * std::pow(ZBoson::mass, 2) *
          M_PI * std::pow(StandardModel::sw, 2) * std::pow(mass(), 3));
}

auto RhNeutrinoGeV::width_l_w() const -> double {
  const double ml = StandardModel::charged_lepton_mass(p_gen);
  if (p_mass < WBoson::mass + ml) {
    return 0.0;
  }
  return (std::pow(StandardModel::qe, 2) *
          std::sqrt(tools::kallen_lambda(std::pow(WBoson::mass, 2),
                                         std::pow(ml, 2),
                                         std::pow(mass(), 2))) *
          (-2 * std::pow(WBoson::mass, 4) + std::pow(ml, 4) +
           std::pow(WBoson::mass, 2) * std::pow(mass(), 2) +
           std::pow(mass(), 4) +
           std::pow(ml, 2) *
               (std::pow(WBoson::mass, 2) - 2 * std::pow(mass(), 2))) *
          std::pow(std::sin(theta()), 2)) /
         (64. * std::pow(WBoson::mass, 2) * M_PI *
          std::pow(StandardModel::sw, 2) * std::pow(mass(), 3));
}

auto RhNeutrinoGeV::width_v_u_u(Gen genu, size_t nevents,
                                size_t batchsize) const
    -> std::pair<double, double> {
  const double mu = StandardModel::up_type_quark_mass(genu);
  if (p_mass < 2 * mu) {
    return std::make_pair(0.0, 0.0);
  }
  const auto msqrd = SquaredAmplitudeNToVUU(*this, genu);
  const std::array<double, 3> fsp_masses = {0.0, mu, mu};
  return Rambo<3>::decay_width(msqrd, p_mass, fsp_masses, nevents, batchsize);
}

auto RhNeutrinoGeV::width_v_d_d(Gen gend, size_t nevents,
                                size_t batchsize) const
    -> std::pair<double, double> {
  const double md = StandardModel::down_type_quark_mass(gend);
  if (p_mass < 2 * md) {
    return std::make_pair(0.0, 0.0);
  }
  const auto msqrd = SquaredAmplitudeNToVDD(*this, gend);
  const std::array<double, 3> fsp_masses = {0.0, md, md};
  return Rambo<3>::decay_width(msqrd, p_mass, fsp_masses, nevents, batchsize);
}

auto RhNeutrinoGeV::width_l_u_d(Gen genu, Gen gend, size_t nevents,
                                size_t batchsize) const
    -> std::pair<double, double> {
  const double ml = StandardModel::charged_lepton_mass(p_gen);
  const double mu = StandardModel::up_type_quark_mass(genu);
  const double md = StandardModel::down_type_quark_mass(gend);
  if (p_mass < ml + mu + md) {
    return std::make_pair(0.0, 0.0);
  }
  const auto msqrd = SquaredAmplitudeNToLUD(*this, genu, gend);
  const std::array<double, 3> fsp_masses = {ml, mu, md};
  return Rambo<3>::decay_width(msqrd, p_mass, fsp_masses, nevents, batchsize);
}

} // namespace blackthorn
