#include "../Utils.h"
#include "blackthorn/Models/RhNeutrino.h"

namespace blackthorn {

// ===========================================================================
// ---- N -> nu + H ----------------------------------------------------------
// ===========================================================================

SquaredAmplitudeNToVH::SquaredAmplitudeNToVH(const RhNeutrinoGeV &model)
    : SquaredAmplitudeNToX<2>(model.mass(), model.theta(), model.gen()) {}

auto SquaredAmplitudeNToVH::operator()(
    const std::array<LVector<double>, 2> & /*ps*/) const -> double {
  constexpr auto mh = Higgs::mass;
  constexpr auto vh = Higgs::vev;
  return (std::pow(p_mass, 2) * (-std::pow(mh, 2) + std::pow(p_mass, 2)) *
          std::pow(std::tan(p_theta), 2)) /
         std::pow(vh, 2);
}

// ===========================================================================
// ---- N -> nu + Z ----------------------------------------------------------
// ===========================================================================

SquaredAmplitudeNToVZ::SquaredAmplitudeNToVZ(const RhNeutrinoGeV &model)
    : SquaredAmplitudeNToX<2>(model.mass(), model.theta(), model.gen()) {}

auto SquaredAmplitudeNToVZ::operator()(
    const std::array<LVector<double>, 2> & /*ps*/) const -> double {
  constexpr auto qe = StandardModel::qe;
  constexpr auto cw = StandardModel::cw;
  constexpr auto sw = StandardModel::sw;
  constexpr auto mz = ZBoson::mass;
  return -0.0625 *
         (std::pow(qe, 2) *
          (2 * std::pow(mz, 4) - std::pow(mz, 2) * std::pow(p_mass, 2) -
           std::pow(p_mass, 4)) *
          std::pow(std::sin(2 * p_theta), 2)) /
         (std::pow(cw, 2) * std::pow(mz, 2) * std::pow(sw, 2));
}

// ===========================================================================
// ---- N -> l + W ----------------------------------------------------------
// ===========================================================================

SquaredAmplitudeNToLW::SquaredAmplitudeNToLW(const RhNeutrinoGeV &model)
    : SquaredAmplitudeNToX<2>(model.mass(), model.theta(), model.gen()),
      p_ml(StandardModel::charged_lepton_mass(p_gen)) {}

auto SquaredAmplitudeNToLW::operator()(
    const std::array<LVector<double>, 2> & /*ps*/) const -> double {
  constexpr auto qe = StandardModel::qe;
  constexpr auto cw = StandardModel::cw;
  constexpr auto sw = StandardModel::sw;
  constexpr auto mw = WBoson::mass;
  return (std::pow(qe, 2) *
          (-2 * std::pow(mw, 4) + std::pow(p_ml, 4) +
           std::pow(mw, 2) * std::pow(p_mass, 2) + std::pow(p_mass, 4) +
           std::pow(p_ml, 2) * (std::pow(mw, 2) - 2 * std::pow(p_mass, 2))) *
          std::pow(std::sin(p_theta), 2)) /
         (4. * std::pow(mw, 2) * std::pow(sw, 2));
}

// ===========================================================================
// ---- N -> nu + u + u ------------------------------------------------------
// ===========================================================================

SquaredAmplitudeNToVUU::SquaredAmplitudeNToVUU(const RhNeutrinoGeV &model,
                                               Gen genu)
    : SquaredAmplitudeNToX<3>(model.mass(), model.theta(), model.gen()),
      p_mu(StandardModel::up_type_quark_mass(genu)),
      p_vertex_nvh(model.feynman_rule_n_v_h()),
      p_vertex_nvz(model.feynman_rule_n_v_z()),
      p_vertex_uuh(model.mass() < Higgs::mass || genu == Gen::Trd
                       ? StandardModel::feynman_rule_u_u_h(genu)
                       : VertexFFS{0, 0}),
      p_vertex_uuz(model.mass() < ZBoson::mass || genu == Gen::Trd
                       ? StandardModel::feynman_rule_u_u_z()
                       : VertexFFV{0, 0}) {}

auto SquaredAmplitudeNToVUU::operator()(
    const std::array<LVector<double>, 3> &ps) const -> double {

  const auto vl_wfs = spinor_ubar(ps[0], 0.0);
  const auto uu_wfs = spinor_ubar(ps[1], p_mu);
  const auto ub_wfs = spinor_v(ps[2], p_mu);

  VectorWf z_wf{};
  ScalarWf h_wf{};

  double msqrd = 0.0;
  for (const auto &vr_wf : p_wfs_n) {
    for (const auto &vl_wf : vl_wfs) {
      z_current(&z_wf, p_vertex_nvz, vl_wf, vr_wf);
      h_current(&h_wf, p_vertex_nvh, vl_wf, vr_wf);
      for (const auto &uu_wf : uu_wfs) {
        for (const auto &ub_wf : ub_wfs) {
          msqrd += std::norm(amplitude(p_vertex_uuh, uu_wf, ub_wf, h_wf) +
                             amplitude(p_vertex_uuz, uu_wf, ub_wf, z_wf));
        }
      }
    }
  }
  return 3.0 * msqrd / 2.0;
}

// ===========================================================================
// ---- N -> nu + d + d ------------------------------------------------------
// ===========================================================================

SquaredAmplitudeNToVDD::SquaredAmplitudeNToVDD(const RhNeutrinoGeV &model,
                                               Gen gend)
    : SquaredAmplitudeNToX<3>(model.mass(), model.theta(), model.gen()),
      p_md(StandardModel::down_type_quark_mass(gend)),
      p_vertex_nvh(model.feynman_rule_n_v_h()),
      p_vertex_nvz(model.feynman_rule_n_v_z()),
      p_vertex_ddh(model.mass() < Higgs::mass
                       ? StandardModel::feynman_rule_d_d_h(gend)
                       : VertexFFS{0, 0}),
      p_vertex_ddz(model.mass() < ZBoson::mass
                       ? StandardModel::feynman_rule_d_d_z()
                       : VertexFFV{0, 0}) {}

auto SquaredAmplitudeNToVDD::operator()(
    const std::array<LVector<double>, 3> &ps) const -> double {

  const auto vl_wfs = spinor_ubar(ps[0], 0.0);
  const auto dd_wfs = spinor_ubar(ps[1], p_md);
  const auto db_wfs = spinor_v(ps[2], p_md);

  VectorWf z_wf{};
  ScalarWf h_wf{};

  double msqrd = 0.0;
  for (const auto &vr_wf : p_wfs_n) {
    for (const auto &vl_wf : vl_wfs) {
      z_current(&z_wf, p_vertex_nvz, vl_wf, vr_wf);
      h_current(&h_wf, p_vertex_nvh, vl_wf, vr_wf);
      for (const auto &uu_wf : dd_wfs) {
        for (const auto &ub_wf : db_wfs) {
          msqrd += std::norm(amplitude(p_vertex_ddh, uu_wf, ub_wf, h_wf) +
                             amplitude(p_vertex_ddz, uu_wf, ub_wf, z_wf));
        }
      }
    }
  }
  return 3.0 * msqrd / 2.0;
}

// ===========================================================================
// ---- N -> l + u + d ------------------------------------------------------
// ===========================================================================

SquaredAmplitudeNToLUD::SquaredAmplitudeNToLUD(const RhNeutrinoGeV &model,
                                               Gen genu, Gen gend)
    : SquaredAmplitudeNToX<3>(model.mass(), model.theta(), model.gen()),
      p_ml(StandardModel::up_type_quark_mass(genu)),
      p_mu(StandardModel::up_type_quark_mass(genu)),
      p_md(StandardModel::down_type_quark_mass(gend)),
      p_vertex_nlw(model.feynman_rule_n_l_w()),
      p_vertex_udw(model.mass() < WBoson::mass
                       ? StandardModel::feynman_rule_u_d_w(genu, gend)
                       : VertexFFV{0, 0}) {}

auto SquaredAmplitudeNToLUD::operator()(
    const std::array<LVector<double>, 3> &ps) const -> double {

  const auto wfs_l = spinor_ubar(ps[0], 0.0);
  const auto wfs_u = spinor_ubar(ps[1], p_mu);
  const auto wfs_d = spinor_v(ps[2], p_mu);

  VectorWf wf_w{};

  double msqrd = 0.0;
  for (const auto &wf_n : p_wfs_n) {
    for (const auto &wf_l : wfs_l) {
      w_current(&wf_w, p_vertex_nlw, wf_l, wf_n);
      for (const auto &wf_u : wfs_u) {
        for (const auto &wf_d : wfs_d) {
          msqrd += std::norm(amplitude(p_vertex_udw, wf_u, wf_d, wf_w));
        }
      }
    }
  }
  return 3.0 * msqrd / 2.0;
}

} // namespace blackthorn
