#include "../Utils.h"
#include "blackthorn/Models/RhNeutrino.h"

namespace blackthorn {

// ===========================================================================
// ---- N -> nu + l + l ------------------------------------------------------
// ===========================================================================

SquaredAmplitudeNToVLL::SquaredAmplitudeNToVLL(const RhNeutrinoGeV &model,
                                               Gen genv, Gen genl1, Gen genl2)
    : SquaredAmplitudeNToX<3>(model.mass(), model.theta(), model.gen()),
      p_ml1(StandardModel::charged_lepton_mass(genl1)),
      p_ml2(StandardModel::charged_lepton_mass(genl2)), p_genv(genv),
      p_genl1(genl1), p_genl2(genl2), p_vertex_nvh(model.feynman_rule_n_v_h()),
      p_vertex_nvz(model.feynman_rule_n_v_z()),
      p_vertex_nlw(model.feynman_rule_n_l_w()),
      // p_vertex_llh(StandardModel::feynman_rule_l_l_h(genl1)),
      // p_vertex_llz(StandardModel::feynman_rule_l_l_z()),
      // p_vertex_vlw(StandardModel::feynman_rule_v_l_w()),
      p_vertex_vlw(model.mass() < WBoson::mass
                       ? StandardModel::feynman_rule_v_l_w()
                       : VertexFFV{0, 0}),
      p_vertex_llh(model.mass() < Higgs::mass
                       ? StandardModel::feynman_rule_l_l_h(genl1)
                       : VertexFFS{0, 0}),
      p_vertex_llz(model.mass() < ZBoson::mass
                       ? StandardModel::feynman_rule_l_l_z()
                       : VertexFFV{0, 0}) {}

SquaredAmplitudeNToVLL::SquaredAmplitudeNToVLL(const RhNeutrinoMeV &model,
                                               Gen genv, Gen genl1, Gen genl2)
    : SquaredAmplitudeNToX<3>(model.mass(), model.theta(), model.gen()),
      p_ml1(StandardModel::charged_lepton_mass(genl1)),
      p_ml2(StandardModel::charged_lepton_mass(genl2)), p_genv(genv),
      p_genl1(genl1), p_genl2(genl2), p_vertex_nvh(model.feynman_rule_n_v_h()),
      p_vertex_nvz(model.feynman_rule_n_v_z()),
      p_vertex_nlw(model.feynman_rule_n_l_w()),
      p_vertex_llh(StandardModel::feynman_rule_l_l_h(genl1)),
      p_vertex_llz(StandardModel::feynman_rule_l_l_z()),
      p_vertex_vlw(StandardModel::feynman_rule_v_l_w()) {}

auto SquaredAmplitudeNToVLL::operator()( // NOLINT
    const std::array<LVector<double>, 3> &ps) const -> double {

  const auto wfs_v = spinor_ubar(ps[0], 0.0);
  const auto wfs_l1 = spinor_ubar(ps[1], p_ml1);
  const auto wfs_l2 = spinor_v(ps[2], p_ml2);

  VectorWf wf_z{};
  VectorWf wf_w{};
  ScalarWf wf_h{};

  // The two W-diagrams require different spinor structures for the
  // neutrinos. In the diagrams connecting the (N, l1) and (nu,l2), the
  // spinor structure is normal (which is what is computed first). In the
  // second diagram connecting (N,l2) and (nu,l1), the neutrino
  // wavefunctions need to be charge conjugated.
  double msqrd{0.0};
#pragma unroll 2
  for (const auto &wf_l1 : wfs_l1) {
#pragma unroll 2
    for (const auto &wf_l2 : wfs_l2) {

      z_current(&wf_z, p_vertex_llz, wf_l1, wf_l2);
      h_current(&wf_h, p_vertex_llh, wf_l1, wf_l2);

      std::complex<double> amp{0, 0};

      if (p_gen == p_genv && p_genl1 == p_genl2) {
#pragma unroll 2
        for (const auto &wf_v : wfs_v) {
#pragma unroll 2
          for (const auto &wf_n : p_wfs_n) {
            // Z-exchange
            amp += amplitude(p_vertex_nvh, wf_v, wf_n, wf_h);
            // H-exchange
            amp += amplitude(p_vertex_nvz, wf_v, wf_n, wf_z);
          }
        }
      }
      // W-exchange with [N, l1] and [nu, l2], -1 for single fermion swap
      if (p_gen == p_genl1 && p_genv == p_genl2) {
#pragma unroll 2
        for (const auto &wf_v : wfs_v) {
#pragma unroll 2
          for (const auto &wf_n : p_wfs_n) {
            w_current(&wf_w, p_vertex_vlw, wf_v, wf_l2);
            amp -= amplitude(p_vertex_nlw, wf_l1, wf_n, wf_w);
          }
        }
      }
      // W-exchange with [CC(N), l2] and [nu, l1]
      if (p_gen == p_genl2 && p_genv == p_genl1) {
#pragma unroll 2
        for (const auto &wf_v : wfs_v) {
#pragma unroll 2
          for (const auto &wf_n : p_wfs_n) {
            w_current(&wf_w, p_vertex_nlw, charge_conjugate(wf_n), wf_l2);
            amp += amplitude(p_vertex_vlw, wf_l1, charge_conjugate(wf_v), wf_w);
          }
        }
      }
      msqrd += std::norm(amp);
    }
  }
  return msqrd / 2.0;
}

// ===========================================================================
// ---- N -> nu + nu + nu ----------------------------------------------------
// ===========================================================================

SquaredAmplitudeNToVVV::SquaredAmplitudeNToVVV(const RhNeutrinoGeV &model,
                                               Gen genv1, Gen genv2, Gen genv3)
    : SquaredAmplitudeNToX<3>(model.mass(), model.theta(), model.gen()),
      p_vertex_nvz(model.feynman_rule_n_v_z()),
      p_vertex_vvz(model.mass() < ZBoson::mass
                       ? StandardModel::feynman_rule_v_v_z()
                       : VertexFFV{0, 0}),
      p_genv1(genv1), p_genv2(genv2), p_genv3(genv3) {}

SquaredAmplitudeNToVVV::SquaredAmplitudeNToVVV(const RhNeutrinoMeV &model,
                                               Gen genv1, Gen genv2, Gen genv3)
    : SquaredAmplitudeNToX<3>(model.mass(), model.theta(), model.gen()),
      p_vertex_nvz(model.feynman_rule_n_v_z()),
      p_vertex_vvz(StandardModel::feynman_rule_v_v_z()), p_genv1(genv1),
      p_genv2(genv2), p_genv3(genv3) {}

auto SquaredAmplitudeNToVVV::operator()( // NOLINT
    const std::array<LVector<double>, 3> &ps) const -> double {

  const auto wfs_v1 = spinor_ubar(ps[0], 0.0);
  const auto wfs_v2 = spinor_ubar(ps[1], 0.0);
  const auto wfs_v3 = spinor_v(ps[2], 0.0);

  // Charge-conjugated wavefunctions needed for
  const std::array<DiracWfI, 2> wfs_v2_cc = {charge_conjugate(wfs_v2[0]),
                                             charge_conjugate(wfs_v2[1])};
  const std::array<DiracWfO, 2> wfs_v3_cc = {charge_conjugate(wfs_v3[0]),
                                             charge_conjugate(wfs_v3[1])};

  VectorWf wf_z{};
  ScalarWf wf_h{};

  double msqrd{0.0};
  for (const auto &wf_n : p_wfs_n) {
    std::complex<double> amp{0.0, 0.0};

    // Diagrams with (N,v1) and (v2,v3)
    if (p_gen == p_genv1 && p_genv2 == p_genv3) {
      for (const auto &wf_v1 : wfs_v1) {
        z_current(&wf_z, p_vertex_nvz, wf_v1, wf_n);
        for (const auto &wf_v2 : wfs_v2) {
          for (const auto &wf_v3 : wfs_v3) {
            amp += amplitude(p_vertex_vvz, wf_v2, wf_v3, wf_z);
          }
        }
      }
    }

    // Diagrams with (N,v2) and (v1,v3)
    if (p_gen == p_genv2 && p_genv1 == p_genv3) {
      for (const auto &wf_v2 : wfs_v2) {
        z_current(&wf_z, p_vertex_nvz, wf_v2, wf_n);
        for (const auto &wf_v1 : wfs_v1) {
          for (const auto &wf_v3 : wfs_v3) {
            // Minus sign for single fermion swap
            amp -= amplitude(p_vertex_vvz, wf_v1, wf_v3, wf_z);
          }
        }
      }
    }

    // Diagrams with (N,v3) and (v1,v2)
    if (p_gen == p_genv3 && p_genv1 == p_genv2) {
      for (const auto &wf_v3 : wfs_v3_cc) {
        z_current(&wf_z, p_vertex_nvz, wf_v3, wf_n);
        for (const auto &wf_v1 : wfs_v1) {
          for (const auto &wf_v2 : wfs_v2_cc) {
            amp += amplitude(p_vertex_vvz, wf_v1, wf_v2, wf_z);
          }
        }
      }
    }

    msqrd += std::norm(amp);
  }
  return msqrd / 2.0;
}

} // namespace blackthorn
