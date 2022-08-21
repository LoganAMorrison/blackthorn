#include "blackthorn/Models/RhNeutrino.h"
#include "blackthorn/Models/StandardModel.h"

namespace blackthorn {

// ===========================================================================
// ---- N -> v + pi0 ---------------------------------------------------------
// ===========================================================================

SquaredAmplitudeNToVPi0::SquaredAmplitudeNToVPi0(const RhNeutrinoMeV &model)
    : SquaredAmplitudeNToX<2>(model.mass(), model.theta(), model.gen()),
      p_vertex(model.feynman_rule_n_v_pi0()) {}

auto SquaredAmplitudeNToVPi0::operator()(
    const std::array<LVector<double>, 2> &ps) const -> double {

  const auto vl_wfs = spinor_ubar(ps[0], 0.0);
  const auto p0_wf = scalar_wf(ps[1], Outgoing);

  double msqrd = 0.0;
#pragma unroll 2
  for (const auto &vr_wf : p_wfs_n) {
#pragma unroll 2
    for (const auto &vl_wf : vl_wfs) {
      msqrd += std::norm(amplitude(p_vertex, vl_wf, vr_wf, p0_wf));
    }
  }
  return msqrd / 2.0;
}

// ===========================================================================
// ---- N -> v + eta ---------------------------------------------------------
// ===========================================================================

SquaredAmplitudeNToVEta::SquaredAmplitudeNToVEta(const RhNeutrinoMeV &model)
    : SquaredAmplitudeNToX<2>(model.mass(), model.theta(), model.gen()),
      p_vertex(model.feynman_rule_n_v_eta()) {}

auto SquaredAmplitudeNToVEta::operator()(
    const std::array<LVector<double>, 2> &ps) const -> double {

  const auto vl_wfs = spinor_ubar(ps[0], 0.0);
  const auto m0_wf = scalar_wf(ps[1], Outgoing);

  double msqrd = 0.0;
#pragma unroll 2
  for (const auto &vr_wf : p_wfs_n) {
#pragma unroll 2
    for (const auto &vl_wf : vl_wfs) {
      msqrd += std::norm(amplitude(p_vertex, vl_wf, vr_wf, m0_wf));
    }
  }
  return msqrd / 2.0;
}

// ===========================================================================
// ---- N -> l + pi ----------------------------------------------------------
// ===========================================================================

SquaredAmplitudeNToLPi::SquaredAmplitudeNToLPi(const RhNeutrinoMeV &model,
                                               Gen gen)
    : SquaredAmplitudeNToX<2>(model.mass(), model.theta(), model.gen()),
      p_ml(StandardModel::charged_lepton_mass(gen)),
      p_vertex(model.feynman_rule_n_lm_pip()) {}

auto SquaredAmplitudeNToLPi::operator()(
    const std::array<LVector<double>, 2> &ps) const -> double {

  const auto vl_wfs = spinor_ubar(ps[0], 0.0);
  const auto pp_wf = scalar_wf(ps[1], Outgoing);

  double msqrd = 0.0;
#pragma unroll 2
  for (const auto &vr_wf : p_wfs_n) {
#pragma unroll 2
    for (const auto &vl_wf : vl_wfs) {
      msqrd += std::norm(amplitude(p_vertex, vl_wf, vr_wf, pp_wf));
    }
  }
  return msqrd / 2.0;
}

// ===========================================================================
// ---- N -> l + k -----------------------------------------------------------
// ===========================================================================

SquaredAmplitudeNToLK::SquaredAmplitudeNToLK(const RhNeutrinoMeV &model)
    : SquaredAmplitudeNToX<2>(model.mass(), model.theta(), model.gen()),
      p_ml(StandardModel::charged_lepton_mass(model.gen())),
      p_vertex(model.feynman_rule_n_lm_kp()) {}

auto SquaredAmplitudeNToLK::operator()(
    const std::array<LVector<double>, 2> &ps) const -> double {

  const auto vl_wfs = spinor_ubar(ps[0], 0.0);
  const auto pk_wf = scalar_wf(ps[1], Outgoing);

  double msqrd = 0.0;
#pragma unroll 2
  for (const auto &vr_wf : p_wfs_n) {
#pragma unroll 2
    for (const auto &vl_wf : vl_wfs) {
      msqrd += std::norm(amplitude(p_vertex, vl_wf, vr_wf, pk_wf));
    }
  }
  return msqrd / 2.0;
}

// ===========================================================================
// ---- N -> nu + pi + pi ----------------------------------------------------
// ===========================================================================

SquaredAmplitudeNToVPiPi::SquaredAmplitudeNToVPiPi(const RhNeutrinoMeV &model)
    : SquaredAmplitudeNToX<3>(model.mass(), model.theta(), model.gen()),
      p_vertex(model.feynman_rule_n_v_pi_pi()) {}

auto SquaredAmplitudeNToVPiPi::operator()(
    const std::array<LVector<double>, 3> &ps) const -> double {

  const auto vl_wfs = spinor_ubar(ps[0], 0.0);
  const auto pp_wf = scalar_wf(ps[1], Outgoing);
  const auto pm_wf = scalar_wf(ps[2], Outgoing);

  double msqrd = 0.0;
#pragma unroll 2
  for (const auto &vr_wf : p_wfs_n) {
#pragma unroll 2
    for (const auto &vl_wf : vl_wfs) {
      msqrd += std::norm(amplitude(p_vertex, vl_wf, vr_wf, pp_wf, pm_wf));
    }
  }
  return msqrd / 2.0;
}

// ===========================================================================
// ---- N -> nu + K + K ------------------------------------------------------
// ===========================================================================

SquaredAmplitudeNToVKK::SquaredAmplitudeNToVKK(const RhNeutrinoMeV &model)
    : SquaredAmplitudeNToX<3>(model.mass(), model.theta(), model.gen()),
      p_vertex(model.feynman_rule_n_v_k_k()) {}

auto SquaredAmplitudeNToVKK::operator()(
    const std::array<LVector<double>, 3> &ps) const -> double {

  const auto vl_wfs = spinor_ubar(ps[0], 0.0);
  const auto pp_wf = scalar_wf(ps[1], Outgoing);
  const auto pm_wf = scalar_wf(ps[2], Outgoing);

  double msqrd = 0.0;
  for (const auto &vr_wf : p_wfs_n) {
    for (const auto &vl_wf : vl_wfs) {
      msqrd += std::norm(amplitude(p_vertex, vl_wf, vr_wf, pp_wf, pm_wf));
    }
  }
  return msqrd / 2.0;
}

// ===========================================================================
// ---- N -> l + pi + pi0 ----------------------------------------------------
// ===========================================================================

static auto l_pi_pi0_pre(const RhNeutrinoMeV &model) -> double {
  const double v2 = std::norm(StandardModel::ckm<Gen::Fst, Gen::Fst>());
  return 2 * std::pow(StandardModel::g_fermi, 2) * std::pow(model.mass(), 4) *
         v2 * std::pow(std::sin(model.theta()), 2);
}

static auto l_pi_pi0_c0(const RhNeutrinoMeV &model) -> double {
  const double ml = StandardModel::charged_lepton_mass(model.gen());
  const double rl = ml / model.mass();
  const double rp = ChargedPion::mass / model.mass();
  const double r0 = NeutralPion::mass / model.mass();
  return std::pow(rl, 4) - 2 * std::pow(rp, 2) +
         std::pow(r0, 2) * (-2 + 4 * std::pow(rp, 2));
}
static auto l_pi_pi0_c1(const RhNeutrinoMeV &model) -> double {
  const double ml = StandardModel::charged_lepton_mass(model.gen());
  const double rl = ml / model.mass();
  return (1 - std::pow(rl, 2));
}
static auto l_pi_pi0_c2(const RhNeutrinoMeV &model) -> double {
  const double ml = StandardModel::charged_lepton_mass(model.gen());
  const double rl = ml / model.mass();
  const double rp = ChargedPion::mass / model.mass();
  const double r0 = NeutralPion::mass / model.mass();
  return -4 * (std::pow(r0, 2) + std::pow(rl, 2) + std::pow(rp, 2));
}

// SquaredAmplitudeNToLPiPi0::SquaredAmplitudeNToLPiPi0(const RhNeutrinoMeV
// &model)
//     : SquaredAmplitudeNToX<3>(model.mass(), model.theta(), model.gen()),
//       p_pre(l_pi_pi0_pre(model)), p_c0(l_pi_pi0_c0(model)),
//       p_c1(l_pi_pi0_c0(model)), p_c2(l_pi_pi0_c0(model)) {}

// auto SquaredAmplitudeNToLPiPi0::operator()(
//     const std::array<LVector<double>, 3> &ps) const -> double {

//   const double m2 = p_mass * p_mass;
//   const double ss = lnorm_sqr(ps[1] + ps[2]) / m2;
//   const double tt = lnorm_sqr(ps[0] + ps[2]) / m2;
//   const double uu = lnorm_sqr(ps[0] + ps[1]) / m2;

//   return p_pre * (p_c0 + p_c1 * ss + p_c2 * tt + 4 * ss * tt + 4 * ss);
// }

SquaredAmplitudeNToLPiPi0::SquaredAmplitudeNToLPiPi0(const RhNeutrinoMeV &model)
    : SquaredAmplitudeNToX<3>(model.mass(), model.theta(), model.gen()),
      p_vertex(model.feynman_rule_n_lm_pip_pi0()) {}

auto SquaredAmplitudeNToLPiPi0::operator()(
    const std::array<LVector<double>, 3> &ps) const -> double {
  const double s = lnorm_sqr(ps[1] + ps[2]);
  const double t = lnorm_sqr(ps[0] + ps[2]);
  const double u = lnorm_sqr(ps[0] + ps[1]);
  const double ml = StandardModel::charged_lepton_mass(p_gen);
  const double mpi = ChargedPion::mass;
  const double mpi0 = NeutralPion::mass;
  const double ckm = std::norm(StandardModel::ckm<Gen::Fst, Gen::Snd>());
  const double gf = StandardModel::g_fermi;
  return 2 * pow(gf, 2) * ckm *
         (pow(ml, 4) + pow(p_mass, 4) + 4 * pow(mpi, 2) * (pow(mpi0, 2) - t) +
          4 * t * (-pow(mpi0, 2) + s + t) - pow(p_mass, 2) * (s + 4 * t) -
          pow(ml, 2) * (-2 * pow(p_mass, 2) + s + 4 * t)) *
         pow(sin(p_theta), 2);
}
} // namespace blackthorn
