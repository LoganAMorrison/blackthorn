#include "blackthorn/Models/RhNeutrino.h"
#include "blackthorn/Models/StandardModel.h"

namespace blackthorn {

// ===========================================================================
// ---- N-L-Pi ---------------------------------------------------------------
// ===========================================================================

auto feynman_rule_n_l_m(double theta, const std::complex<double> &vckm)
    -> VertexFFSDeriv {
  static constexpr auto gg = -2 * FPI * GF;
  const auto g = gg * vckm * sin(theta);
  return VertexFFSDeriv{gg, 0};
}

auto RhNeutrinoMeV::feynman_rule_n_lm_pip() const -> VertexFFSDeriv {
  return feynman_rule_n_l_m(p_theta, std::conj(VUD));
}

auto RhNeutrinoMeV::feynman_rule_n_lm_kp() const -> VertexFFSDeriv {
  return feynman_rule_n_l_m(p_theta, std::conj(VUS));
}

auto RhNeutrinoMeV::feynman_rule_n_lp_pim() const -> VertexFFSDeriv {
  return feynman_rule_n_l_m(p_theta, VUD);
}

auto RhNeutrinoMeV::feynman_rule_n_lp_km() const -> VertexFFSDeriv {
  return feynman_rule_n_l_m(p_theta, VUS);
}

// ===========================================================================
// ---- N-Nu-(Neutral Meson) -------------------------------------------------
// ===========================================================================

auto feynman_rule_n_nu_mn(double theta, double pre) -> VertexFFSDeriv {
  using tools::im;
  using tools::sqr;
  static constexpr auto cw2 = sqr(StandardModel::cw);
  static constexpr auto gg = std::complex<double>{0, -FPI * GF / cw2};

  const auto g = pre * gg * sin(2 * theta);
  return VertexFFSDeriv{g, g};
}

auto RhNeutrinoMeV::feynman_rule_n_v_pi0() const -> VertexFFSDeriv {
  return feynman_rule_n_nu_mn(p_theta, M_SQRT1_2);
}

auto RhNeutrinoMeV::feynman_rule_n_v_eta() const -> VertexFFSDeriv {
  static constexpr double sqrt_1_6 = 0.40824829046386302;
  return feynman_rule_n_nu_mn(p_theta, sqrt_1_6);
}

// ===========================================================================
// ---- N-Nu-Pi-Pi -----------------------------------------------------------
// ===========================================================================

auto feynman_rule_n_nu_mc_mc(double theta) -> VertexFFSS {
  using tools::im;
  using tools::sqr;
  static constexpr auto cw2 = sqr(StandardModel::cw);
  static constexpr auto sw2 = sqr(StandardModel::sw);
  static constexpr double gg = GF * (-1.0 + 2.0 * sw2) / (M_SQRT2 * cw2);

  const auto g = im * gg * sin(2 * theta);
  return VertexFFSS{g, g, -g, -g};
}

auto RhNeutrinoMeV::feynman_rule_n_v_pi_pi() const -> VertexFFSS {
  return feynman_rule_n_nu_mc_mc(p_theta);
}

auto RhNeutrinoMeV::feynman_rule_n_v_k_k() const -> VertexFFSS {
  return feynman_rule_n_nu_mc_mc(p_theta);
}

// ===========================================================================
// ---- N-L-Pi-Pi0 -----------------------------------------------------------
// ===========================================================================

auto feynman_rule_n_l_mn_mc(double theta, std::complex<double> pre)
    -> VertexFFSS {
  static constexpr auto gg = std::complex<double>{0.0, GF};
  const auto g = pre * sin(theta);
  return VertexFFSS{g, 0, -g, 0};
}

auto RhNeutrinoMeV::feynman_rule_n_lm_pip_pi0() const -> VertexFFSS {
  return feynman_rule_n_l_mn_mc(p_theta, -2.0 * std::conj(VUD));
}

auto RhNeutrinoMeV::feynman_rule_n_lp_pim_pi0() const -> VertexFFSS {
  static constexpr auto vckm = StandardModel::ckm<Gen::Fst, Gen::Snd>();
  return feynman_rule_n_l_mn_mc(p_theta, 2.0 * VUD);
}

auto RhNeutrinoMeV::feynman_rule_n_lm_pip_k0() const -> VertexFFSS {
  static constexpr auto vckm = StandardModel::ckm<Gen::Fst, Gen::Snd>();
  return feynman_rule_n_l_mn_mc(p_theta, M_SQRT2 * std::conj(VUS));
}

auto RhNeutrinoMeV::feynman_rule_n_lp_pim_k0() const -> VertexFFSS {
  static constexpr auto vckm = StandardModel::ckm<Gen::Fst, Gen::Snd>();
  return feynman_rule_n_l_mn_mc(p_theta, -M_SQRT2 * VUS);
}

auto RhNeutrinoMeV::feynman_rule_n_lm_kp_pi0() const -> VertexFFSS {
  return feynman_rule_n_l_mn_mc(p_theta, std::conj(VUS));
}

auto RhNeutrinoMeV::feynman_rule_n_lp_km_pi0() const -> VertexFFSS {
  static constexpr auto vckm = StandardModel::ckm<Gen::Fst, Gen::Snd>();
  return feynman_rule_n_l_mn_mc(p_theta, VUS);
}

// ===========================================================================
// ---- N-L-Pi-Photon --------------------------------------------------------
// ===========================================================================

auto feynman_rule_n_l_mc_a(double theta, const std::complex<double> &pre)
    -> VertexFFSV {
  using tools::im;
  using tools::sqr;
  static constexpr auto gg = 2 * QE * GF * FPI;
  const auto g = pre * sin(theta);
  return VertexFFSV{g, 0};
}

auto RhNeutrinoMeV::feynman_rule_n_lm_pip_a() const -> VertexFFSV {
  return feynman_rule_n_l_mc_a(p_theta, -std::conj(VUD));
}

auto RhNeutrinoMeV::feynman_rule_n_lm_kp_a() const -> VertexFFSV {
  return feynman_rule_n_l_mc_a(p_theta, -std::conj(VUS));
}

auto RhNeutrinoMeV::feynman_rule_n_lp_pim_a() const -> VertexFFSV {
  return feynman_rule_n_l_mc_a(p_theta, VUD);
}

auto RhNeutrinoMeV::feynman_rule_n_lp_km_a() const -> VertexFFSV {
  return feynman_rule_n_l_mc_a(p_theta, VUS);
}

// ===========================================================================
// ---- N-Nu-Photon ----------------------------------------------------------
// ===========================================================================

// auto RhNeutrinoMeV::feynman_rule_n_v_a() const;

} // namespace blackthorn
