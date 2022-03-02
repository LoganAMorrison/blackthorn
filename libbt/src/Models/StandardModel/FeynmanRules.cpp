#include "blackthorn/Models/StandardModel.h"
namespace blackthorn {

// ===========================================================================
// ---- Fermion-Fermion-Higgs ------------------------------------------------
// ===========================================================================

template <typename F> struct sm_feynman_rule_ffh { // NOLINT
private:
  static_assert(is_fermion<F>::value, "Type must be a fermion.");
  static constexpr double g = -field_attrs<F>::mass() / Higgs::vev;
  static constexpr std::complex<double> gg = std::complex<double>{0.0, g};

public:
  using value_type = VertexFFS;
  static constexpr value_type value = value_type{gg, gg};
};

auto sm_feynman_rule_uuh(Gen gen) -> VertexFFS {
  if (gen == Gen::Fst) {
    return sm_feynman_rule_ffh<UpQuark>::value;
  }
  if (gen == Gen::Snd) {
    return sm_feynman_rule_ffh<CharmQuark>::value;
  }
  return sm_feynman_rule_ffh<TopQuark>::value;
}

auto sm_feynman_rule_ddh(Gen gen) -> VertexFFS {
  if (gen == Gen::Fst) {
    return sm_feynman_rule_ffh<DownQuark>::value;
  }
  if (gen == Gen::Snd) {
    return sm_feynman_rule_ffh<StrangeQuark>::value;
  }
  return sm_feynman_rule_ffh<BottomQuark>::value;
}

auto sm_feynman_rule_llh(Gen gen) -> VertexFFS {
  if (gen == Gen::Fst) {
    return sm_feynman_rule_ffh<Electron>::value;
  }
  if (gen == Gen::Snd) {
    return sm_feynman_rule_ffh<Muon>::value;
  }
  return sm_feynman_rule_ffh<Tau>::value;
}

// ===========================================================================
// ---- Fermion-Fermion-ZBoson -----------------------------------------------
// ===========================================================================

template <typename F> struct sm_feynman_rule_ffz { // NOLINT
private:
  static_assert(is_fermion<F>::value, "Type must be a fermion.");
  static constexpr double t3 = quantum_numbers<F>::weak_iso_spin();
  static constexpr double q = quantum_numbers<F>::charge();
  static constexpr double gzl = StandardModel::qe *
                                (t3 - q * tools::sqr(StandardModel::sw)) /
                                (StandardModel::sw * StandardModel::cw);
  static constexpr double gzr =
      -q * StandardModel::qe * StandardModel::sw / StandardModel::cw;

public:
  using value_type = VertexFFV;
  static constexpr value_type value = value_type{gzl, gzr};
};

auto sm_feynman_rule_uuz() -> VertexFFV {
  return sm_feynman_rule_ffz<UpQuark>::value;
}

auto sm_feynman_rule_ddz() -> VertexFFV {
  return sm_feynman_rule_ffz<DownQuark>::value;
}

auto sm_feynman_rule_llz() -> VertexFFV {
  return sm_feynman_rule_ffz<Electron>::value;
}

auto sm_feynman_rule_vvz() -> VertexFFV {
  return sm_feynman_rule_ffz<ElectronNeutrino>::value;
}

// ===========================================================================
// ---- Fermion-Fermion-WBoson -----------------------------------------------
// ===========================================================================

auto sm_feynman_rule_lvw() -> VertexFFV {
  static constexpr double g = StandardModel::qe / (M_SQRT2 * StandardModel::sw);
  static constexpr VertexFFV value = VertexFFV{g, 0};
  return value;
}

auto sm_feynman_rule_udw(Gen U, Gen D) -> VertexFFV {
  static constexpr std::complex<double> g =
      StandardModel::qe / (M_SQRT2 * StandardModel::sw);
  const auto ckm = StandardModel::ckm(U, D);
  return VertexFFV{ckm * g, ckm * g};
}

// ===========================================================================
// ---- F-F-H ----------------------------------------------------------------
// ===========================================================================

auto StandardModel::feynman_rule_l_l_h(Gen gen) -> VertexFFS {
  if (gen == Gen::Fst) {
    return feynman_rule_f_f_h<Electron>();
  }
  if (gen == Gen::Snd) {
    return feynman_rule_f_f_h<Muon>();
  }
  return feynman_rule_f_f_h<Tau>();
}

auto StandardModel::feynman_rule_u_u_h(Gen gen) -> VertexFFS {
  if (gen == Gen::Fst) {
    return feynman_rule_f_f_h<UpQuark>();
  }
  if (gen == Gen::Snd) {
    return feynman_rule_f_f_h<CharmQuark>();
  }
  return feynman_rule_f_f_h<TopQuark>();
}

auto StandardModel::feynman_rule_d_d_h(Gen gen) -> VertexFFS {
  if (gen == Gen::Fst) {
    return feynman_rule_f_f_h<DownQuark>();
  }
  if (gen == Gen::Snd) {
    return feynman_rule_f_f_h<StrangeQuark>();
  }
  return feynman_rule_f_f_h<BottomQuark>();
}

// ===========================================================================
// ---- F-F-Z ----------------------------------------------------------------
// ===========================================================================

auto StandardModel::feynman_rule_v_v_z() -> VertexFFV {
  // Rule is independent of generation, so just use first generation
  return feynman_rule_f_f_z<ElectronNeutrino>();
}

auto StandardModel::feynman_rule_l_l_z() -> VertexFFV {
  // Rule is independent of generation, so just use first generation
  return feynman_rule_f_f_z<Electron>();
}

auto StandardModel::feynman_rule_u_u_z() -> VertexFFV {
  // Rule is independent of generation, so just use first generation
  return feynman_rule_f_f_z<UpQuark>();
}

auto StandardModel::feynman_rule_d_d_z() -> VertexFFV {
  // Rule is independent of generation, so just use first generation
  return feynman_rule_f_f_z<DownQuark>();
}

// ===========================================================================
// ---- V-L-W ----------------------------------------------------------------
// ===========================================================================

auto StandardModel::feynman_rule_v_l_w() -> VertexFFV {
  // Rule is independent of generation, so just use first generation
  return feynman_rule_u_d_w<ElectronNeutrino, Electron>();
}

// ===========================================================================
// ---- U-D-W ----------------------------------------------------------------
// ===========================================================================

auto StandardModel::feynman_rule_u_d_w(Gen genu, Gen gend) -> VertexFFV {
  if (genu == Gen::Fst) {
    if (gend == Gen::Fst) {
      return feynman_rule_u_d_w<UpQuark, DownQuark>();
    }
    if (gend == Gen::Snd) {
      return feynman_rule_u_d_w<UpQuark, StrangeQuark>();
    }
    return feynman_rule_u_d_w<UpQuark, BottomQuark>();
  }
  if (genu == Gen::Snd) {
    if (gend == Gen::Fst) {
      return feynman_rule_u_d_w<CharmQuark, DownQuark>();
    }
    if (gend == Gen::Snd) {
      return feynman_rule_u_d_w<CharmQuark, StrangeQuark>();
    }
    return feynman_rule_u_d_w<CharmQuark, BottomQuark>();
  }
  if (gend == Gen::Fst) {
    return feynman_rule_u_d_w<TopQuark, DownQuark>();
  }
  if (gend == Gen::Snd) {
    return feynman_rule_u_d_w<TopQuark, StrangeQuark>();
  }
  return feynman_rule_u_d_w<TopQuark, BottomQuark>();
}

} // namespace blackthorn
