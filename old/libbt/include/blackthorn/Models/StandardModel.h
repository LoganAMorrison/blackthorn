#ifndef BLACKTHORN_MODELS_STANDARD_MODEL_H
#define BLACKTHORN_MODELS_STANDARD_MODEL_H

#include "blackthorn/Amplitudes.h"
#include "blackthorn/Models/Base.h"
#include "blackthorn/Models/Particles.h"
#include "blackthorn/Tools.h"

namespace blackthorn {

class StandardModel {
private:
  // CKM matrix element of u-d
  static constexpr std::complex<double> ckm_ud =
      std::complex<double>{0.9742855469766043, 0.0};
  // CKM matrix element of u-s
  static constexpr std::complex<double> ckm_us =
      std::complex<double>{0.22528978739658034, 0.0};
  // CKM matrix element of u-b
  static constexpr std::complex<double> ckm_ub =
      std::complex<double>{0.003467579534847188, -0.000400673772678475};
  // CKM matrix element of c-d
  static constexpr std::complex<double> ckm_cd =
      std::complex<double>{-0.22523919033512343, 0.000016074829716163196};
  // CKM matrix element of c-s
  static constexpr std::complex<double> ckm_cs =
      std::complex<double>{0.9734329404782395, 3.7170775861643788e-6};
  // CKM matrix element of c-b
  static constexpr std::complex<double> ckm_cb =
      std::complex<double>{0.041177873389110276, 0.0};
  // CKM matrix element of t-d
  static constexpr std::complex<double> ckm_td =
      std::complex<double>{0.005901499687662993, 0.0003900419379730849};
  // CKM matrix element of t-s
  static constexpr std::complex<double> ckm_ts =
      std::complex<double>{-0.04090004814585696, 0.0000901916953960691};
  // CKM matrix element of t-b
  static constexpr std::complex<double> ckm_tb =
      std::complex<double>{0.9991457341628637, 0.0};

  static constexpr std::array<std::array<std::complex<double>, 3>, 3>
      ckm_matrix_ = {
          std::array<std::complex<double>, 3>{ckm_ud, ckm_ud, ckm_ud},
          std::array<std::complex<double>, 3>{ckm_cd, ckm_cd, ckm_cd},
          std::array<std::complex<double>, 3>{ckm_td, ckm_td, ckm_td},
  };

  static constexpr auto gen_to_idx(Gen U) -> size_t {
    if (U == Gen::Fst) {
      return 0;
    }
    if (U == Gen::Snd) {
      return 1;
    }
    return 2;
  }

public:
  // Fermi constant in GeV^-2
  static constexpr double g_fermi = 1.1663787e-5;
  // Fine-structure constant of the electric coupling constant at zero momentum
  static constexpr double alpha_em = 1.0 / 137.0; // at p^2 = 0
  // Electric change coupling at zero momentum
  static constexpr double qe = 0.302862;
  // Sine of the weak mixing angle
  static constexpr double sw = 0.480853;
  // Cosine of the weak mixing angle
  static constexpr double cw = 0.876801;
  // reduced plank mass
  static constexpr double mplank = 2.435e18;

  template <Gen U> static constexpr auto up_type_quark_mass() -> double;
  template <Gen D> static constexpr auto down_type_quark_mass() -> double;
  template <Gen L> static constexpr auto charged_lepton_mass() -> double;

  static constexpr auto up_type_quark_mass(Gen U) -> double;
  static constexpr auto down_type_quark_mass(Gen D) -> double;
  static constexpr auto charged_lepton_mass(Gen L) -> double;

  template <Gen U> static constexpr auto up_type_quark_pdg() -> int;
  template <Gen D> static constexpr auto down_type_quark_pdg() -> int;
  template <Gen L> static constexpr auto charged_lepton_pdg() -> int;
  template <Gen L> static constexpr auto neutrino_pdg() -> int;

  static constexpr auto up_type_quark_pdg(Gen U) -> int;
  static constexpr auto down_type_quark_pdg(Gen D) -> int;
  static constexpr auto charged_lepton_pdg(Gen L) -> int;
  static constexpr auto neutrino_pdg(Gen L) -> int;

  template <Gen U, Gen D> static constexpr auto ckm() -> std::complex<double>;
  static constexpr auto ckm(Gen U, Gen D) -> std::complex<double>;

  template <class F> static auto feynman_rule_f_f_h() -> VertexFFS;
  template <class F> static auto feynman_rule_f_f_z() -> VertexFFV;
  template <class Up, class Down> static auto feynman_rule_u_d_w() -> VertexFFV;
  template <class F> static auto feynman_rule_f_f_a() -> VertexFFV;

  static auto feynman_rule_l_l_h(Gen gen) -> VertexFFS;
  static auto feynman_rule_u_u_h(Gen gen) -> VertexFFS;
  static auto feynman_rule_d_d_h(Gen gen) -> VertexFFS;

  static auto feynman_rule_v_v_z() -> VertexFFV;
  static auto feynman_rule_l_l_z() -> VertexFFV;
  static auto feynman_rule_u_u_z() -> VertexFFV;
  static auto feynman_rule_d_d_z() -> VertexFFV;

  static auto feynman_rule_v_l_w() -> VertexFFV;
  static auto feynman_rule_u_d_w(Gen, Gen) -> VertexFFV;
};

// ===========================================================================
// ---- Quark Quantities -----------------------------------------------------
// ===========================================================================

template <Gen U> constexpr auto StandardModel::up_type_quark_mass() -> double {
  if constexpr (U == Gen::Fst) {
    return UpQuark::mass;
  }
  if constexpr (U == Gen::Snd) {
    return CharmQuark::mass;
  }
  return TopQuark::mass;
}

constexpr auto StandardModel::up_type_quark_mass(Gen U) -> double {
  if (U == Gen::Fst) {
    return UpQuark::mass;
  }
  if (U == Gen::Snd) {
    return CharmQuark::mass;
  }
  return TopQuark::mass;
}

template <Gen D>
constexpr auto StandardModel::down_type_quark_mass() -> double {
  if constexpr (D == Gen::Fst) {
    return DownQuark::mass;
  }
  if constexpr (D == Gen::Snd) {
    return StrangeQuark::mass;
  }
  return BottomQuark::mass;
}

constexpr auto StandardModel::down_type_quark_mass(Gen D) -> double {
  if (D == Gen::Fst) {
    return DownQuark::mass;
  }
  if (D == Gen::Snd) {
    return StrangeQuark::mass;
  }
  return BottomQuark::mass;
}

template <Gen U> constexpr auto StandardModel::up_type_quark_pdg() -> int {
  if constexpr (U == Gen::Fst) {
    return UpQuark::pdg;
  }
  if constexpr (U == Gen::Snd) {
    return CharmQuark::pdg;
  }
  return TopQuark::pdg;
}

constexpr auto StandardModel::up_type_quark_pdg(Gen U) -> int {
  if (U == Gen::Fst) {
    return UpQuark::pdg;
  }
  if (U == Gen::Snd) {
    return CharmQuark::pdg;
  }
  return TopQuark::pdg;
}

template <Gen U> constexpr auto StandardModel::down_type_quark_pdg() -> int {
  if constexpr (U == Gen::Fst) {
    return DownQuark::pdg;
  }
  if constexpr (U == Gen::Snd) {
    return StrangeQuark::pdg;
  }
  return BottomQuark::pdg;
}

constexpr auto StandardModel::down_type_quark_pdg(Gen D) -> int {
  if (D == Gen::Fst) {
    return DownQuark::pdg;
  }
  if (D == Gen::Snd) {
    return StrangeQuark::pdg;
  }
  return BottomQuark::pdg;
}

// ===========================================================================
// ---- Lepton Masses ---------------------------------------------------------
// ===========================================================================

template <Gen L> constexpr auto StandardModel::charged_lepton_mass() -> double {
  if constexpr (L == Gen::Fst) {
    return Electron::mass;
  }
  if constexpr (L == Gen::Snd) {
    return Muon::mass;
  }
  return Tau::mass;
}

constexpr auto StandardModel::charged_lepton_mass(Gen L) -> double {
  if (L == Gen::Fst) {
    return Electron::mass;
  }
  if (L == Gen::Snd) {
    return Muon::mass;
  }
  return Tau::mass;
}

template <Gen L> constexpr auto StandardModel::neutrino_pdg() -> int {
  if constexpr (L == Gen::Fst) {
    return ElectronNeutrino::pdg;
  }
  if constexpr (L == Gen::Snd) {
    return MuonNeutrino::pdg;
  }
  return TauNeutrino::pdg;
}

constexpr auto StandardModel::neutrino_pdg(Gen L) -> int {
  if (L == Gen::Fst) {
    return ElectronNeutrino::pdg;
  }
  if (L == Gen::Snd) {
    return MuonNeutrino::pdg;
  }
  return TauNeutrino::pdg;
}

template <Gen L> constexpr auto StandardModel::charged_lepton_pdg() -> int {
  if constexpr (L == Gen::Fst) {
    return Electron::pdg;
  }
  if constexpr (L == Gen::Snd) {
    return Muon::pdg;
  }
  return Tau::pdg;
}

constexpr auto StandardModel::charged_lepton_pdg(Gen L) -> int {
  if (L == Gen::Fst) {
    return Electron::pdg;
  }
  if (L == Gen::Snd) {
    return Muon::pdg;
  }
  return Tau::pdg;
}

// ===========================================================================
// ---- CKM Matrix -----------------------------------------------------------
// ===========================================================================

template <Gen U, Gen D>
constexpr auto StandardModel::ckm() -> std::complex<double> {

  if constexpr (U == Gen::Fst) {
    if (D == Gen::Fst) {
      return ckm_ud;
    }
    if (D == Gen::Snd) {
      return ckm_us;
    }
    return ckm_ub;
  }
  if constexpr (U == Gen::Snd) {
    if (D == Gen::Fst) {
      return ckm_cd;
    }
    if constexpr (D == Gen::Snd) {
      return ckm_cs;
    }
    return ckm_cb;
  }
  if constexpr (D == Gen::Fst) {
    return ckm_td;
  }
  if constexpr (D == Gen::Snd) {
    return ckm_ts;
  }
  return ckm_tb;
}

constexpr auto StandardModel::ckm(Gen U, Gen D) -> std::complex<double> {
  return ckm_matrix_.at(gen_to_idx(U)).at(gen_to_idx(D));
}

//===========================================================================
//---- Feynman Rules --------------------------------------------------------
//===========================================================================

template <class F> auto StandardModel::feynman_rule_f_f_h() -> VertexFFS {
  static constexpr auto gg = std::complex<double>{0.0, -F::mass / Higgs::vev};
  return VertexFFS{gg, gg};
}

template <class F> auto StandardModel::feynman_rule_f_f_z() -> VertexFFV {
  using tools::im;
  constexpr double qe = StandardModel::qe;
  constexpr double sw = StandardModel::sw;
  constexpr double cw = StandardModel::cw;
  constexpr double q = quantum_numbers<F>::charge();
  constexpr double t3 = quantum_numbers<F>::weak_iso_spin();

  double gzl = qe * (t3 - q * sw * sw) / (sw * cw);
  double gzr = -q * qe * sw / cw;

  return VertexFFV{gzl, gzr};
}

template <class Up, class Down>
auto StandardModel::feynman_rule_u_d_w() -> VertexFFV {
  static_assert(
      ((is_up_type_quark<Up>::value && is_down_type_quark<Down>::value) ||
       (is_neutrino<Up>::value && is_charged_lepton<Down>::value)),
      "Fields must be up-type and down-type quarks or neutrino and charged "
      "lepton.");

  constexpr double qe = StandardModel::qe;
  constexpr double sw = StandardModel::sw;
  constexpr std::complex<double> g = qe / (M_SQRT2 * sw);

  if constexpr (is_up_type_quark<Up>::value &&
                is_down_type_quark<Down>::value) {
    constexpr auto ckm = StandardModel::ckm<Up::gen, Down::gen>();
    return VertexFFV{ckm * g, 0.0};
  }
  if constexpr (is_neutrino<Up>::value && is_charged_lepton<Down>::value) {
    return VertexFFV{g, 0.0};
  }
  // Can't get here
}

template <class F> auto StandardModel::feynman_rule_f_f_a() -> VertexFFV {
  static_assert((is_fermion<F>::value), "Field must be a fermion.");
  constexpr double q = StandardModel::qe * quantum_numbers<F>::charge();
  return VertexFFV{q, q};
}

} // namespace blackthorn

#endif // BLACKTHORN_MODELS_STANDARD_MODEL_H
