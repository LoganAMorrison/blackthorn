#ifndef BLACKTHORN_MODELS_PARTICLES_H
#define BLACKTHORN_MODELS_PARTICLES_H

#include "blackthorn/Models/Base.h"

namespace blackthorn {

constexpr double MEV = 1e-3;

// ===========================================================================
// ---- Field Attributes: Charged Leptons ------------------------------------
// ===========================================================================

struct Electron { // NOLINT
  static constexpr int pdg = 11;
  static constexpr double mass = 0.5109989461e-3;
  static constexpr double width = 0.0;
  static constexpr Gen gen = Gen::Fst;
};
struct Muon { // NOLINT
  static constexpr int pdg = 13;
  static constexpr double mass = 105.6583745e-3;
  static constexpr double width = 0.0;
  static constexpr Gen gen = Gen::Snd;
};
struct Tau { // NOLINT
  static constexpr int pdg = 15;
  static constexpr double mass = 1.77686;
  static constexpr double width = 0.0;
  static constexpr Gen gen = Gen::Trd;
};

template <Gen G> struct ChargedLeptonType;

template <> struct ChargedLeptonType<Gen::Fst> { // NOLINT
  using type = Electron;
};
template <> struct ChargedLeptonType<Gen::Snd> { // NOLINT
  using type = Muon;
};
template <> struct ChargedLeptonType<Gen::Trd> { // NOLINT
  using type = Tau;
};

// ---- Classes ----

template <> struct is_charged_lepton<Electron> { // NOLINT
  static constexpr bool value = true;
};

template <> struct is_charged_lepton<Muon> { // NOLINT
  static constexpr bool value = true;
};

template <> struct is_charged_lepton<Tau> { // NOLINT
  static constexpr bool value = true;
};

// ---- Stability ----

template <> struct is_stable<Electron> { // NOLINT
  static constexpr bool value = true;
};

template <> struct is_stable<Muon> { // NOLINT
  static constexpr bool value = false;
};

template <> struct is_stable<Tau> { // NOLINT
  static constexpr bool value = false;
};

// ---- Attributes ----

template <> struct field_attrs<Electron> { // NOLINT
  static constexpr auto pdg() -> int { return Electron::pdg; };
  static constexpr auto mass() -> double { return Electron::mass; };
  static constexpr auto generation() -> Gen { return Electron::gen; };
};

template <> struct field_attrs<Muon> { // NOLINT
  static constexpr auto pdg() -> int { return Muon::pdg; };
  static constexpr auto mass() -> double { return Muon::mass; };
  static constexpr auto generation() -> Gen { return Muon::gen; };
};

template <> struct field_attrs<Tau> { // NOLINT
  static constexpr auto pdg() -> int { return Tau::pdg; };
  static constexpr auto mass() -> double { return Tau::mass; };
  static constexpr auto generation() -> Gen { return Tau::gen; };
};

// ===========================================================================
// ---- Field Attributes: Neutrinos ------------------------------------------
// ===========================================================================

struct ElectronNeutrino { // NOLINT
  static constexpr int pdg = 12;
  static constexpr double mass = 0.0;
  static constexpr double width = 0.0;
  static constexpr Gen gen = Gen::Fst;
};
struct MuonNeutrino { // NOLINT
  static constexpr int pdg = 14;
  static constexpr double mass = 0.0;
  static constexpr double width = 0.0;
  static constexpr Gen gen = Gen::Snd;
};
struct TauNeutrino { // NOLINT
  static constexpr int pdg = 16;
  static constexpr double mass = 0.0;
  static constexpr double width = 0.0;
  static constexpr Gen gen = Gen::Trd;
};

template <Gen G> struct NeutrinoType;

template <> struct NeutrinoType<Gen::Fst> { // NOLINT
  using type = ElectronNeutrino;
};
template <> struct NeutrinoType<Gen::Snd> { // NOLINT
  using type = MuonNeutrino;
};
template <> struct NeutrinoType<Gen::Trd> { // NOLINT
  using type = TauNeutrino;
};

template <> struct is_neutrino<ElectronNeutrino> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_neutrino<MuonNeutrino> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_neutrino<TauNeutrino> { // NOLINT
  static constexpr bool value = true;
};

template <> struct is_massless<ElectronNeutrino> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_massless<MuonNeutrino> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_massless<TauNeutrino> { // NOLINT
  static constexpr bool value = true;
};

template <> struct is_stable<ElectronNeutrino> { // NOLINT
  static constexpr bool value = true;
};

template <> struct is_stable<MuonNeutrino> { // NOLINT
  static constexpr bool value = true;
};

template <> struct is_stable<TauNeutrino> { // NOLINT
  static constexpr bool value = true;
};

template <> struct is_self_conj<ElectronNeutrino> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_self_conj<MuonNeutrino> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_self_conj<TauNeutrino> { // NOLINT
  static constexpr bool value = true;
};

template <> struct field_attrs<ElectronNeutrino> { // NOLINT
  static constexpr auto pdg() -> int { return ElectronNeutrino::pdg; };
  static constexpr auto generation() -> Gen { return ElectronNeutrino::gen; };
};
template <> struct field_attrs<MuonNeutrino> { // NOLINT
  static constexpr auto pdg() -> int { return MuonNeutrino::pdg; };
  static constexpr auto generation() -> Gen { return MuonNeutrino::gen; };
};
template <> struct field_attrs<TauNeutrino> { // NOLINT
  static constexpr auto pdg() -> int { return TauNeutrino::pdg; };
  static constexpr auto generation() -> Gen { return TauNeutrino::gen; };
};

// ===========================================================================
// ---- Field Attributes: Up-Type Quarks -------------------------------------
// ===========================================================================

struct UpQuark { // NOLINT
  static constexpr int pdg = 2;
  static constexpr double mass = 2.16e-3;
  static constexpr double width = 0.0;
  static constexpr Gen gen = Gen::Fst;
};
struct CharmQuark { // NOLINT
  static constexpr int pdg = 4;
  static constexpr double mass = 1.27;
  static constexpr double width = 0.0;
  static constexpr Gen gen = Gen::Snd;
};
struct TopQuark { // NOLINT
  static constexpr int pdg = 6;
  static constexpr double mass = 172.9;
  static constexpr double width = 1.42;
  static constexpr Gen gen = Gen::Trd;
};

template <> struct is_up_type_quark<UpQuark> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_up_type_quark<CharmQuark> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_up_type_quark<TopQuark> { // NOLINT
  static constexpr bool value = true;
};

template <> struct is_stable<UpQuark> { // NOLINT
  static constexpr bool value = true;
};

template <> struct is_stable<CharmQuark> { // NOLINT
  static constexpr bool value = false;
};

template <> struct is_stable<TopQuark> { // NOLINT
  static constexpr bool value = false;
};

template <> struct field_attrs<UpQuark> { // NOLINT
  static constexpr auto pdg() -> int { return UpQuark::pdg; };
  static constexpr auto mass() -> double { return UpQuark::mass; };
  static constexpr auto generation() -> Gen { return UpQuark::gen; };
};
template <> struct field_attrs<CharmQuark> { // NOLINT
  static constexpr auto pdg() -> int { return CharmQuark::pdg; };
  static constexpr auto mass() -> double { return CharmQuark::mass; };
  static constexpr auto generation() -> Gen { return CharmQuark::gen; };
};
template <> struct field_attrs<TopQuark> { // NOLINT
  static constexpr auto pdg() -> int { return TopQuark::pdg; };
  static constexpr auto mass() -> double { return TopQuark::mass; };
  static constexpr auto generation() -> Gen { return TopQuark::gen; };
};

// ===========================================================================
// ---- Down-Type Quarks -----------------------------------------------------
// ===========================================================================

struct DownQuark { // NOLINT
  static constexpr int pdg = 1;
  static constexpr double mass = 4.67e-3;
  static constexpr double width = 0.0;
  static constexpr Gen gen = Gen::Fst;
};
struct StrangeQuark { // NOLINT
  static constexpr int pdg = 1;
  static constexpr double mass = 95.0e-3;
  static constexpr double width = 0.0;
  static constexpr Gen gen = Gen::Fst;
};
struct BottomQuark { // NOLINT
  static constexpr int pdg = 5;
  static constexpr double mass = 4.18;
  static constexpr double width = 0.0;
  static constexpr Gen gen = Gen::Trd;
};

// ---- Classes ----

template <> struct is_down_type_quark<DownQuark> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_down_type_quark<StrangeQuark> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_down_type_quark<BottomQuark> { // NOLINT
  static constexpr bool value = true;
};

template <> struct is_stable<DownQuark> { // NOLINT
  static constexpr bool value = true;
};

template <> struct is_stable<StrangeQuark> { // NOLINT
  static constexpr bool value = true;
};

template <> struct is_stable<BottomQuark> { // NOLINT
  static constexpr bool value = false;
};

template <> struct field_attrs<DownQuark> { // NOLINT
  static constexpr auto pdg() -> int { return DownQuark::pdg; };
  static constexpr auto mass() -> double { return DownQuark::mass; };
  static constexpr auto generation() -> Gen { return DownQuark::gen; };
};
template <> struct field_attrs<StrangeQuark> { // NOLINT
  static constexpr auto pdg() -> int { return StrangeQuark::pdg; };
  static constexpr auto mass() -> double { return StrangeQuark::mass; };
  static constexpr auto generation() -> Gen { return StrangeQuark::gen; };
};
template <> struct field_attrs<BottomQuark> { // NOLINT
  static constexpr auto pdg() -> int { return BottomQuark::pdg; };
  static constexpr auto mass() -> double { return BottomQuark::mass; };
  static constexpr auto generation() -> Gen { return BottomQuark::gen; };
};

// ===========================================================================
// ---- Bosons ---------------------------------------------------------------
// ===========================================================================

struct Gluon { // NOLINT
  static constexpr int pdg = 21;
  static constexpr double mass = 0.0;
  static constexpr double width = 0.0;
};
struct Photon { // NOLINT
  static constexpr int pdg = 22;
  static constexpr double mass = 0.0;
  static constexpr double width = 0.0;
};
struct ZBoson { // NOLINT
  static constexpr int pdg = 23;
  static constexpr double mass = 91.18760;
  static constexpr double width = 2.49520;
};
struct WBoson { // NOLINT
  static constexpr int pdg = 24;
  static constexpr double mass = 80.385003;
  static constexpr double width = 2.08500;
};
struct Higgs { // NOLINT
  static constexpr int pdg = 25;
  static constexpr double mass = 125.00;
  static constexpr double width = 0.00374;
  static constexpr bool self_conj = true;
  static constexpr double vev = 246.21965;
};

// ---- Classes ----

template <> struct is_vector_boson<Gluon> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_vector_boson<Photon> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_vector_boson<ZBoson> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_vector_boson<WBoson> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_scalar_boson<Higgs> { // NOLINT
  static constexpr bool value = true;
};

// ---- Massless ----

template <> struct is_massless<Gluon> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_massless<Photon> { // NOLINT
  static constexpr bool value = true;
};

// ---- Self Conjugate ----

template <> struct is_self_conj<Gluon> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_self_conj<Photon> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_self_conj<ZBoson> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_self_conj<Higgs> { // NOLINT
  static constexpr bool value = true;
};

// ---- Stability ----

template <> struct is_stable<Gluon> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_stable<Photon> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_stable<ZBoson> { // NOLINT
  static constexpr bool value = false;
};
template <> struct is_stable<WBoson> { // NOLINT
  static constexpr bool value = false;
};
template <> struct is_stable<Higgs> { // NOLINT
  static constexpr bool value = false;
};

// ---- Attributes ----

template <> struct field_attrs<Gluon> { // NOLINT
  static constexpr auto pdg() -> int { return Gluon::pdg; };
};
template <> struct field_attrs<Photon> { // NOLINT
  static constexpr auto pdg() -> int { return Photon::pdg; };
};
template <> struct field_attrs<ZBoson> { // NOLINT
  static constexpr auto pdg() -> int { return ZBoson::pdg; };
  static constexpr auto mass() -> double { return ZBoson::mass; };
  static constexpr auto width() -> double { return ZBoson::width; };
};
template <> struct field_attrs<WBoson> { // NOLINT
  static constexpr auto pdg() -> int { return WBoson::pdg; };
  static constexpr auto mass() -> double { return WBoson::mass; };
  static constexpr auto width() -> double { return WBoson::width; };
};
template <> struct field_attrs<Higgs> { // NOLINT
  static constexpr auto pdg() -> int { return Higgs::pdg; };
  static constexpr auto mass() -> double { return Higgs::mass; };
  static constexpr auto width() -> double { return Higgs::width; };
  static constexpr auto vev() -> double { return Higgs::vev; };
};

// ---- Quantum Numbers ----

template <> struct quantum_numbers<WBoson> { // NOLINT
  static constexpr auto charge() -> double { return 1.0; }
};

//============================================================================
//---- Neutral Mesons --------------------------------------------------------
//============================================================================

struct NeutralPion { // NOLINT
  static constexpr double decay_const = 0.091924;
  static constexpr double mass = 0.1349768;
  static constexpr int pdg = 111;

  /// BR(??, ??) = (98.823??0.034) %
  static constexpr double BR_PI0_TO_A_A = 98.823e-2;
  /// BR(e+, e???, ??) = (1.174??0.035) %
  static constexpr double BR_PI0_TO_E_E_A = 1.174e-2;
  /// BR(e+, e+, e???, e???) = (3.34??0.16 )??10???5
  static constexpr double BR_PI0_TO_E_E_E_E = 3.34e-5;
};

struct Eta { // NOLINT
  static constexpr double mass = 0.547862;
  static constexpr int pdg = 211;

  /// BR(?? -> ??, ??) = (39.41??0.20) %
  static constexpr double BR_ETA_TO_A_A = 39.41e-2;
  /// BR(?? -> ??0, ??0, ??0) = (32.68??0.23) %
  static constexpr double BR_ETA_TO_PI0_PI0_PI0 = 32.68e-2;
  /// BR(?? -> ??+, ?????, ??0) = (22.92??0.28) %
  static constexpr double BR_ETA_TO_PI_PI_PI0 = 22.92e-2;
  /// BR(?? -> ??+, ?????, ??) = ( 4.22??0.08) %
  static constexpr double BR_ETA_TO_PI_PI_A = 4.22e-2;
  /// BR(?? -> ??0, ??, ??) = ( 2.56??0.22)??10???4
  static constexpr double BR_ETA_TO_PI0_A_A = 2.56e-4;

  /// BR(?? -> e+, e???, ??) = ( 6.9??0.4 )??10???3
  static constexpr double BR_ETA_TO_E_E_A = 6.9e-3;
  /// BR(?? -> ??+, ?????, ??) = ( 3.1??0.4 )??10???4
  static constexpr double BR_ETA_TO_MU_MU_A = 3.1e-4;
  /// BR(?? -> ??+, ?????, e+, e???)  = ( 2.68??0.11)??10???4
  static constexpr double BR_ETA_TO_PI_PI_E_E = 2.68e-4;
  /// BR(?? -> e+, e???, e+, e???) = ( 2.40??0.22)??10???5
  static constexpr double BR_ETA_TO_E_E_E_E = 2.40e-5;
  /// BR(?? -> ??+, ?????) = ( 5.8??0.8 )??10???6
  static constexpr double BR_ETA_TO_MU_MU = 5.8e-6;
};

struct NeutralKaon { // NOLINT
  static constexpr double mass = 0.497611;
  static constexpr int pdg = 311;
};

struct ShortKaon { // NOLINT
  static constexpr double mass = NeutralKaon::mass;
  static constexpr int pdg = 310;

  /// BR(??+, ?????) = (69.20??0.05) %
  static constexpr double BR_KS_TO_PI_PI = 69.20e-2;
  /// BR(??0, ??0) = (30.69??0.05) %
  static constexpr double BR_KS_TO_PI0_PI0 = 30.69e-2;

  /// BR(??+, ?????, ??) = ( 1.79??0.05)??10???3
  static constexpr double BR_KS_TO_PI_PI_A = 1.79e-3;
  /// BR(????, e???, ??e) =  ( 7.04??0.08)??10???4
  static constexpr double BR_KS_TO_PI_E_NUE = 7.04e-4;
  /// BR(??+, ?????, e+, e???) = ( 4.79??0.15)??10???5
  static constexpr double BR_KS_TO_PI_PI_E_E = 4.79e-5;
  /// BR(??, ??) = ( 2.63??0.17)??10???6
  static constexpr double BR_KS_TO_A_A = 2.63e-6;
  /// BR(??+, ?????, ??0) = ( 3.5+1.1???0.9)??10???7
  static constexpr double BR_KS_TO_PI_PI_PI0 = 3.5e-7;
  /// BR(??0, ??, ??) =  ( 4.9??1.8 )??10???8
  static constexpr double BR_KS_TO_PI0_A_A = 4.9e-8;
  /// BR(??0, e+, e???) = ( 3.0+1.5???1.2)??10???9
  static constexpr double BR_KS_TO_PI0_E_E = 3e-9;
  /// BR(??0, ??+, ?????) = ( 2.9+1.5???1.2)??10???9
  static constexpr double BR_KS_TO_PI0_MU_MU = 2.9e-9;
};

struct LongKaon { // NOLINT
  static constexpr double mass = NeutralKaon::mass;
  static constexpr int pdg = 130;

  /// BR(????, e???, ??e) = (40.55??0.11 ) %
  static constexpr double BR_KL_TO_PI_E_NUE = 40.55e-2;
  /// BR(????, ?????, ????) =  (27.04??0.07 ) %
  static constexpr double BR_KL_TO_PI_MU_NUMU = 27.04e-2;
  /// BR(??0, ??0, ??0) = (19.52??0.12 ) %
  static constexpr double BR_KL_TO_PI0_PI0_PI0 = 19.52e-2;
  /// BR(??+, ?????, ??0) = (12.54??0.05 ) %
  static constexpr double BR_KL_TO_PI_PI_PI0 = 12.54e-2;

  /// BR(??+, ?????) = ( 1.967??0.010)??10???3
  static constexpr double BR_KL_TO_PI_PI = 1.967e-3;
  /// BR(??0, ??0) = ( 8.64??0.06 )??10???4
  static constexpr double BR_KL_TO_PI0_PI0 = 8.64e-4;
  /// BR(??, ??) = ( 5.47??0.04 )??10???4
  static constexpr double BR_KL_TO_A_A = 5.47e-4;
  /// BR(??0, ????, e???, ??) = ( 5.20??0.11 )??10???5
  static constexpr double BR_KL_TO_PI0_PI_E_NU = 5.20e-5;
  /// BR(????, e???, ??, e+, e???) = ( 1.26??0.04 )??10???5
  static constexpr double BR_KL_TO_PI_E_E_E_NU = 1.26e-5;
};

// ---- Classes ----

template <> struct is_scalar_boson<NeutralPion> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_scalar_boson<Eta> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_scalar_boson<NeutralKaon> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_scalar_boson<LongKaon> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_scalar_boson<ShortKaon> { // NOLINT
  static constexpr bool value = true;
};

// ---- Self Conjugate ----

template <> struct is_self_conj<NeutralPion> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_self_conj<Eta> { // NOLINT
  static constexpr bool value = true;
};

// ---- Stability ----

template <> struct is_stable<NeutralPion> { // NOLINT
  static constexpr bool value = false;
};
template <> struct is_stable<Eta> { // NOLINT
  static constexpr bool value = false;
};
template <> struct is_stable<NeutralKaon> { // NOLINT
  static constexpr bool value = false;
};
template <> struct is_stable<LongKaon> { // NOLINT
  static constexpr bool value = false;
};
template <> struct is_stable<ShortKaon> { // NOLINT
  static constexpr bool value = false;
};

// ---- Attributes ----

template <> struct field_attrs<NeutralPion> { // NOLINT
  static constexpr auto pdg() -> int { return NeutralPion::pdg; };
  static constexpr auto mass() -> double { return NeutralPion::mass; };
};
template <> struct field_attrs<Eta> { // NOLINT
  static constexpr auto pdg() -> int { return Eta::pdg; };
  static constexpr auto mass() -> double { return Eta::mass; };
};
template <> struct field_attrs<NeutralKaon> { // NOLINT
  static constexpr auto pdg() -> int { return NeutralKaon::pdg; };
  static constexpr auto mass() -> double { return NeutralKaon::mass; };
};
template <> struct field_attrs<LongKaon> { // NOLINT
  static constexpr auto pdg() -> int { return LongKaon::pdg; };
  static constexpr auto mass() -> double { return LongKaon::mass; };
};
template <> struct field_attrs<ShortKaon> { // NOLINT
  static constexpr auto pdg() -> int { return ShortKaon::pdg; };
  static constexpr auto mass() -> double { return ShortKaon::mass; };
};

//============================================================================
//---- Charged Mesons --------------------------------------------------------
//============================================================================

struct ChargedPion { // NOLINT
  static constexpr double mass = 0.13957039;
  static constexpr double decay_const = 0.092214;
  static constexpr int pdg = 211;

  /// ????? -> ?????? + ????? + ?? vector form factor
  static constexpr double ff_vec = 0.0259;
  /// ????? -> ?????? + ????? + ?? vector form factor slope: V??(x) = V??(x) * (1 + slope*x)
  static constexpr double ff_vec_slope = 0.095;
  /// ????? -> ?????? + ????? + ?? axial-vector form factor
  static constexpr double ff_axi = 0.0119;
  /// ????? -> ?????? + ????? + ?? epsilon factor
  static constexpr double ff_eps = 1.0;

  /// BR(????? -> ????? + ????) = (99.98770??0.00004) %
  static constexpr double BR_PI_TO_MU_NUMU = 0.9998770;
  /// BR(????? -> e??? + ??e) = ( 1.230??0.004  )??10???4
  static constexpr double BR_PI_TO_E_NUE = 1.230e-4;
};

struct ChargedKaon { // NOLINT
  static constexpr double mass = 0.493677;
  static constexpr double decay_const = 0.1104;
  static constexpr int pdg = 321;
  /// K??? -> ?????? + ????? + ?? vector form factor
  static constexpr double ff_vec = 0.096;
  /// K??? -> ?????? + ????? + ?? vector form factor slope: VK(x) = VK(x) * (1 + slope*x)
  static constexpr double ff_vec_slope = 0.0;
  /// K??? -> ?????? + ????? + ?? axial-vector form factor
  static constexpr double ff_axi = 0.042;
  /// K??? -> ?????? + ????? + ?? epsilon factor
  static constexpr double ff_eps = -1.0;

  /// BR(K -> ??+, ????) = (63.56 ?? 0.11) %
  static constexpr double BR_K_TO_MU_NUMU = 63.56e-2;
  /// BR(K -> ??+, ??0) = (20.67 ?? 0.08 ) %
  static constexpr double BR_K_TO_PI_PI0 = 20.67e-2;
  /// BR(K -> ??+, ??+, ?????) = (5.583 ?? 0.024) %
  static constexpr double BR_K_TO_PI_PI_PI = 5.583e-2;
  // BR(K -> ??0, e+, ??e) = (5.07 ?? 0.04) %
  static constexpr double BR_K_TO_E_NUE_PI0 = 5.07e-2;
  // BR(K -> ??0, ??+, ????)   (3.352 ?? 0.033) %
  static constexpr double BR_K_TO_MU_NUMU_PI0 = 3.352e-2;
  // BR(K -> ??+, ??0, ??0) = (1.760 ?? 0.023) %
  static constexpr double BR_K_TO_PI_PI0_PI0 = 1.760e-2;

  // BR(K -> e+, ??e) = (1.582 ?? 0.007)??10???5
  static constexpr double BR_K_TO_E_NUE = 1.582e-5;
  // BR(K -> ??0, ??0, e+, ??e) = (2.55 ?? 0.04)??10???5
  static constexpr double BR_K_TO_E_NUE_PI0_PI0 = 2.55e-5;
  // BR(K -> ??+, ?????, e+, ??e) =  (4.247 ?? 0.024)??10???5
  static constexpr double BR_K_TO_E_NUE_PI_PI = 4.247e-5;
  // Taken from Pythia8306 (can't find in PDG)
  static constexpr double BR_K_TO_MU_NUMU_PI0_PI0 = 0.0000140;
  // BR(K -> ??+, ?????, ??+, ????) =  (1.4 ?? 0.9)??10???5
  static constexpr double BR_K_TO_MU_NUMU_PI_PI = 1.4e-5;
  // BR(K -> e+, ??e, e+, e???) =  (2.48 ?? 0.20 )??10???8
  static constexpr double BR_K_TO_E_E_E_NUE = 2.48e-8;
  // BR(K -> ??+, ????, e+, e???) =  (7.06 ?? 0.31 )??10???8
  static constexpr double BR_K_TO_MU_E_E_NUMU = 7.06e-8;
  // BR(K -> e+, ??e, ??+, ?????) =  (1.7 ?? 0.5  )??10???8
  static constexpr double BR_K_TO_MU_MU_E_NUE = 1.7e-8;
  // BR(K -> ??+, e+, e???) = (3.00 ?? 0.09 )??10???7
  static constexpr double BR_K_TO_PI_E_E = 3.00e-7;
  // BR(K -> ??+, ??+, ?????) = (9.4 ?? 0.6  )??10???8
  static constexpr double BR_K_TO_PI_MU_MU = 9.4e-8;
};

// ---- Classes ----

template <> struct is_scalar_boson<ChargedPion> { // NOLINT
  static constexpr bool value = true;
};
template <> struct is_scalar_boson<ChargedKaon> { // NOLINT
  static constexpr bool value = true;
};

// ---- Attributes ----

template <> struct field_attrs<ChargedPion> { // NOLINT
  static constexpr auto pdg() -> int { return ChargedPion::pdg; };
  static constexpr auto mass() -> double { return ChargedPion::mass; };
};
template <> struct field_attrs<ChargedKaon> { // NOLINT
  static constexpr auto pdg() -> int { return ChargedKaon::pdg; };
  static constexpr auto mass() -> double { return ChargedKaon::mass; };
};

// ---- Quantum Numbers ----

template <> struct quantum_numbers<ChargedPion> { // NOLINT
  static constexpr auto charge() -> double { return 1.0; }
};
template <> struct quantum_numbers<ChargedKaon> { // NOLINT
  static constexpr auto charge() -> double { return 1.0; }
};

// ---- Stability ----

template <> struct is_stable<ChargedPion> { // NOLINT
  static constexpr bool value = false;
};
template <> struct is_stable<ChargedKaon> { // NOLINT
  static constexpr bool value = false;
};

} // namespace blackthorn

#endif // BLACKTHORN_MODELS_PARTICLES_H
