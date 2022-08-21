#include "blackthorn/Spectra.h"
#include "blackthorn/Models/RhNeutrino.h"
#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/Tools.h"
#include <type_traits>

using pyvec = pybind11::array_t<double>;
using stdvec = std::vector<double>;
using nuvec = std::array<std::vector<double>, 3>;

namespace blackthorn {

using Rhn = blackthorn::RhNeutrinoMeV;

static auto gen_to_int(Gen gen) -> int {
  if (gen == Gen::Fst) {
    return 1;
  }
  if (gen == Gen::Snd) {
    return 2;
  }
  if (gen == Gen::Trd) {
    return 3;
  }
  return 0;
}

// =========================================================================
// ---- Templated Photon Helper Functions ----------------------------------
// =========================================================================

// Helper class to compute the photon spectrum given a 2-body final state.
template <class P1, class P2, class X>
auto dndx_photon(const X &x, double beta, double mass) -> X { // NOLINT
  DecaySpectrum<P1, P2> dspec(mass);
  return dspec.dndx_photon(x, beta);
}

// Helper class to compute the photon spectrum from a 3-body final state.
template <class P1, class P2, class P3, class X, class Msqrd>
auto dndx_photon(const X &x, double beta, double mass, Msqrd msqrd) // NOLINT
    -> X {
  DecaySpectrum<P1, P2, P3> dspec(mass, msqrd);
  return dspec.dndx_photon(x, beta);
}

// Helper class to compute the spectrum for N -> ℓ + P
template <class P, class X>
auto dndx_photon_lepton(const X &x, double beta, double mass, Gen gen)
    -> X { // NOLINT
  switch (gen) {
  case Gen::Fst:
    return dndx_photon<Electron, P>(x, beta, mass);
  case Gen::Snd:
    return dndx_photon<Muon, P>(x, beta, mass);
  default:
    return dndx_photon<Tau, P>(x, beta, mass);
  }
}

// Helper class to compute the spectrum for N -> ν + P
template <class P, class X>
auto dndx_photon_neutrino(const X &x, double beta, double mass, Gen gen)
    -> X { // NOLINT
  switch (gen) {
  case Gen::Fst:
    return dndx_photon<ElectronNeutrino, P>(x, beta, mass);
  case Gen::Snd:
    return dndx_photon<MuonNeutrino, P>(x, beta, mass);
  default:
    return dndx_photon<TauNeutrino, P>(x, beta, mass);
  }
}

// =========================================================================
// ---- Templated Neutrino Helper Functions --------------------------------
// =========================================================================

template <class X> class NeutrinoSpecRetType {
public:
  using type = std::remove_reference_t<std::remove_const_t<X>>;
  static auto zeros_like(const X &xs) -> X { return tools::zeros_like(xs); }
};
template <> class NeutrinoSpecRetType<double> {
public:
  using type = std::array<double, 3>;
  static auto zeros_like(double /*x*/) -> type { return {0.0, 0.0, 0.0}; }
};

template <> class NeutrinoSpecRetType<std::vector<double>> {
public:
  using type = std::array<std::vector<double>, 3>;
  static auto zeros_like(const std::vector<double> &xs) -> type {
    return {tools::zeros_like(xs), tools::zeros_like(xs),
            tools::zeros_like(xs)};
  }
};

template <class P1, class P2, class X>
auto dndx_neutrino(const X &x, double beta, double mass) -> // NOLINT
    typename NeutrinoSpecRetType<X>::type {
  DecaySpectrum<P1, P2> dspec(mass);
  return dspec.dndx_neutrino(x, beta);
}

template <class P1, class P2, class P3, class X, class Msqrd>
auto dndx_neutrino(const X &x, double beta, double mass, Msqrd msqrd) // NOLINT
    -> typename NeutrinoSpecRetType<X>::type {
  DecaySpectrum<P1, P2, P3> dspec(mass, msqrd);
  return dspec.dndx_neutrino(x, beta);
}

template <class P1, class P2, class P3, class Msqrd>
auto dndx_neutrino(const std::vector<double> &x, double beta, // NOLINT
                   double mass,                               // NOLINT
                   Msqrd msqrd)                               // NOLINT
    -> std::array<std::vector<double>, 3> {
  DecaySpectrum<P1, P2, P3> dspec(mass, msqrd);
  return dspec.dndx_neutrino(x, beta);
}

template <class P, class X>
auto dndx_neutrino_lepton(const X &x, double beta, double mass, Gen G) ->
    typename NeutrinoSpecRetType<X>::type { // NOLINT
  switch (G) {
  case Gen::Fst: {
    return dndx_neutrino<Electron, P>(x, beta, mass);
  }
  case Gen::Snd: {
    return dndx_neutrino<Muon, P>(x, beta, mass);
  }
  default:
    return dndx_neutrino<Tau, P>(x, beta, mass);
  }
}

template <class P, class X>
auto dndx_neutrino_neutrino(const X &x, double beta, double mass, Gen G) ->
    typename NeutrinoSpecRetType<X>::type { // NOLINT
  switch (G) {
  case Gen::Fst: {
    return dndx_neutrino<ElectronNeutrino, P>(x, beta, mass);
  }
  case Gen::Snd: {
    return dndx_neutrino<MuonNeutrino, P>(x, beta, mass);
  }
  default:
    return dndx_neutrino<TauNeutrino, P>(x, beta, mass);
  }
}

// =========================================================================
// ---- Templated Positron Helper Functions --------------------------------
// =========================================================================

template <class P1, class P2, class X>
auto dndx_positron(const X &x, double beta, double mass) -> X { // NOLINT
  DecaySpectrum<P1, P2> dspec(mass);
  return dspec.dndx_positron(x, beta);
}

template <class P1, class P2, class P3, class X, class Msqrd>
auto dndx_positron(const X &x, double beta, double mass, Msqrd msqrd) // NOLINT
    -> X {
  DecaySpectrum<P1, P2, P3> dspec(mass, msqrd);
  return dspec.dndx_positron(x, beta);
}

template <class P, class X>
auto dndx_positron_lepton(const X &x, double beta, double mass, Gen G)
    -> X { // NOLINT
  switch (G) {
  case Gen::Fst:
    return dndx_positron<Electron, P>(x, beta, mass);
  case Gen::Snd:
    return dndx_positron<Muon, P>(x, beta, mass);
  default:
    return dndx_positron<Tau, P>(x, beta, mass);
  }
}

template <class P, class X>
auto dndx_positron_neutrino(const X &x, double beta, double mass, Gen G)
    -> X { // NOLINT
  switch (G) {
  case Gen::Fst:
    return dndx_positron<ElectronNeutrino, P>(x, beta, mass);
  case Gen::Snd:
    return dndx_positron<MuonNeutrino, P>(x, beta, mass);
  default:
    return dndx_positron<TauNeutrino, P>(x, beta, mass);
  }
}

// =========================================================================
// ---- Photon Spectra Functions -------------------------------------------
// =========================================================================

auto Rhn::dndx_photon_l_pi(double x, double beta) const -> double {
  return dndx_photon_lepton<ChargedPion>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_photon_l_pi(const pyvec &x, double beta) const -> pyvec {
  return dndx_photon_lepton<ChargedPion>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_photon_l_pi(const stdvec &x, double beta) const -> stdvec {
  return dndx_photon_lepton<ChargedPion>(x, beta, p_mass, p_gen);
}

auto Rhn::dndx_photon_l_k(double x, double beta) const -> double {
  return dndx_photon_lepton<ChargedKaon>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_photon_l_k(const pyvec &x, double beta) const -> pyvec {
  return dndx_photon_lepton<ChargedKaon>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_photon_l_k(const stdvec &x, double beta) const -> stdvec {
  return dndx_photon_lepton<ChargedKaon>(x, beta, p_mass, p_gen);
}

auto Rhn::dndx_photon_v_pi0(double x, double beta) const -> double {
  return dndx_photon_neutrino<NeutralPion>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_photon_v_pi0(const pyvec &x, double beta) const -> pyvec {
  return dndx_photon_neutrino<NeutralPion>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_photon_v_pi0(const stdvec &x, double beta) const -> stdvec {
  return dndx_photon_neutrino<NeutralPion>(x, beta, p_mass, p_gen);
}

auto Rhn::dndx_photon_v_eta(double x, double beta) const -> double {
  return dndx_photon_neutrino<NeutralPion>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_photon_v_eta(const pyvec &x, double beta) const -> pyvec {
  return dndx_photon_neutrino<NeutralPion>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_photon_v_eta(const stdvec &x, double beta) const -> stdvec {
  return dndx_photon_neutrino<NeutralPion>(x, beta, p_mass, p_gen);
}

template <class X>
auto p_dndx_photon_v_l_l(const X &xs, double beta, Gen genv, Gen genl1,
                         Gen genl2, const Rhn &model) -> X {
  const auto msqrd = SquaredAmplitudeNToVLL(model, genv, genl1, genl2);
  const double mass = model.mass();
  using EV = ElectronNeutrino;
  using MV = MuonNeutrino;
  using TV = TauNeutrino;
  using E = Electron;
  using M = Muon;
  using T = Tau;

  if (genv == Gen::Null || genl1 == Gen::Null || genl2 == Gen::Null) {
    return tools::zeros_like(xs);
  }

  const int a = (gen_to_int(genv) - 1);
  const int b = (gen_to_int(genl1) - 1);
  const int c = (gen_to_int(genl2) - 1);
  const int i = 9 * a + 3 * b + c;

  switch (i) {
  // 0-8: N -> νe + ℓ + ℓ
  case 0:
    return dndx_photon<EV, E, E>(xs, beta, mass, msqrd);
  case 1:
    return dndx_photon<EV, E, M>(xs, beta, mass, msqrd);
  case 2:
    return dndx_photon<EV, E, T>(xs, beta, mass, msqrd);
  case 3:
    return dndx_photon<EV, M, E>(xs, beta, mass, msqrd);
  case 4:
    return dndx_photon<EV, M, M>(xs, beta, mass, msqrd);
  case 5:
    return dndx_photon<EV, M, T>(xs, beta, mass, msqrd);
  case 6:
    return dndx_photon<EV, T, E>(xs, beta, mass, msqrd);
  case 7:
    return dndx_photon<EV, T, M>(xs, beta, mass, msqrd);
  case 8:
    return dndx_photon<EV, T, T>(xs, beta, mass, msqrd);
  // 9-17: N -> νμ + ℓ + ℓ
  case 9:
    return dndx_photon<MV, E, E>(xs, beta, mass, msqrd);
  case 10:
    return dndx_photon<MV, E, M>(xs, beta, mass, msqrd);
  case 11:
    return dndx_photon<MV, E, T>(xs, beta, mass, msqrd);
  case 12:
    return dndx_photon<MV, M, E>(xs, beta, mass, msqrd);
  case 13:
    return dndx_photon<MV, M, M>(xs, beta, mass, msqrd);
  case 14:
    return dndx_photon<MV, M, T>(xs, beta, mass, msqrd);
  case 15:
    return dndx_photon<MV, T, E>(xs, beta, mass, msqrd);
  case 16:
    return dndx_photon<MV, T, M>(xs, beta, mass, msqrd);
  case 17:
    return dndx_photon<MV, T, T>(xs, beta, mass, msqrd);
  // 18-26: N -> ντ + ℓ + ℓ
  case 18:
    return dndx_photon<TV, E, E>(xs, beta, mass, msqrd);
  case 19:
    return dndx_photon<TV, E, M>(xs, beta, mass, msqrd);
  case 20:
    return dndx_photon<TV, E, T>(xs, beta, mass, msqrd);
  case 21:
    return dndx_photon<TV, M, E>(xs, beta, mass, msqrd);
  case 22:
    return dndx_photon<TV, M, M>(xs, beta, mass, msqrd);
  case 23:
    return dndx_photon<TV, M, T>(xs, beta, mass, msqrd);
  case 24:
    return dndx_photon<TV, T, E>(xs, beta, mass, msqrd);
  case 25:
    return dndx_photon<TV, T, M>(xs, beta, mass, msqrd);
  case 26:
    return dndx_photon<TV, T, T>(xs, beta, mass, msqrd);
  default:
    return tools::zeros_like(xs);
  }
}
auto Rhn::dndx_photon_v_l_l(double x, double beta, Gen gv, Gen gl1,
                            Gen gl2) const -> double {
  return p_dndx_photon_v_l_l(x, beta, gv, gl1, gl2, *this);
}
auto Rhn::dndx_photon_v_l_l(const pyvec &x, double beta, Gen gv, Gen gl1,
                            Gen gl2) const -> pyvec {
  return p_dndx_photon_v_l_l(x, beta, gv, gl1, gl2, *this);
}
auto Rhn::dndx_photon_v_l_l(const stdvec &x, double beta, Gen gv, Gen gl1,
                            Gen gl2) const -> stdvec {
  return p_dndx_photon_v_l_l(x, beta, gv, gl1, gl2, *this);
}

template <class X>
auto p_dndx_photon_l_pi_pi0(const X &xs, double beta, const Rhn &model) -> X {
  const auto msqrd = SquaredAmplitudeNToLPiPi0(model);
  const double mass = model.mass();
  using CP = ChargedPion;
  using NP = NeutralPion;

  if (model.gen() == Gen::Fst) {
    return dndx_photon<Electron, CP, NP>(xs, beta, mass, msqrd);
  }
  if (model.gen() == Gen::Snd) {
    return dndx_photon<Muon, CP, NP>(xs, beta, mass, msqrd);
  }
  if (model.gen() == Gen::Trd) {
    return dndx_photon<Tau, CP, NP>(xs, beta, mass, msqrd);
  }

  return tools::zeros_like(xs);
}
auto Rhn::dndx_photon_l_pi_pi0(double x, double beta) const -> double {
  return p_dndx_photon_l_pi_pi0(x, beta, *this);
}
auto Rhn::dndx_photon_l_pi_pi0(const pyvec &x, double beta) const -> pyvec {
  return p_dndx_photon_l_pi_pi0(x, beta, *this);
}
auto Rhn::dndx_photon_l_pi_pi0(const stdvec &x, double beta) const -> stdvec {
  return p_dndx_photon_l_pi_pi0(x, beta, *this);
}

template <class X>
auto p_dndx_photon_v_pi_pi(const X &xs, double beta, const Rhn &model) -> X {
  const auto msqrd = SquaredAmplitudeNToLPiPi0(model);
  const double mass = model.mass();
  using P = ChargedPion;

  if (model.gen() == Gen::Fst) {
    using V = ElectronNeutrino;
    return dndx_photon<V, P, P>(xs, beta, mass, msqrd);
  }
  if (model.gen() == Gen::Snd) {
    using V = MuonNeutrino;
    return dndx_photon<V, P, P>(xs, beta, mass, msqrd);
  }
  if (model.gen() == Gen::Trd) {
    using V = TauNeutrino;
    return dndx_photon<V, P, P>(xs, beta, mass, msqrd);
  }

  return tools::zeros_like(xs);
}
auto Rhn::dndx_photon_v_pi_pi(double x, double beta) const -> double {
  return p_dndx_photon_v_pi_pi(x, beta, *this);
}
auto Rhn::dndx_photon_v_pi_pi(const pyvec &x, double beta) const -> pyvec {
  return p_dndx_photon_v_pi_pi(x, beta, *this);
}
auto Rhn::dndx_photon_v_pi_pi(const stdvec &x, double beta) const -> stdvec {
  return p_dndx_photon_v_pi_pi(x, beta, *this);
}

// =========================================================================
// ---- Neutrino Spectra Functions -----------------------------------------
// =========================================================================

auto Rhn::dndx_neutrino_l_pi(double x, double beta) const -> NeutrinoPt {
  return dndx_neutrino_neutrino<ChargedPion>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_neutrino_l_pi(const pyvec &x, double beta) const -> pyvec {
  return dndx_neutrino_neutrino<ChargedPion>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_neutrino_l_pi(const stdvec &x, double beta) const -> nuvec {
  return dndx_neutrino_neutrino<ChargedPion>(x, beta, p_mass, p_gen);
}

auto Rhn::dndx_neutrino_l_k(double x, double beta) const -> NeutrinoPt {
  return dndx_neutrino_neutrino<ChargedKaon>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_neutrino_l_k(const pyvec &x, double beta) const -> pyvec {
  return dndx_neutrino_neutrino<ChargedKaon>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_neutrino_l_k(const stdvec &x, double beta) const -> nuvec {
  return dndx_neutrino_neutrino<ChargedKaon>(x, beta, p_mass, p_gen);
}

auto Rhn::dndx_neutrino_v_pi0(double x, double beta) const -> NeutrinoPt {
  return dndx_neutrino_neutrino<NeutralPion>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_neutrino_v_pi0(const pyvec &x, double beta) const -> pyvec {
  return dndx_neutrino_neutrino<NeutralPion>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_neutrino_v_pi0(const stdvec &x, double beta) const -> nuvec {
  return dndx_neutrino_neutrino<NeutralPion>(x, beta, p_mass, p_gen);
}

auto Rhn::dndx_neutrino_v_eta(double x, double beta) const -> NeutrinoPt {
  return dndx_neutrino_neutrino<Eta>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_neutrino_v_eta(const pyvec &x, double beta) const -> pyvec {
  return dndx_neutrino_neutrino<Eta>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_neutrino_v_eta(const stdvec &x, double beta) const -> nuvec {
  return dndx_neutrino_neutrino<Eta>(x, beta, p_mass, p_gen);
}

template <class X>
auto p_dndx_neutrino_v_l_l(const X &xs, double beta, Gen genv, Gen genl1,
                           Gen genl2, const Rhn &model) ->
    typename NeutrinoSpecRetType<X>::type {
  const auto msqrd = SquaredAmplitudeNToVLL(model, genv, genl1, genl2);
  const double mass = model.mass();
  using EV = ElectronNeutrino;
  using MV = MuonNeutrino;
  using TV = TauNeutrino;
  using E = Electron;
  using M = Muon;
  using T = Tau;

  std::cout << "Here!"
            << "\n";

  if (genv == Gen::Null || genl1 == Gen::Null || genl2 == Gen::Null) {
    return NeutrinoSpecRetType<X>::zeros_like(xs);
  }

  const int a = (gen_to_int(genv) - 1);
  const int b = (gen_to_int(genl1) - 1);
  const int c = (gen_to_int(genl2) - 1);
  const int i = 9 * a + 3 * b + c;

  switch (i) {
    // 0-8: N -> νe + ℓ + ℓ
  case 0:
    return dndx_neutrino<EV, E, E>(xs, beta, mass, msqrd);
  case 1:
    return dndx_neutrino<EV, E, M>(xs, beta, mass, msqrd);
  case 2:
    return dndx_neutrino<EV, E, T>(xs, beta, mass, msqrd);
  case 3:
    return dndx_neutrino<EV, M, E>(xs, beta, mass, msqrd);
  case 4:
    return dndx_neutrino<EV, M, M>(xs, beta, mass, msqrd);
  case 5:
    return dndx_neutrino<EV, M, T>(xs, beta, mass, msqrd);
  case 6:
    return dndx_neutrino<EV, T, E>(xs, beta, mass, msqrd);
  case 7:
    return dndx_neutrino<EV, T, M>(xs, beta, mass, msqrd);
  case 8:
    return dndx_neutrino<EV, T, T>(xs, beta, mass, msqrd);
    // 9-17: N -> νμ + ℓ + ℓ
  case 9:
    return dndx_neutrino<MV, E, E>(xs, beta, mass, msqrd);
  case 10:
    return dndx_neutrino<MV, E, M>(xs, beta, mass, msqrd);
  case 11:
    return dndx_neutrino<MV, E, T>(xs, beta, mass, msqrd);
  case 12:
    return dndx_neutrino<MV, M, E>(xs, beta, mass, msqrd);
  case 13:
    return dndx_neutrino<MV, M, M>(xs, beta, mass, msqrd);
  case 14:
    return dndx_neutrino<MV, M, T>(xs, beta, mass, msqrd);
  case 15:
    return dndx_neutrino<MV, T, E>(xs, beta, mass, msqrd);
  case 16:
    return dndx_neutrino<MV, T, M>(xs, beta, mass, msqrd);
  case 17:
    return dndx_neutrino<MV, T, T>(xs, beta, mass, msqrd);
    // 18-26: N -> ντ + ℓ + ℓ
  case 18:
    return dndx_neutrino<TV, E, E>(xs, beta, mass, msqrd);
  case 19:
    return dndx_neutrino<TV, E, M>(xs, beta, mass, msqrd);
  case 20:
    return dndx_neutrino<TV, E, T>(xs, beta, mass, msqrd);
  case 21:
    return dndx_neutrino<TV, M, E>(xs, beta, mass, msqrd);
  case 22:
    return dndx_neutrino<TV, M, M>(xs, beta, mass, msqrd);
  case 23:
    return dndx_neutrino<TV, M, T>(xs, beta, mass, msqrd);
  case 24:
    return dndx_neutrino<TV, T, E>(xs, beta, mass, msqrd);
  case 25:
    return dndx_neutrino<TV, T, M>(xs, beta, mass, msqrd);
  case 26:
    return dndx_neutrino<TV, T, T>(xs, beta, mass, msqrd);
  default:
    return NeutrinoSpecRetType<X>::zeros_like(xs);
  }
}
auto Rhn::dndx_neutrino_v_l_l(double x, double beta, Gen g1, Gen g2,
                              Gen g3) const -> NeutrinoPt {
  return p_dndx_neutrino_v_l_l(x, beta, g1, g2, g3, *this);
}
auto Rhn::dndx_neutrino_v_l_l(const pyvec &x, double beta, Gen g1, Gen g2,
                              Gen g3) const -> pyvec {
  return p_dndx_neutrino_v_l_l(x, beta, g1, g2, g3, *this);
}
auto Rhn::dndx_neutrino_v_l_l(const stdvec &x, double beta, Gen g1, Gen g2,
                              Gen g3) const -> nuvec {
  return p_dndx_neutrino_v_l_l(x, beta, g1, g2, g3, *this);
}

template <class X>
auto p_dndx_neutrino_l_pi_pi0(const X &xs, double beta, const Rhn &model) ->
    typename NeutrinoSpecRetType<X>::type {
  const auto msqrd = SquaredAmplitudeNToLPiPi0(model);
  const double mass = model.mass();
  using CP = ChargedPion;
  using NP = NeutralPion;

  switch (model.gen()) {
  case Gen::Fst:
    return dndx_neutrino<Electron, CP, NP>(xs, beta, mass, msqrd);
  case Gen::Snd:
    return dndx_neutrino<Muon, CP, NP>(xs, beta, mass, msqrd);
  case Gen::Trd:
    return dndx_neutrino<Tau, CP, NP>(xs, beta, mass, msqrd);
  default:
    return NeutrinoSpecRetType<X>::zeros_like(xs);
  }
}
auto Rhn::dndx_neutrino_l_pi_pi0(double x, double beta) const -> NeutrinoPt {
  return p_dndx_neutrino_l_pi_pi0(x, beta, *this);
}
auto Rhn::dndx_neutrino_l_pi_pi0(const pyvec &x, double beta) const -> pyvec {
  return p_dndx_neutrino_l_pi_pi0(x, beta, *this);
}
auto Rhn::dndx_neutrino_l_pi_pi0(const stdvec &x, double beta) const -> nuvec {
  return p_dndx_neutrino_l_pi_pi0(x, beta, *this);
}

template <class X>
auto p_dndx_neutrino_v_pi_pi(const X &xs, double beta, const Rhn &model) ->
    typename NeutrinoSpecRetType<X>::type {
  const auto msqrd = SquaredAmplitudeNToLPiPi0(model);
  const double mass = model.mass();
  using P = ChargedPion;

  switch (model.gen()) {
  case Gen::Fst:
    return dndx_neutrino<ElectronNeutrino, P, P>(xs, beta, model.mass(), msqrd);
  case Gen::Snd:
    return dndx_neutrino<MuonNeutrino, P, P>(xs, beta, model.mass(), msqrd);
  case Gen::Trd:
    return dndx_neutrino<TauNeutrino, P, P>(xs, beta, model.mass(), msqrd);
  default:
    return NeutrinoSpecRetType<X>::zeros_like(xs);
  }
}
auto Rhn::dndx_neutrino_v_pi_pi(double x, double beta) const -> NeutrinoPt {
  return p_dndx_neutrino_v_pi_pi(x, beta, *this);
}
auto Rhn::dndx_neutrino_v_pi_pi(const pyvec &x, double beta) const -> pyvec {
  return p_dndx_neutrino_v_pi_pi(x, beta, *this);
}
auto Rhn::dndx_neutrino_v_pi_pi(const stdvec &x, double beta) const -> nuvec {
  return p_dndx_neutrino_v_pi_pi(x, beta, *this);
}

// =========================================================================
// ---- Positron Spectra Functions -----------------------------------------
// =========================================================================

auto Rhn::dndx_positron_l_pi(double x, double beta) const -> double {
  return dndx_positron_lepton<ChargedPion>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_positron_l_pi(const pyvec &x, double beta) const -> pyvec {
  return dndx_positron_lepton<ChargedPion>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_positron_l_pi(const stdvec &x, double beta) const -> stdvec {
  return dndx_positron_lepton<ChargedPion>(x, beta, p_mass, p_gen);
}

auto Rhn::dndx_positron_l_k(double x, double beta) const -> double {
  return dndx_positron_lepton<ChargedKaon>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_positron_l_k(const pyvec &x, double beta) const -> pyvec {
  return dndx_positron_lepton<ChargedKaon>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_positron_l_k(const stdvec &x, double beta) const -> stdvec {
  return dndx_positron_lepton<ChargedKaon>(x, beta, p_mass, p_gen);
}

auto Rhn::dndx_positron_v_pi0(double x, double beta) const -> double {
  return dndx_positron_neutrino<NeutralPion>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_positron_v_pi0(const pyvec &x, double beta) const -> pyvec {
  return dndx_positron_neutrino<NeutralPion>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_positron_v_pi0(const stdvec &x, double beta) const -> stdvec {
  return dndx_positron_neutrino<NeutralPion>(x, beta, p_mass, p_gen);
}

auto Rhn::dndx_positron_v_eta(double x, double beta) const -> double {
  return dndx_positron_neutrino<Eta>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_positron_v_eta(const pyvec &x, double beta) const -> pyvec {
  return dndx_positron_neutrino<Eta>(x, beta, p_mass, p_gen);
}
auto Rhn::dndx_positron_v_eta(const stdvec &x, double beta) const -> stdvec {
  return dndx_positron_neutrino<Eta>(x, beta, p_mass, p_gen);
}

template <class X>
auto p_dndx_positron_v_l_l(const X &xs, double beta, Gen genv, Gen genl1,
                           Gen genl2, const Rhn &model) -> X {
  const auto msqrd = SquaredAmplitudeNToVLL(model, genv, genl1, genl2);
  const double mass = model.mass();
  using EV = ElectronNeutrino;
  using MV = MuonNeutrino;
  using TV = TauNeutrino;
  using E = Electron;
  using M = Muon;
  using T = Tau;

  if (genv == Gen::Null || genl1 == Gen::Null || genl2 == Gen::Null) {
    return tools::zeros_like(xs);
  }

  const int a = (gen_to_int(genv) - 1);
  const int b = (gen_to_int(genl1) - 1);
  const int c = (gen_to_int(genl2) - 1);
  const int i = 9 * a + 3 * b + c;

  switch (i) {
    // 0-8: N -> νe + ℓ + ℓ
  case 0:
    return dndx_positron<EV, E, E>(xs, beta, mass, msqrd);
  case 1:
    return dndx_positron<EV, E, M>(xs, beta, mass, msqrd);
  case 2:
    return dndx_positron<EV, E, T>(xs, beta, mass, msqrd);
  case 3:
    return dndx_positron<EV, M, E>(xs, beta, mass, msqrd);
  case 4:
    return dndx_positron<EV, M, M>(xs, beta, mass, msqrd);
  case 5:
    return dndx_positron<EV, M, T>(xs, beta, mass, msqrd);
  case 6:
    return dndx_positron<EV, T, E>(xs, beta, mass, msqrd);
  case 7:
    return dndx_positron<EV, T, M>(xs, beta, mass, msqrd);
  case 8:
    return dndx_positron<EV, T, T>(xs, beta, mass, msqrd);
    // 9-17: N -> νμ + ℓ + ℓ
  case 9:
    return dndx_positron<MV, E, E>(xs, beta, mass, msqrd);
  case 10:
    return dndx_positron<MV, E, M>(xs, beta, mass, msqrd);
  case 11:
    return dndx_positron<MV, E, T>(xs, beta, mass, msqrd);
  case 12:
    return dndx_positron<MV, M, E>(xs, beta, mass, msqrd);
  case 13:
    return dndx_positron<MV, M, M>(xs, beta, mass, msqrd);
  case 14:
    return dndx_positron<MV, M, T>(xs, beta, mass, msqrd);
  case 15:
    return dndx_positron<MV, T, E>(xs, beta, mass, msqrd);
  case 16:
    return dndx_positron<MV, T, M>(xs, beta, mass, msqrd);
  case 17:
    return dndx_positron<MV, T, T>(xs, beta, mass, msqrd);
    // 18-26: N -> ντ + ℓ + ℓ
  case 18:
    return dndx_positron<TV, E, E>(xs, beta, mass, msqrd);
  case 19:
    return dndx_positron<TV, E, M>(xs, beta, mass, msqrd);
  case 20:
    return dndx_positron<TV, E, T>(xs, beta, mass, msqrd);
  case 21:
    return dndx_positron<TV, M, E>(xs, beta, mass, msqrd);
  case 22:
    return dndx_positron<TV, M, M>(xs, beta, mass, msqrd);
  case 23:
    return dndx_positron<TV, M, T>(xs, beta, mass, msqrd);
  case 24:
    return dndx_positron<TV, T, E>(xs, beta, mass, msqrd);
  case 25:
    return dndx_positron<TV, T, M>(xs, beta, mass, msqrd);
  case 26:
    return dndx_positron<TV, T, T>(xs, beta, mass, msqrd);
  default:
    return tools::zeros_like(xs);
  }
}

auto Rhn::dndx_positron_v_l_l(double x, double beta, Gen gv, Gen gl1,
                              Gen gl2) const -> double {
  return p_dndx_positron_v_l_l(x, beta, gv, gl1, gl2, *this);
}
auto Rhn::dndx_positron_v_l_l(const pyvec &x, double beta, Gen gv, Gen gl1,
                              Gen gl2) const -> pyvec {
  return p_dndx_positron_v_l_l(x, beta, gv, gl1, gl2, *this);
}
auto Rhn::dndx_positron_v_l_l(const stdvec &x, double beta, Gen gv, Gen gl1,
                              Gen gl2) const -> stdvec {
  return p_dndx_positron_v_l_l(x, beta, gv, gl1, gl2, *this);
}

template <class X>
auto p_dndx_positron_l_pi_pi0(const X &xs, double beta, const Rhn &model) -> X {
  const auto msqrd = SquaredAmplitudeNToLPiPi0(model);
  const double mass = model.mass();
  using CP = ChargedPion;
  using NP = NeutralPion;

  switch (model.gen()) {
  case Gen::Fst:
    return dndx_positron<Electron, CP, NP>(xs, beta, mass, msqrd);
  case Gen::Snd:
    return dndx_positron<Muon, CP, NP>(xs, beta, mass, msqrd);
  case Gen::Trd:
    return dndx_positron<Tau, CP, NP>(xs, beta, mass, msqrd);
  default:
    return tools::zeros_like(xs);
  }
}

auto Rhn::dndx_positron_l_pi_pi0(double x, double beta) const -> double {
  return p_dndx_positron_l_pi_pi0(x, beta, *this);
}
auto Rhn::dndx_positron_l_pi_pi0(const pyvec &x, double beta) const -> pyvec {
  return p_dndx_positron_l_pi_pi0(x, beta, *this);
}
auto Rhn::dndx_positron_l_pi_pi0(const stdvec &x, double beta) const -> stdvec {
  return p_dndx_positron_l_pi_pi0(x, beta, *this);
}

template <class X>
auto p_dndx_positron_v_pi_pi(const X &xs, double beta, const Rhn &model) -> X {
  const auto msqrd = SquaredAmplitudeNToLPiPi0(model);
  const double mass = model.mass();
  using P = ChargedPion;

  switch (model.gen()) {
  case Gen::Fst:
    return dndx_positron<ElectronNeutrino, P, P>(xs, beta, mass, msqrd);
  case Gen::Snd:
    return dndx_positron<MuonNeutrino, P, P>(xs, beta, mass, msqrd);
  case Gen::Trd:
    return dndx_positron<TauNeutrino, P, P>(xs, beta, mass, msqrd);
  default:
    return tools::zeros_like(xs);
  }
}

auto Rhn::dndx_positron_v_pi_pi(double x, double beta) const -> double {
  return p_dndx_positron_v_pi_pi(x, beta, *this);
}
auto Rhn::dndx_positron_v_pi_pi(const pyvec &x, double beta) const -> pyvec {
  return p_dndx_positron_v_pi_pi(x, beta, *this);
}
auto Rhn::dndx_positron_v_pi_pi(const stdvec &x, double beta) const -> stdvec {
  return p_dndx_positron_v_pi_pi(x, beta, *this);
}

} // namespace blackthorn
