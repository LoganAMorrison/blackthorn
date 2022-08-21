#include "blackthorn/Models/RhNeutrino.h"
#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/Spectra.h"
#include "blackthorn/Tools.h"

namespace blackthorn {

// =========================================================================
// ---- Photon Spectra Functions -------------------------------------------
// =========================================================================

template <class P1, class P2, class X>
auto dndx_photon(const X &x, double beta, double mass) -> X { // NOLINT
  DecaySpectrum<P1, P2> dspec(mass);
  return dspec.dndx_photon(x, beta);
}

template <class P, class X>
auto dndx_photon_lepton(const X &x, double beta, double mass, Gen G)
    -> X { // NOLINT
  switch (G) {
  case Gen::Fst: {
    return dndx_photon<Electron, P>(x, beta, mass);
  }
  case Gen::Snd: {
    return dndx_photon<Muon, P>(x, beta, mass);
  }
  default:
    return dndx_photon<Tau, P>(x, beta, mass);
  }
}

template <class P, class X>
auto dndx_photon_neutrino(const X &x, double beta, double mass, Gen G)
    -> X { // NOLINT
  switch (G) {
  case Gen::Fst: {
    return dndx_photon<ElectronNeutrino, P>(x, beta, mass);
  }
  case Gen::Snd: {
    return dndx_photon<MuonNeutrino, P>(x, beta, mass);
  }
  default:
    return dndx_photon<TauNeutrino, P>(x, beta, mass);
  }
}

auto RhNeutrinoMeV::dndx_l_pi(const py::array_t<double> &xs) const
    -> py::array_t<double> {
  return dndx_photon_lepton<ChargedPion>(xs, 0.0, p_mass, p_gen);
}
auto RhNeutrinoMeV::dndx_l_pi(double x, double beta) const -> double {
  return dndx_photon_lepton<ChargedPion>(x, beta, p_mass, p_gen);
}
auto RhNeutrinoMeV::dndx_l_pi(const py::array_t<double> &xs, double e) const
    -> py::array_t<double> {
  const double beta = tools::beta(e, p_mass);
  return dndx_photon_lepton<ChargedPion>(xs, beta, p_mass, p_gen);
}

auto RhNeutrinoMeV::dndx_l_k(const py::array_t<double> &xs) const
    -> py::array_t<double> {
  return dndx_photon_lepton<ChargedKaon>(xs, 0.0, p_mass, p_gen);
}
auto RhNeutrinoMeV::dndx_l_k(const py::array_t<double> &xs, double e) const
    -> py::array_t<double> {
  const double beta = tools::beta(e, p_mass);
  return dndx_photon_lepton<ChargedKaon>(xs, beta, p_mass, p_gen);
}

auto RhNeutrinoMeV::dndx_v_pi0(double x, double beta) const -> double {
  return dndx_photon_neutrino<NeutralPion>(x, beta, p_mass, p_gen);
}
auto RhNeutrinoMeV::dndx_v_pi0(const py::array_t<double> &xs) const
    -> py::array_t<double> {
  return dndx_photon_neutrino<NeutralPion>(xs, 0.0, p_mass, p_gen);
}
auto RhNeutrinoMeV::dndx_v_pi0(const std::vector<double> &xs) const
    -> std::vector<double> {
  return dndx_photon_neutrino<NeutralPion>(xs, 0.0, p_mass, p_gen);
}
auto RhNeutrinoMeV::dndx_v_pi0(const py::array_t<double> &xs, double e) const
    -> py::array_t<double> {
  const double beta = tools::beta(e, p_mass);
  return dndx_photon_neutrino<NeutralPion>(xs, beta, p_mass, p_gen);
}

auto RhNeutrinoMeV::dndx_v_eta(const py::array_t<double> &xs) const
    -> py::array_t<double> {
  return dndx_photon_neutrino<Eta>(xs, 0.0, p_mass, p_gen);
}
auto RhNeutrinoMeV::dndx_v_eta(const py::array_t<double> &xs, double e) const
    -> py::array_t<double> {
  const double beta = tools::beta(e, p_mass);
  return dndx_photon_neutrino<Eta>(xs, beta, p_mass, p_gen);
}

// =========================================================================
// ---- Neutrino Spectra Functions -----------------------------------------
// =========================================================================

template <class P1, class P2, class X>
auto dndx_neutrino(const X &x, double beta, double mass) -> X { // NOLINT
  DecaySpectrum<P1, P2> dspec(mass);
  return dspec.dndx_neutrino(x, beta);
}

template <class P, class X>
auto dndx_neutrino_lepton(const X &x, double beta, double mass, Gen G)
    -> X { // NOLINT
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
auto dndx_neutrino_neutrino(const X &x, double beta, double mass, Gen G)
    -> X { // NOLINT
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

auto RhNeutrinoMeV::dndx_neutrino_l_pi(const pyarray &xs) const -> pyarray {
  return dndx_neutrino_neutrino<ChargedPion>(xs, 0.0, p_mass, p_gen);
}
auto RhNeutrinoMeV::dndx_neutrino_l_pi(const pyarray &xs, double e) const
    -> pyarray {
  const double beta = tools::beta(e, p_mass);
  return dndx_neutrino_neutrino<ChargedPion>(xs, beta, p_mass, p_gen);
}

auto RhNeutrinoMeV::dndx_neutrino_l_k(const pyarray &xs) const -> pyarray {
  return dndx_neutrino_neutrino<ChargedKaon>(xs, 0.0, p_mass, p_gen);
}
auto RhNeutrinoMeV::dndx_neutrino_l_k(const pyarray &xs, double e) const
    -> pyarray {
  const double beta = tools::beta(e, p_mass);
  return dndx_neutrino_neutrino<ChargedKaon>(xs, beta, p_mass, p_gen);
}

auto RhNeutrinoMeV::dndx_neutrino_v_pi0(const pyarray &xs) const -> pyarray {
  return dndx_neutrino_neutrino<NeutralPion>(xs, 0.0, p_mass, p_gen);
}
auto RhNeutrinoMeV::dndx_neutrino_v_pi0(const pyarray &xs, double e) const
    -> pyarray {
  const double beta = tools::beta(e, p_mass);
  return dndx_neutrino_neutrino<NeutralPion>(xs, beta, p_mass, p_gen);
}

auto RhNeutrinoMeV::dndx_neutrino_v_eta(const pyarray &xs) const -> pyarray {
  return dndx_neutrino_neutrino<Eta>(xs, 0.0, p_mass, p_gen);
}
auto RhNeutrinoMeV::dndx_neutrino_v_eta(const pyarray &xs, double e) const
    -> pyarray {
  const double beta = tools::beta(e, p_mass);
  return dndx_neutrino_neutrino<Eta>(xs, beta, p_mass, p_gen);
}

// =========================================================================
// ---- Positron Spectra Functions -----------------------------------------
// =========================================================================

template <class P1, class P2, class X>
auto dndx_positron(const X &x, double beta, double mass) -> X { // NOLINT
  DecaySpectrum<P1, P2> dspec(mass);
  return dspec.dndx_positron(x, beta);
}

template <class P, class X>
auto dndx_positron_lepton(const X &x, double beta, double mass, Gen G)
    -> X { // NOLINT
  switch (G) {
  case Gen::Fst: {
    return dndx_positron<Electron, P>(x, beta, mass);
  }
  case Gen::Snd: {
    return dndx_positron<Muon, P>(x, beta, mass);
  }
  default:
    return dndx_positron<Tau, P>(x, beta, mass);
  }
}

template <class P, class X>
auto dndx_positron_neutrino(const X &x, double beta, double mass, Gen G)
    -> X { // NOLINT
  switch (G) {
  case Gen::Fst: {
    return dndx_positron<ElectronNeutrino, P>(x, beta, mass);
  }
  case Gen::Snd: {
    return dndx_positron<MuonNeutrino, P>(x, beta, mass);
  }
  default:
    return dndx_positron<TauNeutrino, P>(x, beta, mass);
  }
}

auto RhNeutrinoMeV::dndx_positron_l_pi(const pyarray &xs) const -> pyarray {
  return dndx_positron_lepton<ChargedPion>(xs, 0.0, p_mass, p_gen);
}
auto RhNeutrinoMeV::dndx_positron_l_pi(const pyarray &xs, double e) const
    -> pyarray {
  const double beta = tools::beta(e, p_mass);
  return dndx_positron_lepton<ChargedPion>(xs, beta, p_mass, p_gen);
}

auto RhNeutrinoMeV::dndx_positron_l_k(const pyarray &xs) const -> pyarray {
  return dndx_positron_lepton<ChargedKaon>(xs, 0.0, p_mass, p_gen);
}
auto RhNeutrinoMeV::dndx_positron_l_k(const pyarray &xs, double e) const
    -> pyarray {
  const double beta = tools::beta(e, p_mass);
  return dndx_positron_lepton<ChargedKaon>(xs, beta, p_mass, p_gen);
}

auto RhNeutrinoMeV::dndx_positron_v_pi0(const pyarray &xs) const -> pyarray {
  return dndx_positron_neutrino<NeutralPion>(xs, 0.0, p_mass, p_gen);
}
auto RhNeutrinoMeV::dndx_positron_v_pi0(const pyarray &xs, double e) const
    -> pyarray {
  const double beta = tools::beta(e, p_mass);
  return dndx_positron_neutrino<NeutralPion>(xs, beta, p_mass, p_gen);
}

auto RhNeutrinoMeV::dndx_positron_v_eta(const pyarray &xs) const -> pyarray {
  return dndx_positron_neutrino<Eta>(xs, 0.0, p_mass, p_gen);
}
auto RhNeutrinoMeV::dndx_positron_v_eta(const pyarray &xs, double e) const
    -> pyarray {
  const double beta = tools::beta(e, p_mass);
  return dndx_positron_neutrino<Eta>(xs, beta, p_mass, p_gen);
}

} // namespace blackthorn
