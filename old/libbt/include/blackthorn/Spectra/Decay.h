#ifndef BLACKTHORN_SPECTRA_DECAY_H
#define BLACKTHORN_SPECTRA_DECAY_H

#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/PhaseSpace.h"
#include "blackthorn/Tools.h"
#include <algorithm>
#include <array>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <cmath>
#include <execution>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace blackthorn {

/// Structure holding the neutrino spectrum for different neutrino flavors.
template <typename T> struct NeutrinoSpectrum {
  T electron;
  T muon;
  T tau;
};

template <typename P> class decay_spectrum {
public:
  /// Compute the radiative decay spectrum, dN/dE, from the decay of a parent
  /// particle into all possible finals states and a photon.
  ///
  /// @param photon_energy Energy of the photon
  /// @param parent_energy Energy of the decaying particle.
  static auto dnde_photon(double /*photon_energy*/, double /*parent_energy*/)
      -> double {
    return 0.0;
  }
  static auto dnde_photon(const std::vector<double> &photon_energy,
                          double /*parent_energy*/) -> std::vector<double> {
    return std::vector<double>(photon_energy.size(), 0.0);
  }
  static auto dnde_photon(const py::array_t<double> &photon_energy,
                          double /*parent_energy*/) -> py::array_t<double> {
    return tools::zeros_like(photon_energy);
  }

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a parent
  /// particle into all possible finals states and a positron.
  ///
  /// @param positron_energy Energy of the positron
  /// @param parent_energy Energy of the decaying particle.
  static auto dnde_positron(double /*ep*/, double /*parent_energy*/) -> double {
    return 0.0;
  }
  static auto dnde_positron(const std::vector<double> &positron_energy,
                            double /*parent_energy*/) -> std::vector<double> {
    return std::vector<double>(positron_energy.size(), 0.0);
  }
  static auto dnde_positron(const py::array_t<double> &positron_energy,
                            double /*parent_energy*/) -> py::array_t<double> {
    return tools::zeros_like(positron_energy);
  }

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a parent
  /// particle into all possible finals states and neutrinos.
  ///
  /// @param neutrino_energy Energy of the neutrino
  /// @param parent_energy Energy of the decaying particle.
  ///
  /// @returns spectrum Structure containing dN/dE for each neutrino flavor
  static auto dnde_neutrino(double /*neutrino_energy*/,
                            double /*parent_energy*/)
      -> NeutrinoSpectrum<double> {
    return {0.0, 0.0, 0.0};
  }
  static auto dnde_neutrino(const std::vector<double> &neutrino_energy,
                            double /*parent_energy*/)
      -> NeutrinoSpectrum<std::vector<double>> {
    return {std::vector<double>(neutrino_energy.size(), 0.0),
            std::vector<double>(neutrino_energy.size(), 0.0),
            std::vector<double>(neutrino_energy.size(), 0.0)};
  }

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a parent
  /// particle into all possible finals states and neutrino of specified
  /// generation.
  ///
  /// @param neutrino_energy Energy of the neutrino
  /// @param parent_energy Energy of the decaying particle
  /// @param gen Generation of the neutrino
  static auto dnde_neutrino(double /*neutrino_energy*/,
                            double /*parent_energy*/, Gen /*gen*/) -> double {
    return 0.0;
  }
};

//============================================================================
//---- Neutral Pion Decay Spectra --------------------------------------------
//============================================================================

template <> class decay_spectrum<NeutralPion> {
public:
  /// Compute the radiative decay spectrum, dN/dE, from the decay of a neutral
  /// pion into two photons.
  ///
  /// @param egam Energy of the photon
  /// @param epi Energy of the pion
  static auto dnde_photon(double egam, double epi) -> double;

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a neutral
  /// pion into two photons.
  ///
  /// @param egams Photon energies.
  /// @param epi Energy of the pion
  static auto dnde_photon(const std::vector<double> &egams, double epi)
      -> std::vector<double>;

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a neutral
  /// pion into two photons.
  ///
  /// @param egams Photon energies.
  /// @param epi Energy of the pion
  static auto dnde_photon(const py::array_t<double> &egams, double epi)
      -> py::array_t<double>;

  /// Compute the radiative decay spectrum, dN/dx, from the decay of a neutral
  /// pion into two photons.
  ///
  /// @param x Scaled photon energy x = 2 egam / cme.
  /// @param epi Energy of the pion
  /// @param cme Center of mass energy.
  static auto dndx_photon(double x, double epi, double cme) -> double;

  /// Compute the radiative decay spectrum, dN/dx, from the decay of a neutral
  /// pion into two photons.
  ///
  /// @param xs Scaled photon energies x = 2 egam / cme.
  /// @param epi Energy of the pion
  /// @param cme Center of mass energy.
  static auto dndx_photon(const std::vector<double> &xs, double epi, double cme)
      -> std::vector<double>;

  /// Compute the radiative decay spectrum, dN/dx, from the decay of a neutral
  /// pion into two photons.
  ///
  /// @param xs Scaled photon energies x = 2 egam / cme.
  /// @param epi Energy of the pion
  /// @param cme Center of mass energy.
  static auto dndx_photon(const py::array_t<double> &xs, double epi, double cme)
      -> py::array_t<double>;

  // Positron and neutrino

  static auto dnde_positron(double /*ep*/, double /*epi*/) -> double {
    return 0.0;
  }
  static auto dnde_positron(const std::vector<double> &positron_energy,
                            double /*epi*/) -> std::vector<double> {
    return std::vector<double>(positron_energy.size(), 0.0);
  }
  static auto dnde_positron(const py::array_t<double> &positron_energy,
                            double /*epi*/) -> py::array_t<double> {
    return tools::zeros_like(positron_energy);
  }

  static auto dnde_neutrino(double /*neutrino_energy*/, double /*epi*/)
      -> NeutrinoSpectrum<double> {
    return {0.0, 0.0, 0.0};
  }
  static auto dnde_neutrino(const std::vector<double> &neutrino_energy,
                            double /*epi*/)
      -> NeutrinoSpectrum<std::vector<double>> {
    return {std::vector<double>(neutrino_energy.size(), 0.0),
            std::vector<double>(neutrino_energy.size(), 0.0),
            std::vector<double>(neutrino_energy.size(), 0.0)};
  }
  static auto dnde_neutrino(double /*neutrino_energy*/, double /*epi*/,
                            Gen /*gen*/) -> double {
    return 0.0;
  }
};

//============================================================================
//---- Muon Decay Spectra ----------------------------------------------------
//============================================================================

template <> class decay_spectrum<Muon> {
private:
  static constexpr double me = Electron::mass;
  static constexpr double mmu = Muon::mass;
  static constexpr double ratio_e_mu_mass_sq = (me / mmu) * (me / mmu);
  static constexpr double alpha_em = StandardModel::alpha_em;

public:
  // =========================================================================
  // ---- Photon -------------------------------------------------------------
  // =========================================================================

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a muon.
  ///
  /// @param egam Energy of the photon
  /// @param emu Energy of the muon
  static auto dnde_photon(double egam, double emu) -> double;

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a muon.
  ///
  /// @param egams Energies of the photon
  /// @param emu Energy of the muon
  static auto dnde_photon(const std::vector<double> &egams, double emu)
      -> std::vector<double>;

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a muon.
  ///
  /// @param egams Energies of the photon
  /// @param emu Energy of the muon
  static auto dnde_photon(const py::array_t<double> &egams, double emu)
      -> py::array_t<double>;

  // =========================================================================
  // ---- Positron -----------------------------------------------------------
  // =========================================================================

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a muon into
  /// positrons.
  ///
  /// @param ep Energy of the positron
  /// @param emu Energy of the muon
  static auto dnde_positron(double ep, double muon_energy) -> double;

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a muon into
  /// positrons.
  ///
  /// @param eps Energies of the positron
  /// @param emu Energy of the muon
  static auto dnde_positron(const std::vector<double> &eps, double muon_energy)
      -> std::vector<double>;

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a muon into
  /// positrons.
  ///
  /// @param eps Energies of the positron
  /// @param emu Energy of the muon
  static auto dnde_positron(const py::array_t<double> &eps, double muon_energy)
      -> py::array_t<double>;

  // =========================================================================
  // ---- Nuetrino -----------------------------------------------------------
  // =========================================================================

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a muon into
  /// nuetrinos.
  ///
  /// @param enu Energy of the neutrino
  /// @param emu Energy of the muon
  static auto dnde_neutrino(double e, double muon_energy)
      -> NeutrinoSpectrum<double>;

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a muon into
  /// nuetrinos.
  ///
  /// @param e Energy of the neutrino
  /// @param muon_energy Energy of the muon
  /// @param g Generation of neutrino.
  static auto dnde_neutrino(double e, double muon_energy, Gen g) -> double;

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a muon into
  /// neutrinos.
  ///
  /// @param enus Energies of the positron
  /// @param emu Energy of the muon
  static auto dnde_neutrino(const std::vector<double> &enus, double emu)
      -> NeutrinoSpectrum<std::vector<double>>;
};

//============================================================================
//---- Charged Pion Decay Spectra --------------------------------------------
//============================================================================

template <> class decay_spectrum<ChargedPion> { // NOLINT
public:
  /// Compute the radiative decay spectrum, dN/dE, from the decay of a charged
  /// pion.
  ///
  /// @param egam Energy of the photon
  /// @param epi Energy of the charged pion.
  static auto dnde_photon(double egam, double epi) -> double;

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a charged
  /// pion.
  ///
  /// @param egams Energies of the photon
  /// @param epi Energy of the charged pion.
  static auto dnde_photon(const std::vector<double> &egams, double epi)
      -> std::vector<double>;

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a charged
  /// pion.
  ///
  /// @param egams Energies of the photon
  /// @param epi Energy of the charged pion.
  static auto dnde_photon(const py::array_t<double> &egams, double epi)
      -> py::array_t<double>;

  // =========================================================================
  // ---- Positron -----------------------------------------------------------
  // =========================================================================

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a muon into
  /// positrons.
  ///
  /// @param ep Energy of the positron
  /// @param epi Energy of the charged pion.
  static auto dnde_positron(double ep, double epi) -> double;

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a muon into
  /// positrons.
  ///
  /// @param eps Energies of the positron
  /// @param epi Energy of the charged pion.
  static auto dnde_positron(const std::vector<double> &eps, double epi)
      -> std::vector<double>;

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a muon into
  /// positrons.
  ///
  /// @param eps Energies of the positron
  /// @param epi Energy of the charged pion.
  static auto dnde_positron(const py::array_t<double> &eps, double epi)
      -> py::array_t<double>;

  // =========================================================================
  // ---- Nuetrino -----------------------------------------------------------
  // =========================================================================

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a muon into
  /// nuetrinos.
  ///
  /// @param enu Energy of the neutrino
  /// @param epi Energy of the charged pion.
  static auto dnde_neutrino(double enu, double epi) -> NeutrinoSpectrum<double>;

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a muon into
  /// neutrinos.
  ///
  /// @param enus Energies of the positron
  /// @param epi Energy of the charged pion.
  static auto dnde_neutrino(const std::vector<double> &enus, double epi)
      -> NeutrinoSpectrum<std::vector<double>>;

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a muon into
  /// neutrinos.
  ///
  /// @param enus Energies of the neutrino
  /// @param epi Energy of the charged pion.
  static auto dnde_neutrino(const py::array_t<double> &enus, double epi)
      -> py::array_t<double>;
};

// ============================================================================
// ---- Charged Kaon Decay Spectra --------------------------------------------
// ============================================================================

template <> class decay_spectrum<ChargedKaon> { // NOLINT

public:
  /// Compute the radiative decay spectrum, dN/dE, from the decay of a charged
  /// kaon.
  ///
  /// @param egam Energy of the photon
  /// @param ek Energy of the charged kaon.
  static auto dnde_photon(double egam, double ek) -> double;

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a charged
  /// kaon.
  ///
  /// @param egams Energies of the photon
  /// @param ek Energy of the charged kaon.
  static auto dnde_photon(const std::vector<double> &egams, double ek)
      -> std::vector<double>;

  /// Compute the radiative decay spectrum, dN/dE, from the decay of a charged
  /// kaon.
  ///
  /// @param egams Energies of the photon
  /// @param ek Energy of the charged kaon.
  static auto dnde_photon(const py::array_t<double> &egams, double ek)
      -> py::array_t<double>;

  // Positron and neutrino

  static auto dnde_positron(double e, double ek) -> double;

  static auto dnde_positron(const std::vector<double> &positron_energy,
                            double ek) -> std::vector<double>;

  static auto dnde_positron(const py::array_t<double> &positron_energy,
                            double ek) -> py::array_t<double>;

  static auto dnde_neutrino(double neutrino_energy, double ek)
      -> NeutrinoSpectrum<double>;
  static auto dnde_neutrino(double neutrino_energy, double ek, Gen gen)
      -> double;

  static auto dnde_neutrino(const std::vector<double> &neutrino_energy,
                            double ek) -> NeutrinoSpectrum<std::vector<double>>;
};

} // namespace blackthorn

#endif // BLACKTHORN_SPECTRA_DECAY_H
