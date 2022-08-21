#ifndef BLACKTHORN_RH_NEUTRINO_HPP
#define BLACKTHORN_RH_NEUTRINO_HPP

#include "blackthorn/Amplitudes.h"
#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/PhaseSpace.h"
#include "blackthorn/Spectra/Base.h"
#include "blackthorn/Spectra/Decay.h"
#include "blackthorn/Spectra/Pythia.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

namespace blackthorn {

static constexpr double FPI = NeutralPion::decay_const;
static constexpr double GF = StandardModel::g_fermi;
static constexpr double QE = StandardModel::qe;

static constexpr std::complex<double> VUD =
    StandardModel::ckm<Gen::Fst, Gen::Fst>();
static constexpr std::complex<double> VUS =
    StandardModel::ckm<Gen::Fst, Gen::Snd>();

using pyarray = py::array_t<double>;

template <size_t N> class SquaredAmplitudeNToX {
protected:
  const double p_mass;                                   // NOLINT
  const double p_theta;                                  // NOLINT
  const Gen p_gen;                                       // NOLINT
  const std::array<DiracWf<FermionFlow::In>, 2> p_wfs_n; // NOLINT

public:
  SquaredAmplitudeNToX(double mass, double theta, Gen gen)
      : p_mass(mass), p_theta(theta), p_gen(gen),
        p_wfs_n(spinor_u(LVector<double>{mass, 0, 0, 0}, mass)) {}

  [[nodiscard]] auto mass() const -> const double & { return p_mass; }
  [[nodiscard]] auto gen() const -> const Gen & { return p_gen; }

  virtual auto operator()(const std::array<LVector<double>, N> &) const
      -> double = 0;
};

// ===========================================================================
// ---- RhNeutrino MeV -------------------------------------------------------
// ===========================================================================

class RhNeutrinoMeV {
public:
  static constexpr size_t DEFAULT_NEVENTS = 10'000;
  static constexpr size_t DEFAULT_BATCHSIZE = 100;

private:
  double p_mass;
  double p_theta;
  Gen p_gen;

  std::function<double(double, double)> p_dnde_l;

  static constexpr double mass_min = 0.0;
  static constexpr double mass_max = 1.0;

  /// Feynman rule for: N -> ℓ⁻ + π⁺
  [[nodiscard]] auto feynman_rule_n_lm_pip() const -> VertexFFSDeriv;

  /// Feynman rule for: N -> ℓ⁺ + π⁻
  [[nodiscard]] auto feynman_rule_n_lp_pim() const -> VertexFFSDeriv;

  /// Feynman rule for: N -> ℓ⁻ + K⁺
  [[nodiscard]] auto feynman_rule_n_lm_kp() const -> VertexFFSDeriv;

  /// Feynman rule for: N -> ℓ⁺ + K⁻
  [[nodiscard]] auto feynman_rule_n_lp_km() const -> VertexFFSDeriv;

  /// Feynman rule for: N -> ν + γ
  [[nodiscard]] auto feynman_rule_n_v_a() const -> VertexFFSDeriv;

  /// Feynman rule for: N -> ν + π⁰
  [[nodiscard]] auto feynman_rule_n_v_pi0() const -> VertexFFSDeriv;

  /// Feynman rule for: N -> ν + η
  [[nodiscard]] auto feynman_rule_n_v_eta() const -> VertexFFSDeriv;

  /// Feynman rule for: N -> ν + π⁺ + π⁻
  [[nodiscard]] auto feynman_rule_n_v_pi_pi() const -> VertexFFSS;

  /// Feynman rule for: N -> ν + K⁺ + K⁻
  [[nodiscard]] auto feynman_rule_n_v_k_k() const -> VertexFFSS;

  /// Feynman rule for: N -> ℓ⁻ + π⁺ + π⁰
  [[nodiscard]] auto feynman_rule_n_lm_pip_pi0() const -> VertexFFSS;

  /// Feynman rule for: N -> ℓ⁺ + π⁻ + π⁰
  [[nodiscard]] auto feynman_rule_n_lp_pim_pi0() const -> VertexFFSS;

  /// Feynman rule for: N -> ℓ⁻ + π⁺ + K⁰
  [[nodiscard]] auto feynman_rule_n_lm_pip_k0() const -> VertexFFSS;

  /// Feynman rule for: N -> ℓ⁺ + π⁻ + K⁰
  [[nodiscard]] auto feynman_rule_n_lp_pim_k0() const -> VertexFFSS;

  /// Feynman rule for: N -> ℓ⁻ + K⁺ + π⁰
  [[nodiscard]] auto feynman_rule_n_lm_kp_pi0() const -> VertexFFSS;

  /// Feynman rule for: N -> ℓ⁺ + K⁻ + π⁰
  [[nodiscard]] auto feynman_rule_n_lp_km_pi0() const -> VertexFFSS;

  /// Radiative feynman rule for: N -> ℓ⁻ + π⁺ + γ
  [[nodiscard]] auto feynman_rule_n_lm_pip_a() const -> VertexFFSV;

  /// Radiative feynman rule for: N -> ℓ⁺ + π⁻ + γ
  [[nodiscard]] auto feynman_rule_n_lp_pim_a() const -> VertexFFSV;

  /// Radiative feynman rule for: N -> ℓ⁺ + K⁻ + γ
  [[nodiscard]] auto feynman_rule_n_lm_kp_a() const -> VertexFFSV;

  /// Radiative feynman rule for: N -> ℓ⁺ + K⁻ + γ
  [[nodiscard]] auto feynman_rule_n_lp_km_a() const -> VertexFFSV;

  /// Feynman rule for: N -> ν + h
  [[nodiscard]] auto feynman_rule_n_v_h() const -> VertexFFS;

  /// Feynman rule for: N -> ν + Z
  [[nodiscard]] auto feynman_rule_n_v_z() const -> VertexFFV;

  /// Feynman rule for: N -> ℓ + W + γ
  [[nodiscard]] auto feynman_rule_n_l_w() const -> VertexFFV;

  friend class SquaredAmplitudeNToLPi;
  friend class SquaredAmplitudeNToLK;
  friend class SquaredAmplitudeNToVPi0;
  friend class SquaredAmplitudeNToVEta;
  friend class SquaredAmplitudeNToVPiPi;
  friend class SquaredAmplitudeNToVKK;
  friend class SquaredAmplitudeNToLPiPi0;
  friend class SquaredAmplitudeNToVLL;
  friend class SquaredAmplitudeNToVVV;

public:
  RhNeutrinoMeV(double mass, double theta, Gen gen) // NOLINT
      : p_mass(mass), p_theta(theta), p_gen(gen) {
    if (mass > mass_max) {
      // throw std::invalid_argument("Invalid mass. Must be less than " +
      //                             std::to_string(mass_max) + ".");
      if (gen == Gen::Snd) {
        p_dnde_l = [](double eg, double el) {
          return decay_spectrum<Muon>::dnde_photon(eg, el);
        };
      } else {
        p_dnde_l = [](double /*eg*/, double /*el*/) { return 0; };
      }
    }
  }

  [[nodiscard]] auto mass() const -> const double & { return p_mass; }
  auto mass() -> double & { return p_mass; }

  [[nodiscard]] auto theta() const -> const double & { return p_theta; }
  auto theta() -> double & { return p_theta; }

  [[nodiscard]] auto gen() const -> const Gen & { return p_gen; }
  auto gen() -> Gen & { return p_gen; }

  // =========================================================================
  // ---- Partial Width Functions --------------------------------------------
  // =========================================================================

  /// Compute the partial width for N -> ℓ⁻ + π⁺
  [[nodiscard]] auto width_l_pi() const -> double;

  /// Compute the partial width for N -> ℓ⁻ + K⁺.
  [[nodiscard]] auto width_l_k() const -> double;

  /// Compute the partial width for N -> ν + γ .
  [[nodiscard]] auto width_v_a() const -> double;

  /// Compute the partial width for N -> ν + π⁰.
  [[nodiscard]] auto width_v_pi0() const -> double;

  /// Compute the partial width for N -> ν + η.
  [[nodiscard]] auto width_v_eta() const -> double;

  /// Compute the partial width for N -> ν + ν + ν.
  ///
  /// @param genv1 generation of 1st final-state neutrino.
  /// @param genv2 generation of 2nd final-state neutrino.
  /// @param genv3 generation of 3rd final-state neutrino.
  /// @param nevents number of phase-space events to use.
  [[nodiscard]] auto width_v_v_v(Gen, Gen, Gen, size_t = DEFAULT_NEVENTS,
                                 size_t = DEFAULT_BATCHSIZE) const
      -> std::pair<double, double>;

  /// Compute the partial widths for N -> ν + ν + ν, summing over all
  /// possible generations.
  ///
  /// @param nevents number of phase-space events to use.
  [[nodiscard]] auto width_v_v_v(size_t = DEFAULT_NEVENTS,
                                 size_t = DEFAULT_BATCHSIZE) const
      -> std::pair<double, double>;

  /// Compute the partial widths for N -> ν + ℓ⁻ + ℓ⁺.
  ///
  /// @param genv1 generation of 1st final-state neutrino.
  /// @param genv2 generation of 2nd final-state neutrino.
  /// @param genv3 generation of 3rd final-state neutrino.
  /// @param nevents number of phase-space events to use.
  [[nodiscard]] auto width_v_l_l(Gen, Gen, Gen, size_t = DEFAULT_NEVENTS,
                                 size_t = DEFAULT_BATCHSIZE) const
      -> std::pair<double, double>;

  /// Compute the partial widths for N -> ν + π⁺ + π⁻.
  ///
  /// @param nevents number of phase-space events to use.
  [[nodiscard]] auto width_v_pi_pi(size_t = DEFAULT_NEVENTS,
                                   size_t = DEFAULT_BATCHSIZE) const
      -> std::pair<double, double>;

  // [[nodiscard]] auto width_v_k_k() const -> std::pair<double, double>;

  /// Compute the partial widths for N -> ℓ⁻ + π⁺ + π⁰.
  ///
  /// @param nevents number of phase-space events to use.
  [[nodiscard]] auto width_l_pi_pi0(size_t = DEFAULT_NEVENTS,
                                    size_t = DEFAULT_BATCHSIZE) const
      -> std::pair<double, double>;

  // =========================================================================
  // ---- Photon Spectra Functions -------------------------------------------
  // =========================================================================

  // clang-format off

  /// Compute the dN/dx from N -> ℓ⁻ + π⁺ in rest frame of RH neutrino.
  ///
  /// @param x Value of x = 2 * eng_gamma / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_photon_l_pi(double x, double beta) const -> double;
  [[nodiscard]] auto dndx_photon_l_pi(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_photon_l_pi(const std::vector<double> &x, double beta) const -> std::vector<double>;

  /// Compute the dN/dx from N -> ℓ⁻ + K⁺ for a boosted RH neutrino.
  ///
  /// @param x Value of x = 2 * eng_gamma / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_photon_l_k(double x, double beta) const -> double;
  [[nodiscard]] auto dndx_photon_l_k(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_photon_l_k(const std::vector<double> &x, double beta) const -> std::vector<double>;

  /// Compute the dN/dx from N -> ν + π⁰ for a boosted RH neutrino.
  ///
  /// @param x Value of x = 2 * eng_gamma / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_photon_v_pi0(double x, double beta) const -> double;
  [[nodiscard]] auto dndx_photon_v_pi0(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_photon_v_pi0(const std::vector<double> &x, double beta) const -> std::vector<double>;

  /// Compute the dN/dx from N -> ν + η for a boosted RH neutrino.
  ///
  /// @param x Value of x = 2 * eng_gamma / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_photon_v_eta(double x, double beta) const -> double;
  [[nodiscard]] auto dndx_photon_v_eta(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_photon_v_eta(const std::vector<double> &x, double beta) const -> std::vector<double>;

  /// Compute the dN/dx from N -> ν + π⁺ + π⁻ for a boosted RH neutrino.
  ///
  /// @param x Value of x = 2 * eng_gamma / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_photon_v_pi_pi(double x, double beta) const -> double;
  [[nodiscard]] auto dndx_photon_v_pi_pi(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_photon_v_pi_pi(const std::vector<double> &x, double beta) const -> std::vector<double>;

  /// Compute the dN/dx from N -> ν + ℓ⁻ + ℓ⁺ for a boosted RH neutrino.
  ///
  /// @param x Value of x = 2 * eng_gamma / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  /// @param gv Generation of the final state neutrino
  /// @param gl1 Generation of the 1st final state lepton
  /// @param gl2 Generation of the 2nd final state lepton
  [[nodiscard]] auto dndx_photon_v_l_l(double x, double beta, Gen gv, Gen gl1, Gen gl2) const -> double;
  [[nodiscard]] auto dndx_photon_v_l_l(const pyarray &x, double beta, Gen gv, Gen gl1, Gen gl2) const -> pyarray;
  [[nodiscard]] auto dndx_photon_v_l_l(const std::vector<double> &x, double beta, Gen gv, Gen gl1, Gen gl2) const -> std::vector<double>;

  /// Compute the dN/dx from N -> ℓ⁻ + π⁺ + π⁰ in rest frame of RH neutrino.
  ///
  /// @param x Value of x = 2 * eng_gamma / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_photon_l_pi_pi0(double x, double beta) const -> double;
  [[nodiscard]] auto dndx_photon_l_pi_pi0(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_photon_l_pi_pi0(const std::vector<double> &x, double beta) const -> std::vector<double>;

  // clang-format on

  // =========================================================================
  // ---- Neutrino Spectra Functions -----------------------------------------
  // =========================================================================

  using NeutrinoVec = std::array<std::vector<double>, 3>;
  using NeutrinoPt = std::array<double, 3>;

  // clang-format off

  /// Compute the dN/dx from N -> ℓ⁻ + π⁺ in rest frame of RH neutrino.
  ///
  /// @param x Value of x = 2 * neutrino_energy / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_neutrino_l_pi(double x, double beta) const -> NeutrinoPt;
  [[nodiscard]] auto dndx_neutrino_l_pi(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_neutrino_l_pi(const std::vector<double> &x, double beta) const -> NeutrinoVec;

  /// Compute the dN/dx from N -> ℓ⁻ + K⁺ in rest frame of RH neutrino.
  ///
  /// @param x Value of x = 2 * neutrino_energy / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_neutrino_l_k(double x, double beta) const -> NeutrinoPt;
  [[nodiscard]] auto dndx_neutrino_l_k(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_neutrino_l_k(const std::vector<double> &x, double beta) const -> NeutrinoVec;

  /// Compute the dN/dx from N -> ν + π⁰ in rest frame of RH neutrino.
  ///
  /// @param x Value of x = 2 * neutrino_energy / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_neutrino_v_pi0(double x, double beta) const -> NeutrinoPt;
  [[nodiscard]] auto dndx_neutrino_v_pi0(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_neutrino_v_pi0(const std::vector<double> &x, double beta) const -> NeutrinoVec;

  /// Compute the dN/dx from N -> ν + η in rest frame of RH neutrino.
  ///
  /// @param x Value of x = 2 * neutrino_energy / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_neutrino_v_eta(double x, double beta) const -> NeutrinoPt;
  [[nodiscard]] auto dndx_neutrino_v_eta(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_neutrino_v_eta(const std::vector<double> &x, double beta) const -> NeutrinoVec;

  /// Compute the dN/dx from N -> ν + π⁺ + π⁻ in rest frame of RH neutrino.
  ///
  /// @param x Value of x = 2 * neutrino_energy / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_neutrino_v_pi_pi(double x, double beta) const -> NeutrinoPt;
  [[nodiscard]] auto dndx_neutrino_v_pi_pi(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_neutrino_v_pi_pi(const std::vector<double> &x, double beta) const -> NeutrinoVec;

  /// Compute the dN/dx from N -> ν + ℓ⁻ + ℓ⁺ in rest frame of RH neutrino.
  ///
  /// @param x Value of x = 2 * neutrino_energy / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  /// @param gv Generation of the final state neutrino
  /// @param gl1 Generation of the 1st final state lepton
  /// @param gl2 Generation of the 2nd final state lepton
  [[nodiscard]] auto dndx_neutrino_v_l_l(double x, double beta, Gen gv, Gen gl1, Gen gl2) const -> NeutrinoPt;
  [[nodiscard]] auto dndx_neutrino_v_l_l(const pyarray &x, double beta, Gen gv, Gen gl1, Gen gl2) const -> pyarray;
  [[nodiscard]] auto dndx_neutrino_v_l_l(const std::vector<double> &x, double beta, Gen gv, Gen gl1, Gen gl2) const -> NeutrinoVec;

  /// Compute the dN/dx from N -> ℓ⁻ + π⁺ + π⁰ in rest frame of RH neutrino.
  ///
  /// @param x Value of x = 2 * neutrino_energy / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_neutrino_l_pi_pi0(double x, double beta) const -> NeutrinoPt;
  [[nodiscard]] auto dndx_neutrino_l_pi_pi0(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_neutrino_l_pi_pi0(const std::vector<double> &x, double beta) const -> NeutrinoVec;

  // clang-format on

  // =========================================================================
  // ---- Positron Spectra Functions -----------------------------------------
  // =========================================================================

  // clang-format off

  /// Compute the dN/dx from N -> ℓ⁻ + π⁺ in rest frame of RH neutrino.
  ///
  /// @param x Value of x = 2 * positron_energy / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_positron_l_pi(double x, double beta) const -> double;
  [[nodiscard]] auto dndx_positron_l_pi(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_positron_l_pi(const std::vector<double> &x, double beta) const -> std::vector<double>;

  /// Compute the dN/dx from N -> ℓ⁻ + K⁺ for a boosted RH neutrino.
  ///
  /// @param x Value of x = 2 * positron_energy / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_positron_l_k(double x, double beta) const -> double;
  [[nodiscard]] auto dndx_positron_l_k(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_positron_l_k(const std::vector<double> &x, double beta) const -> std::vector<double>;

  /// Compute the dN/dx from N -> ν + π⁰ for a boosted RH neutrino.
  ///
  /// @param x Value of x = 2 * positron_energy / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_positron_v_pi0(double x, double beta) const -> double;
  [[nodiscard]] auto dndx_positron_v_pi0(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_positron_v_pi0(const std::vector<double> &x, double beta) const -> std::vector<double>;

  /// Compute the dN/dx from N -> ν + η for a boosted RH neutrino.
  ///
  /// @param x Value of x = 2 * positron_energy / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_positron_v_eta(double x, double beta) const -> double;
  [[nodiscard]] auto dndx_positron_v_eta(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_positron_v_eta(const std::vector<double> &x, double beta) const -> std::vector<double>;

  /// Compute the dN/dx from N -> ν + π⁺ + π⁻ for a boosted RH neutrino.
  ///
  /// @param x Value of x = 2 * positron_energy / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_positron_v_pi_pi(double x, double beta) const -> double;
  [[nodiscard]] auto dndx_positron_v_pi_pi(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_positron_v_pi_pi(const std::vector<double> &x, double beta) const -> std::vector<double>;

  /// Compute the dN/dx from N -> ν + ℓ⁻ + ℓ⁺ for a boosted RH neutrino.
  ///
  /// @param x Value of x = 2 * positron_energy / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  /// @param gv Generation of the final state neutrino
  /// @param gl1 Generation of the 1st final state lepton
  /// @param gl2 Generation of the 2nd final state lepton
  [[nodiscard]] auto dndx_positron_v_l_l(double x, double beta, Gen gv, Gen gl1, Gen gl2) const -> double;
  [[nodiscard]] auto dndx_positron_v_l_l(const pyarray &x, double beta, Gen gv, Gen gl1, Gen gl2) const -> pyarray;
  [[nodiscard]] auto dndx_positron_v_l_l(const std::vector<double> &x, double beta, Gen gv, Gen gl1, Gen gl2) const -> std::vector<double>;

  /// Compute the dN/dx from N -> ℓ⁻ + π⁺ + π⁰ in rest frame of RH neutrino.
  ///
  /// @param x Value of x = 2 * positron_energy / mass_n where dN/dx is evaluated.
  /// @param beta Boost velocity of the RH neutrino.
  [[nodiscard]] auto dndx_positron_l_pi_pi0(double x, double beta) const -> double;
  [[nodiscard]] auto dndx_positron_l_pi_pi0(const pyarray &x, double beta) const -> pyarray;
  [[nodiscard]] auto dndx_positron_l_pi_pi0(const std::vector<double> &x, double beta) const -> std::vector<double>;

  // clang-format on
};

// ===========================================================================
// ---- RhNeutrino GeV -------------------------------------------------------
// ===========================================================================

class RhNeutrinoGeV {
public:
  static constexpr size_t DEFAULT_NEVENTS = 10'000;
  static constexpr size_t DEFAULT_BATCHSIZE = 100;

private:
  double p_mass;  // NOLINT
  double p_theta; // NOLINT
  Gen p_gen;      // NOLINT

  using dpair = std::pair<double, double>;

  static constexpr double mass_min = 5.0;
  static constexpr double mass_max = StandardModel::mplank;

  /// Feynman rule for: N -> ν + h
  [[nodiscard]] auto feynman_rule_n_v_h() const -> VertexFFS;

  /// Feynman rule for: N -> ν + Z
  [[nodiscard]] auto feynman_rule_n_v_z() const -> VertexFFV;

  /// Feynman rule for: N -> ℓ + W
  [[nodiscard]] auto feynman_rule_n_l_w() const -> VertexFFV;

  friend class SquaredAmplitudeNToVH;
  friend class SquaredAmplitudeNToVZ;
  friend class SquaredAmplitudeNToLW;
  friend class SquaredAmplitudeNToVUU;
  friend class SquaredAmplitudeNToVDD;
  friend class SquaredAmplitudeNToLUD;
  friend class SquaredAmplitudeNToVLL;
  friend class SquaredAmplitudeNToVVV;

public:
  RhNeutrinoGeV(double mass, double theta, Gen gen) // NOLINT
      : p_mass(mass), p_theta(theta), p_gen(gen) {
    // if (mass < mass_min) {
    //   throw std::invalid_argument("Invalid mass. Must be greater than " +
    //                               std::to_string(mass_min) + ".");
    // }
  }

  [[nodiscard]] auto mass() const -> const double & { return p_mass; }
  auto mass() -> double & { return p_mass; }

  [[nodiscard]] auto theta() const -> const double & { return p_theta; }
  auto theta() -> double & { return p_theta; }

  [[nodiscard]] auto gen() const -> const Gen & { return p_gen; }
  auto gen() -> Gen & { return p_gen; }

  /// Compute the partial width for N -> ν + h
  [[nodiscard]] auto width_v_h() const -> double;

  /// Compute the partial width for N -> ν + Z
  [[nodiscard]] auto width_v_z() const -> double;

  /// Compute the partial width for N -> ℓ + W
  [[nodiscard]] auto width_l_w() const -> double;

  // =========================================================================
  // ---- Partial Width Functions --------------------------------------------
  // =========================================================================

  /// Compute the partial widths for N -> ν + u + u.
  ///
  /// @param genu Generation of the final-state up-type quarks.
  /// @param nevents Number of phase-space events to use.
  /// @param batchsize Number of events computed per batch.
  [[nodiscard]] auto width_v_u_u(Gen, size_t = DEFAULT_NEVENTS,
                                 size_t = DEFAULT_BATCHSIZE) const -> dpair;

  /// Compute the partial widths for N -> ν + d + d.
  ///
  /// @param genu Generation of the final-state down-type quarks.
  /// @param nevents Number of phase-space events to use.
  /// @param batchsize Number of events computed per batch.
  [[nodiscard]] auto width_v_d_d(Gen, size_t = DEFAULT_NEVENTS,
                                 size_t = DEFAULT_BATCHSIZE) const -> dpair;

  /// Compute the partial widths for N -> ℓ + u + d.
  ///
  /// @param genu Generation of the final-state up-type quark.
  /// @param gend Generation of the final-state down-type quark.
  /// @param nevents Number of phase-space events to use.
  /// @param batchsize Number of events computed per batch.
  [[nodiscard]] auto width_l_u_d(Gen, Gen, size_t = DEFAULT_NEVENTS,
                                 size_t = DEFAULT_BATCHSIZE) const -> dpair;

  /// Compute the partial widths for N -> ν + ν + ν.
  ///
  /// @param gen1 Generation of the 1st final-state neutrino.
  /// @param gen2 Generation of the 2nd final-state neutrino.
  /// @param gen3 Generation of the 3rd final-state neutrino.
  /// @param nevents Number of phase-space events to use.
  /// @param batchsize Number of events computed per batch.
  [[nodiscard]] auto width_v_v_v(Gen, Gen, Gen, size_t = DEFAULT_NEVENTS,
                                 size_t = DEFAULT_BATCHSIZE) const -> dpair;

  /// Compute the partial widths for N -> ν + ν + ν, summing over all
  /// final-state neutrino generations.
  ///
  /// @param nevents Number of phase-space events to use.
  /// @param batchsize Number of events computed per batch.
  [[nodiscard]] auto width_v_v_v(size_t = DEFAULT_NEVENTS,
                                 size_t = DEFAULT_BATCHSIZE) const -> dpair;

  /// Compute the partial widths for N -> ν + ℓ⁻ + ℓ⁺, summing over all
  /// final-state neutrino generations.
  ///
  /// @param genv Generation of the final-state neutrino.
  /// @param genl1 Generation of the 1st final-state lepton.
  /// @param genl2 Generation of the 2nd final-state lepton.
  /// @param nevents Number of phase-space events to use.
  /// @param batchsize Number of events computed per batch.
  [[nodiscard]] auto width_v_l_l(Gen, Gen, Gen, size_t = DEFAULT_NEVENTS,
                                 size_t = DEFAULT_BATCHSIZE) const -> dpair;

  // =========================================================================
  // ---- Spectra Functions --------------------------------------------------
  // =========================================================================

  /// Compute the photon spectum dN/dx from N -> ν + h.
  ///
  /// @param xmin Minimum value of x = 2 * eng_gamma / mass_n
  /// @param xmax Maximum value of x = 2 * eng_gamma / mass_n
  /// @param nbins Number of bins use in constructing histogram.
  /// @param nevents Number of phase-space events to use.
  [[nodiscard]] auto
  dndx_v_h(double xmin, double xmax, unsigned int nbins,
           unsigned int nevents = PythiaSpectrum<2>::DEFAULT_NEVENTS) const
      -> PythiaSpectrum<2>::SpectrumType;

  /// Compute the photon spectum dN/dx from N -> ν + Z.
  ///
  /// @param xmin Minimum value of x = 2 * eng_gamma / mass_n
  /// @param xmax Maximum value of x = 2 * eng_gamma / mass_n
  /// @param nbins Number of bins use in constructing histogram.
  /// @param nevents Number of phase-space events to use.
  [[nodiscard]] auto
  dndx_v_z(double xmin, double xmax, unsigned int nbins,
           unsigned int nevents = PythiaSpectrum<2>::DEFAULT_NEVENTS) const
      -> PythiaSpectrum<2>::SpectrumType;

  /// Compute the photon spectum dN/dx from N -> ℓ + W.
  ///
  /// @param xmin Minimum value of x = 2 * eng_gamma / mass_n
  /// @param xmax Maximum value of x = 2 * eng_gamma / mass_n
  /// @param nbins Number of bins use in constructing histogram.
  /// @param nevents Number of phase-space events to use.
  [[nodiscard]] auto
  dndx_l_w(double xmin, double xmax, unsigned int nbins,
           unsigned int nevents = PythiaSpectrum<2>::DEFAULT_NEVENTS) const
      -> PythiaSpectrum<2>::SpectrumType;

  [[nodiscard]] auto dndx_l_w_fsr(double x) const -> double;

  /// Compute the photon spectum dN/dx from N -> ν + u + u.
  ///
  /// @param xmin Minimum value of x = 2 * eng_gamma / mass_n
  /// @param xmax Maximum value of x = 2 * eng_gamma / mass_n
  /// @param nbins Number of bins use in constructing histogram.
  /// @param genu Generation of final-state up-type quarks.
  /// @param nevents Number of phase-space events to use.
  [[nodiscard]] auto
  dndx_v_u_u(double xmin, double xmax, unsigned int nbins, Gen genu,
             unsigned int nevents = PythiaSpectrum<2>::DEFAULT_NEVENTS) const
      -> PythiaSpectrum<3>::SpectrumType;

  /// Compute the photon spectum dN/dx from N -> ν + d + d.
  ///
  /// @param xmin Minimum value of x = 2 * eng_gamma / mass_n
  /// @param xmax Maximum value of x = 2 * eng_gamma / mass_n
  /// @param nbins Number of bins use in constructing histogram.
  /// @param gend Generation of final-state down-type quarks.
  /// @param nevents Number of phase-space events to use.
  [[nodiscard]] auto
  dndx_v_d_d(double xmin, double xmax, unsigned int nbins, Gen gend,
             unsigned int nevents = PythiaSpectrum<2>::DEFAULT_NEVENTS) const
      -> PythiaSpectrum<3>::SpectrumType;

  /// Compute the photon spectum dN/dx from N -> ℓ + u + d.
  ///
  /// @param xmin Minimum value of x = 2 * eng_gamma / mass_n
  /// @param xmax Maximum value of x = 2 * eng_gamma / mass_n
  /// @param nbins Number of bins use in constructing histogram.
  /// @param genu Generation of final-state up-type quark.
  /// @param gend Generation of final-state down-type quark.
  /// @param nevents Number of phase-space events to use.
  [[nodiscard]] auto
  dndx_l_u_d(double xmin, double xmax, unsigned int nbins, Gen genu, Gen gend,
             unsigned int nevents = PythiaSpectrum<2>::DEFAULT_NEVENTS) const
      -> PythiaSpectrum<3>::SpectrumType;

  /// Compute the photon spectum dN/dx from N -> ν + ℓ⁻ + ℓ⁺.
  ///
  /// @param xmin Minimum value of x = 2 * eng_gamma / mass_n
  /// @param xmax Maximum value of x = 2 * eng_gamma / mass_n
  /// @param nbins Number of bins use in constructing histogram.
  /// @param genv Generation of the final-state neutrino.
  /// @param genl1 Generation of the 1st final-state lepton.
  /// @param genl2 Generation of the 2nd final-state lepton.
  /// @param nevents Number of phase-space events to use.
  [[nodiscard]] auto
  dndx_v_l_l(double xmin, double xmax, unsigned int nbins, Gen genv, Gen genl1,
             Gen genl2,
             unsigned int nevents = PythiaSpectrum<2>::DEFAULT_NEVENTS) const
      -> PythiaSpectrum<3>::SpectrumType;
};

// ===========================================================================
// ---- Generic Squared Matrix Elements --------------------------------------
// ===========================================================================

// template <size_t N> class SquaredAmplitudeNToX {
// protected:
//   const double p_mass;                                   // NOLINT
//   const double p_theta;                                  // NOLINT
//   const Gen p_gen;                                       // NOLINT
//   const std::array<DiracWf<FermionFlow::In>, 2> p_wfs_n; // NOLINT

// public:
//   SquaredAmplitudeNToX(double mass, double theta, Gen gen)
//       : p_mass(mass), p_theta(theta), p_gen(gen),
//         p_wfs_n(spinor_u(LVector<double>{mass, 0, 0, 0}, mass)) {}

//   [[nodiscard]] auto mass() const -> const double & { return p_mass; }
//   [[nodiscard]] auto gen() const -> const Gen & { return p_gen; }

//   virtual auto operator()(const std::array<LVector<double>, N> &) const
//       -> double = 0;
// };

class SquaredAmplitudeNToVVV : SquaredAmplitudeNToX<3> {
private:
  using DiracWfI = DiracWf<FermionFlow::In>;
  using DiracWfO = DiracWf<FermionFlow::Out>;

  const VertexFFS p_vertex_nvh;
  const VertexFFV p_vertex_nvz;
  const VertexFFS p_vertex_vvh;
  const VertexFFV p_vertex_vvz;
  const Gen p_genv1;
  const Gen p_genv2;
  const Gen p_genv3;

public:
  SquaredAmplitudeNToVVV(const RhNeutrinoGeV &, Gen, Gen, Gen);
  SquaredAmplitudeNToVVV(const RhNeutrinoMeV &, Gen, Gen, Gen);
  auto operator()(const std::array<LVector<double>, 3> &ps) const
      -> double override;
};

class SquaredAmplitudeNToVLL : SquaredAmplitudeNToX<3> {
private:
  const double p_ml1;
  const double p_ml2;
  const Gen p_genv;
  const Gen p_genl1;
  const Gen p_genl2;

  // Diagram1 vertices
  const VertexFFS p_vertex_nvh;
  const VertexFFS p_vertex_llh;
  // Diagram2 vertices
  const VertexFFV p_vertex_nvz;
  const VertexFFV p_vertex_llz;
  // Diagram3 vertices
  const VertexFFV p_vertex_nlw;
  const VertexFFV p_vertex_vlw;

public:
  SquaredAmplitudeNToVLL(const RhNeutrinoMeV &, Gen, Gen, Gen);
  SquaredAmplitudeNToVLL(const RhNeutrinoGeV &, Gen, Gen, Gen);
  auto operator()(const std::array<LVector<double>, 3> &ps) const
      -> double override;
};

// ===========================================================================
// ---- MeV Squared Matrix Elements ------------------------------------------
// ===========================================================================

/// Squared Amplitude for N -> nu + pi0.
class SquaredAmplitudeNToVPi0 : SquaredAmplitudeNToX<2> {
private:
  const VertexFFSDeriv p_vertex;

public:
  explicit SquaredAmplitudeNToVPi0(const RhNeutrinoMeV &);
  auto operator()(const std::array<LVector<double>, 2> &ps) const
      -> double override;
};

class SquaredAmplitudeNToVEta : SquaredAmplitudeNToX<2> {
private:
  const VertexFFSDeriv p_vertex;

public:
  explicit SquaredAmplitudeNToVEta(const RhNeutrinoMeV &);
  auto operator()(const std::array<LVector<double>, 2> &ps) const
      -> double override;
};

/// Squared Amplitude for N -> l + pi.
class SquaredAmplitudeNToLPi : SquaredAmplitudeNToX<2> {
private:
  const double p_ml;
  const VertexFFSDeriv p_vertex;

public:
  explicit SquaredAmplitudeNToLPi(const RhNeutrinoMeV &, Gen);
  auto operator()(const std::array<LVector<double>, 2> &ps) const
      -> double override;
};

/// Squared Amplitude for N -> l + K.
class SquaredAmplitudeNToLK : SquaredAmplitudeNToX<2> {
private:
  const double p_ml;
  const VertexFFSDeriv p_vertex;

public:
  explicit SquaredAmplitudeNToLK(const RhNeutrinoMeV &);
  auto operator()(const std::array<LVector<double>, 2> &ps) const
      -> double override;
};

/// Squared Amplitude for N -> nu + pi + pi.
class SquaredAmplitudeNToVPiPi : SquaredAmplitudeNToX<3> {
private:
  const VertexFFSS p_vertex;

public:
  explicit SquaredAmplitudeNToVPiPi(const RhNeutrinoMeV &);
  auto operator()(const std::array<LVector<double>, 3> &ps) const
      -> double override;
};

/// Squared Amplitude for N -> nu + K + K.
class SquaredAmplitudeNToVKK : SquaredAmplitudeNToX<3> {
private:
  const VertexFFSS p_vertex;

public:
  explicit SquaredAmplitudeNToVKK(const RhNeutrinoMeV &);
  auto operator()(const std::array<LVector<double>, 3> &ps) const
      -> double override;
};

/// Squared Amplitude for N -> l + pi + pi.
// class SquaredAmplitudeNToLPiPi0 : SquaredAmplitudeNToX<3> {
// private:
//   const double p_pre;
//   const double p_c0;
//   const double p_c1;
//   const double p_c2;

// public:
//   explicit SquaredAmplitudeNToLPiPi0(const RhNeutrinoMeV &);
//   auto operator()(const std::array<LVector<double>, 3> &ps) const
//       -> double override;
// };

class SquaredAmplitudeNToLPiPi0 : SquaredAmplitudeNToX<3> {
private:
  const VertexFFSS p_vertex;

public:
  explicit SquaredAmplitudeNToLPiPi0(const RhNeutrinoMeV &);
  auto operator()(const std::array<LVector<double>, 3> &ps) const
      -> double override;
};

// ===========================================================================
// ---- GeV Squared Matrix Elements ------------------------------------------
// ===========================================================================

class SquaredAmplitudeNToVH : SquaredAmplitudeNToX<2> {
public:
  explicit SquaredAmplitudeNToVH(const RhNeutrinoGeV &);
  auto operator()(const std::array<LVector<double>, 2> &ps) const
      -> double override;
};
class SquaredAmplitudeNToVZ : SquaredAmplitudeNToX<2> {
public:
  explicit SquaredAmplitudeNToVZ(const RhNeutrinoGeV &);
  auto operator()(const std::array<LVector<double>, 2> &ps) const
      -> double override;
};
class SquaredAmplitudeNToLW : SquaredAmplitudeNToX<2> {
private:
  const double p_ml;

public:
  explicit SquaredAmplitudeNToLW(const RhNeutrinoGeV &);
  auto operator()(const std::array<LVector<double>, 2> &ps) const
      -> double override;
};

class SquaredAmplitudeNToVUU : SquaredAmplitudeNToX<3> {
private:
  const double p_mu;
  const VertexFFS p_vertex_nvh;
  const VertexFFV p_vertex_nvz;
  const VertexFFS p_vertex_uuh;
  const VertexFFV p_vertex_uuz;

public:
  SquaredAmplitudeNToVUU(const RhNeutrinoGeV &, Gen);
  auto operator()(const std::array<LVector<double>, 3> &ps) const
      -> double override;
};

class SquaredAmplitudeNToVDD : SquaredAmplitudeNToX<3> {
private:
  const double p_md;
  const VertexFFS p_vertex_nvh;
  const VertexFFV p_vertex_nvz;
  const VertexFFS p_vertex_ddh;
  const VertexFFV p_vertex_ddz;

public:
  SquaredAmplitudeNToVDD(const RhNeutrinoGeV &, Gen);
  auto operator()(const std::array<LVector<double>, 3> &ps) const
      -> double override;
};

class SquaredAmplitudeNToLUD : SquaredAmplitudeNToX<3> {
private:
  const double p_ml;
  const double p_mu;
  const double p_md;
  const VertexFFV p_vertex_nlw;
  const VertexFFV p_vertex_udw;

public:
  SquaredAmplitudeNToLUD(const RhNeutrinoGeV &, Gen, Gen);
  auto operator()(const std::array<LVector<double>, 3> &ps) const
      -> double override;
};

} // namespace blackthorn

#endif // BLACKTHORN_RH_NEUTRINO_HPP
