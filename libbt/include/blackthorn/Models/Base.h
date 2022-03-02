#ifndef BLACKTHORN_MODELS_BASE_H
#define BLACKTHORN_MODELS_BASE_H

#include "blackthorn/Amplitudes.h"
#include <type_traits> // NOLINT

namespace blackthorn {

enum class EnergyScale {
  MeV = 0,
  GeV = 1,
};

/**
 * Enumeration specifying the generation of a sm fermion.
 */
enum class Gen {
  // First generation sm fermion.
  Null = -1,
  // First generation sm fermion.
  Fst = 0,
  // Second generation sm fermion.
  Snd = 1,
  // Third generation sm fermion.
  Trd = 2,
};

// ===========================================================================
// ---- Quark Traits ---------------------------------------------------------
// ===========================================================================

template <class F> struct is_up_type_quark {
  static constexpr bool value = false;
};
template <class F>
inline constexpr bool is_up_type_quark_v = is_up_type_quark<F>::value;

template <class F> struct is_down_type_quark {
  static constexpr bool value = false;
};
template <class F>
inline constexpr bool is_down_type_quark_v = is_down_type_quark<F>::value;

template <class F> struct is_quark {
  static constexpr bool value =
      is_up_type_quark<F>::value || is_down_type_quark<F>::value;
};
template <class F> inline constexpr bool is_quark_v = is_quark<F>::value;

//===========================================================================
//---- Lepton Traits --------------------------------------------------------
//===========================================================================

template <class F> struct is_neutrino { static constexpr bool value = false; };
template <class F> inline constexpr bool is_neutrino_v = is_neutrino<F>::value;

template <class F> struct is_charged_lepton {
  static constexpr bool value = false;
};
template <class F>
inline constexpr bool is_charged_lepton_v = is_charged_lepton<F>::value;

template <class F> struct is_lepton {
  static constexpr bool value =
      is_neutrino<F>::value || is_charged_lepton<F>::value;
};
template <class F> inline constexpr bool is_lepton_v = is_lepton<F>::value;

// ===========================================================================
// ---- Classification Traits ------------------------------------------------
// ===========================================================================

template <class F> struct is_up_type {
  static constexpr bool value =
      is_neutrino<F>::value || is_up_type_quark<F>::value;
};
template <class F> inline constexpr bool is_up_type_v = is_up_type<F>::value;

template <class F> struct is_down_type {
  static constexpr bool value =
      is_charged_lepton<F>::value || is_down_type_quark<F>::value;
};
template <class F>
inline constexpr bool is_down_type_v = is_down_type<F>::value;

template <class F> struct is_fermion {
  static constexpr bool value =
      (is_neutrino<F>::value || is_up_type_quark<F>::value ||
       is_charged_lepton<F>::value || is_down_type_quark<F>::value);
};
template <class F> inline constexpr bool is_fermion_v = is_fermion<F>::value;

template <class F> struct is_dirac_fermion {
  static constexpr bool value =
      (is_up_type_quark<F>::value || is_charged_lepton<F>::value ||
       is_down_type_quark<F>::value);
};
template <class F>
inline constexpr bool is_dirac_fermion_v = is_dirac_fermion<F>::value;

template <class F> struct is_majorana_fermion {
  static constexpr bool value = is_neutrino<F>::value;
};
template <class F>
inline constexpr bool is_majorana_fermion_v = is_majorana_fermion<F>::value;

template <class F> struct is_vector_boson {
  static constexpr bool value = false;
};
template <class F>
inline constexpr bool is_vector_boson_v = is_vector_boson<F>::value;

template <class F> struct is_scalar_boson {
  static constexpr bool value = false;
};
template <class F>
inline constexpr bool is_scalar_boson_v = is_scalar_boson<F>::value;

template <class F> struct is_massless { static constexpr bool value = false; };
template <class F> inline constexpr bool is_massless_v = is_massless<F>::value;

template <class F> struct is_self_conj { static constexpr bool value = false; };
template <class F>
inline constexpr bool is_self_conj_v = is_self_conj<F>::value;

template <class F> struct is_stable { static constexpr bool value = true; };
template <class F> inline constexpr bool is_stable_v = is_stable<F>::value;

// ===========================================================================
// ---- Field Attributes -----------------------------------------------------
// ===========================================================================

template <class F> struct field_attrs {
  static constexpr auto pdg() -> int { return 0; };
  static constexpr auto mass() -> double { return 0.0; };
  static constexpr auto width() -> double { return 0.0; };
  static constexpr auto generation() -> Gen { return Gen::Null; };
};

// ===========================================================================
// ---- Quantum Numbers ------------------------------------------------------
// ===========================================================================

/**
 * Trait specifying the quantum numbers of a quantum field.
 */
template <class F> struct quantum_numbers {

  /**
   * The electric charge of the field in units of the electron charge
   */
  static constexpr auto charge() -> double {
    if constexpr (is_charged_lepton<F>::value) {
      return -1.0;
    }
    if constexpr (is_up_type_quark<F>::value) {
      return 2.0 / 3.0;
    }
    if constexpr (is_down_type_quark<F>::value) {
      return -1.0 / 3.0;
    }
    return 0.0;
  }

  /**
   * The eigen-value under SU(2)_L
   */
  static constexpr auto weak_iso_spin() -> double {
    if constexpr (is_up_type_quark<F>::value || is_neutrino<F>::value) {
      return 0.5;
    }
    if constexpr (is_down_type_quark<F>::value || is_charged_lepton<F>::value) {
      return -0.5;
    }
    return 0.0;
  }
};

// ===========================================================================
// ---- Feynman Rules --------------------------------------------------------
// ===========================================================================

template <typename... Fields> struct feynman_rule;

} // namespace blackthorn

#endif // BLACKTHORN_MODELS_BASE_H
