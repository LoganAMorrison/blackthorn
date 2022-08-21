#ifndef BLACKTHORN_AMPLITUDES_HPP
#define BLACKTHORN_AMPLITUDES_HPP

#include "blackthorn/Tensors.h"
#include "blackthorn/Wavefunctions.h"
#include <array>
#include <cassert>
#include <complex>
#include <iostream>
#include <stdexcept>

namespace blackthorn {

// ===========================================================================
// ---- Loop Calculations ----------------------------------------------------
// ===========================================================================

class TensorCoeffsA {
private:
  using ValueType = std::complex<double>;
  std::vector<ValueType> p_tn;
  std::vector<ValueType> p_tnuv;
  size_t p_r;

public:
  TensorCoeffsA(std::vector<ValueType> tn, std::vector<ValueType> tnuv, int r)
      : p_tn(std::move(tn)), p_tnuv(std::move(tnuv)), p_r(r) {}
  [[nodiscard]] auto coeff(size_t) const -> const ValueType &;
  [[nodiscard]] auto coeff_uv(size_t) const -> const ValueType &;
};

class TensorCoeffsB {
public:
  using ValueType = std::complex<double>;
  using ArrType = std::vector<std::vector<ValueType>>;

private:
  std::vector<ValueType> p_tn;
  std::vector<ValueType> p_tnuv;
  size_t p_r;

public:
  // TensorCoeffsB(ArrType tn, ArrType tnuv)
  //     : p_tn(std::move(tn)), p_tnuv(std::move(tnuv)) {}
  TensorCoeffsB(std::vector<ValueType> tn, std::vector<ValueType> tnuv, int r)
      : p_tn(std::move(tn)), p_tnuv(std::move(tnuv)), p_r(r) {}
  [[nodiscard]] auto coeff(size_t, size_t) const -> const ValueType &;
  [[nodiscard]] auto coeff_uv(size_t, size_t) const -> const ValueType &;

  [[nodiscard]] auto coeff(size_t) const -> const ValueType &;
  [[nodiscard]] auto coeff_uv(size_t) const -> const ValueType &;
};

class TensorCoeffsC {
private:
  using ValueType = std::complex<double>;
  std::vector<ValueType> p_tn;
  std::vector<ValueType> p_tnuv;
  size_t p_r;

public:
  TensorCoeffsC(std::vector<ValueType> tn, std::vector<ValueType> tnuv, int r)
      : p_tn(std::move(tn)), p_tnuv(std::move(tnuv)), p_r(r) {}
  [[nodiscard]] auto coeff(size_t, size_t, size_t) const -> const ValueType &;
  [[nodiscard]] auto coeff_uv(size_t, size_t, size_t) const
      -> const ValueType &;

  [[nodiscard]] auto coeff(size_t) const -> const ValueType &;
  [[nodiscard]] auto coeff_uv(size_t) const -> const ValueType &;
};

class Loop {
private:
  auto init() -> void;
  auto initevent() -> void;
  using ValueType = std::complex<double>;

public:
  Loop();

  [[nodiscard]] auto scalarA0(const ValueType &) -> ValueType;

  [[nodiscard]] auto scalarB0(const ValueType &, const ValueType &,
                              const ValueType &) -> ValueType;

  [[nodiscard]] auto scalarC0(const ValueType &, const ValueType &,
                              const ValueType &, const ValueType &,
                              const ValueType &, const ValueType &)
      -> ValueType;

  [[nodiscard]] auto tensor_coeffs_a(const ValueType &, int) -> TensorCoeffsA;

  [[nodiscard]] auto tensor_coeffs_b(const ValueType &, const ValueType &,
                                     const ValueType &, int) -> TensorCoeffsB;

  [[nodiscard]] auto tensor_coeffs_c(const ValueType &, const ValueType &,
                                     const ValueType &, const ValueType &,
                                     const ValueType &, const ValueType &, int)
      -> TensorCoeffsC;

  [[nodiscard]] auto tensor_coeffs_tn(std::vector<ValueType>,
                                      std::vector<ValueType>, int, int)
      -> std::pair<std::vector<ValueType>, std::vector<ValueType>>;
};

// ===========================================================================
// ---- Fermion-Fermion-Scalar Vertex ----------------------------------------
// ===========================================================================

/// Vertex representing fermion-fermion-scalar vertex corresponding to the
/// interaction: S * fbar * (gL * PL + gR * PR) * f
struct VertexFFS {
  std::complex<double> left;
  std::complex<double> right;
} __attribute__((aligned(32)));

/// Vertex representing fermion-fermion-scalar vertex corresponding to the
/// interaction: d[S,mu] * fbar * gamma[mu] * (gL * PL + gR * PR) * f
struct VertexFFSDeriv {
  std::complex<double> left;
  std::complex<double> right;
} __attribute__((aligned(32)));

// ===========================================================================
// ---- Fermion-Fermion-Vector Vertex ----------------------------------------
// ===========================================================================

/// Vertex representing fermion-fermion-vector vertex corresponding to the
/// interaction: V[mu] * fbar * gamma[mu] * (gL * PL + gR * PR) * f
struct VertexFFV {
  std::complex<double> left;
  std::complex<double> right;
} __attribute__((aligned(32)));

// ===========================================================================
// ---- Fermion-Fermion-Scalar-Vector Vertex ---------------------------------
// ===========================================================================

/// Vertex representing fermion-fermion-vector vertex corresponding to the
/// interaction: V[mu] * S * fbar * gamma[mu] * (gL * PL + gR * PR) * f
struct VertexFFSV {
  std::complex<double> left;
  std::complex<double> right;
} __attribute__((aligned(32)));

// ===========================================================================
// ---- Fermion-Fermion-Scalar-Vector Vertex ---------------------------------
// ===========================================================================

/// Vertex representing fermion-fermion-scalar-scalar vertex corresponding to
/// the interaction:
///     S1 * d[S2, mu] * fbar * gamma[mu] * (gL1 * PL + gR1 * PR) * f
///   + S2 * d[S1, mu] * fbar * gamma[mu] * (gL2 * PL + gR2 * PR) * f
///   + S1 * S2 * fbar * gamma[mu] * (gL3 * PL + gR3 * PR) * f
struct VertexFFSS {
  std::complex<double> left1;
  std::complex<double> right1;
  std::complex<double> left2;
  std::complex<double> right2;
  std::complex<double> left3;
  std::complex<double> right3;
} __attribute__((aligned(128)));

// ===========================================================================
// ---- Propagators ----------------------------------------------------------
// ===========================================================================

class Propagator {
private:
  static auto cmplx_mass_sqr(double, double) -> std::complex<double>;
  static auto propagator_den(const LVector<double> &, double, double)
      -> std::complex<double>;
  static auto propagator_den(const LVector<double> &) -> std::complex<double>;

public:
  template <class Wf> static auto attach(Wf *, double, double) -> void;
  template <class Wf> static auto attach(Wf *) -> void;
};

// ===========================================================================
// ---- Currents -------------------------------------------------------------
// ===========================================================================

class Current {
private:
  using DiracWfI = DiracWf<FermionFlow::In>;
  using DiracWfO = DiracWf<FermionFlow::Out>;

  template <class V, class... WfIns>
  static auto fuse_into(ScalarWf *, const V &, const WfIns &...) -> void;

  template <class V, class... WfIns>
  static auto fuse_into(DiracWfI *, const V &, const WfIns &...) -> void;

  template <class V, class... WfIns>
  static auto fuse_into(DiracWfO *, const V &, const WfIns &...) -> void;

  template <class V, class... WfIns>
  static auto fuse_into(VectorWf *, const V &, const WfIns &...) -> void;

public:
  template <class V, class... WfIns>
  static auto generate(ScalarWf *out, const V &v, double mass, double width,
                       const WfIns &...wfs) -> void {
    fuse_into(out, v, wfs...);
    Propagator::attach(out, mass, width);
  }

  template <class V, class... WfIns>
  static auto generate(DiracWfI *out, const V &v, double mass, double width,
                       const WfIns &...wfs) -> void {
    fuse_into(out, v, wfs...);
    Propagator::attach(out, mass, width);
  }

  template <class V, class... WfIns>
  static auto generate(DiracWfO *out, const V &v, double mass, double width,
                       const WfIns &...wfs) -> void {
    fuse_into(out, v, wfs...);
    Propagator::attach(out, mass, width);
  }

  template <class V, class... WfIns>
  static auto generate(VectorWf *out, const V &v, double mass, double width,
                       const WfIns &...wfs) -> void {
    fuse_into(out, v, wfs...);
    Propagator::attach(out, mass, width);
  }

  template <class V, class... WfIns>
  static auto generate(VectorWf *out, const V &v, const WfIns &...wfs) -> void {
    fuse_into(out, v, wfs...);
    Propagator::attach(out);
  }
};

// ===========================================================================
// ---- Amplitudes -----------------------------------------------------------
// ===========================================================================

/**
 * Compute the a fermion-fermion-scalar amplitude.
 * @param v fermion-fermion-scalar vertex structure
 * @param fo flow-out fermion wavefunction
 * @param fi flow-in fermion wavefunction
 * @param phi scalar wavefunction
 * @return the amplitude <0|vertex|psi_in,psi_out,phi>
 */
auto amplitude(const VertexFFS &, const DiracWf<FermionFlow::Out> &,
               const DiracWf<FermionFlow::In> &, const ScalarWf &)
    -> std::complex<double>;

/**
 * Compute the a fermion-fermion-scalar amplitude.
 * @param v fermion-fermion-scalar vertex structure
 * @param fo flow-out fermion wavefunction
 * @param fi flow-in fermion wavefunction
 * @param phi scalar wavefunction
 * @return the amplitude <0|vertex|psi_in,psi_out,phi>
 */
auto amplitude(const VertexFFSDeriv &, const DiracWf<FermionFlow::Out> &,
               const DiracWf<FermionFlow::In> &, const ScalarWf &)
    -> std::complex<double>;

/**
 * Compute the a fermion-fermion-vector amplitude.
 * @param v fermion-fermion-vector vertex structure
 * @param fo flow-out fermion wavefunction
 * @param fi flow-in fermion wavefunction
 * @param eps vector wavefunction
 * @return the amplitude <0|vertex|psi_in,psi_out,eps>
 */
auto amplitude(const VertexFFV &, const DiracWf<FermionFlow::Out> &,
               const DiracWf<FermionFlow::In> &, const VectorWf &)
    -> std::complex<double>;

auto amplitude(const VertexFFSS &v, const DiracWf<FermionFlow::Out> &fo,
               const DiracWf<FermionFlow::In> &fi, const ScalarWf &phi1,
               const ScalarWf &phi2) -> std::complex<double>;

} // namespace blackthorn

#endif // BLACKTHORN_AMPLITUDES_HPP
