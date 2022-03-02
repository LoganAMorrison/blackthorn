#ifndef BLACKTHORN_WAVEFUNCTIONS_H
#define BLACKTHORN_WAVEFUNCTIONS_H

#include "blackthorn/Tensors.h"
#include <array>     // NOLINT
#include <complex>   // NOLINT
#include <iostream>  // NOLINT
#include <stdexcept> // NOLINT

namespace blackthorn {

// ===========================================================================
// ---- Classification -------------------------------------------------------
// ===========================================================================

enum class FermionFlow {
  In,
  Out,
};

enum InOut {
  In,
  Out,
};

static constexpr FermionFlow FlowIn = FermionFlow::In;
static constexpr FermionFlow FlowOut = FermionFlow::Out;

static constexpr InOut Incoming = InOut::In;
static constexpr InOut Outgoing = InOut::Out;

// ===========================================================================
// ---- Scalar Wavefunction --------------------------------------------------
// ===========================================================================

class ScalarWf {
public:
  using ValueType = std::complex<double>;
  using WfType = std::complex<double>;
  using MomentumType = LVector<double>;

private:
  WfType p_data;
  MomentumType p_momentum;

public:
  ScalarWf(const WfType &wf, const MomentumType &p)
      : p_data(wf), p_momentum(p) {}
  ScalarWf() : p_data(WfType{}), p_momentum(MomentumType{}) {}

  [[nodiscard]] auto wavefunction() const -> const WfType & { return p_data; }
  auto wavefunction() -> WfType & { return p_data; }

  [[nodiscard]] auto momentum() const -> const MomentumType & {
    return p_momentum;
  }
  auto momentum() -> MomentumType & { return p_momentum; }

  static auto momentum_flow() -> double { return 1; }

  [[nodiscard]] auto momentum_with_flow() const -> MomentumType {
    return momentum_flow() * p_momentum;
  }

  friend auto operator<<(std::ostream &os, const ScalarWf &wf)
      -> std::ostream & {
    os << "ScalarWf(" << wf.wavefunction() << ")";
    return os;
  }

  template <typename T> auto operator*=(const T &rhs) -> void { p_data *= rhs; }
  template <typename T> auto operator/=(const T &rhs) -> void { p_data /= rhs; }
};

auto scalar_wf(ScalarWf *, const LVector<double> &, InOut) -> void;
auto scalar_wf(const LVector<double> &, InOut) -> ScalarWf;

// ===========================================================================
// ---- Dirac Wavefunction ---------------------------------------------------
// ===========================================================================

template <FermionFlow F> class DiracWf {
public:
  using ValueType = std::complex<double>;
  using WfType = std::array<ValueType, 4>;
  using MomentumType = LVector<double>;

private:
  WfType p_data;
  MomentumType p_momentum;

public:
  DiracWf(const WfType &wf, const MomentumType &p)
      : p_data(wf), p_momentum(p) {}
  DiracWf() : p_data(WfType{}), p_momentum(MomentumType{}) {}

  auto operator[](size_t i) const -> const ValueType & { return p_data[i]; }
  auto operator[](size_t i) -> ValueType & { return p_data[i]; }

  [[nodiscard]] auto wavefunction() const -> const WfType & { return p_data; }
  auto wavefunction() -> WfType & { return p_data; }

  [[nodiscard]] auto momentum() const -> const MomentumType & {
    return p_momentum;
  }
  auto momentum() -> MomentumType & { return p_momentum; }

  auto momentum(size_t i) -> double & { return p_momentum[i]; }
  [[nodiscard]] auto momentum(size_t i) const -> const double & {
    return p_momentum[i];
  }

  static auto momentum_flow() -> double {
    return F == FermionFlow::In ? 1.0 : -1.0;
  }

  [[nodiscard]] auto momentum_with_flow() const -> MomentumType {
    return (F == FermionFlow::In ? 1.0 : -1.0) * momentum_flow() * p_momentum;
  }

  friend auto operator<<(std::ostream &os, const DiracWf &p) -> std::ostream & {
    os << "DiracWf(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3]
       << ")";
    return os;
  }

  [[nodiscard]] auto e() const -> const double & { return p_momentum[0]; }
  [[nodiscard]] auto px() const -> const double & { return p_momentum[1]; }
  [[nodiscard]] auto py() const -> const double & { return p_momentum[2]; }
  [[nodiscard]] auto pz() const -> const double & { return p_momentum[3]; }

  template <typename T> auto operator*=(const T &rhs) -> void {
    p_data[0] *= rhs;
    p_data[1] *= rhs;
    p_data[2] *= rhs;
    p_data[3] *= rhs;
  }

  template <typename T> auto operator/=(const T &rhs) -> void {
    p_data[0] /= rhs;
    p_data[1] /= rhs;
    p_data[2] /= rhs;
    p_data[3] /= rhs;
  }
};

auto spinor_u(const LVector<double> &, double, int) -> DiracWf<FermionFlow::In>;
auto spinor_v(const LVector<double> &, double, int) -> DiracWf<FermionFlow::In>;
auto spinor_ubar(const LVector<double> &, double, int)
    -> DiracWf<FermionFlow::Out>;
auto spinor_vbar(const LVector<double> &, double, int)
    -> DiracWf<FermionFlow::Out>;

auto spinor_u(const LVector<double> &, double)
    -> std::array<DiracWf<FermionFlow::In>, 2>;
auto spinor_v(const LVector<double> &, double)
    -> std::array<DiracWf<FermionFlow::In>, 2>;
auto spinor_ubar(const LVector<double> &, double)
    -> std::array<DiracWf<FermionFlow::Out>, 2>;
auto spinor_vbar(const LVector<double> &, double)
    -> std::array<DiracWf<FermionFlow::Out>, 2>;

auto spinor_u(DiracWf<FermionFlow::In> *, const LVector<double> &, double, int)
    -> void;
auto spinor_v(DiracWf<FermionFlow::In> *, const LVector<double> &, double, int)
    -> void;
auto spinor_ubar(DiracWf<FermionFlow::Out> *, const LVector<double> &, double,
                 int) -> void;
auto spinor_vbar(DiracWf<FermionFlow::Out> *, const LVector<double> &, double,
                 int) -> void;

auto charge_conjugate(const DiracWf<FermionFlow::In> &)
    -> DiracWf<FermionFlow::Out>;
auto charge_conjugate(const DiracWf<FermionFlow::Out> &)
    -> DiracWf<FermionFlow::In>;

// ===========================================================================
// ---- Vector Wavefunction --------------------------------------------------
// ===========================================================================

class VectorWf {
public:
  using ValueType = std::complex<double>;
  using WfType = LVector<std::complex<double>>;
  using MomentumType = LVector<double>;

private:
  WfType p_data;
  MomentumType p_momentum;

public:
  VectorWf(const WfType &wf, const MomentumType &p)
      : p_data(wf), p_momentum(p) {}
  VectorWf() : p_data(WfType{}), p_momentum(MomentumType{}) {}

  auto operator[](unsigned int i) const -> const ValueType & {
    return p_data[i];
  }
  auto operator[](unsigned int i) -> ValueType & { return p_data[i]; }

  [[nodiscard]] auto wavefunction() const -> const WfType & { return p_data; }
  auto wavefunction() -> WfType & { return p_data; }

  [[nodiscard]] auto momentum() const -> const MomentumType & {
    return p_momentum;
  }
  auto momentum() -> MomentumType & { return p_momentum; }

  [[nodiscard]] auto momentum(unsigned int i) const -> const double & {
    return p_momentum[i];
  }
  auto momentum(unsigned int i) -> double & { return p_momentum[i]; }

  static auto momentum_flow() -> double { return 1; }

  [[nodiscard]] auto momentum_with_flow() const -> MomentumType {
    return momentum_flow() * p_momentum;
  }

  friend auto operator<<(std::ostream &os, const VectorWf &wf)
      -> std::ostream & {
    os << "VectorWf(" << wf[0] << ", " << wf[1] << ", " << wf[2] << ", "
       << wf[3] << ")";
    return os;
  }

  template <typename T> auto operator*=(const T &rhs) -> void {
    p_data[0] *= rhs;
    p_data[1] *= rhs;
    p_data[2] *= rhs;
    p_data[3] *= rhs;
  }

  template <typename T> auto operator/=(const T &rhs) -> void {
    p_data[0] /= rhs;
    p_data[1] /= rhs;
    p_data[2] /= rhs;
    p_data[3] /= rhs;
  }
};

/**
 * Fill in the vector wavefunction given the momentum, mass, spin and if the
 * vector is in the initial or final state.
 * @param wf wavefunction structure to fill
 * @param k four-momentum of the state
 * @param m mass of the state
 * @param spin spin of the state. Can be -1, 0, or 1.
 * @param final-state if `FinalState`, then polarization is conjugated
 */
auto vector_wf(VectorWf *wf, const LVector<double> &, double, int, InOut)
    -> void;

/**
 * Create a vector wavefunction given the momentum, mass, spin and if the
 * vector is in the initial or final state.
 * @param k four-momentum of the state
 * @param m mass of the state
 * @param spin spin of the state. Can be -1, 0, or 1.
 * @param final-state if `FinalState`, then polarization is conjugated
 */
auto vector_wf(const LVector<double> &, double, int, InOut) -> VectorWf;

auto vector_wf(const LVector<double> &, double, InOut)
    -> std::array<VectorWf, 3>;

auto vector_wf(const LVector<double> &, InOut) -> std::array<VectorWf, 2>;

} // namespace blackthorn

#endif // BLACKTHORN_WAVEFUNCTIONS_HPP
