#include "blackthorn/Tensors.h"
#include "blackthorn/Tools.h"
#include "blackthorn/Wavefunctions.h"

namespace blackthorn {

using cmplx = std::complex<double>;

// ===========================================================================
// ---- Dirac Wavefunction ---------------------------------------------------
// ===========================================================================

static auto check_spin(int spin) -> void {
  if (spin != 1 && spin != -1) {

    throw std::invalid_argument("Invalid fermion spin. Must be -1 or 1, got: " +
                                std::to_string(spin));
  }
}

auto spinor_u(DiracWf<FlowIn> *wf, const LVector<double> &p, double mass,
              int spin) -> void {
  check_spin(spin);
  const auto e = p[0];
  const auto px = p[1];
  const auto py = p[2];
  const auto pz = p[3];
  const auto pm = std::hypot(px, py, pz);
  const auto wp = std::sqrt(std::abs(e + pm));
  const auto wm = mass / wp;
  const auto lam = static_cast<double>(spin);

  cmplx x1;
  cmplx x2;

  if (tools::zero_or_subnormal(pm + pz)) {
    x1 = cmplx{lam, 0};
    x2 = cmplx{0, 0};
  } else {
    const auto den = 1.0 / sqrt(2 * pm * (pm + pz));
    x1 = cmplx{lam * px * den, py * den};
    x2 = cmplx{(pm + pz) * den, 0};
  }

  if (spin == 1) {
    (*wf)[0] = wm * x2;
    (*wf)[1] = wm * x1;
    (*wf)[2] = wp * x2;
    (*wf)[3] = wp * x1;
  } else {
    (*wf)[0] = wp * x1;
    (*wf)[1] = wp * x2;
    (*wf)[2] = wm * x1;
    (*wf)[3] = wm * x2;
  }
  wf->momentum()[0] = e;
  wf->momentum()[1] = px;
  wf->momentum()[2] = py;
  wf->momentum()[3] = pz;
}

auto spinor_v(DiracWf<FlowIn> *wf, const LVector<double> &p, double mass,
              int spin) -> void {
  check_spin(spin);
  const auto e = p[0];
  const auto px = p[1];
  const auto py = p[2];
  const auto pz = p[3];
  const auto pm = std::hypot(px, py, pz);
  const auto wp = std::sqrt(std::abs(e + pm));
  const auto wm = mass / wp;
  const auto pmz = std::abs(pm + pz);
  const auto lam = static_cast<double>(spin);
  const auto isz = pmz < std::numeric_limits<double>::epsilon();

  cmplx x1;
  cmplx x2;

  if (isz) {
    x1 = cmplx{-lam, 0};
    x2 = cmplx{0, 0};
  } else {
    const auto den = 1.0 / sqrt(2 * pm * pmz);
    x1 = cmplx{-lam * px * den, py * den};
    x2 = cmplx{(pm + pz) * den, 0};
  }

  if (spin == 1) {
    (*wf)[0] = -wp * x1;
    (*wf)[1] = -wp * x2;
    (*wf)[2] = wm * x1;
    (*wf)[3] = wm * x2;
  } else {
    (*wf)[0] = wm * x2;
    (*wf)[1] = wm * x1;
    (*wf)[2] = -wp * x2;
    (*wf)[3] = -wp * x1;
  }
  wf->momentum()[0] = -e;
  wf->momentum()[1] = -px;
  wf->momentum()[2] = -py;
  wf->momentum()[3] = -pz;
}

auto spinor_ubar(DiracWf<FlowOut> *wf, const LVector<double> &p, double mass,
                 int spin) -> void {
  check_spin(spin);
  const auto e = p[0];
  const auto px = p[1];
  const auto py = p[2];
  const auto pz = p[3];
  const auto pm = std::hypot(px, py, pz);
  const auto wp = std::sqrt(std::abs(e + pm));
  const auto wm = mass / wp;
  const auto pmz = std::abs(pm + pz);
  const auto lam = static_cast<double>(spin);
  const auto isz = pmz < std::numeric_limits<double>::epsilon();

  cmplx x1;
  cmplx x2;

  if (isz) {
    x1 = cmplx{lam, 0};
    x2 = cmplx{0, 0};
  } else {
    const auto den = 1.0 / sqrt(2 * pm * pmz);
    x1 = cmplx{lam * px * den, -py * den};
    // x1 = cmplx{lam * px * den, py * den};
    x2 = cmplx{(pm + pz) * den, 0};
  }

  if (spin == 1) {
    (*wf)[0] = wp * x2;
    (*wf)[1] = wp * x1;
    (*wf)[2] = wm * x2;
    (*wf)[3] = wm * x1;
  } else {
    (*wf)[0] = wm * x1;
    (*wf)[1] = wm * x2;
    (*wf)[2] = wp * x1;
    (*wf)[3] = wp * x2;
  }
  wf->momentum()[0] = e;
  wf->momentum()[1] = px;
  wf->momentum()[2] = py;
  wf->momentum()[3] = pz;
}

auto spinor_vbar(DiracWf<FlowOut> *wf, const LVector<double> &p, double mass,
                 int spin) -> void {
  check_spin(spin);
  const auto e = p[0];
  const auto px = p[1];
  const auto py = p[2];
  const auto pz = p[3];
  const auto pm = std::hypot(px, py, pz);
  const auto wp = std::sqrt(std::abs(e + pm));
  const auto wm = mass / wp;
  const auto pmz = std::abs(pm + pz);
  const auto lam = static_cast<double>(spin);
  const auto isz = pmz < std::numeric_limits<double>::epsilon();

  cmplx x1;
  cmplx x2;

  if (isz) {
    x1 = cmplx{-lam, 0};
    x2 = cmplx{0, 0};
  } else {
    const auto den = 1.0 / sqrt(2 * pm * pmz);
    x1 = cmplx{-lam * px * den, -py * den};
    // x1 = cmplx{-lam * px * den, py * den};
    x2 = cmplx{(pm + pz) * den, 0};
  }

  if (spin == 1) {
    (*wf)[0] = wm * x1;
    (*wf)[1] = wm * x2;
    (*wf)[2] = -wp * x1;
    (*wf)[3] = -wp * x2;
  } else {
    (*wf)[0] = -wp * x2;
    (*wf)[1] = -wp * x1;
    (*wf)[2] = wm * x2;
    (*wf)[3] = wm * x1;
  }
  wf->momentum()[0] = -e;
  wf->momentum()[1] = -px;
  wf->momentum()[2] = -py;
  wf->momentum()[3] = -pz;
}

auto spinor_u(const LVector<double> &p, double mass, int spin)
    -> DiracWf<FlowIn> {
  DiracWf<FlowIn> wf{};
  spinor_u(&wf, p, mass, spin);
  return wf;
}

auto spinor_v(const LVector<double> &p, double mass, int spin)
    -> DiracWf<FlowIn> {
  DiracWf<FlowIn> wf{};
  spinor_v(&wf, p, mass, spin);
  return wf;
}

auto spinor_ubar(const LVector<double> &p, double mass, int spin)
    -> DiracWf<FlowOut> {
  DiracWf<FlowOut> wf{};
  spinor_ubar(&wf, p, mass, spin);
  return wf;
}

auto spinor_vbar(const LVector<double> &p, double mass, int spin)
    -> DiracWf<FlowOut> {
  DiracWf<FlowOut> wf{};
  spinor_vbar(&wf, p, mass, spin);
  return wf;
}

auto spinor_u(const LVector<double> &p, double mass)
    -> std::array<DiracWf<FlowIn>, 2> {
  return {spinor_u(p, mass, 1), spinor_u(p, mass, -1)};
}

auto spinor_v(const LVector<double> &p, double mass)
    -> std::array<DiracWf<FlowIn>, 2> {
  return {spinor_v(p, mass, 1), spinor_v(p, mass, -1)};
}

auto spinor_ubar(const LVector<double> &p, double mass)
    -> std::array<DiracWf<FlowOut>, 2> {
  return {spinor_ubar(p, mass, 1), spinor_ubar(p, mass, -1)};
}

auto spinor_vbar(const LVector<double> &p, double mass)
    -> std::array<DiracWf<FlowOut>, 2> {
  return {spinor_vbar(p, mass, 1), spinor_vbar(p, mass, -1)};
}

// ===========================================================================
// ---- Charge Conjugation ---------------------------------------------------
// ===========================================================================

auto charge_conjugate(const DiracWf<FermionFlow::In> &fi)
    -> DiracWf<FermionFlow::Out> {
  return DiracWf<FermionFlow::Out>{{fi[1], -fi[0], -fi[3], fi[2]},
                                   -fi.momentum()};
}

auto charge_conjugate(const DiracWf<FermionFlow::Out> &fo)
    -> DiracWf<FermionFlow::In> {
  return DiracWf<FermionFlow::In>{{-fo[1], fo[0], fo[3], -fo[2]},
                                  -fo.momentum()};
}
} // namespace blackthorn
