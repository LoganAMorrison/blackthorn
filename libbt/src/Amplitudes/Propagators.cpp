#include "blackthorn/Amplitudes.h"
#include "blackthorn/Tools.h"
#include "blackthorn/Wavefunctions.h"

namespace blackthorn {

auto Propagator::cmplx_mass_sqr(double mass, double width)
    -> std::complex<double> {
  using tools::im;
  using tools::sqr;
  return sqr(mass) - im * mass * width;
}

auto Propagator::propagator_den(const LVector<double> &p, double mass,
                                double width) -> std::complex<double> {
  using tools::im;
  using tools::sqr;
  return tools::im / (lnorm_sqr(p) - sqr(mass) + im * mass * width);
}

auto Propagator::propagator_den(const LVector<double> &p)
    -> std::complex<double> {
  using tools::im;
  using tools::sqr;
  return tools::im / lnorm_sqr(p);
}

template <>
auto Propagator::attach(ScalarWf *wf, double mass, double width) -> void {
  wf->wavefunction() *= propagator_den(wf->momentum(), mass, width);
}

template <> auto Propagator::attach(ScalarWf *wf) -> void {
  wf->wavefunction() *= propagator_den(wf->momentum());
}

template <> auto Propagator::attach(DiracWf<FermionFlow::In> *psi) -> void {
  using tools::im;

  const auto e = psi->momentum(0);
  const auto px = psi->momentum(1);
  const auto py = psi->momentum(2);
  const auto pz = psi->momentum(3);

  const auto psi0 = (*psi)[0];
  const auto psi1 = (*psi)[1];
  const auto psi2 = (*psi)[2];
  const auto psi3 = (*psi)[3];

  const auto p = psi->momentum();
  const auto den = propagator_den(p);

  (*psi)[0] = den * ((e - pz) * psi2 - (px - im * py) * psi3);
  (*psi)[1] = den * (-((px + im * py) * psi2) + (e + pz) * psi3);
  (*psi)[2] = den * ((e + pz) * psi0 + (px - im * py) * psi1);
  (*psi)[3] = den * ((px + im * py) * psi0 + (e - pz) * psi1);
}

template <>
auto Propagator::attach(DiracWf<FermionFlow::In> *psi, double m, double width)
    -> void {
  using tools::im;

  const auto e = psi->momentum(0);
  const auto px = psi->momentum(1);
  const auto py = psi->momentum(2);
  const auto pz = psi->momentum(3);

  const auto psi0 = (*psi)[0];
  const auto psi1 = (*psi)[1];
  const auto psi2 = (*psi)[2];
  const auto psi3 = (*psi)[3];

  const auto p = psi->momentum();
  const auto den = propagator_den(p, m, width);

  (*psi)[0] = den * (m * psi0 + (e - pz) * psi2 - (px - im * py) * psi3);
  (*psi)[1] = den * (m * psi1 - (px + im * py) * psi2 + (e + pz) * psi3);
  (*psi)[2] = den * ((e + pz) * psi0 + (px - im * py) * psi1 + m * psi2);
  (*psi)[3] = den * ((px + im * py) * psi0 + (e - pz) * psi1 + m * psi3);
}

template <> auto Propagator::attach(DiracWf<FermionFlow::Out> *psi) -> void {
  using tools::im;

  const auto e = psi->momentum(0);
  const auto px = psi->momentum(1);
  const auto py = psi->momentum(2);
  const auto pz = psi->momentum(3);

  const auto psi0 = (*psi)[0];
  const auto psi1 = (*psi)[1];
  const auto psi2 = (*psi)[2];
  const auto psi3 = (*psi)[3];

  const auto p = psi->momentum();
  const auto den = propagator_den(p);

  (*psi)[0] = den * ((e + pz) * psi2 + (px + im * py) * psi3);
  (*psi)[1] = den * ((px - im * py) * psi2 + (e - pz) * psi3);
  (*psi)[2] = den * ((e - pz) * psi0 - (px + im * py) * psi1);
  (*psi)[3] = den * (-(px * psi0) + im * py * psi0 + (e + pz) * psi1);
}

template <>
auto Propagator::attach(DiracWf<FermionFlow::Out> *psi, double m, double width)
    -> void {
  using tools::im;

  const auto e = psi->momentum(0);
  const auto px = psi->momentum(1);
  const auto py = psi->momentum(2);
  const auto pz = psi->momentum(3);

  const auto psi0 = (*psi)[0];
  const auto psi1 = (*psi)[1];
  const auto psi2 = (*psi)[2];
  const auto psi3 = (*psi)[3];

  const auto p = psi->momentum();
  const auto den = propagator_den(p, m, width);

  (*psi)[0] = den * (m * psi0 + (e + pz) * psi2 + (px + im * py) * psi3);
  (*psi)[1] = den * (m * psi1 + (px - im * py) * psi2 + (e - pz) * psi3);
  (*psi)[2] = den * ((e - pz) * psi0 - (px + im * py) * psi1 + m * psi2);
  (*psi)[3] = den * (-((px - im * py) * psi0) + (e + pz) * psi1 + m * psi3);
}

template <>
auto Propagator::attach(VectorWf *eps, double m, double width) -> void {
  const auto p = eps->momentum();
  const auto wf = eps->wavefunction();

  const auto den = propagator_den(p, m, width);
  const auto invcm2 =
      m == 0.0 ? 0.0 : 1.0 / Propagator::cmplx_mass_sqr(m, width);
  eps->wavefunction() = (-wf + p * dot(p, wf) * invcm2) * den;
}

template <> auto Propagator::attach(VectorWf *eps) -> void {
  const auto p = eps->momentum();
  *eps *= -tools::im / lnorm_sqr(p);
}

} // namespace blackthorn
