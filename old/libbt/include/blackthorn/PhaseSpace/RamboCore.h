#ifndef BLACKTHORN_PHASE_SPACE_RAMBO_CORE_H
#define BLACKTHORN_PHASE_SPACE_RAMBO_CORE_H

#include "blackthorn/Tensors.h"
#include <array>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <numeric>
#include <random>

namespace blackthorn::rambo_impl {

template <size_t N> using MomentaType = std::array<LVector<double>, N>;

template <size_t N> auto massless_weight(double cme) -> double {
  using boost::math::factorial;
  using boost::math::pow;

  static_assert(N >= 2, "Number of final-state particles must be >= 2.");

  static constexpr double k1_2PI = 1.591'549'430'918'953'5e-1;
  static constexpr double PRE_2 = 3.978'873'577'297'383'6e-2;
  static constexpr double PRE_3 = 1.259'825'563'796'855e-4;
  static constexpr double PRE_4 = 1.329'656'430'278'884e-7;
  static constexpr double PRE_5 = 7.016'789'757'994'902e-11;

  switch (N) {
  case 2:
    return PRE_2;
  case 3:
    return PRE_3 * pow<2>(cme);
  case 4:
    return PRE_4 * pow<4>(cme);
  case 5:
    return PRE_5 * pow<6>(cme);
  default:
    break;
  }

  const double wgt =
      pow<N - 1>(M_PI_2) * pow<2 * N - 4>(cme) * pow<3 * N - 4>(k1_2PI);

  // Compute (n-1)! * (n-2)!
  const auto fact_nm2 = factorial<double>(N - 2);
  const double fact = 1.0 / static_cast<double>((N - 1) * fact_nm2 * fact_nm2);
  return fact * wgt;
}

// Generate a `n` random four-momenta with energies distributated according to
// E⋅exp(-E)
template <size_t N> auto initialize_momenta(MomentaType<N> *momenta) -> void {
  static thread_local std::random_device rd{};
  static thread_local std::mt19937 generator{rd()};
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  // 2*pi
  static constexpr double k2PI = 6.2831853071795865;

#pragma unroll N
  for (auto &p : *momenta) {
    const double rho1 = distribution(generator);
    const double rho2 = distribution(generator);
    const double rho3 = distribution(generator);
    const double rho4 = distribution(generator);

    const double ctheta = 2 * rho1 - 1;
    const double stheta = sqrt(1 - ctheta * ctheta);
    const double phi = k2PI * rho2;
    const double e = -log(rho3 * rho4);

    p[0] = e;
    p[1] = e * stheta * cos(phi);
    p[2] = e * stheta * sin(phi);
    p[3] = e * ctheta;
  }
}

template <size_t N>
auto boost_momenta(MomentaType<N> *momenta, double cme) -> void {
  const LVector<double> sum_qs =
      std::accumulate(momenta->begin(), momenta->end(), LVector<double>{});

  const double invmass = 1.0 / lnorm(sum_qs);
  // boost vector
  const double bx = -invmass * sum_qs[1];
  const double by = -invmass * sum_qs[2];
  const double bz = -invmass * sum_qs[3];
  // boost factors
  const double x = cme * invmass;
  const double g = sum_qs[0] * invmass;
  const double a = 1.0 / (1.0 + g);

#pragma unroll N
  for (auto &p : *momenta) {
    const double bdotq = bx * p[1] + by * p[2] + bz * p[3];
    const double fact = std::fma(a, bdotq, p[0]);

    p[0] = x * std::fma(g, p[0], bdotq);
    p[1] = x * std::fma(fact, bx, p[1]);
    p[2] = x * std::fma(fact, by, p[2]);
    p[3] = x * std::fma(fact, bz, p[3]);
  }
}

template <size_t N>
auto compute_scale_factor(const MomentaType<N> &ps, double cme,
                          const std::array<double, N> &ms) -> double {
  using tools::sqr;
  static constexpr size_t max_iter = 50;
  static constexpr double tol = 1e-10;
  // initial guess
  const double msum = std::accumulate(ms.begin(), ms.end(), 0);
  double xi = sqrt(1.0 - sqr(msum / cme));

  size_t itercount = 0;
  while (true) {
    double f = -cme;
    double df = 0.0;
    // Compute residual and derivative
#pragma unroll N
    for (size_t i = 0; i < N; i++) {
      const double e = ps[i][0];
      const double deltaf = std::hypot(ms[i], xi * e);
      f += deltaf;
      df += xi * sqr(e) / deltaf;
    }

    // Newton correction
    const double deltaxi = -f / df;
    xi += deltaxi;

    itercount += 1;
    if (std::abs(deltaxi) < tol || itercount > max_iter) {
      break;
    }
  }
  return xi;
}

template <size_t N>
auto correct_masses(MomentaType<N> *ps, double cme,
                    const std::array<double, N> &ms) -> void {
  const double xi = compute_scale_factor(*ps, cme, ms);
#pragma unroll N
  for (size_t i = 0; i < N; i++) {
    ps->at(i)[0] = std::hypot(ms[i], xi * ps->at(i)[0]);
    ps->at(i)[1] *= xi;
    ps->at(i)[2] *= xi;
    ps->at(i)[3] *= xi;
  }
}

template <size_t N>
auto wgt_rescale_factor(const MomentaType<N> &ps, double cme) -> double {
  using tools::sqr;
  double t1 = 0.0;
  double t2 = 0.0;
  double t3 = 1.0;
#pragma unroll N
  for (const auto &p : ps) {
    const double modsqr = lnorm3_sqr(p);
    const double mod = sqrt(modsqr);
    const double inveng = 1.0 / p[0];
    t1 += mod / cme;
    t2 += modsqr * inveng;
    t3 *= mod * inveng;
  }
  t1 = pow(t1, static_cast<double>(2 * N - 3));
  t2 = 1.0 / t2;
  return t1 * t2 * t3 * cme;
}

// Fill momenta in center-of-mass frame with center-of-mass energy `cme`
// and `masses`.
template <size_t N>
auto generate_momenta(MomentaType<N> *momenta, double cme,
                      const std::array<double, N> &ms) -> void {
  initialize_momenta(momenta);
  boost_momenta(momenta, cme);
  correct_masses(momenta, cme, ms);
}

template <size_t N, class MSqrd>
auto generate_wgt(MSqrd msqrd, MomentaType<N> *momenta, double cme,
                  const std::array<double, N> &ms, double base_wgt) -> double {
  generate_momenta(momenta, cme, ms);
  return wgt_rescale_factor(*momenta, cme) * msqrd(*momenta) * base_wgt;
}

} // namespace blackthorn::rambo_impl

#endif // BLACKTHORN_PHASE_SPACE_RAMBO_CORE_H
