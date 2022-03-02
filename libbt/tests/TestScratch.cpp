#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "Pythia8/Pythia.h"
#include "Tools.h"
#include "blackthorn/Models/Particles.h"
#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/PhaseSpace/RamboCore.h"
#include "blackthorn/Spectra/Quad.h"
#include "blackthorn/Tools.h"
#include "boost/math/quadrature/gauss_kronrod.hpp"
#include <catch2/catch.hpp>
#include <iostream>
#include <random>

namespace bt = blackthorn;

template <class MSqrd> class ThreeBodyPhaseSpace { // NOLINT
  using Momentum = bt::LVector<double>;
  using Momenta = std::array<Momentum, 3>;

  MSqrd p_msqrd;
  double p_m0;
  double p_m1;
  double p_m2;
  double p_m3;

public:
  ThreeBodyPhaseSpace(MSqrd msqrd, double m0, double m1, double m2, double m3)
      : p_msqrd(msqrd), p_m0(m0), p_m1(m1), p_m2(m2), p_m3(m3) {}

  auto tbounds(double s) -> std::pair<double, double>;
  auto fill_momenta(double s, double t, Momenta *momenta) -> void;
  auto rotate(double tx, double ty, double tz, Momenta *momenta) -> void;
  auto integrate_t(double s, bool use_boost) -> double;
};

// clang-format off

static double rescale_error(double err, double res_abs, double res_asc);

template <size_t N, class F>
auto qk(F f, double a, double b, const std::array<double, N> &xgk, const std::array<double, N> &wgk, const std::array<double, N / 2> &wg, double *abserr, double *res_abs, double *res_asc) -> double;

template <class F>
auto qk15(F f, double a, double b, double *abserr, double *resabs, double *resasc) -> double;

// clang-format on

auto msqrd_mu_to_e_nue_numu(const std::array<bt::LVector<double>, 3> &momenta)
    -> double {
  constexpr double mmu = bt::Muon::mass;
  constexpr double me = bt::Electron::mass;
  constexpr double gf = bt::StandardModel::g_fermi;
  const double t = bt::lnorm_sqr(momenta[0] + momenta[2]);
  return 16 * gf * gf * (mmu * mmu - t) * (t - me * me);
}

// TEST_CASE("Three") {
//   constexpr double mmu = bt::Muon::mass;
//   constexpr double me = bt::Electron::mass;
//   const double smin = 1e-4;
//   const double smax = bt::tools::sqr(mmu - me);
//   auto ss = bt::tools::linspace(smin, smax, 100);

//   ThreeBodyPhaseSpace tbps(msqrd_mu_to_e_nue_numu, mmu, me, 0.0, 0.0);

//   const double s = ss[40];
//   std::array<bt::LVector<double>, 3> momenta{};
//   auto tbs = tbps.tbounds(s);
//   const double t = (tbs.second - tbs.first) / 2.0;
//   tbps.fill_momenta(s, t, &momenta);
//   for (size_t i = 0; i < 3; ++i) {
//     std::cout << momenta[i].e() << ", " << momenta[i].px() << ", "
//               << momenta[i].py() << ", " << momenta[i].pz() << "\n";
//   }
//   std::cout << s << ", " << bt::lnorm_sqr(momenta[1] + momenta[2]) << "\n";
//   std::cout << t << ", " << bt::lnorm_sqr(momenta[0] + momenta[2]) << "\n";

//   auto p = momenta[0] + momenta[1] + momenta[2];
//   std::cout << p.e() << ", " << p.px() << ", " << p.py() << ", " << p.pz()
//             << "\n";
// }

TEST_CASE("Muon Decay", "") {
  constexpr double mmu = bt::Muon::mass;
  constexpr double me = bt::Electron::mass;
  constexpr size_t NPTS = 100;
  const double smin = 1e-4;
  const double smax = bt::tools::sqr(mmu - me);
  auto ss = bt::tools::linspace(smin, smax * 0.99, NPTS);

  ThreeBodyPhaseSpace tbps(msqrd_mu_to_e_nue_numu, mmu, me, 0.0, 0.0);

  SECTION("Boost Vs. QNG") {
    std::array<double, NPTS> bgk{};
    std::array<double, NPTS> qng{};

    auto do_boost = [&]() {
      for (size_t i = 0; i < NPTS; ++i) {
        bgk[i] = tbps.integrate_t(ss[i], true);
      }
    };
    auto do_qng = [&]() {
      for (size_t i = 0; i < NPTS; ++i) {
        qng[i] = tbps.integrate_t(ss[i], false);
      }
    };

    BENCHMARK("Boost") { do_boost(); };
    BENCHMARK("QNG") { do_qng(); };

    for (size_t i = 0; i < NPTS; ++i) {
      std::cout << ss[i] << ", " << qng[i] << ", " << bgk[i] << "\n";
    }
  }
}

template <class MSqrd>
auto ThreeBodyPhaseSpace<MSqrd>::integrate_t(double s, bool use_boost)
    -> double {
  using bt::FixedGaussKronrod;
  using bt::LVector;
  std::array<LVector<double>, 3> momenta;
  auto tbs = tbounds(s);
  const double pre = 1.0 / (256.0 * pow(p_m0 * M_PI, 3));

  auto f = [&](double t) {
    fill_momenta(s, t, &momenta);
    return p_msqrd(momenta);
  };

  if (use_boost) {
    using boost::math::quadrature::gauss_kronrod;
    return pre * gauss_kronrod<double, 15>::integrate(f, tbs.first, tbs.second,
                                                      5, 1e-7);
  }

  return pre *
         FixedGaussKronrod::qk(f, tbs.first, tbs.second, 1e-7, 1e-3).result;
}

template <class MSqrd>
auto ThreeBodyPhaseSpace<MSqrd>::tbounds(double s)
    -> std::pair<double, double> {
  using bt::tools::kallen_lambda;
  const double m02 = p_m0 * p_m0;
  const double m12 = p_m1 * p_m1;
  const double m22 = p_m2 * p_m2;
  const double m32 = p_m3 * p_m3;
  const double m2sum = m02 + m12 + m22 + m32;

  const double p01 = 0.5 * sqrt(kallen_lambda(s, m02, m12) / s);
  const double p23 = 0.5 * sqrt(kallen_lambda(s, m22, m32) / s);

  const double f = 0.5 * (-s + m2sum - (m02 - m12) * (m22 - m32) / s);
  const double g = 2.0 * p01 * p23;

  return {f - g, f + g};
}

template <class MSqrd>
auto ThreeBodyPhaseSpace<MSqrd>::fill_momenta(double s, double t,
                                              Momenta *momenta) -> void {
  using bt::tools::energy_one_cm;
  using bt::tools::sqr;
  using bt::tools::two_body_three_momentum;

  static thread_local std::random_device rd{};
  static thread_local std::mt19937 generator{rd()};
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  const double m02 = p_m0 * p_m0;
  const double m12 = p_m1 * p_m1;
  const double m22 = p_m2 * p_m2;
  const double m32 = p_m3 * p_m3;

  const double e1 = (m02 + m12 - s) / (2 * p_m0);
  const double e2 = (m02 + m22 - t) / (2 * p_m0);
  const double e3 = p_m0 - e1 - e2;

  const double p1 = std::sqrt(sqr(e1) - m12);
  const double p2 = std::sqrt(sqr(e2) - m22);
  const double p3 = std::sqrt(sqr(e3) - m32);

  // Put p1 along z-axis
  momenta->at(0).e() = e1;
  momenta->at(0).px() = 0.0;
  momenta->at(0).py() = 0.0;
  momenta->at(0).pz() = p1;

  const double zmax = std::min(1.0, (p1 + p3) / p2);
  const double zmin = std::max((p1 - p3) / p2, -1.0);
  const double z = (zmax - zmin) * distribution(generator) + zmin;
  const double phi = 2.0 * M_PI * distribution(generator);

  momenta->at(1).e() = e2;
  momenta->at(1).px() = p2 * cos(phi) * sqrt(1.0 - z * z);
  momenta->at(1).py() = p2 * sin(phi) * sqrt(1.0 - z * z);
  momenta->at(1).pz() = p2 * z;

  momenta->at(2) = -momenta->at(1) - momenta->at(0);
  momenta->at(2).e() += p_m0;

  // Rotate by arbitrary angles
  const double tx = 2.0 * M_PI * distribution(generator);
  const double ty = 2.0 * M_PI * distribution(generator);
  const double tz = 2.0 * M_PI * distribution(generator);

  // rotate(tx, ty, tz, momenta);
}

template <class MSqrd>
auto ThreeBodyPhaseSpace<MSqrd>::rotate(double tx, double ty, double tz,
                                        Momenta *momenta) -> void {
  const double cx = cos(tx);
  const double cy = cos(ty);
  const double cz = cos(tz);

  const double sx = sin(tx);
  const double sy = sin(ty);
  const double sz = sin(tz);
#pragma unroll 3
  for (size_t i = 0; i < 3; ++i) {
    const double px = momenta->at(i).px();
    const double py = momenta->at(i).py();
    const double pz = momenta->at(i).pz();

    momenta->at(i).px() = px * cy * cz + cx * (pz * cz * sy - py * sz) +
                          sx * (py * cz * sy + pz * sz);
    momenta->at(i).py() = -(pz * cz * sx) + (px * cy + py * sx * sy) * sz +
                          cx * (py * cz + pz * sy * sz);
    momenta->at(i).pz() = pz * cx * cy + py * cy * sx - px * sy;
  }
}

double rescale_error(double err, double res_abs, double res_asc) {

  double scaled_err = std::abs(err);

  if (!(res_asc == 0) && !(scaled_err == 0.0)) {
    const double scale = std::pow(200.0 * scaled_err / res_asc, 1.5);
    scaled_err = scale < 1.0 ? res_asc * scale : res_asc;
  }

  if (res_abs > std::numeric_limits<double>::min() /
                    (50.0 * std::numeric_limits<double>::epsilon())) {
    const double min_err =
        50.0 * std::numeric_limits<double>::epsilon() * res_abs;
    if (min_err > scaled_err) {
      scaled_err = min_err;
    }
  }
  return scaled_err;
}

template <size_t N, class F>
auto qk(F f, double a, double b, const std::array<double, N> &xgk,
        const std::array<double, N> &wgk, const std::array<double, N / 2> &wg,
        double *abserr, double *res_abs, double *res_asc) -> double {

  static constexpr size_t N1 = (N - 1) / 2;
  static constexpr size_t N2 = N / 2;

  std::array<double, N - 1> fv1{};
  std::array<double, N - 1> fv2{};

  const double center = 0.5 * (a + b);
  const double half_len = 0.5 * (b - a);
  const double abs_half_len = std::abs(half_len);
  const double f_center = f(center);

  double res_gauss = 0.0;
  double res_kronrod = f_center * wgk[N - 1];

  *res_abs = std::abs(res_kronrod);
  *res_asc = 0.0;

  if (N % 2 == 0) {
    res_gauss = f_center * wg[N / 2 - 1];
  }
#pragma unroll((N - 1) / 2 + 1)
  for (size_t j = 0; j < (N - 1) / 2; ++j) {
    const auto jtw = j * 2 + 1;
    const auto abscissa = half_len * xgk[jtw];
    const auto fval1 = f(center - abscissa);
    const auto fval2 = f(center + abscissa);
    const auto fsum = fval1 + fval2;
    fv1[jtw] = fval1;
    fv2[jtw] = fval2;

    res_gauss += wg[j] * fsum;
    res_kronrod += wgk[jtw] * fsum;
    *res_abs += wgk[jtw] * (std::abs(fval1) + std::abs(fval2));
  }

#pragma unroll(N / 2 + 1)
  for (size_t j = 0; j < (N / 2); ++j) {
    const auto jtwm1 = j * 2;
    const auto abscissa = half_len * xgk[jtwm1];
    const auto fval1 = f(center - abscissa);
    const auto fval2 = f(center + abscissa);
    fv1[jtwm1] = fval1;
    fv2[jtwm1] = fval2;
    res_kronrod += wgk[jtwm1] * (fval1 + fval2);
    *res_abs += wgk[jtwm1] * (std::abs(fval1) + std::abs(fval2));
  }

  const double mean = res_kronrod * 0.5;
  *res_asc = wgk[N - 1] * std::abs(f_center - mean);

#pragma unroll N
  for (size_t j = 0; j < N - 1; ++j) {
    *res_asc += wgk[j] * (std::abs(fv1[j] - mean) + std::abs(fv2[j] - mean));
  }

  // scale by the width of the integration region
  const double err = (res_kronrod - res_gauss) * half_len;

  res_kronrod *= half_len;
  *res_abs *= abs_half_len;
  *res_asc *= abs_half_len;

  *abserr = rescale_error(err, *res_abs, *res_asc);

  return res_kronrod;
}

template <class F>
auto qk15(F f, double a, double b, double *abserr, double *resabs,
          double *resasc) -> double {
  // gauss quadrature weights and kronron quadrature abscissae and weights
  // as evaluated with 80 decimal digit arithmetic by l. w. fullerton,
  // bell labs, nov. 1981.

  static const std::array<double, 8> XGK = {
      0.99145537112081263921, 0.94910791234275852453, 0.86486442335976907279,
      0.74153118559939443986, 0.58608723546769113029, 0.40584515137739716691,
      0.20778495500789846760, 0.00000000000000000000};
  static const std::array<double, 8> WGK = {
      0.02293532201052922496, 0.06309209262997855329, 0.10479001032225018384,
      0.14065325971552591875, 0.16900472663926790283, 0.19035057806478540991,
      0.20443294007529889241, 0.20948214108472782801};
  static const std::array<double, 4> WG = {
      0.12948496616886969327, 0.27970539148927666790, 0.38183005050511894495,
      0.41795918367346938776};

  return qk<8, F>(f, a, b, XGK, WGK, WG, abserr, resabs, resasc);
}
