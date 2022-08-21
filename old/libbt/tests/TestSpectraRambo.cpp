#define CATCH_CONFIG_MAIN

#include "blackthorn/Models/RhNeutrino.h"
#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/Spectra.h"
#include <catch2/catch.hpp>
#include <fmt/format.h>

namespace bt = blackthorn;

auto width_h_to_bb() -> double {
  static constexpr double MH = bt::Higgs::mass;
  static constexpr double MW = bt::WBoson::mass;
  static constexpr double MB = bt::BottomQuark::mass;
  static constexpr double r = MB / MH;
  static constexpr double EL = bt::StandardModel::qe;
  static constexpr double SW = bt::StandardModel::sw;

  return (pow(EL, 2) * pow(MH, 3) * pow(r, 2) * pow(1 - 4 * pow(r, 2), 1.5)) /
         (32. * pow(MW, 2) * M_PI * pow(SW, 2));
}

auto msqrd_h_to_bba(const bt::MomentaType<3> &p) -> double {
  using bt::tools::sqr;
  static constexpr double MH = bt::Higgs::mass;
  static constexpr double MW = bt::WBoson::mass;
  static constexpr double MB = bt::BottomQuark::mass;
  static constexpr double r2 = sqr(MB / MH);
  static constexpr double EL = bt::StandardModel::qe;
  static constexpr double SW = bt::StandardModel::sw;

  const double s = bt::lnorm_sqr(p[1] + p[2]);
  const double t = bt::lnorm_sqr(p[0] + p[2]);
  const double x = 1 - s / (MH * MH) + r2;
  const double y = 1 - t / (MH * MH) + r2;

  const double num =
      8 * pow(r2, 3) * pow(-2 + x + y, 2) +
      r2 * (-1 + x) * (-1 + y) *
          (2 + pow(x, 2) + 2 * x * (-1 + y) - 2 * y + pow(y, 2)) +
      sqr(r2) * (pow(x, 2) * (6 - 8 * y) + 2 * y * (-4 + 3 * y) -
                 4 * x * (2 - 5 * y + 2 * pow(y, 2)));
  const double den = pow(-1 + x, 2) * pow(-1 + y, 2);
  const double pre = sqr(sqr(EL) * MH) / sqr(3 * MW * SW);

  return pre * num / den;
}

auto msqrd_h_to_bba_h(const bt::MomentaType<3> &p) -> double {
  using bt::tools::sqr;
  static constexpr double MH = bt::Higgs::mass;
  static constexpr double MW = bt::WBoson::mass;
  static constexpr double MB = bt::BottomQuark::mass;
  static constexpr double r2 = sqr(MB / MH);
  static constexpr double EL = bt::StandardModel::qe;
  static constexpr double SW = bt::StandardModel::sw;

  auto ph = p[0] + p[1] + p[2];

  const auto wf_h = bt::scalar_wf(ph, bt::Incoming);
  const auto wf_b1s = bt::spinor_ubar(p[0], MB);
  const auto wf_b2s = bt::spinor_v(p[1], MB);
  const auto wf_as = bt::vector_wf(p[2], bt::Outgoing);

  bt::DiracWf<bt::FlowOut> wf_b1_off{};
  bt::DiracWf<bt::FlowIn> wf_b2_off{};

  const auto vbba = bt::StandardModel::feynman_rule_f_f_a<bt::BottomQuark>();
  const auto vbbh = bt::StandardModel::feynman_rule_f_f_h<bt::BottomQuark>();

  double msqrd = 0.0;
#pragma unroll 3
  for (const auto &wf_a : wf_as) {
#pragma unroll 2
    for (const auto &wf_b1 : wf_b1s) {
      bt::Current::generate(&wf_b1_off, vbba, MB, 0.0, wf_b1, wf_a);
#pragma unroll 2
      for (const auto &wf_b2 : wf_b2s) {
        bt::Current::generate(&wf_b2_off, vbba, MB, 0.0, wf_b2, wf_a);
        msqrd += std::norm(bt::amplitude(vbbh, wf_b1_off, wf_b2, wf_h) +
                           bt::amplitude(vbbh, wf_b1, wf_b2_off, wf_h));
      }
    }
  }
  return msqrd;
}

auto dndx_h_to_bba_analytic(double x) -> double {
  using bt::tools::sqr;
  const double r = bt::BottomQuark::mass / bt::Higgs::mass;
  const double r2 = sqr(r);
  const double rfac = 1 - 4 * r2;

  if (rfac < x || x < 0) {
    return 0.0;
  }

  const double pre = bt::StandardModel::alpha_em / (9 * M_PI * pow(rfac, 1.5));
  const double sqrtfac = sqrt((1 - x) * (rfac - x));
  const double split = 2 * rfac * sqrtfac;
  const double logpre = sqr(x) + 2 * rfac * (1 - 2 * r2 - x);
  const double logfac = log((1 - x + sqrtfac) / (1 - x - sqrtfac));
  return -pre * (split - logpre * logfac) / x;
}

TEST_CASE("H->b+bbar, analytic_msqrd") {
  const double whbb = width_h_to_bb();
  const double xmin = 1e-6;
  const double xmax = 0.99;
  std::array<double, 2> ms = {bt::BottomQuark::mass, bt::BottomQuark::mass};
  const auto xs = bt::tools::geomspace(xmin, xmax, 50);

  for (const auto &x : xs) {
    const double egam = bt::Higgs::mass * x / 2;
    const double dndx_a = dndx_h_to_bba_analytic(x);
    auto dndx_r = bt::photon_spectrum_rambo(msqrd_h_to_bba, egam,
                                            bt::Higgs::mass, ms, whbb, 50000);
    const double v0 = dndx_r.first * bt::Higgs::mass / 2;
    const double dv = dndx_r.second * bt::Higgs::mass / 2;
    const double v = dndx_a;
    // fmt::print("{: e} {: e} {: e} {: e}\n", x, dndx_a, dndx_r.first,
    //            dndx_a / dndx_r.first);

    CHECK(std::abs((v - v0) / v0) < std::max(2 * dv, 1e-1));
  }
}

TEST_CASE("H->b+bbar, helas msqrd") {
  const double whbb = width_h_to_bb();
  const double xmin = 1e-6;
  const double xmax = 0.99;
  std::array<double, 2> ms = {bt::BottomQuark::mass, bt::BottomQuark::mass};
  const auto xs = bt::tools::geomspace(xmin, xmax, 50);

  for (const auto &x : xs) {
    const double egam = bt::Higgs::mass * x / 2;
    const double dndx_a = dndx_h_to_bba_analytic(x);
    auto dndx_r = bt::photon_spectrum_rambo(msqrd_h_to_bba_h, egam,
                                            bt::Higgs::mass, ms, whbb, 50000);
    const double v0 = dndx_r.first * bt::Higgs::mass / 2;
    const double dv = dndx_r.second * bt::Higgs::mass / 2;
    const double v = dndx_a;
    // fmt::print("{: e} {: e} {: e} {: e}\n", x, dndx_a, dndx_r.first,
    //            dndx_a / dndx_r.first);

    CHECK(std::abs((v - v0) / v0) < std::max(2 * dv, 1e-1));
  }
}
