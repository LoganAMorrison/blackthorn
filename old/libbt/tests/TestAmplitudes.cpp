#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "Tools.h"
#include "blackthorn/Amplitudes.h"
#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/PhaseSpace.h"
#include "blackthorn/Tensors.h"
#include <algorithm>
#include <catch2/catch.hpp>
#include <fmt/format.h>
#include <numeric>

using namespace blackthorn; // NOLINT

auto width_t_to_bw_analytic() -> double;
auto width_t_to_bw() -> std::pair<double, double>;
auto width_mu_to_e_nu_nu_analytic() -> std::pair<double, double>;
auto width_mu_to_e_nu_nu() -> std::pair<double, double>;
auto cross_section_ee_to_mm_analytic(double cme) -> std::pair<double, double>;
auto cross_section_ee_to_mm(double cme) -> std::pair<double, double>;

TEST_CASE("Test Widths", "[widths]") { // NOLINT
  using tools::print_cross_section;
  using tools::print_width;
  using tools::UNICODE_MU;
  using tools::UNICODE_NU;

  SECTION("t -> b + W") {
    const auto res = width_t_to_bw();
    const auto res_an = width_t_to_bw_analytic();
    const auto frac_diff = fractional_diff(res_an, res.first);

    print_width("estimate", res.first, res.second, "t", {"b", "w"});
    print_width("actual", res_an, "t", {"b", "w"});
    fmt::print("frac. diff. (%) = {}\n", frac_diff * 100.0);

    CHECK(frac_diff <= 1.0);
    BENCHMARK("analytic msqrd") { return width_t_to_bw_analytic(); };
    BENCHMARK("helas msqrd") { return width_t_to_bw(); };
  }

  SECTION("mu -> e + nu + nu") {
    const auto res = width_mu_to_e_nu_nu();
    const auto res_an = width_mu_to_e_nu_nu_analytic();
    const auto frac_diff = fractional_diff(res_an.first, res.first);

    print_width("estimate", res.first, res.second, UNICODE_MU,
                {"e", "ve", "vm"});
    print_width("actual", res_an.first, res.second, UNICODE_MU,
                {"e", "ve", "vm"});
    print_fractional_diff(res_an.first, res.first);

    CHECK(frac_diff <= 1.0);
    BENCHMARK("analytic msqrd") { return width_mu_to_e_nu_nu_analytic(); };
    BENCHMARK("helas msqrd") { return width_mu_to_e_nu_nu(); };
  }

  SECTION("e + e -> m + m") {
    const auto res = cross_section_ee_to_mm(1e3);
    const auto res_an = cross_section_ee_to_mm_analytic(1e3);
    const auto frac_diff = fractional_diff(res_an.first, res.first);

    print_cross_section("estimate", res.first, res.second, "e", "e",
                        {UNICODE_MU, UNICODE_MU});
    print_cross_section("actual", res_an.first, res_an.second, "e", "e",
                        {UNICODE_MU, UNICODE_MU});
    print_fractional_diff(res_an.first, res.first);

    CHECK(frac_diff <= 1.0);
    BENCHMARK("analytic msqrd") {
      return cross_section_ee_to_mm_analytic(1e3);
    };
    BENCHMARK("helas msqrd") { return cross_section_ee_to_mm(1e3); };
  }
}

auto width_t_to_bw_analytic() -> double {
  using tools::kallen_lambda;
  return (pow(StandardModel::qe, 2) *
          (pow(BottomQuark::mass, 4) + pow(TopQuark::mass, 4) +
           pow(TopQuark::mass, 2) * pow(WBoson::mass, 2) -
           2 * pow(WBoson::mass, 4) +
           pow(BottomQuark::mass, 2) *
               (-2 * pow(TopQuark::mass, 2) + pow(WBoson::mass, 2))) *
          sqrt(kallen_lambda(pow(BottomQuark::mass, 2), pow(TopQuark::mass, 2),
                             pow(WBoson::mass, 2)))) /
         (64. * pow(TopQuark::mass, 3) * pow(WBoson::mass, 2) * M_PI *
          pow(StandardModel::sw, 2));
}

auto width_t_to_bw() -> std::pair<double, double> {
  const std::array<double, 2> fsp_masses{BottomQuark::mass, WBoson::mass};
  constexpr double mw = WBoson::mass;
  constexpr double mt = TopQuark::mass;
  constexpr double mb = BottomQuark::mass;
  auto pt = LVector<double>{mt, 0.0, 0.0, 0.0};

  auto twf = [](const LVector<double> &p, int spin) {
    return spinor_u(p, mt, spin);
  };
  auto bwf = [](const LVector<double> &p, int spin) {
    return spinor_ubar(p, mb, spin);
  };
  auto wwf = [](const LVector<double> &p, int spin) {
    return vector_wf(p, mw, spin, Outgoing);
  };

  const auto t_wfs = {twf(pt, 1), twf(pt, -1)};

  const auto msqrd = [&pt, &t_wfs, &wwf, &twf,
                      &bwf](const std::array<LVector<double>, 2> &ps) {
    auto pb = ps[0];
    auto pw = ps[1];

    const VertexFFV v{
        tools::im * StandardModel::qe / (M_SQRT2 * StandardModel::sw), 0.0};

    const auto b_wfs = {bwf(pb, 1), bwf(pb, -1)};
    const auto w_wfs = {wwf(pw, 1), wwf(pw, 0), wwf(pw, -1)};

    double msqrd = 0.0;
    for (const auto &t_wf : t_wfs) {
      for (const auto &b_wf : b_wfs) {
        for (const auto &w_wf : w_wfs) {
          msqrd += std::norm(amplitude(v, b_wf, t_wf, w_wf));
        }
      }
    }
    return msqrd / 2.0;
  };
  return Rambo<2>::decay_width(msqrd, TopQuark::mass, fsp_masses);
}

auto width_mu_to_e_nu_nu_analytic() -> std::pair<double, double> {
  const std::array<double, 3> fsp_masses{Electron::mass, 0.0, 0.0};
  const size_t nevents = 100'000;

  auto msqrd = [](const std::array<LVector<double>, 3> &ps) {
    const double s = lnorm_sqr(ps[1] + ps[2]);
    // const double t = lnorm_sqr(ps[0] + ps[2]);
    const double u = lnorm_sqr(ps[0] + ps[1]);
    return 0.5 *
           (pow(StandardModel::qe, 4) *
            (4 * pow(WBoson::mass, 4) * (pow(Muon::mass, 2) - s - u) * (s + u) +
             pow(Electron::mass, 4) *
                 (-pow(Muon::mass, 4) + pow(Muon::mass, 2) * u) +
             pow(Electron::mass, 2) *
                 (pow(Muon::mass, 4) * u + 4 * pow(WBoson::mass, 4) * (s + u) -
                  pow(Muon::mass, 2) *
                      (4 * pow(WBoson::mass, 4) + 4 * pow(WBoson::mass, 2) * s +
                       pow(u, 2))))) /
           (4. * pow(WBoson::mass, 4) * pow(StandardModel::sw, 4) *
            (pow(WBoson::mass, 4) + pow(u, 2) +
             pow(WBoson::mass, 2) * (-2 * u + pow(WBoson::width, 2))));
  };

  return Rambo<3>::decay_width(msqrd, Muon::mass, fsp_masses, nevents);
}

auto width_mu_to_e_nu_nu() -> std::pair<double, double> {
  const std::array<double, 3> fsp_masses{Electron::mass, 0.0, 0.0};
  const size_t nevents = 100'000;

  auto msqrd = [](const std::array<LVector<double>, 3> &ps) {
    auto pm = std::accumulate(ps.begin(), ps.end(), LVector<double>{});

    const VertexFFV v{StandardModel::qe / (M_SQRT2 * StandardModel::sw), 0.0};
    constexpr double mm = Muon::mass;
    constexpr double me = Electron::mass;

    const auto Muon_wfs = {spinor_u(pm, mm, 1), spinor_u(pm, mm, -1)};
    const auto nu_Muon_wfs = {spinor_ubar(ps[2], 0.0, 1),
                              spinor_ubar(ps[2], 0.0, -1)};
    const auto Electron_wfs = {spinor_ubar(ps[0], me, 1),
                               spinor_ubar(ps[0], me, -1)};
    const auto nu_Electron_wfs = {spinor_v(ps[1], 0.0, 1),
                                  spinor_v(ps[1], 0.0, -1)};

    VectorWf wf_w{};

    double msqrd = 0.0;
    for (const auto &wf_m : Muon_wfs) {
      for (const auto &wf_num : nu_Muon_wfs) {
        Current::generate(&wf_w, v, WBoson::mass, WBoson::width, wf_num, wf_m);
        for (const auto &wf_e : Electron_wfs) {
          for (const auto &wf_nue : nu_Electron_wfs) {
            msqrd += std::norm(amplitude(v, wf_e, wf_nue, wf_w));
          }
        }
      }
    }
    return msqrd / 2.0;
  };
  return Rambo<3>::decay_width(msqrd, Muon::mass, fsp_masses, nevents);
}

auto cross_section_ee_to_mm_analytic(double cme) -> std::pair<double, double> {
  using tools::sqr;
  using tools::two_body_three_momentum;
  constexpr double mm = Muon::mass;
  constexpr double me = Electron::mass;
  const std::array<double, 2> fsp_masses{mm, mm};

  const auto pmag = two_body_three_momentum(cme, me, me);
  const auto p1 = LVector<double>{std::hypot(me, pmag), 0.0, 0.0, pmag};

  auto msqrd = [&p1](const std::array<LVector<double>, 2> &ps) {
    const double s = lnorm_sqr(ps[0] + ps[1]);
    const double t = lnorm_sqr(p1 - ps[0]);
    const double u = lnorm_sqr(p1 - ps[1]);
    return (64 * pow(StandardModel::qe, 4) * pow(Electron::mass, 2) *
                pow(Muon::mass, 2) +
            16 * pow(StandardModel::qe, 4) * pow(Muon::mass, 2) *
                (-2 * pow(Electron::mass, 2) + s) +
            16 * pow(StandardModel::qe, 4) * pow(Electron::mass, 2) *
                (-2 * pow(Muon::mass, 2) + s) +
            8 * pow(StandardModel::qe, 4) *
                pow(pow(Electron::mass, 2) + pow(Muon::mass, 2) - t, 2) +
            8 * pow(StandardModel::qe, 4) *
                pow(pow(Electron::mass, 2) + pow(Muon::mass, 2) - u, 2)) /
           (4. * pow(s, 2));
  };
  return Rambo<2>::cross_section(msqrd, cme, me, me, fsp_masses);
}

auto cross_section_ee_to_mm(double cme) -> std::pair<double, double> {
  using tools::sqr;
  using tools::two_body_three_momentum;
  constexpr double mm = Muon::mass;
  constexpr double me = Electron::mass;
  const std::array<double, 2> fsp_masses{mm, mm};

  const auto pmag = two_body_three_momentum(cme, me, me);
  const auto p1 = LVector<double>{std::hypot(me, pmag), 0.0, 0.0, pmag};
  const auto p2 = LVector<double>{std::hypot(me, pmag), 0.0, 0.0, -pmag};

  auto msqrd = [&p1, &p2](const std::array<LVector<double>, 2> &ps) {
    auto p3 = ps[0];
    auto p4 = ps[1];

    const VertexFFV v{StandardModel::qe, StandardModel::qe};

    const auto wf1s = {spinor_u(p1, me, 1), spinor_u(p1, me, -1)};
    const auto wf2s = {spinor_vbar(p2, me, 1), spinor_vbar(p2, me, -1)};
    const auto wf3s = {spinor_ubar(p3, mm, 1), spinor_ubar(p3, mm, -1)};
    const auto wf4s = {spinor_v(p4, mm, 1), spinor_v(p4, mm, -1)};

    VectorWf wfg{};

    double msqrd = 0.0;
    for (const auto &wf1 : wf1s) {
      for (const auto &wf2 : wf2s) {
        Current::generate(&wfg, v, wf2, wf1);
        for (const auto &wf3 : wf3s) {
          for (const auto &wf4 : wf4s) {
            msqrd += std::norm(amplitude(v, wf3, wf4, wfg));
          }
        }
      }
    }
    return msqrd / 4.0;
  };
  return Rambo<2>::cross_section(msqrd, cme, me, me, fsp_masses);
}
