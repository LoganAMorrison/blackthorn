#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "../Tools.h"
#include "blackthorn/Amplitudes.h"
#include "blackthorn/Models/RhNeutrino.h"
#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/PhaseSpace.h"
#include "blackthorn/Tensors.h"
#include <algorithm>
#include <catch2/catch.hpp>
#include <fmt/format.h>
#include <numeric>
#include <tuple>

using namespace blackthorn; // NOLINT

auto width_analytic(double mass, double theta, Gen genu)
    -> std::pair<double, double>;

auto gen_to_string(Gen gen) -> std::string {
  if (gen == Gen::Fst) {
    return "fst";
  }
  if (gen == Gen::Snd) {
    return "snd";
  }
  return "trd";
}

auto print_model(const RhNeutrinoGeV &model) -> void {
  std::cout << "RhNetrinoGeV(mass: " << model.mass()
            << ", theta: " << model.mass()
            << ", gen: " << gen_to_string(model.gen()) << ")\n";
}

TEST_CASE("Test N -> nu + H", "[widths][vh]") { // NOLINT
  using tools::print_width;
  using tools::UNICODE_MU;
  using tools::UNICODE_NU;

  std::cout << std::scientific;

  constexpr Gen genn = Gen::Fst;
  const std::string l = "e";
  auto model = RhNeutrinoGeV(1e3, 1e-3, genn);
  print_model(model);

  auto res = model.width_v_h();
  print_width("estimate", res, "N", {"v", "H"});
  std::cout << "\n";
}

TEST_CASE("Test N -> nu + Z", "[widths][vz]") { // NOLINT
  using tools::print_width;
  using tools::UNICODE_MU;
  using tools::UNICODE_NU;

  std::cout << std::scientific;

  constexpr Gen genn = Gen::Fst;
  const std::string l = "e";
  auto model = RhNeutrinoGeV(1e3, 1e-3, genn);
  print_model(model);

  auto res = model.width_v_z();
  print_width("estimate", res, "N", {"v", "Z"});
  std::cout << "\n";
}

TEST_CASE("Test N -> nu + u + u", "[widths][vuu]") { // NOLINT
  using tools::print_cross_section;
  using tools::print_width;
  using tools::UNICODE_MU;
  using tools::UNICODE_NU;

  SECTION("N -> nu + u + u") {
    constexpr Gen genn = Gen::Fst;
    const std::string l = "e";
    auto model = RhNeutrinoGeV(50.0, 1e-3, genn);

    const Gen genu = Gen::Fst;
    const std::string fs = "u";

    auto res = model.width_v_u_u(genu);
    auto res_an = width_analytic(model.mass(), model.theta(), genu);
    print_width("estimate", res.first, res.second, "N", {"v", fs, fs});
    print_width("analytic", res_an.first, res_an.second, "N", {"v", fs, fs});
    print_fractional_diff(res_an.first, res.first);
    std::cout << "\n";
  }

  SECTION("N -> nu + c + c") {
    constexpr Gen genn = Gen::Fst;
    const std::string l = "e";
    auto model = RhNeutrinoGeV(50.0, 1e-3, genn);

    const Gen genu = Gen::Snd;
    const std::string fs = "c";

    auto res = model.width_v_u_u(genu);
    auto res_an = width_analytic(model.mass(), model.theta(), genu);
    print_width("estimate", res.first, res.second, "N", {"v", fs, fs});
    print_width("analytic", res_an.first, res_an.second, "N", {"v", fs, fs});
    print_fractional_diff(res_an.first, res.first);
    std::cout << "\n";
  }

  SECTION("N -> nu + t + t") {
    constexpr Gen genn = Gen::Fst;
    const std::string l = "e";
    auto model = RhNeutrinoGeV(1e3, 1e-3, genn);

    const Gen genu = Gen::Trd;
    const std::string fs = "t";

    auto res = model.width_v_u_u(genu);
    auto res_an = width_analytic(model.mass(), model.theta(), genu);
    print_width("estimate", res.first, res.second, "N", {"v", fs, fs});
    print_width("analytic", res_an.first, res_an.second, "N", {"v", fs, fs});
    print_fractional_diff(res_an.first, res.first);
    std::cout << "\n";
  }
}

auto msqrd_analytic(const std::array<LVector<double>, 3> &ps, double mass,
                    double theta, Gen genu) -> double {
  const double mu = StandardModel::up_type_quark_mass(genu);
  const double qe = StandardModel::qe;
  const double sw = StandardModel::sw;
  const double cw = StandardModel::cw;
  const double mz = ZBoson::mass;
  const double wz = ZBoson::width;
  const double mh = Higgs::mass;
  const double wh = Higgs::width;
  const double mvr = mass;
  const double s = lnorm_sqr(ps[1] + ps[2]);
  const double t = lnorm_sqr(ps[0] + ps[2]);
  const double u = lnorm_sqr(ps[0] + ps[1]);

  const double z_prop_den = tools::sqr(s - mz * mz) + tools::sqr(mz * wz);
  const double h_prop_den = tools::sqr(s - mh * mh) + tools::sqr(mh * wh);

  const double msqrd_h =
      (2 * std::pow(mu, 2) * std::pow(mass, 2) * (std::pow(mass, 2) - s) *
       (-4 * std::pow(mu, 2) + s) * std::pow(std::tan(theta), 2)) /
      (h_prop_den * std::pow(Higgs::vev, 4));

  const double msqrd_z =
      (std::pow(qe, 4) *
       (std::pow(sw, 4) *
            (36 * std::pow(mu, 6) * (2 * std::pow(ZBoson::mass, 2) - s) +
             17 * std::pow(ZBoson::mass, 4) *
                 (-std::pow(t, 2) - std::pow(u, 2) +
                  std::pow(mvr, 2) * (t + u)) +
             std::pow(mu, 4) *
                 (-34 * std::pow(ZBoson::mass, 4) +
                  36 * std::pow(mvr, 2) * (2 * std::pow(ZBoson::mass, 2) - s) -
                  72 * std::pow(ZBoson::mass, 2) * (t + u) + 36 * s * (t + u)) +
             std::pow(mu, 2) *
                 (std::pow(ZBoson::mass, 4) * (-16 * s + 34 * (t + u)) -
                  18 * std::pow(ZBoson::mass, 2) *
                      (std::pow(s, 2) - std::pow(t + u, 2)) +
                  9 * s * (std::pow(s, 2) - std::pow(t + u, 2)) -
                  9 * std::pow(mvr, 2) *
                      (2 * std::pow(ZBoson::mass, 4) + s * (s - 2 * (t + u)) +
                       std::pow(ZBoson::mass, 2) * (-2 * s + 4 * (t + u))))) +
        9 * std::pow(cw, 4) *
            (std::pow(mu, 6) * (8 * std::pow(ZBoson::mass, 2) - 4 * s) +
             std::pow(ZBoson::mass, 4) * (-std::pow(t, 2) - std::pow(u, 2) +
                                          std::pow(mvr, 2) * (t + u)) +
             std::pow(mu, 4) *
                 (-2 * std::pow(ZBoson::mass, 4) +
                  std::pow(mvr, 2) * (8 * std::pow(ZBoson::mass, 2) - 4 * s) -
                  8 * std::pow(ZBoson::mass, 2) * (t + u) + 4 * s * (t + u)) +
             std::pow(mu, 2) *
                 (2 * std::pow(ZBoson::mass, 4) * (t + u) +
                  s * (std::pow(s, 2) - std::pow(t + u, 2)) +
                  2 * std::pow(ZBoson::mass, 2) *
                      (-std::pow(s, 2) + std::pow(t + u, 2)) -
                  std::pow(mvr, 2) *
                      (2 * std::pow(ZBoson::mass, 4) + s * (s - 2 * (t + u)) +
                       std::pow(ZBoson::mass, 2) * (-2 * s + 4 * (t + u))))) +
        6 * std::pow(cw, 2) * std::pow(sw, 2) *
            (12 * std::pow(mu, 6) * (2 * std::pow(ZBoson::mass, 2) - s) +
             std::pow(ZBoson::mass, 4) * (std::pow(t, 2) + std::pow(u, 2) -
                                          std::pow(mvr, 2) * (t + u)) +
             2 * std::pow(mu, 4) *
                 (std::pow(ZBoson::mass, 4) +
                  6 * std::pow(mvr, 2) * (2 * std::pow(ZBoson::mass, 2) - s) -
                  12 * std::pow(ZBoson::mass, 2) * (t + u) + 6 * s * (t + u)) -
             std::pow(mu, 2) *
                 (-3 * std::pow(s, 3) + 3 * s * std::pow(t + u, 2) +
                  2 * std::pow(ZBoson::mass, 4) * (-4 * s + t + u) +
                  6 * std::pow(ZBoson::mass, 2) *
                      (std::pow(s, 2) - std::pow(t + u, 2)) +
                  3 * std::pow(mvr, 2) *
                      (2 * std::pow(ZBoson::mass, 4) + s * (s - 2 * (t + u)) +
                       std::pow(ZBoson::mass, 2) * (-2 * s + 4 * (t + u)))))) *
       std::pow(std::sin(2 * theta), 2)) /
      (288. * std::pow(cw, 4) * std::pow(ZBoson::mass, 4) * z_prop_den *
       std::pow(sw, 4));

  return 3 * (msqrd_h + msqrd_z);
}

auto width_analytic(double mass, double theta, Gen genu)
    -> std::pair<double, double> {

  const double mu = StandardModel::up_type_quark_mass(genu);
  const std::array<double, 3> masses = {0.0, mu, mu};

  const auto msqrd = [mass, theta,
                      genu](const std::array<LVector<double>, 3> &ps) {
    return msqrd_analytic(ps, mass, theta, genu);
  };

  return Rambo<3>::decay_width(msqrd, mass, masses);
}
