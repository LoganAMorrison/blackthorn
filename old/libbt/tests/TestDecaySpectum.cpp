#define CATCH_CONFIG_MAIN

#include "blackthorn/Models/Particles.h"
#include "blackthorn/Models/RhNeutrino.h"
#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/Spectra.h"
#include <catch2/catch.hpp>
#include <chrono>
#include <fmt/format.h>
#include <iomanip>

using namespace blackthorn;

using DVec = std::vector<double>;

TEST_CASE("Spectra from: mu -> e + ve + vm", "[muon]") {
  const size_t npts = 100;
  const double emu = Muon::mass * 2;
  const auto es = tools::geomspace(emu * 1e-6, emu, npts);

  SECTION("Photon") {}

  SECTION("Positron") {
    const auto dndes_rest = decay_spectrum<Muon>::dnde_positron(es, Muon::mass);
    const auto dndes_boosted = decay_spectrum<Muon>::dnde_positron(es, emu);

    std::cout << std::scientific;
    for (size_t i = 0; i < npts; ++i) { // NOLINT
      fmt::print("{:1.5e}, {:1.5e}, {:1.5e}\n", es[i], dndes_rest[i],
                 dndes_boosted[i]);
    }
  }

  SECTION("Neutrino") {
    std::cout << std::scientific;
    for (size_t i = 0; i < npts; ++i) { // NOLINT
      if (i == npts - 1) {
        std::cout << ""
                  << "";
      }
      const double dnder_e =
          decay_spectrum<Muon>::dnde_neutrino(es[i], Muon::mass, Gen::Fst);
      const double dndeb_e =
          decay_spectrum<Muon>::dnde_neutrino(es[i], emu, Gen::Fst);
      const double dnder_m =
          decay_spectrum<Muon>::dnde_neutrino(es[i], Muon::mass, Gen::Snd);
      const double dndeb_m =
          decay_spectrum<Muon>::dnde_neutrino(es[i], emu, Gen::Snd);
      fmt::print("{:1.5e}, {:1.5e}, {:1.5e}, {:1.5e}, {:1.5e}\n", es[i],
                 dnder_e, dndeb_e, dnder_m, dndeb_m);
    }
  }
}

TEST_CASE("Spectra from: N -> l + pi", "[rhn][lepton][charged-pion]") {
  const size_t npts = 100;
  const double m = 0.4;
  const double beta = 0.4;

  DecaySpectrum<ChargedPion, Muon> dspec(m);
  RhNeutrinoMeV model(m, 1e-3, Gen::Snd);

  auto xs = tools::geomspace(1e-6, 1.0, npts);

  std::cout << std::scientific;
  SECTION("Photon") {
    fmt::print("{:<11}  {:<11}  {:<11}  {:<11}\n", "x", "dndx1", "dndx2",
               "ratio");
    for (size_t i = 0; i < npts; ++i) { // NOLINT
      const double x = xs[i];
      const double dndx1 = dspec.dndx_photon(x, beta);
      const double dndx2 = model.dndx_photon_l_pi(x, beta);
      const double ratio = dndx2 / dndx1;
      fmt::print("{:1.5e}, {:1.5e}, {:1.5e}, {:1.5e}\n", x, dndx1, dndx2,
                 ratio);
    }
  }

  SECTION("Positron") {
    fmt::print("{:<11}  {:<11}  {:<11}  {:<11}\n", "x", "dndx1", "dndx2",
               "ratio");
    for (size_t i = 0; i < npts; ++i) { // NOLINT
      const double x = xs[i];
      const double dndx1 = dspec.dndx_positron(x, beta);
      fmt::print("{:1.5e}, {:1.5e}\n", x, dndx1);
    }
  }

  SECTION("Neutrino") {
    std::cout << std::scientific;
    fmt::print("{:<11}  {:<11}  {:<11}  {:<11}\n", "x", "dndx1", "dndx2",
               "ratio");
    for (size_t i = 0; i < npts; ++i) { // NOLINT
      const double x = xs[i];
      const double dndx_e = dspec.dndx_neutrino(x, beta, Gen::Fst);
      const double dndx_m = dspec.dndx_neutrino(x, beta, Gen::Snd);
      fmt::print("{:1.5e}, {:1.5e}, {:1.5e}\n", x, dndx_e, dndx_m);
    }
  }
}

TEST_CASE("Spectrum from: N -> nu + pi0", "[rhn][neutrino][neutral-pion]") {
  const size_t npts = 100;
  const double m = 0.4;
  const double beta = 0.2;
  DecaySpectrum<NeutralPion, ElectronNeutrino> dspec(m);
  RhNeutrinoMeV model(m, 1e-3, Gen::Fst);

  auto xs = tools::geomspace(1e-6, 1.0, npts);

  std::cout << std::scientific;

  SECTION("Photon") {
    fmt::print("{:<11}  {:<11}  {:<11}  {:<11}\n", "x", "dndx1", "dndx2",
               "ratio");
    for (size_t i = 0; i < npts; ++i) { // NOLINT
      const double x = xs[i];
      const double dndx1 = dspec.dndx_photon(x, beta);
      const double dndx2 = model.dndx_photon_v_pi0(x, beta);
      const double ratio = dndx1 / dndx2;
      fmt::print("{:1.5e}, {:1.5e}, {:1.5e}, {:1.5e}\n", x, dndx1, dndx2,
                 ratio);
    }
  }

  SECTION("Positron") {}

  SECTION("Neutrino") {}
}

TEST_CASE("Spectrum from: N -> nu + l + l", "[rhn][neutrino][lepton]") {
  const size_t npts = 100;
  const double m = 0.4;
  const double beta = 0.4;

  constexpr auto g0 = Gen::Fst;
  constexpr auto g1 = Gen::Fst;
  constexpr auto g2 = Gen::Fst;
  constexpr auto g3 = Gen::Fst;
  using P1 = NeutrinoType<g1>::type;
  using P2 = ChargedLeptonType<g2>::type;
  using P3 = ChargedLeptonType<g3>::type;
  using TUnit = std::milli;

  SECTION("Electron") {
    RhNeutrinoMeV model(m, 1e-3, g0);
    SquaredAmplitudeNToVLL msqrd(model, g1, g2, g3);
    DecaySpectrum<P1, P2, P3> dspec(m, msqrd);

    SECTION("Photon") {
      double en = m / sqrt(1 - beta * beta);

      auto xs = tools::geomspace(1e-6, 1.0, npts);
      auto start = std::chrono::high_resolution_clock::now();
      auto dndx1s = dspec.dndx_photon(xs, beta);
      auto stop1 = std::chrono::high_resolution_clock::now();
      auto dndx2s = model.dndx_photon_v_l_l(xs, beta, g1, g2, g3);
      auto stop2 = std::chrono::high_resolution_clock::now();

      auto t1 = std::chrono::duration<double, TUnit>(stop1 - start);
      auto t2 = std::chrono::duration<double, TUnit>(stop2 - stop1);

      std::cout << std::scientific;
      fmt::print(
          "\n-----------------------------------------------------------\n");
      fmt::print("timings: new = {:.2e} ms, old = {:.2e} ms\n", t1.count(),
                 t2.count());
      fmt::print("{:<11}  {:<11}  {:<11}  {:<11}\n", "x", "dndx1", "dndx2",
                 "ratio");
      for (size_t i = 0; i < npts; ++i) {
        const double x = xs[i];
        const double dndx1 = dndx1s[i];
        const double dndx2 = dndx2s[i];
        const double ratio = dndx2 / dndx1;

        fmt::print("{:1.5e}, {:1.5e}, {:1.5e}, {:1.5e}\n", x, dndx1, dndx2,
                   ratio);
      }
    }
    SECTION("Positron") {
      double en = m / sqrt(1 - beta * beta);

      auto xs = tools::geomspace(1e-6, 1.0, npts);
      auto start = std::chrono::high_resolution_clock::now();
      auto dndx1s = dspec.dndx_positron(xs, beta);
      auto stop1 = std::chrono::high_resolution_clock::now();
      auto dndx2s = model.dndx_positron_v_l_l(xs, beta, g1, g2, g3);
      auto stop2 = std::chrono::high_resolution_clock::now();

      auto t1 = std::chrono::duration<double, TUnit>(stop1 - start);
      auto t2 = std::chrono::duration<double, TUnit>(stop2 - stop1);

      std::cout << std::scientific;
      fmt::print(
          "\n-----------------------------------------------------------\n");
      fmt::print("timings: new = {:.2e} ms, old = {:.2e} ms\n", t1.count(),
                 t2.count());
      fmt::print("{:<11}  {:<11}  {:<11}  {:<11}\n", "x", "dndx1", "dndx2",
                 "ratio");
      for (size_t i = 0; i < npts; ++i) {
        const double x = xs[i];
        const double dndx1 = dndx1s[i];
        const double dndx2 = dndx2s[i];
        const double ratio = dndx2 / dndx1;

        fmt::print("{:1.5e}, {:1.5e}, {:1.5e}, {:1.5e}\n", x, dndx1, dndx2,
                   ratio);
      }
    }
    SECTION("Neutrino") {}
  }

  SECTION("Muon") {
    RhNeutrinoMeV model(m, 1e-3, Gen::Fst);
    SquaredAmplitudeNToVLL msqrd(model, g1, g2, g3);
    DecaySpectrum<P1, P2, P3> dspec(m, msqrd);

    SECTION("Photon") {

      auto xs = tools::geomspace(1e-6, 1.0, npts);
      auto start = std::chrono::high_resolution_clock::now();
      auto dndx1s = dspec.dndx_photon(xs, beta);
      auto stop1 = std::chrono::high_resolution_clock::now();
      auto dndx2s = model.dndx_photon_v_l_l(xs, beta, g1, g2, g3);
      auto stop2 = std::chrono::high_resolution_clock::now();

      auto t1 = std::chrono::duration<double, TUnit>(stop1 - start);
      auto t2 = std::chrono::duration<double, TUnit>(stop2 - stop1);

      std::cout << std::scientific;
      fmt::print(
          "\n-----------------------------------------------------------\n");
      fmt::print("timings: new = {:.2e} ms, old = {:.2e} ms\n", t1.count(),
                 t2.count());
      fmt::print("{:<11}  {:<11}  {:<11}  {:<11}\n", "x", "dndx1", "dndx2",
                 "ratio");
      for (size_t i = 0; i < npts; ++i) {
        const double x = xs[i];
        const double dndx1 = dndx1s[i];
        const double dndx2 = dndx2s[i];
        const double ratio = dndx2 / dndx1;

        fmt::print("{:1.5e}, {:1.5e}, {:1.5e}, {:1.5e}\n", x, dndx1, dndx2,
                   ratio);
      }
    }
    SECTION("Positron") {}
    SECTION("Neutrino") {}
  }
}

TEST_CASE("Spectrum from: N -> l + pi + pi0",
          "[rhn][charged-lepton][charged-pion][neutral-pion]") {

  const size_t npts = 100;
  const double m = 0.4;
  const double beta = 0.4;
  const double theta = 1e-3;

  using TUnit = std::milli;

  SECTION("Electron") {
    constexpr Gen G = Gen::Fst;

    using P1 = ChargedLeptonType<G>::type;
    using P2 = ChargedPion;
    using P3 = NeutralPion;
    constexpr double ml = P1::mass;

    RhNeutrinoMeV model(m, 1e-3, G);
    SquaredAmplitudeNToLPiPi0 msqrd(model);
    DecaySpectrum<P1, P2, P3> dspec(m, msqrd);

    SECTION("Photon") {
      auto xs = tools::geomspace(1e-6, 1.0, npts);
      auto start = std::chrono::high_resolution_clock::now();
      auto dndx1s = dspec.dndx_photon(xs, beta);
      auto stop1 = std::chrono::high_resolution_clock::now();
      auto dndx2s = model.dndx_photon_l_pi_pi0(xs, beta);
      auto stop2 = std::chrono::high_resolution_clock::now();

      auto t1 = std::chrono::duration<double, TUnit>(stop1 - start);
      auto t2 = std::chrono::duration<double, TUnit>(stop2 - stop1);

      std::cout << std::scientific;
      fmt::print(
          "\n-----------------------------------------------------------\n");
      fmt::print("timings: new = {:.2e} ms, old = {:.2e} ms\n", t1.count(),
                 t2.count());
      fmt::print("{:<11}  {:<11}  {:<11}  {:<11}\n", "x", "dndx1", "dndx2",
                 "ratio");
      for (size_t i = 0; i < npts; ++i) {
        const double x = xs[i];
        const double dndx1 = dndx1s[i];
        const double dndx2 = dndx2s[i];
        const double ratio = dndx2 / dndx1;

        fmt::print("{:1.5e}, {:1.5e}, {:1.5e}, {:1.5e}\n", x, dndx1, dndx2,
                   ratio);
      }
    }
    SECTION("Positron") {}
    SECTION("Neutrino") {}
  }

  SECTION("Muon") {
    constexpr Gen G = Gen::Snd;

    using P1 = ChargedLeptonType<G>::type;
    using P2 = ChargedPion;
    using P3 = NeutralPion;
    constexpr double ml = P1::mass;

    RhNeutrinoMeV model(m, 1e-3, G);
    SquaredAmplitudeNToLPiPi0 msqrd(model);
    DecaySpectrum<P1, P2, P3> dspec(m, msqrd);

    SECTION("Photon") {
      auto xs = tools::geomspace(1e-6, 1.0, npts);
      auto start = std::chrono::high_resolution_clock::now();
      auto dndx1s = dspec.dndx_photon(xs, beta);
      auto stop1 = std::chrono::high_resolution_clock::now();
      auto dndx2s = model.dndx_photon_l_pi_pi0(xs, beta);
      auto stop2 = std::chrono::high_resolution_clock::now();

      auto t1 = std::chrono::duration<double, TUnit>(stop1 - start);
      auto t2 = std::chrono::duration<double, TUnit>(stop2 - stop1);

      std::cout << std::scientific;
      fmt::print(
          "\n-----------------------------------------------------------\n");
      fmt::print("timings: new = {:.2e} ms, old = {:.2e} ms\n", t1.count(),
                 t2.count());
      fmt::print("{:<11}  {:<11}  {:<11}  {:<11}\n", "x", "dndx1", "dndx2",
                 "ratio");
      for (size_t i = 0; i < npts; ++i) {
        const double x = xs[i];
        const double dndx1 = dndx1s[i];
        const double dndx2 = dndx2s[i];
        const double ratio = dndx2 / dndx1;

        fmt::print("{:1.5e}, {:1.5e}, {:1.5e}, {:1.5e}\n", x, dndx1, dndx2,
                   ratio);
      }
    }
    SECTION("Positron") {}
    SECTION("Neutrino") {}
  }
}
