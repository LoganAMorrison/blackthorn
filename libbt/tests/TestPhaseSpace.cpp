#include "Tools.h"
#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/PhaseSpace.h"
#include "blackthorn/Tensors.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>

using namespace blackthorn; // NOLINT

auto msqrd_mu_to_e_nu_nu(const std::array<LVector<double>, 3> &ps) -> double {
  using tools::sqr;
  const auto t = lnorm_sqr(ps[0] + ps[2]);
  constexpr double mmu = Muon::mass;
  return 16 * sqr(StandardModel::g_fermi) * t * (sqr(mmu) - t);
}

TEST(TestPs, TestMomentumConservation) { // NOLINT
  const double m = 10.0;
  const std::array<double, 3> fsp_masses{1.0, 2.0, 3.0};
  const size_t nevents = 10;

  const auto events =
      Rambo<3>::generate_phase_space(msqrd_flat<3>, m, fsp_masses, nevents);

  for (const auto &event : events) {
    const auto psum = std::accumulate(event.momenta().begin(),
                                      event.momenta().end(), LVector<double>{});
    ASSERT_LT(std::abs(psum[0] - m), 1e-10);
    ASSERT_LT(std::abs(psum[1]), 1e-10);
    ASSERT_LT(std::abs(psum[2]), 1e-10);
    ASSERT_LT(std::abs(psum[3]), 1e-10);
  }
}

TEST(TestPs, TestCorrectMasses) { // NOLINT
  const double cme = 10.0;
  const std::array<double, 3> masses{1.0, 2.0, 3.0};
  const size_t nevents = 10;

  const auto events =
      Rambo<3>::generate_phase_space(msqrd_flat<3>, cme, masses, nevents);

  for (const auto &event : events) {
    for (size_t i = 0; i < masses.size(); i++) {
      auto m = masses.at(i);
      ASSERT_LT(std::abs(lnorm_sqr(event.momenta(i)) - m * m), 1e-10);
    }
  }
}

TEST(TestPs, MuonDecay) { // NOLINT

  const std::array<double, 3> fsp_masses{0.0, 0.0, 0.0};
  const size_t nevents = 100'000;

  const double width = tools::sqr(StandardModel::g_fermi) * pow(Muon::mass, 5) /
                       (192 * pow(M_PI, 3));
  const auto res = Rambo<3>::decay_width(msqrd_mu_to_e_nu_nu, Muon::mass,
                                         fsp_masses, nevents);
  const auto frac_diff = fractional_diff(width, res.first);

  std::cout << "estimate = " << res.first << " +- " << res.second << std::endl;
  std::cout << "actual = " << width << std::endl;
  std::cout << "frac. diff. (%) = " << frac_diff * 100.0 << std::endl;

  ASSERT_LE(frac_diff, 1.0);
}
