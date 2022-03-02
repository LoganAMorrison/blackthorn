#ifndef BLACKTHORN_SPECTRA_RAMBO_H
#define BLACKTHORN_SPECTRA_RAMBO_H

#include "blackthorn/PhaseSpace.h"
#include <boost/math/statistics/univariate_statistics.hpp>

namespace blackthorn {

namespace detail {

template <size_t N>
static auto generate_photon_momentum(MomentaType<N> *k, double e) -> void {
  static thread_local std::random_device rd{};
  static thread_local std::mt19937 generator{rd()};
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  const double phi = 2 * M_PI * distribution(generator);
  const double ct = 2 * distribution(generator) - 1;
  const double st = sqrt(1 - ct * ct);
  (*k)[N - 1][0] = e;
  (*k)[N - 1][1] = e * cos(phi) * st;
  (*k)[N - 1][2] = e * sin(phi) * st;
  (*k)[N - 1][3] = e * ct;
}

template <class MSqrd, size_t N>
auto integrate_photon_spectrum_rambo(MSqrd msqrd, double photon_energy,
                                     double m,
                                     const std::array<double, N> &fsp_masses,
                                     size_t nevents)
    -> std::pair<double, double> {
  using boost::math::statistics::mean_and_sample_variance;
  using tools::sqr;

  const double mass_sum =
      std::accumulate(fsp_masses.cbegin(), fsp_masses.cend(), 0.0);
  const double x = 2 * photon_energy / m;
  const double r2 = sqr(mass_sum / m);

  if (x <= 0 || 1 - r2 <= x) {
    return std::make_pair(0.0, 0.0);
  }

  // Total energy of the final state particles (excluding the photon) in their
  // rest frame
  const double cme = m * sqrt(1 - x);
  // Energy of the photon in the rest frame where final state particles
  // (excluding the photon)
  const double egam = photon_energy * m / cme;
  const double base_wgt = rambo_impl::massless_weight<N>(cme);

  std::vector<double> weights(nevents);
  MomentaType<N> momenta{};
  std::array<LVector<double>, N + 1> all_momenta{};

  // generate 'local' events
  for (size_t i = 0; i < nevents; i++) { // NOLINT
    rambo_impl::generate_momenta(&momenta, cme, fsp_masses);
    weights[i] = rambo_impl::wgt_rescale_factor(momenta, cme) * base_wgt;
#pragma unroll N
    for (size_t j = 0; j < N; ++j) {
      all_momenta[j] = momenta[j];
    }
    generate_photon_momentum(&all_momenta, egam);
    weights[i] *= msqrd(all_momenta);
  }
  // Compute mean and sum of squares
  // auto mv = tools::mean_var_welford(weights);
  // mv.second = sqrt(mv.second / static_cast<double>(nevents));
  // return mv;
  auto mv = mean_and_sample_variance(std::execution::par, weights.begin(),
                                     weights.end());
  mv.second = sqrt(mv.second / static_cast<double>(nevents));
  return mv;
}
} // namespace detail

template <class MSqrd, size_t N>
auto photon_spectrum_rambo(MSqrd msqrd, double photon_energy, double m,
                           const std::array<double, N> &fsp_masses,
                           double non_rad, size_t nevents)
    -> std::pair<double, double> {
  auto res = detail::integrate_photon_spectrum_rambo(msqrd, photon_energy, m,
                                                     fsp_masses, nevents);
  const double pre = photon_energy / (8 * M_PI * M_PI * non_rad * m);
  res.first *= pre;
  res.second *= pre;
  return res;
}

template <class MSqrd, size_t N>
auto photon_spectrum_rambo(MSqrd msqrd, double photon_energy, double m1,
                           double m2, double cme,
                           const std::array<double, N> &fsp_masses,
                           double non_rad, size_t nevents)
    -> std::pair<double, double> {
  auto res = integrate_photon_spectrum_rambo(msqrd, photon_energy, cme,
                                             fsp_masses, nevents);
  const double p = tools::two_body_three_momentum(cme, m1, m2);
  const double pre = photon_energy / (16 * M_PI * M_PI * non_rad * p * cme);
  res.first *= pre;
  res.second *= pre;
  return res;
}

} // namespace blackthorn

#endif // BLACKTHORN_SPECTRA_RAMBO_H
