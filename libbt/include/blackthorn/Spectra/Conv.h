#ifndef BLACKTHORN_SPECTRA_CONV_H
#define BLACKTHORN_SPECTRA_CONV_H

#include "blackthorn/PhaseSpace.h"
#include <algorithm>
#include <array>
#include <boost/histogram.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <cmath>
#include <numeric>

namespace blackthorn {

namespace bh = boost::histogram;
using LinAxis = bh::axis::regular<double>;
using LogAxis = bh::axis::regular<double, bh::axis::transform::log>;
template <class Axis> using EnergyHist = bh::histogram<std::tuple<Axis>>;

/**
 * Generate energy distrubtions for final-state particles.
 */
template <class MSqrd, size_t N, class Axis>
auto energy_distributions(MSqrd msqrd, const double cme,
                          const std::array<double, N> &masses,
                          const std::array<unsigned int, N> &nbins,
                          size_t nevents) -> std::array<EnergyHist<Axis>, N> {
  const double msum = std::accumulate(masses.cbegin(), masses.cend(), 0);

  auto hs = std::array<bh::histogram<std::tuple<Axis>>, N>{};
#pragma unroll N
  for (size_t i = 0; i < N; ++i) {
    const double m = masses[i];
    const double emax = tools::energy_one_cm(cme, m, msum - m);
    hs[i] = bh::make_histogram(LinAxis(nbins[i], m, emax));
  }

  RamboEventGenerator<N, MSqrd> generator{msqrd, cme, masses};
  PhaseSpaceEvent<N> event{};

  for (size_t i = 0; i < nevents; ++i) {
    generator.fill(&event);
#pragma unroll N
    for (size_t j = 0; j < N; ++j) {
      hs[j](event.momenta()[j].e(), bh::weight(event.weight()));
    }
  }

  for (size_t i = 0; i < N; ++i) {
    double norm = 0.0;
    for (auto &&h : bh::indexed(hs[i])) {
      const double p = *h;
      const double de = h.bin().width();
      norm += p * de;
    }
    if (norm > 0) {
      hs[i] /= norm;
    }
  }

  return hs;
}

/**
 * Generate energy distrubtions for final-state particles.
 */
template <class MSqrd, size_t N>
auto energy_distributions_linear(MSqrd msqrd, const double cme,
                                 const std::array<double, N> &masses,
                                 const std::array<unsigned int, N> &nbins,
                                 size_t nevents)
    -> std::array<EnergyHist<LinAxis>, N> {
  return energy_distributions<MSqrd, N, LinAxis>(msqrd, cme, masses, nbins,
                                                 nevents);
}

/**
 * Generate energy distrubtions for final-state particles.
 */
template <class MSqrd, size_t N>
auto energy_distributions_log(MSqrd msqrd, const double cme,
                              const std::array<double, N> &masses,
                              const std::array<unsigned int, N> &nbins,
                              size_t nevents)
    -> std::array<EnergyHist<LogAxis>, N> {
  return energy_distributions<MSqrd, N, LogAxis>(msqrd, cme, masses, nbins,
                                                 nevents);
}

/**
 * Generate energy distrubtions for final-state particles.
 */
template <size_t I, size_t J, class MSqrd, size_t N, class Axis>
auto invariant_mass_distributions(MSqrd msqrd, const double cme,
                                  const std::array<double, N> &masses,
                                  unsigned int nbins, size_t nevents)
    -> EnergyHist<Axis> {

  static_assert(N >= 3, "`N` must be greater than 2.");
  static_assert(I < N, "`I` must be less than `N`.");
  static_assert(J < N, "`J` must be less than `N`.");

  const double msum = std::accumulate(masses.cbegin(), masses.cend(), 0);
  const double mi = masses[I];
  const double mj = masses[J];
  const double mmij = msum - mi - mj;

  auto hist = bh::make_histogram(Axis(nbins, mi + mj, cme - mmij));

  RamboEventGenerator<N, MSqrd> generator{msqrd, cme, masses};
  PhaseSpaceEvent<N> event{};

  for (size_t n = 0; n < nevents; ++n) {
    generator.fill(&event);
    double ss = lnorm(event.momenta()[I] + event.momenta()[J]);
    hist(ss, bh::weight(event.weight()));
  }
  double norm = 0.0;
  for (auto &&h : bh::indexed(hist)) {
    const double p = *h;
    const double de = h.bin().width();
    norm += p * de;
  }
  if (norm > 0) {
    hist /= norm;
  }

  return hist;
}

/**
 * Generate energy distrubtions for final-state particles.
 */
template <size_t I, size_t J, class MSqrd, size_t N>
auto invariant_mass_distributions_linear(MSqrd msqrd, const double cme,
                                         const std::array<double, N> &masses,
                                         unsigned int nbins, size_t nevents)
    -> EnergyHist<LinAxis> {
  return invariant_mass_distributions<I, J, MSqrd, N, LinAxis>(
      msqrd, cme, masses, nbins, nevents);
}

template <class F>
auto convolve(F f, double e, const EnergyHist<LinAxis> &hist) -> double {
  double conv = 0.0;
  for (auto &&h : boost::histogram::indexed(hist)) { // NOLINT
    const double p = *h;
    const double ep = h.bin().center();
    const double de = h.bin().width();
    conv += p * f(e, ep) * de;
  }
  return conv;
}

template <class F>
auto convolve(F f, double e, const EnergyHist<LogAxis> &hist) -> double {
  double conv = 0.0;
  for (auto &&h : boost::histogram::indexed(hist)) { // NOLINT
    const double p = *h;
    const double ep = h.bin().center();
    const double el = h.bin().lower();
    const double eh = h.bin().upper();
    const double dlog = log(eh / el);
    conv += p * f(e, ep) * ep * dlog;
  }
  return conv;
}

} // namespace blackthorn

#endif // BLACKTHORN_SPECTRA_CONV_H
