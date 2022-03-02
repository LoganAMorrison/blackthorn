#ifndef BLACKTHORN_PHASE_SPACE_RAMBO_INTEGRATOR_H
#define BLACKTHORN_PHASE_SPACE_RAMBO_INTEGRATOR_H

#include "blackthorn/PhaseSpace/RamboCore.h"
#include "blackthorn/PhaseSpace/Types.h"
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <numeric>
#include <random>
#include <thread>

namespace blackthorn {

template <size_t N> class Rambo {
private:
  // 2*pi
  static constexpr double k2PI = 2 * M_PI;
  // 1/(2pi)
  static constexpr double k1_2PI = 1.0 / k2PI;

  static constexpr size_t DEFAULT_NEVENTS = 10'000;
  static constexpr size_t DEFAULT_BATCHSIZE = 100;

  // ========================================================================
  // ---- Validation --------------------------------------------------------
  // ========================================================================

  static auto check_nfsp() -> void {
    if constexpr (N == 0 || N == 1) {
      throw std::invalid_argument(
          "Number of final-state particles must be >= 2.");
    }
  }

  static auto channel_open(double cme, const std::array<double, N> &masses)
      -> bool {
    const double msum = std::accumulate(masses.begin(), masses.end(), 0.0);
    return cme > msum;
  }

  template <class MSqrd>
  static auto generate_event(MSqrd msqrd, double cme,
                             const std::array<double, N> &ms, double base_wgt)
      -> PhaseSpaceEvent<N> {
    MomentaType<N> momenta{};
    rambo_impl::generate_momenta(&momenta, cme, ms);
    double weight = rambo_impl::wgt_rescale_factor(momenta, cme) *
                    msqrd(momenta) * base_wgt;
    return PhaseSpaceEvent<N>(momenta, weight);
  }

public:
  // ===========================================================================
  // ---- API Functions --------------------------------------------------------
  // ===========================================================================

  /**
   * Generate a single PhaseSpaceEvent.
   *
   * @param msqrd Function to compute squared matrix element.
   * @param cme Center-of-mass energy
   * @param masses Masses of the final-state particles
   */
  template <class MSqrd>
  static auto generate_event(MSqrd msqrd, double cme,
                             const std::array<double, N> &ms)
      -> PhaseSpaceEvent<N> {
    const auto base_wgt = rambo_impl::massless_weight<N>(cme);
    return generate_event(msqrd, cme, ms, base_wgt);
  }

  /**
   * Fill the the PhaseSpaceEvent with a new weight and momenta.
   *
   * @param[out] event PhaseSpaceEvent to fill.
   * @param[in] msqrd Function to compute squared matrix element.
   * @param[in] cme Center-of-mass energy
   * @param[in] masses Masses of the final-state particles
   */
  template <class MSqrd>
  static auto fill_event(PhaseSpaceEvent<N> *event, MSqrd msqrd, double cme,
                         const std::array<double, N> &ms) -> void {
    rambo_impl::generate_momenta(&event->p_momenta, cme, ms);
    const auto base_wgt = rambo_impl::massless_weight<N>(cme);
    event->p_weight = rambo_impl::wgt_rescale_factor(event->p_momenta, cme) *
                      msqrd(event->p_momenta) * base_wgt;
  }

  /**
   * Integrate over an N-particle phase-space.
   *
   * @param cme Center-of-mass energy
   * @param masses Masses of the final-state particles
   * @param msqrd Function to compute squared matrix element given the momenta
   * @param nevents Number of phase-space points to sample
   * @param batchsize Number of phase-space points process at a time
   */
  template <class MSqrd>
  static auto integrate_phase_space(MSqrd msqrd, double cme,
                                    const std::array<double, N> &ms,
                                    const size_t nevents = DEFAULT_NEVENTS,
                                    const size_t batchsize = DEFAULT_BATCHSIZE)
      -> std::pair<double, double> {
    using tools::sqr;
    const double base_wgt = rambo_impl::massless_weight<N>(cme);
    const double inv_nevents = 1.0 / static_cast<double>(nevents);

    double mean = 0.0;
    double m2 = 0.0;
    double count = 0.0;

    std::mutex mutex;

    const auto adder = [&mean, &m2, &count, &mutex, &cme, &ms, &base_wgt,
                        &msqrd](size_t n) {
      std::vector<double> weights(n);
      MomentaType<N> momenta{};

      // generate 'local' events
      for (size_t i = 0; i < n; i++) { // NOLINT
        weights[i] =
            rambo_impl::generate_wgt(msqrd, &momenta, cme, ms, base_wgt);
      }
      // Compute mean and sum of squares
      auto lmv = tools::mean_sum_sqrs_welford(weights);

      // Lock access and add results to 'global' values
      std::lock_guard<std::mutex> g(mutex);
      const double avgb = std::get<0>(lmv);
      const double m2b = std::get<1>(lmv);
      const double nb = std::get<2>(lmv);

      const double na = count;
      count += nb;

      const double delta = mean - avgb;

      mean = (na * mean + nb * avgb) / count;
      m2 += m2b + sqr(delta) * na * nb / count;
    };

    // Compute the number of different threads to launch
    size_t nbatches = nevents / batchsize;
    const size_t remaining = nevents % batchsize;

    // Create threads and launch, then wait for them to finish
    std::vector<std::thread> threads;
    threads.reserve(nbatches + (remaining > 0 ? 1 : 0));

    for (size_t i = 0; i < nbatches; i++) { // NOLINT
      threads.emplace_back(adder, batchsize);
    }
    if (remaining > 0) {
      threads.emplace_back(adder, remaining);
    }
    for (auto &t : threads) { // NOLINT
      t.join();
    }

    // Compute global mean and std
    const double var = m2 / static_cast<double>(count - 1);
    const double std = sqrt(var * inv_nevents);

    return std::make_pair(mean, std);
  }

  /**
   * Generate many phase-space events
   *
   * @param cme Center-of-mass energy
   * @param masses Masses of the final-state particles
   * @param msqrd Function to compute squared matrix element given the momenta
   * @param nevents Number of phase-space points to sample
   * @param batchsize Number of phase-space points process at a time
   */
  template <class MSqrd>
  static auto generate_phase_space(MSqrd msqrd, double cme,
                                   const std::array<double, N> &ms,
                                   const size_t nevents = DEFAULT_NEVENTS,
                                   const size_t batchsize = DEFAULT_BATCHSIZE)
      -> std::vector<PhaseSpaceEvent<N>> {

    const double base_wgt = rambo_impl::massless_weight<N>(cme);

    std::mutex mutex;
    const auto nthreads = std::thread::hardware_concurrency();

    std::vector<PhaseSpaceEvent<N>> events{};
    events.reserve(nevents);

    const auto batch_generate = [&events, &mutex, &cme, &ms, &base_wgt,
                                 &msqrd](size_t n) {
      std::vector<PhaseSpaceEvent<N>> levents{};
      events.reserve(n);

      for (size_t i = 0; i < n; i++) { // NOLINT
        levents.emplace_back(generate_event(msqrd, cme, ms, base_wgt));
      }
      // Lock and push events to master event holder
      std::lock_guard<std::mutex> g(mutex);
      for (size_t i = 0; i < n; i++) { // NOLINT
        events.emplace_back(std::move(levents[i]));
      }
    };

    const auto nbatches = nevents / batchsize;
    const auto remaining = nevents % batchsize;

    std::vector<std::thread> threads;
    threads.reserve(nthreads);

    for (size_t i = 0; i < nbatches; i++) { // NOLINT
      threads.emplace_back(batch_generate, batchsize);
    }
    if (remaining > 0) { // NOLINT
      threads.emplace_back(batch_generate, remaining);
    }
    for (auto &t : threads) { // NOLINT
      t.join();
    }

    return events;
  }

  /**
   * Compute the decay width of a particle.
   *
   * @param m Mass of the decaying particle
   * @param masses Masses of the final-state particles
   * @param msqrd Function to compute squared matrix element given the momenta
   * @param nevents Number of phase-space points to sample
   * @param batchsize Number of phase-space points process at a time
   */
  template <class MSqrd>
  static auto decay_width(MSqrd msqrd, const double m,
                          const std::array<double, N> &masses,
                          const size_t nevents = DEFAULT_NEVENTS,
                          const size_t batchsize = DEFAULT_BATCHSIZE)
      -> std::pair<double, double> {
    check_nfsp();
    if (!channel_open(m, masses)) {
      return std::make_pair(0.0, 0.0);
    }
    auto result = integrate_phase_space(msqrd, m, masses, nevents, batchsize);
    result.first = result.first / (2.0 * m);
    result.second = result.second / (2.0 * m);
    return result;
  }

  /**
   * Compute the decay width or scattering cross-section.
   *
   * @param cme Center-of-mass energy
   * @param m1 Mass of 1st incoming particle
   * @param m2 Mass of 2nd incoming particle
   * @param masses Masses of the final-state particles
   * @param msqrd Function to compute squared matrix element given the momenta
   * @param nevents Number of phase-space points to sample
   * @param batchsize Number of phase-space points process at a time
   */
  template <class MSqrd>
  static auto cross_section(MSqrd msqrd, const double cme, double m1, double m2,
                            const std::array<double, N> &masses,
                            const size_t nevents = DEFAULT_NEVENTS,
                            const size_t batchsize = DEFAULT_BATCHSIZE)
      -> std::pair<double, double> {
    check_nfsp();
    if (!channel_open(cme, masses) || m1 + m2 > cme) {
      return std::make_pair(0.0, 0.0);
    }

    auto res = integrate_phase_space(msqrd, cme, masses, nevents, batchsize);

    const auto mu1 = tools::sqr(m1 / cme);
    const auto mu2 = tools::sqr(m2 / cme);
    // (2 * E1) * (2 * E2) * vrel
    const auto den =
        2.0 * cme * cme * std::sqrt(tools::kallen_lambda(1.0, mu1, mu2));

    res.first /= den;
    res.second /= den;
    return res;
  }
};

template <>
template <class MSqrd>
auto Rambo<2>::integrate_phase_space(MSqrd msqrd, double cme,
                                     const std::array<double, 2> &ms,
                                     size_t /*nevents*/, size_t /*batchsize*/)
    -> std::pair<double, double> {
  using boost::math::quadrature::gauss_kronrod;
  using tools::sqr;
  constexpr unsigned N = 15;
  constexpr unsigned max_depth = 5;
  constexpr double tol = 1e-9;
  constexpr double lb = -1.0;
  constexpr double ub = 1.0;

  const double m1 = ms[0];
  const double m2 = ms[1];
  const double e1 = (sqr(cme) + sqr(m1) - sqr(m2)) / (2 * cme);
  const double e2 = (sqr(cme) - sqr(m1) + sqr(m2)) / (2 * cme);
  const double pmag = tools::two_body_three_momentum(cme, m1, m2);
  const double pre = pmag / (8 * M_PI * cme);
  MomentaType<2> momenta{};

  auto integrand = [&momenta, &e1, &e2, &pmag, &msqrd](double z) {
    const double sz = std::sqrt(1 - z * z);
    momenta[0] = {e1, pmag * sz, 0.0, pmag * z};
    momenta[1] = {e2, -pmag * sz, 0.0, -pmag * z};
    return msqrd(momenta);
  };
  double error = 0.0;
  double result = gauss_kronrod<double, N>::integrate(integrand, lb, ub,
                                                      max_depth, tol, &error);

  return std::make_pair(pre * result, pre * error);
}

} // namespace blackthorn

#endif // BLACKTHORN_PHASE_SPACE_RAMBO_INTEGRATOR_H
