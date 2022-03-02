#include "blackthorn/PhaseSpace.h"

namespace blackthorn {

// 2*pi
static constexpr double k2PI = 2 * M_PI;
// 1/(2pi)
static constexpr double k1_2PI = 1.0 / k2PI;

static auto massless_weight(double cme, size_t nn) -> double {
  const auto n = static_cast<double>(nn);
  const auto nm1 = n - 1;
  double wgt = pow(M_PI_2, nm1) * pow(cme, 2 * n - 4) * pow(k1_2PI, 3 * n - 4);

  // Compute (n-1)! * (n-2)!
  double fact_nm2 = 1.0;
  for (size_t i = 2; i < nn - 2; i++) {
    fact_nm2 *= static_cast<double>(i);
  }
  const double fact_nm1 = nm1 * fact_nm2;
  const double fact = 1.0 / (fact_nm1 * fact_nm2);
  return fact * wgt;
}

RamboEventGenerator::RamboEventGenerator(double cme, std::vector<double> masses,
                                         MSqrdType msqrd)
    : p_cme(cme), p_masses(std::move(masses)), p_msqrd(std::move(msqrd)),
      p_n(p_masses.size()), p_base_wgt(massless_weight(cme, p_n)) {
  using tools::sqr;
  const double msum = std::accumulate(p_masses.begin(), p_masses.end(), 0.0);
  p_xi0 = sqrt(1.0 - sqr(msum / p_cme));
}

// Generate a `n` random four-momenta with energies distributated according to
// E⋅exp(-E)
auto RamboEventGenerator::initialize_momenta(MomentaType *momenta) -> void {
  static thread_local std::random_device rd{};
  static thread_local std::mt19937 generator{rd()};
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

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
auto RamboEventGenerator::boost_momenta(MomentaType *momenta) const -> void {
  const LVector<double> sum_qs =
      std::accumulate(momenta->begin(), momenta->end(), LVector<double>{});

  const double invmass = 1.0 / lnorm(sum_qs);
  // boost vector
  const double bx = -invmass * sum_qs[1];
  const double by = -invmass * sum_qs[2];
  const double bz = -invmass * sum_qs[3];
  // boost factors
  const double x = p_cme * invmass;
  const double g = sum_qs[0] * invmass;
  const double a = 1.0 / (1.0 + g);

  for (auto &p : *momenta) {
    const double bdotq = bx * p[1] + by * p[2] + bz * p[3];
    const double fact = std::fma(a, bdotq, p[0]);

    p[0] = x * std::fma(g, p[0], bdotq);
    p[1] = x * std::fma(fact, bx, p[1]);
    p[2] = x * std::fma(fact, by, p[2]);
    p[3] = x * std::fma(fact, bz, p[3]);
  }
}

auto RamboEventGenerator::compute_scale_factor(const MomentaType &ps) const
    -> double {
  using tools::sqr;
  constexpr int MAX_ITER = 50;
  constexpr double TOL = 1e-10;

  double xi = p_xi0;
  size_t itercount = 0;
  while (true) {
    double f = -p_cme;
    double df = 0.0;
    // Compute residual and derivative
    for (size_t i = 0; i < p_n; i++) {
      const double e = ps[i][0];
      const double deltaf = std::hypot(p_masses[i], xi * e);
      f += deltaf;
      df += xi * sqr(e) / deltaf;
    }

    // Newton correction
    const double deltaxi = -f / df;
    xi += deltaxi;

    itercount += 1;
    if (std::abs(deltaxi) < TOL || itercount > MAX_ITER) {
      break;
    }
  }
  return xi;
}

auto RamboEventGenerator::correct_masses(MomentaType *ps) const -> void {
  const double xi = compute_scale_factor(*ps);
  for (size_t i = 0; i < p_n; i++) {
    ps->at(i)[0] = std::hypot(p_masses[i], xi * ps->at(i)[0]);
    ps->at(i)[1] *= xi;
    ps->at(i)[2] *= xi;
    ps->at(i)[3] *= xi;
  }
}

auto RamboEventGenerator::wgt_rescale_factor(const MomentaType &ps) const
    -> double {
  using tools::sqr;
  double t1 = 0.0;
  double t2 = 0.0;
  double t3 = 1.0;
  for (const auto &p : ps) {
    const double modsqr = lnorm3_sqr(p);
    const double mod = sqrt(modsqr);
    const double inveng = 1.0 / p[0];
    t1 += mod / p_cme;
    t2 += modsqr * inveng;
    t3 *= mod * inveng;
  }
  t1 = pow(t1, static_cast<double>(2 * p_n - 3));
  t2 = 1.0 / t2;
  return t1 * t2 * t3 * p_cme;
}

auto RamboEventGenerator::generate(MomentaType *ps, double *wgt) const -> void {
  initialize_momenta(ps);
  boost_momenta(ps);
  correct_masses(ps);
  *wgt = p_base_wgt * wgt_rescale_factor(*ps) * p_msqrd(*ps);
}

auto RamboEventGenerator::generate() const -> std::pair<MomentaType, double> {
  MomentaType ps{};
  initialize_momenta(&ps);
  boost_momenta(&ps);
  correct_masses(&ps);
  double wgt = p_base_wgt * wgt_rescale_factor(ps) * p_msqrd(ps);
  return std::make_pair(ps, wgt);
}

} // namespace blackthorn
