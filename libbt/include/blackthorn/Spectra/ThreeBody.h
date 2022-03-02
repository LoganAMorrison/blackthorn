/// Defines a specialization to `DecaySpectrum` for the case of a decay into a
/// three-body final state.
///
/// The three-body final states are handled as follows.
///
/// Decay: For decays, we compute a coarse-grain energy histogram for each final
/// state and convolve it with the final state's decay spectrum:
///   (dN/dE)(E) = ∫dε P(ε) (dN/dE)(E, ε) ≈ ∑ᵢ P(εᵢ) (dN/dE)(E, εᵢ)
/// where P(ε) is the probability of having an energy ε.
///
/// Final-State Radiation: When computing the photon spectrum, we compute a
/// coarse-grain invariant-mass distribution for the three pairs (P1,P2),
/// (P1,P3) and (P2,P3). We then convolve these distributions with the
/// Altarelli-Parisi approximation. We end up computing the FSR twice for each
/// charged particle, so we average over the two to obtain a better result.
///   (dN/dE)(E) = ∫dsᵤᵥ P(sᵤᵥ) (APᵤ(E, sᵤᵥ) + APᵥ(E, sᵤᵥ))
///
/// Final-State is Product: In the case were a final-state particle is the
/// product, e.g. computing the neutrino spectrum for X -> nu + ell + ell, we
/// don't use the coarse-grain energy distribution since it isn't accurate
/// enough and is expensive to generate an accurate distribution. Instead, we
/// numerically integrate the squared matrix element using boost's
/// `gauss_kronrod`.

#ifndef BLACKTHORN_SPECTRA_THREE_BODY_H
#define BLACKTHORN_SPECTRA_THREE_BODY_H

#include "blackthorn/PhaseSpace.h"
#include "blackthorn/Spectra/Base.h"
#include "blackthorn/Spectra/Boost.h"
#include "blackthorn/Spectra/Conv.h"
#include "blackthorn/Spectra/Decay.h"
#include "blackthorn/Spectra/Rambo.h"
#include "blackthorn/Spectra/Splitting.h"
#include "blackthorn/Tensors.h"
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <iterator>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <tuple>

namespace blackthorn {

// ===========================================================================
// ---- Three-Body final state -----------------------------------------------
// ===========================================================================

template <class P1, class P2, class P3> class DecaySpectrum<P1, P2, P3> {
  using Momentum = LVector<double>;
  using Momenta = std::array<Momentum, 3>;
  using MSqrdType = std::function<double(const Momenta &)>;
  using EHist = EnergyHist<LinAxis>;

  static constexpr double M1 = P1::mass;
  static constexpr double M2 = P2::mass;
  static constexpr double M3 = P3::mass;
  static constexpr std::array<double, 3> fsp_masses = {M1, M2, M3};
  static constexpr unsigned int NBINS = 25;
  static constexpr size_t NEVENTS = 10000;
  static constexpr size_t NEVENTS_FSR = 1000;

  double p_parent_mass;
  MSqrdType p_msqrd;
  double p_width{};
  std::array<EHist, 3> p_edists{};
  EHist p_s_dist{};
  EHist p_t_dist{};
  EHist p_u_dist{};

  template <class P> auto get_masses() -> std::array<double, 3>;
  template <class P> auto get_squared_masses() -> std::array<double, 3>;

  template <class P>
  auto integration_t_bounds(double s) -> std::pair<double, double>;

  template <class P>
  auto fill_momenta(double s, double t, Momenta *momenta) -> void;

  auto rotate_momenta(Momenta *momenta) -> void;

  auto make_energy_distributions() -> void;
  auto make_invariant_mass_distributions() -> void;

  template <class P, class Product>
  auto p_decay_spectrum(double e, double ep) -> double;

  template <class P, class Product>
  auto convolved_decay(double e, const EHist &dist) -> double;
  template <class P>
  auto convolved_decay_neutrino(double e, const EHist &dist)
      -> NeutrinoSpectrum<double>;
  template <class F1, class F2>
  auto convolved_fsr(double e, const EHist &dist) -> double;

  auto decay_photon(double e) -> double;
  auto fsr_photon(double e) -> double;

  template <class P, class Product>
  auto dnde_rest_frame_final_state_product(double e) -> double;

  auto dnde_photon_rest_frame(double e) -> double;
  auto dnde_positron_rest_frame(double e) -> double;
  auto dnde_neutrino_rest_frame(double e) -> NeutrinoSpectrum<double>;
  template <class Product> auto dnde_rest_frame(double e) -> double;

  template <class Product> auto dnde(double e, double beta) -> double;
  template <class Product>
  auto dnde(const std::vector<double> &es, double beta) -> std::vector<double>;
  template <class Product>
  auto dnde(const py::array_t<double> &es, double beta) -> py::array_t<double>;

  template <class Product> auto dndx(double x, double beta) -> double;
  template <class Product>
  auto dndx(const std::vector<double> &xs, double beta) -> std::vector<double>;
  template <class Product>
  auto dndx(const py::array_t<double> &xs, double beta) -> py::array_t<double>;

public:
  /// Construct an decay spectrum object to compute the decay spectrum of a
  /// particle X into three particles A, B, and C and either a photon, neutrino
  /// or positron.
  ///
  /// @param parent_mass Mass of the decaying particle.
  /// @param msqrd Squared matrix element for X -> A + B + C
  /// @param msqrd_rad Squared matrix element for X -> A + B + C + photon.
  explicit DecaySpectrum(double parent_mass, MSqrdType msqrd = msqrd_flat<3>)
      : p_parent_mass(parent_mass), p_msqrd(std::move(msqrd)) {
    auto ms = get_masses<P1>();
    p_width = Rambo<3>::decay_width(p_msqrd, p_parent_mass, ms).first;
    make_energy_distributions();
    make_invariant_mass_distributions();
  }

  auto parent_mass() const -> const double & { return p_parent_mass; }
  auto parent_mass(double mass) -> void {
    p_parent_mass = mass;
    make_energy_distributions();
    make_invariant_mass_distributions();
  }

  /// Compute the spectrum dN/dE from the decay of a particle X into three
  /// particles A, B, C and a photon. X -> A + B + C + photon.
  ///
  /// @param e Energy of the product.
  /// @param beta Boost velocity of the decaying particle.
  auto dnde_photon(double e, double beta) -> double {
    return dnde<Photon>(e, beta);
  }
  auto dnde_photon(const std::vector<double> &es, double beta)
      -> std::vector<double> {
    return dnde<Photon>(es, beta);
  }
  auto dnde_photon(const py::array_t<double> &es, double beta)
      -> py::array_t<double> {
    return dnde<Photon>(es, beta);
  }

  /// Compute the spectrum dN/dx from the decay of a particle X into three
  /// particles A, B, C and a photon. X -> A + B + C + photon.
  ///
  /// @param x Scaled energy: x = 2 E / m, where m is the mass of X.
  /// @param beta Boost velocity of the decaying particle.
  auto dndx_photon(double x, double beta) -> double { // NOLINT
    return dndx<Photon>(x, beta);
  }
  auto dndx_photon(const std::vector<double> &xs, double beta)
      -> std::vector<double> { // NOLINT
    return dndx<Photon>(xs, beta);
  }
  auto dndx_photon(const py::array_t<double> &xs, double beta)
      -> py::array_t<double> { // NOLINT
    return dndx<Photon>(xs, beta);
  }

  /// Compute the spectrum dN/dE from the decay of a particle X into three
  /// particles A, B, C and a positron. X -> A + B + C + e^+.
  ///
  /// @param e Energy of the product.
  /// @param beta Boost velocity of the decaying particle.
  auto dnde_positron(double e, double beta) -> double {
    return dnde<Electron>(e, beta);
  }
  auto dnde_positron(const std::vector<double> &es, double beta)
      -> std::vector<double> {
    return dnde<Electron>(es, beta);
  }
  auto dnde_positron(const py::array_t<double> &es, double beta)
      -> py::array_t<double> {
    return dnde<Electron>(es, beta);
  }

  /// Compute the spectrum dN/dx from the decay of a particle X into three
  /// particles A, B, C and a positron. X -> A + B + C + e^+.
  ///
  /// @param x Scaled energy: x = 2 E / m, where m is the mass of X.
  /// @param beta Boost velocity of the decaying particle.
  auto dndx_positron(double x, double beta) -> double { // NOLINT
    return dndx<Electron>(x, beta);
  }
  auto dndx_positron(const std::vector<double> &xs, double beta)
      -> std::vector<double> { // NOLINT
    return dndx<Electron>(xs, beta);
  }
  auto dndx_positron(const py::array_t<double> &xs, double beta)
      -> py::array_t<double> { // NOLINT
    return dndx<Electron>(xs, beta);
  }

  /// Compute the spectrum dN/dE from the decay of a particle X into three
  /// particles A, B + C and a neutrino. X -> A + B + C + nu.
  ///
  /// @param e Energy of the product.
  /// @param beta Boost velocity of the decaying particle.
  auto dnde_neutrino(double e, double beta, Gen g) -> double {
    switch (g) {
    case Gen::Fst:
      return dnde<ElectronNeutrino>(e, beta);
    case Gen::Snd:
      return dnde<MuonNeutrino>(e, beta);
    default:
      return dnde<TauNeutrino>(e, beta);
    }
  }
  auto dnde_neutrino(const std::vector<double> &es, double beta, Gen g)
      -> std::vector<double> {
    switch (g) {
    case Gen::Fst:
      return dnde<ElectronNeutrino>(es, beta);
    case Gen::Snd:
      return dnde<MuonNeutrino>(es, beta);
    default:
      return dnde<TauNeutrino>(es, beta);
    }
  }
  auto dnde_neutrino(const py::array_t<double> &es, double beta, Gen g)
      -> py::array_t<double> {
    switch (g) {
    case Gen::Fst:
      return dnde<ElectronNeutrino>(es, beta);
    case Gen::Snd:
      return dnde<MuonNeutrino>(es, beta);
    default:
      return dnde<TauNeutrino>(es, beta);
    }
  }

  /// Compute the spectrum dN/dx from the decay of a particle X into three
  /// particles A, B and a neutrino. X -> A + B + nu.
  ///
  /// @param x Scaled energy: x = 2 E / m, where m is the mass of X.
  /// @param beta Boost velocity of the decaying particle.
  /// @param g Generation of the neutrino to compute spectrum of
  auto dndx_neutrino(double x, double beta, Gen g) -> double { // NOLINT
    switch (g) {
    case Gen::Fst:
      return dndx<ElectronNeutrino>(x, beta);
    case Gen::Snd:
      return dndx<MuonNeutrino>(x, beta);
    default:
      return dndx<TauNeutrino>(x, beta);
    }
  }
  auto dndx_neutrino(const std::vector<double> &xs, double beta, Gen g)
      -> std::vector<double> { // NOLINT
    switch (g) {
    case Gen::Fst:
      return dndx<ElectronNeutrino>(xs, beta);
    case Gen::Snd:
      return dndx<MuonNeutrino>(xs, beta);
    default:
      return dndx<TauNeutrino>(xs, beta);
    }
  }
  auto dndx_neutrino(const py::array_t<double> &xs, double beta, Gen g)
      -> py::array_t<double> { // NOLINT
    switch (g) {
    case Gen::Fst:
      return dndx<ElectronNeutrino>(xs, beta);
    case Gen::Snd:
      return dndx<MuonNeutrino>(xs, beta);
    default:
      return dndx<TauNeutrino>(xs, beta);
    }
  }
  /// Compute the spectrum dN/dx from the decay of a particle X into three
  /// particles A, B and a neutrino. X -> A + B + nu.
  ///
  /// @param x Scaled energy: x = 2 E / m, where m is the mass of X.
  /// @param beta Boost velocity of the decaying particle.
  /// @param g Generation of the neutrino to compute spectrum of
  auto dndx_neutrino(double x, double beta) -> std::array<double, 3> { // NOLINT
    return {dndx<ElectronNeutrino>(x, beta), dndx<MuonNeutrino>(x, beta),
            dndx<TauNeutrino>(x, beta)};
  }
  auto dndx_neutrino(const py::array_t<double> &xs, double beta)
      -> py::array_t<double> { // NOLINT
    auto x = xs.unchecked<1>();
    const size_t size = x.shape(0);
    std::vector<double> xv(size, 0.0);
    for (size_t i = 0; i < size; ++i) {
      xv[i] = x(i);
    }

    auto vresult = dndx_neutrino(xv, beta);

    size_t nnu = 3;
    py::array_t<double, py::array::c_style> result({nnu, size});
    auto r = result.mutable_unchecked<2>();
    for (size_t i = 0; i < x.shape(0); ++i) { // NOLINT
      r(0, i) = vresult[0][i];
      r(1, i) = vresult[1][i];
      r(2, i) = vresult[2][i];
    }
    return result;
  }
  auto dndx_neutrino(const std::vector<double> &xs, double beta)
      -> std::array<std::vector<double>, 3> { // NOLINT
    std::array<std::vector<double>, 3> result = {
        std::vector<double>(xs.size(), 0.0),
        std::vector<double>(xs.size(), 0.0),
        std::vector<double>(xs.size(), 0.0)};

    const auto fe = [&](double x) { return dndx<ElectronNeutrino>(x, beta); };
    const auto fm = [&](double x) { return dndx<ElectronNeutrino>(x, beta); };
    const auto ft = [&](double x) { return dndx<ElectronNeutrino>(x, beta); };
    result[0] = tools::vectorized_par(fe, xs);
    result[1] = tools::vectorized_par(fm, xs);
    result[2] = tools::vectorized_par(ft, xs);

    return result;
  }
};

// ===========================================================================
// ---- Private dN/dE and dN/dx functions ------------------------------------
// ===========================================================================

template <class P1, class P2, class P3>
template <class P>
auto DecaySpectrum<P1, P2, P3>::get_masses() -> std::array<double, 3> {
  if constexpr (std::is_same_v<P, P1>) {
    return {P1::mass, P2::mass, P3::mass};
  } else if constexpr (std::is_same_v<P, P2>) {
    return {P2::mass, P1::mass, P3::mass};
  } else {
    return {P3::mass, P1::mass, P2::mass};
  }
}
template <class P1, class P2, class P3>
template <class P>
auto DecaySpectrum<P1, P2, P3>::get_squared_masses() -> std::array<double, 3> {
  using tools::sqr;
  if constexpr (std::is_same_v<P, P1>) {
    return {sqr(P1::mass), sqr(P2::mass), sqr(P3::mass)};
  } else if constexpr (std::is_same_v<P, P2>) {
    return {sqr(P2::mass), sqr(P1::mass), sqr(P3::mass)};
  } else {
    return {sqr(P3::mass), sqr(P1::mass), sqr(P2::mass)};
  }
}

template <class P1, class P2, class P3>
template <class P>
auto DecaySpectrum<P1, P2, P3>::integration_t_bounds(double s)
    -> std::pair<double, double> {
  using tools::kallen_lambda;
  using tools::sqr;

  const double m02 = sqr(p_parent_mass);
  const auto [m12, m22, m32] = get_squared_masses<P>();

  const double m2sum = m02 + m12 + m22 + m32;
  const double p01 = 0.5 * sqrt(kallen_lambda(s, m02, m12) / s);
  const double p23 = 0.5 * sqrt(kallen_lambda(s, m22, m32) / s);
  const double f = 0.5 * (-s + m2sum - (m02 - m12) * (m22 - m32) / s);
  const double g = 2.0 * p01 * p23;
  return {f - g, f + g};
}

// We fill the momenta as follows:
//  - align p1 along z-axis: p1 = (e1;0,0,|p1|),
//  - pick p2z and p3z using:
//      cos(t2) * |p2| + cos(t3) * |p3| = |p1|.               [Eqn. 1]
//    Since the magnitudes of momenta are fixed (by knowining s,t and masses),
//    we find:
//      cos(t3) = (|p1| - cos(t2) * |p2|) / |p3|              [Eqn. 2]
//    Since -1 < cos(t3) < 1, this means that
//      (|p1| - |p3|) / |p2| < cos(t2) < (|p1| + |p3|) / |p2| [Eqn. 3]
//    We pick a random z = cos(t2) using these constraints.
//  - We then pick an angle between p2x and p2y:
//      tan(phi) = p2y / p2x.                                 [Eqn. 4]
//    p2 is now fixed.
//  - p3 is then fixed total momentum in CM frame:
//      P = (M;0,0,0) = p1 + p2 + p3                          [Eqn. 5]
template <class P1, class P2, class P3>
template <class P>
auto DecaySpectrum<P1, P2, P3>::fill_momenta(double s, double t, // NOLINT
                                             Momenta *momenta) -> void {
  using tools::energy_one_cm;
  using tools::sqr;
  using tools::two_body_three_momentum;

  static thread_local std::random_device rd{};
  static thread_local std::mt19937 generator{rd()};
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  const double m02 = sqr(p_parent_mass);
  const auto [m12, m22, m32] = get_squared_masses<P>();

  const double e1 = (m02 + m12 - s) / (2 * p_parent_mass);
  const double e2 = (m02 + m22 - t) / (2 * p_parent_mass);
  const double e3 = p_parent_mass - e1 - e2;

  const double p1 = std::sqrt(sqr(e1) - m12);
  const double p2 = std::sqrt(sqr(e2) - m22);
  const double p3 = std::sqrt(sqr(e3) - m32);

  // Put p1 along z-axis
  momenta->at(0).e() = e1;
  momenta->at(0).px() = 0.0;
  momenta->at(0).py() = 0.0;
  momenta->at(0).pz() = p1;

  // Bounds fixed by [Eqn. 3] above
  const double zmax = std::min(1.0, (p1 + p3) / p2);
  const double zmin = std::max((p1 - p3) / p2, -1.0);
  const double z = (zmax - zmin) * distribution(generator) + zmin;
  // Angle betwen p2x and p2y
  const double phi = 2.0 * M_PI * distribution(generator);

  momenta->at(1).e() = e2;
  momenta->at(1).px() = p2 * cos(phi) * sqrt(1.0 - z * z);
  momenta->at(1).py() = p2 * sin(phi) * sqrt(1.0 - z * z);
  momenta->at(1).pz() = p2 * z;

  // p3 fixed by momentum conservation
  momenta->at(2) = -momenta->at(1) - momenta->at(0);
  momenta->at(2).e() += p_parent_mass;
}

template <class P1, class P2, class P3>
auto DecaySpectrum<P1, P2, P3>::rotate_momenta(Momenta *momenta) -> void {
  static thread_local std::random_device rd{};
  static thread_local std::mt19937 generator{rd()};
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  const double tx = 2.0 * M_PI * distribution(generator);
  const double ty = 2.0 * M_PI * distribution(generator);
  const double tz = 2.0 * M_PI * distribution(generator);

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

// ===========================================================================
// ---- Constructing Distributions -------------------------------------------
// ===========================================================================

template <class P1, class P2, class P3>
auto DecaySpectrum<P1, P2, P3>::make_energy_distributions() -> void {
  p_edists = energy_distributions_linear(p_msqrd, p_parent_mass, fsp_masses,
                                         {NBINS, NBINS, NBINS}, NEVENTS);
}

template <class P1, class P2, class P3>
auto DecaySpectrum<P1, P2, P3>::make_invariant_mass_distributions() -> void {
  p_s_dist = invariant_mass_distributions_linear<1, 2>(
      p_msqrd, p_parent_mass, fsp_masses, NBINS, NEVENTS);
  p_t_dist = invariant_mass_distributions_linear<0, 2>(
      p_msqrd, p_parent_mass, fsp_masses, NBINS, NEVENTS);
  p_u_dist = invariant_mass_distributions_linear<0, 1>(
      p_msqrd, p_parent_mass, fsp_masses, NBINS, NEVENTS);
}

// ===========================================================================
// ---- Convienience Decay Spectrum Wrapper ----------------------------------
// ===========================================================================

template <class P1, class P2, class P3>
template <class P, class Product>
auto DecaySpectrum<P1, P2, P3>::p_decay_spectrum(double e, double ep)
    -> double {
  if constexpr (std::is_same_v<Product, Photon>) {
    return decay_spectrum<P>::dnde_photon(e, ep);
  } else if constexpr (std::is_same_v<Product, Electron>) {
    return decay_spectrum<P>::dnde_positron(e, ep);
  } else if constexpr (std::is_same_v<Product, ElectronNeutrino>) {
    return decay_spectrum<P>::dnde_neutrino(e, ep).electron;
  } else if constexpr (std::is_same_v<Product, MuonNeutrino>) {
    return decay_spectrum<P>::dnde_neutrino(e, ep).muon;
  } else if constexpr (std::is_same_v<Product, TauNeutrino>) {
    return decay_spectrum<P>::dnde_neutrino(e, ep).tau;
  }
  return 0.0;
}

// ===========================================================================
// ---- Convolutions ---------------------------------------------------------
// ===========================================================================

// Compute the convolved decay spectrum from particle P
// energy distribution `dist` and product energy `e`.
template <class P1, class P2, class P3>
template <class P, class Product>
auto DecaySpectrum<P1, P2, P3>::convolved_decay(double e, const EHist &dist)
    -> double {
  // Don't try to compute decay convolution if P is a stable particle
  if constexpr (is_stable_v<P>) {
    return 0.0;
  }

  double val = 0.0;

#pragma unroll NBINS
  for (const auto &&h : boost::histogram::indexed(dist)) {
    const double dp = *h * h.bin().width();
    const double em = h.bin().center();
    val += dp * p_decay_spectrum<P, Product>(e, em);
  }
  return val;
}

template <class P1, class P2, class P3>
template <class P>
auto DecaySpectrum<P1, P2, P3>::convolved_decay_neutrino(double e,
                                                         const EHist &dist)
    -> NeutrinoSpectrum<double> {
  NeutrinoSpectrum<double> val = {0.0, 0.0, 0.0};
  // Don't try to compute decay convolution if P is a stable particle
  if constexpr (is_stable_v<P>) {
    return val;
  }
#pragma unroll NBINS
  for (const auto &&h : boost::histogram::indexed(dist)) {
    const double dp = *h * h.bin().width();
    const double em = h.bin().center();
    const NeutrinoSpectrum<double> res =
        decay_spectrum<P>::dnde_neutrino(e, em);
    val.electron += dp * res.electron;
    val.muon += dp * res.muon;
    val.tau += dp * res.tau;
  }
  return val;
}

// Compute the convolved FSR spectrum from particles F1 and F2 with
// invariant-mass distribution `dist` and product energy `e`.
template <class P1, class P2, class P3>
template <class F1, class F2>
auto DecaySpectrum<P1, P2, P3>::convolved_fsr(double e, const EHist &dist)
    -> double {
  using tools::sqr;
  double val = 0.0;
#pragma unroll NBINS
  for (const auto &&h : boost::histogram::indexed(dist)) {
    const double dp = *h * h.bin().width();
    const double s = sqr(h.bin().center());
    val += dp * (dnde_photon_fsr<F1>(e, s) + dnde_photon_fsr<F2>(e, s));
  }
  return val;
}

// ===========================================================================
// ----  ------------------------------------
// ===========================================================================

template <class P1, class P2, class P3>
auto DecaySpectrum<P1, P2, P3>::decay_photon(double e) -> double {
  return convolved_decay<P1, Photon>(e, p_edists[0]) +
         convolved_decay<P2, Photon>(e, p_edists[1]) +
         convolved_decay<P3, Photon>(e, p_edists[2]);
}

// ===========================================================================
// ---- Computing Final-State Radiation --------------------------------------
// ===========================================================================

// Compute the FSR off the charged final state particles. Each final state
// particle is convolved over two different invariant-mass distributions.
//  Particle 1: t and u distributions
//  Particle 2: s and u distributions
//  Particle 3: s and t distributions
// where:
//  P = p1 + p2 + p3,
//  s = (P - p1)^2 = (p2 + p3)^2,
//  t = (P - p2)^2 = (p1 + p3)^2, and
//  u = (P - p3)^2 = (p1 + p2)^2
template <class P1, class P2, class P3>
auto DecaySpectrum<P1, P2, P3>::fsr_photon(double e) -> double {
  double res = 0.0;
  return 0.5 * (convolved_fsr<P2, P3>(e, p_s_dist) +
                convolved_fsr<P1, P3>(e, p_t_dist) +
                convolved_fsr<P1, P2>(e, p_u_dist));
}

// ===========================================================================
// ---- Rest Frame Spectra ---------------------------------------------------
// ===========================================================================

template <class P1, class P2, class P3>
template <class P, class Product>
auto DecaySpectrum<P1, P2, P3>::dnde_rest_frame_final_state_product(double e)
    -> double {
  using boost::math::quadrature::gauss_kronrod;
  using tools::sqr;

  if constexpr (std::is_same_v<P, Product>) {
    const double m0 = p_parent_mass;
    const double s = (sqr(m0) + sqr(P::mass) - 2 * m0 * e);
    const auto [m1, m2, m3] = get_masses<P>();
    static constexpr double epsilon = std::numeric_limits<double>::epsilon();

    if (s <= sqr(m2 + m3) + epsilon || sqr(m0 - m1) - epsilon <= s) {
      return 0.0;
    }

    Momenta momenta{};

    std::pair<double, double> tbs = integration_t_bounds<P>(s);
    const double pre = 1.0 / (256.0 * pow(m0 * M_PI, 3));

    auto f = [&](double t) {
      fill_momenta<P>(s, t, &momenta);
      return p_msqrd(momenta);
    };

    const auto result = pre * gauss_kronrod<double, 15>::integrate(
                                  f, tbs.first, tbs.second, 5, 1e-7);
    return result / (2.0 * p_parent_mass * p_width);
  } else {
    return 0.0;
  }
}

template <class P1, class P2, class P3>
auto DecaySpectrum<P1, P2, P3>::dnde_photon_rest_frame(double e) -> double {
  const double fsr = fsr_photon(e);
  const double decay = decay_photon(e);
  const double final_state =
      (dnde_rest_frame_final_state_product<P1, Photon>(e) +
       dnde_rest_frame_final_state_product<P2, Photon>(e) +
       dnde_rest_frame_final_state_product<P3, Photon>(e));
  return fsr + decay + final_state;
}

template <class P1, class P2, class P3>
auto DecaySpectrum<P1, P2, P3>::dnde_positron_rest_frame(double e) -> double {
  using P = Electron;
  const double decay = convolved_decay<P1, P>(e, p_edists[0]) +
                       convolved_decay<P2, P>(e, p_edists[1]) +
                       convolved_decay<P3, P>(e, p_edists[2]);
  const double final_state = (dnde_rest_frame_final_state_product<P1, P>(e) +
                              dnde_rest_frame_final_state_product<P2, P>(e) +
                              dnde_rest_frame_final_state_product<P3, P>(e));
  return decay + final_state;
}

template <class P1, class P2, class P3>
auto DecaySpectrum<P1, P2, P3>::dnde_neutrino_rest_frame(double e)
    -> NeutrinoSpectrum<double> {
  using RetType = NeutrinoSpectrum<double>;
  const RetType res1 = convolved_decay_neutrino<P1>(e, p_edists[0]);
  const RetType res2 = convolved_decay_neutrino<P2>(e, p_edists[1]);
  const RetType res3 = convolved_decay_neutrino<P3>(e, p_edists[2]);

  const double final_state_e =
      (dnde_rest_frame_final_state_product<P1, ElectronNeutrino>(e) +
       dnde_rest_frame_final_state_product<P2, ElectronNeutrino>(e) +
       dnde_rest_frame_final_state_product<P3, ElectronNeutrino>(e));
  const double final_state_mu =
      (dnde_rest_frame_final_state_product<P1, MuonNeutrino>(e) +
       dnde_rest_frame_final_state_product<P2, MuonNeutrino>(e) +
       dnde_rest_frame_final_state_product<P3, MuonNeutrino>(e));
  const double final_state_tau =
      (dnde_rest_frame_final_state_product<P1, TauNeutrino>(e) +
       dnde_rest_frame_final_state_product<P2, TauNeutrino>(e) +
       dnde_rest_frame_final_state_product<P3, TauNeutrino>(e));

  return {
      res1.electron + res2.electron + res3.electron + final_state_e,
      res1.muon + res2.muon + res3.muon + final_state_mu,
      res1.tau + res2.tau + res3.tau + final_state_tau,
  };
}

template <class P1, class P2, class P3>
template <class Product>
auto DecaySpectrum<P1, P2, P3>::dnde_rest_frame(double e) -> double {
  if constexpr (std::is_same_v<Product, Photon>) {
    return dnde_photon_rest_frame(e);
  } else if constexpr (std::is_same_v<Product, Electron>) {
    return dnde_positron_rest_frame(e);
  } else if constexpr (std::is_same_v<Product, ElectronNeutrino>) {
    return dnde_neutrino_rest_frame(e).electron;
  } else if constexpr (std::is_same_v<Product, MuonNeutrino>) {
    return dnde_neutrino_rest_frame(e).muon;
  } else if constexpr (std::is_same_v<Product, TauNeutrino>) {
    return dnde_neutrino_rest_frame(e).tau;
  }

  return 0.0;
}

// ===========================================================================
// ---- Private dN/dE and dN/dx functions ------------------------------------
// ===========================================================================

template <class P1, class P2, class P3>
template <class Product>
auto DecaySpectrum<P1, P2, P3>::dnde(double e, double beta) -> double {
  using boost::math::quadrature::gauss_kronrod;
  static constexpr unsigned int GK_N = 15;
  static constexpr unsigned int GK_MAX_LIMIT = 7;
  constexpr double md = Product::mass;

  if (beta < 0.0 || 1.0 < beta || e < md) {
    return 0.0;
  }

  // If we are sufficiently close to the parent's rest-frame, use the
  // rest-frame result.
  if (beta < std::numeric_limits<double>::epsilon()) {
    return dnde_rest_frame<Product>(e);
  }

  const double gamma = 1.0 / std::sqrt(1.0 - beta * beta);

  const auto integrand = [&](double e1) -> double {
    if constexpr (is_massless<Product>::value) {
      return dnde_rest_frame<Product>(e1) / e1;
    } else {
      return dnde_rest_frame<Product>(e1) / sqrt(e1 * e1 - md * md);
    }
  };

  const double k = is_massless<Product>::value ? e : sqrt(e * e - md * md);
  const double eminus = gamma * (e - beta * k);
  const double eplus = gamma * (e + beta * k);
  const double pre = 1.0 / (2.0 * gamma * beta);

  return pre * gauss_kronrod<double, GK_N>::integrate(integrand, eminus, eplus,
                                                      GK_MAX_LIMIT, 1e-6);
}

template <class P1, class P2, class P3>
template <class Product>
auto DecaySpectrum<P1, P2, P3>::dndx(double x, double beta) // NOLINT
    -> double {
  if (beta < 0.0 || 1.0 < beta) {
    return 0.0;
  }

  const double gamma = 1.0 / std::sqrt(1.0 - beta * beta);
  const double ep = p_parent_mass * gamma;
  const double e = x * ep / 2.0;
  const double l_dnde = dnde<Product>(e, beta);

  return l_dnde * ep / 2.0;
}

template <class P1, class P2, class P3>
template <class Product>
auto DecaySpectrum<P1, P2, P3>::dnde(const std::vector<double> &es, double beta)
    -> std::vector<double> {
  const auto f = [&](double e) { return dnde<Product>(e, beta); };
  return tools::vectorized_par(f, es);
}

template <class P1, class P2, class P3>
template <class Product>
auto DecaySpectrum<P1, P2, P3>::dnde(const py::array_t<double> &e, double beta)
    -> py::array_t<double> {
  auto es = e.unchecked<1>();
  std::vector<double> es_(es.size(), 0.0);
  for (size_t i = 0; i < es.size(); ++i) {
    es_[i] = es(i);
  }
  auto res_ = dnde<Product>(es_, beta);
  auto result = py::array_t<double>(e.request());
  auto r = result.mutable_unchecked<1>();
  for (size_t i = 0; i < es.size(); ++i) {
    r(i) = res_[i];
  }
  return result;
}

template <class P1, class P2, class P3>
template <class Product>
auto DecaySpectrum<P1, P2, P3>::dndx(const std::vector<double> &xs, double beta)
    -> std::vector<double> { // NOLINT
  const auto f = [&](double x) { return dndx<Product>(x, beta); };
  return tools::vectorized_par(f, xs);
}

template <class P1, class P2, class P3>
template <class Product>
auto DecaySpectrum<P1, P2, P3>::dndx(const py::array_t<double> &x, double beta)
    -> py::array_t<double> {
  auto xs = x.unchecked<1>();
  std::vector<double> xs_(xs.size(), 0.0);
  for (size_t i = 0; i < xs.size(); ++i) {
    xs_[i] = xs(i);
  }
  std::vector<double> res_ = dndx<Product>(xs_, beta);
  auto result = py::array_t<double>(x.request());
  auto r = result.mutable_unchecked<1>();
  for (size_t i = 0; i < xs.size(); ++i) {
    r(i) = res_[i];
  }
  return result;
}

// template <class P1, class P2, class P3>
// template <class Product>
// auto DecaySpectrum<P1, P2, P3>::dnde(const py::array_t<double> &es, double
// beta)
//     -> py::array_t<double> {
//   auto e = es.unchecked<1>();
//   auto result = py::array_t<double>(es.request());
//   auto res = result.mutable_unchecked<1>();

//   for (py::ssize_t i = 0; i < e.shape(0); ++i) { // NOLINT
//     res(i) = dnde<Product>(e(i), beta);          // NOLINT
//   }

//   return result;
// }

// template <class P1, class P2, class P3>
// template <class Product>
// auto DecaySpectrum<P1, P2, P3>::dndx(const py::array_t<double> &xs, double
// beta)
//     -> py::array_t<double> {
//   auto x = xs.unchecked<1>();
//   auto result = py::array_t<double>(xs.request());
//   auto res = result.mutable_unchecked<1>();

//   for (py::ssize_t i = 0; i < x.shape(0); ++i) { // NOLINT
//     res(i) = dndx<Product>(x(i), beta);          // NOLINT
//   }

//   return result;
// }

} // namespace blackthorn

#endif // BLACKTHORN_SPECTRA_THREE_BODY_H
