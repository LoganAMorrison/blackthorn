#ifndef BLACKTHORN_SPECTRA_TWO_BODY_H
#define BLACKTHORN_SPECTRA_TWO_BODY_H

#include "blackthorn/PhaseSpace.h"
#include "blackthorn/Spectra/Base.h"
#include "blackthorn/Spectra/Boost.h"
#include "blackthorn/Spectra/Conv.h"
#include "blackthorn/Spectra/Decay.h"
#include "blackthorn/Spectra/Rambo.h"
#include "blackthorn/Spectra/Splitting.h"
#include "blackthorn/Tensors.h"
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <tuple>

namespace blackthorn {

namespace py = pybind11;

// ===========================================================================
// ---- Two-Body Final State Specialization ----------------------------------
// ===========================================================================

/// Class for computing spectra from the decay of a particle into a two-body
/// final state.
template <class P1, class P2> class DecaySpectrum<P1, P2> {
  double p_parent_mass;

protected:
  static constexpr double M1 = P1::mass;
  static constexpr double M2 = P2::mass;

  virtual auto fsr(double e) -> double {
    using tools::sqr;
    const double m = p_parent_mass;
    if (m < M1 + M2) {
      return 0.0;
    }

    return dnde_photon_fsr<P1>(e, m * m) + dnde_photon_fsr<P2>(e, m * m);
  }

  auto decay_photon(double e) -> double {
    const double m = p_parent_mass;
    if (m < M1 + M2) {
      return 0.0;
    }
    const double e1 = tools::energy_one_cm(m, M1, M2);
    const double e2 = tools::energy_one_cm(m, M2, M1);
    return decay_spectrum<P1>::dnde_photon(e, e1) +
           decay_spectrum<P2>::dnde_photon(e, e2);
  }

  auto dnde_photon_rest_frame(double e) -> double {
    return decay_photon(e) + fsr(e);
  }

  auto dnde_positron_rest_frame(double e) -> double {
    const double m = p_parent_mass;
    if (m < M1 + M2) {
      return 0.0;
    }
    const double e1 = tools::energy_one_cm(m, M1, M2);
    const double e2 = tools::energy_one_cm(m, M2, M1);
    return decay_spectrum<P1>::dnde_positron(e, e1) +
           decay_spectrum<P2>::dnde_positron(e, e2);
  }

  auto dnde_neutrino_rest_frame(double e) -> NeutrinoSpectrum<double> {
    const double m = p_parent_mass;
    if (m < M1 + M2) {
      return {0.0, 0.0, 0.0};
    }
    const double e1 = tools::energy_one_cm(m, M1, M2);
    const double e2 = tools::energy_one_cm(m, M2, M1);
    const NeutrinoSpectrum<double> res1 =
        decay_spectrum<P1>::dnde_neutrino(e, e1);
    const NeutrinoSpectrum<double> res2 =
        decay_spectrum<P2>::dnde_neutrino(e, e2);
    return {res1.electron + res2.electron, res1.muon + res2.muon, 0.0};
  }

  template <class Product> auto dnde_rest_frame(double e) -> double {
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

  template <class Product, class F1>
  auto dnde_delta_function_single(double e2, double e0, double beta) -> double {
    using tools::energy_one_cm;
    static constexpr double eps = std::numeric_limits<double>::epsilon();
    const double gamma = sqrt(1.0 - tools::sqr(beta));

    if constexpr (std::is_same_v<F1, Product>) {
      if (e2 - F1::mass < eps) {
        return 0.0;
      }

      const double k2 = sqrt(tools::sqr(e2) - tools::sqr(F1::mass));
      const double e1plus = gamma * (e2 + beta * k2);
      const double e1minus = std::max(gamma * (e2 - beta * k2), F1::mass);

      if (e0 <= e1minus || e1plus <= e0 || e1plus - e1minus < eps) {
        return 0.0;
      }

      const double k0 = sqrt(tools::sqr(e0) - tools::sqr(F1::mass));
      return 1.0 / k0;
    } else {
      return 0.0;
    }
  }

  template <class Product>
  auto dnde_delta_function(double e, double beta) -> double {
    using tools::energy_one_cm;
    const double e01 = energy_one_cm(p_parent_mass, P1::mass, P2::mass);
    const double e02 = energy_one_cm(p_parent_mass, P2::mass, P1::mass);
    return (dnde_delta_function_single<Product, P1>(e, e01, beta) +
            dnde_delta_function_single<Product, P2>(e, e02, beta));
  }

  template <class Product> auto dnde(double e, double beta) -> double {
    using boost::math::quadrature::gauss_kronrod;
    using tools::sqr;
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

    const double delta = dnde_delta_function<Product>(e, beta);

    const double dec_fsr = gauss_kronrod<double, GK_N>::integrate(
        integrand, eminus, eplus, GK_MAX_LIMIT, 1e-6);

    return pre * (delta + dec_fsr);
  }

  template <class Product>
  auto dndx(double x, double beta) -> double { // NOLINT
    if (beta < 0.0 || 1.0 < beta) {
      return 0.0;
    }

    const double gamma = 1.0 / std::sqrt(1.0 - beta * beta);
    const double ep = p_parent_mass * gamma;
    const double e = x * ep / 2.0;
    const double l_dnde = dnde<Product>(e, beta);

    return l_dnde * ep / 2.0;
  }

  template <class Product>
  auto dnde(const std::vector<double> &es, double beta) -> std::vector<double> {
    const auto f = [&](double e) { return dnde<Product>(e, beta); };
    return tools::vectorized_par(f, es);
  }
  // template <class Product>
  // auto dnde(const py::array_t<double> &es, double beta) ->
  // py::array_t<double> {
  //   auto e = es.unchecked<1>();
  //   auto result = py::array_t<double>(es.request());
  //   auto res = result.mutable_unchecked<1>();

  //   for (py::ssize_t i = 0; i < e.shape(0); ++i) { // NOLINT
  //     res(i) = dnde<Product>(e(i), beta);          // NOLINT
  //   }

  //   return result;
  // }

  template <class Product>
  auto dnde(const py::array_t<double> &e, double beta) -> py::array_t<double> {
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

  template <class Product>
  auto dndx(const std::vector<double> &xs, double beta)
      -> std::vector<double> { // NOLINT
    const auto f = [&](double x) { return dndx<Product>(x, beta); };
    return tools::vectorized_par(f, xs);
  }
  // template <class Product>
  // auto dndx(const py::array_t<double> &xs, double beta) ->
  // py::array_t<double> {
  //   auto x = xs.unchecked<1>();
  //   auto result = py::array_t<double>(xs.request());
  //   auto res = result.mutable_unchecked<1>();

  //   for (py::ssize_t i = 0; i < x.shape(0); ++i) { // NOLINT
  //     res(i) = dndx<Product>(x(i), beta);          // NOLINT
  //   }

  //   return result;
  // }

  template <class Product>
  auto dndx(const py::array_t<double> &x, double beta) -> py::array_t<double> {
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

public:
  explicit DecaySpectrum(double parent_mass) : p_parent_mass(parent_mass) {}

  /// Compute the spectrum dN/dE from the decay of a particle X into a two
  /// particles A, B and a photon. X -> A + B + photon.
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

  /// Compute the spectrum dN/dx from the decay of a particle X into a two
  /// particles A, B and a photon. X -> A + B + photon.
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

  /// Compute the spectrum dN/dE from the decay of a particle X into a two
  /// particles A, B and a positron. X -> A + B + e^+.
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

  /// Compute the spectrum dN/dx from the decay of a particle X into a two
  /// particles A, B and a positron. X -> A + B + e^+.
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

  // /// Compute the spectrum dN/dE from the decay of a particle X into a two
  // /// particles A, B and a neutrino. X -> A + B + nu.
  // ///
  // /// @param e Energy of the product.
  // /// @param beta Boost velocity of the decaying particle.
  // auto dnde_neutrino(double e, double beta, Gen g) -> double {
  //   switch (g) {
  //   case Gen::Fst:
  //     return dnde<ElectronNeutrino>(e, beta);
  //   case Gen::Snd:
  //     return dnde<MuonNeutrino>(e, beta);
  //   default:
  //     return dnde<TauNeutrino>(e, beta);
  //   }
  // }
  // auto dnde_neutrino(const std::vector<double> &es, double beta, Gen g)
  //     -> std::vector<double> {
  //   switch (g) {
  //   case Gen::Fst:
  //     return dnde<ElectronNeutrino>(es, beta);
  //   case Gen::Snd:
  //     return dnde<MuonNeutrino>(es, beta);
  //   default:
  //     return dnde<TauNeutrino>(es, beta);
  //   }
  // }
  // auto dnde_neutrino(const py::array_t<double> &es, double beta, Gen g)
  //     -> py::array_t<double> {
  //   switch (g) {
  //   case Gen::Fst:
  //     return dnde<ElectronNeutrino>(es, beta);
  //   case Gen::Snd:
  //     return dnde<MuonNeutrino>(es, beta);
  //   default:
  //     return dnde<TauNeutrino>(es, beta);
  //   }
  // }
  // auto dnde_neutrino(const py::array_t<double> &es, double beta)
  //     -> py::array_t<double> { // NOLINT
  //   auto e = es.unchecked<1>();
  //   size_t n1 = e.shape(0);
  //   size_t n2 = 3;
  //   py::array_t<double, py::array::c_style> result({n2, n1});
  //   auto r = result.mutable_unchecked<2>();

  //   for (size_t i = 0; i < e.shape(0); ++i) { // NOLINT
  //     r(0, i) = dnde<ElectronNeutrino>(e(i), beta);
  //     r(1, i) = dnde<MuonNeutrino>(e(i), beta);
  //     r(2, i) = dnde<TauNeutrino>(e(i), beta);
  //   }

  //   return result;
  // }
  // auto dnde_neutrino(double e, double beta) -> std::array<double, 3> { //
  // NOLINT
  //   return {dnde<ElectronNeutrino>(e, beta), dnde<MuonNeutrino>(e, beta),
  //           dnde<TauNeutrino>(e, beta)};
  // }
  // auto dnde_neutrino(const std::vector<double> &e, double beta)
  //     -> std::array<std::vector<double>, 3> { // NOLINT
  //   return {dnde<ElectronNeutrino>(e, beta), dnde<MuonNeutrino>(e, beta),
  //           dnde<TauNeutrino>(e, beta)};
  // }

  // /// Compute the spectrum dN/dx from the decay of a particle X into a two
  // /// particles A, B and a neutrino. X -> A + B + nu.
  // ///
  // /// @param x Scaled energy: x = 2 E / m, where m is the mass of X.
  // /// @param beta Boost velocity of the decaying particle.
  // auto dndx_neutrino(double x, double beta, Gen g) -> double { // NOLINT
  //   switch (g) {
  //   case Gen::Fst:
  //     return dndx<ElectronNeutrino>(x, beta);
  //   case Gen::Snd:
  //     return dndx<MuonNeutrino>(x, beta);
  //   default:
  //     return dndx<TauNeutrino>(x, beta);
  //   }
  // }
  // auto dndx_neutrino(const std::vector<double> &xs, double beta, Gen g)
  //     -> std::vector<double> { // NOLINT
  //   switch (g) {
  //   case Gen::Fst:
  //     return dndx<ElectronNeutrino>(xs, beta);
  //   case Gen::Snd:
  //     return dndx<MuonNeutrino>(xs, beta);
  //   default:
  //     return dndx<TauNeutrino>(xs, beta);
  //   }
  // }
  // auto dndx_neutrino(const py::array_t<double> &xs, double beta, Gen g)
  //     -> py::array_t<double> { // NOLINT
  //   switch (g) {
  //   case Gen::Fst:
  //     return dndx<ElectronNeutrino>(xs, beta);
  //   case Gen::Snd:
  //     return dndx<MuonNeutrino>(xs, beta);
  //   default:
  //     return dndx<TauNeutrino>(xs, beta);
  //   }
  // }
  // auto dndx_neutrino(const py::array_t<double> &xs, double beta)
  //     -> py::array_t<double> { // NOLINT
  //   auto x = xs.unchecked<1>();
  //   size_t n1 = x.shape(0);
  //   size_t n2 = 3;
  //   py::array_t<double, py::array::c_style> result({n2, n1});
  //   auto r = result.mutable_unchecked<2>();

  //   for (size_t i = 0; i < x.shape(0); ++i) { // NOLINT
  //     r(0, i) = dndx<ElectronNeutrino>(x(i), beta);
  //     r(1, i) = dndx<MuonNeutrino>(x(i), beta);
  //     r(2, i) = dndx<TauNeutrino>(x(i), beta);
  //   }

  //   return result;
  // }
  // auto dndx_neutrino(double x, double beta) -> std::array<double, 3> { //
  // NOLINT
  //   return {dnde<ElectronNeutrino>(x, beta), dnde<MuonNeutrino>(x, beta),
  //           dnde<TauNeutrino>(x, beta)};
  // }
  // auto dndx_neutrino(const std::vector<double> &x, double beta)
  //     -> std::array<std::vector<double>, 3> { // NOLINT
  //   return {dndx<ElectronNeutrino>(x, beta), dndx<MuonNeutrino>(x, beta),
  //           dndx<TauNeutrino>(x, beta)};
  // }

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

} // namespace blackthorn

#endif // BLACKTHORN_SPECTRA_TWO_BODY_H
