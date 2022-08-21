#ifndef BLACKTHORN_SPECTRA_BASE_H
#define BLACKTHORN_SPECTRA_BASE_H

#include "blackthorn/PhaseSpace.h"
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

static constexpr auto zero(double /*x*/) -> double { return 0.0; }

/// Compute the FSR spectrum dN/dx off a charged particle.
///
/// @param x Value of 2*energy/sqrt(s)
/// @param s Squared center-of-mass energy. For decays, sqrt(s) = mass. For
/// annihilations, sqrt(s) = 2m.
template <class F>
static auto dndx_photon_fsr(double x, double s) -> double { // NOLINT
  using tools::sqr;
  static constexpr double mf = F::mass;
  static constexpr double qf = quantum_numbers<F>::charge();

  if constexpr (qf == 0.0) {
    return 0.0;
  } else {
    const double pre = sqr(qf) * StandardModel::alpha_em / (2.0 * M_PI);
    const double xm = 1.0 - x;

    double kernel = 0.0;
    if constexpr (is_fermion<F>::value) {
      const double mu2 = sqr(mf) / s;
      if (M_E * mu2 < xm) {
        kernel = (1.0 + sqr(xm)) / x * (log(s * xm / sqr(mf)) - 1.0);
      }
    } else if constexpr (is_scalar_boson<F>::value) {
      const double mu2 = sqr(mf) / s;
      if (M_E * mu2 < xm) {
        kernel = 2.0 * xm / x * (log(s * xm / sqr(mf)) - 1.0);
      }
    } else if constexpr (is_vector_boson<F>::value) {
      const double y = s * sqr(xm) / sqr(2 * mf);
      const double lf1 = log(y) + 2.0 * log(1.0 - sqrt(1.0 - 1.0 / y));
      const double lf2 = log(s / sqr(mf));
      kernel = 2 * (x / xm * lf1 + xm / x * lf2 + x * xm * lf2);
    }

    return pre * kernel;
  }
}

template <class F>
static auto dnde_photon_fsr(double e, double s) -> double { // NOLINT
  const double e_to_x = 2.0 / sqrt(s);
  const double x = e * e_to_x;
  return dndx_photon_fsr<F>(x, s) * e_to_x;
}

// ===========================================================================
// ---- Templates ------------------------------------------------------------
// ===========================================================================

/// Primary template class for computing decay spectra into photons.
template <class... FinalStates> class DecaySpectrum {
  using FinalStatesTuple = std::tuple<FinalStates...>;
  static constexpr size_t N = std::tuple_size<FinalStatesTuple>::value;
};

} // namespace blackthorn

#endif // BLACKTHORN_SPECTRA_BASE_H
