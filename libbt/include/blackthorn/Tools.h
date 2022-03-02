#ifndef BLACKTHORN_TOOLS_H
#define BLACKTHORN_TOOLS_H

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <execution>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

namespace blackthorn::tools {

/// 1 / (128 pi^3)
static constexpr double k1_128PI3 = 0.25196511275937101e-3;

static constexpr std::complex<double> im = std::complex<double>{0.0, 1.0};

/**
 * Helper function to square a number.
 */
template <typename T> constexpr auto sqr(const T &x) -> decltype(x * x) {
  return x * x;
}

/**
 * Helper function to square the absolute value of a number.
 */
template <typename T>
constexpr auto abs2(const T &x) -> decltype(std::abs(x) * std::abs(x)) {
  return std::abs(x) * std::abs(x);
}

// ===========================================================================
// ---- Array Creation -------------------------------------------------------
// ===========================================================================

/**
 * Create an vector of linearly space values.
 *
 * @tparam N number of points in the range.
 * @param start first number in the range.
 * @param end last number in the range.
 */
template <size_t N>
auto linspace(const double start, const double end) -> std::array<double, N> {
  std::array<double, N> lst{};
  const double step = (end - start) / static_cast<double>(N - 1);
  std::generate(lst.begin(), lst.end(), [&, n = 0]() mutable {
    const double val = step * static_cast<double>(n) + start;
    n++;
    return val;
  });
  return lst;
}

/**
 * Create an vector of `n` logarithmically spaced points starting from
 * `pow(base, start)` and ending with `pow(base, end)`.
 *
 * @tparam N number of points in the range.
 * @param start power of the first number in the range.
 * @param end power of the last number in the range.
 * @param base base of the start and end points.
 */
template <size_t N>
auto logspace(const double start, const double end, const double base = 10.0)
    -> std::array<double, N> {
  std::array<double, N> lst{};
  const double step = (end - start) / static_cast<double>(N - 1);
  std::generate(lst.begin(), lst.end(), [&, n = 0]() mutable {
    const double val = step * static_cast<double>(n) + start;
    n++;
    return std::pow(base, val);
  });
  return lst;
}

/**
 * Create an vector of `N` logarithmically spaced points starting from `start`
 * and ending with `end`.
 *
 * @tparam N number of points in the range.
 * @param start first number in the range.
 * @param end last number in the range.
 */
template <size_t N>
auto geomspace(const double start, const double end) -> std::array<double, N> {
  return logspace<N>(log10(start), log10(end), 10.0);
}

/**
 * Create an vector of linearly space values.
 *
 * @param start first number in the range.
 * @param end last number in the range.
 * @param n number of points in the range.
 */
auto linspace(double start, double end, size_t n) -> std::vector<double>;

/**
 * Create an vector of `n` logarithmically spaced points starting from `pow(10,
 * start)` and ending with `pow(10, end)`.
 *
 * @param start log10 of the first number in the range.
 * @param end log10 of the last number in the range.
 * @param n number of points in the range.
 */
auto logspace(double start, double end, size_t n, double base = 10.0)
    -> std::vector<double>;

/**
 * Create an vector of `n` logarithmically spaced points starting from `start`
 * and ending with `end`.
 *
 * @param start first number in the range.
 * @param end last number in the range.
 * @param n number of points in the range.
 */
auto geomspace(double start, double end, size_t n) -> std::vector<double>;

template <class F>
auto vectorized(F f, const std::vector<double> &xs) -> std::vector<double> {
  std::vector<double> res(xs.size(), 0.0);
  std::transform(xs.begin(), xs.end(), res.begin(), f);
  return res;
}

// ===========================================================================
// ---- Vectorization --------------------------------------------------------
// ===========================================================================

template <class F>
auto vectorized_par(F f, const std::vector<double> &xs) -> std::vector<double> {
  std::vector<double> res(xs.size(), 0.0);
  std::transform(std::execution::par, xs.begin(), xs.end(), res.begin(), f);
  return res;
}

template <class F>
auto vectorized(F f, const py::array_t<double> &xs) -> py::array_t<double> {
  py::buffer_info bufe = xs.request();
  if (bufe.ndim != 1) {
    throw std::runtime_error("Number of dimensions must be 1.");
  }
  auto spec = py::array_t<double>(bufe.size);
  py::buffer_info bufs = spec.request();

  auto *ptre = static_cast<double *>(bufe.ptr);
  auto *ptrs = static_cast<double *>(bufs.ptr);

  for (size_t i = 0; i < bufe.shape[0]; ++i) { // NOLINT
    ptrs[i] = f(ptre[i]);                      // NOLINT
  }
  return spec;
}

// =========================================
// ---- Numpy ------------------------------
// =========================================

/// Check that the buffer has length 1 and return buffer into.
auto get_buffer_and_check_dim(const py::array_t<double> &xs) -> py::buffer_info;

/// Return a numpy array with zeros with the same shape as input array.
auto zeros_like(const py::array_t<double> &xs) -> py::array_t<double>;

auto zeros_like(const std::vector<double> &xs) -> std::vector<double>;

inline auto zeros_like(double /*x*/) -> double { return 0.0; };

// =========================================
// ---- Compile-time non-negative power ----
// =========================================

// Even power
template <unsigned int N, unsigned int M = N % 2> struct positive_power {
  template <typename T> static auto result(T base) -> T {
    T power = positive_power<N / 2>::result(base);
    return power * power;
  }
};

// Odd power
template <unsigned int N> struct positive_power<N, 1> {
  template <typename T> static auto result(T base) -> T {
    T power = positive_power<N / 2>::result(base);
    return base * power * power;
  }
};

template <> struct positive_power<1, 1> { // NOLINT
  template <typename T> static auto result(T base) -> T { return base; }
};

template <> struct positive_power<0, 0> { // NOLINT
  template <typename T> static auto result(T /*base*/) -> T { return T(1); }
};

// Compute base^p.
template <unsigned int N, typename T> static auto powi(T base) -> T {
  return positive_power<N>::result(base);
}

// ================================
// ---- Compile-Time Factorial ----
// ================================

template <unsigned int N> struct factorial {
  static auto result() -> unsigned int {
    return factorial<N - 1>::result() * N;
  }
};

template <> struct factorial<1> { // NOLINT
  static auto result() -> unsigned int { return 1; }
};

template <> struct factorial<0> { // NOLINT
  static auto result() -> unsigned int { return 1; }
};

template <unsigned int N> auto fact() -> unsigned int {
  return factorial<N>::result();
}

// ===========================================================================
// ---- Two-Point Disc Funcitons ---------------------------------------------
// ===========================================================================

/**
 * Structure holding functions to compute the discontinuities of a scalar
 * two-point function.
 */
struct disc_scalar { // NOLINT
  static auto fermion(double s, double m) -> double {
    if (s > 4 * m * m) {
      return (s - 4 * m * m) * sqrt(s * (s - 4 * m * m)) / (8 * M_PI * s);
    }
    return 0.0;
  }
  static auto scalar(double s, double m) -> double {
    if (s > 4 * m * m) {
      return sqrt(s * (s - 4 * m * m)) / (16 * M_PI * s);
    }
    return 0.0;
  }
  static auto vector(double s, double m) -> double {
    const double mu = (s / (m * m));
    if (4 * mu * mu < 1) {
      return (12.0 - 4.0 * mu + mu * mu) * sqrt(1.0 - 4.0 * mu * mu) /
             (64 * M_PI);
    }
    return 0.0;
  }
};

/**
 * Structure holding functions to compute the discontinuities of a vector
 * two-point function.
 */
struct disc_vector { // NOLINT
  static auto fermion(double s, double m) -> double {
    if (s > 4 * m * m) {
      return (s + 2 * m * m) * sqrt(s * (s - 4 * m * m)) / (12.0 * M_PI * s);
    }
    return 0.0;
  }
};

// ===========================================================================
// ---- Special Functions ----------------------------------------------------
// ===========================================================================

/**
 * Compute the dilogarithm.
 */
auto dilog(const std::complex<double> &z) -> std::complex<double>;

/**
 * Compute the Scalar C0 function C0(s1, s12, s2; m1, m2, m3) w/ s1=s12=0 and
 * m2=m3.
 */
auto scalar_c0_1(double s2, double m1, double m2) -> std::complex<double>;

/**
 * Compute the Scalar C0 function C0(s1, s12, s2; m1, m2, m3) w/ s1=s12=0 and
 * m1=m2.
 */
auto scalar_c0_2(double s2, double m1, double m3) -> std::complex<double>;

// ===========================================================================
// ---- Statistics -----------------------------------------------------------
// ===========================================================================

auto mean_sum_sqrs_welford(const std::vector<double> &wgts)
    -> std::tuple<double, double, double>;

auto mean_var_welford(const std::vector<double> &wgts)
    -> std::pair<double, double>;

auto mean_var(const std::vector<double> &wgts) -> std::pair<double, double>;

// ===========================================================================
// ---- Kinematic Functions --------------------------------------------------
// ===========================================================================

/**
 * Compute the gamma boost factor.
 *
 * @param e energy of the particle
 * @param m mass of the particle
 */
static constexpr auto gamma(double e, double m) -> double { return e / m; }

static auto gamma(double beta) -> double { return 1.0 / sqrt(1.0 - sqr(beta)); }

/**
 * @breif Compute the velocity of a particle given its energy and mass.
 * @param e Energy of the particle
 * @param m Mass of the particle
 */
inline auto beta(double e, double m) -> double {
  return sqrt(1.0 - tools::sqr(m / e));
}

/**
 * Compute the Kallen-Lambda function (triangle functions).
 */
template <typename T>
static auto kallen_lambda(const T a, const T b, const T c) -> T {
  return std::fma(a, a - 2 * c, std::fma(b, b - 2 * a, c * (c - 2 * b)));
}

/**
 * Compute the momentum of shared by a two-body final state.
 */
template <typename T>
static auto two_body_three_momentum(const T cme, const T m1, const T m2) -> T {
  return std::sqrt(kallen_lambda(sqr(cme), sqr(m1), sqr(m2))) / (2 * cme);
}

/**
 * Compute the energy of particle 1 in the center-of-mass frame from a process
 * of the form X -> 1 + 2, given by:
 *
 *     E1 = (q^2 + m1^2 - m2^2) / (2 * q)
 *
 * where `q` is the center-of-mass energy, `m1` is the mass of particle 1 and
 * `m2` is the mass of the second particle.
 *
 * @param q center-of-mass energy
 * @param m1 mass of particle 1
 * @param m2 mass of particle 2
 */
static constexpr auto energy_one_cm(double q, double m1, double m2) -> double {
  return (tools::sqr(q) + tools::sqr(m1) - tools::sqr(m2)) / (2.0 * q);
}

// ===========================================================================
// ---- Check Functions ------------------------------------------------------
// ===========================================================================

auto zero_or_subnormal(double) -> bool;

// ===========================================================================
// ---- Printing Tools -------------------------------------------------------
// ===========================================================================

const std::string UNICODE_CAP_GAMMA = "\u0393"; // NOLINT
const std::string UNICODE_LARROW = "\u2192";    // NOLINT
const std::string UNICODE_PM = "\u00B1";        // NOLINT
const std::string UNICODE_MU = "\u03BC";        // NOLINT
const std::string UNICODE_NU = "\u03BD";        // NOLINT
const std::string UNICODE_GAMMA = "\u03B3";     // NOLINT
const std::string UNICODE_SIGMA = "\u03C3";     // NOLINT

auto print_width(double, double, const std::string &,
                 const std::vector<std::string> &...) -> void;

auto print_width(double, const std::string &,
                 const std::vector<std::string> &...) -> void;

auto print_width(const std::string &, double, double, const std::string &,
                 const std::vector<std::string> &) -> void;

auto print_width(const std::string &, double, const std::string &,
                 const std::vector<std::string> &) -> void;

auto print_cross_section(double, double, const std::string &,
                         const std::string &,
                         const std::vector<std::string> &...) -> void;

auto print_cross_section(double, const std::string &, const std::string &,
                         const std::vector<std::string> &...) -> void;

auto print_cross_section(const std::string &, double, double,
                         const std::string &, const std::string &,
                         const std::vector<std::string> &) -> void;

auto print_cross_section(const std::string &, double, const std::string &,
                         const std::string &, const std::vector<std::string> &)
    -> void;

} // namespace blackthorn::tools

#endif // BLACKTHORN_TOOLS_HPP
