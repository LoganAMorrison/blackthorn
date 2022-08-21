#include "Collier.h"
#include "blackthorn/Amplitudes.h"

namespace blackthorn {

static constexpr int DEFAULT_NMAX = 6;
static constexpr int DEFAULT_RMAX = 6;
static constexpr int DEFAULT_MODE = 1;

Loop::Loop() { init(); }

auto Loop::init() -> void { // NOLINT
  int nmax = DEFAULT_NMAX;
  int rmax = DEFAULT_RMAX;
  int mode = DEFAULT_MODE;
  int noreset = 0;
  collier_initialize(&nmax, &rmax);
}

auto Loop::initevent() -> void { // NOLINT
  collier_initialize_event();
}

auto Loop::scalarA0(const std::complex<double> &m1) -> std::complex<double> {
  initevent();
  std::complex<double> a0{};
  std::complex<double> m12_ = m1 * m1;
  collier_scalarA0(&a0, &m12_);
  return a0;
}

auto Loop::scalarB0(const std::complex<double> &s,
                    const std::complex<double> &m1,
                    const std::complex<double> &m2) -> std::complex<double> {
  initevent();
  std::complex<double> b0{};
  std::complex<double> s_ = s;
  std::complex<double> m12_ = m1 * m1;
  std::complex<double> m22_ = m2 * m2;
  collier_scalarB0(&b0, &s_, &m12_, &m22_);
  return b0;
}

auto Loop::scalarC0(const std::complex<double> &s1,
                    const std::complex<double> &s12,
                    const std::complex<double> &s2,
                    const std::complex<double> &m1,
                    const std::complex<double> &m2,
                    const std::complex<double> &m3) -> std::complex<double> {
  initevent();
  std::complex<double> c0{};
  std::complex<double> s1_ = s1;
  std::complex<double> s12_ = s12;
  std::complex<double> s2_ = s2;
  std::complex<double> m12_ = m1 * m1;
  std::complex<double> m22_ = m2 * m2;
  std::complex<double> m32_ = m3 * m3;
  collier_scalarC0(&c0, &s1_, &s12_, &s2_, &m12_, &m22_, &m32_);
  return c0;
}

auto TensorCoeffsA::coeff(size_t i) const -> const ValueType & {
  return p_tn[i];
}

auto TensorCoeffsA::coeff_uv(size_t i) const -> const ValueType & {
  return p_tnuv[i];
}

auto TensorCoeffsB::coeff(size_t i, size_t j) const -> const ValueType & {
  const size_t n = p_r / 2 + 1;
  return p_tn[n * j + i];
  // return p_tn[j][i];
}

auto TensorCoeffsB::coeff_uv(size_t i, size_t j) const -> const ValueType & {
  const size_t n = p_r / 2 + 1;
  return p_tn[n * j + i];
  // return p_tn[j][i];
}

auto TensorCoeffsB::coeff(size_t i) const -> const ValueType & {
  return p_tn[i];
}

auto TensorCoeffsB::coeff_uv(size_t i) const -> const ValueType & {
  return p_tn[i];
}

auto TensorCoeffsC::coeff(size_t i, size_t j, size_t k) const
    -> const ValueType & {
  const size_t n = p_r / 2 + 1;
  const size_t m = p_r + 1;
  return p_tn[n * m * k + n * j + i];
}

auto TensorCoeffsC::coeff_uv(size_t i, size_t j, size_t k) const
    -> const ValueType & {
  const size_t n = p_r / 2 + 1;
  const size_t m = p_r + 1;
  return p_tn[n * m * k + n * j + i];
}

auto TensorCoeffsC::coeff(size_t i) const -> const ValueType & {
  return p_tn[i];
}

auto TensorCoeffsC::coeff_uv(size_t i) const -> const ValueType & {
  return p_tn[i];
}

auto Loop::tensor_coeffs_a(const ValueType &m0, int r) -> TensorCoeffsA {
  initevent();
  int n = 1;
  int rmax = r;
  size_t s = (r / 2 + 1);
  std::vector<ValueType> tn(s, 0);
  std::vector<ValueType> tnuv(s, 0);
  ValueType m02 = m0 * m0;
  collier_coeffs_a(tn.data(), tnuv.data(), &m02, &n);
  return TensorCoeffsA(tn, tnuv, r);
}

auto Loop::tensor_coeffs_b(const ValueType &s, const ValueType &m0,
                           const ValueType &m1, int r) -> TensorCoeffsB {
  initevent();
  int rmax = r;
  int n = 2;
  int nc = collier_get_nc(&n, &r) + 1;
  std::vector<ValueType> a{};
  std::vector<ValueType> auv{};
  a.resize(nc);
  auv.resize(nc);
  auto s_ = s;
  auto m02 = m0 * m0;
  auto m12 = m1 * m1;
  collier_coeffs_b(a.data(), auv.data(), &s_, &m02, &m12, &rmax);
  return TensorCoeffsB(a, auv, r);
}

auto Loop::tensor_coeffs_c(const ValueType &s1, const ValueType &s12,
                           const ValueType &s2, const ValueType &m0,
                           const ValueType &m1, const ValueType &m2, int r)
    -> TensorCoeffsC {
  initevent();
  int rmax = r;
  int n = 3;
  int nc = collier_get_nc(&n, &r) + 1;
  std::vector<ValueType> c{};
  std::vector<ValueType> cuv{};
  c.resize(nc);
  cuv.resize(nc);
  auto s1_ = s1;
  auto s12_ = s12;
  auto s2_ = s2;
  auto m02 = m0 * m0;
  auto m12 = m1 * m1;
  auto m22 = m2 * m2;
  collier_coeffs_c(c.data(), cuv.data(), &s1_, &s12_, &s2_, &m02, &m12, &m22,
                   &rmax);
  return TensorCoeffsC(c, cuv, r);
}

auto Loop::tensor_coeffs_tn(std::vector<ValueType> sarr,
                            std::vector<ValueType> m2arr, int n, int r)
    -> std::pair<std::vector<ValueType>, std::vector<ValueType>> {
  initevent();
  int rmax = r;
  int nc = collier_get_nc(&n, &r) + 1;
  std::cout << n << std::endl;
  std::cout << r << std::endl;
  std::cout << nc << std::endl;
  std::vector<ValueType> tn{};
  std::vector<ValueType> tnuv{};
  tn.resize(nc);
  tnuv.resize(nc);
  collier_coeffs_tn(tn.data(), tnuv.data(), sarr.data(), m2arr.data(), &n, &r);
  return std::make_pair(tn, tnuv);
}

} // namespace blackthorn
