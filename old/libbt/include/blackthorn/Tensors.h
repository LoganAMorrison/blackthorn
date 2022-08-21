#ifndef BLACKTHORN_TENSORS_H
#define BLACKTHORN_TENSORS_H

#include "blackthorn/Tools.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace blackthorn {

template <typename T, size_t N> class SVector {
private:
  std::array<T, N> p_data{};

public:
  explicit SVector(std::array<T, N> data) : p_data(std::move(data)) {}
  SVector() = default;

  // ==========================================================================
  // ---- Access Operations ---------------------------------------------------
  // ==========================================================================

  auto operator[](size_t i) const -> const T & { return p_data[i]; }
  auto operator[](size_t i) -> T & { return p_data[i]; }

  auto at(size_t i) const -> const T & { return p_data.at(i); }
  auto at(size_t i) -> T & { return p_data.at(i); }

  // ==========================================================================
  // ---- Printing Operations -------------------------------------------------
  // ==========================================================================

  friend auto operator<<(std::ostream &os, const SVector<T, N> &p)
      -> std::ostream & {
    os << "SVector("
       << "\n";
    for (const auto &x : p.p_data) {
      os << x << ", ";
    }
    os << ")";
    return os;
  }

  // ==========================================================================
  // ---- Unary Math Operations -----------------------------------------------
  // ==========================================================================

  auto operator-() const -> SVector {
    SVector<T, N> vec{};
    for (size_t i = 0; i < N; i++) {
      vec[i] = -p_data[i];
    }
    return vec;
  }

  // ==========================================================================
  // ---- Non-Modifying Binary Math Operations --------------------------------
  // ==========================================================================

  template <typename S>
  auto operator+(const SVector<S, N> &rhs) const -> SVector {
    SVector<T, N> vec{};
    for (size_t i = 0; i < N; i++) {
      vec[i] = p_data[i] + rhs[i];
    }
    return vec;
  }

  template <typename S>
  auto operator-(const SVector<S, N> &rhs) const -> SVector {
    SVector<T, N> vec{};
    for (size_t i = 0; i < N; i++) {
      vec[i] = p_data[i] - rhs[i];
    }
    return vec;
  }

  template <typename S> auto operator*(const S &rhs) const -> SVector {
    SVector<T, N> vec{};
    for (size_t i = 0; i < N; i++) {
      vec[i] = p_data[i] * rhs;
    }
    return vec;
  }

  template <typename S> auto operator/(const S &rhs) const -> SVector {
    SVector<T, N> vec{};
    for (size_t i = 0; i < N; i++) {
      vec[i] = p_data[i] / rhs;
    }
    return vec;
  }

  // ==========================================================================
  // ---- Modifying Binary Math Operations ------------------------------------
  // ==========================================================================

  template <typename S> auto operator+=(const SVector<S, N> &rhs) -> void {
    for (size_t i = 0; i < N; i++) {
      p_data[i] += static_cast<T>(rhs[i]);
    }
  }

  template <typename S> auto operator-=(const SVector<S, N> &rhs) -> void {
    for (size_t i = 0; i < N; i++) {
      p_data[i] -= static_cast<T>(rhs[i]);
    }
  }

  template <typename S> auto operator*=(const S &rhs) -> void {
    for (size_t i = 0; i < N; i++) {
      p_data[i] *= static_cast<T>(rhs);
    }
  }

  template <typename S> auto operator/=(const S &rhs) -> void {
    for (size_t i = 0; i < N; i++) {
      p_data[i] /= static_cast<T>(rhs);
    }
  }
  // ==========================================================================
  // ---- Friend Math Operations ----------------------------------------------
  // ==========================================================================

  template <typename U>
  friend auto operator*(const SVector<T, N> &p, const U &rhs)
      -> SVector<decltype(T{} + U{}), N> {
    SVector<T, N> vec{};
    for (size_t i = 0; i < N; i++) {
      vec[i] = p[i] * rhs;
    }
    return vec;
  }

  template <typename U>
  friend auto operator*(const U &rhs, const SVector<T, N> &p)
      -> SVector<decltype(T{} + U{}), N> {
    return p * rhs;
  }

  template <typename U>
  friend auto operator/(const SVector<T, N> &p, const U &rhs)
      -> SVector<decltype(T{} + U{}), N> {
    SVector<T, N> vec{};
    for (size_t i = 0; i < N; i++) {
      vec[i] = p[i] / rhs;
    }
    return vec;
  }
};

// ==========================================================================
// ---- Lorentz Vectors -----------------------------------------------------
// ==========================================================================

template <typename T> class LVector {
private:
  std::array<T, 4> p_data{};

public:
  explicit LVector(std::array<T, 4> data) : p_data(std::move(data)) {}
  explicit LVector(SVector<T, 4> data) : p_data(std::move(data)) {}
  LVector(T x0, T x1, T x2, T x3) : p_data({x0, x1, x2, x3}) {}
  LVector() = default;

  // ---- Access Operations ---------------------------------------------------

  auto e() const -> const T & { return p_data[0]; }
  auto e() -> T & { return p_data[0]; }

  auto px() const -> const T & { return p_data[1]; }
  auto px() -> T & { return p_data[1]; }

  auto py() const -> const T & { return p_data[2]; }
  auto py() -> T & { return p_data[2]; }

  auto pz() const -> const T & { return p_data[3]; }
  auto pz() -> T & { return p_data[3]; }

  auto operator[](size_t i) const -> const T & { return p_data[i]; }
  auto operator[](size_t i) -> T & { return p_data[i]; }

  auto at(size_t i) const -> const T & { return p_data.at(i); }
  auto at(size_t i) -> T & { return p_data.at(i); }

  auto data() const noexcept -> const T * { return p_data.data(); }
  auto data() noexcept -> T * { return p_data.data(); }

  // ---- Printing Operations -------------------------------------------------

  friend auto operator<<(std::ostream &os, const LVector<T> &p)
      -> std::ostream & {
    os << "LVector(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3]
       << ")";
    return os;
  }

  // ---- Unary Math Operations -----------------------------------------------

  auto operator-() const -> LVector<decltype(-T{})> {
    return LVector(-p_data[0], -p_data[1], -p_data[2], -p_data[3]);
  }

  // ---- Non-Modifying Binary Math Operations --------------------------------

  template <typename S>
  auto operator+(const LVector<S> &rhs) const -> LVector<decltype(T{} + S{})> {
    return LVector(p_data[0] + rhs.p_data[0], p_data[1] + rhs.p_data[1],
                   p_data[2] + rhs.p_data[2], p_data[3] + rhs.p_data[3]);
  }

  template <typename S>
  auto operator-(const LVector<S> &rhs) const -> LVector<decltype(T{} - S{})> {
    return LVector(p_data[0] - rhs.p_data[0], p_data[1] - rhs.p_data[1],
                   p_data[2] - rhs.p_data[2], p_data[3] - rhs.p_data[3]);
  }

  // ---- Modifying Binary Math Operations ------------------------------------

  template <typename S> auto operator+=(const LVector<S> &rhs) -> void {
    p_data[0] += static_cast<T>(rhs.p_data[0]);
    p_data[1] += static_cast<T>(rhs.p_data[1]);
    p_data[2] += static_cast<T>(rhs.p_data[2]);
    p_data[3] += static_cast<T>(rhs.p_data[3]);
  }

  template <typename S> auto operator-=(const LVector<S> &rhs) -> void {
    p_data[0] -= static_cast<T>(rhs.p_data[0]);
    p_data[1] -= static_cast<T>(rhs.p_data[1]);
    p_data[2] -= static_cast<T>(rhs.p_data[2]);
    p_data[3] -= static_cast<T>(rhs.p_data[3]);
  }

  template <typename S> auto operator*=(const S &rhs) -> void {
    p_data[0] *= static_cast<T>(rhs);
    p_data[1] *= static_cast<T>(rhs);
    p_data[2] *= static_cast<T>(rhs);
    p_data[3] *= static_cast<T>(rhs);
  }

  template <typename S> auto operator/=(const S &rhs) -> void {
    p_data[0] /= static_cast<T>(rhs);
    p_data[1] /= static_cast<T>(rhs);
    p_data[2] /= static_cast<T>(rhs);
    p_data[3] /= static_cast<T>(rhs);
  }

  // ---- Friend Math Operations ----------------------------------------------

  template <typename U>
  friend auto operator*(const LVector<T> &p, const U &rhs)
      -> LVector<decltype(U{} * T{})> {
    return LVector<decltype(U{} * T{})>(p[0] * rhs, p[1] * rhs, p[2] * rhs,
                                        p[3] * rhs);
  }

  template <typename U>
  friend auto operator*(const U &rhs, const LVector<T> &p)
      -> LVector<decltype(U{} * T{})> {
    return p * rhs;
  }

  template <typename U>
  friend auto operator/(const LVector<T> &p, const U &rhs)
      -> LVector<decltype(T{} / U{})> {
    return LVector<decltype(T{} / U{})>(p[0] / rhs, p[1] / rhs, p[2] / rhs,
                                        p[3] / rhs);
  }
};

// ==========================================================================
// ---- Norms ---------------------------------------------------------------
// ==========================================================================

template <typename T>
inline auto lnorm_sqr(const LVector<T> &lv) -> decltype(tools::abs2(T{})) {
  using tools::abs2;
  return abs2(lv.e()) - (abs2(lv.px()) + abs2(lv.py()) + abs2(lv.pz()));
}

template <typename T>
inline auto lnorm(const LVector<T> &lv) -> decltype(sqrt(std::abs(T{} * T{}))) {
  const auto m2 = lnorm_sqr(lv);
  return sqrt(std::abs(m2));
}

template <typename T>
inline auto lnorm3_sqr(const LVector<T> &lv) -> decltype(tools::abs2(T{})) {
  using tools::abs2;
  return abs2(lv.px()) + abs2(lv.py()) + abs2(lv.pz());
}

template <typename T>
inline auto lnorm3(const LVector<T> &lv)
    -> decltype(sqrt(std::abs(T{} * T{}))) {
  return sqrt(lnorm3_sqr(lv));
}

// ==========================================================================
// ---- Operations on LVectors ----------------------------------------------
// ==========================================================================

template <typename T, typename S>
inline auto dot(const LVector<T> &lv1, const LVector<S> &lv2)
    -> decltype(T{} * S{}) {
  return lv1[0] * lv2[0] -
         (lv1[1] * lv2[1] + lv1[2] * lv2[2] + lv1[3] * lv2[3]);
}

// ==========================================================================
// ---- Dirac Vectors -------------------------------------------------------
// ==========================================================================

template <typename T> class DVector {
private:
  std::array<T, 4> p_data{};

public:
  explicit DVector(std::array<T, 4> data) : p_data(std::move(data)) {}
  explicit DVector(SVector<T, 4> data) : p_data(std::move(data)) {}
  DVector(T x0, T x1, T x2, T x3) : p_data({x0, x1, x2, x3}) {}
  DVector() = default;

  // ---- Access Operations ---------------------------------------------------

  auto operator[](size_t i) const -> const T & { return p_data[i]; }
  auto operator[](size_t i) -> T & { return p_data[i]; }

  auto at(size_t i) const -> const T & { return p_data.at(i); }
  auto at(size_t i) -> T & { return p_data.at(i); }

  // ---- Printing Operations -------------------------------------------------

  friend auto operator<<(std::ostream &os, const DVector<T> &p)
      -> std::ostream & {
    os << "DVector(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3]
       << ")";
    return os;
  }

  // ---- Unary Math Operations -----------------------------------------------

  auto operator-() const -> DVector<decltype(-T{})> {
    return DVector(-p_data[0], -p_data[1], -p_data[2], -p_data[3]);
  }

  // ---- Non-Modifying Binary Math Operations --------------------------------

  template <typename S>
  auto operator+(const DVector<S> &rhs) const -> DVector<decltype(T{} + S{})> {
    return DVector(p_data[0] + rhs.p_data[0], p_data[1] + rhs.p_data[1],
                   p_data[2] + rhs.p_data[2], p_data[3] + rhs.p_data[3]);
  }

  template <typename S>
  auto operator-(const DVector<S> &rhs) const -> DVector<decltype(T{} - S{})> {
    return DVector(p_data[0] - rhs.p_data[0], p_data[1] - rhs.p_data[1],
                   p_data[2] - rhs.p_data[2], p_data[3] - rhs.p_data[3]);
  }

  // template <typename S>
  // auto operator*(const S &rhs) const -> DVector<decltype(T{} * S{})> {
  //   return DVector(p_data[0] * rhs, p_data[1] * rhs, p_data[2] * rhs,
  //                  p_data[3] * rhs);
  // }

  // template <typename S>
  // auto operator/(const S &rhs) const -> DVector<decltype(T{} / S{})> {
  //   return DVector(p_data[0] / rhs, p_data[1] / rhs, p_data[2] / rhs,
  //                  p_data[3] / rhs);
  // }

  // ---- Modifying Binary Math Operations ------------------------------------

  template <typename S> auto operator+=(const DVector<S> &rhs) -> void {
    p_data[0] += static_cast<T>(rhs.p_data[0]);
    p_data[1] += static_cast<T>(rhs.p_data[1]);
    p_data[2] += static_cast<T>(rhs.p_data[2]);
    p_data[3] += static_cast<T>(rhs.p_data[3]);
  }

  template <typename S> auto operator-=(const DVector<S> &rhs) -> void {
    p_data[0] -= static_cast<T>(rhs.p_data[0]);
    p_data[1] -= static_cast<T>(rhs.p_data[1]);
    p_data[2] -= static_cast<T>(rhs.p_data[2]);
    p_data[3] -= static_cast<T>(rhs.p_data[3]);
  }

  template <typename S> auto operator*=(const S &rhs) -> void {
    p_data[0] *= static_cast<T>(rhs);
    p_data[1] *= static_cast<T>(rhs);
    p_data[2] *= static_cast<T>(rhs);
    p_data[3] *= static_cast<T>(rhs);
  }

  template <typename S> auto operator/=(const S &rhs) -> void {
    p_data[0] /= static_cast<T>(rhs);
    p_data[1] /= static_cast<T>(rhs);
    p_data[2] /= static_cast<T>(rhs);
    p_data[3] /= static_cast<T>(rhs);
  }

  // ---- Friend Math Operations ----------------------------------------------

  template <typename U>
  friend auto operator*(const DVector<T> &p, const U &rhs)
      -> DVector<decltype(U{} * T{})> {
    return DVector<decltype(U{} * T{})>(p[0] * rhs, p[1] * rhs, p[2] * rhs,
                                        p[3] * rhs);
  }

  template <typename U>
  friend auto operator*(const U &rhs, const DVector<T> &p)
      -> DVector<decltype(U{} * T{})> {
    return p * rhs;
  }

  template <typename U>
  friend auto operator/(const DVector<T> &p, const U &rhs)
      -> DVector<decltype(T{} / U{})> {
    return DVector<decltype(T{} / U{})>(p[0] / rhs, p[1] / rhs, p[2] / rhs,
                                        p[3] / rhs);
  }
};

// ==========================================================================
// ---- Norms ---------------------------------------------------------------
// ==========================================================================

template <typename T>
inline auto norm_sqr(const DVector<T> &v) -> decltype(tools::abs2(T{})) {
  using tools::abs2;
  return abs2(v[0]) + abs2(v[1]) + abs2(v[2]) + abs2(v[3]);
}

template <typename T>
inline auto norm(const DVector<T> &v) -> decltype(sqrt(std::abs(T{} * T{}))) {
  const auto m2 = norm_sqr(v);
  return sqrt(std::abs(m2));
}

// ==========================================================================
// ---- Operations on LVectors ----------------------------------------------
// ==========================================================================

template <typename T, typename S>
inline auto dot(const DVector<T> &v1, const DVector<S> &v2)
    -> decltype(T{} * S{}) {
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] + v1[3] * v2[3];
}

} // namespace blackthorn

#endif // BLACKTHORN_TENSORS_H
