#include "blackthorn/Amplitudes.h"
#include "blackthorn/Tools.h"
#include "blackthorn/Wavefunctions.h"

namespace blackthorn {

template <typename Wf> struct MomentumFlow { // NOLINT
  static constexpr double value = 1.0;
};

template <> struct MomentumFlow<DiracWf<FermionFlow::In>> { // NOLINT
  static constexpr double value = -1.0;
};

// Structure for summing the momenta for N-1 particles to produce the momentum
// of the Nth particle in a vertex.
template <class Wf> static auto momentum_sum(const Wf &wf) -> LVector<double> {
  return MomentumFlow<Wf>::value * wf.momentum();
}

template <class Wf, class... Wfs>
static auto momentum_sum(const Wf &wf, const Wfs &...wfs) -> LVector<double> {
  return MomentumFlow<Wf>::value * wf.momentum() + momentum_sum(wfs...);
}

template <class WfOut, class... Wfs>
static auto momentum_sum(WfOut * /*out*/, const Wfs &...wfs)
    -> LVector<double> {
  return MomentumFlow<WfOut>::value * momentum_sum(wfs...);
}

// ===========================================================================
// ---- Fermion-Fermion-Scalar Vertex ----------------------------------------
// ===========================================================================

template <>
auto Current::fuse_into(DiracWfI *out, const VertexFFS &v, const DiracWfI &fi,
                        const ScalarWf &phi) -> void {
  out->momentum() = momentum_sum(out, fi, phi);
  out->wavefunction() = {v.left * phi.wavefunction() * fi[0],
                         v.left * phi.wavefunction() * fi[1],
                         v.right * phi.wavefunction() * fi[2],
                         v.right * phi.wavefunction() * fi[3]};
}

template <>
auto Current::fuse_into(DiracWfO *out, const VertexFFS &v, const DiracWfO &fo,
                        const ScalarWf &phi) -> void {
  out->momentum() = momentum_sum(out, fo, phi);
  out->wavefunction() = {v.left * phi.wavefunction() * fo[0],
                         v.left * phi.wavefunction() * fo[1],
                         v.right * phi.wavefunction() * fo[2],
                         v.right * phi.wavefunction() * fo[3]};
}

template <>
auto Current::fuse_into(ScalarWf *out, const VertexFFS &v, const DiracWfO &fo,
                        const DiracWfI &fi) -> void {
  out->momentum() = momentum_sum(out, fo, fi);
  out->wavefunction() = v.left * fi[0] * fo[0] + v.left * fi[1] * fo[1] +
                        v.right * fi[2] * fo[2] + v.right * fi[3] * fo[3];
}

// ===========================================================================
// ---- Fermion-Fermion-Scalar Deriv Vertex ----------------------------------
// ===========================================================================

template <>
auto Current::fuse_into(DiracWfI *out, const VertexFFSDeriv &v,
                        const DiracWfI &fi, const ScalarWf &phi) -> void {
  using tools::im;
  const auto p = phi.momentum();
  const auto phiwf = phi.wavefunction();
  out->momentum() = momentum_sum(out, fi, phi);
  ;
  out->wavefunction() = {phiwf * (-(v.right * fi[3] * (p[1] - im * p[2])) +
                                  v.right * fi[2] * (p[0] - p[3])),
                         phiwf * (-(v.right * fi[2] * (p[1] + im * p[2])) +
                                  v.right * fi[3] * (p[0] + p[3])),
                         phiwf * (v.left * fi[1] * (p[1] - im * p[2]) +
                                  v.left * fi[0] * (p[0] + p[3])),
                         phiwf * (v.left * fi[0] * (p[1] + im * p[2]) +
                                  v.left * fi[1] * (p[0] - p[3]))};
}

template <>
auto Current::fuse_into(DiracWfO *out, const VertexFFSDeriv &v,
                        const DiracWfO &fo, const ScalarWf &phi) -> void {
  using tools::im;
  const auto p = phi.momentum();
  const auto phiwf = phi.wavefunction();
  out->momentum() = momentum_sum(out, fo, phi);
  out->wavefunction() = {
      phiwf * (v.left * fo[3] * (p[1] + im * p[2]) +
               v.left * fo[2] * (p[0] + p[3])),
      phiwf * (v.left * fo[2] * (p[1] - im * p[2]) +
               v.left * fo[3] * (p[0] - p[3])),
      -(v.right * phiwf *
        (fo[1] * (p[1] + im * p[2]) + fo[0] * (-p[0] + p[3]))),
      phiwf * (-(v.right * fo[0] * (p[1] - im * p[2])) +
               v.right * fo[1] * (p[0] + p[3]))};
}

template <>
auto Current::fuse_into(ScalarWf *out, const VertexFFSDeriv &v,
                        const DiracWfO &fo, const DiracWfI &fi) -> void {
  using tools::im;
  const auto p = momentum_sum(out, fo, fi);
  out->momentum() = p;
  out->wavefunction() =
      (v.right * fi[2] *
           (-(fo[1] * (p[1] + im * p[2])) + fo[0] * (p[0] - p[3])) +
       v.left * fi[1] * (fo[2] * (p[1] - im * p[2]) + fo[3] * (p[0] - p[3])) +
       v.right * fi[3] *
           (-(fo[0] * (p[1] - im * p[2])) + fo[1] * (p[0] + p[3])) +
       v.left * fi[0] * (fo[3] * (p[1] + im * p[2]) + fo[2] * (p[0] + p[3])));
}

// ===========================================================================
// ---- Fermion-Fermion-Vector Vertex ----------------------------------------
// ===========================================================================

template <>
auto Current::fuse_into(DiracWfI *out, const VertexFFV &v, const DiracWfI &fi,
                        const VectorWf &eps) -> void {
  using tools::im;
  out->momentum() = momentum_sum(out, fi, eps);
  out->wavefunction() = {-(v.right * fi[3] * (eps[1] - im * eps[2])) +
                             v.right * fi[2] * (eps[0] - eps[3]),
                         -(v.right * fi[2] * (eps[1] + im * eps[2])) +
                             v.right * fi[3] * (eps[0] + eps[3]),
                         v.left * fi[1] * (eps[1] - im * eps[2]) +
                             v.left * fi[0] * (eps[0] + eps[3]),
                         v.left * fi[0] * (eps[1] + im * eps[2]) +
                             v.left * fi[1] * (eps[0] - eps[3])};
}

template <>
auto Current::fuse_into(DiracWfO *out, const VertexFFV &v, const DiracWfO &fo,
                        const VectorWf &eps) -> void {
  using tools::im;
  out->momentum() = momentum_sum(out, fo, eps);
  out->wavefunction() = {v.left * fo[3] * (eps[1] + im * eps[2]) +
                             v.left * fo[2] * (eps[0] + eps[3]),
                         v.left * fo[2] * (eps[1] - im * eps[2]) +
                             v.left * fo[3] * (eps[0] - eps[3]),
                         -(v.right * (fo[1] * (eps[1] + im * eps[2]) +
                                      fo[0] * (-eps[0] + eps[3]))),
                         -(v.right * fo[0] * (eps[1] - im * eps[2])) +
                             v.right * fo[1] * (eps[0] + eps[3])};
}

template <>
auto Current::fuse_into(VectorWf *out, const VertexFFV &v,
                        const DiracWf<FermionFlow::Out> &fo,
                        const DiracWf<FermionFlow::In> &fi) -> void {
  using tools::im;
  out->momentum() = momentum_sum(out, fo, fi);
  out->wavefunction() = {
      v.right * fi[2] * fo[0] + v.right * fi[3] * fo[1] +
          v.left * fi[0] * fo[2] + v.left * fi[1] * fo[3],
      v.right * (fi[3] * fo[0] + fi[2] * fo[1]) -
          v.left * (fi[1] * fo[2] + fi[0] * fo[3]),
      -im * (v.right * fi[3] * fo[0] - v.right * fi[2] * fo[1] -
             v.left * fi[1] * fo[2] + v.left * fi[0] * fo[3]),
      v.right * fi[2] * fo[0] - v.right * fi[3] * fo[1] -
          v.left * fi[0] * fo[2] + v.left * fi[1] * fo[3]};
}

// ===========================================================================
// ---- Fermion-Fermion-Scalar-Vector Vertex ---------------------------------
// ===========================================================================

template <>
auto Current::fuse_into(DiracWfI *out, const VertexFFSV &v, const DiracWfI &fi,
                        const ScalarWf &phi, const VectorWf &eps) -> void {
  using tools::im;
  const auto phiwf = phi.wavefunction();
  out->momentum() = momentum_sum(out, fi, phi, eps);
  out->wavefunction() = {phiwf * (-(v.right * fi[3] * (eps[1] - im * eps[2])) +
                                  v.right * fi[2] * (eps[0] - eps[3])),
                         phiwf * (-(v.right * fi[2] * (eps[1] + im * eps[2])) +
                                  v.right * fi[3] * (eps[0] + eps[3])),
                         phiwf * (v.left * fi[1] * (eps[1] - im * eps[2]) +
                                  v.left * fi[0] * (eps[0] + eps[3])),
                         phiwf * (v.left * fi[0] * (eps[1] + im * eps[2]) +
                                  v.left * fi[1] * (eps[0] - eps[3]))};
}

template <>
auto Current::fuse_into(DiracWfO *out, const VertexFFSV &v, const DiracWfO &fo,
                        const ScalarWf &phi, const VectorWf &eps) -> void {
  using tools::im;
  const auto phiwf = phi.wavefunction();
  out->momentum() = momentum_sum(out, fo, phi, eps);
  out->wavefunction() = {v.left * fo[3] * (eps[1] + im * eps[2]) +
                             v.left * fo[2] * (eps[0] + eps[3]),
                         v.left * fo[2] * (eps[1] - im * eps[2]) +
                             v.left * fo[3] * (eps[0] - eps[3]),
                         -(v.right * (fo[1] * (eps[1] + im * eps[2]) +
                                      fo[0] * (-eps[0] + eps[3]))),
                         -(v.right * fo[0] * (eps[1] - im * eps[2])) +
                             v.right * fo[1] * (eps[0] + eps[3])};
}

template <>
auto Current::fuse_into(VectorWf *out, const VertexFFV &v,
                        const DiracWf<FermionFlow::Out> &fo,
                        const DiracWf<FermionFlow::In> &fi, const ScalarWf &phi)
    -> void {
  using tools::im;
  const auto phiwf = phi.wavefunction();
  out->momentum() = momentum_sum(out, fo, fi, phi);
  out->wavefunction() = {
      phiwf * (v.right * fi[2] * fo[0] + v.right * fi[3] * fo[1] +
               v.left * fi[0] * fo[2] + v.left * fi[1] * fo[3]),
      phiwf * (v.right * (fi[3] * fo[0] + fi[2] * fo[1]) -
               v.left * (fi[1] * fo[2] + fi[0] * fo[3])),
      phiwf * (-im * (v.right * fi[3] * fo[0] - v.right * fi[2] * fo[1] -
                      v.left * fi[1] * fo[2] + v.left * fi[0] * fo[3])),
      phiwf * (v.right * fi[2] * fo[0] - v.right * fi[3] * fo[1] -
               v.left * fi[0] * fo[2] + v.left * fi[1] * fo[3])};
}

template <>
auto Current::fuse_into(ScalarWf *out, const VertexFFV &v,
                        const DiracWf<FermionFlow::Out> &fo,
                        const DiracWf<FermionFlow::In> &fi, const VectorWf &eps)
    -> void {
  using tools::im;
  out->momentum() = momentum_sum(out, fo, fi, eps);
  out->wavefunction() =
      v.right * fi[2] *
          (-(fo[1] * (eps[1] + im * eps[2])) + fo[0] * (eps[0] - eps[3])) +
      v.left * fi[1] *
          (fo[2] * (eps[1] - im * eps[2]) + fo[3] * (eps[0] - eps[3])) +
      v.right * fi[3] *
          (-(fo[0] * (eps[1] - im * eps[2])) + fo[1] * (eps[0] + eps[3])) +
      v.left * fi[0] *
          (fo[3] * (eps[1] + im * eps[2]) + fo[2] * (eps[0] + eps[3]));
}

// ===========================================================================
// ---- Fermion-Fermion-Scalar-Scalar Vertex ---------------------------------
// ===========================================================================

template <>
auto Current::fuse_into(DiracWfI *out, const VertexFFSS &v, const DiracWfI &fi,
                        const ScalarWf &phi1, const ScalarWf &phi2) -> void {
  using tools::im;
  const auto phi1wf = phi1.wavefunction();
  const auto phi2wf = phi2.wavefunction();
  const auto p1 = phi1.momentum();
  const auto p2 = phi2.momentum();
  out->momentum() = momentum_sum(out, fi, phi1, phi2);
  out->wavefunction() = {
      v.left3 * phi1wf * phi2wf * fi[0] -
          fi[3] * (v.right1 * phi2wf * (p1[1] - im * p1[2]) +
                   v.right2 * phi1wf * (p2[1] - im * p2[2])) +
          fi[2] * (v.right1 * phi2wf * (p1[0] - p1[3]) +
                   v.right2 * phi1wf * (p2[0] - p2[3])),
      v.left3 * phi1wf * phi2wf * fi[1] -
          fi[2] * (v.right1 * phi2wf * (p1[1] + im * p1[2]) +
                   v.right2 * phi1wf * (p2[1] + im * p2[2])) +
          fi[3] * (v.right1 * phi2wf * (p1[0] + p1[3]) +
                   v.right2 * phi1wf * (p2[0] + p2[3])),
      v.right3 * phi1wf * phi2wf * fi[2] +
          v.left1 * phi2wf *
              (fi[1] * (p1[1] - im * p1[2]) + fi[0] * (p1[0] + p1[3])) +
          v.left2 * phi1wf *
              (fi[1] * (p2[1] - im * p2[2]) + fi[0] * (p2[0] + p2[3])),
      v.right3 * phi1wf * phi2wf * fi[3] +
          v.left1 * phi2wf *
              (fi[0] * (p1[1] + im * p1[2]) + fi[1] * (p1[0] - p1[3])) +
          v.left2 * phi1wf *
              (fi[0] * (p2[1] + im * p2[2]) + fi[1] * (p2[0] - p2[3]))};
}

template <>
auto Current::fuse_into(DiracWfO *out, const VertexFFSS &v, const DiracWfO &fo,
                        const ScalarWf &phi1, const ScalarWf &phi2) -> void {
  using tools::im;
  const auto phi1wf = phi1.wavefunction();
  const auto phi2wf = phi2.wavefunction();
  const auto p1 = phi1.momentum();
  const auto p2 = phi2.momentum();
  out->momentum() = momentum_sum(out, fo, phi1, phi2);
  out->wavefunction() = {
      v.left3 * phi1wf * phi2wf * fo[0] +
          v.left1 * phi2wf *
              (fo[3] * (p1[1] + im * p1[2]) + fo[2] * (p1[0] + p1[3])) +
          v.left2 * phi1wf *
              (fo[3] * (p2[1] + im * p2[2]) + fo[2] * (p2[0] + p2[3])),
      v.left3 * phi1wf * phi2wf * fo[1] +
          v.left1 * phi2wf *
              (fo[2] * (p1[1] - im * p1[2]) + fo[3] * (p1[0] - p1[3])) +
          v.left2 * phi1wf *
              (fo[2] * (p2[1] - im * p2[2]) + fo[3] * (p2[0] - p2[3])),
      v.right3 * phi1wf * phi2wf * fo[2] -
          fo[1] * (v.right1 * phi2wf * (p1[1] + im * p1[2]) +
                   v.right2 * phi1wf * (p2[1] + im * p2[2])) +
          fo[0] * (v.right1 * phi2wf * (p1[0] - p1[3]) +
                   v.right2 * phi1wf * (p2[0] - p2[3])),
      v.right3 * phi1wf * phi2wf * fo[3] -
          fo[0] * (v.right1 * phi2wf * (p1[1] - im * p1[2]) +
                   v.right2 * phi1wf * (p2[1] - im * p2[2])) +
          fo[1] * (v.right1 * phi2wf * (p1[0] + p1[3]) +
                   v.right2 * phi1wf * (p2[0] + p2[3]))};
}

template <>
auto Current::fuse_into(ScalarWf * /*out*/, const VertexFFSS & /*v*/,
                        const DiracWf<FermionFlow::Out> & /*fo*/,
                        const DiracWf<FermionFlow::In> & /*fi*/,
                        const ScalarWf & /*phi*/) -> void {
  // TODO: Figure out what to do here. Since there are two scalars, there isn't
  // an obvious way to tell which on to yield
  throw std::runtime_error("Unimplemented");
}

} // namespace blackthorn
