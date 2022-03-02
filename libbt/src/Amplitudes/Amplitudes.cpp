#include "blackthorn/Amplitudes.h"
#include "blackthorn/Tools.h"
#include "blackthorn/Wavefunctions.h"

namespace blackthorn {

// ===========================================================================
// ---- Amplitudes -----------------------------------------------------------
// ===========================================================================

auto amplitude(const VertexFFS &v, const DiracWf<FermionFlow::Out> &fo,
               const DiracWf<FermionFlow::In> &fi, const ScalarWf &phi)
    -> std::complex<double> {
  return phi.wavefunction() * (v.left * (fi[0] * fo[0] + fi[1] * fo[1]) +
                               v.right * (fi[2] * fo[2] + fi[3] * fo[3]));
}

auto amplitude(const VertexFFSDeriv &v, const DiracWf<FermionFlow::Out> &fo,
               const DiracWf<FermionFlow::In> &fi, const ScalarWf &phi)
    -> std::complex<double> {
  using tools::im;
  const auto p = phi.momentum();
  return phi.wavefunction() *
         (v.right * fi[2] *
              (-(fo[1] * (p[1] + im * p[2])) + fo[0] * (p[0] - p[3])) +
          v.left * fi[1] *
              (fo[2] * (p[1] - im * p[2]) + fo[3] * (p[0] - p[3])) +
          v.right * fi[3] *
              (-(fo[0] * (p[1] - im * p[2])) + fo[1] * (p[0] + p[3])) +
          v.left * fi[0] *
              (fo[3] * (p[1] + im * p[2]) + fo[2] * (p[0] + p[3])));
}

auto amplitude(const VertexFFV &v, const DiracWf<FermionFlow::Out> &fo,
               const DiracWf<FermionFlow::In> &fi, const VectorWf &eps)
    -> std::complex<double> {
  using tools::im;
  return v.right * fi[2] *
             (-(fo[1] * (eps[1] + im * eps[2])) + fo[0] * (eps[0] - eps[3])) +
         v.left * fi[1] *
             (fo[2] * (eps[1] - im * eps[2]) + fo[3] * (eps[0] - eps[3])) +
         v.right * fi[3] *
             (-(fo[0] * (eps[1] - im * eps[2])) + fo[1] * (eps[0] + eps[3])) +
         v.left * fi[0] *
             (fo[3] * (eps[1] + im * eps[2]) + fo[2] * (eps[0] + eps[3]));
}

auto amplitude(const VertexFFSS &v, const DiracWf<FermionFlow::Out> &fo,
               const DiracWf<FermionFlow::In> &fi, const ScalarWf &phi1,
               const ScalarWf &phi2) -> std::complex<double> {
  using tools::im;
  const auto p1 = phi1.momentum();
  const auto p2 = phi2.momentum();
  const auto phi1wf = phi1.wavefunction();
  const auto phi2wf = phi2.wavefunction();

  return fi[2] * (v.right3 * phi1wf * phi2wf * fo[2] -
                  fo[1] * (v.right1 * phi2wf * (p1[1] + im * p1[2]) +
                           v.right2 * phi1wf * (p2[1] + im * p2[2])) +
                  fo[0] * (v.right1 * phi2wf * (p1[0] - p1[3]) +
                           v.right2 * phi1wf * (p2[0] - p2[3]))) +
         fi[1] *
             (v.left3 * phi1wf * phi2wf * fo[1] +
              v.left1 * phi2wf *
                  (fo[2] * (p1[1] - im * p1[2]) + fo[3] * (p1[0] - p1[3])) +
              v.left2 * phi1wf *
                  (fo[2] * (p2[1] - im * p2[2]) + fo[3] * (p2[0] - p2[3]))) +
         fi[3] * (v.right3 * phi1wf * phi2wf * fo[3] -
                  fo[0] * (v.right1 * phi2wf * (p1[1] - im * p1[2]) +
                           v.right2 * phi1wf * (p2[1] - im * p2[2])) +
                  fo[1] * (v.right1 * phi2wf * (p1[0] + p1[3]) +
                           v.right2 * phi1wf * (p2[0] + p2[3]))) +
         fi[0] * (v.left3 * phi1wf * phi2wf * fo[0] +
                  v.left1 * phi2wf *
                      (fo[3] * (p1[1] + im * p1[2]) + fo[2] * (p1[0] + p1[3])) +
                  v.left2 * phi1wf *
                      (fo[3] * (p2[1] + im * p2[2]) + fo[2] * (p2[0] + p2[3])));
}

} // namespace blackthorn
