#include "Tools.h"
#include "Helas.h"
#include <fmt/core.h>

namespace blackthorn {

constexpr size_t SWF_SIZE = 3;
constexpr size_t DWF_SIZE = 6;
constexpr size_t VWF_SIZE = 6;

using cmplx = std::complex<double>;

auto print_fractional_diff(double actual, double estimate) -> void {
  const auto frac_diff = fractional_diff(actual, estimate);
  fmt::print("frac. diff. (%) = {}\n", frac_diff * 100.0);
}

static auto extract_momentum(const std::vector<cmplx> &wf) -> LVector<double> {
  const auto size = wf.size();
  const double e = wf[size - 2].real();
  const double px = wf[size - 1].real();
  const double py = wf[size - 1].imag();
  const double pz = wf[size - 2].imag();
  return LVector<double>{e, px, py, pz};
}

static auto extract_swf(const std::vector<cmplx> &wf) -> cmplx { return wf[0]; }

static auto extract_dwf(const std::array<cmplx, DWF_SIZE> &wf)
    -> std::array<cmplx, DWF_SIZE - 2> {
  return {wf[0], wf[1], wf[2], wf[3]};
}

static auto extract_dwf(const std::vector<cmplx> &wf)
    -> std::array<cmplx, DWF_SIZE - 2> {
  return {wf[0], wf[1], wf[2], wf[3]};
}

static auto extract_vwf(const std::array<cmplx, VWF_SIZE> &wf)
    -> LVector<cmplx> {
  return {wf[0], wf[1], wf[2], wf[3]};
}

static auto extract_vwf(const std::vector<cmplx> &wf) -> LVector<cmplx> {
  return {wf[0], wf[1], wf[2], wf[3]};
}

static auto to_helas(const DiracWf<FlowIn> &wf) -> std::vector<cmplx> {
  const auto p = wf.momentum();
  return {wf[0], wf[1], wf[2], wf[3], cmplx{p[0], p[3]}, cmplx{p[1], p[2]}};
}

static auto to_helas(const DiracWf<FlowOut> &wf) -> std::vector<cmplx> {
  const auto p = wf.momentum();
  return {wf[0], wf[1], wf[2], wf[3], cmplx{p[0], p[3]}, cmplx{p[1], p[2]}};
}

static auto to_helas(const VectorWf &wf) -> std::vector<cmplx> {
  const auto p = wf.momentum();
  return {wf[0], wf[1], wf[2], wf[3], cmplx{p[0], p[3]}, cmplx{p[1], p[2]}};
}

static auto to_helas(const ScalarWf &wf) -> std::vector<cmplx> {
  const auto p = wf.momentum();
  return {wf.wavefunction(), cmplx{p[0], p[3]}, cmplx{p[1], p[2]}};
}

static auto lvector_to_vector(const LVector<double> &p) -> std::vector<double> {
  std::vector<double> pp(4, 0.0);
  for (size_t i = 0; i < 4; ++i) {
    pp[i] = p[i];
  }
  return pp;
}

/**
 * Call HELAS sxxxxx to create a final-state scalar wavefunction.
 * @param p four-momentum
 */
auto helas_scalar_wf_final_state(LVector<double> p) -> ScalarWf {
  int nss = 1;
  std::vector<cmplx> wf(SWF_SIZE, 0);
  std::vector<double> pp = lvector_to_vector(p);
  sxxxxx_(p.data(), &nss, wf.data());
  return {extract_swf(wf), extract_momentum(wf)};
}

/**
 * Call HELAS sxxxxx to create a initial-state scalar wavefunction.
 * @param p four-momentum
 */
auto helas_scalar_wf_initial_state(LVector<double> p) -> ScalarWf {
  int nss = -1;
  std::vector<cmplx> wf(SWF_SIZE, 0);
  std::vector<double> pp = lvector_to_vector(p);
  sxxxxx_(p.data(), &nss, wf.data());
  return {extract_swf(wf), extract_momentum(wf)};
}

/**
 * Call HELAS ixxxxx to create a u-spinor.
 * @param p Four-momentum
 * @param mass Mass of the fermion
 * @param spin Spin of the fermion (-1 or 1)
 */
auto helas_spinor_u(LVector<double> p, double mass, int spin)
    -> DiracWf<FlowIn> {
  int nhel = spin;
  int nsf = 1;
  std::vector<cmplx> wf(DWF_SIZE, 0);
  std::vector<double> pp = lvector_to_vector(p);
  ixxxxx_(pp.data(), &mass, &nhel, &nsf, wf.data());
  return {extract_dwf(wf), extract_momentum(wf)};
}

/**
 * Call HELAS ixxxxx to create a v-spinor.
 * @param p Four-momentum
 * @param mass Mass of the fermion
 * @param spin Spin of the fermion (-1 or 1)
 */
auto helas_spinor_v(LVector<double> p, double mass, int spin)
    -> DiracWf<FlowIn> {
  int nhel = spin;
  int nsf = -1;
  std::vector<cmplx> wf(DWF_SIZE, 0);
  std::vector<double> pp = lvector_to_vector(p);
  ixxxxx_(pp.data(), &mass, &nhel, &nsf, wf.data());
  return {extract_dwf(wf), extract_momentum(wf)};
}

/**
 * Call HELAS oxxxxx to create a ubar-spinor.
 * @param p Four-momentum
 * @param mass Mass of the fermion
 * @param spin Spin of the fermion (-1 or 1)
 */
auto helas_spinor_ubar(LVector<double> p, double mass, int spin)
    -> DiracWf<FlowOut> {
  int nhel = spin;
  int nsf = 1;
  std::vector<cmplx> wf(DWF_SIZE, 0);
  std::vector<double> pp = lvector_to_vector(p);
  oxxxxx_(p.data(), &mass, &nhel, &nsf, wf.data());
  return {extract_dwf(wf), extract_momentum(wf)};
}

/**
 * Call HELAS oxxxxx to create a vbar-spinor.
 * @param p Four-momentum
 * @param mass Mass of the fermion
 * @param spin Spin of the fermion (-1 or 1)
 */
auto helas_spinor_vbar(LVector<double> p, double mass, int spin)
    -> DiracWf<FlowOut> {
  int nhel = spin;
  int nsf = -1;
  std::vector<cmplx> wf(DWF_SIZE, 0);
  std::vector<double> pp = lvector_to_vector(p);
  oxxxxx_(p.data(), &mass, &nhel, &nsf, wf.data());
  return {extract_dwf(wf), extract_momentum(wf)};
}

/**
 * Call HELAS vxxxxx to create a final-state vector wavefunction.
 * @param p Four-momentum
 * @param mass Mass of the vector
 * @param spin Spin of the vector (-1, 0 or 1)
 */
auto helas_vector_wf_final_state(LVector<double> p, double mass, int spin)
    -> VectorWf {
  int nhel = spin;
  int nsf = 1;
  std::vector<cmplx> wf(VWF_SIZE, 0);
  std::vector<double> pp = lvector_to_vector(p);
  vxxxxx_(p.data(), &mass, &nhel, &nsf, wf.data());
  return {extract_vwf(wf), extract_momentum(wf)};
}

/**
 * Call HELAS vxxxxx to create a initial-state vector wavefunction.
 * @param p Four-momentum
 * @param mass Mass of the vector
 * @param spin Spin of the vector (-1, 0 or 1)
 */
auto helas_vector_wf_initial_state(LVector<double> p, double mass, int spin)
    -> VectorWf {
  int nhel = spin;
  int nsf = -1;
  std::vector<cmplx> wf(VWF_SIZE, 0);
  std::vector<double> pp = lvector_to_vector(p);
  vxxxxx_(p.data(), &mass, &nhel, &nsf, wf.data());
  return {extract_vwf(wf), extract_momentum(wf)};
}

/**
 * Call HELAS hioxxx to create an off-shell scalar wf.
 */
auto helas_offshell_scalar(const DiracWf<FlowIn> &fi,
                           const DiracWf<FlowOut> &fo, const VertexFFS &v,
                           double mass, double width) -> ScalarWf {
  auto fih = to_helas(fi);
  auto foh = to_helas(fo);
  std::vector<cmplx> gc = {v.left, v.right};
  std::vector<cmplx> wf(SWF_SIZE, 0);
  hioxxx_(fih.data(), foh.data(), gc.data(), &mass, &width, wf.data());
  return {extract_swf(wf), extract_momentum(wf)};
}

/**
 * Call HELAS jioxxx to create an off-shell scalar wf.
 */
auto helas_offshell_vector(const DiracWf<FlowIn> &fi,
                           const DiracWf<FlowOut> &fo, const VertexFFV &v,
                           double mass, double width) -> VectorWf {
  auto fih = to_helas(fi);
  auto foh = to_helas(fo);
  std::vector<cmplx> gc = {v.left, v.right};
  std::vector<cmplx> wf(VWF_SIZE, 0);
  jioxxx_(fih.data(), foh.data(), gc.data(), &mass, &width, wf.data());
  return {extract_vwf(wf), extract_momentum(wf)};
}

} // namespace blackthorn
