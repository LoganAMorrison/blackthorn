#include "Helas.h"
#include "blackthorn/Tensors.h"
#include "blackthorn/Wavefunctions.h"
#include <array>
#include <complex>
#include <memory>

namespace bt = blackthorn;
using cmplx = std::complex<double>;

constexpr size_t SWF_SIZE = 3;
constexpr size_t DWF_SIZE = 6;
constexpr size_t VWF_SIZE = 6;

template <size_t N>
static auto extract_momentum(const std::array<cmplx, N> &wf)
    -> bt::LVector<double> {
  const double e = wf[SWF_SIZE - 2].real();
  const double px = wf[SWF_SIZE - 1].real();
  const double py = wf[SWF_SIZE - 1].imag();
  const double pz = wf[SWF_SIZE - 2].imag();
  return bt::LVector<double>{e, px, py, pz};
}

static auto extract_momentum(const std::vector<cmplx> &wf)
    -> bt::LVector<double> {
  const double e = wf[SWF_SIZE - 2].real();
  const double px = wf[SWF_SIZE - 1].real();
  const double py = wf[SWF_SIZE - 1].imag();
  const double pz = wf[SWF_SIZE - 2].imag();
  return bt::LVector<double>{e, px, py, pz};
}

template <size_t N>
static auto extract_wavefunction(const std::array<cmplx, N> &wf)
    -> bt::LVector<double>;

static auto extract_swf(const std::array<cmplx, SWF_SIZE> &wf) -> cmplx {
  return wf[0];
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
    -> bt::LVector<cmplx> {
  return {wf[0], wf[1], wf[2], wf[3]};
}

static auto extract_vwf(const std::vector<cmplx> &wf) -> bt::LVector<cmplx> {
  return {wf[0], wf[1], wf[2], wf[3]};
}

/**
 * Call HELAS sxxxxx.
 * @param p four-momentum
 * @param finalstate If true, wavefunction is outgoing.
 */
auto helas_sxxxxx(bt::LVector<double> p, bool finalstate) -> bt::ScalarWf {
  int nss = finalstate ? 1 : -1;
  std::vector<cmplx> wf(SWF_SIZE, 0);
  std::vector<double> pp = {p[0], p[1], p[2], p[3]};
  sxxxxx_(p.data(), &nss, wf.data());
  return {extract_swf(wf), extract_momentum(wf)};
}

/**
 * Call HELAS ixxxxx.
 * @param p Four-momentum
 * @param mass Mass of the fermion
 * @param spin Spin of the fermion (-1 or 1)
 * @param anti If true, wavefunction is "v", else "u"
 */
auto helas_ixxxxx(bt::LVector<double> p, double mass, int spin, bool anti)
    -> bt::DiracWf<bt::FlowIn> {
  int nhel = spin;
  int nsf = anti ? -1 : 1;
  std::vector<cmplx> wf(SWF_SIZE, 0);
  std::vector<double> pp = {p[0], p[1], p[2], p[3]};
  ixxxxx_(pp.data(), &mass, &nhel, &nsf, wf.data());
  return {extract_dwf(wf), extract_momentum(wf)};
}

/**
 * Call HELAS oxxxxx.
 * @param p Four-momentum
 * @param mass Mass of the fermion
 * @param spin Spin of the fermion (-1 or 1)
 * @param anti If true, wavefunction is "vbar", else "ubar"
 */
auto helas_oxxxxx(bt::LVector<double> p, double mass, int spin, bool anti)
    -> bt::DiracWf<bt::FlowOut> {
  int nhel = spin;
  int nsf = anti ? -1 : 1;
  std::vector<cmplx> wf(DWF_SIZE, 0);
  std::vector<double> pp = {p[0], p[1], p[2], p[3]};
  ixxxxx_(p.data(), &mass, &nhel, &nsf, wf.data());
  return {extract_dwf(wf), extract_momentum(wf)};
}

/**
 * Call HELAS vxxxxx.
 * @param p Four-momentum
 * @param mass Mass of the vector
 * @param spin Spin of the vector (-1, 0 or 1)
 * @param finalstate If true, wavefunction is outgoing.
 */
auto helas_vxxxxx(bt::LVector<double> p, double mass, int spin, bool finalstate)
    -> bt::VectorWf {
  int nhel = spin;
  int nsf = finalstate ? 1 : -1;
  std::vector<cmplx> wf(VWF_SIZE, 0);
  std::vector<double> pp = {p[0], p[1], p[2], p[3]};
  ixxxxx_(p.data(), &mass, &nhel, &nsf, wf.data());
  return {extract_vwf(wf), extract_momentum(wf)};
}
