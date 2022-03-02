#include "blackthorn/Wavefunctions.h"

namespace blackthorn {

auto scalar_wf(ScalarWf *wf, const LVector<double> &p, InOut inout) -> void {
  const double s = inout == Incoming ? 1.0 : -1.0;
  wf->wavefunction() = std::complex<double>{1.0, 0.0};
  wf->momentum() = s * p;
}

auto scalar_wf(const LVector<double> &p, InOut inout) -> ScalarWf {
  ScalarWf wf{};
  scalar_wf(&wf, p, inout);
  return wf;
}

} // namespace blackthorn
