#include "blackthorn/Wavefunctions.h"

namespace blackthorn {

using cmplx = std::complex<double>;

auto vector_wf(VectorWf *wf, const LVector<double> &k, double mass, int spin,
               InOut io) -> void {
  const double e = k[0];
  const double kx = k[1];
  const double ky = k[2];
  const double kz = k[3];

  const double s = io == Outgoing ? -1.0 : 1.0;

  wf->momentum()[0] = -s * k[0];
  wf->momentum()[1] = -s * k[1];
  wf->momentum()[2] = -s * k[2];
  wf->momentum()[3] = -s * k[3];

  const auto km = std::hypot(kx, ky, kz);
  const auto kt = std::hypot(kx, ky);

  const auto massless = std::fpclassify(mass) != FP_NORMAL;
  const auto km_zero = std::fpclassify(km) != FP_NORMAL;
  const auto kt_zero = std::fpclassify(kt) != FP_NORMAL;

  if (spin == 0) {
    if (!massless) {
      if (km_zero) {
        (*wf)[0] = 0.0;
        (*wf)[1] = 0.0;
        (*wf)[2] = 0.0;
        (*wf)[3] = 1.0;
      } else {
        const auto n = e / (mass * km);
        (*wf)[0] = km / mass;
        (*wf)[1] = kx * n;
        (*wf)[2] = ky * n;
        (*wf)[3] = kz * n;
      }
    }
  } else if (spin == 1 || spin == -1) {
    if (kt_zero) {
      (*wf)[0] = 0.0;
      (*wf)[1] = -spin * M_SQRT1_2;
      (*wf)[2] = cmplx{0.0, -s * std::copysign(1.0, kz) * M_SQRT1_2};
      (*wf)[3] = 0.0;
    } else {
      const auto kxt = kx / kt * M_SQRT1_2;
      const auto kyt = ky / kt * M_SQRT1_2;
      const auto kzm = kz / km;
      const auto ktm = kt / km * M_SQRT1_2;

      (*wf)[0] = 0.0;
      (*wf)[1] = cmplx{-spin * kxt * kzm, +s * kyt};
      (*wf)[2] = cmplx{-spin * kyt * kzm, -s * kxt};
      (*wf)[3] = cmplx{+spin * ktm, 0.0};
    }

  } else {
    throw std::invalid_argument("Invalid spin " + std::to_string(spin) +
                                ". Must be -1,0, or 1.");
  }
}

auto vector_wf(const LVector<double> &p, double mass, int spin, InOut io)
    -> VectorWf {
  VectorWf wf{};
  vector_wf(&wf, p, mass, spin, io);
  return wf;
}

auto vector_wf(const LVector<double> &p, double mass, InOut io)
    -> std::array<VectorWf, 3> {
  return {
      vector_wf(p, mass, 1, io),
      vector_wf(p, mass, 0, io),
      vector_wf(p, mass, -1, io),
  };
}

auto vector_wf(const LVector<double> &p, InOut io) -> std::array<VectorWf, 2> {
  return {
      vector_wf(p, 0, 1, io),
      vector_wf(p, 0, -1, io),
  };
}

} // namespace blackthorn
