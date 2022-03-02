#include "blackthorn/Spectra/Boost.h"
#include "blackthorn/Spectra/Decay.h"

namespace blackthorn {

auto decay_spectrum<NeutralPion>::dnde_photon(double egam, double epi)
    -> double {
  static constexpr double mpi0 = NeutralPion::mass;

  if (epi < mpi0) {
    return 0.0;
  }

  return 2 * NeutralPion::BR_PI0_TO_A_A *
         boost_delta_function(mpi0 / 2, egam, 0.0, tools::beta(epi, mpi0));
}

auto decay_spectrum<NeutralPion>::dndx_photon(double x, double epi, double cme)
    -> double {
  static constexpr double mpi0 = NeutralPion::mass;
  static constexpr double br = 0.9882;
  if (epi < mpi0) {
    return 0.0;
  }
  double egam = cme * x / 2.0;
  return cme * br *
         boost_delta_function(mpi0 / 2, egam, 0.0, tools::beta(epi, mpi0));
}

auto decay_spectrum<NeutralPion>::dnde_photon(const std::vector<double> &egams,
                                              const double epi)
    -> std::vector<double> {
  const auto f = [&](double x) { return dnde_photon(x, epi); };
  return tools::vectorized_par(f, egams);
}

auto decay_spectrum<NeutralPion>::dndx_photon(const std::vector<double> &xs,
                                              const double epi, double cme)
    -> std::vector<double> {
  const auto f = [&](double x) { return dndx_photon(x, epi, cme); };
  return tools::vectorized_par(f, xs);
}

auto decay_spectrum<NeutralPion>::dnde_photon(const py::array_t<double> &egams,
                                              double epi)
    -> py::array_t<double> {
  const auto f = [&](double x) { return dnde_photon(x, epi); };
  return tools::vectorized(f, egams);
}

auto decay_spectrum<NeutralPion>::dndx_photon(const py::array_t<double> &xs,
                                              double epi, double cme)
    -> py::array_t<double> {
  const auto f = [&](double x) { return dndx_photon(x, epi, cme); };
  return tools::vectorized(f, xs);
}

} // namespace blackthorn
