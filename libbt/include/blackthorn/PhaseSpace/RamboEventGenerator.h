#ifndef BLACKTHORN_PHASE_SPACE_RAMBO_EVENT_GENERATOR_H
#define BLACKTHORN_PHASE_SPACE_RAMBO_EVENT_GENERATOR_H

#include "blackthorn/PhaseSpace/RamboCore.h"
#include "blackthorn/PhaseSpace/Types.h"

namespace blackthorn {

template <size_t N, class MSqrd> class RamboEventGenerator {
public:
  using MomentaType = std::array<LVector<double>, N>;

private:
  MSqrd p_msqrd;
  double p_cme;
  std::array<int, N> p_pdgs;
  std::array<double, N> p_masses;
  double p_base_wgt;

public:
  RamboEventGenerator(MSqrd msqrd, double cme, std::array<double, N> masses)
      : p_msqrd(msqrd), p_cme(cme), p_masses(masses),
        p_base_wgt(rambo_impl::massless_weight<N>(cme)) {}

  [[nodiscard]] auto generate() const -> PhaseSpaceEvent<N> {
    MomentaType momenta{};
    rambo_impl::generate_momenta(momenta, p_cme, p_masses);
    double weight = rambo_impl::wgt_rescale_factor(momenta, p_cme) *
                    p_msqrd(momenta) * p_base_wgt;
    return PhaseSpaceEvent<N>(momenta, weight);
  }

  auto fill(PhaseSpaceEvent<N> *event) const -> void {
    rambo_impl::generate_momenta(&event->p_momenta, p_cme, p_masses);
    event->p_weight = rambo_impl::wgt_rescale_factor(event->p_momenta, p_cme) *
                      p_msqrd(event->p_momenta) * p_base_wgt;
  }
};

} // namespace blackthorn

#endif // BLACKTHORN_PHASE_SPACE_RAMBO_EVENT_GENERATOR_H
