#ifndef BLACKTHORN_PHASE_SPACE_TYPES_H
#define BLACKTHORN_PHASE_SPACE_TYPES_H

#include "blackthorn/Tensors.h"

namespace blackthorn {

/// Alias for momenta used in a `PhaseSpaceEvent`
template <size_t N> using MomentaType = std::array<LVector<double>, N>;

/// Matrix element that just returns 1. Used when the matrix element is unknown.
template <size_t N>
constexpr auto msqrd_flat(const MomentaType<N> & /*momenta*/) -> double {
  return 1.0;
}

/// Structure for holding the momenta and weight of a event created by a phase
/// space generator.
template <size_t N> class PhaseSpaceEvent {
private:
  MomentaType<N> p_momenta{};
  double p_weight{0};

  template <size_t M> friend class Rambo;
  template <size_t M, class MSqrd> friend class RamboEventGenerator;

public:
  PhaseSpaceEvent(MomentaType<N> t_momenta, double t_weight)
      : p_momenta(std::move(t_momenta)), p_weight(t_weight) {}

  PhaseSpaceEvent() = default;

  auto weight() -> double & { return p_weight; }
  [[nodiscard]] auto weight() const -> const double & { return p_weight; }

  auto momenta() -> MomentaType<N> & { return p_momenta; }
  [[nodiscard]] auto momenta() const -> const MomentaType<N> & {
    return p_momenta;
  }

  auto momenta(size_t i) -> LVector<double> & { return p_momenta[i]; }
  [[nodiscard]] auto momenta(size_t i) const -> const LVector<double> & {
    return p_momenta[i];
  }
};

} // namespace blackthorn

#endif // BLACKTHORN_PHASE_SPACE_TYPES_H
