#include "blackthorn/Amplitudes.h"
#include "blackthorn/Models/RhNeutrino.h"
#include "blackthorn/Models/StandardModel.h"
#include <boost/math/constants/constants.hpp>

namespace blackthorn {

using dpair = std::pair<double, double>;

static auto symmetry_factor_vvv(Gen g1, Gen g2, Gen g3) -> double {
  using boost::math::double_constants::third;
  if (g1 == g2 && g1 == g3) {
    return 0.5 * third;
  }
  if (g1 == g2 || g1 == g3 || g2 == g3) {
    return 0.5;
  }
  return 1.0;
}

template <class Model>
static auto width_n_to_v_v_v(const Model &model, Gen genv1, Gen genv2,
                             Gen genv3, size_t nevents, size_t batchsize)
    -> dpair {
  const auto msqrd = SquaredAmplitudeNToVVV(model, genv1, genv2, genv3);
  const std::array<double, 3> fsp_masses{0.0, 0.0, 0};
  const dpair result = Rambo<3>::decay_width(msqrd, model.mass(), fsp_masses,
                                             nevents, batchsize);
  const double s = symmetry_factor_vvv(genv1, genv2, genv3);
  return {result.first * s, result.second * s};
}

template <class Model>
auto width_n_to_v_v_v(const Model &model, size_t nevents, size_t batchsize)
    -> std::pair<double, double> {
  Gen g2 = Gen::Null;
  Gen g3 = Gen::Null;
  Gen genn = model.gen();

  if (genn == Gen::Fst) {
    g2 = Gen::Snd;
    g3 = Gen::Trd;
  } else if (genn == Gen::Snd) {
    g2 = Gen::Fst;
    g3 = Gen::Trd;
  } else {
    g2 = Gen::Fst;
    g3 = Gen::Snd;
  }
  const double s1 = symmetry_factor_vvv(genn, genn, genn);
  const auto msqrd1 = SquaredAmplitudeNToVVV(model, genn, genn, genn);
  const double s2 = symmetry_factor_vvv(genn, g2, g2);
  const auto msqrd2 = SquaredAmplitudeNToVVV(model, genn, g2, g2);
  const double s3 = symmetry_factor_vvv(genn, g3, g3);
  const auto msqrd3 = SquaredAmplitudeNToVVV(model, genn, g3, g3);

  auto msqrd = [&msqrd1, &msqrd2, &msqrd3, s1, s2,
                s3](const std::array<LVector<double>, 3> &ps) {
    return s1 * msqrd1(ps) + s2 * msqrd2(ps) + s3 * msqrd3(ps);
  };

  const std::array<double, 3> fsp_masses{0.0, 0.0, 0.0};
  return Rambo<3>::decay_width(msqrd, model.mass(), fsp_masses, nevents,
                               batchsize);
}

template <class Model>
auto width_n_to_v_l_l(const Model &model, Gen genv, Gen genl1, Gen genl2,
                      size_t nevents, size_t batchsize)
    -> std::pair<double, double> {
  const double ml1 = StandardModel::charged_lepton_mass(genl1);
  const double ml2 = StandardModel::charged_lepton_mass(genl2);
  if (model.mass() < ml1 + ml2) {
    return std::make_pair(0.0, 0.0);
  }
  const auto msqrd = SquaredAmplitudeNToVLL(model, genv, genl1, genl2);
  const std::array<double, 3> fsp_masses = {0.0, ml1, ml2};
  return Rambo<3>::decay_width(msqrd, model.mass(), fsp_masses, nevents,
                               batchsize);
}

auto RhNeutrinoGeV::width_v_v_v(Gen genv1, Gen genv2, Gen genv3, size_t nevents,
                                size_t batchsize) const
    -> std::pair<double, double> {
  return width_n_to_v_v_v(*this, genv1, genv2, genv3, nevents, batchsize);
}

auto RhNeutrinoGeV::width_v_v_v(size_t nevents, size_t batchsize) const
    -> std::pair<double, double> {
  return width_n_to_v_v_v(*this, nevents, batchsize);
}
auto RhNeutrinoGeV::width_v_l_l(Gen genv1, Gen genl1, Gen genl2, size_t nevents,
                                size_t batchsize) const
    -> std::pair<double, double> {
  return width_n_to_v_l_l(*this, genv1, genl1, genl2, nevents, batchsize);
}

auto RhNeutrinoMeV::width_v_v_v(Gen genv1, Gen genv2, Gen genv3, size_t nevents,
                                size_t batchsize) const
    -> std::pair<double, double> {
  return width_n_to_v_v_v(*this, genv1, genv2, genv3, nevents, batchsize);
}
auto RhNeutrinoMeV::width_v_v_v(size_t nevents, size_t batchsize) const
    -> std::pair<double, double> {
  return width_n_to_v_v_v(*this, nevents, batchsize);
}

auto RhNeutrinoMeV::width_v_l_l(Gen genv1, Gen genl1, Gen genl2, size_t nevents,
                                size_t batchsize) const
    -> std::pair<double, double> {
  return width_n_to_v_l_l(*this, genv1, genl1, genl2, nevents, batchsize);
}

} // namespace blackthorn
