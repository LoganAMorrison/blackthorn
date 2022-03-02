#include "blackthorn/Models/RhNeutrino.h"
#include "blackthorn/Models/StandardModel.h"

namespace blackthorn {

// ===========================================================================
// ---- N-V-H ----------------------------------------------------------------
// ===========================================================================

auto RhNeutrinoMeV::feynman_rule_n_v_h() const -> VertexFFS {
  const auto g = p_mass * tan(p_theta) / Higgs::vev;
  return VertexFFS{g, -g};
}

auto RhNeutrinoGeV::feynman_rule_n_v_h() const -> VertexFFS {
  const auto g = p_mass * tan(p_theta) / Higgs::vev;
  return VertexFFS{g, -g};
}

// ===========================================================================
// ---- N-V-Z ----------------------------------------------------------------
// ===========================================================================

auto RhNeutrinoMeV::feynman_rule_n_v_z() const -> VertexFFV {
  const auto g = -StandardModel::qe * sin(2 * p_theta) /
                 (4 * StandardModel::cw * StandardModel::sw);
  return VertexFFV{g, g};
}

auto RhNeutrinoGeV::feynman_rule_n_v_z() const -> VertexFFV {
  const auto g = -StandardModel::qe * sin(2 * p_theta) /
                 (4 * StandardModel::cw * StandardModel::sw);
  return VertexFFV{g, g};
}

// ===========================================================================
// ---- N-L-W ----------------------------------------------------------------
// ===========================================================================

auto RhNeutrinoGeV::feynman_rule_n_l_w() const -> VertexFFV {
  const auto g = tools::im * StandardModel::qe * sin(p_theta) /
                 (M_SQRT2 * StandardModel::sw);
  return VertexFFV{g, 0.0};
}

auto RhNeutrinoMeV::feynman_rule_n_l_w() const -> VertexFFV {
  const auto g = tools::im * StandardModel::qe * sin(p_theta) /
                 (M_SQRT2 * StandardModel::sw);
  return VertexFFV{g, 0.0};
}

} // namespace blackthorn
