#include "Utils.h"

namespace blackthorn {

auto h_current(ScalarWf *wf, const VertexFFS &v,
               const DiracWf<FermionFlow::Out> &fo,
               const DiracWf<FermionFlow::In> &fi) -> void {
  constexpr double mh = Higgs::mass;
  constexpr double wh = Higgs::width;
  Current::generate(wf, v, mh, wh, fo, fi);
}

auto z_current(VectorWf *wf, const VertexFFV &v,
               const DiracWf<FermionFlow::Out> &fo,
               const DiracWf<FermionFlow::In> &fi) -> void {
  constexpr double mz = ZBoson::mass;
  constexpr double wz = ZBoson::width;
  Current::generate(wf, v, mz, wz, fo, fi);
}

auto w_current(VectorWf *wf, const VertexFFV &v,
               const DiracWf<FermionFlow::Out> &fo,
               const DiracWf<FermionFlow::In> &fi) -> void {
  constexpr double mw = WBoson::mass;
  constexpr double ww = WBoson::width;
  Current::generate(wf, v, mw, ww, fo, fi);
}

auto all_same(Gen gen1, Gen gen2, Gen gen3, Gen gen4) -> bool {
  return (gen1 == gen2 && gen1 == gen3 && gen1 == gen4);
}

} // namespace blackthorn
