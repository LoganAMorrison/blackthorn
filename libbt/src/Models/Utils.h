#include "blackthorn/Models/Particles.h"
#include "blackthorn/Models/RhNeutrino.h"
#include "blackthorn/Models/StandardModel.h"

namespace blackthorn {

auto h_current(ScalarWf *wf, const VertexFFS &v,
               const DiracWf<FermionFlow::Out> &fo,
               const DiracWf<FermionFlow::In> &fi) -> void;

auto z_current(VectorWf *wf, const VertexFFV &v,
               const DiracWf<FermionFlow::Out> &fo,
               const DiracWf<FermionFlow::In> &fi) -> void;

auto w_current(VectorWf *wf, const VertexFFV &v,
               const DiracWf<FermionFlow::Out> &fo,
               const DiracWf<FermionFlow::In> &fi) -> void;

auto all_same(Gen gen1, Gen gen2, Gen gen3, Gen gen4) -> bool;

} // namespace blackthorn
