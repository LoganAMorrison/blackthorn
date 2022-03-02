#ifndef BLACKTHORN_TESTS_TOOLS_H
#define BLACKTHORN_TESTS_TOOLS_H

#include "blackthorn/Amplitudes.h"
#include "blackthorn/Tensors.h"
#include "blackthorn/Wavefunctions.h"
#include <cmath>

namespace blackthorn {

constexpr auto fractional_diff(double actual, double estimate) -> double {
  return std::abs((actual - estimate) / actual);
}

auto print_fractional_diff(double actual, double estimate) -> void;

/**
 * Call HELAS sxxxxx to create a final-state scalar wavefunction.
 * @param p four-momentum
 */
auto helas_scalar_wf_final_state(LVector<double> p) -> ScalarWf;

/**
 * Call HELAS sxxxxx to create a initial-state scalar wavefunction.
 * @param p four-momentum
 */
auto helas_scalar_wf_initial_state(LVector<double> p) -> ScalarWf;

/**
 * Call HELAS ixxxxx to create a u-spinor.
 * @param p Four-momentum
 * @param mass Mass of the fermion
 * @param spin Spin of the fermion (-1 or 1)
 */
auto helas_spinor_u(LVector<double> p, double mass, int spin)
    -> DiracWf<FlowIn>;

/**
 * Call HELAS ixxxxx to create a v-spinor.
 * @param p Four-momentum
 * @param mass Mass of the fermion
 * @param spin Spin of the fermion (-1 or 1)
 */
auto helas_spinor_v(LVector<double> p, double mass, int spin)
    -> DiracWf<FlowIn>;

/**
 * Call HELAS oxxxxx to create a ubar-spinor.
 * @param p Four-momentum
 * @param mass Mass of the fermion
 * @param spin Spin of the fermion (-1 or 1)
 */
auto helas_spinor_ubar(LVector<double> p, double mass, int spin)
    -> DiracWf<FlowOut>;

/**
 * Call HELAS oxxxxx to create a vbar-spinor.
 * @param p Four-momentum
 * @param mass Mass of the fermion
 * @param spin Spin of the fermion (-1 or 1)
 */
auto helas_spinor_vbar(LVector<double> p, double mass, int spin)
    -> DiracWf<FlowOut>;

/**
 * Call HELAS vxxxxx to create a final-state vector wavefunction.
 * @param p Four-momentum
 * @param mass Mass of the vector
 * @param spin Spin of the vector (-1, 0 or 1)
 */
auto helas_vector_wf_final_state(LVector<double> p, double mass, int spin)
    -> VectorWf;

/**
 * Call HELAS vxxxxx to create a initial-state vector wavefunction.
 * @param p Four-momentum
 * @param mass Mass of the vector
 * @param spin Spin of the vector (-1, 0 or 1)
 */
auto helas_vector_wf_initial_state(LVector<double> p, double mass, int spin)
    -> VectorWf;

/**
 * Call HELAS hioxxx to create an off-shell scalar wf.
 * @param fi Flow-in fermion wavefunction
 * @param fo Flow-out fermion wavefunction
 * @param v Vertex
 * @param mass Mass of scalar
 * @param width Width of scalar
 */
auto helas_offshell_scalar(const DiracWf<FlowIn> &fi,
                           const DiracWf<FlowOut> &fo, const VertexFFS &v,
                           double mass, double width) -> ScalarWf;

/**
 * Call HELAS jioxxx to create an off-shell vector wf.
 * @param fi Flow-in fermion wavefunction
 * @param fo Flow-out fermion wavefunction
 * @param v Vertex
 * @param mass Mass of vector
 * @param width Width of vector
 */
auto helas_offshell_vector(const DiracWf<FlowIn> &fi,
                           const DiracWf<FlowOut> &fo, const VertexFFV &v,
                           double mass, double width) -> VectorWf;

} // namespace blackthorn

#endif // BLACKTHORN_TESTS_TOOLS_H
