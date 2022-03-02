#define CATCH_CONFIG_MAIN

#include "Helas.h"
#include "Tools.h"
#include "blackthorn/Amplitudes.h"
#include "blackthorn/PhaseSpace.h"
#include "blackthorn/Tensors.h"
#include <algorithm>
#include <catch2/catch.hpp>
#include <numeric>

using namespace blackthorn;

using cmplx = std::complex<double>;

static constexpr size_t NEVENTS = 1000;
static constexpr size_t BATCH_SIZE = 100;
static constexpr size_t NFSPS = 4;
static constexpr double m1 = 0.0;
static constexpr double m2 = 1e-10;
static constexpr double m3 = 0.1;
static constexpr double m4 = 0.3;
static constexpr std::array<double, NFSPS> masses = {m1, m2, m3, m4};
static constexpr double cme = 1.0;

static constexpr auto msqrd(const std::array<LVector<double>, NFSPS> & /*ps*/)
    -> double {
  return 1.0;
}

static auto generate_dummy_events() -> std::vector<PhaseSpaceEvent<4>> {
  return Rambo<4>::generate_phase_space(msqrd, cme, {m1, m2, m3, m4}, NEVENTS,
                                        BATCH_SIZE);
}

static auto compare_with_helas(const DiracWf<FlowIn> &bt, // NOLINT
                               const DiracWf<FlowIn> &helas) -> void {
  CHECK(bt[0].real() == Approx(helas[0].real()));
  CHECK(bt[0].imag() == Approx(helas[0].imag()));
  CHECK(bt[1].real() == Approx(helas[1].real()));
  CHECK(bt[1].imag() == Approx(helas[1].imag()));
  CHECK(bt[2].real() == Approx(helas[2].real()));
  CHECK(bt[2].imag() == Approx(helas[2].imag()));
  CHECK(bt[3].real() == Approx(helas[3].real()));
  CHECK(bt[3].imag() == Approx(helas[3].imag()));
}

static auto compare_with_helas(const DiracWf<FlowOut> &bt, // NOLINT
                               const DiracWf<FlowOut> &helas) -> void {
  CHECK(bt[0].real() == Approx(helas[0].real()));
  CHECK(bt[0].imag() == Approx(helas[0].imag()));
  CHECK(bt[1].real() == Approx(helas[1].real()));
  CHECK(bt[1].imag() == Approx(helas[1].imag()));
  CHECK(bt[2].real() == Approx(helas[2].real()));
  CHECK(bt[2].imag() == Approx(helas[2].imag()));
  CHECK(bt[3].real() == Approx(helas[3].real()));
  CHECK(bt[3].imag() == Approx(helas[3].imag()));
}

static auto compare_with_helas(const VectorWf &bt, // NOLINT
                               const VectorWf &helas) -> void {
  CHECK(bt[0].real() == Approx(helas[0].real()));
  CHECK(bt[0].imag() == Approx(helas[0].imag()));
  CHECK(bt[1].real() == Approx(helas[1].real()));
  CHECK(bt[1].imag() == Approx(helas[1].imag()));
  CHECK(bt[2].real() == Approx(helas[2].real()));
  CHECK(bt[2].imag() == Approx(helas[2].imag()));
  CHECK(bt[3].real() == Approx(helas[3].real()));
  CHECK(bt[3].imag() == Approx(helas[3].imag()));
}

static auto compare_with_helas(const ScalarWf &bt, // NOLINT
                               const ScalarWf &helas) -> void {
  const auto wfb = bt.wavefunction();
  const auto wfh = helas.wavefunction();
  CHECK(wfb.real() == Approx(wfh.real()));
  CHECK(wfb.imag() == Approx(wfh.imag()));
}

// ===========================================================================
// ---- Dirac Spinor-U Wavefunctions -----------------------------------------
// ===========================================================================

TEST_CASE("SpinorU w/ spin=+1", "[dirac][spinor_u][spin=+1]") {
  auto events = generate_dummy_events();
  constexpr int spin = 1;

  auto compare = [](const LVector<double> &p, double m) {
    const auto bt = spinor_u(p, m, spin);
    const auto helas = helas_spinor_u(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (size_t i = 0; i < masses.size(); i++) {
    const double m = masses.at(i);
    for (const auto &event : events) {
      const auto p = event.momenta(i);
      compare(p, m);
    }
  }
}

TEST_CASE("SpinorU w/ spin=-1", "[dirac][spinor_u][spin=-1]") {
  auto events = generate_dummy_events();
  constexpr int spin = -1;

  auto compare = [](const LVector<double> &p, double m) {
    const auto bt = spinor_u(p, m, spin);
    const auto helas = helas_spinor_u(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (size_t i = 0; i < masses.size(); i++) {
    const double m = masses.at(i);
    for (const auto &event : events) {
      const auto p = event.momenta(i);
      compare(p, m);
    }
  }
}

// ===========================================================================
// ---- Dirac Spinor-V Wavefunctions -----------------------------------------
// ===========================================================================

TEST_CASE("SpinorV w/ spin=+1", "[dirac][spinor_v][spin=+1]") {
  auto events = generate_dummy_events();
  constexpr int spin = 1;

  auto compare = [](const LVector<double> &p, double m) {
    const auto bt = spinor_v(p, m, spin);
    const auto helas = helas_spinor_v(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (size_t i = 0; i < masses.size(); i++) {
    const double m = masses.at(i);
    for (const auto &event : events) {
      const auto p = event.momenta(i);
      compare(p, m);
    }
  }
}

TEST_CASE("SpinorV w/ spin=-1", "[dirac][spinor_v][spin=-1]") {
  auto events = generate_dummy_events();
  constexpr int spin = -1;

  auto compare = [](const LVector<double> &p, double m) {
    const auto bt = spinor_v(p, m, spin);
    const auto helas = helas_spinor_v(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (size_t i = 0; i < masses.size(); i++) {
    const double m = masses.at(i);
    for (const auto &event : events) {
      const auto p = event.momenta(i);
      compare(p, m);
    }
  }
}

// ===========================================================================
// ---- Dirac Spinor-UBar Wavefunctions --------------------------------------
// ===========================================================================

TEST_CASE("SpinorUBar w/ spin=+1", "[dirac][spinor_ubar][spin=+1]") {
  auto events = generate_dummy_events();
  constexpr int spin = 1;

  auto compare = [](const LVector<double> &p, double m) {
    const auto bt = spinor_ubar(p, m, spin);
    const auto helas = helas_spinor_ubar(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (size_t i = 0; i < masses.size(); i++) {
    const double m = masses.at(i);
    for (const auto &event : events) {
      const auto p = event.momenta(i);
      compare(p, m);
    }
  }
}

TEST_CASE("SpinorUBar w/ spin=-1", "[dirac][spinor_ubar][spin=-1]") {
  auto events = generate_dummy_events();
  constexpr int spin = -1;

  auto compare = [](const LVector<double> &p, double m) {
    const auto bt = spinor_ubar(p, m, spin);
    const auto helas = helas_spinor_ubar(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (size_t i = 0; i < masses.size(); i++) {
    const double m = masses.at(i);
    for (const auto &event : events) {
      const auto p = event.momenta(i);
      compare(p, m);
    }
  }
}

// ===========================================================================
// ---- Dirac Spinor-VBar Wavefunctions --------------------------------------
// ===========================================================================

TEST_CASE("SpinorVBar w/ spin=+1", "[dirac][spinor_vbar][spin=+1]") {
  auto events = generate_dummy_events();
  constexpr int spin = 1;

  auto compare = [](const LVector<double> &p, double m) {
    const auto bt = spinor_vbar(p, m, spin);
    const auto helas = helas_spinor_vbar(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (size_t i = 0; i < masses.size(); i++) {
    const double m = masses.at(i);
    for (const auto &event : events) {
      const auto p = event.momenta(i);
      compare(p, m);
    }
  }
}

TEST_CASE("SpinorVBar w/ spin=-1", "[dirac][spinor_vbar][spin=-1]") {
  auto events = generate_dummy_events();
  constexpr int spin = -1;

  auto compare = [](const LVector<double> &p, double m) {
    const auto bt = spinor_vbar(p, m, spin);
    const auto helas = helas_spinor_vbar(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (size_t i = 0; i < masses.size(); i++) {
    const double m = masses.at(i);
    for (const auto &event : events) {
      const auto p = event.momenta(i);
      compare(p, m);
    }
  }
}

// ===========================================================================
// ---- Outgoing Vector Wavefunctions ----------------------------------------
// ===========================================================================

TEST_CASE("Outgoing vector w/ spin=+1", "[vector][outgoing][spin=+1]") {
  auto events = generate_dummy_events();
  constexpr int spin = 1;

  auto compare = [](const LVector<double> &p, double m) {
    const auto bt = vector_wf(p, m, spin, Outgoing);
    const auto helas = helas_vector_wf_final_state(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (size_t i = 1; i < masses.size(); i++) {
    const double m = masses.at(i);
    for (const auto &event : events) {
      const auto p = event.momenta(i);
      compare(p, m);
    }
  }
}

TEST_CASE("Outgoing vector w/ spin=-1", "[vector][outgoing][spin=-1]") {
  auto events = generate_dummy_events();
  constexpr int spin = -1;

  auto compare = [](const LVector<double> &p, double m) {
    const auto bt = vector_wf(p, m, spin, Outgoing);
    const auto helas = helas_vector_wf_final_state(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (size_t i = 1; i < masses.size(); i++) {
    const double m = masses.at(i);
    for (const auto &event : events) {
      const auto p = event.momenta(i);
      compare(p, m);
    }
  }
}

TEST_CASE("Outgoing vector w/ spin=0", "[vector][outgoing][spin=0]") {
  auto events = generate_dummy_events();
  constexpr int spin = 0;

  auto compare = [](const LVector<double> &p, double m) {
    const auto bt = vector_wf(p, m, spin, Outgoing);
    const auto helas = helas_vector_wf_final_state(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (size_t i = 1; i < masses.size(); i++) {
    const double m = masses.at(i);
    for (const auto &event : events) {
      const auto p = event.momenta(i);
      compare(p, m);
    }
  }
}

TEST_CASE("Outgoing vector w/ spin=+1, m=0",
          "[vector][outgoing][spin=+1][m=0]") {
  auto events = generate_dummy_events();
  constexpr int spin = 1;
  constexpr double m = m1;

  auto compare = [](const LVector<double> &p) {
    const auto bt = vector_wf(p, m, spin, Outgoing);
    const auto helas = helas_vector_wf_final_state(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (const auto &event : events) {
    const auto p = event.momenta(0);
    compare(p);
  }
}

TEST_CASE("Outgoing vector w/ spin=-1, m=0",
          "[vector][outgoing][spin=-1][m=0]") {
  auto events = generate_dummy_events();
  constexpr int spin = -1;
  constexpr double m = m1;

  auto compare = [](const LVector<double> &p) {
    const auto bt = vector_wf(p, m, spin, Outgoing);
    const auto helas = helas_vector_wf_final_state(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (const auto &event : events) {
    const auto p = event.momenta(0);
    compare(p);
  }
}

// ===========================================================================
// ---- Incoming Vector Wavefunctions ----------------------------------------
// ===========================================================================

TEST_CASE("Incoming vector w/ spin=+1", "[vector][incoming][spin=+1]") {
  auto events = generate_dummy_events();
  constexpr int spin = 1;

  auto compare = [](const LVector<double> &p, double m) {
    const auto bt = vector_wf(p, m, spin, Incoming);
    const auto helas = helas_vector_wf_initial_state(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (size_t i = 1; i < masses.size(); i++) {
    const double m = masses.at(i);
    for (const auto &event : events) {
      const auto p = event.momenta(i);
      compare(p, m);
    }
  }
}

TEST_CASE("Incoming vector w/ spin=-1", "[vector][incoming][spin=-1]") {
  auto events = generate_dummy_events();
  constexpr int spin = -1;

  auto compare = [](const LVector<double> &p, double m) {
    const auto bt = vector_wf(p, m, spin, Incoming);
    const auto helas = helas_vector_wf_initial_state(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (size_t i = 1; i < masses.size(); i++) {
    const double m = masses.at(i);
    for (const auto &event : events) {
      const auto p = event.momenta(i);
      compare(p, m);
    }
  }
}

TEST_CASE("Incoming vector w/ spin=0", "[vector][incoming][spin=0]") {
  auto events = generate_dummy_events();
  constexpr int spin = 0;

  auto compare = [](const LVector<double> &p, double m) {
    const auto bt = vector_wf(p, m, spin, Incoming);
    const auto helas = helas_vector_wf_initial_state(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (size_t i = 1; i < masses.size(); i++) {
    if (masses.at(i) == 0) {
      continue;
    }
    const double m = masses.at(i);
    for (const auto &event : events) {
      const auto p = event.momenta(i);
      compare(p, m);
    }
  }
}

TEST_CASE("Incoming vector w/ spin=+1, m=0",
          "[vector][incoming][spin=+1][m=0]") {
  auto events = generate_dummy_events();
  constexpr int spin = 1;
  constexpr double m = m1;

  auto compare = [](const LVector<double> &p) {
    const auto bt = vector_wf(p, m, spin, Incoming);
    const auto helas = helas_vector_wf_initial_state(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (const auto &event : events) {
    const auto p = event.momenta(0);
    compare(p);
  }
}

TEST_CASE("Incoming vector w/ spin=-1, m=0",
          "[vector][incoming][spin=-1][m=0]") {
  auto events = generate_dummy_events();
  constexpr int spin = -1;
  constexpr double m = m1;

  auto compare = [](const LVector<double> &p) {
    const auto bt = vector_wf(p, m, spin, Incoming);
    const auto helas = helas_vector_wf_initial_state(p, m, spin);
    compare_with_helas(bt, helas);
  };

  for (const auto &event : events) {
    const auto p = event.momenta(0);
    compare(p);
  }
}

// ===========================================================================
// ---- Off-Shell Scalar Wavefunctions ---------------------------------------
// ===========================================================================

TEST_CASE("Off-shell scalar", "[scalar][offshell][from-ubar-u]") {
  static thread_local std::random_device rd{};
  static thread_local std::mt19937 generator{rd()};
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  const std::array<double, 4> ms = {1, 2, 3, 4};
  auto events =
      Rambo<4>::generate_phase_space(msqrd, 25.0, ms, NEVENTS, BATCH_SIZE);
  constexpr int spin = -1;
  constexpr double m = m1;

  auto compare = [](const LVector<double> &p1, double m1, int spin1,
                    const LVector<double> &p2, double m2, int spin2,
                    double mass, double width, VertexFFS v) {
    const auto fobt = spinor_ubar(p1, m1, spin1);
    const auto fibt = spinor_u(p2, m2, spin2);
    const auto foh = helas_spinor_ubar(p1, m1, spin1);
    const auto fih = helas_spinor_u(p2, m2, spin2);
    auto bt = ScalarWf{};
    Current::generate(&bt, v, mass, width, fobt, fibt);
    auto v2 = VertexFFS{-tools::im * v.left, -tools::im * v.right};
    const auto helas = helas_offshell_scalar(fih, foh, v2, mass, width);
    compare_with_helas(bt, helas);
  };

  for (const auto &event : events) {
    const auto p1 = event.momenta(0);
    const auto p2 = event.momenta(1);
    const auto m1 = ms[0];
    const auto m2 = ms[1];
    const auto m3 = ms[2];
    const auto w = 1.0;
    const auto v = VertexFFS{distribution(generator), distribution(generator)};
    compare(p1, m1, 1, p2, m2, 1, m3, w, v);
  }
}

// ===========================================================================
// ---- Off-Shell Vector Wavefunctions ---------------------------------------
// ===========================================================================

TEST_CASE("Off-shell vector", "[vector][offshell][from-ubar-u]") {
  static thread_local std::random_device rd{};
  static thread_local std::mt19937 generator{rd()};
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  const std::array<double, 4> ms = {1, 2, 3, 4};
  auto events =
      Rambo<4>::generate_phase_space(msqrd, 25.0, ms, NEVENTS, BATCH_SIZE);
  constexpr int spin = -1;
  constexpr double m = m1;

  auto compare = [](const LVector<double> &p1, double m1, int spin1,
                    const LVector<double> &p2, double m2, int spin2,
                    double mass, double width, VertexFFV v) {
    const auto fobt = spinor_ubar(p1, m1, spin1);
    const auto fibt = spinor_u(p2, m2, spin2);
    const auto foh = helas_spinor_ubar(p1, m1, spin1);
    const auto fih = helas_spinor_u(p2, m2, spin2);
    auto bt = VectorWf{};
    Current::generate(&bt, v, mass, width, fobt, fibt);
    auto v2 = VertexFFV{-tools::im * v.left, -tools::im * v.right};
    const auto helas = helas_offshell_vector(fih, foh, v2, mass, width);
    compare_with_helas(bt, helas);
  };

  for (const auto &event : events) {
    const auto p1 = event.momenta(0);
    const auto p2 = event.momenta(1);
    const auto m1 = ms[0];
    const auto m2 = ms[1];
    const auto m3 = ms[2];
    const auto w = 1.0;
    const auto v = VertexFFV{distribution(generator), distribution(generator)};
    compare(p1, m1, 1, p2, m2, 1, m3, w, v);
  }
}
