#define CATCH_CONFIG_MAIN

#include "Helas.h"
#include "TestData.h"
#include "Tools.h"
#include "blackthorn/Amplitudes.h"
#include "blackthorn/PhaseSpace.h"
#include "blackthorn/Tensors.h"
#include <algorithm>
#include <catch2/catch.hpp>
#include <numeric>

using namespace blackthorn;

using cmplx = std::complex<double>;

static const double SQRT_3_2 = std::sqrt(3.0 / 2.0);
static const double SQRT_1_2 = std::sqrt(1.0 / 2.0);
static const double SQRT_2_1 = std::sqrt(2.0 / 1.0);

// ===========================================================================
// ---- Spinor-U -------------------------------------------------------------
// ===========================================================================

TEST_CASE("SpinorU", "[spinor_u]") { // NOLINT
  DiracWf<FlowIn> wf{};
  DVector<cmplx> expected;

  SECTION("random data, spin = 1") {
    for (size_t i = 0; i < N_TEST_PTS; ++i) {
      auto p = SPINOR_U_PS.at(i);
      expected = SPINOR_U_U_WFS.at(i);
      spinor_u(&wf, p, lnorm(p), 1);

      for (size_t j = 0; j < 4; ++j) {
        CHECK(wf[j].real() == Approx(expected[j].real()));
        CHECK(wf[j].imag() == Approx(expected[j].imag()));
      }
    }
  }
  SECTION("random data, spin = -1") {
    for (size_t i = 0; i < N_TEST_PTS; ++i) {
      auto p = SPINOR_U_PS.at(i);
      expected = SPINOR_U_D_WFS.at(i);
      spinor_u(&wf, p, lnorm(p), -1);

      for (size_t j = 0; j < 4; ++j) {
        CHECK(wf[j].real() == Approx(expected[j].real()));
        CHECK(wf[j].imag() == Approx(expected[j].imag()));
      }
    }
  }

  SECTION("pm = 0, pz = 0, m > 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, 0.0};
    expected = {0.0, 1.0, 0.0, 1.0};
    spinor_u(&wf, p, 1.0, 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = 0, pz = 0, m > 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, 0.0};
    expected = {-1.0, 0.0, -1.0, 0.0};
    spinor_u(&wf, p, 1.0, -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }

  SECTION("pm = pz > 0, m > 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, 0.5};
    expected = {SQRT_1_2, 0.0, SQRT_3_2, 0.0};
    spinor_u(&wf, p, lnorm(p), 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz > 0, m > 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, 0.5};
    expected = {0.0, SQRT_3_2, 0.0, SQRT_1_2};
    spinor_u(&wf, p, lnorm(p), -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz < 0, m > 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, -0.5};

    expected = {0.0, SQRT_1_2, 0.0, SQRT_3_2};
    spinor_u(&wf, p, lnorm(p), 1);

    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz < 0, m > 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, -0.5};
    expected = {-SQRT_3_2, 0.0, -SQRT_1_2, 0.0};
    spinor_u(&wf, p, lnorm(p), -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz > 0, m = 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, 1.0};
    expected = {0.0, 0.0, SQRT_2_1, 0.0};
    spinor_u(&wf, p, 0, 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz > 0, m = 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, 1.0};
    expected = {0.0, SQRT_2_1, 0.0, 0.0};
    spinor_u(&wf, p, 0, -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz < 0, m = 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, -1.0};
    expected = {0.0, 0.0, 0.0, SQRT_2_1};
    spinor_u(&wf, p, 0, 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz < 0, m = 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, -1.0};
    expected = {-SQRT_2_1, 0.0, 0.0, 0.0};
    spinor_u(&wf, p, 0, -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
}

// ===========================================================================
// ---- Spinor-V -------------------------------------------------------------
// ===========================================================================

TEST_CASE("SpinorV", "[spinor_v]") {
  DiracWf<FlowIn> wf{};
  DVector<cmplx> expected;

  SECTION("random data, spin = 1") {
    for (size_t i = 0; i < N_TEST_PTS; ++i) {
      auto p = SPINOR_V_PS.at(i);
      expected = SPINOR_V_U_WFS.at(i);
      spinor_v(&wf, p, lnorm(p), 1);

      for (size_t j = 0; j < 4; ++j) {
        CHECK(wf[j].real() == Approx(expected[j].real()));
        CHECK(wf[j].imag() == Approx(expected[j].imag()));
      }
    }
  }
  SECTION("random data, spin = -1") {
    for (size_t i = 0; i < N_TEST_PTS; ++i) {
      auto p = SPINOR_V_PS.at(i);
      expected = SPINOR_V_D_WFS.at(i);
      spinor_v(&wf, p, lnorm(p), -1);

      for (size_t j = 0; j < 4; ++j) {
        CHECK(wf[j].real() == Approx(expected[j].real()));
        CHECK(wf[j].imag() == Approx(expected[j].imag()));
      }
    }
  }

  SECTION("pm = 0, pz = 0, m > 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, 0.0};
    expected = {1.0, 0.0, -1.0, 0.0};
    spinor_v(&wf, p, 1.0, 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = 0, pz = 0, m > 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, 0.0};
    expected = {0.0, 1.0, 0.0, -1.0};
    spinor_v(&wf, p, 1.0, -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz > 0, m > 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, 0.5};
    expected = {0.0, -SQRT_3_2, 0.0, SQRT_1_2};
    spinor_v(&wf, p, lnorm(p), 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz > 0, m > 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, 0.5};
    expected = {SQRT_1_2, 0.0, -SQRT_3_2, 0.0};
    spinor_v(&wf, p, lnorm(p), -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz < 0, m > 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, -0.5};
    expected = {SQRT_3_2, 0.0, -SQRT_1_2, 0.0};
    spinor_v(&wf, p, lnorm(p), 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz < 0, m > 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, -0.5};
    expected = {0.0, SQRT_1_2, 0.0, -SQRT_3_2};
    spinor_v(&wf, p, lnorm(p), -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz > 0, m = 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, 1.0};
    expected = {0.0, -SQRT_2_1, 0.0, 0.0};
    spinor_v(&wf, p, 0, 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz > 0, m = 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, 1.0};
    expected = {0.0, 0.0, -SQRT_2_1, 0.0};
    spinor_v(&wf, p, 0, -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz < 0, m = 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, -1.0};
    expected = {SQRT_2_1, 0.0, 0.0, 0.0};
    spinor_v(&wf, p, 0, 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz < 0, m = 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, -1.0};
    expected = {0.0, 0.0, 0.0, -SQRT_2_1};
    spinor_v(&wf, p, 0, -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
}

// ===========================================================================
// ---- Spinor-UBar ----------------------------------------------------------
// ===========================================================================

TEST_CASE("SpinorUbar", "[spinor_ubar]") {
  DiracWf<FlowOut> wf{};
  DVector<cmplx> expected;

  SECTION("random data, spin = 1") {
    for (size_t i = 0; i < N_TEST_PTS; ++i) {
      auto p = SPINOR_UBAR_PS[i];
      expected = SPINOR_UBAR_U_WFS[i];
      spinor_ubar(&wf, p, lnorm(p), 1);

      for (size_t j = 0; j < 4; ++j) {
        CHECK(wf[j].real() == Approx(expected[j].real()));
        CHECK(wf[j].imag() == Approx(expected[j].imag()));
      }
    }
  }
  SECTION("random data, spin = -1") {
    for (size_t i = 0; i < N_TEST_PTS; ++i) {
      auto p = SPINOR_UBAR_PS[i];
      expected = SPINOR_UBAR_D_WFS[i];
      spinor_ubar(&wf, p, lnorm(p), -1);

      for (size_t j = 0; j < 4; ++j) {
        CHECK(wf[j].real() == Approx(expected[j].real()));
        CHECK(wf[j].imag() == Approx(expected[j].imag()));
      }
    }
  }
  SECTION("pm = 0, pz = 0, m > 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, 0.0};
    expected = {0.0, 1.0, 0.0, 1.0};
    spinor_ubar(&wf, p, 1.0, 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = 0, pz = 0, m > 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, 0.0};
    expected = {-1.0, 0.0, -1.0, 0.0};
    spinor_ubar(&wf, p, 1.0, -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz > 0, m > 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, 0.5};
    expected = {SQRT_3_2, 0.0, SQRT_1_2, 0.0};
    spinor_ubar(&wf, p, lnorm(p), 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz > 0, m > 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, 0.5};
    expected = {0.0, SQRT_1_2, 0.0, SQRT_3_2};
    spinor_ubar(&wf, p, lnorm(p), -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz < 0, m > 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, -0.5};
    expected = {0.0, SQRT_3_2, 0.0, SQRT_1_2};
    spinor_ubar(&wf, p, lnorm(p), 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz < 0, m > 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, -0.5};
    expected = {-SQRT_1_2, 0.0, -SQRT_3_2, 0.0};
    spinor_ubar(&wf, p, lnorm(p), -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz > 0, m = 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, 1.0};
    expected = {SQRT_2_1, 0.0, 0.0, 0.0};
    spinor_ubar(&wf, p, 0, 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz > 0, m = 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, 1.0};
    expected = {0.0, 0.0, 0.0, SQRT_2_1};
    spinor_ubar(&wf, p, 0, -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz < 0, m = 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, -1.0};
    expected = {0.0, SQRT_2_1, 0.0, 0.0};
    spinor_ubar(&wf, p, 0, 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz < 0, m = 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, -1.0};
    expected = {0.0, 0.0, -SQRT_2_1, 0.0};
    spinor_ubar(&wf, p, 0, -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
}

// ===========================================================================
// ---- Spinor-VBar ----------------------------------------------------------
// ===========================================================================

TEST_CASE("SpinorVBar", "[spinor_vbar]") {
  DiracWf<FlowOut> wf{};
  DVector<cmplx> expected;

  SECTION("random data, spin = 1") {
    for (size_t i = 0; i < N_TEST_PTS; ++i) {
      auto p = SPINOR_VBAR_PS[i];
      expected = SPINOR_VBAR_U_WFS[i];
      spinor_vbar(&wf, p, lnorm(p), 1);

      for (size_t j = 0; j < 4; ++j) {
        CHECK(wf[j].real() == Approx(expected[j].real()));
        CHECK(wf[j].imag() == Approx(expected[j].imag()));
      }
    }
  }
  SECTION("random data, spin = -1") {
    for (size_t i = 0; i < N_TEST_PTS; ++i) {
      auto p = SPINOR_VBAR_PS[i];
      expected = SPINOR_VBAR_D_WFS[i];
      spinor_vbar(&wf, p, lnorm(p), -1);

      for (size_t j = 0; j < 4; ++j) {
        CHECK(wf[j].real() == Approx(expected[j].real()));
        CHECK(wf[j].imag() == Approx(expected[j].imag()));
      }
    }
  }
  SECTION("pm = 0, pz = 0, m > 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, 0.0};
    expected = {-1.0, 0.0, 1.0, 0.0};
    spinor_vbar(&wf, p, 1.0, 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = 0, pz = 0, m > 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, 0.0};
    expected = {0.0, -1.0, 0.0, 1.0};
    spinor_vbar(&wf, p, 1.0, -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz > 0, m > 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, 0.5};

    expected = {0.0, SQRT_1_2, 0.0, -SQRT_3_2};
    spinor_vbar(&wf, p, lnorm(p), 1);

    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz > 0, m > 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, 0.5};
    expected = {-SQRT_3_2, 0.0, SQRT_1_2, 0.0};
    spinor_vbar(&wf, p, lnorm(p), -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz < 0, m > 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, -0.5};

    expected = {-SQRT_1_2, 0.0, SQRT_3_2, 0.0};
    spinor_vbar(&wf, p, lnorm(p), 1);

    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz < 0, m > 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, -0.5};
    expected = {0.0, -SQRT_3_2, 0.0, SQRT_1_2};
    spinor_vbar(&wf, p, lnorm(p), -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz > 0, m = 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, 1.0};
    expected = {0.0, 0.0, 0.0, -SQRT_2_1};
    spinor_vbar(&wf, p, 0, 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz > 0, m = 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, 1.0};
    expected = {-SQRT_2_1, 0.0, 0.0, 0.0};
    spinor_vbar(&wf, p, 0, -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz < 0, m = 0, spin = 1") {
    auto p = LVector<double>{1.0, 0, 0, -1.0};
    expected = {0.0, 0.0, SQRT_2_1, 0.0};
    spinor_vbar(&wf, p, 0, 1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
  SECTION("pm = pz < 0, m = 0, spin = -1") {
    auto p = LVector<double>{1.0, 0, 0, -1.0};
    expected = {0.0, -SQRT_2_1, 0.0, 0.0};
    spinor_vbar(&wf, p, 0, -1);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(wf[i].real() == Approx(expected[i].real()));
      CHECK(wf[i].imag() == Approx(expected[i].imag()));
    }
  }
}
