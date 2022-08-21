#define CATCH_CONFIG_MAIN

#include "TestData.h"
#include "blackthorn/Amplitudes.h"
#include "blackthorn/PhaseSpace.h"
#include "blackthorn/Tensors.h"
#include <algorithm>
#include <catch2/catch.hpp>
#include <numeric>

using namespace blackthorn;
using tools::im;

using cmplx = std::complex<double>;

static const double SQRT_3_2 = std::sqrt(3.0 / 2.0);
static const double SQRT_1_2 = std::sqrt(1.0 / 2.0);
static const double SQRT_1_3 = std::sqrt(1.0 / 3.0);
static const double SQRT_2_3 = std::sqrt(2.0 / 3.0);
static const double SQRT_2_1 = std::sqrt(2.0 / 1.0);

TEST_CASE("VectorWf", "[vector_wf]") {
  VectorWf wf{};
  LVector<cmplx> expected;
  LVector<double> k;

  // =========================================================================
  // ---- Final-State --------------------------------------------------------
  // =========================================================================

  SECTION("random data, final-state, spin = 1") {
    for (size_t i = 0; i < N_TEST_PTS; ++i) {
      auto p = POL_PS.at(i);
      expected = POL_U_WFS.at(i);
      vector_wf(&wf, p, lnorm(p), 1, Outgoing);
      for (size_t j = 0; j < 4; ++j) {
        CHECK(wf[j].real() == Approx(expected[j].real()));
        CHECK(wf[j].imag() == -Approx(expected[j].imag()));
      }
    }
  }
  SECTION("random data, final-state, spin = 0") {
    for (size_t i = 0; i < N_TEST_PTS; ++i) {
      auto p = POL_PS.at(i);
      expected = POL_L_WFS.at(i);
      vector_wf(&wf, p, lnorm(p), 0, Outgoing);
      for (size_t j = 0; j < 4; ++j) {
        CHECK(wf[j].real() == Approx(expected[j].real()));
        CHECK(wf[j].imag() == -Approx(expected[j].imag()));
      }
    }
  }
  SECTION("random data, final-state, spin = -1") {
    for (size_t i = 0; i < N_TEST_PTS; ++i) {
      auto p = POL_PS.at(i);
      expected = POL_D_WFS.at(i);
      vector_wf(&wf, p, lnorm(p), -1, Outgoing);
      for (size_t j = 0; j < 4; ++j) {
        CHECK(wf[j].real() == Approx(expected[j].real()));
        CHECK(wf[j].imag() == -Approx(expected[j].imag()));
      }
    }
  }

  // =========================================================================
  // ---- initial-State ------------------------------------------------------
  // =========================================================================

  SECTION("random data, initial-state, spin = 1") {
    for (size_t i = 0; i < N_TEST_PTS; ++i) {
      auto p = POL_PS.at(i);
      expected = POL_U_WFS.at(i);
      vector_wf(&wf, p, lnorm(p), 1, Incoming);

      for (size_t j = 0; j < 4; ++j) {
        CHECK(wf[j].real() == Approx(expected[j].real()));
        CHECK(wf[j].imag() == Approx(expected[j].imag()));
      }
    }
  }
  SECTION("random data, initial-state, spin = 0") {
    for (size_t i = 0; i < N_TEST_PTS; ++i) {
      auto p = POL_PS.at(i);
      expected = POL_L_WFS.at(i);
      vector_wf(&wf, p, lnorm(p), 0, Incoming);
      for (size_t j = 0; j < 4; ++j) {
        CHECK(wf[j].real() == Approx(expected[j].real()));
        CHECK(wf[j].imag() == -Approx(expected[j].imag()));
      }
    }
  }
  SECTION("random data, initial-state, spin = -1") {
    for (size_t i = 0; i < N_TEST_PTS; ++i) {
      auto p = POL_PS.at(i);
      expected = POL_D_WFS.at(i);
      vector_wf(&wf, p, lnorm(p), -1, Incoming);
      for (size_t j = 0; j < 4; ++j) {
        CHECK(wf[j].real() == Approx(expected[j].real()));
        CHECK(wf[j].imag() == Approx(expected[j].imag()));
      }
    }
  }

  // =========================================================================
  // ---- kt = 0, kz > 0, m > 0 ----------------------------------------------
  // =========================================================================

  SECTION("kt = 0, kz > 0, m > 0, initial-state, spin = 1") {
    k = {1.0, 0.0, 0.0, 0.5};
    expected = {0.0, -SQRT_1_2, -im * SQRT_1_2, 0.0};
    vector_wf(&wf, k, lnorm(k), 1, Incoming);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }
  SECTION("kt = 0, kz > 0, m > 0, initial-state, spin = 0") {
    k = {1.0, 0.0, 0.0, 0.5};
    expected = {SQRT_1_3, 0.0, 0.0, 2.0 * SQRT_1_3};
    vector_wf(&wf, k, lnorm(k), 0, Incoming);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }
  SECTION("kt = 0, kz > 0, m > 0, initial-state, spin = -1") {
    k = {1.0, 0.0, 0.0, 0.5};
    expected = {0.0, SQRT_1_2, -im * SQRT_1_2, 0.0};
    vector_wf(&wf, k, lnorm(k), -1, Incoming);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }

  SECTION("kt = 0, kz > 0, m > 0, final-state, spin = 1") {
    k = {1.0, 0.0, 0.0, 0.5};
    expected = {0.0, -SQRT_1_2, im * SQRT_1_2, 0.0};
    vector_wf(&wf, k, lnorm(k), 1, Outgoing);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }
  SECTION("kt = 0, kz > 0, m > 0, final-state, spin = 0") {
    k = {1.0, 0.0, 0.0, 0.5};
    expected = {SQRT_1_3, 0.0, 0.0, 2.0 * SQRT_1_3};
    vector_wf(&wf, k, lnorm(k), 0, Outgoing);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }
  SECTION("kt = 0, kz > 0, m > 0, final-state, spin = -1") {
    k = {1.0, 0.0, 0.0, 0.5};
    expected = {0.0, SQRT_1_2, im * SQRT_1_2, 0.0};
    vector_wf(&wf, k, lnorm(k), -1, Outgoing);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }

  // =========================================================================
  // ---- kt = 0, kz < 0, m > 0 ----------------------------------------------
  // =========================================================================

  SECTION("kt = 0, kz < 0, m > 0, initial-state, spin = 1") {
    k = {1.0, 0.0, 0.0, -0.5};
    expected = {0.0, -SQRT_1_2, im * SQRT_1_2, 0.0};
    vector_wf(&wf, k, lnorm(k), 1, Incoming);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }
  SECTION("kt = 0, kz < 0, m > 0, initial-state, spin = 0") {
    k = {1.0, 0.0, 0.0, -0.5};
    expected = {SQRT_1_3, 0.0, 0.0, -2.0 * SQRT_1_3};
    vector_wf(&wf, k, lnorm(k), 0, Incoming);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }
  SECTION("kt = 0, kz < 0, m > 0, initial-state, spin = -1") {
    k = {1.0, 0.0, 0.0, -0.5};
    expected = {0.0, SQRT_1_2, im * SQRT_1_2, 0.0};
    vector_wf(&wf, k, lnorm(k), -1, Incoming);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }

  SECTION("kt = 0, kz < 0, m > 0, final-state, spin = 1") {
    k = {1.0, 0.0, 0.0, -0.5};
    expected = {0.0, -SQRT_1_2, -im * SQRT_1_2, 0.0};
    vector_wf(&wf, k, lnorm(k), 1, Outgoing);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }
  SECTION("kt = 0, kz < 0, m > 0, final-state, spin = 0") {
    k = {1.0, 0.0, 0.0, -0.5};
    expected = {SQRT_1_3, 0.0, 0.0, -2.0 * SQRT_1_3};
    vector_wf(&wf, k, lnorm(k), 0, Outgoing);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }
  SECTION("kt = 0, kz < 0, m > 0, final-state, spin = -1") {
    k = {1.0, 0.0, 0.0, -0.5};
    expected = {0.0, SQRT_1_2, -im * SQRT_1_2, 0.0};
    vector_wf(&wf, k, lnorm(k), -1, Outgoing);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }

  // =========================================================================
  // ---- kt = 0, kz > 0, m = 0 ----------------------------------------------
  // =========================================================================

  SECTION("kt = 0, kz > 0, m = 0, initial-state, spin = 1") {
    k = {1.0, 0.0, 0.0, 1.0};
    expected = {0.0, -SQRT_1_2, -im * SQRT_1_2, 0.0};
    vector_wf(&wf, k, 0, 1, Incoming);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }
  SECTION("kt = 0, kz > 0, m = 0, initial-state, spin = -1") {
    k = {1.0, 0.0, 0.0, 1.0};
    expected = {0.0, SQRT_1_2, -im * SQRT_1_2, 0.0};
    vector_wf(&wf, k, 0, -1, Incoming);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }
  SECTION("kt = 0, kz > 0, m = 0, final-state, spin = 1") {
    k = {1.0, 0.0, 0.0, 1.0};
    expected = {0.0, -SQRT_1_2, im * SQRT_1_2, 0.0};
    vector_wf(&wf, k, 0, 1, Outgoing);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }
  SECTION("kt = 0, kz > 0, m = 0, final-state, spin = -1") {
    k = {1.0, 0.0, 0.0, 1.0};
    expected = {0.0, SQRT_1_2, im * SQRT_1_2, 0.0};
    vector_wf(&wf, k, 0, -1, Outgoing);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }

  // =========================================================================
  // ---- kt = 0, kz < 0, m = 0 ----------------------------------------------
  // =========================================================================

  SECTION("kt = 0, kz < 0, m = 0, initial-state, spin = 1") {
    k = {1.0, 0.0, 0.0, -1.0};
    expected = {0.0, -SQRT_1_2, im * SQRT_1_2, 0.0};
    vector_wf(&wf, k, 0, 1, Incoming);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }
  SECTION("kt = 0, kz < 0, m = 0, initial-State, spin = -1") {
    k = {1.0, 0.0, 0.0, -1.0};
    expected = {0.0, SQRT_1_2, im * SQRT_1_2, 0.0};
    vector_wf(&wf, k, 0, -1, Incoming);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }
  SECTION("kt = 0, kz < 0, m = 0, final-state, spin = 1") {
    k = {1.0, 0.0, 0.0, -1.0};
    expected = {0.0, -SQRT_1_2, -im * SQRT_1_2, 0.0};
    vector_wf(&wf, k, 0, 1, Outgoing);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }
  SECTION("kt = 0, kz < 0, m = 0, final-state, spin = -1") {
    k = {1.0, 0.0, 0.0, -1.0};
    expected = {0.0, SQRT_1_2, -im * SQRT_1_2, 0.0};
    vector_wf(&wf, k, 0, -1, Outgoing);
    for (size_t j = 0; j < 4; ++j) {
      CHECK(wf[j].real() == Approx(expected[j].real()));
      CHECK(wf[j].imag() == Approx(expected[j].imag()));
    }
  }
}
