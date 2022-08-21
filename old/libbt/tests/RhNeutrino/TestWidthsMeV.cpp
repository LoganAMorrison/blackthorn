#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "../Tools.h"
#include "blackthorn/Amplitudes.h"
#include "blackthorn/Models/RhNeutrino.h"
#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/PhaseSpace.h"
#include "blackthorn/Tensors.h"
#include <algorithm>
#include <catch2/catch.hpp>
#include <fmt/format.h>
#include <numeric>
#include <tuple>

using namespace blackthorn; // NOLINT

auto gen_to_string(Gen gen) -> std::string {
  if (gen == Gen::Fst) {
    return "fst";
  }
  if (gen == Gen::Snd) {
    return "snd";
  }
  return "trd";
}

auto print_model(const RhNeutrinoMeV &model) -> void {
  std::cout << "RhNetrinoMeV(mass: " << model.mass()
            << ", theta: " << model.theta()
            << ", gen: " << gen_to_string(model.gen()) << ")\n";
}

TEST_CASE("Test N -> nu + photon", "[widths][va]") { // NOLINT
  using tools::print_width;
  using tools::UNICODE_MU;
  using tools::UNICODE_NU;

  std::cout << std::scientific;

  constexpr Gen genn = Gen::Fst;
  const std::string l = "e";
  auto model = RhNeutrinoMeV(0.4, 1e-3, genn);
  print_model(model);

  auto res = model.width_v_a();
  print_width("estimate", res, "N", {"v", "a"});
  std::cout << "\n";
}
