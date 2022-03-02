#include "ChargedMesonRadiativeDecay.h"
#include "blackthorn/Spectra/Base.h"
#include "blackthorn/Spectra/Boost.h"
#include "blackthorn/Spectra/Decay.h"

namespace blackthorn {

template <size_t N>
static auto interp(double x, const std::array<double, N> &xs,
                   const std::array<double, N> &ys) {
  if (x < xs.front() || x > xs.back()) {
    return 0.0;
  }
#pragma unroll N
  for (size_t i = 0; i < N - 1; ++i) {
    if (xs[i] < x && x < xs[i + 1]) {
      const double ya = ys[i];
      const double yb = ys[i + 1];
      const double xa = xs[i];
      const double xb = xs[i + 1];
      const double m = (yb - ya) / (xb - xa);
      const double b = (yb * xa - ya * xb) / (xa - xb);
      return m * x + b;
    }
  }
  return 0.0;
}

// ===========================================================================
// ---- K⁺ -> μ⁺ + νμ --------------------------------------------------------
// ===========================================================================

// Radiative decay spectrum for K⁺ -> μ⁺ + νμ in kaon rest frame.
class dnde_mu_nu {
  using Self = ChargedKaon;
  // μ⁺ energy in: K⁺ -> μ⁺ + νμ
  static constexpr double eng_mu_k_rf =
      tools::energy_one_cm(Self::mass, Muon::mass, 0.0);
  static constexpr double br = Self::BR_K_TO_MU_NUMU;

public:
  static auto dnde_photon(double e, double br_threshold) // NOLINT
      -> double {
    if (br < br_threshold) {
      return 0.0;
    }
    double result = 0.0;
    // contribution: K⁺ -> μ⁺ + νμ + γ
    result += br * dnde_x_to_lva<Self, Muon>(e);
    // contribution: K⁺ -> [μ⁺ -> e⁺ + νe + νμ + γ] + νμ
    result += br * decay_spectrum<Muon>::dnde_photon(e, eng_mu_k_rf);
    return result;
  }

  static auto dnde_positron(double e, double br_threshold) // NOLINT
      -> double {
    if (br < br_threshold) {
      return 0.0;
    }
    return br * decay_spectrum<Muon>::dnde_positron(e, eng_mu_k_rf);
  }
  static auto dnde_neutrino(double e, double br_threshold) // NOLINT
      -> NeutrinoSpectrum<double> {
    if (br < br_threshold) {
      return {0.0, 0.0, 0.0};
    }
    const auto mu = decay_spectrum<Muon>::dnde_neutrino(e, eng_mu_k_rf);
    return {mu.electron * br, mu.muon * br, mu.tau * br};
  }
};

// ===========================================================================
// ---- K⁺ -> π⁺ + π⁰ --------------------------------------------------------
// ===========================================================================

// Radiative decay spectrum for K⁺ -> π⁺ + π⁰ in kaon rest frame.
class dnde_pi_pi0 {
  using Self = ChargedKaon;
  static constexpr double eng_pi =
      tools::energy_one_cm(Self::mass, ChargedPion::mass, NeutralPion::mass);
  static constexpr double eng_pi0 = tools::energy_one_cm(
      ChargedKaon::mass, NeutralPion::mass, ChargedPion::mass);
  static constexpr double br = Self::BR_K_TO_PI_PI0;

public:
  static auto dnde_photon(double e, double br_threshold) // NOLINT
      -> double {
    if (br > br_threshold) {
      return 0.0;
    }
    return br * (decay_spectrum<ChargedPion>::dnde_photon(e, eng_pi) +
                 decay_spectrum<NeutralPion>::dnde_photon(e, eng_pi0));
  }
  static auto dnde_positron(double e, double br_threshold) // NOLINT
      -> double {
    if (br > br_threshold) {
      return 0.0;
    }
    return br * decay_spectrum<ChargedPion>::dnde_positron(e, eng_pi);
  }
  static auto dnde_neutrino(double e, double br_threshold) // NOLINT
      -> NeutrinoSpectrum<double> {
    if (br > br_threshold) {
      return {0.0, 0.0, 0.0};
    }
    const auto cp = decay_spectrum<ChargedPion>::dnde_neutrino(e, eng_pi);
    return {br * cp.electron, br * cp.muon, br * cp.tau};
  }
};

// ===========================================================================
// ---- K⁺ -> π⁺ + π⁺ + π⁻ ---------------------------------------------------
// ===========================================================================

// Radiative decay spectrum for K⁺ -> π⁺ + π⁺ + π⁻ in kaon rest frame.
class dnde_pi_pi_pi {
  using Self = ChargedKaon;
  static constexpr size_t NPTS = 25;
  static constexpr double br = Self::BR_K_TO_PI_PI_PI;
  // Energies centers of pion energy histogram
  static constexpr std::array<double, NPTS> epis = {
      0.14053199, 0.14245519, 0.14437838, 0.14630158, 0.14822478,
      0.15014798, 0.15207117, 0.15399437, 0.15591757, 0.15784077,
      0.15976396, 0.16168716, 0.16361036, 0.16553355, 0.16745675,
      0.16937995, 0.17130315, 0.17322634, 0.17514954, 0.17707274,
      0.17899594, 0.18091913, 0.18284233, 0.18476553, 0.18668873};
  static constexpr std::array<double, NPTS> inv_masses = {
      0.07886901, 0.08076789, 0.08266677, 0.08456564, 0.08646452,
      0.0883634,  0.09026227, 0.09216115, 0.09406003, 0.0959589,
      0.09785778, 0.09975666, 0.10165553, 0.10355441, 0.10545329,
      0.10735216, 0.10925104, 0.11114992, 0.11304879, 0.11494767,
      0.11684655, 0.11874542, 0.1206443,  0.12254318, 0.12444205};
  // Sum of all energy probability distrutions
  static constexpr std::array<double, NPTS> ps = {
      18.31571897, 33.13558448, 42.25466585, 49.45539952, 55.29858152,
      60.2342205,  64.30283612, 67.93114142, 71.18829728, 73.53772299,
      75.68278585, 77.41268225, 78.63814092, 79.32001934, 79.52327244,
      79.30849553, 78.52728658, 76.99406483, 74.8842747,  71.90197147,
      67.35277015, 61.98301612, 54.31593727, 43.63607248, 24.76731519};
  // Sum of all invariant mass distrutions (s, t and u)
  static constexpr std::array<double, NPTS> prob_inv_mass = {
      26.22156308, 44.5294645,  55.3310214,  63.07873886, 68.28636587,
      72.77335753, 75.81587833, 78.10811495, 79.48935211, 80.48702039,
      80.64143369, 80.33970351, 79.51198949, 78.34196716, 76.50202121,
      74.4323964,  72.05825195, 68.93434341, 65.26018019, 60.79082045,
      56.02725394, 50.02980863, 42.78605442, 33.62844184, 18.61796449};

public:
  static auto dnde_photon(double e, double br_threshold) // NOLINT
      -> double {
    if (br < br_threshold) {
      return 0.0;
    }
    static constexpr double de = epis[1] - epis[0];
    static constexpr double ds = inv_masses[1] - inv_masses[0];
    double res = 0.0;
#pragma unroll NPTS
    for (size_t i = 0; i < NPTS; ++i) {
      res += de * ps[i] *                                          // NOLINT
             decay_spectrum<ChargedPion>::dnde_photon(e, epis[i]); // NOLINT
      res += ds * prob_inv_mass[i] *                               // NOLINT
             dnde_photon_fsr<ChargedPion>(e, inv_masses[i]);       // NOLINT
    }
    return br * res;
  }

  static auto dnde_positron(double e, double br_threshold) // NOLINT
      -> double {
    if (br < br_threshold) {
      return 0.0;
    }
    static constexpr double de = epis[1] - epis[0];
    double res = 0.0;
#pragma unroll NPTS
    for (size_t i = 0; i < NPTS; ++i) {
      res += de * ps[i] *                                            // NOLINT
             decay_spectrum<ChargedPion>::dnde_positron(e, epis[i]); // NOLINT
    }
    return br * res;
  }

  static auto dnde_neutrino(double e, double br_threshold) // NOLINT
      -> NeutrinoSpectrum<double> {
    if (br < br_threshold) {
      return {0.0, 0.0, 0.0};
    }
    static constexpr double de = epis[1] - epis[0];
    double res_e = 0.0;
    double res_m = 0.0;
#pragma unroll NPTS
    for (size_t i = 0; i < NPTS; ++i) {
      const auto res =
          decay_spectrum<ChargedPion>::dnde_neutrino(e, epis[i]); // NOLINT
      res_e += de * ps[i] * res.electron;                         // NOLINT
      res_m += de * ps[i] * res.muon;                             // NOLINT
    }
    return {br * res_e, br * res_m, 0.0};
  }
};

// ===========================================================================
// ---- K⁺ -> π⁺ + π⁰ + π⁰ ---------------------------------------------------
// ===========================================================================

class dnde_pi_pi0_pi0 {
  using Self = ChargedKaon;
  static constexpr size_t NPTS = 25;
  static constexpr double br = Self::BR_K_TO_PI_PI0_PI0;
  // Energies centers of neutral pion energy histogram
  static constexpr std::array<double, NPTS> enps = {
      0.13605624, 0.13821513, 0.14037402, 0.14253291, 0.1446918,
      0.14685069, 0.14900957, 0.15116846, 0.15332735, 0.15548624,
      0.15764513, 0.15980401, 0.1619629,  0.16412179, 0.16628068,
      0.16843957, 0.17059846, 0.17275734, 0.17491623, 0.17707512,
      0.17923401, 0.1813929,  0.18355179, 0.18571067, 0.18786956};
  // Energies centers of charges pion energy histogram
  static constexpr std::array<double, NPTS> ecps = {
      0.14063417, 0.14276174, 0.14488931, 0.14701687, 0.14914444,
      0.15127201, 0.15339958, 0.15552714, 0.15765471, 0.15978228,
      0.16190984, 0.16403741, 0.16616498, 0.16829254, 0.17042011,
      0.17254768, 0.17467525, 0.17680281, 0.17893038, 0.18105795,
      0.18318551, 0.18531308, 0.18744065, 0.18956821, 0.19169578};
  // Energies centers of charges pion energy histogram
  static constexpr std::array<double, NPTS> inv_masses = {
      0.07886901, 0.08076789, 0.08266677, 0.08456564, 0.08646452,
      0.0883634,  0.09026227, 0.09216115, 0.09406003, 0.0959589,
      0.09785778, 0.09975666, 0.10165553, 0.10355441, 0.10545329,
      0.10735216, 0.10925104, 0.11114992, 0.11304879, 0.11494767,
      0.11684655, 0.11874542, 0.1206443,  0.12254318, 0.12444205};
  // Sum of neutral pion energy probability distrutions
  static constexpr std::array<double, NPTS> pnps = {
      12.28928787, 22.12425523, 27.84552276, 32.39570972, 35.92953124,
      38.68391095, 41.18530533, 42.96059154, 44.39666069, 45.59569091,
      46.41170912, 46.91065473, 47.01317075, 47.05455325, 46.63620471,
      45.83953193, 44.89975928, 43.52020258, 41.50375639, 39.39604067,
      36.7131146,  33.11014783, 28.6748182,  22.68053376, 12.63207681};
  // Charged pion energy probabilities
  static constexpr std::array<double, NPTS> pcps = {
      3.68832701,  6.92639232,  9.13431031,  11.02397439, 12.77783234,
      14.3703162,  15.82601179, 17.28993901, 18.68095221, 20.02055908,
      21.07653447, 22.25149912, 23.18971512, 24.20071169, 24.85178589,
      25.55147754, 25.89255313, 26.05050601, 25.87981321, 25.553746,
      24.68261575, 23.18369194, 20.86129677, 17.10166323, 9.95422247};
  // Charged pion energy probabilities
  static constexpr std::array<double, NPTS> prob_inv_mass = {
      6.72938217,  11.7907873,  15.0057192,  17.51940175, 19.44581822,
      21.14650185, 22.54927204, 23.87393793, 24.7111995,  25.52297305,
      26.12673552, 26.60593174, 26.80271582, 26.86310925, 26.83561915,
      26.54757235, 26.08185309, 25.53721261, 24.43375209, 23.20186867,
      21.71756572, 19.77470335, 17.17163865, 13.66287695, 7.67616684};

public:
  static auto dnde_photon(double e, double br_threshold) // NOLINT
      -> double {
    if (br < br_threshold) {
      return 0.0;
    }
    static constexpr double de0 = enps[1] - enps[0];
    static constexpr double dep = ecps[1] - ecps[0];
    static constexpr double ds = inv_masses[1] - inv_masses[0];
    double res = 0.0;
#pragma unroll NPTS
    for (size_t i = 0; i < NPTS; ++i) {
      res += de0 * pnps[i] *                                       // NOLINT
             decay_spectrum<NeutralPion>::dnde_photon(e, enps[i]); // NOLINT
      res += dep * pcps[i] *                                       // NOLINT
             decay_spectrum<ChargedPion>::dnde_photon(e, ecps[i]); // NOLINT
      res += ds * prob_inv_mass[i] *                               // NOLINT
             dnde_photon_fsr<ChargedPion>(e, inv_masses[i]);       // NOLINT
    }
    return br * res;
  }

  static auto dnde_positron(double e, double br_threshold) // NOLINT
      -> double {
    if (br < br_threshold) {
      return 0.0;
    }
    static constexpr double dep = ecps[1] - ecps[0];
    double res = 0.0;
#pragma unroll NPTS
    for (size_t i = 0; i < NPTS; ++i) {
      res += dep * pcps[i] *                                         // NOLINT
             decay_spectrum<ChargedPion>::dnde_positron(e, ecps[i]); // NOLINT
    }
    return br * res;
  }

  static auto dnde_neutrino(double e, double br_threshold) // NOLINT
      -> NeutrinoSpectrum<double> {
    if (br < br_threshold) {
      return {0.0, 0.0, 0.0};
    }
    static constexpr double dep = ecps[1] - ecps[0];
    double res_e = 0.0;
    double res_m = 0.0;
#pragma unroll NPTS
    for (size_t i = 0; i < NPTS; ++i) {
      const auto res =
          decay_spectrum<ChargedPion>::dnde_neutrino(e, ecps[i]); // NOLINT
      res_e += dep * pcps[i] * res.electron;                      // NOLINT
      res_m += dep * pcps[i] * res.muon;                          // NOLINT
    }
    return {br * res_e, br * res_m, 0.0};
  }
};

// ===========================================================================
// ---- K⁺ -> π⁰ + e⁺ + νe ---------------------------------------------------
// ===========================================================================

class dnde_pi0_e_nu {
  using Self = ChargedKaon;
  static constexpr size_t NPTS = 25;
  static constexpr double br = Self::BR_K_TO_E_NUE_PI0;
  // Energies centers of pion energy histogram
  static constexpr std::array<double, NPTS> epis = {
      0.13758307, 0.14279561, 0.14800815, 0.15322069, 0.15843323,
      0.16364577, 0.16885831, 0.17407086, 0.1792834,  0.18449594,
      0.18970848, 0.19492102, 0.20013356, 0.2053461,  0.21055864,
      0.21577118, 0.22098372, 0.22619626, 0.2314088,  0.23662134,
      0.24183388, 0.24704642, 0.25225897, 0.25747151, 0.26268405};
  // Energies centers of electron energy histogram
  static constexpr std::array<double, NPTS> els = {
      0.00506851, 0.01418354, 0.02329857, 0.03241359, 0.04152862,
      0.05064365, 0.05975868, 0.0688737,  0.07798873, 0.08710376,
      0.09621879, 0.10533381, 0.11444884, 0.12356387, 0.1326789,
      0.14179392, 0.15090895, 0.16002398, 0.16913901, 0.17825403,
      0.18736906, 0.19648409, 0.20559911, 0.21471414, 0.22382917};
  // Energies centers of electron-neutrino energy histogram
  static constexpr std::array<double, NPTS> evs = {
      0.00456493, 0.01369479, 0.02282464, 0.0319545,  0.04108436,
      0.05021422, 0.05934407, 0.06847393, 0.07760379, 0.08673365,
      0.09586351, 0.10499336, 0.11412322, 0.12325308, 0.13238294,
      0.14151279, 0.15064265, 0.15977251, 0.16890237, 0.17803222,
      0.18716208, 0.19629194, 0.2054218,  0.21455165, 0.22368151};
  // Energies centers of electron-neutrino energy histogram
  static constexpr std::array<double, NPTS> inv_masses = {
      0.00456493, 0.01369479, 0.02282464, 0.0319545,  0.04108436,
      0.05021422, 0.05934407, 0.06847393, 0.07760379, 0.08673365,
      0.09586351, 0.10499336, 0.11412322, 0.12325308, 0.13238294,
      0.14151279, 0.15064265, 0.15977251, 0.16890237, 0.17803222,
      0.18716208, 0.19629194, 0.2054218,  0.21455165, 0.22368151};
  // Probabilities P(E) for pion
  static constexpr std::array<double, NPTS> ppis = {
      0.05399869, 0.25404215,  0.5443153,   0.90636359,  1.33608116,
      1.81737548, 2.3520427,   2.94592689,  3.59247262,  4.26869723,
      4.99130105, 5.77062695,  6.58858674,  7.42932551,  8.31989454,
      9.26705327, 10.24102622, 11.20392835, 12.23979408, 13.36650717,
      14.4754074, 15.70550833, 16.82120141, 18.1082829,  19.24526569};
  // Probabilities P(E) for electron
  static constexpr std::array<double, NPTS> pls = {
      0.0330978,  0.2076308,  0.52590394, 0.97758683, 1.5280329,
      2.17281285, 2.87878445, 3.63907274, 4.38101266, 5.13070331,
      5.88261707, 6.52802282, 7.13941624, 7.59269515, 7.95206047,
      8.13585361, 8.12773825, 7.88392224, 7.38981395, 6.6941439,
      5.68583215, 4.44384868, 3.0056247,  1.48491278, 0.28780084};
  // Probabilities P(E) for electron-neutrino
  static constexpr std::array<double, NPTS> pvs = {
      0.02869022, 0.19451648, 0.50953769, 0.95206818, 1.49292183,
      2.13835846, 2.84857884, 3.59496632, 4.36271182, 5.10060696,
      5.84732321, 6.52199475, 7.08946374, 7.59705664, 7.91910514,
      8.1296687,  8.0929875,  7.89452871, 7.45403707, 6.70611235,
      5.72454925, 4.49716494, 3.02374799, 1.51114168, 0.29889449};
  // Probabilities P(E) for electron-neutrino
  static constexpr std::array<double, NPTS> prob_inv_mass = {
      0.02864568, 0.19599746, 0.51343323, 0.95280411, 1.50211814,
      2.14469718, 2.85085579, 3.58774335, 4.35910012, 5.09329214,
      5.81993441, 6.48871195, 7.12626326, 7.61355447, 7.94009797,
      8.11235226, 8.12208017, 7.88462623, 7.4482406,  6.70834274,
      5.7233828,  4.46921854, 3.03095432, 1.51368682, 0.30059927};

public:
  static auto dnde_photon(double e, double br_threshold) // NOLINT
      -> double {
    if (br < br_threshold) {
      return 0.0;
    }
    static constexpr double dep = epis[1] - epis[0];
    static constexpr double ds = inv_masses[1] - inv_masses[0];
    double res = 0;
#pragma unroll NPTS
    for (size_t i = 0; i < NPTS; ++i) {
      res += dep * ppis[i] *                                         // NOLINT
             decay_spectrum<NeutralPion>::dnde_positron(e, epis[i]); // NOLINT
      res += ds * prob_inv_mass[i] *                                 // NOLINT
             dnde_photon_fsr<Electron>(e, inv_masses[i]);            // NOLINT
    }
    return br * res;
  }

  static auto dnde_positron(double e, double br_threshold) // NOLINT
      -> double {
    // clang-format off
    if (br < br_threshold) { return 0.0; }
    // clang-format on
    return br * interp(e, els, pls);
  }

  static auto dnde_neutrino(double e, double br_threshold) // NOLINT
      -> NeutrinoSpectrum<double> {
    // clang-format off
    if (br < br_threshold) { return {0.0, 0.0, 0.0}; }
    // clang-format on
    return {br * interp(e, evs, pvs), 0.0, 0.0};
  }
};

// ===========================================================================
// ---- K⁺ -> π⁰ + μ⁺ + νμ ---------------------------------------------------
// ===========================================================================

class dnde_pi0_mu_nu {
  using Self = ChargedKaon;
  static constexpr size_t NPTS = 25;
  static constexpr double br = Self::BR_K_TO_MU_NUMU_PI0;
  // Energies centers of pion energy histogram
  static constexpr std::array<double, NPTS> epis = {
      0.13735694, 0.14211723, 0.14687751, 0.15163779, 0.15639808,
      0.16115836, 0.16591865, 0.17067893, 0.17543922, 0.1801995,
      0.18495978, 0.18972007, 0.19448035, 0.19924064, 0.20400092,
      0.20876121, 0.21352149, 0.21828177, 0.22304206, 0.22780234,
      0.23256263, 0.23732291, 0.24208319, 0.24684348, 0.25160376};
  // Energies centers of muon energy histogram
  static constexpr std::array<double, NPTS> els = {
      0.10833907, 0.11370046, 0.11906185, 0.12442324, 0.12978462,
      0.13514601, 0.1405074,  0.14586879, 0.15123018, 0.15659157,
      0.16195296, 0.16731435, 0.17267573, 0.17803712, 0.18339851,
      0.1887599,  0.19412129, 0.19948268, 0.20484407, 0.21020546,
      0.21556685, 0.22092823, 0.22628962, 0.23165101, 0.2370124};
  // Energies centers of muon-neutrino energy histogram
  static constexpr std::array<double, NPTS> evs = {
      0.00376383, 0.01129149, 0.01881916, 0.02634682, 0.03387448,
      0.04140214, 0.04892981, 0.05645747, 0.06398513, 0.07151279,
      0.07904046, 0.08656812, 0.09409578, 0.10162344, 0.10915111,
      0.11667877, 0.12420643, 0.13173409, 0.13926176, 0.14678942,
      0.15431708, 0.16184474, 0.16937241, 0.17690007, 0.18442773};
  // Energies centers of muon-neutrino energy histogram
  static constexpr std::array<double, NPTS> inv_masses = {
      0.00257357, 0.0077202,  0.01286682, 0.01801344, 0.02316006,
      0.02830669, 0.03345331, 0.03859993, 0.04374656, 0.04889318,
      0.0540398,  0.05918642, 0.06433305, 0.06947967, 0.07462629,
      0.07977292, 0.08491954, 0.09006616, 0.09521278, 0.10035941,
      0.10550603, 0.11065265, 0.11579928, 0.1209459,  0.12609252};
  // Probabilities P(E) for pion
  static constexpr std::array<double, NPTS> ppis = {
      0.92740826,  1.85324124,  2.59817324,  3.30826756,  4.02343886,
      4.75486668,  5.52240508,  6.28965003,  7.029923,    7.85691194,
      8.60157278,  9.47400012,  10.25876454, 11.07870559, 11.76608042,
      12.52434412, 13.16499019, 13.69906517, 14.06759035, 14.13035486,
      13.91794976, 12.91601512, 10.98198068, 7.32531911,  2.00047201};
  // Probabilities P(E) for muon
  static constexpr std::array<double, NPTS> pls = {
      2.47676132,  4.55809774,  5.95698744,  7.04648858,  8.00918964,
      8.77666118,  9.47534546,  10.01685361, 10.46325085, 10.77125052,
      11.07301301, 11.01410955, 10.98347422, 10.7834086,  10.46418194,
      9.9504274,   9.2972945,   8.49383935,  7.55065968,  6.41801919,
      5.21117149,  3.87054834,  2.48135026,  1.16339006,  0.213062};
  // Probabilities P(E) for muon-neutrino
  static constexpr std::array<double, NPTS> pvs = {
      0.02396644, 0.16403659, 0.43682254, 0.82059677, 1.30134408,
      1.88647482, 2.50242522, 3.19364071, 3.94955927, 4.70125484,
      5.45378062, 6.24586391, 6.92678225, 7.59985847, 8.20558756,
      8.66907103, 9.06436734, 9.34705526, 9.39762098, 9.28375363,
      8.94544807, 8.3070916,  7.32068034, 5.83514763, 3.26113194};
  // Probabilities P(E) for muon-neutrino
  static constexpr std::array<double, NPTS> prob_inv_mass = {
      19.52664808, 18.24593425, 16.99068179, 15.85725397, 14.6697194,
      13.55309612, 12.42217214, 11.39122763, 10.35698221, 9.38731174,
      8.45821146,  7.55669784,  6.67089689,  5.84945723,  5.06471292,
      4.32718572,  3.6326503,   2.9875266,   2.39965969,  1.85016421,
      1.3565369,   0.92188689,  0.54980105,  0.25686087,  0.05504305};

public:
  static auto dnde_photon(double e, double br_threshold) // NOLINT
      -> double {
    if (br < br_threshold) {
      return 0.0;
    }
    static constexpr double dep = epis[1] - epis[0];
    static constexpr double del = els[1] - els[0];
    static constexpr double ds = inv_masses[1] - inv_masses[0];
    const double x = 2 * e / Self::mass;
    double res = 0.0;
#pragma unroll NPTS
    for (size_t i = 0; i < NPTS; ++i) {
      res += del * pls[i] *                                        // NOLINT
             decay_spectrum<Muon>::dnde_photon(e, els[i]);         // NOLINT
      res += dep * ppis[i] *                                       // NOLINT
             decay_spectrum<NeutralPion>::dnde_photon(e, epis[i]); // NOLINT
      res += ds * prob_inv_mass[i] *                               // NOLINT
             dnde_photon_fsr<Muon>(e, inv_masses[i]);              // NOLINT
    }
    return br * res;
  }

  static auto dnde_positron(double e, double br_threshold) // NOLINT
      -> double {
    if (br < br_threshold) {
      return 0.0;
    }
    static constexpr double del = els[1] - els[0];
    double res = 0.0;
#pragma unroll NPTS
    for (size_t i = 0; i < NPTS; ++i) {
      res += del * pls[i] *                                  // NOLINT
             decay_spectrum<Muon>::dnde_positron(e, els[i]); // NOLINT
    }
    return br * res;
  }

  static auto dnde_neutrino(double e, double br_threshold) // NOLINT
      -> NeutrinoSpectrum<double> {
    if (br < br_threshold) {
      return {0.0, 0.0, 0.0};
    }
    static constexpr double del = els[1] - els[0];
    double res_e = 0.0;
    double res_m = interp(e, evs, pvs);
#pragma unroll NPTS
    for (size_t i = 0; i < NPTS; ++i) {
      const auto res = decay_spectrum<Muon>::dnde_neutrino(e, els[i]); // NOLINT
      res_e += del * pls[i] * res.electron;                            // NOLINT
      res_m += del * pls[i] * res.muon;                                // NOLINT
    }
    return {br * res_e, br * res_m, 0.0};
  }
};

// Kaon radiative decay spectrum for a kaon at rest
static double dnde_photon_rf(double egam, double br_threshold) {
  using Self = ChargedKaon;

  double result = 0.0;

  result += dnde_mu_nu::dnde_photon(egam, br_threshold);
  result += dnde_pi_pi0::dnde_photon(egam, br_threshold);
  result += dnde_pi_pi_pi::dnde_photon(egam, br_threshold);
  result += dnde_pi_pi0_pi0::dnde_photon(egam, br_threshold);
  result += dnde_pi0_e_nu::dnde_photon(egam, br_threshold);
  result += dnde_pi0_mu_nu::dnde_photon(egam, br_threshold);

  return result;
}

static double dnde_positron_rf(double egam, double br_threshold) {
  using Self = ChargedKaon;

  double result = 0.0;

  result += dnde_mu_nu::dnde_positron(egam, br_threshold);
  result += dnde_pi_pi0::dnde_positron(egam, br_threshold);
  result += dnde_pi_pi_pi::dnde_positron(egam, br_threshold);
  result += dnde_pi_pi0_pi0::dnde_positron(egam, br_threshold);
  result += dnde_pi0_e_nu::dnde_positron(egam, br_threshold);
  result += dnde_pi0_mu_nu::dnde_positron(egam, br_threshold);

  return result;
}

static auto dnde_neutrino_rf(double egam, double br_threshold)
    -> NeutrinoSpectrum<double> {
  using Self = ChargedKaon;
  NeutrinoSpectrum<double> res = {0.0, 0.0, 0.0};
  NeutrinoSpectrum<double> result = {0.0, 0.0, 0.0};

  res = dnde_mu_nu::dnde_neutrino(egam, br_threshold);
  result.electron += res.electron;
  result.muon += res.muon;

  res = dnde_pi_pi0::dnde_neutrino(egam, br_threshold);
  result.electron += res.electron;
  result.muon += res.muon;

  res = dnde_pi_pi_pi::dnde_neutrino(egam, br_threshold);
  result.electron += res.electron;
  result.muon += res.muon;

  res = dnde_pi_pi0_pi0::dnde_neutrino(egam, br_threshold);
  result.electron += res.electron;
  result.muon += res.muon;

  res = dnde_pi0_e_nu::dnde_neutrino(egam, br_threshold);
  result.electron += res.electron;
  result.muon += res.muon;

  res = dnde_pi0_mu_nu::dnde_neutrino(egam, br_threshold);
  result.electron += res.electron;
  result.muon += res.muon;

  return result;
}

auto decay_spectrum<ChargedKaon>::dnde_photon(double egam, double ek)
    -> double {
  using Self = ChargedKaon;
  if (ek < Self::mass) {
    return 0.0;
  }
  const auto spec_rf = [&](double eg) { return dnde_photon_rf(eg, 0.0); };
  return boost_spectrum(spec_rf, ek, Self::mass, egam, 0.0);
}

auto decay_spectrum<ChargedKaon>::dnde_positron(double e, double ek) -> double {
  using Self = ChargedKaon;
  if (ek < Self::mass) {
    return 0.0;
  }
  const auto spec_rf = [&](double eg) { return dnde_positron_rf(eg, 0.0); };
  return boost_spectrum(spec_rf, ek, Self::mass, e, Electron::mass);
}

auto decay_spectrum<ChargedKaon>::dnde_neutrino(double neutrino_energy,
                                                double ek)
    -> NeutrinoSpectrum<double> {
  using Self = ChargedKaon;

  if (ek < Self::mass) {
    return {0.0, 0.0, 0.0};
  }

  const auto electron = [&](double eg) {
    return dnde_neutrino_rf(eg, 0.0).electron;
  };
  const auto muon = [&](double eg) { return dnde_neutrino_rf(eg, 0.0).muon; };

  return {boost_spectrum(electron, ek, Self::mass, neutrino_energy, 0.0),
          boost_spectrum(muon, ek, Self::mass, neutrino_energy, 0.0), 0.0};
}

// ===========================================================================
// ---- Vectorized Versions --------------------------------------------------
// ===========================================================================

auto decay_spectrum<ChargedKaon>::dnde_photon(const std::vector<double> &egams,
                                              double ek)
    -> std::vector<double> {
  const auto f = [&](double x) { return dnde_photon(x, ek); };
  return tools::vectorized_par(f, egams);
}

auto decay_spectrum<ChargedKaon>::dnde_photon(const py::array_t<double> &egams,
                                              double ek)
    -> py::array_t<double> {
  const auto f = [&](double x) { return dnde_photon(x, ek); };
  return tools::vectorized(f, egams);
}

auto decay_spectrum<ChargedKaon>::dnde_positron(
    const std::vector<double> &positron_energy, double ek)
    -> std::vector<double> {
  const auto f = [&](double x) { return dnde_positron(x, ek); };
  return tools::vectorized_par(f, positron_energy);
}

auto decay_spectrum<ChargedKaon>::dnde_positron(
    const py::array_t<double> &positron_energy, double ek)
    -> py::array_t<double> {
  const auto f = [&](double x) { return dnde_photon(x, ek); };
  return tools::vectorized(f, positron_energy);
}

auto decay_spectrum<ChargedKaon>::dnde_neutrino(double neutrino_energy,
                                                double ek, Gen gen) -> double {
  auto result = dnde_neutrino(neutrino_energy, ek);
  switch (gen) {
  case Gen::Fst:
    return result.electron;
  case Gen::Snd:
    return result.muon;
  case Gen::Trd:
    return result.tau;
  default:
    return 0.0;
  }
}

auto decay_spectrum<ChargedKaon>::dnde_neutrino(
    const std::vector<double> &neutrino_energy, double ek)
    -> NeutrinoSpectrum<std::vector<double>> {
  std::vector<double> res_e(neutrino_energy.size(), 0.0);
  std::vector<double> res_mu(neutrino_energy.size(), 0.0);
  std::vector<double> res_tau(neutrino_energy.size(), 0.0);
  std::transform(neutrino_energy.begin(), neutrino_energy.end(), res_e.begin(),
                 [&](double x) { return dnde_neutrino(x, ek, Gen::Fst); });
  std::transform(neutrino_energy.begin(), neutrino_energy.end(), res_e.begin(),
                 [&](double x) { return dnde_neutrino(x, ek, Gen::Snd); });
  return {res_e, res_mu, res_tau};
}

} // namespace blackthorn
