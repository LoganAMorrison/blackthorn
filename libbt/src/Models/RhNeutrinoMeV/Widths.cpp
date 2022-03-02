#include "blackthorn/Amplitudes.h"
#include "blackthorn/Models/RhNeutrino.h"
#include "blackthorn/Models/StandardModel.h"

namespace blackthorn {

// ===========================================================================
// ---- N -> Nu + Pi0 --------------------------------------------------------
// ===========================================================================

auto RhNeutrinoMeV::width_v_pi0() const -> double {
  constexpr auto m0 = NeutralPion::mass;
  constexpr auto cw = StandardModel::cw;
  if (p_mass < m0) {
    return 0.0;
  }
  return (pow(FPI, 2) * pow(GF, 2) * std::abs(pow(m0, 2) - pow(p_mass, 2)) *
          (-pow(m0, 2) + pow(p_mass, 2)) * pow(sin(2 * p_theta), 2)) /
         (32.0 * pow(cw, 4) * M_PI * p_mass);
}

// ===========================================================================
// ---- N -> Nu + eta --------------------------------------------------------
// ===========================================================================

auto RhNeutrinoMeV::width_v_eta() const -> double {
  constexpr auto m0 = Eta::mass;
  constexpr auto cw = StandardModel::cw;
  if (p_mass < m0) {
    return 0.0;
  }
  return (std::pow(FPI, 2) * std::pow(GF, 2) *
          std::sqrt(std::pow(std::pow(m0, 2) - std::pow(p_mass, 2), 2)) *
          (-std::pow(m0, 2) + std::pow(p_mass, 2)) *
          std::pow(std::sin(2 * p_theta), 2)) /
         (96. * std::pow(cw, 4) * M_PI * p_mass);
}

// ===========================================================================
// ---- N -> L + Pi ----------------------------------------------------------
// ===========================================================================

auto RhNeutrinoMeV::width_l_pi() const -> double {
  constexpr auto mpi = ChargedPion::mass;
  const auto ml = StandardModel::charged_lepton_mass(p_gen);
  if (p_mass < ml + mpi) {
    return 0.0;
  }
  return (pow(FPI, 2) * pow(GF, 2) * std::norm(VUD) *
          std::sqrt(
              tools::kallen_lambda(pow(mpi, 2), pow(ml, 2), pow(p_mass, 2))) *
          (pow(ml, 4) - pow(mpi, 2) * pow(p_mass, 2) + pow(p_mass, 4) -
           pow(ml, 2) * (pow(mpi, 2) + 2 * pow(p_mass, 2))) *
          pow(sin(p_theta), 2)) /
         (8. * M_PI * pow(p_mass, 3));
}

// ===========================================================================
// ---- N -> L + K -----------------------------------------------------------
// ===========================================================================

auto RhNeutrinoMeV::width_l_k() const -> double {
  constexpr auto mk = ChargedKaon::mass;
  const auto ml = StandardModel::charged_lepton_mass(p_gen);
  if (p_mass < ml + mk) {
    return 0.0;
  }
  return (pow(FPI, 2) * pow(GF, 2) * std::norm(VUS) *
          std::sqrt(
              tools::kallen_lambda(pow(mk, 2), pow(ml, 2), pow(p_mass, 2))) *
          (pow(ml, 4) - pow(mk, 2) * pow(p_mass, 2) + pow(p_mass, 4) -
           pow(ml, 2) * (pow(mk, 2) + 2 * pow(p_mass, 2))) *
          pow(sin(p_theta), 2)) /
         (8. * M_PI * pow(p_mass, 3));
}

// ===========================================================================
// ---- N -> Nu + M + M ------------------------------------------------------
// ===========================================================================

auto RhNeutrinoMeV::width_v_pi_pi(size_t nevents, size_t batchsize) const
    -> std::pair<double, double> {
  constexpr auto mpi = ChargedPion::mass;
  if (p_mass < 2 * mpi) {
    return std::make_pair(0.0, 0.0);
  }
  const auto msqrd = SquaredAmplitudeNToVPiPi(*this);
  const std::array<double, 3> fsp_masses = {0.0, mpi, mpi};
  return Rambo<3>::decay_width(msqrd, p_mass, fsp_masses, nevents, batchsize);
}

// auto RhNeutrinoMeV::width_v_k_k() const -> std::pair<double, double> {
//   constexpr auto mk = ChargedKaon::mass;
//   if (p_mass < 2 * mk) {
//     return std::make_pair(0.0, 0.0);
//   }
//   const auto msqrd = SquaredAmplitudeNToVKK(*this);
//   const std::array<double, 3> fsp_masses = {0.0, mk, mk};
//   return Rambo<3>::decay_width(msqrd, p_mass, fsp_masses);
// }

// ===========================================================================
// ---- N -> L + Pi + Pi0 ----------------------------------------------------
// ===========================================================================

auto RhNeutrinoMeV::width_l_pi_pi0(size_t nevents, size_t batchsize) const
    -> std::pair<double, double> {
  constexpr auto mpi = ChargedPion::mass;
  constexpr auto mpi0 = NeutralPion::mass;
  const auto ml = StandardModel::charged_lepton_mass(p_gen);
  if (p_mass < mpi + ml + mpi0) {
    return std::make_pair(0.0, 0.0);
  }
  const auto msqrd = SquaredAmplitudeNToLPiPi0(*this);
  const std::array<double, 3> fsp_masses = {ml, mpi, mpi0};
  return Rambo<3>::decay_width(msqrd, p_mass, fsp_masses, nevents, batchsize);
}

// ===========================================================================
// ---- N -> nu + photon -----------------------------------------------------
// ===========================================================================

auto RhNeutrinoMeV::width_v_a() const -> double {
  using tools::im;
  using tools::sqr;
  static thread_local Loop loop{};
  constexpr auto mw = WBoson::mass;
  constexpr auto sw = StandardModel::sw;
  constexpr auto qe = StandardModel::qe;
  const auto mvr = p_mass;
  const auto ml = StandardModel::charged_lepton_mass(p_gen);
  const auto mvr2 = sqr(mvr);
  const auto ml2 = sqr(ml);
  const auto mw2 = sqr(mw);
  const auto ct = cos(p_theta);
  const auto st = sin(p_theta);

  // PVA[0, ml]
  const auto PVAL = loop.scalarA0(ml);
  // PVA[0, mw]
  const auto PVAW = loop.scalarA0(mw);

  // PVB[0, 0, 0, ml, ml] = B0
  const auto pvb1 = loop.tensor_coeffs_b(0, ml, ml, 0);
  // PVB[0, 0][0, ml, MW] = B0
  // PVB[0, 1][0, ml, MW] = B1
  const auto pvb2 = loop.tensor_coeffs_b(0, ml, mw, 1);
  // PVB[0, 0][mvr^2, ml, MW] = B0
  // PVB[0, 1][mvr^2, ml, MW] = B1
  const auto pvb3 = loop.tensor_coeffs_b(mvr2, ml, mw, 1);
  // PVB[0, 0][0, MW, MW] = B0
  // PVB[0, 1][0, MW, MW] = B1
  // PVB[1, 0][0, MW, MW] = B00
  const auto pvb4 = loop.tensor_coeffs_b(0, mw, mw, 2);

  // PVC[0, 0, 0][0, 0, mvr^2, ml, ml, MW] = C0
  // PVC[0, 1, 0][0, 0, mvr^2, ml, ml, MW] = C1
  // PVC[0, 0, 1][0, 0, mvr^2, ml, ml, MW] = C2
  // PVC[1, 0, 0][0, 0, mvr^2, ml, ml, MW] = C00
  // PVC[0, 2, 0][0, 0, mvr^2, ml, ml, MW] = C11
  // PVC[0, 1, 1][0, 0, mvr^2, ml, ml, MW] = C12
  // PVC[0, 0, 2][0, 0, mvr^2, ml, ml, MW] = C22
  const auto pvc1 = loop.tensor_coeffs_c(0, 0, mvr2, ml, ml, mw, 2);

  // PVC[0, 0, 0][mvr^2, 0, 0, ml, MW, MW] = C0
  // PVC[0, 1, 0][mvr^2, 0, 0, ml, MW, MW] = C1
  // PVC[0, 0, 1][mvr^2, 0, 0, ml, MW, MW] = C2
  // PVC[1, 0, 0][mvr^2, 0, 0, ml, MW, MW] = C00
  // PVC[0, 2, 0][mvr^2, 0, 0, ml, MW, MW] = C11
  // PVC[0, 1, 1][mvr^2, 0, 0, ml, MW, MW] = C12
  // PVC[0, 0, 2][mvr^2, 0, 0, ml, MW, MW] = C22
  const auto pvc2 = loop.tensor_coeffs_c(mvr2, 0, 0, ml, mw, mw, 2);

  const auto PVB1_00 = pvb1.coeff(0);

  const auto PVB2_00 = pvb2.coeff(0);
  const auto PVB2_01 = pvb2.coeff(1);

  const auto PVB3_00 = pvb3.coeff(0);
  const auto PVB3_01 = pvb3.coeff(1);

  const auto PVB4_10 = pvb4.coeff(2);

  const auto PVC1_000 = pvc1.coeff(0);
  const auto PVC1_010 = pvc1.coeff(1);
  const auto PVC1_001 = pvc1.coeff(2);
  const auto PVC1_100 = pvc1.coeff(3);
  const auto PVC1_020 = pvc1.coeff(4);
  const auto PVC1_011 = pvc1.coeff(5);
  const auto PVC1_002 = pvc1.coeff(6);

  const auto PVC2_000 = pvc2.coeff(0);
  const auto PVC2_010 = pvc2.coeff(1);
  const auto PVC2_001 = pvc2.coeff(2);
  const auto PVC2_100 = pvc2.coeff(3);
  const auto PVC2_020 = pvc2.coeff(4);
  const auto PVC2_011 = pvc2.coeff(5);
  const auto PVC2_002 = pvc2.coeff(6);

  const auto c1 =
      -2 * std::pow(mw, 2) *
          (PVC1_001 + PVC1_011 - PVC2_002 + PVC2_010 - PVC2_011) +
      std::pow(ml, 2) * (PVC1_000 + PVC1_001 - PVC1_011 + PVC2_000 +
                         2.0 * PVC2_001 + PVC2_002 + PVC2_010 + PVC2_011) +
      std::pow(mvr, 2) *
          (PVC1_001 + PVC1_002 + PVC1_011 + PVC2_010 + PVC2_011 + PVC2_020);
  const auto c2 =
      PVB1_00 - PVB2_00 +
      std::pow(mw, 2) * (PVC1_000 - 2.0 * (PVC1_001 + PVC1_011 - PVC2_002 +
                                           PVC2_010 - PVC2_011)) +
      std::pow(ml, 2) * (PVC1_001 - PVC1_011 + PVC2_000 + 2.0 * PVC2_001 +
                         PVC2_002 + PVC2_010 + PVC2_011) +
      std::pow(mvr, 2) * (-PVC1_000 - PVC1_001 + PVC1_002 - PVC1_010 +
                          PVC1_011 + PVC2_010 + PVC2_011 + PVC2_020);
  const auto c3 =
      mvr *
      (-PVB2_01 + PVB3_01 +
       std::pow(mvr, 2) * (PVC1_001 + PVC2_000 + 2.0 * PVC2_001 + PVC2_002 +
                           2.0 * (PVC2_010 + PVC2_011) + PVC2_020) -
       std::pow(ml, 2) * (PVC1_001 + PVC1_002 + PVC2_000 + 2.0 * PVC2_001 +
                          PVC2_002 + 2.0 * (PVC2_010 + PVC2_011) + PVC2_020) -
       std::pow(mw, 2) *
           (PVC1_001 + 2.0 * PVC1_002 + PVC2_000 + 3.0 * PVC2_001 +
            2.0 * PVC2_002 + 3.0 * PVC2_010 + 4.0 * PVC2_011 + 2.0 * PVC2_020));
  const auto c4 = -(
      mvr *
      (std::pow(ml, 2) * (PVC1_001 + PVC1_002 + PVC2_000 + 2.0 * PVC2_001 +
                          PVC2_002 + 2.0 * (PVC2_010 + PVC2_011) + PVC2_020) -
       std::pow(mvr, 2) * (PVC1_001 + PVC1_002 + PVC2_000 + 2.0 * PVC2_001 +
                           PVC2_002 + 2.0 * (PVC2_010 + PVC2_011) + PVC2_020) +
       std::pow(mw, 2) * (PVC1_001 + 2.0 * PVC1_002 + PVC2_000 +
                          3.0 * PVC2_001 + 2.0 * PVC2_002 + 3.0 * PVC2_010 +
                          4.0 * PVC2_011 + 2.0 * PVC2_020)));
  const auto c5 =
      (-PVAL + 2.0 * PVAW - 2.0 * PVB4_10 - std::pow(mvr, 4) * PVC1_001 +
       std::pow(mw, 2) * (PVB2_00 - 4.0 * (PVC1_100 + PVC2_100)) +
       std::pow(ml, 2) *
           (PVB2_00 + PVB3_00 - std::pow(mvr, 2) * (PVC1_000 + PVC1_001) -
            2.0 * (PVC1_100 + PVC2_100)) +
       std::pow(mvr, 2) *
           (-PVB2_00 +
            2.0 *
                (PVC1_100 + std::pow(mw, 2) * (PVC1_001 + PVC2_001 + PVC2_010) +
                 PVC2_100))) /
      2.0;
  const auto c6 =
      (-PVAL + 2.0 * PVAW - 2.0 * PVB4_10 +
       std::pow(mvr, 4) * (PVC1_000 + PVC1_001 + PVC1_010) +
       std::pow(mw, 2) * (PVB2_00 - 4.0 * (PVC1_100 + PVC2_100)) +
       std::pow(ml, 2) * (PVB2_00 + PVB3_00 - 2.0 * (PVC1_100 + PVC2_100)) -
       std::pow(mvr, 2) *
           (PVB1_00 + std::pow(ml, 2) * PVC1_001 +
            std::pow(mw, 2) *
                (PVC1_000 - 2.0 * (PVC1_001 + PVC2_001 + PVC2_010)) -
            2.0 * (PVC1_100 + PVC2_100))) /
      2.0;
  const auto c7 = -0.5 * im * mvr *
                  (-PVB1_00 - std::pow(ml, 2) * (PVC1_000 + PVC1_001) +
                   std::pow(mvr, 2) * (PVC1_000 + PVC1_001 + PVC1_010) +
                   std::pow(mw, 2) * (-PVC1_000 + 2.0 * PVC1_001 +
                                      4.0 * (PVC2_001 + PVC2_010)) +
                   2.0 * (PVC1_100 + PVC2_100));
  const auto c8 = -0.5 * im * mvr *
                  (-PVB1_00 - std::pow(ml, 2) * (PVC1_000 + PVC1_001) +
                   std::pow(mvr, 2) * (PVC1_000 + PVC1_001 + PVC1_010) +
                   std::pow(mw, 2) * (-PVC1_000 + 2.0 * PVC1_001 +
                                      4.0 * (PVC2_001 + PVC2_010)) +
                   2.0 * (PVC1_100 + PVC2_100));

  const auto msqrd =
      -im / 2.0 * std::pow(mvr, 2) *
      (4.0 * im * std::pow(std::abs(c5), 2) +
       4.0 * im * std::pow(std::abs(c6), 2) +
       mvr * (4.0 * im * mvr * std::pow(std::abs(c7), 2) +
              4.0 * im * mvr * std::pow(std::abs(c8), 2) +
              std::pow(mvr, 2) * c8 * std::conj(c1) +
              std::pow(mvr, 2) * c7 * std::conj(c2) + mvr * c7 * std::conj(c3) +
              mvr * c8 * std::conj(c4) + 6.0 * c8 * std::conj(c5) +
              6.0 * c7 * std::conj(c6) - std::pow(mvr, 2) * c2 * std::conj(c7) -
              mvr * c3 * std::conj(c7) - 6.0 * c6 * std::conj(c7) -
              (mvr * (mvr * c1 + c4) + 6.0 * c5) * std::conj(c8)));

  return std::real(msqrd) *
         (std::pow(ct, 2) * std::pow(GF, 2) * std::pow(st, 2) *
          StandardModel::alpha_em) /
         (32. * mvr * std::pow(M_PI, 4));
}

} // namespace blackthorn
