#include "blackthorn/Models/StandardModel.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace bt = blackthorn;

PYBIND11_MODULE(constants, m) { // NOLINT
  m.attr("G_FERMI") = bt::StandardModel::g_fermi;
  m.attr("ALPHA_EM") = bt::StandardModel::alpha_em;
  m.attr("QE") = bt::StandardModel::qe;
  m.attr("SW") = bt::StandardModel::sw;
  m.attr("CW") = bt::StandardModel::cw;
  m.attr("HIGGS_VEV") = bt::Higgs::vev;
  m.attr("CKM_UD") = bt::StandardModel::ckm<bt::Gen::Fst, bt::Gen::Fst>();
  m.attr("CKM_US") = bt::StandardModel::ckm<bt::Gen::Fst, bt::Gen::Snd>();
  m.attr("CKM_UB") = bt::StandardModel::ckm<bt::Gen::Fst, bt::Gen::Trd>();
  m.attr("CKM_CD") = bt::StandardModel::ckm<bt::Gen::Snd, bt::Gen::Fst>();
  m.attr("CKM_CS") = bt::StandardModel::ckm<bt::Gen::Snd, bt::Gen::Snd>();
  m.attr("CKM_CB") = bt::StandardModel::ckm<bt::Gen::Snd, bt::Gen::Trd>();
  m.attr("CKM_TD") = bt::StandardModel::ckm<bt::Gen::Trd, bt::Gen::Fst>();
  m.attr("CKM_TS") = bt::StandardModel::ckm<bt::Gen::Trd, bt::Gen::Snd>();
  m.attr("CKM_TB") = bt::StandardModel::ckm<bt::Gen::Trd, bt::Gen::Trd>();

  m.attr("MASS_ELECTRON") = bt::Electron::mass;
  m.attr("MASS_MUON") = bt::Muon::mass;
  m.attr("MASS_TAU") = bt::Tau::mass;
  m.attr("MASS_UP_QUARK") = bt::UpQuark::mass;
  m.attr("MASS_CHARM_QUARK") = bt::CharmQuark::mass;
  m.attr("MASS_TOP_QUARK") = bt::TopQuark::mass;
  m.attr("MASS_DOWN_QUARK") = bt::DownQuark::mass;
  m.attr("MASS_STRANGE_QUARK") = bt::StrangeQuark::mass;
  m.attr("MASS_BOTTOM_QUARK") = bt::BottomQuark::mass;
  m.attr("MASS_ZBOSON") = bt::ZBoson::mass;
  m.attr("MASS_WBOSON") = bt::WBoson::mass;
  m.attr("MASS_HIGGS") = bt::Higgs::mass;
  m.attr("MASS_NEUTRAL_PION") = bt::NeutralPion::mass;
  m.attr("MASS_NEUTRAL_KAON") = bt::NeutralKaon::mass;
  m.attr("MASS_ETA") = bt::Eta::mass;
  m.attr("MASS_CHARGED_PION") = bt::ChargedPion::mass;
  m.attr("MASS_CHARGED_KAON") = bt::ChargedKaon::mass;

  m.attr("PDG_ELECTRON") = bt::Electron::pdg;
  m.attr("PDG_MUON") = bt::Muon::pdg;
  m.attr("PDG_TAU") = bt::Tau::pdg;
  m.attr("PDG_ELECTRON_NEUTRINO") = bt::ElectronNeutrino::pdg;
  m.attr("PDG_MUON_NEUTRINO") = bt::MuonNeutrino::pdg;
  m.attr("PDG_TAU_NEUTRINO") = bt::TauNeutrino::pdg;
  m.attr("PDG_UP_QUARK") = bt::UpQuark::pdg;
  m.attr("PDG_CHARM_QUARK") = bt::CharmQuark::pdg;
  m.attr("PDG_TOP_QUARK") = bt::TopQuark::pdg;
  m.attr("PDG_DOWN_QUARK") = bt::DownQuark::pdg;
  m.attr("PDG_STRANGE_QUARK") = bt::StrangeQuark::pdg;
  m.attr("PDG_BOTTOM_QUARK") = bt::BottomQuark::pdg;
  m.attr("PDG_ZBOSON") = bt::ZBoson::pdg;
  m.attr("PDG_WBOSON") = bt::WBoson::pdg;
  m.attr("PDG_HIGGS") = bt::Higgs::pdg;
  m.attr("PDG_NEUTRAL_PION") = bt::NeutralPion::pdg;
  m.attr("PDG_NEUTRAL_KAON") = bt::NeutralKaon::pdg;
  m.attr("PDG_ETA") = bt::Eta::pdg;
  m.attr("PDG_CHARGED_PION") = bt::ChargedPion::pdg;
  m.attr("PDG_CHARGED_KAON") = bt::ChargedKaon::pdg;
}
