#include "blackthorn/Spectra/Decay.h"
#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace bt = blackthorn;
namespace py = pybind11;

class Electron {
private:
  using F = bt::Electron;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr bt::Gen gen = F::gen;
  static constexpr double width = F::width;
};

class Muon {
private:
  using F = bt::Muon;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr bt::Gen gen = F::gen;
  static constexpr double width = F::width;

  static auto dnde_photon(double e, double emu) -> double {
    return blackthorn::decay_spectrum<F>::dnde_photon(e, emu);
  }
  static auto dnde_photon(const std::vector<double> &e, double emu)
      -> std::vector<double> {
    return blackthorn::decay_spectrum<F>::dnde_photon(e, emu);
  }
};

class Tau {
private:
  using F = bt::Tau;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr bt::Gen gen = F::gen;
  static constexpr double width = F::width;
};

class ElectronNeutrino {
private:
  using F = bt::ElectronNeutrino;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr bt::Gen gen = F::gen;
  static constexpr double width = F::width;
};

class MuonNeutrino {
private:
  using F = bt::MuonNeutrino;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr bt::Gen gen = F::gen;
  static constexpr double width = F::width;
};

class TauNeutrino {
private:
  using F = bt::TauNeutrino;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr bt::Gen gen = F::gen;
  static constexpr double width = F::width;
};

class UpQuark {
private:
  using F = bt::UpQuark;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr bt::Gen gen = F::gen;
  static constexpr double width = F::width;
};

class CharmQuark {
private:
  using F = bt::CharmQuark;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr bt::Gen gen = F::gen;
  static constexpr double width = F::width;
};

class TopQuark {
private:
  using F = bt::TopQuark;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr bt::Gen gen = F::gen;
  static constexpr double width = F::width;
};

class DownQuark {
private:
  using F = bt::DownQuark;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr bt::Gen gen = F::gen;
  static constexpr double width = F::width;
};

class StrangeQuark {
private:
  using F = bt::StrangeQuark;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr bt::Gen gen = F::gen;
  static constexpr double width = F::width;
};

class BottomQuark {
private:
  using F = bt::BottomQuark;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr bt::Gen gen = F::gen;
  static constexpr double width = F::width;
};

class Gluon {
private:
  using F = bt::Gluon;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr double width = F::width;
};

class Photon {
private:
  using F = bt::Photon;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr double width = F::width;
};

class ZBoson {
private:
  using F = bt::ZBoson;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr double width = F::width;
};

class WBoson {
private:
  using F = bt::WBoson;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr double width = F::width;
};

class Higgs {
private:
  using F = bt::Higgs;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;
  static constexpr double width = F::width;
  static constexpr double vev = F::vev;
};

class NeutralPion {
private:
  using F = bt::NeutralPion;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;

  static auto dnde_photon(double e, double epi) -> double {
    return blackthorn::decay_spectrum<F>::dnde_photon(e, epi);
  }
  static auto dnde_photon(const std::vector<double> &e, double epi)
      -> std::vector<double> {
    return blackthorn::decay_spectrum<F>::dnde_photon(e, epi);
  }
};

class ChargedPion {
private:
  using F = bt::ChargedPion;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;

  static auto dnde_photon(double e, double epi) -> double {
    return blackthorn::decay_spectrum<F>::dnde_photon(e, epi);
  }
  static auto dnde_photon(const std::vector<double> &e, double epi)
      -> std::vector<double> {
    return blackthorn::decay_spectrum<F>::dnde_photon(e, epi);
  }
};

class ChargedKaon {
private:
  using F = bt::ChargedKaon;

public:
  static constexpr double mass = F::mass;
  static constexpr int pdg = F::pdg;

  static auto dnde_photon(double e, double ek) -> double {
    return blackthorn::decay_spectrum<F>::dnde_photon(e, ek);
  }
  static auto dnde_photon(const std::vector<double> &e, double ek)
      -> std::vector<double> {
    return blackthorn::decay_spectrum<F>::dnde_photon(e, ek);
  }
};

template <class Field>
static auto define_dnde_photon(py::class_<Field> *f) -> void {
  static std::string doc_single = R"Doc(
Compute the photon decay spectrum.

Parameters
----------
egam: float
  Energy of the photon.
eng: float
  Energy of the decaying particle.

Returns
-------
dnde: float
  Value of the photon spectrum dN/dE.
)Doc";

  static std::string doc_vec = R"Doc(
Compute the photon decay spectrum.

Parameters
----------
egams: array-like
  Array of photon energies.
eng: float
  Energy of the decaying particle.

Returns
-------
dnde: array-like
  Photon spectrum dN/dE evaluated at the input energies.
)Doc";

  f->def_static(
      "dnde_photon",
      [](double e, double emu) { return Field::dnde_photon(e, emu); },
      doc_single.c_str(), py::arg("egam"), py::arg("eng"));
  f->def_static(
      "dnde_photon",
      [](const std::vector<double> &e, double emu) {
        return Field::dnde_photon(e, emu);
      },
      doc_vec.c_str(), py::arg("egams"), py::arg("eng"));
}

template <class Field> static auto define_mass(py::class_<Field> *f) -> void {
  f->def_property_readonly_static(
      "mass", [](py::object /*self*/) { return Field::mass; }, "mass in GeV");
}

template <class Field> static auto define_pdg(py::class_<Field> *f) -> void {
  f->def_property_readonly_static(
      "pdg", [](py::object /*self*/) { return Field::pdg; }, "PDG number");
}

template <class Field> static auto define_gen(py::class_<Field> *f) -> void {
  f->def_property_readonly_static(
      "gen", [](py::object /*self*/) { return Field::gen; },
      "Generation of the fermion");
}

PYBIND11_MODULE(fields, m) { // NOLINT
  // To define Gen
  py::module_::import("blackthorn.rh_neutrino");

  // =========================================================================
  // ---- Leptons ------------------------------------------------------------
  // =========================================================================

  auto electron = py::class_<Electron>(m, "Electron");
  electron.def(py::init<>());
  define_mass(&electron);
  define_pdg(&electron);
  define_gen(&electron);

  auto muon = py::class_<Muon>(m, "Muon");
  muon.def(py::init<>());
  define_mass(&muon);
  define_pdg(&muon);
  define_gen(&muon);
  define_dnde_photon(&muon);

  auto tau = py::class_<Tau>(m, "Tau");
  tau.def(py::init<>());
  define_mass(&tau);
  define_pdg(&tau);
  define_gen(&tau);

  // =========================================================================
  // ---- Neutrinos ----------------------------------------------------------
  // =========================================================================

  auto electron_nu = py::class_<ElectronNeutrino>(m, "ElectronNeutrino");
  electron_nu.def(py::init<>());
  define_mass(&electron_nu);
  define_pdg(&electron_nu);
  define_gen(&electron_nu);

  auto muon_nu = py::class_<MuonNeutrino>(m, "MuonNeutrino");
  muon_nu.def(py::init<>());
  define_mass(&muon_nu);
  define_pdg(&muon_nu);
  define_gen(&muon_nu);

  auto tau_nu = py::class_<TauNeutrino>(m, "TauNeutrino");
  tau_nu.def(py::init<>());
  define_mass(&tau_nu);
  define_pdg(&tau_nu);
  define_gen(&tau_nu);

  // =========================================================================
  // ---- Up-Type Quarks -----------------------------------------------------
  // =========================================================================

  auto qu = py::class_<UpQuark>(m, "UpQuark");
  qu.def(py::init<>());
  define_mass(&qu);
  define_pdg(&qu);
  define_gen(&qu);

  auto qc = py::class_<CharmQuark>(m, "CharmQuark");
  qc.def(py::init<>());
  define_mass(&qc);
  define_pdg(&qc);
  define_gen(&qc);

  auto qt = py::class_<TopQuark>(m, "TopQuark");
  qt.def(py::init<>());
  define_mass(&qt);
  define_pdg(&qt);
  define_gen(&qt);

  // =========================================================================
  // ---- Down-Type Quarks ---------------------------------------------------
  // =========================================================================

  auto qd = py::class_<DownQuark>(m, "DownQuark");
  qd.def(py::init<>());
  define_mass(&qd);
  define_pdg(&qd);
  define_gen(&qd);

  auto qs = py::class_<StrangeQuark>(m, "StrangeQuark");
  qs.def(py::init<>());
  define_mass(&qs);
  define_pdg(&qs);
  define_gen(&qs);

  auto qb = py::class_<BottomQuark>(m, "BottomQuark");
  qb.def(py::init<>());
  define_mass(&qb);
  define_pdg(&qb);
  define_gen(&qb);

  // =========================================================================
  // ---- Gauge Fields -------------------------------------------------------
  // =========================================================================

  auto gluon = py::class_<Gluon>(m, "Gluon");
  gluon.def(py::init<>());
  define_mass(&gluon);
  define_pdg(&gluon);

  auto photon = py::class_<Photon>(m, "Photon");
  photon.def(py::init<>());
  define_mass(&photon);
  define_pdg(&photon);

  auto zboson = py::class_<ZBoson>(m, "ZBoson");
  zboson.def(py::init<>());
  define_mass(&zboson);
  define_pdg(&zboson);

  auto wboson = py::class_<WBoson>(m, "WBoson");
  wboson.def(py::init<>());
  define_mass(&wboson);
  define_pdg(&wboson);

  auto higgs = py::class_<Higgs>(m, "Higgs");
  higgs.def(py::init<>());
  define_mass(&higgs);
  define_pdg(&higgs);
  higgs.def_property_readonly_static(
      "vev", [](py::object /*self*/) { return bt::Higgs::pdg; }, "Higgs vev");

  // =========================================================================
  // ---- Charged Pion -------------------------------------------------------
  // =========================================================================

  auto chgpion = py::class_<ChargedPion>(m, "ChargedPion");
  chgpion.def(py::init<>());
  define_mass(&chgpion);
  define_pdg(&chgpion);
  define_dnde_photon(&chgpion);

  // =========================================================================
  // ---- Charged Pion -------------------------------------------------------
  // =========================================================================

  auto neupion = py::class_<NeutralPion>(m, "NeutralPion");
  neupion.def(py::init<>());
  define_mass(&neupion);
  define_pdg(&neupion);
  define_dnde_photon(&neupion);

  // =========================================================================
  // ---- Charged Kaon -------------------------------------------------------
  // =========================================================================

  auto chgkaon = py::class_<ChargedKaon>(m, "ChargedKaon");
  chgkaon.def(py::init<>());
  define_mass(&chgkaon);
  define_pdg(&chgkaon);
  define_dnde_photon(&chgkaon);
}
