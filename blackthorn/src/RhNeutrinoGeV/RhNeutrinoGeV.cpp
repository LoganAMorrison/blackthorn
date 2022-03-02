#include "RhNeutrinoGeV.h"
#include "blackthorn/Spectra/Conv.h"
#include <tuple>

namespace blackthorn::py_interface::gev {

auto add_edist_v_u_u(py::class_<RhNeutrinoGeV> *rhn) -> void;
auto add_edist_v_d_d(py::class_<RhNeutrinoGeV> *rhn) -> void;
auto add_edist_l_u_d(py::class_<RhNeutrinoGeV> *rhn) -> void;
auto add_edist_v_l_l(py::class_<RhNeutrinoGeV> *rhn) -> void;

auto add_inv_mass_dist_v_u_u(py::class_<RhNeutrinoGeV> *rhn) -> void;
auto add_inv_mass_dist_v_d_d(py::class_<RhNeutrinoGeV> *rhn) -> void;
auto add_inv_mass_dist_v_l_l(py::class_<RhNeutrinoGeV> *rhn) -> void;
auto add_inv_mass_dist_l_u_d(py::class_<RhNeutrinoGeV> *rhn) -> void;

auto config_class(py::class_<RhNeutrinoGeV> *rhn) -> void {
  rhn->def(py::init<double, double, Gen>(), "");

  // ---- Accessors ----

  rhn->def_property(
      "mass", [](const RhNeutrinoGeV &model) { return model.mass(); },
      [](RhNeutrinoGeV &model) { return model.mass(); });
  rhn->def_property(
      "theta", [](const RhNeutrinoGeV &model) { return model.theta(); },
      [](RhNeutrinoGeV &model) { return model.theta(); });
  rhn->def_property(
      "gen", [](const RhNeutrinoGeV &model) { return model.gen(); },
      [](RhNeutrinoGeV &model) { return model.gen(); });

  // ---- Widths ----
  add_width_v_h(rhn);
  add_width_v_z(rhn);
  add_width_l_w(rhn);
  add_width_v_u_u(rhn);
  add_width_v_d_d(rhn);
  add_width_l_u_d(rhn);
  add_width_v_l_l(rhn);
  add_width_v_v_v(rhn);

  // ---- dndx ----
  add_dndx_v_h(rhn);
  add_dndx_v_z(rhn);
  add_dndx_l_w(rhn);
  add_dndx_v_u_u(rhn);
  add_dndx_v_d_d(rhn);
  add_dndx_l_u_d(rhn);
  add_dndx_v_l_l(rhn);

  // ---- energy distrubtions ----
  add_edist_v_u_u(rhn);
  add_edist_v_d_d(rhn);
  add_edist_l_u_d(rhn);
  add_edist_v_l_l(rhn);

  // ---- inv mass distrubtions ----
  add_inv_mass_dist_v_u_u(rhn);
  add_inv_mass_dist_v_d_d(rhn);
  add_inv_mass_dist_v_l_l(rhn);
  add_inv_mass_dist_l_u_d(rhn);
}

auto add_constructor(RhnClassType *rhn) -> void {
  static std::string doc = R"Doc(
Create a RhNeutrinoGeV object.

Parameters
----------
mass: float
  Mass of the right-handed neutrino.
theta: float
  Mixing angle between right-handed and active neutrino.
gen: Gen
  Generation of active neutrino the right-handed neturino mixes with.
)Doc";

  rhn->def(py::init<double, double, Gen>(), doc.c_str());
}

auto add_edist_v_u_u(py::class_<RhNeutrinoGeV> *rhn) -> void {
  static std::string doc = R"Doc(
Compute the energy distrubtions for final state particles from N->v+u+u.

Parameters
----------
genu: Gen
  Generation of the up-quarks.
nbins: List[int]
  List containing the number of bins to use for each particle.
nevents: int
  Number of events to use in generating the distrubtions.
)Doc";

  rhn->def(
      "v_u_u_energy_distributions",
      [](const blackthorn::RhNeutrinoGeV &model, Gen genu,
         const std::array<unsigned int, 3> &nbins, size_t nevents) {
        const auto msqrd = SquaredAmplitudeNToVUU(model, genu);

        const auto mu = StandardModel::up_type_quark_mass(genu);
        const std::array<double, 3> fsp_masses = {0.0, mu, mu};

        const auto dists = blackthorn::energy_distributions_linear(
            msqrd, model.mass(), fsp_masses, nbins, nevents);

        std::vector<std::pair<std::vector<double>, std::vector<double>>>
            out_dists(3);

        for (size_t i = 0; i < 3; i++) {
          out_dists[i].first.reserve(nbins[i]);
          out_dists[i].second.reserve(nbins[i]);
          for (auto &&h : blackthorn::bh::indexed(dists[i])) {
            const double p = *h;
            const double c = h.bin().center();
            out_dists[i].first.push_back(c);
            out_dists[i].second.push_back(p);
          }
        }

        return out_dists;
      },
      doc.c_str(), py::arg("genu"), py::arg("nbins"), py::arg("nevents"));
}

auto add_edist_v_d_d(py::class_<RhNeutrinoGeV> *rhn) -> void {
  static std::string doc = R"Doc(
Compute the energy distrubtions for final state particles from N->v+d+d.

Parameters
----------
gend: Gen
  Generation of the down-quarks.
nbins: List[int]
  List containing the number of bins to use for each particle.
nevents: int
  Number of events to use in generating the distrubtions.
)Doc";

  rhn->def(
      "v_d_d_energy_distributions",
      [](const blackthorn::RhNeutrinoGeV &model, Gen gend,
         const std::array<unsigned int, 3> &nbins, size_t nevents) {
        const auto msqrd = SquaredAmplitudeNToVDD(model, gend);

        const auto md = StandardModel::down_type_quark_mass(gend);
        const std::array<double, 3> fsp_masses = {0.0, md, md};

        const auto dists = blackthorn::energy_distributions_linear(
            msqrd, model.mass(), fsp_masses, nbins, nevents);

        std::vector<std::pair<std::vector<double>, std::vector<double>>>
            out_dists(3);

        for (size_t i = 0; i < 3; i++) {
          out_dists[i].first.reserve(nbins[i]);
          out_dists[i].second.reserve(nbins[i]);
          for (auto &&h : blackthorn::bh::indexed(dists[i])) {
            const double p = *h;
            const double c = h.bin().center();
            out_dists[i].first.push_back(c);
            out_dists[i].second.push_back(p);
          }
        }

        return out_dists;
      },
      doc.c_str(), py::arg("gend"), py::arg("nbins"), py::arg("nevents"));
}

auto add_edist_l_u_d(py::class_<RhNeutrinoGeV> *rhn) -> void {
  static std::string doc = R"Doc(
Compute the energy distrubtions for final state particles from N->l+u+l.

Parameters
----------
genu: Gen
  Generation of the up-quark.
gend: Gen
  Generation of the down-quark.
nbins: List[int]
  List containing the number of bins to use for each particle.
nevents: int
  Number of events to use in generating the distrubtions.
)Doc";

  rhn->def(
      "l_u_d_energy_distributions",
      [](const blackthorn::RhNeutrinoGeV &model, Gen genu, Gen gend,
         const std::array<unsigned int, 3> &nbins, size_t nevents) {
        const auto msqrd = SquaredAmplitudeNToLUD(model, genu, gend);

        const auto ml = StandardModel::charged_lepton_mass(model.gen());
        const auto mu = StandardModel::up_type_quark_mass(genu);
        const auto md = StandardModel::down_type_quark_mass(gend);
        const std::array<double, 3> fsp_masses = {ml, mu, md};

        const auto dists = blackthorn::energy_distributions_linear(
            msqrd, model.mass(), fsp_masses, nbins, nevents);

        std::vector<std::pair<std::vector<double>, std::vector<double>>>
            out_dists(3);

        for (size_t i = 0; i < 3; i++) {
          out_dists[i].first.reserve(nbins[i]);
          out_dists[i].second.reserve(nbins[i]);
          for (auto &&h : blackthorn::bh::indexed(dists[i])) {
            const double p = *h;
            const double c = h.bin().center();
            out_dists[i].first.push_back(c);
            out_dists[i].second.push_back(p);
          }
        }

        return out_dists;
      },
      doc.c_str(), py::arg("genu"), py::arg("gend"), py::arg("nbins"),
      py::arg("nevents"));
}

auto add_edist_v_l_l(py::class_<RhNeutrinoGeV> *rhn) -> void {

  static std::string doc = R"Doc(
Compute the energy distrubtions for final state particles from N->nu+l+l.

Parameters
----------
genv: Gen
  Generation of the neutrino.
genl1: Gen
  Generation of the first lepton.
genl2: Gen
  Generation of the second lepton.
nbins: List[int]
  List containing the number of bins to use for each particle.
nevents: int
  Number of events to use in generating the distrubtions.
)Doc";

  rhn->def(
      "v_l_l_energy_distributions",
      [](const blackthorn::RhNeutrinoGeV &model, Gen genv, Gen genl1, Gen genl2,
         const std::array<unsigned int, 3> &nbins, size_t nevents) {
        const auto msqrd = SquaredAmplitudeNToVLL(model, genv, genl1, genl2);

        const auto ml1 = StandardModel::charged_lepton_mass(genl1);
        const auto ml2 = StandardModel::charged_lepton_mass(genl2);
        const std::array<double, 3> fsp_masses = {0.0, ml1, ml2};

        const auto dists = blackthorn::energy_distributions_linear(
            msqrd, model.mass(), fsp_masses, nbins, nevents);

        std::vector<std::pair<std::vector<double>, std::vector<double>>>
            out_dists(3);

        for (size_t i = 0; i < 3; i++) {
          out_dists[i].first.reserve(nbins[i]);
          out_dists[i].second.reserve(nbins[i]);
          for (auto &&h : blackthorn::bh::indexed(dists[i])) {
            const double p = *h;
            const double c = h.bin().center();
            out_dists[i].first.push_back(c);
            out_dists[i].second.push_back(p);
          }
        }

        return out_dists;
      },
      doc.c_str(), py::arg("genv"), py::arg("genl1"), py::arg("genl2"),
      py::arg("nbins"), py::arg("nevents"));
}

auto add_inv_mass_dist_v_u_u(py::class_<RhNeutrinoGeV> *rhn) -> void {

  static std::string doc = R"Doc(
Compute the invariant mass distrubtion for final state up-quarks from N->nu+u+u.

Parameters
----------
genu: Gen
  Generation of the up-quarks.
nbins: int
  Number of bins to use in distrubtion. 
nevents: int
  Number of events to use in generating the distrubtions.

Returns
-------
ms: List[float]
  Masses for the distrubtion.
ps: List[float]
  Probabilities for the distrubtion.
)Doc";

  rhn->def(
      "inv_mass_distribution_v_u_u",
      [](const blackthorn::RhNeutrinoGeV &model, Gen genu,
         const unsigned int nbins, size_t nevents) {
        const auto msqrd = SquaredAmplitudeNToVUU(model, genu);

        const auto mu = StandardModel::up_type_quark_mass(genu);
        const std::array<double, 3> fsp_masses = {0.0, mu, mu};

        const auto dist = blackthorn::invariant_mass_distributions_linear<1, 2>(
            msqrd, model.mass(), fsp_masses, nbins, nevents);

        std::pair<std::vector<double>, std::vector<double>> out_dist{};
        out_dist.first.reserve(nbins);
        out_dist.second.reserve(nbins);

        for (auto &&h : blackthorn::bh::indexed(dist)) {
          const double p = *h;
          const double c = h.bin().center();
          out_dist.first.push_back(c);
          out_dist.second.push_back(p);
        }

        return out_dist;
      },
      doc.c_str(), py::arg("genu"), py::arg("nbins"), py::arg("nevents"));
}

auto add_inv_mass_dist_v_d_d(py::class_<RhNeutrinoGeV> *rhn) -> void {

  static std::string doc = R"Doc(
Compute the invariant mass distrubtion for final state down-quarks from N->nu+d+d.

Parameters
----------
gend: Gen
  Generation of the down-quarks.
nbins: int
  Number of bins to use in distrubtion. 
nevents: int
  Number of events to use in generating the distrubtions.

Returns
-------
ms: List[float]
  Masses for the distrubtion.
ps: List[float]
  Probabilities for the distrubtion.
)Doc";

  rhn->def(
      "inv_mass_distribution_v_d_d",
      [](const blackthorn::RhNeutrinoGeV &model, Gen gend,
         const unsigned int nbins, size_t nevents) {
        const auto msqrd = SquaredAmplitudeNToVDD(model, gend);

        const auto md = StandardModel::down_type_quark_mass(gend);
        const std::array<double, 3> fsp_masses = {0.0, md, md};

        const auto dist = blackthorn::invariant_mass_distributions_linear<1, 2>(
            msqrd, model.mass(), fsp_masses, nbins, nevents);

        std::pair<std::vector<double>, std::vector<double>> out_dist{};
        out_dist.first.reserve(nbins);
        out_dist.second.reserve(nbins);

        for (auto &&h : blackthorn::bh::indexed(dist)) {
          const double p = *h;
          const double c = h.bin().center();
          out_dist.first.push_back(c);
          out_dist.second.push_back(p);
        }

        return out_dist;
      },
      doc.c_str(), py::arg("gend"), py::arg("nbins"), py::arg("nevents"));
}

auto add_inv_mass_dist_v_l_l(py::class_<RhNeutrinoGeV> *rhn) -> void {

  static std::string doc = R"Doc(
Compute the invariant mass distrubtion for final state charged-leptons from N->nu+l+l.

Parameters
----------
genv: Gen
  Generation of the neutrino.
genl1: Gen
  Generation of the first charged lepton.
genl2: Gen
  Generation of the second charged lepton.
nbins: int
  Number of bins to use in distrubtion. 
nevents: int
  Number of events to use in generating the distrubtions.

Returns
-------
ms: List[float]
  Masses for the distrubtion.
ps: List[float]
  Probabilities for the distrubtion.
)Doc";

  rhn->def(
      "inv_mass_distribution_v_l_l",
      [](const blackthorn::RhNeutrinoGeV &model, Gen genv, Gen genl1, Gen genl2,
         const unsigned int nbins, size_t nevents) {
        const auto msqrd = SquaredAmplitudeNToVLL(model, genv, genl1, genl2);

        const auto ml1 = StandardModel::charged_lepton_mass(genl1);
        const auto ml2 = StandardModel::charged_lepton_mass(genl2);
        const std::array<double, 3> fsp_masses = {0.0, ml1, ml2};

        const auto dist = blackthorn::invariant_mass_distributions_linear<1, 2>(
            msqrd, model.mass(), fsp_masses, nbins, nevents);

        std::pair<std::vector<double>, std::vector<double>> out_dist{};
        out_dist.first.reserve(nbins);
        out_dist.second.reserve(nbins);

        for (auto &&h : blackthorn::bh::indexed(dist)) {
          const double p = *h;
          const double c = h.bin().center();
          out_dist.first.push_back(c);
          out_dist.second.push_back(p);
        }

        return out_dist;
      },
      doc.c_str(), py::arg("genv"), py::arg("genl1"), py::arg("genl2"),
      py::arg("nbins"), py::arg("nevents"));
}

auto add_inv_mass_dist_l_u_d(py::class_<RhNeutrinoGeV> *rhn) -> void {

  static std::string doc = R"Doc(
Compute the invariant mass distrubtions for lepton + up-quark and
up-quark + down-quark.

Parameters
----------
genu: Gen
  Generation of the up-quark.
gend: Gen
  Generation of the down-quark.
nbins: int
  Number of bins to use in distrubtion. 
nevents: int
  Number of events to use in generating the distrubtions.

Returns
-------
ms_lu: List[float]
  Masses for the lepton + up-quark distrubtion.
ps_lu: List[float]
  Probabilities for the lepton + up-quark distrubtion.
ms_ud: List[float]
  Masses for the up-quark + down-quark distrubtion.
ps_ud: List[float]
  Probabilities for the up-quark + down-quark distrubtion.
)Doc";

  rhn->def(
      "inv_mass_distributions_l_u_d",
      [](const blackthorn::RhNeutrinoGeV &model, Gen genu, Gen gend,
         const unsigned int nbins, size_t nevents) {
        const auto msqrd = SquaredAmplitudeNToLUD(model, genu, gend);

        const auto ml = StandardModel::charged_lepton_mass(model.gen());
        const auto mu = StandardModel::up_type_quark_mass(genu);
        const auto md = StandardModel::up_type_quark_mass(gend);
        const std::array<double, 3> fsp_masses = {ml, mu, md};

        const auto dist_lu =
            blackthorn::invariant_mass_distributions_linear<0, 1>(
                msqrd, model.mass(), fsp_masses, nbins, nevents);
        const auto dist_ud =
            blackthorn::invariant_mass_distributions_linear<1, 2>(
                msqrd, model.mass(), fsp_masses, nbins, nevents);

        std::vector<double> ms_lu{};
        ms_lu.reserve(nbins);
        std::vector<double> ms_ud{};
        ms_ud.reserve(nbins);
        std::vector<double> ps_lu{};
        ps_lu.reserve(nbins);
        std::vector<double> ps_ud{};
        ps_ud.reserve(nbins);

        for (auto &&h : blackthorn::bh::indexed(dist_lu)) {
          const double p = *h;
          const double c = h.bin().center();
          ms_lu.push_back(c);
          ps_lu.push_back(p);
        }
        for (auto &&h : blackthorn::bh::indexed(dist_ud)) {
          const double p = *h;
          const double c = h.bin().center();
          ms_ud.push_back(c);
          ps_ud.push_back(p);
        }

        return std::make_tuple(ms_lu, ps_lu, ms_ud, ps_ud);
      },
      doc.c_str(), py::arg("genu"), py::arg("gend"), py::arg("nbins"),
      py::arg("nevents"));
}

} // namespace blackthorn::py_interface::gev
