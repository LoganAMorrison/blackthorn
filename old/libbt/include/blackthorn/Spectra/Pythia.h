#ifndef BLACKTHORN_PYTHIA_SPECTRA_H
#define BLACKTHORN_PYTHIA_SPECTRA_H

#include "blackthorn/Models/StandardModel.h"
#include "blackthorn/PhaseSpace.h"
#include <Pythia8/Pythia.h>
#include <boost/histogram.hpp>

namespace blackthorn {

namespace bp = boost::histogram;
using Pythia8::Pythia;

template <size_t N> class PythiaSpectrum {
  using LogAxis = bp::axis::regular<double, bp::axis::transform::log>;
  using Histogram = bp::histogram<std::tuple<LogAxis>>;

public:
  static constexpr double DEFAULT_XMIN = 1e-6;
  static constexpr double DEFAULT_XMAX = 1.0;
  static constexpr unsigned int DEFAULT_NBINS = 100;
  static constexpr unsigned int DEFAULT_NEVENTS = 10'000;

private:
  Pythia p_pythia{};
  double p_xmin = 1e-6;
  double p_xmax = 1.0;
  unsigned int p_nbins = DEFAULT_NBINS;
  unsigned int p_nevents = DEFAULT_NEVENTS;
  bool p_log_scale = true;

public:
  using SpectrumType = std::pair<std::vector<double>, std::vector<double>>;

  explicit PythiaSpectrum(double xmin = DEFAULT_XMIN,
                          double xmax = DEFAULT_XMAX,
                          unsigned int nbins = DEFAULT_XMAX,
                          unsigned int nevents = DEFAULT_NEVENTS,
                          bool log_scale = true)
      : p_xmin(xmin), p_xmax(xmax), p_nbins(nbins), p_nevents(nevents),
        p_log_scale(log_scale) {
    pythia_init();
  }

  auto xmin() const -> const double & { return p_xmin; }
  auto xmax() const -> const double & { return p_xmax; }
  auto nbins() const -> const unsigned int & { return p_nbins; }
  auto log_scale() const -> const bool & { return p_log_scale; }

  auto xmin() -> double & { return p_xmin; }
  auto xmax() -> double & { return p_xmax; }
  auto nbins() -> unsigned int & { return p_nbins; }
  auto log_scale() -> bool & { return p_log_scale; }

private:
  // ===========================================================================
  // ---- Initializations ------------------------------------------------------
  // ===========================================================================

  auto pythia_init() -> void {
    pythia_init_init();
    pythia_init_process_level();
    pythia_init_next();
    pythia_init_decays();
    p_pythia.init();
  }

  auto pythia_init_init() -> void {
    p_pythia.readString("Init:showChangedParticleData = off");
    p_pythia.readString("Init:showChangedSettings = off");
    p_pythia.readString("Init:showMultipartonInteractions = off");
    p_pythia.readString("Init:showProcesses = off");
  }

  auto pythia_init_process_level() -> void {
    p_pythia.readString("ProcessLevel:all = off");
  }

  auto pythia_init_next() -> void {
    p_pythia.readString("Next:numberShowInfo = 0");
    p_pythia.readString("Next:numberShowProcess = 0");
    p_pythia.readString("Next:numberShowEvent = 0");
    p_pythia.readString("Next:numberCount = 0");
  }

  auto pythia_init_decays() -> void {
    p_pythia.particleData.mayDecay(Muon::pdg, true);
    p_pythia.particleData.mayDecay(NeutralPion::pdg, true);
    p_pythia.particleData.mayDecay(Eta::pdg, true);
    p_pythia.particleData.mayDecay(NeutralKaon::pdg, true);
    p_pythia.particleData.mayDecay(ShortKaon::pdg, true);
    p_pythia.particleData.mayDecay(LongKaon::pdg, true);
    p_pythia.particleData.mayDecay(ChargedPion::pdg, true);
    p_pythia.particleData.mayDecay(ChargedKaon::pdg, true);
  }

  // ===========================================================================
  // ---- Event Record Modifications -------------------------------------------
  // ===========================================================================

  auto generate_lifetime(int pdg, int idx) -> void {
    // Generate a lifetime to give decay away from a primary vertex
    if (p_pythia.particleData.canDecay(pdg)) {
      p_pythia.event[idx].tau(p_pythia.event[idx].tau0() * p_pythia.rndm.exp());
    }
  }

  auto append_outgoing(int pdg, double e, double px, double py, double pz,
                       double mass) -> void {
    static constexpr int HARD_OUTGOING_CODE = 23;
    static constexpr int COLOR_TAG = 101;
    // If 1 <= id <= 6, then we have a quark
    const int col =
        (DownQuark::pdg <= pdg && pdg <= TopQuark::pdg) ? COLOR_TAG : 0;
    // If -6 <= id <= -1, then we have an anti quark
    const int acol =
        (-TopQuark::pdg <= pdg && pdg <= -DownQuark::pdg) ? COLOR_TAG : 0;
    const int idx = p_pythia.event.append(pdg, HARD_OUTGOING_CODE, col, acol,
                                          px, py, pz, e, mass);
    generate_lifetime(pdg, idx);
  }

  auto append_outgoing(int pdg, const LVector<double> &p, double mass) -> void {
    append_outgoing(pdg, p[0], p[1], p[2], p[3], mass);
  }

  auto append_outgoing(int pdg, const LVector<double> &p) -> void {
    const auto mass = lnorm(p);
    append_outgoing(pdg, p, mass);
  }

  auto add_to_evnt_rec(const std::array<LVector<double>, N> &momenta,
                       const std::array<int, N> &pdgs) -> void {
    // Fill final-state particles of hard process into event record
    for (size_t i = 0; i < pdgs.size(); i++) {
      const auto pdg = pdgs[i];
      append_outgoing(pdg, momenta[i]);
    }
  }

  // ===========================================================================
  // ---- Spectrum Generation --------------------------------------------------
  // ===========================================================================

  auto get_masses(const std::array<int, N> &final_states) const
      -> std::array<double, N> {
    std::array<double, N> masses{};
    for (size_t i = 0; i < final_states.size(); i++) {
      masses[i] = p_pythia.particleData.m0(final_states.at(i));
    }
    return masses;
  }

  auto create_histogram() const -> Histogram {
    return boost::histogram::make_histogram(LogAxis{p_nbins, p_xmin, p_xmax});
  }

  auto add_to_hist(Histogram *hist, double mass, int product,
                   double hard_wgt) const -> void {
    // Loop over all particles in the event and add requested type to a
    // histogram
    for (const auto &event : p_pythia.event) {
      // Only count final state particles
      if (event.isFinal()) {
        const int id = event.id();
        if (id == product) {
          const double eng = event.e();
          const double wgt = p_pythia.info.weight() * hard_wgt;
          (*hist)(2 * eng / mass, boost::histogram::weight(wgt));
        }
      }
    }
  }

  auto hist_to_vectors(Histogram *hist, double norm) const
      -> std::pair<std::vector<double>, std::vector<double>> {
    // Put Histogram into vector
    std::vector<double> xs;
    std::vector<double> spec;

    xs.reserve(p_nbins);
    spec.reserve(p_nbins);

    for (auto &&h : bp::indexed(*hist)) {
      xs.emplace_back(h.bin().center());
      spec.emplace_back((*h) * norm);
    }
    return std::make_pair(xs, spec);
  }

public:
  template <class MSqrd>
  auto decay_spectrum(double mass, int product,
                      const std::array<int, N> &final_states, MSqrd msqrd)
      -> SpectrumType {

    // Create the hard process event generator
    auto masses = get_masses(final_states);
    RamboEventGenerator generator(msqrd, mass, masses);
    // RamboEventGenerator::MomentaType momenta(final_states.size());
    double hard_wgt = 0.0;
    PhaseSpaceEvent<N> event{};

    auto spectrum = create_histogram();
    double integral = 0.0;

    if (std::accumulate(masses.begin(), masses.end(), 0.0) < mass) {
      for (size_t n = 0; n < p_nevents; n++) {
        p_pythia.event.reset();
        generator.fill(&event);
        integral += event.weight();
        add_to_evnt_rec(event.momenta(), final_states);
        if (!p_pythia.next()) {
          continue;
        }
        add_to_hist(&spectrum, mass, product, event.weight());
      }
      const double norm = mass / (2 * integral);
      return hist_to_vectors(&spectrum, norm);
    }
    return hist_to_vectors(&spectrum, 0.0);
  }
};

} // namespace blackthorn

#endif // BLACKTHORN_PYTHIA_SPECTRA_H
