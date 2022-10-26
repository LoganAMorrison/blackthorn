# import pytest
# from pytest import approx
# import numpy as np


# from hazma import spectra
# from hazma import parameters
# from blackthorn import fields


# def test_dnde_muon_photon_rest_frame():
#     xs = np.geomspace(1e-3, 0.9, 100)

#     es_hazma = xs * parameters.muon_mass / 2.0
#     dnde_hazma = spectra.dnde_photon_muon(es_hazma, parameters.muon_mass) * 1e3

#     es_bt = xs * fields.Muon.mass / 2.0
#     dnde_bt = fields.Muon.dnde_photon(es_bt, fields.Muon.mass)

#     for h, b in zip(dnde_hazma, dnde_bt):
#         assert b == approx(h, rel=1e-3, abs=0)


# def test_dnde_muon_photon_boosted():
#     gamma = 2.0
#     xs = np.geomspace(1e-6, 1.0, 100)

#     emu_mev = gamma * parameters.muon_mass
#     emu_gev = gamma * fields.Muon.mass

#     es_hazma = xs * emu_mev / 2.0
#     dnde_hazma = spectra.dnde_photon_muon(es_hazma, emu_mev) * 1e3

#     es_bt = xs * emu_gev / 2.0
#     dnde_bt = fields.Muon.dnde_photon(es_bt, emu_gev)

#     for h, b in zip(dnde_hazma, dnde_bt):
#         assert b == approx(h, rel=1e-3, abs=0)


# def test_dnde_muon_positron_rest_frame():
#     xs = np.geomspace(1e-3, 0.5, 100)

#     es_hazma = xs * parameters.muon_mass / 2.0
#     dnde_hazma = spectra.dnde_positron_muon(es_hazma, parameters.muon_mass) * 1e3

#     es_bt = xs * fields.Muon.mass / 2.0
#     dnde_bt = fields.Muon.dnde_photon(es_bt, fields.Muon.mass)

#     for h, b in zip(dnde_hazma, dnde_bt):
#         assert b == approx(h, rel=1e-3, abs=0)


# def test_dnde_muon_positron_boosted():
#     gamma = 2.0
#     xs = np.geomspace(1e-6, 1.0, 100)

#     emu_mev = gamma * parameters.muon_mass
#     emu_gev = gamma * fields.Muon.mass

#     es_hazma = xs * emu_mev / 2.0
#     dnde_hazma = spectra.dnde_positron_muon(es_hazma, emu_mev) * 1e3

#     es_bt = xs * emu_gev / 2.0
#     dnde_bt = fields.Muon.dnde_positron(es_bt, emu_gev)

#     for h, b in zip(dnde_hazma, dnde_bt):
#         assert b == approx(h, rel=1e-3, abs=0)


# def test_dnde_charged_pion_photon_rest_frame():
#     xs = np.geomspace(1e-3, 0.9, 100)
#     epi_mev = parameters.charged_pion_mass
#     epi_gev = fields.ChargedPion.mass

#     es_hazma = xs * epi_mev / 2.0
#     dnde_hazma = spectra.dnde_photon_charged_pion(es_hazma, epi_mev) * 1e3

#     es_bt = xs * epi_gev / 2.0
#     dnde_bt = fields.ChargedPion.dnde_photon(es_bt, epi_gev)

#     for h, b in zip(dnde_hazma, dnde_bt):
#         assert b == approx(h, rel=1e-2, abs=0)


# def test_dnde_charged_pion_photon_boosted():
#     gamma = 2.0
#     xs = np.geomspace(1e-3, 0.9, 100)
#     epi_mev = gamma * parameters.charged_pion_mass
#     epi_gev = gamma * fields.ChargedPion.mass

#     es_hazma = xs * epi_mev / 2.0
#     dnde_hazma = spectra.dnde_photon_charged_pion(es_hazma, epi_mev) * 1e3

#     es_bt = xs * epi_gev / 2.0
#     dnde_bt = fields.ChargedPion.dnde_photon(es_bt, epi_gev)

#     for h, b in zip(dnde_hazma, dnde_bt):
#         assert b == approx(h, rel=1e-2, abs=0)
