import enum

import numpy as np


class Gen(enum.IntEnum):
    Fst = 0
    Snd = 1
    Trd = 2


# Fermi constant in GeV^-2
GF = 1.1663787e-5
# Fine-structure constant of the electric coupling constant at zero momentum
ALPHA_EM = 1.0 / 137.0  # at p^2 = 0
# Electric change coupling at zero momentum
QE = 0.302862
# Sine of the weak mixing angle
SW = 0.480853
# Cosine of the weak mixing angle
CW = 0.876801

_ckm_lam = 0.22650
_ckm_a = 0.790
_ckm_rho = 0.141
_ckm_eta = 0.357

_s12 = _ckm_lam
_s23 = _ckm_a * _ckm_lam**2
_s13 = _ckm_a * _ckm_lam**3 * np.sqrt(_ckm_rho**2 + _ckm_eta**2)
_delta = np.arctan2(_ckm_eta, _ckm_rho)
_c12 = np.sqrt(1 - _s12**2)
_c13 = np.sqrt(1 - _s13**2)
_c23 = np.sqrt(1 - _s23**2)

# CKM matrix element of u-d
CKM_UD = _c12 * _c13
# CKM matrix element of u-s
CKM_US = _c13 * _s12
# CKM matrix element of u-b
CKM_UB = _s13 * np.exp(1j * _delta)
# CKM matrix element of c-d
CKM_CD = -(_c23 * _s12) - _c12 * np.exp(1j * _delta) * _s13 * _s23
# CKM matrix element of c-s
CKM_CS = _c12 * _c23 - np.exp(1j * _delta) * _s12 * _s13 * _s23
# CKM matrix element of c-b
CKM_CB = _c13 * _s23
# CKM matrix element of t-d
CKM_TD = -(_c12 * _c23 * np.exp(1j * _delta) * _s13) + _s12 * _s23
# CKM matrix element of t-s
CKM_TS = -(_c23 * np.exp(1j * _delta) * _s12 * _s13) - _c12 * _s23
# CKM matrix element of t-b
CKM_TB = _c13 * _c23

CKM = np.array(
    [
        [CKM_UD, CKM_US, CKM_UB],
        [CKM_CD, CKM_CS, CKM_CB],
        [CKM_TD, CKM_TS, CKM_TB],
    ]
)
