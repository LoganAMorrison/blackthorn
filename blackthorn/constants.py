"""Useful constants for working with RHN models."""

# pylint: disable=invalid-name

import enum

import numpy as np


class Gen(enum.IntEnum):
    """Enumeration for fermion generations."""

    Fst = 0
    Snd = 1
    Trd = 2

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


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

_CKM_LAM = 0.22650
_CKM_A = 0.790
_CKM_RHO = 0.141
_CKM_ETA = 0.357

_SIN12 = _CKM_LAM
_SIN23 = _CKM_A * _CKM_LAM**2
_SIN13 = _CKM_A * _CKM_LAM**3 * np.sqrt(_CKM_RHO**2 + _CKM_ETA**2)
_DELTA = np.arctan2(_CKM_ETA, _CKM_RHO)
_COS12 = np.sqrt(1 - _SIN12**2)
_COS13 = np.sqrt(1 - _SIN13**2)
_COS23 = np.sqrt(1 - _SIN23**2)

# CKM matrix element of u-d
CKM_UD = _COS12 * _COS13
# CKM matrix element of u-s
CKM_US = _COS13 * _SIN12
# CKM matrix element of u-b
CKM_UB = _SIN13 * np.exp(1j * _DELTA)
# CKM matrix element of c-d
CKM_CD = -(_COS23 * _SIN12) - _COS12 * np.exp(1j * _DELTA) * _SIN13 * _SIN23
# CKM matrix element of c-s
CKM_CS = _COS12 * _COS23 - np.exp(1j * _DELTA) * _SIN12 * _SIN13 * _SIN23
# CKM matrix element of c-b
CKM_CB = _COS13 * _SIN23
# CKM matrix element of t-d
CKM_TD = -(_COS12 * _COS23 * np.exp(1j * _DELTA) * _SIN13) + _SIN12 * _SIN23
# CKM matrix element of t-s
CKM_TS = -(_COS23 * np.exp(1j * _DELTA) * _SIN12 * _SIN13) - _COS12 * _SIN23
# CKM matrix element of t-b
CKM_TB = _COS13 * _COS23

CKM = np.array(
    [
        [CKM_UD, CKM_US, CKM_UB],
        [CKM_CD, CKM_CS, CKM_CB],
        [CKM_TD, CKM_TS, CKM_TB],
    ]
)
