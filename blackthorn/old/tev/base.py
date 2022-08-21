"""
blah
"""
from dataclasses import dataclass

from ..constants import Gen


@dataclass
class RhNeutrinoTeVBase:
    mass: float
    theta: float
    gen: Gen
