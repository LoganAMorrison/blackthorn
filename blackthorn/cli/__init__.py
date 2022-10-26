from cleo.application import Application

from .branching_fractions import GenerateBranchingFractions
from .spectra import GenerateSpectra


def generate():
    app = Application(name="generate")
    app.add(GenerateBranchingFractions())
    app.add(GenerateSpectra())
    app.run()
