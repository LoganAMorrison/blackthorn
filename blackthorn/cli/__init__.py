from cleo.application import Application

from .branching_fractions import GenerateBranchingFractions


def generate():
    app = Application(name="generate")
    app.add(GenerateBranchingFractions())
    app.run()
