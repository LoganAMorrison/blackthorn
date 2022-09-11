# `blackthorn`

---

| [**Overview**](#overview)
| [**Installation**](#install)
| [**Usage**](#usage)

## <a id="overview">Overview</a>

`blackthorn` is a tool for generating photon, positron and neutrino spectra
dark-matter annihilations into or decays of right-handed (RH) neutrinos.
`blackthorn` is capable of computing spectra for nearly any RH-neutrino mass.
To handle the large mass range, `blackthorn` uses:

- [`hazma`](https://github.com/LoganAMorrison/Hazma) for RH-neutrino masses < 1 GeV,
- [`PPC4DMID`](http://www.marcocirelli.net/PPPC4DMID.html) for RH-neutrino masses > 5 GeV and < 1 TeV,
- [`HDMSpectra`](https://github.com/nickrodd/HDMSpectra) for RH-neutrino masses > 1 TeV.

## 📦 <a id="installation">Installation</a>

## 🚀 <a id="usage">Usage</a>

### CLI

`blackthorn` comes with a command-line-interface to generate spectra. The
format for executing the CLI is:

```shell
python -m blackthorn.generate_spectrum [OPTIONS] MASS GENERATION OUTPUT
```

The required positional arguments are:

- `MASS`: Mass of the RH-neutrino in GeV,
- `GENERATION`: Generation of the RH-neutrino (1, 2, or 3)
- `OUTPUT`: File where output should be stored. The output is Javascript-object notation (JSON).

The common options are:

- `--all`: Generate spectra for photon, positron and neutrinos,
- `--photon`: Turn on photon spectra generation (ignored if `--all` is set),
- `--positron`: Turn on positron spectra generation (ignored if `--all` is set),
- `--neutrino`: Turn on neutrino spectra generation (ignored if `--all` is set),
- `--x-min`: Set the minimum value of `x=2E/sqrt(s)` (default is `1e-4`),
- `--x-max`: Set the minimum value of `x=2E/sqrt(s)` (default is `1`),
- `--n`: Set the number of `x=2E/sqrt(s)` values (default is `100`),
- `--scale`: Scaling of the `x=2E/sqrt(s)` values ("log" or "linear", default is "log"),
- `--eps`: Energy resolution for convolving spectra in order to resolve
  delta-function contributions (default is `0.05`),
- `--dm-mass`: Mass of the dark-matter. If specified, spectra will be generated
  assume DM annihilation into two RH-neutrinos.

For example, to generate the photon, neutrino and positron spectra from the
decay a 100 TeV, first-generation RH-neutrino, use:

```shell
python -m blackthorn.generate_spectrum --all 1e5 1 output.json
```

### Python

To use `blackthorn` from Python, the most important classes are:

- Models:
  - `blackthorn.models.RhNeutrinoMeV`: Class for RH-neutrinos with a mass < 1 GeV,
  - `blackthorn.models.RhNeutrinoGeV`: Class for RH-neutrinos with a mass > 5 GeV and < 1 TeV,
  - `blackthorn.models.RhNeutrinoTeV`: Class for RH-neutrinos with a mass > 1 TeV.
- Constants:
  - `blackthorn.constants.Gen`: Enumeration of generations (`Fst`, `Snd`, and `Trd`).
- Fields:
  - `blackthorn.fields.Photon`: Class representing the photon,
  - `blackthorn.fields.Positron`: Class representing the photon,
  - `blackthorn.fields.ElectronNeutrino`: Class representing the electron-neutrino,
  - `blackthorn.fields.MuonNeutrino`: Class representing the muon-neutrino,
  - `blackthorn.fields.TauNeutrino`: Class representing the tau-neutrino.
- Spectra:
  - `blackthorn.spectrum_utils.Spectrum`: Class for working with spectra (boosting, convolving)

Each of the model classes is instantiated with `MODEL(mass: float, theta: float, gen: Gen)` where `MODEL=RhNeutrinoMeV,RhNeutrinoGeV,RhNeutrinoTeV`. For
example, to construct a RH-neutrino model, use:

```python
from blackthorn.models import RhNeutrinoMeV, RhNeutrinoGeV, RhNeutrinoTeV
from blackthorn.constants import Gen

# RH-neutrino with mass 500 MeV, mixing angle 10^-3 with the first generation
model = RhNeutrinoMeV(0.5, 1e-3, Gen.Fst)

# RH-neutrino with mass 100 GeV, mixing angle 10^-3 with the second generation
model = RhNeutrinoGeV(100.0, 1e-3, Gen.Snd)

# RH-neutrino with mass 100 TeV, mixing angle 10^-3 with the third generation
model = RhNeutrinoTeV(100e3, 1e-3, Gen.Trd)
```

Spectrum generation for each model is identical. To generate spectra dN/dx, use:

```python
import numpy as np
from blackthorn import models
from blackthorn.constants import Gen
from blackthorn import fields

# 100 x=2E/sqrt(s) logarithmically spaced
x = np.geomspace(1e-4, 1, 100)

# Generate the photon spectrum
model = models.RhNeutrinoMeV(0.5, 1e-3, Gen.Fst)
spectrum = model.dndx(x, fields.Photon)

# Generate the positron spectrum
model = models.RhNeutrinoGeV(100.0, 1e-3, Gen.Snd)
spectrum = model.dndx(x, fields.Positron)

# Generate the electron-neutrino spectrum
model = models.RhNeutrinoTeV(100e3, 1e-3, Gen.Trd)
spectrum = model.dndx(x, fields.ElectronNeutrino)
```

The return value of `dndx` is a `Spectrum` object. The `Spectrum` objects has
various properties and functions for working with the underlying spectrum.
Below are some commonly used functionalities:

```python
# ... Same as above

# Get the x values (numpy array)
spectrum.x

# Get the dN/dx values (numpy array)
spectrum.dndx

# Boost the spectrum (beta is the boost velocity)
spectrum.boost(beta=0.5)

# Convolve the spectrum (eps is the energy resolution)
spectrum.convolve(eps=0.1)
```

For more fine-grained control of the convolution, use `total_conv_spectrum_fn`
method, available for each model (note this function return dN/dE, _not_ dN/dx):

```python
import numpy as np
from blackthorn import models
from blackthorn.constants import Gen
from blackthorn import fields

e_min = 1e-4
e_max = 0.5

# Specify a function to compute energy resolution
def energy_res(e):
  return ... # Return the energy resolution as %

# Generate the photon spectrum **dN/dE**
model = models.RhNeutrinoMeV(0.5, 1e-3, Gen.Fst)
spectrum = model.total_conv_spectrum_fn(
  e_min=1e-4,            # Minimum energy needed
  e_max=0.5,             # Maximum energy needed
  energy_res=energy_res, # See above
  product=fields.Photon, # Product to generate spectrum for
  npts=1000,             # Number of interpolation points to use
  # cme=...              # Center-of-mass energy (for DM annihilations)
)
```

#### Decay Branching Fractions and Partial Widths

To compute decay branching fractions and/or partial decay widths, use:

```python
from blackthorn import models
from blackthorn.constants import Gen

model = RhNeutrinoTeV(1e6, 1e-3, Gen.Fst)

# Compute the decay branching rations. Returns a dictionary.
model.branching_fractions()
# Output:
# {
#   "ve h": ..., # electron-neutrino and Higgs
#   "ve z": ..., # electron-neutrino and Z
#   "e w": ...,  # electron and W (both charge states)
# }

# Compute the partial decay widths
model.partial_widths()
# Output: Same format as `branching_fractions`
```
