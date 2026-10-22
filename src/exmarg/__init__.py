"""
exmarg: analytic marginalization over the extrinsic parameters of a
compact binary signal in two detectors.

modules:
  detectors  where the detectors are, what they hear, their noise
  profiles   the template: expected signal shape per mass/spin
  data       making segments and the matched filter series z
  numerics   log-arithmetic helpers (no physics inside)
  tables     the W tables (GORFs): sky/orientation/distance averages
  geometry   the GORFs in frequency and the truncation deficit
  search     the analysis of segments: read the summed tables at
             every arrival time and lag, on the CPU or the GPU

extrinsics (sky, time, phase, polarization, inclination, distance)
are never sampled; they live inside the tables.  formalism:
paper/main.pdf.
"""

from .profiles import profile
from .data import make_data, make_event
from .numerics import moments, ln_offset
from .tables import w_table_dl, w_table_par, rebin, patterns, noise_mean
from .priors import use_distance_prior, madau_dickinson_ymin
from . import detectors, geometry, search
