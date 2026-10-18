"""
exmarg: analytic marginalization over the extrinsic parameters of a
compact binary signal in two detectors.

seven small modules, one pipeline:
  detectors  where the detectors are, what they hear, their noise
  profiles   the template: expected signal shape per mass/spin
  data       making segments and the matched filter series z
  numerics   the log-arithmetic armor (no physics inside)
  tables     the W tables: sky/orientation/distance averages,
             precomputed once per loudness node
  kernel     lookup tables compiled from W; scoring is lookups
  evaluate   Evaluator (bracketing, ln Lambda, scans) + emcee

extrinsics (sky, time, phase, polarization, inclination, distance)
are never sampled; they live inside the tables.  formalism:
paper/main.pdf.
"""

from .profiles import profile
from .data import make_data, make_event
from .numerics import moments, ln_offset
from .tables import (w_table_dl, w_table_par, rebin, taylor,
                     taylor_order, patterns, noise_mean)
from .kernel import (Kernel, SumKernel, sum_lnlam_pair,
                     save_sumstore, load_sumstore)
from .evaluate import (Evaluator, SumEvaluator, load_store,
                       mc_q_to_m1m2, zeta)
from .priors import use_distance_prior, madau_dickinson_ymin
from . import detectors, geometry
