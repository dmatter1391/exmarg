"""
the distance prior of the W tables.

the W tables average the distance weight M_n(A) = int p(y) y^(2n) exp(-y^2 A) dy over the prior p(y) of
y = D0 / D.  two priors are available:

  'volume'          uniform in volume, p(y) = 3 ymin^3 / y^4 for y >= ymin (closed form, numerics.py)
  'madau-dickinson' the distance distribution of mergers whose rate follows the Madau-Dickinson history
                    out to z = 10, the prior of the simulated year of the paper (tabulated in
                    data/demo_population.npz, evaluated once per order on a grid in ln A)

    import exmarg
    exmarg.use_distance_prior('madau-dickinson')
    tau, W = exmarg.w_table_par(u, u, exmarg.madau_dickinson_ymin(), n_max)
"""
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from . import numerics, tables

_POP = np.load(Path(__file__).parent / 'data' / 'demo_population.npz')
_LNY, _P_LNY = _POP['lny'], _POP['p_lny']
_LNW = np.log(np.maximum(_P_LNY * np.gradient(_LNY), 1e-300))
_LNA = np.linspace(np.log(1e-6), np.log(1e6), 4001)
_CACHE = {}


def madau_dickinson_ymin():
    """the smallest y = D0 / D of the Madau-Dickinson prior (the distance at z = 10)"""
    return float(_POP['ymin'])


def _md_table(n):
    """ln M_n on the ln A grid, computed once per order"""
    if n not in _CACHE:
        y2 = np.exp(2.0 * _LNY)
        _CACHE[n] = logsumexp(_LNW[None, :] + n * np.log(y2)[None, :] - np.exp(_LNA)[:, None] * y2[None, :], axis=1)
    return _CACHE[n]


def ln_moment_madau_dickinson(n, x_, lnAf, ymin):
    """ln M_n(A) of the Madau-Dickinson prior, with the call signature of numerics.ln_moment_n"""
    if abs(ymin - madau_dickinson_ymin()) > 1e-9:
        raise ValueError('build the tables at ymin = madau_dickinson_ymin() = %.4f' % madau_dickinson_ymin())
    return np.interp(lnAf, _LNA, _md_table(n))


def use_distance_prior(name):
    """select the distance prior of every W table built afterwards"""
    if name == 'volume':
        tables.ln_moment_n = numerics.ln_moment_n
    elif name == 'madau-dickinson':
        tables.ln_moment_n = ln_moment_madau_dickinson
    else:
        raise ValueError("the distance prior is 'volume' or 'madau-dickinson', not %r" % name)
