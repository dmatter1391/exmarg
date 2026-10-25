"""
the log-arithmetic armor, quarantined here so the physics modules
read like the formulas.

why it exists: the formulas contain factorials and powers that leave
float64 (Gamma(n) overflows at n ~ 171, u^-n underflows) even though
every PHYSICAL result is a modest number.  so every product is
carried as a sum of logs and exponentiated once, bounded; and every
sum of positive terms goes through logsumexp.  nothing in this file
is physics — deleting the armor changes no formula, only crashes the
arithmetic.
"""

import numpy as np
from scipy.special import erfc, gammaln, gammaincc

A_FLOOR_X = 1e-10      # floor on x = ymin^2 A, so logs stay finite
LN_CAP = 700.0         # exp cap; only reached where the companion
                       # factor already makes the product negligible


def bounded_exp(lnv):
    '''exp with the overflow guard'''
    return np.exp(np.minimum(lnv, LN_CAP))


def logsumexp(v, axis=None):
    '''ln sum exp(v), overflow safe'''
    v = np.asarray(v, float)
    top = v.max(axis=axis, keepdims=True)
    out = top + np.log(np.exp(v - top).sum(axis=axis, keepdims=True))
    return float(out) if axis is None else np.squeeze(out, axis=axis)


def ln_offset(n, u1, u2):
    '''per order storage offset o_n = ln n! + n ln min(u1, u2);
    keeps stored W rows O(1) at every order.  undo with exp(-o_n).'''
    return (gammaln(np.asarray(n, float) + 1.0)
            + np.asarray(n, float) * np.log(np.minimum(u1, u2)))


def ln_moment_n(n, x_, lnAf, ymin):
    '''
    ln M_n of the distance integral, one order:

      M_n(A) = (3/2) ymin^3 A^{(3-2n)/2} Gamma(n - 3/2, ymin^2 A)

    from x_ = clip(ymin^2 A, A_FLOOR_X) and lnAf = ln x_ - 2 ln ymin.
    n >= 2 uses lngamma + the regularized upper incomplete gamma
    (inside [0, 1], stable at any order); n = 0, 1 have negative half
    integer index and come from the two step downward ladder off
    Gamma(1/2, x) = sqrt(pi) erfc(sqrt(x)), which is bounded.
    '''
    base = np.log(1.5) + 3.0 * np.log(ymin)
    if n >= 2:
        s = n - 1.5
        q = np.maximum(gammaincc(s, x_), 1e-300)
        return base + (1.5 - n) * lnAf + gammaln(s) + np.log(q)
    ex = np.exp(-x_)
    g_ph = np.sqrt(np.pi) * erfc(np.sqrt(x_))
    g_mh = (g_ph - x_**(-0.5) * ex) / (-0.5)
    G = g_mh if n == 1 else (g_mh - x_**(-1.5) * ex) / (-1.5)
    return base + (1.5 - n) * lnAf + np.log(np.maximum(G, 1e-300))


def ln_moments(A, n_max, ymin):
    '''list of ln M_n arrays, n = 0 .. n_max'''
    A = np.asarray(A, float)
    x_ = np.clip(ymin**2 * A, A_FLOOR_X, None)
    lnAf = np.log(x_) - 2.0 * np.log(ymin)
    return [ln_moment_n(n, x_, lnAf, ymin) for n in range(n_max + 1)]


def moments(A, n_max, ymin):
    '''absolute M_n (exponentiated); small n use only'''
    return [np.exp(l) for l in ln_moments(A, n_max, ymin)]
