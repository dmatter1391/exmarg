"""
SIMULATION ONLY: synthetic segments for tests and calibration.

this module never touches the processing chain — real data enters
the pipeline directly at evaluate.zeta, and nothing here is imported
by the evaluation side.  generation and processing must never mix.
"""

import numpy as np

from .detectors import S, Fplus_Fcross, c, x, basis
from .profiles import profile


class Event:
    pass


def make_event(m1, m2, s1z, s2z, T, fmin, fmax, sky, psi_s, chi_s,
               d_max_mpc=10000.0, ymin=0.4):
    '''template + geometry container for one synthetic event.

    ONE physical choice fixes everything: the distance prior reaches
    D_max (uniform in volume up to there).  the reference distance
    is then D0 = ymin D_max, and the template u at D0 is whatever it
    is — check it lies inside the store ladder before analysis.
    orientation never enters D0 — the event loudness comes from the
    y and orientation draws, as in a real population.  NOTE: fmin
    must keep the chirp inside the T segment (a 15+10 chirp from
    20 Hz lasts ~4.3 s and wraps T = 4; use fmin = 25 there).
    '''
    nu, f, a1, psi0 = profile(m1, m2, s1z, s2z, T, fmin, fmax, 1.0)
    sig2 = T * S('H', f) / 2.0
    D0 = ymin * d_max_mpc
    R = {}
    for I in 'HL':
        Fp, Fx = Fplus_Fcross(I, np.array([sky[0]]),
                              np.array([sky[1]]),
                              np.array([psi_s]))
        R[I] = complex((0.5 * (1 + chi_s**2) * Fp
                        + 1j * chi_s * Fx)[0])
    ev = Event()
    ev.m1, ev.m2, ev.s1z, ev.s2z = m1, m2, s1z, s2z
    ev.T, ev.fmin, ev.fmax, ev.ymin = T, fmin, fmax, ymin
    ev.nu, ev.f, ev.psi0 = nu, f, psi0
    ev.a, ev.D0 = a1 / D0, D0
    ev.sig2 = {'H': sig2, 'L': sig2.copy()}
    ev.u = {I: (ev.a**2 / ev.sig2[I]).sum() for I in 'HL'}
    ev.sky, ev.psi_s, ev.chi_s, ev.R_true = sky, psi_s, chi_s, R
    ev.rho_mf = float(np.sqrt(2 * (abs(R['H'])**2 + abs(R['L'])**2)
                              * ev.u['H']))
    return ev


def make_data(case, seed, inject=True, phi_s=1.3, t_s=3.7, y_s=1.0,
              noise=True):
    '''
    synthetic frequency domain data for one segment, both detectors:
    Gaussian noise plus, if inject, the case's template at its true
    sky position/orientation.

    Outputs
    * dtilde {'H': (B,), 'L': (B,)}
    '''
    rng = np.random.default_rng(seed)
    d = {}
    for I in 'HL':
        if noise:
            g = rng.standard_normal((2, case.f.size))
            d[I] = np.sqrt(case.sig2[I] / 2.0) * (g[0] + 1j * g[1])
        else:
            d[I] = np.zeros(case.f.size, dtype=complex)
    if inject:
        n_hat = basis(np.array([case.sky[0]]),
                      np.array([case.sky[1]]))[0][0]
        for I in 'HL':
            tau_I = n_hat @ (x[I] - x['L']) / c
            d[I] = d[I] + (y_s * case.R_true[I] * np.exp(-1j * phi_s)
                           * case.a * np.exp(1j * case.psi0)
                           * np.exp(-2j * np.pi * case.f
                                    * (t_s + tau_I)))
    return d
